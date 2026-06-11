import os

os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _init_distributed():
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    return local_rank, dist.get_rank(), dist.get_world_size()


def _split_cp_inputs_reference(inputs: torch.Tensor, cp_size: int, cp_rank: int, dim: int) -> torch.Tensor:
    if dim < 0:
        dim = (dim + inputs.ndim) % inputs.ndim
    inputs = inputs.view(
        *inputs.shape[:dim],
        2 * cp_size,
        inputs.shape[dim] // (2 * cp_size),
        *inputs.shape[dim + 1:],
    )
    index = torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], device=inputs.device)
    return inputs.index_select(dim, index).reshape(*inputs.shape[:dim], -1, *inputs.shape[dim + 2:])


def _build_head_perm_for_split_sections(split_sections, cp_size: int, device: torch.device) -> torch.Tensor:
    assert all(s % cp_size == 0 for s in split_sections), split_sections
    offset = 0
    parts = []
    for size in split_sections:
        parts.append(torch.arange(offset, offset + size, device=device, dtype=torch.long).view(cp_size, -1))
        offset += size
    return torch.cat(parts, dim=-1).view(-1)


def _undo_attention_load_balancing(input_: torch.Tensor, cp_size: int) -> torch.Tensor:
    chunks = torch.chunk(input_, chunks=2 * cp_size, dim=0)
    order = [2 * i for i in range(cp_size)] + [2 * cp_size - 2 * i - 1 for i in range(cp_size)]
    return torch.cat([chunks[i] for i in order], dim=0)


def _redo_attention_load_balancing(input_: torch.Tensor, cp_size: int) -> torch.Tensor:
    chunks = torch.chunk(input_, chunks=2 * cp_size, dim=0)
    order = [None] * (2 * cp_size)
    order[::2] = range(cp_size)
    order[1::2] = reversed(range(cp_size, 2 * cp_size))
    return torch.cat([chunks[i] for i in order], dim=0)


def _a2a_cp2hp(input_: torch.Tensor, cp_group, undo_attention_load_balancing: bool = True) -> torch.Tensor:
    cp_size = dist.get_world_size(group=cp_group)
    if cp_size == 1:
        return input_
    s_in, b_in, h_in = input_.shape
    h_out = h_in // cp_size
    send = input_.view(s_in, b_in, cp_size, h_out).permute(2, 0, 1, 3).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=cp_group)
    output = recv.reshape(cp_size * s_in, b_in, h_out)
    if undo_attention_load_balancing:
        output = _undo_attention_load_balancing(output, cp_size)
    return output


def _a2a_hp2cp(input_: torch.Tensor, cp_group, redo_attention_load_balancing: bool = True) -> torch.Tensor:
    cp_size = dist.get_world_size(group=cp_group)
    if cp_size == 1:
        return input_
    if redo_attention_load_balancing:
        input_ = _redo_attention_load_balancing(input_, cp_size)
    s_in, b_in, h_in = input_.shape
    s_out = s_in // cp_size
    send = input_.view(cp_size, s_out, b_in, h_in).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=cp_group)
    return recv.permute(1, 2, 0, 3).reshape(s_out, b_in, h_in * cp_size)


def _get_parameter_local_cp(param: torch.Tensor, dim: int, cp_group, split_sections=None) -> torch.Tensor:
    cp_size = dist.get_world_size(group=cp_group)
    cp_rank = dist.get_rank(group=cp_group)
    if cp_size == 1:
        return param
    if split_sections is not None:
        return torch.cat(
            [_get_parameter_local_cp(p, dim, cp_group) for p in torch.split(param, split_sections, dim=dim)],
            dim=dim,
        )
    slices = [slice(None)] * param.dim()
    dim_size = param.size(dim=dim)
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    return param[tuple(slices)]


def _build_thd_cp_a2a_perm(cu_seqlens: torch.Tensor, cp_size: int, t_global: int):
    cu = cu_seqlens.to(dtype=torch.long)
    seq_lens = torch.diff(cu)
    if (seq_lens % (2 * cp_size) != 0).any():
        raise ValueError(f'each packed sequence length must be divisible by {2 * cp_size}: {seq_lens.tolist()}')
    t_local = t_global // cp_size
    positions = torch.arange(t_global, device=cu.device)
    seq_idx = torch.bucketize(positions, cu[1:], right=True)
    halves = seq_lens // (2 * cp_size)
    local_starts = cu[:-1] // cp_size
    global_starts = cu[:-1]
    half_i = halves[seq_idx]
    pos_in_seq = positions - global_starts[seq_idx]
    natural_chunk = pos_in_seq // half_i
    offset = pos_in_seq - natural_chunk * half_i
    lb_chunk = torch.where(natural_chunk < cp_size, 2 * natural_chunk, 4 * cp_size - 2 * natural_chunk - 1)
    rank = lb_chunk // 2
    half_within_rank = lb_chunk - 2 * rank
    k = half_within_rank * half_i + offset
    idx = rank * t_local + local_starts[seq_idx] + k
    inv = torch.empty_like(idx)
    inv[idx] = positions
    return idx, inv


def _make_case(seq_len: int, batch: int, cp_size: int):
    torch.manual_seed(1234)
    dtype = torch.bfloat16
    device = torch.device('cuda')
    key_head_dim = 8
    value_head_dim = 8
    num_key_heads = cp_size
    num_value_heads = 3 * num_key_heads
    qk_dim = key_head_dim * num_key_heads
    v_dim = value_head_dim * num_value_heads
    in_proj_dim = 2 * qk_dim + 2 * v_dim + 2 * num_value_heads
    conv_dim = 2 * qk_dim + v_dim
    conv_kernel = 4
    qkvzba = (torch.randn(seq_len, batch, in_proj_dim, device=device, dtype=dtype) * 0.2).contiguous()
    conv_weight = (torch.randn(conv_dim, 1, conv_kernel, device=device, dtype=dtype) * 0.1).contiguous()
    conv_bias = (torch.randn(conv_dim, device=device, dtype=dtype) * 0.1).contiguous()
    A = torch.empty(num_value_heads, device=device, dtype=torch.float32).uniform_(1.0, 4.0)
    A_log = A.log().to(dtype).contiguous()
    dt_bias = torch.ones(num_value_heads, device=device, dtype=dtype)
    out_norm_weight = (torch.randn(value_head_dim, device=device, dtype=dtype) * 0.1 + 1.0).contiguous()
    dims = {
        'qk_dim': qk_dim,
        'v_dim': v_dim,
        'key_head_dim': key_head_dim,
        'value_head_dim': value_head_dim,
        'num_key_heads': num_key_heads,
        'num_value_heads': num_value_heads,
        'conv_kernel': conv_kernel,
    }
    return qkvzba, conv_weight, conv_bias, A_log, dt_bias, out_norm_weight, dims


def _slice_dims(dims, cp_size):
    return {
        'qk_dim': dims['qk_dim'] // cp_size,
        'v_dim': dims['v_dim'] // cp_size,
        'key_head_dim': dims['key_head_dim'],
        'value_head_dim': dims['value_head_dim'],
        'num_key_heads': dims['num_key_heads'] // cp_size,
        'num_value_heads': dims['num_value_heads'] // cp_size,
        'conv_kernel': dims['conv_kernel'],
    }


def _simple_causal_recurrent(query, key, value, g, beta):
    batch, seq_len, num_heads, key_dim = key.shape
    value_dim = value.shape[-1]
    state = torch.zeros(batch, num_heads, key_dim, value_dim, device=key.device, dtype=torch.float32)
    outputs = []
    for i in range(seq_len):
        decay = torch.exp(g[:, i].float()).unsqueeze(-1).unsqueeze(-1)
        state = state * decay
        update_value = value[:, i].float() * beta[:, i].float().unsqueeze(-1)
        state = state + key[:, i].float().unsqueeze(-1) * update_value.unsqueeze(-2)
        outputs.append(torch.einsum('bhk,bhkv->bhv', query[:, i].float(), state).to(query.dtype))
    return torch.stack(outputs, dim=1)


def _gdn_like_core(qkvzba, conv_weight, conv_bias, A_log, dt_bias, out_norm_weight, dims, cu_seqlens=None):
    seq_len, batch, _ = qkvzba.shape
    qk_dim = dims['qk_dim']
    v_dim = dims['v_dim']
    key_head_dim = dims['key_head_dim']
    value_head_dim = dims['value_head_dim']
    num_key_heads = dims['num_key_heads']
    num_value_heads = dims['num_value_heads']
    qkvzba = qkvzba.transpose(0, 1)
    qkv, gate, beta, alpha = torch.split(
        qkvzba,
        [2 * qk_dim + v_dim, v_dim, num_value_heads, num_value_heads],
        dim=-1,
    )
    conv_chunks = []
    seq_ranges = [(0, seq_len)] if cu_seqlens is None else list(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()))
    for start, end in seq_ranges:
        chunk = qkv[:, start:end].transpose(1, 2).contiguous()
        chunk = F.conv1d(
            chunk,
            conv_weight,
            conv_bias,
            padding=dims['conv_kernel'] - 1,
            groups=conv_weight.shape[0],
        )[..., :end - start]
        conv_chunks.append(F.silu(chunk).transpose(1, 2))
    qkv = torch.cat(conv_chunks, dim=1).contiguous()

    query_key, value = torch.split(qkv, [2 * qk_dim, v_dim], dim=-1)
    query_key = query_key.reshape(batch, seq_len, -1, key_head_dim)
    query_key_fp32 = query_key.float()
    query_key = (query_key_fp32 * torch.rsqrt((query_key_fp32 * query_key_fp32).sum(dim=-1, keepdim=True) + 1e-6)).to(
        query_key.dtype)
    query, key = torch.split(query_key, [num_key_heads, num_key_heads], dim=2)
    value = value.reshape(batch, seq_len, -1, value_head_dim)
    repeat_factor = num_value_heads // num_key_heads
    if repeat_factor > 1:
        query = query.repeat_interleave(repeat_factor, dim=2)
        key = key.repeat_interleave(repeat_factor, dim=2)

    gate = gate.reshape(batch, seq_len, -1, value_head_dim).contiguous()
    beta = beta.reshape(batch, seq_len, -1).sigmoid().contiguous()
    alpha = alpha.reshape(batch, seq_len, -1)
    g = -A_log.exp() * F.softplus(alpha.float() + dt_bias)
    if cu_seqlens is None:
        out = _simple_causal_recurrent(query.contiguous(), key.contiguous(), value.contiguous(), g.contiguous(), beta)
    else:
        out_chunks = []
        for start, end in seq_ranges:
            out_chunks.append(
                _simple_causal_recurrent(
                    query[:, start:end].contiguous(),
                    key[:, start:end].contiguous(),
                    value[:, start:end].contiguous(),
                    g[:, start:end].contiguous(),
                    beta[:, start:end].contiguous(),
                ))
        out = torch.cat(out_chunks, dim=1)
    rms = torch.rsqrt(out.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    out = (out.float() * rms * out_norm_weight.float()).to(out.dtype)
    out = out * F.silu(gate.float()).to(out.dtype)
    return out.reshape(batch, seq_len, -1).transpose(0, 1).contiguous()


def _assert_close(name, actual, expected, atol=6e-2, rtol=6e-2):
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max()
    denom = expected.float().abs().clamp_min(1e-6)
    max_rel = (diff / denom).max()
    stats = torch.stack([max_abs, max_rel])
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(f'{name}: max_abs={stats[0].item():.6f}, max_rel={stats[1].item():.6f}', flush=True)
    if stats[0].item() > atol and stats[1].item() > rtol:
        raise AssertionError(
            f'{name} mismatch: max_abs={stats[0].item():.6f}, max_rel={stats[1].item():.6f}, '
            f'atol={atol}, rtol={rtol}')


def _head_perm(dims, cp_size):
    return _build_head_perm_for_split_sections(
        (
            dims['qk_dim'],
            dims['qk_dim'],
            dims['v_dim'],
            dims['v_dim'],
            dims['num_value_heads'],
            dims['num_value_heads'],
        ),
        cp_size,
        torch.device('cuda'),
    )


def _check_unpacked(cp_group, rank, cp_size):
    if rank == 0:
        print('checking unpacked GDN-like CP equivalence', flush=True)
    qkvzba, conv_weight, conv_bias, A_log, dt_bias, out_norm_weight, dims = _make_case(
        seq_len=64, batch=1, cp_size=cp_size)
    ref = _gdn_like_core(qkvzba, conv_weight, conv_bias, A_log, dt_bias, out_norm_weight, dims)

    local = _split_cp_inputs_reference(qkvzba, cp_size, rank, dim=0).contiguous()
    local = local.index_select(-1, _head_perm(dims, cp_size))
    hp = _a2a_cp2hp(local, cp_group)
    conv_sections = [dims['qk_dim'], dims['qk_dim'], dims['v_dim']]
    cp_out_hp = _gdn_like_core(
        hp,
        _get_parameter_local_cp(conv_weight, dim=0, cp_group=cp_group, split_sections=conv_sections),
        _get_parameter_local_cp(conv_bias, dim=0, cp_group=cp_group, split_sections=conv_sections),
        _get_parameter_local_cp(A_log, dim=0, cp_group=cp_group),
        _get_parameter_local_cp(dt_bias, dim=0, cp_group=cp_group),
        out_norm_weight,
        _slice_dims(dims, cp_size),
    )
    value_start = rank * dims['v_dim'] // cp_size
    value_end = (rank + 1) * dims['v_dim'] // cp_size
    _assert_close('unpacked HP output', cp_out_hp, ref[..., value_start:value_end])
    _assert_close('unpacked roundtrip output', _a2a_hp2cp(cp_out_hp, cp_group),
                  _split_cp_inputs_reference(ref, cp_size, rank, dim=0).contiguous())


def _check_packed(cp_group, rank, cp_size):
    if rank == 0:
        print('checking packed GDN-like CP equivalence', flush=True)
    lengths = torch.tensor([16, 24, 8], device='cuda', dtype=torch.long)
    cu_seqlens = torch.cat([torch.zeros(1, device='cuda', dtype=torch.long), lengths.cumsum(dim=0)])
    seq_len = int(cu_seqlens[-1].item())
    qkvzba, conv_weight, conv_bias, A_log, dt_bias, out_norm_weight, dims = _make_case(
        seq_len=seq_len, batch=1, cp_size=cp_size)
    ref = _gdn_like_core(
        qkvzba, conv_weight, conv_bias, A_log, dt_bias, out_norm_weight, dims, cu_seqlens=cu_seqlens)

    local_parts = []
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        local_parts.append(_split_cp_inputs_reference(qkvzba[start:end], cp_size, rank, dim=0))
    local = torch.cat(local_parts, dim=0).contiguous().index_select(-1, _head_perm(dims, cp_size))
    hp = _a2a_cp2hp(local, cp_group, undo_attention_load_balancing=False)
    thd_idx, thd_inv = _build_thd_cp_a2a_perm(cu_seqlens, cp_size, seq_len)
    hp = hp.index_select(0, thd_idx)

    conv_sections = [dims['qk_dim'], dims['qk_dim'], dims['v_dim']]
    cp_out_hp = _gdn_like_core(
        hp,
        _get_parameter_local_cp(conv_weight, dim=0, cp_group=cp_group, split_sections=conv_sections),
        _get_parameter_local_cp(conv_bias, dim=0, cp_group=cp_group, split_sections=conv_sections),
        _get_parameter_local_cp(A_log, dim=0, cp_group=cp_group),
        _get_parameter_local_cp(dt_bias, dim=0, cp_group=cp_group),
        out_norm_weight,
        _slice_dims(dims, cp_size),
        cu_seqlens=cu_seqlens,
    )
    value_start = rank * dims['v_dim'] // cp_size
    value_end = (rank + 1) * dims['v_dim'] // cp_size
    _assert_close('packed HP output', cp_out_hp, ref[..., value_start:value_end])

    cp_out = _a2a_hp2cp(cp_out_hp.index_select(0, thd_inv), cp_group, redo_attention_load_balancing=False)
    expected_parts = []
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        expected_parts.append(_split_cp_inputs_reference(ref[start:end], cp_size, rank, dim=0))
    _assert_close('packed roundtrip output', cp_out, torch.cat(expected_parts, dim=0).contiguous())


def main():
    _, rank, cp_size = _init_distributed()
    if cp_size < 2:
        raise AssertionError('check_gdn_cp_equivalence.py requires at least 2 ranks.')
    cp_group = dist.group.WORLD
    _check_unpacked(cp_group, rank, cp_size)
    _check_packed(cp_group, rank, cp_size)
    if rank == 0:
        print(f'GDN-like CP equivalence checks OK with cp_size={cp_size}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
