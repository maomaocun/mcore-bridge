import os
import importlib.util
from pathlib import Path

os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fla.modules.convolution import causal_conv1d
from fla.modules.l2norm import l2norm
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def _init_distributed():
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    return local_rank, dist.get_rank(), dist.get_world_size()


def _load_gdn_helpers():
    module_path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/modules/gated_delta_net.py'
    spec = importlib.util.spec_from_file_location('mcore_bridge_gdn_for_fla_check', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module._build_head_perm_for_split_sections,
        module._build_thd_cp_a2a_perm,
        module.get_parameter_local_cp,
        module.tensor_a2a_cp2hp,
        module.tensor_a2a_hp2cp,
    )


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
        chunks = torch.chunk(output, chunks=2 * cp_size, dim=0)
        order = [2 * i for i in range(cp_size)] + [2 * cp_size - 2 * i - 1 for i in range(cp_size)]
        output = torch.cat([chunks[i] for i in order], dim=0)
    return output


def _a2a_hp2cp(input_: torch.Tensor, cp_group, redo_attention_load_balancing: bool = True) -> torch.Tensor:
    cp_size = dist.get_world_size(group=cp_group)
    if cp_size == 1:
        return input_
    if redo_attention_load_balancing:
        chunks = torch.chunk(input_, chunks=2 * cp_size, dim=0)
        order = [None] * (2 * cp_size)
        order[::2] = range(cp_size)
        order[1::2] = reversed(range(cp_size, 2 * cp_size))
        input_ = torch.cat([chunks[i] for i in order], dim=0)
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


def _make_realish_qwen_case(seq_len: int, batch: int):
    torch.manual_seed(1234)
    dtype = torch.bfloat16
    device = torch.device('cuda')
    dims = {
        'qk_dim': 512,
        'v_dim': 1536,
        'key_head_dim': 128,
        'value_head_dim': 128,
        'num_key_heads': 4,
        'num_value_heads': 12,
        'conv_kernel': 4,
    }
    in_proj_dim = 2 * dims['qk_dim'] + 2 * dims['v_dim'] + 2 * dims['num_value_heads']
    conv_dim = 2 * dims['qk_dim'] + dims['v_dim']
    qkvzba = (torch.randn(seq_len, batch, in_proj_dim, device=device, dtype=dtype) * 0.2).contiguous()
    conv_weight = (torch.randn(conv_dim, 1, dims['conv_kernel'], device=device, dtype=dtype) * 0.1).contiguous()
    conv_bias = (torch.randn(conv_dim, device=device, dtype=dtype) * 0.1).contiguous()
    A = torch.empty(dims['num_value_heads'], device=device, dtype=torch.float32).uniform_(1.0, 4.0)
    A_log = A.log().to(dtype).contiguous()
    dt_bias = torch.ones(dims['num_value_heads'], device=device, dtype=dtype)
    out_norm_weight = (torch.randn(dims['value_head_dim'], device=device, dtype=dtype) * 0.1 + 1.0).contiguous()
    return qkvzba, conv_weight, conv_bias, A_log, dt_bias, out_norm_weight, dims


def _clone_case_with_grads(case):
    cloned = []
    for item in case[:-1]:
        cloned.append(item.detach().clone().requires_grad_(True))
    cloned.append(case[-1])
    return tuple(cloned)


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


def _gdn_fla_core(qkvzba, conv_weight, conv_bias, A_log, dt_bias, out_norm_weight, dims, cu_seqlens=None):
    seq_len, batch, _ = qkvzba.shape
    qk_dim = dims['qk_dim']
    v_dim = dims['v_dim']
    key_head_dim = dims['key_head_dim']
    value_head_dim = dims['value_head_dim']
    num_key_heads = dims['num_key_heads']
    num_value_heads = dims['num_value_heads']

    qkvzba = qkvzba.transpose(0, 1)
    qkv, gate, beta, alpha = torch.split(qkvzba, [2 * qk_dim + v_dim, v_dim, num_value_heads, num_value_heads], dim=-1)
    qkv = causal_conv1d(
        x=qkv.reshape(batch, seq_len, -1).contiguous(),
        weight=conv_weight.squeeze(1),
        bias=conv_bias,
        activation='silu',
        cu_seqlens=cu_seqlens,
    )[0]
    query_key, value = torch.split(qkv, [2 * qk_dim, v_dim], dim=-1)
    query_key = query_key.reshape(batch, seq_len, -1, key_head_dim)
    query_key = l2norm(query_key.contiguous())
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
    out, _ = chunk_gated_delta_rule(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        g=g.contiguous(),
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
    )
    rms = torch.rsqrt(out.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    out = (out.float() * rms * out_norm_weight.float()).to(out.dtype)
    out = out * F.silu(gate.float()).to(out.dtype)
    return out.reshape(batch, seq_len, -1).transpose(0, 1).contiguous()


def _assert_close(name, actual, expected, atol=1e-1, rtol=1e-1):
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


def _head_perm_from_helper(build_head_perm, dims, cp_size):
    return build_head_perm(
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


def _gather_param_grad(local_grad, cp_group):
    grad = torch.zeros_like(local_grad) if local_grad is None else local_grad.detach().clone()
    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=cp_group)
    return grad


def _check_packed_fla(cp_group, rank, cp_size):
    if rank == 0:
        print('checking packed real-operator GDN CP equivalence', flush=True)
    lengths = torch.tensor([1024], device='cuda', dtype=torch.long)
    cu_seqlens = torch.cat([torch.zeros(1, device='cuda', dtype=torch.long), lengths.cumsum(dim=0)])
    seq_len = int(cu_seqlens[-1].item())
    qkvzba, conv_weight, conv_bias, A_log, dt_bias, out_norm_weight, dims = _make_realish_qwen_case(
        seq_len=seq_len, batch=1)
    ref = _gdn_fla_core(qkvzba, conv_weight, conv_bias, A_log, dt_bias, out_norm_weight, dims, cu_seqlens=cu_seqlens)

    local = _split_cp_inputs_reference(qkvzba, cp_size, rank, dim=0).contiguous().index_select(-1, _head_perm(dims, cp_size))
    hp = _a2a_cp2hp(local, cp_group, undo_attention_load_balancing=False)
    thd_idx, thd_inv = _build_thd_cp_a2a_perm(cu_seqlens, cp_size, seq_len)
    hp = hp.index_select(0, thd_idx)

    conv_sections = [dims['qk_dim'], dims['qk_dim'], dims['v_dim']]
    cp_out_hp = _gdn_fla_core(
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
    _assert_close('packed FLA HP output', cp_out_hp, ref[..., value_start:value_end])

    cp_out = _a2a_hp2cp(cp_out_hp.index_select(0, thd_inv), cp_group, redo_attention_load_balancing=False)
    expected = _split_cp_inputs_reference(ref, cp_size, rank, dim=0).contiguous()
    _assert_close('packed FLA roundtrip output', cp_out, expected)


def _check_packed_fla_backward(cp_group, rank, cp_size, helpers):
    build_head_perm, build_thd_perm, get_parameter_local_cp, tensor_a2a_cp2hp, tensor_a2a_hp2cp = helpers
    if rank == 0:
        print('checking packed real-operator GDN CP backward equivalence', flush=True)
    lengths = torch.tensor([256], device='cuda', dtype=torch.long)
    cu_seqlens = torch.cat([torch.zeros(1, device='cuda', dtype=torch.long), lengths.cumsum(dim=0)])
    seq_len = int(cu_seqlens[-1].item())
    base_case = _make_realish_qwen_case(seq_len=seq_len, batch=1)
    ref_case = _clone_case_with_grads(base_case)
    cp_case = _clone_case_with_grads(base_case)
    ref_qkvzba, ref_conv_weight, ref_conv_bias, ref_A_log, ref_dt_bias, ref_out_norm_weight, dims = ref_case
    cp_qkvzba, cp_conv_weight, cp_conv_bias, cp_A_log, cp_dt_bias, cp_out_norm_weight, _ = cp_case

    torch.manual_seed(2468)
    grad_out = torch.randn(seq_len, 1, dims['v_dim'], device='cuda', dtype=torch.float32) * 0.05

    ref = _gdn_fla_core(
        ref_qkvzba,
        ref_conv_weight,
        ref_conv_bias,
        ref_A_log,
        ref_dt_bias,
        ref_out_norm_weight,
        dims,
        cu_seqlens=cu_seqlens,
    )
    (ref.float() * grad_out).sum().backward()

    head_perm = _head_perm_from_helper(build_head_perm, dims, cp_size)
    local = _split_cp_inputs_reference(cp_qkvzba.detach(), cp_size, rank, dim=0)
    local = local.contiguous().index_select(-1, head_perm).requires_grad_(True)
    hp = tensor_a2a_cp2hp(
        local,
        seq_dim=0,
        head_dim=-1,
        cp_group=cp_group,
        undo_attention_load_balancing=False,
    )
    thd_idx, thd_inv = build_thd_perm(cu_seqlens, cp_size, seq_len)
    hp = hp.index_select(0, thd_idx)

    conv_sections = [dims['qk_dim'], dims['qk_dim'], dims['v_dim']]
    cp_out_hp = _gdn_fla_core(
        hp,
        get_parameter_local_cp(cp_conv_weight, dim=0, cp_group=cp_group, split_sections=conv_sections),
        get_parameter_local_cp(cp_conv_bias, dim=0, cp_group=cp_group, split_sections=conv_sections),
        get_parameter_local_cp(cp_A_log, dim=0, cp_group=cp_group),
        get_parameter_local_cp(cp_dt_bias, dim=0, cp_group=cp_group),
        cp_out_norm_weight,
        _slice_dims(dims, cp_size),
        cu_seqlens=cu_seqlens,
    )
    cp_out = tensor_a2a_hp2cp(
        cp_out_hp.index_select(0, thd_inv),
        seq_dim=0,
        head_dim=-1,
        cp_group=cp_group,
        redo_attention_load_balancing=False,
    )
    local_grad_out = _split_cp_inputs_reference(grad_out, cp_size, rank, dim=0).contiguous()
    (cp_out.float() * local_grad_out).sum().backward()

    expected_qkv_grad = _split_cp_inputs_reference(ref_qkvzba.grad, cp_size, rank, dim=0)
    expected_qkv_grad = expected_qkv_grad.contiguous().index_select(-1, head_perm)
    _assert_close('packed FLA qkvzba grad', local.grad, expected_qkv_grad, atol=2e-2, rtol=8e-2)
    _assert_close(
        'packed FLA conv_weight grad',
        _gather_param_grad(cp_conv_weight.grad, cp_group),
        ref_conv_weight.grad,
        atol=2e-2,
        rtol=8e-2,
    )
    _assert_close(
        'packed FLA conv_bias grad',
        _gather_param_grad(cp_conv_bias.grad, cp_group),
        ref_conv_bias.grad,
        atol=2e-2,
        rtol=8e-2,
    )
    _assert_close(
        'packed FLA A_log grad',
        _gather_param_grad(cp_A_log.grad, cp_group),
        ref_A_log.grad,
        atol=2e-2,
        rtol=8e-2,
    )
    _assert_close(
        'packed FLA dt_bias grad',
        _gather_param_grad(cp_dt_bias.grad, cp_group),
        ref_dt_bias.grad,
        atol=2e-2,
        rtol=8e-2,
    )
    _assert_close(
        'packed FLA out_norm_weight grad',
        _gather_param_grad(cp_out_norm_weight.grad, cp_group),
        ref_out_norm_weight.grad,
        atol=2e-2,
        rtol=8e-2,
    )


def main():
    _, rank, cp_size = _init_distributed()
    if cp_size < 2:
        raise AssertionError('check_gdn_cp_fla_equivalence.py requires at least 2 ranks.')
    cp_group = dist.group.WORLD
    _check_packed_fla(cp_group, rank, cp_size)
    _check_packed_fla_backward(cp_group, rank, cp_size, _load_gdn_helpers())
    if rank == 0:
        print(f'GDN FLA CP equivalence checks OK with cp_size={cp_size}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
