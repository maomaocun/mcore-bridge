import importlib.util
import os
from pathlib import Path

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


def _load_gdn_helpers():
    module_path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/modules/gated_delta_net.py'
    spec = importlib.util.spec_from_file_location('mcore_bridge_gdn_for_backward_check', module_path)
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


def _make_case(seq_len: int, batch: int, cp_size: int):
    torch.manual_seed(4321)
    device = torch.device('cuda')
    dtype = torch.float32
    key_head_dim = 4
    value_head_dim = 4
    num_key_heads = 2 * cp_size
    num_value_heads = 2 * num_key_heads
    qk_dim = key_head_dim * num_key_heads
    v_dim = value_head_dim * num_value_heads
    in_proj_dim = 2 * qk_dim + 2 * v_dim + 2 * num_value_heads
    conv_dim = 2 * qk_dim + v_dim
    conv_kernel = 3
    scale = 0.08

    qkvzba = (torch.randn(seq_len, batch, in_proj_dim, device=device, dtype=dtype) * scale).requires_grad_(True)
    conv_weight = (torch.randn(conv_dim, 1, conv_kernel, device=device, dtype=dtype) * scale).requires_grad_(True)
    conv_bias = (torch.randn(conv_dim, device=device, dtype=dtype) * scale).requires_grad_(True)
    A_log = torch.empty(num_value_heads, device=device, dtype=dtype).uniform_(0.0, 0.7).requires_grad_(True)
    dt_bias = torch.zeros(num_value_heads, device=device, dtype=dtype, requires_grad=True)
    out_norm_weight = (torch.randn(value_head_dim, device=device, dtype=dtype) * scale + 1.0).requires_grad_(True)
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


def _clone_case(case):
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

    seq_ranges = [(0, seq_len)] if cu_seqlens is None else list(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()))
    conv_chunks = []
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


def _assert_close(name, actual, expected, atol=5e-5, rtol=5e-4):
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max()
    denom = expected.float().abs().clamp_min(1e-6)
    max_rel = (diff / denom).max()
    stats = torch.stack([max_abs, max_rel])
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(f'{name}: max_abs={stats[0].item():.8f}, max_rel={stats[1].item():.8f}', flush=True)
    if stats[0].item() > atol and stats[1].item() > rtol:
        raise AssertionError(
            f'{name} mismatch: max_abs={stats[0].item():.8f}, max_rel={stats[1].item():.8f}, '
            f'atol={atol}, rtol={rtol}')


def _head_perm(build_head_perm, dims, cp_size):
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


def _run_backward_case(name, cp_group, rank, cp_size, packed, helpers):
    build_head_perm, build_thd_perm, get_parameter_local_cp, tensor_a2a_cp2hp, tensor_a2a_hp2cp = helpers
    if rank == 0:
        print(f'checking {name} GDN-like CP backward equivalence', flush=True)

    if packed:
        lengths = torch.tensor([16, 24, 8], device='cuda', dtype=torch.long)
        cu_seqlens = torch.cat([torch.zeros(1, device='cuda', dtype=torch.long), lengths.cumsum(dim=0)])
        seq_len = int(cu_seqlens[-1].item())
    else:
        cu_seqlens = None
        seq_len = 64

    ref_case = _make_case(seq_len=seq_len, batch=1, cp_size=cp_size)
    cp_case = _clone_case(ref_case)
    ref_qkvzba, ref_conv_weight, ref_conv_bias, ref_A_log, ref_dt_bias, ref_out_norm_weight, dims = ref_case
    cp_qkvzba, cp_conv_weight, cp_conv_bias, cp_A_log, cp_dt_bias, cp_out_norm_weight, _ = cp_case

    torch.manual_seed(8765)
    ref_grad_out = torch.randn(seq_len, 1, dims['v_dim'], device='cuda', dtype=torch.float32) * 0.07

    ref = _gdn_like_core(
        ref_qkvzba,
        ref_conv_weight,
        ref_conv_bias,
        ref_A_log,
        ref_dt_bias,
        ref_out_norm_weight,
        dims,
        cu_seqlens=cu_seqlens,
    )
    (ref * ref_grad_out).sum().backward()

    head_perm = _head_perm(build_head_perm, dims, cp_size)
    if packed:
        local_parts = []
        local_grad_parts = []
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            local_parts.append(_split_cp_inputs_reference(cp_qkvzba.detach()[start:end], cp_size, rank, dim=0))
            local_grad_parts.append(_split_cp_inputs_reference(ref_grad_out[start:end], cp_size, rank, dim=0))
        local_qkvzba = torch.cat(local_parts, dim=0).contiguous().index_select(-1, head_perm)
        local_grad_out = torch.cat(local_grad_parts, dim=0).contiguous()
        hp = tensor_a2a_cp2hp(
            local_qkvzba.requires_grad_(True),
            seq_dim=0,
            head_dim=-1,
            cp_group=cp_group,
            undo_attention_load_balancing=False,
        )
        thd_idx, thd_inv = build_thd_perm(cu_seqlens, cp_size, seq_len)
        hp = hp.index_select(0, thd_idx)
    else:
        local_qkvzba = _split_cp_inputs_reference(cp_qkvzba.detach(), cp_size, rank, dim=0)
        local_qkvzba = local_qkvzba.contiguous().index_select(-1, head_perm).requires_grad_(True)
        local_grad_out = _split_cp_inputs_reference(ref_grad_out, cp_size, rank, dim=0).contiguous()
        hp = tensor_a2a_cp2hp(local_qkvzba, seq_dim=0, head_dim=-1, cp_group=cp_group)

    conv_sections = [dims['qk_dim'], dims['qk_dim'], dims['v_dim']]
    cp_out_hp = _gdn_like_core(
        hp,
        get_parameter_local_cp(cp_conv_weight, dim=0, cp_group=cp_group, split_sections=conv_sections),
        get_parameter_local_cp(cp_conv_bias, dim=0, cp_group=cp_group, split_sections=conv_sections),
        get_parameter_local_cp(cp_A_log, dim=0, cp_group=cp_group),
        get_parameter_local_cp(cp_dt_bias, dim=0, cp_group=cp_group),
        cp_out_norm_weight,
        _slice_dims(dims, cp_size),
        cu_seqlens=cu_seqlens,
    )

    if packed:
        cp_out = tensor_a2a_hp2cp(
            cp_out_hp.index_select(0, thd_inv),
            seq_dim=0,
            head_dim=-1,
            cp_group=cp_group,
            redo_attention_load_balancing=False,
        )
    else:
        cp_out = tensor_a2a_hp2cp(cp_out_hp, seq_dim=0, head_dim=-1, cp_group=cp_group)
    (cp_out * local_grad_out).sum().backward()

    if packed:
        expected_qkv_grad_parts = []
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            expected_qkv_grad_parts.append(_split_cp_inputs_reference(ref_qkvzba.grad[start:end], cp_size, rank, dim=0))
        expected_qkv_grad = torch.cat(expected_qkv_grad_parts, dim=0).contiguous().index_select(-1, head_perm)
    else:
        expected_qkv_grad = _split_cp_inputs_reference(ref_qkvzba.grad, cp_size, rank, dim=0)
        expected_qkv_grad = expected_qkv_grad.contiguous().index_select(-1, head_perm)

    _assert_close(f'{name} qkvzba grad', local_qkvzba.grad, expected_qkv_grad)
    _assert_close(f'{name} conv_weight grad', _gather_param_grad(cp_conv_weight.grad, cp_group), ref_conv_weight.grad)
    _assert_close(f'{name} conv_bias grad', _gather_param_grad(cp_conv_bias.grad, cp_group), ref_conv_bias.grad)
    _assert_close(f'{name} A_log grad', _gather_param_grad(cp_A_log.grad, cp_group), ref_A_log.grad)
    _assert_close(f'{name} dt_bias grad', _gather_param_grad(cp_dt_bias.grad, cp_group), ref_dt_bias.grad)
    _assert_close(
        f'{name} out_norm_weight grad',
        _gather_param_grad(cp_out_norm_weight.grad, cp_group),
        ref_out_norm_weight.grad,
    )


def main():
    _, rank, cp_size = _init_distributed()
    if cp_size < 2:
        raise AssertionError('check_gdn_cp_backward.py requires at least 2 ranks.')
    cp_group = dist.group.WORLD
    helpers = _load_gdn_helpers()
    _run_backward_case('unpacked', cp_group, rank, cp_size, packed=False, helpers=helpers)
    _run_backward_case('packed', cp_group, rank, cp_size, packed=True, helpers=helpers)
    if rank == 0:
        print(f'GDN-like CP backward checks OK with cp_size={cp_size}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
