#!/usr/bin/env python3
"""Distributed QSA TP/CP operator smoke.

Run with ``torchrun --nproc_per_node=8 qsa_distributed_smoke.py --mode tp``
or 2/4/8 ranks and ``--mode cp``.  The script checks output, LSE, dQ and the
owner-accumulated dK/dV rather than only exercising a forward shape.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from mcore_bridge.model.modules.qsa_attention import qsa_sparse_forward
from mcore_bridge.model.modules.qsa_cp_exchange import qsa_exchange_selected_kv


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('tp', 'cp'), required=True)
    return parser.parse_args()


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (actual.float() - expected.float()).abs().max()


def _run_tp(rank: int, world: int) -> dict:
    if world != 8:
        raise RuntimeError(f'TP smoke requires world_size=8, got {world}')
    device = torch.device('cuda', torch.cuda.current_device())
    torch.manual_seed(901)
    seq_len, batch, q_heads, kv_heads, head_dim, slots = 64, 1, 24, 2, 16, 9
    q0 = torch.randn(seq_len, batch, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k0 = torch.randn(seq_len, batch, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v0 = torch.randn_like(k0)
    route_ids = torch.randint(0, seq_len, (batch, seq_len, slots), device=device, dtype=torch.int32)
    route_ids = torch.minimum(
        route_ids, torch.arange(seq_len, device=device, dtype=torch.int32).view(1, seq_len, 1))
    route_lengths = torch.minimum(
        torch.arange(seq_len, device=device, dtype=torch.int32).view(1, seq_len) + 1,
        torch.tensor(slots, device=device, dtype=torch.int32),
    ).expand(batch, seq_len).contiguous()
    grad_out = torch.randn_like(q0)

    q_start, q_end = rank * (q_heads // world), (rank + 1) * (q_heads // world)
    kv_head = rank // (world // kv_heads)
    q_local = q0[:, :, q_start:q_end].detach().clone().requires_grad_()
    k_local = k0[:, :, kv_head:kv_head + 1].detach().clone().requires_grad_()
    v_local = v0[:, :, kv_head:kv_head + 1].detach().clone().requires_grad_()
    local_out, local_lse = qsa_sparse_forward(
        q_local, k_local, v_local, route_ids, route_lengths,
        softmax_scale=head_dim ** -0.5, backend='triton', require_backend=True)
    (local_out * grad_out[:, :, q_start:q_end]).sum().backward()

    q_ref = q0.detach().clone().requires_grad_()
    k_ref = k0.detach().clone().requires_grad_()
    v_ref = v0.detach().clone().requires_grad_()
    full_out, full_lse = qsa_sparse_forward(
        q_ref, k_ref, v_ref, route_ids, route_lengths,
        softmax_scale=head_dim ** -0.5, backend='triton', require_backend=True)
    (full_out * grad_out).sum().backward()

    outputs = [torch.empty_like(local_out) for _ in range(world)]
    lses = [torch.empty_like(local_lse) for _ in range(world)]
    dqs = [torch.empty_like(q_local.grad) for _ in range(world)]
    dks = [torch.empty_like(k_local.grad) for _ in range(world)]
    dvs = [torch.empty_like(v_local.grad) for _ in range(world)]
    dist.all_gather(outputs, local_out)
    dist.all_gather(lses, local_lse)
    dist.all_gather(dqs, q_local.grad)
    dist.all_gather(dks, k_local.grad)
    dist.all_gather(dvs, v_local.grad)
    if rank == 0:
        output = torch.cat(outputs, dim=2)
        lse = torch.cat(lses, dim=1)
        d_q = torch.cat(dqs, dim=2)
        d_k = torch.cat((torch.stack(dks[:4]).sum(0), torch.stack(dks[4:]).sum(0)), dim=2)
        d_v = torch.cat((torch.stack(dvs[:4]).sum(0), torch.stack(dvs[4:]).sum(0)), dim=2)
        return {
            'mode': 'tp8',
            'output_max_abs': float(_max_error(output, full_out).item()),
            'lse_max_abs': float(_max_error(lse, full_lse).item()),
            'dq_max_abs': float(_max_error(d_q, q_ref.grad).item()),
            'dk_owner_sum_max_abs': float(_max_error(d_k, k_ref.grad).item()),
            'dv_owner_sum_max_abs': float(_max_error(d_v, v_ref.grad).item()),
            'finite': bool(torch.isfinite(output).all() and torch.isfinite(d_q).all()
                          and torch.isfinite(d_k).all() and torch.isfinite(d_v).all()),
        }
    return {}


def _run_cp(rank: int, world: int) -> dict:
    if world not in (2, 4, 8):
        raise RuntimeError(f'CP smoke requires world_size in {{2,4,8}}, got {world}')
    device = torch.device('cuda', torch.cuda.current_device())
    torch.manual_seed(902)
    # Keep four tokens per zigzag half-chunk for stable small-shape coverage:
    # the physical row is CP-aligned while the final logical row is padding.
    global_len, batch, q_heads, kv_heads, head_dim, slots = 8 * world, 1, 4, 2, 8, 9
    q_global = torch.randn(global_len, batch, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_global = torch.randn(global_len, batch, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_global = torch.randn_like(k_global)
    route_ids = torch.randint(0, global_len, (batch, global_len, slots), device=device, dtype=torch.int32)
    route_ids = torch.minimum(
        route_ids, torch.arange(global_len, device=device, dtype=torch.int32).view(1, global_len, 1))
    route_lengths = torch.minimum(
        torch.arange(global_len, device=device, dtype=torch.int32).view(1, global_len) + 1,
        torch.tensor(slots, device=device, dtype=torch.int32),
    ).expand(batch, global_len).contiguous()
    # Physical CP row is aligned to 16, while the logical sample ends at 15;
    # the final aligned lane is a no-loss/no-route padding query.
    route_lengths[:, -1] = 0
    route_ids[:, -1, :] = -1
    grad_out = torch.randn_like(q_global)

    chunk = global_len // (2 * world)
    local_positions = torch.tensor(
        list(range(rank * chunk, (rank + 1) * chunk))
        + list(range((2 * world - rank - 1) * chunk, (2 * world - rank) * chunk)),
        device=device, dtype=torch.int32)
    local_q = q_global.index_select(0, local_positions).detach().clone().requires_grad_()
    local_k = k_global.index_select(0, local_positions).detach().clone().requires_grad_()
    local_v = v_global.index_select(0, local_positions).detach().clone().requires_grad_()
    local_routes = route_ids.index_select(1, local_positions.to(torch.long))
    local_lengths = route_lengths.index_select(1, local_positions.to(torch.long))
    exchanged_k, exchanged_v, mapped_routes = qsa_exchange_selected_kv(
        local_k, local_v, local_routes, local_lengths, global_len,
        dist.group.WORLD, partition_mode='zigzag')
    local_out, local_lse = qsa_sparse_forward(
        local_q, exchanged_k, exchanged_v, mapped_routes, local_lengths,
        softmax_scale=head_dim ** -0.5, backend='triton', require_backend=True,
        causal=False, query_positions=local_positions, dkv_accum_dtype='fp32')
    (local_out * grad_out.index_select(0, local_positions.to(torch.long))).sum().backward()

    # The owner must receive dK/dV contributions from every CP query rank, so
    # the reference loss covers the full global query set. Local output/dQ are
    # compared after slicing the same zigzag positions.
    ref_q = q_global.detach().clone().requires_grad_()
    ref_k = k_global.detach().clone().requires_grad_()
    ref_v = v_global.detach().clone().requires_grad_()
    ref_out, ref_lse = qsa_sparse_forward(
        ref_q, ref_k, ref_v, route_ids, route_lengths,
        softmax_scale=head_dim ** -0.5, backend='triton', require_backend=True,
        query_positions=torch.arange(global_len, device=device, dtype=torch.int32),
        dkv_accum_dtype='fp32')
    (ref_out * grad_out).sum().backward()
    local_ref_out = ref_out.index_select(0, local_positions.to(torch.long))
    local_ref_lse = ref_lse.index_select(2, local_positions.to(torch.long))
    local_ref_dq = ref_q.grad.index_select(0, local_positions.to(torch.long))
    local_ref_dk = ref_k.grad.index_select(0, local_positions.to(torch.long))
    local_ref_dv = ref_v.grad.index_select(0, local_positions.to(torch.long))
    errors = torch.stack([
        _max_error(local_out, local_ref_out),
        _max_error(local_lse, local_ref_lse),
        _max_error(local_q.grad, local_ref_dq),
        _max_error(local_k.grad, local_ref_dk),
        _max_error(local_v.grad, local_ref_dv),
    ])
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)
    if rank == 0:
        return {
            'mode': f'cp{world}_selected_exchange',
            'global_len': global_len,
            'local_positions_rank0': local_positions.tolist(),
            'max_errors_output_lse_dq_dk_dv': [float(x) for x in errors.tolist()],
            'finite': bool(torch.isfinite(errors).all()),
        }
    return {}


def main() -> None:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError('distributed QSA smoke requires CUDA')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    result = _run_tp(rank, world) if args.mode == 'tp' else _run_cp(rank, world)
    if rank == 0:
        print(json.dumps(result, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
