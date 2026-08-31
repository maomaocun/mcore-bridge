# Copyright (c) ModelScope Contributors. All rights reserved.
"""Context-parallel selected-K/V owner exchange for QSA.

The indexer produces either global token positions or compact complete-block
IDs for each local query. This module maps them to CP owners, sends each unique
remote local offset once, and returns a compact per-rank K/V cache plus remapped
token indices. Compact input is expanded directly into the final cache-local
route without materializing an intermediate global-token route. The exchange
is an autograd boundary: backward sends remote dK/dV back to owners and reduces
duplicate requests with ``index_add_``.

This is separate from QSA attention so it can be benchmarked against a
full-K/V all-gather and eventually overlapped with the indexer. It supports
SBHD/unpacked CP only; THD and dynamic CP need segment-aware plans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.autograd import Function


@dataclass
class _QSAOwnerPlan:
    request_mask: torch.Tensor
    requests_by_owner: Tuple[torch.Tensor, ...]
    send_splits: List[int]
    local_key_len: int
    cp_group: object
    rank: int


def _owner_and_local_offsets(global_seq_len: int, cp_size: int, device: torch.device, partition_mode: str):
    positions = torch.arange(global_seq_len, device=device, dtype=torch.long)
    local_key_len = global_seq_len // cp_size
    if partition_mode == 'contiguous':
        return positions // local_key_len, positions.remainder(local_key_len), local_key_len
    if partition_mode != 'zigzag':
        raise ValueError(f'unsupported QSA CP partition mode: {partition_mode!r}')
    if global_seq_len % (2 * cp_size):
        raise ValueError(
            f'QSA zigzag CP requires global sequence divisible by 2*cp_size, got '
            f'sequence={global_seq_len}, cp_size={cp_size}')
    chunk_len = global_seq_len // (2 * cp_size)
    chunk = positions // chunk_len
    within = positions.remainder(chunk_len)
    owner = torch.where(chunk < cp_size, chunk, 2 * cp_size - 1 - chunk)
    local_offset = torch.where(chunk < cp_size, within, chunk_len + within)
    return owner, local_offset, local_key_len


def _expand_compact_owner_route(
    block_indices: torch.Tensor,
    topk_length: torch.Tensor,
    query_positions: torch.Tensor,
    route_block_size: int,
) -> torch.Tensor:
    """Populate the final-width owner route from compact block IDs."""

    if block_indices.is_cuda:
        from .qsa_triton import qsa_expand_compact_route_triton

        return qsa_expand_compact_route_triton(
            block_indices,
            topk_length,
            query_positions,
            route_block_size,
        )
    from .qsa_attention import qsa_expand_block_route

    return qsa_expand_block_route(
        block_indices,
        topk_length,
        query_positions,
        route_block_size,
    )


def _build_owner_plan(
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    global_seq_len: int,
    cp_group,
    partition_mode: str,
    route_block_size: int = 1,
    query_positions: Optional[torch.Tensor] = None,
    _map_in_place: bool = False,
) -> Tuple[_QSAOwnerPlan, torch.Tensor]:
    """Build owner requests and a cache-local token route.

    ``route_block_size=1`` consumes the public token route.  A larger value
    consumes compact complete-block IDs and reconstructs block lanes plus the
    causal tail directly into the final mapped token route.
    """

    route_block_size = int(route_block_size)
    if route_block_size <= 0:
        raise ValueError('QSA CP owner route_block_size must be positive')
    cp_size = cp_group.size()
    rank = cp_group.rank()
    if topk_indices.ndim != 3 or topk_length.ndim != 2:
        raise ValueError('QSA CP owner exchange expects topk_indices=[B,Sq,K] and topk_length=[B,Sq]')
    if topk_length.shape != topk_indices.shape[:2]:
        raise ValueError('QSA CP owner exchange route/length shape mismatch')
    if global_seq_len % cp_size:
        raise ValueError(f'global sequence {global_seq_len} is not divisible by CP size {cp_size}')
    if route_block_size > 1:
        if query_positions is None:
            raise ValueError(
                'QSA compact CP owner exchange requires global query_positions')
        if topk_indices.is_cuda:
            from .qsa_triton import (
                qsa_cp_compact_remap_route_triton,
                qsa_cp_compact_request_mask_triton,
            )

            request_mask = qsa_cp_compact_request_mask_triton(
                topk_indices,
                topk_length,
                query_positions,
                route_block_size,
                global_seq_len,
                cp_size,
                rank,
                partition_mode,
            )
            local_key_len = global_seq_len // cp_size
            plan, cache_offsets = _finalize_owner_plan(
                request_mask, local_key_len, cp_group, rank)
            mapped = qsa_cp_compact_remap_route_triton(
                topk_indices,
                topk_length,
                query_positions,
                cache_offsets,
                route_block_size,
                global_seq_len,
                cp_size,
                rank,
                partition_mode,
            )
            return plan, mapped
        # The CPU implementation remains an intentionally independent
        # compatibility reference rather than mirroring the Triton kernels.
        mapped = _expand_compact_owner_route(
            topk_indices,
            topk_length,
            query_positions,
            route_block_size,
        )
        return _build_owner_plan(
            mapped,
            topk_length,
            global_seq_len,
            cp_group,
            partition_mode,
            _map_in_place=True,
        )

    if topk_indices.is_cuda:
        from .qsa_triton import (
            qsa_cp_remap_route_triton,
            qsa_cp_request_mask_triton,
        )

        request_mask = qsa_cp_request_mask_triton(
            topk_indices,
            topk_length,
            global_seq_len,
            cp_size,
            rank,
            partition_mode,
        )
        local_key_len = global_seq_len // cp_size
        plan, cache_offsets = _finalize_owner_plan(
            request_mask, local_key_len, cp_group, rank)
        mapped = qsa_cp_remap_route_triton(
            topk_indices,
            topk_length,
            cache_offsets,
            global_seq_len,
            cp_size,
            rank,
            partition_mode,
            in_place=_map_in_place,
        )
        return plan, mapped

    owner_by_position, local_by_position, local_key_len = _owner_and_local_offsets(
        global_seq_len, cp_size, topk_indices.device, partition_mode)
    safe = topk_indices.to(torch.long).clamp(0, global_seq_len - 1)
    slots = torch.arange(topk_indices.shape[-1], device=topk_indices.device, dtype=torch.long)
    valid = ((slots[None, None, :] < topk_length.to(torch.long)[..., None]) &
             (topk_indices >= 0) & (topk_indices < global_seq_len))
    owners = owner_by_position[safe]
    local_offsets = local_by_position[safe]

    request_mask = torch.zeros(
        cp_size, local_key_len, device=topk_indices.device, dtype=torch.uint8)
    for owner in range(cp_size):
        if owner == rank:
            continue
        mask = valid & (owners == owner)
        request_mask[owner, local_offsets[mask]] = 1

    plan, cache_offsets = _finalize_owner_plan(
        request_mask, local_key_len, cp_group, rank)
    if _map_in_place:
        mapped = topk_indices
        mapped.fill_(-1)
        for owner in range(cp_size):
            owner_mask = valid & (owners == owner)
            mapped[owner_mask] = cache_offsets[
                owner, local_offsets[owner_mask]]
    else:
        mapped = torch.where(
            valid,
            cache_offsets[owners, local_offsets],
            torch.full_like(topk_indices, -1, dtype=torch.int32),
        )
    return plan, mapped


def _finalize_owner_plan(
    request_mask: torch.Tensor,
    local_key_len: int,
    cp_group,
    rank: int,
) -> Tuple[_QSAOwnerPlan, torch.Tensor]:
    """Freeze sorted requests and build owner/local-to-cache offsets."""

    cp_size = cp_group.size()
    cache_offsets = torch.full(
        (cp_size, local_key_len),
        -1,
        device=request_mask.device,
        dtype=torch.int32,
    )
    cache_offsets[rank] = torch.arange(
        local_key_len, device=request_mask.device, dtype=torch.int32)
    requests_by_owner = []
    remote_offset = local_key_len
    for owner in range(cp_size):
        if owner == rank:
            request = torch.empty(
                0, device=request_mask.device, dtype=torch.int32)
        else:
            request = request_mask[owner].nonzero(
                as_tuple=False).flatten().to(torch.int32)
            if request.numel():
                cache_offsets[owner, request.to(torch.long)] = torch.arange(
                    remote_offset,
                    remote_offset + request.numel(),
                    device=request_mask.device,
                    dtype=torch.int32,
                )
            remote_offset += int(request.numel())
        requests_by_owner.append(request)

    plan = _QSAOwnerPlan(
        request_mask=request_mask,
        requests_by_owner=tuple(requests_by_owner),
        send_splits=[int(request.numel()) for request in requests_by_owner],
        local_key_len=local_key_len,
        cp_group=cp_group,
        rank=rank,
    )
    return plan, cache_offsets


def _all_request_masks(request_mask: torch.Tensor, cp_group) -> torch.Tensor:
    gathered = [torch.empty_like(request_mask) for _ in range(cp_group.size())]
    dist.all_gather(gathered, request_mask, group=cp_group)
    return torch.stack(gathered, dim=0)


def _exchange_forward(tensor: torch.Tensor, plan: _QSAOwnerPlan, all_masks: torch.Tensor):
    recv_masks = all_masks[:, plan.rank]
    recv_splits = [int(row.sum().item()) for row in recv_masks]
    send_parts = []
    for requester, rows in enumerate(recv_splits):
        offsets = recv_masks[requester].nonzero(as_tuple=False).flatten().to(torch.long)
        send_parts.append(tensor.index_select(0, offsets) if rows else tensor[:0])
    send_tensor = torch.cat(send_parts, dim=0).contiguous() if send_parts else tensor[:0]
    remote = torch.empty(
        (sum(plan.send_splits), *tensor.shape[1:]), device=tensor.device, dtype=tensor.dtype)
    dist.all_to_all_single(
        remote,
        send_tensor,
        output_split_sizes=plan.send_splits,
        input_split_sizes=recv_splits,
        group=plan.cp_group,
    )
    return torch.cat((tensor, remote), dim=0), recv_splits


class _QSAOwnerExchangeFunction(Function):
    @staticmethod
    def forward(ctx, key: torch.Tensor, value: torch.Tensor, request_mask: torch.Tensor, plan: _QSAOwnerPlan):
        all_masks = _all_request_masks(request_mask, plan.cp_group)
        cache_key, recv_splits = _exchange_forward(key, plan, all_masks)
        cache_value, value_recv_splits = _exchange_forward(value, plan, all_masks)
        if recv_splits != value_recv_splits:
            raise RuntimeError('QSA key/value owner exchange split mismatch')
        ctx.plan = plan
        ctx.all_masks = all_masks
        ctx.recv_splits = recv_splits
        return cache_key, cache_value

    @staticmethod
    def backward(ctx, grad_cache_key: torch.Tensor, grad_cache_value: torch.Tensor):
        plan = ctx.plan
        all_masks = ctx.all_masks
        recv_splits = ctx.recv_splits
        result = []
        for grad_cache in (grad_cache_key, grad_cache_value):
            local_grad = grad_cache[:plan.local_key_len].clone()
            remote_grad = grad_cache[plan.local_key_len:].contiguous()
            returned = torch.empty(
                (sum(recv_splits), *grad_cache.shape[1:]),
                device=grad_cache.device,
                dtype=grad_cache.dtype)
            dist.all_to_all_single(
                returned,
                remote_grad,
                output_split_sizes=recv_splits,
                input_split_sizes=plan.send_splits,
                group=plan.cp_group,
            )
            start = 0
            for requester, rows in enumerate(recv_splits):
                if rows:
                    offsets = all_masks[requester, plan.rank].nonzero(as_tuple=False).flatten().to(torch.long)
                    local_grad.index_add_(0, offsets, returned[start:start + rows])
                start += rows
            result.append(local_grad)
        return result[0], result[1], None, None


def qsa_exchange_selected_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    global_seq_len: int,
    cp_group,
    partition_mode: str = 'zigzag',
    route_block_size: int = 1,
    query_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exchange unique selected K/V and return cache-local token indices.

    Compact input avoids an intermediate global-token route.  The returned
    route is always token based because the owner cache packs arbitrary remote
    tokens and does not expose a globally contiguous block address space.
    """

    cp_size = int(cp_group.size())
    route_block_size = int(route_block_size)
    if cp_size <= 0:
        raise ValueError(f'QSA CP group size must be positive, got {cp_size}')
    if route_block_size <= 0:
        raise ValueError(
            f'QSA CP owner route_block_size must be positive, got '
            f'{route_block_size}')
    if key.ndim != 4 or value.shape != key.shape:
        raise ValueError(f'QSA CP owner exchange expects local K/V [S,B,H,D], got {key.shape}, {value.shape}')
    if topk_indices.ndim != 3 or topk_length.shape != topk_indices.shape[:2]:
        raise ValueError(
            'QSA CP owner exchange expects topk_indices=[B,Sq,K] and '
            'topk_length=[B,Sq]')
    if global_seq_len <= 0 or global_seq_len % cp_size:
        raise ValueError(
            f'global sequence {global_seq_len} is not divisible by CP size '
            f'{cp_size}')
    if key.shape[0] != global_seq_len // cp_size:
        raise ValueError(
            f'QSA CP local key length mismatch: key={key.shape[0]}, global={global_seq_len}, cp={cp_size}')
    if cp_size == 1:
        if route_block_size == 1:
            return key, value, topk_indices
        if query_positions is None:
            raise ValueError(
                'QSA compact CP owner exchange requires global query_positions')
        mapped = _expand_compact_owner_route(
            topk_indices,
            topk_length,
            query_positions,
            route_block_size,
        )
        return key, value, mapped
    plan, mapped = _build_owner_plan(
        topk_indices,
        topk_length,
        global_seq_len,
        cp_group,
        partition_mode,
        route_block_size=route_block_size,
        query_positions=query_positions,
    )
    cache_key, cache_value = _QSAOwnerExchangeFunction.apply(key, value, plan.request_mask, plan)
    return cache_key, cache_value, mapped


__all__ = ['qsa_exchange_selected_kv']
