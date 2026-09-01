# Copyright (c) ModelScope Contributors. All rights reserved.
"""Selected-KV attention for Qwen4-Exp QSA.

This module intentionally keeps the reference implementation next to the
backend dispatch.  It gives the Triton kernel a small, auditable contract:

* query/key/value use Megatron's ``[sequence, batch, heads, head_dim]`` layout;
* ``topk_indices`` is ``[batch, query_sequence, K]`` and contains token
  positions for the compatibility ABI, or complete-block IDs when
  ``route_block_size > 1``;
* ``topk_length`` is ``[batch, query_sequence]`` and invalid index slots are
  ``-1``;
* optional ``query_positions`` carries global positions when CP uses a
  non-contiguous (zigzag) local query order;
* ``selected_token_group_size`` describes the QSA compression ratio for the
  optional block-level segmented dK/dV reduction;
* the output is ``[sequence, batch, query_heads, head_dim]`` and LSE is
  ``[batch, query_heads, sequence]``.

The torch implementation is tiled over queries and selected K/V.  It is not a
dense-mask fallback: its largest attention scratch tensor is bounded by
``query_tile * K * heads_per_kv``.  The custom autograd functions recompute
those tiles during backward and scatter-add duplicate selected K/V entries,
so training never saves an ``S x S`` probability matrix.  Packed THD inputs
use O(T) segment metadata and fixed-grid Triton launches on the production
atomic path.
"""

from __future__ import annotations

import math
import os
import weakref
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.autograd import Function

from mcore_bridge.utils import get_logger

logger = get_logger()

QSA_BACKENDS = ("none", "torch", "triton", "cudnn")


class _QSAContextParallelGather(Function):
    """Gather sequence shards from a CP/sequence group and reduce gradients.

    This is the correctness CP bridge.  It is deliberately separate from the
    selected-KV exchange planned for the production CP backend: forward uses a
    full gathered K/V tensor, but the custom backward prevents remote-rank K/V
    gradients from being silently dropped by a plain ``dist.all_gather``.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, cp_group, partition_mode: str):
        ctx.cp_group = cp_group
        ctx.partition_mode = partition_mode
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return tensor
        world_size = torch.distributed.get_world_size(group=cp_group)
        if world_size <= 1:
            ctx.world_size = 1
            return tensor
        local_length = tensor.shape[0]
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor.contiguous(), group=cp_group)
        stacked = torch.cat(gathered, dim=0)
        global_length = local_length * world_size
        if partition_mode == 'contiguous':
            global_positions = torch.arange(global_length, device=tensor.device, dtype=torch.long)
        elif partition_mode == 'zigzag':
            if global_length % (2 * world_size):
                raise ValueError(
                    f'QSA zigzag sequence gather requires global length divisible by 2*world_size; '
                    f'got length={global_length}, world_size={world_size}')
            positions = torch.arange(global_length, device=tensor.device, dtype=torch.long)
            chunk_length = global_length // (2 * world_size)
            chunk = positions // chunk_length
            within = positions.remainder(chunk_length)
            owner = torch.where(chunk < world_size, chunk, 2 * world_size - 1 - chunk)
            local_offset = torch.where(chunk < world_size, within, chunk_length + within)
            global_positions = owner * local_length + local_offset
        else:
            raise ValueError(f'unsupported QSA sequence partition mode: {partition_mode!r}')
        ctx.local_positions = torch.nonzero(
            global_positions // local_length == torch.distributed.get_rank(group=cp_group),
            as_tuple=False,
        ).flatten()
        ctx.world_size = world_size
        return stacked.index_select(0, global_positions)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if getattr(ctx, 'world_size', 1) <= 1:
            return grad_output, None, None
        local_grad = grad_output.index_select(0, ctx.local_positions).contiguous()
        torch.distributed.all_reduce(local_grad, group=ctx.cp_group)
        return local_grad, None, None


def qsa_reconstruct_cp_tensor(tensor: torch.Tensor, cp_group, partition_mode: str = 'zigzag') -> torch.Tensor:
    """Gather a sequence-sharded tensor with an autograd-safe reverse reduction."""

    return _QSAContextParallelGather.apply(tensor, cp_group, partition_mode)


class QSAKernelError(RuntimeError):
    """Base error for an unavailable or invalid QSA selected-KV backend."""


class QSAKernelUnavailable(QSAKernelError):
    """Raised when a requested backend cannot run for the current process."""


@dataclass(frozen=True)
class QSAResolvedBackend:
    requested: str
    actual: str
    fallback_reason: Optional[str] = None


# Packed cu_seqlens is structural batch metadata and is normally shared by
# every QSA layer. Retain the validated host tuple while that exact tensor is
# alive, but key it by object identity instead of Tensor equality (which is
# elementwise) and invalidate it on PyTorch-tracked in-place mutations.
_PACKED_BOUNDARY_CACHE = {}


def _packed_boundaries(cu_seqlens: torch.Tensor) -> tuple[int, ...]:
    cache_key = id(cu_seqlens)
    try:
        version = int(cu_seqlens._version)
    except RuntimeError:
        # Inference tensors intentionally have no version counter. They are
        # still safe to cache under the packed contract that cu_seqlens is
        # immutable structural metadata; ordinary training tensors retain
        # mutation-aware invalidation through the version comparison.
        version = None
    cached = _PACKED_BOUNDARY_CACHE.get(cache_key)
    if (cached is not None and cached[0]() is cu_seqlens
            and cached[1] == version):
        return cached[2]
    if cu_seqlens.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            'QSA packed cu_seqlens must be validated by one warmup call '
            'before CUDA Graph capture')
    boundaries = tuple(
        cu_seqlens.detach().to(device='cpu', dtype=torch.long).tolist())

    def drop(reference, key=cache_key):
        entry = _PACKED_BOUNDARY_CACHE.get(key)
        if entry is not None and entry[0] is reference:
            _PACKED_BOUNDARY_CACHE.pop(key, None)

    reference = weakref.ref(cu_seqlens, drop)
    _PACKED_BOUNDARY_CACHE[cache_key] = (reference, version, boundaries)
    return boundaries


def _device_capability(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cpu"
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        return ".".join(str(x) for x in torch.cuda.get_device_capability(index))
    except (RuntimeError, TypeError, ValueError):
        return "unknown"


def _triton_sm90_available(device: torch.device) -> bool:
    try:
        from .qsa_triton import is_sm90

        return is_sm90(device)
    except (ImportError, RuntimeError, TypeError, ValueError):
        return False


def resolve_qsa_backend(
    requested: str,
    device: torch.device,
    require: bool = False,
) -> QSAResolvedBackend:
    """Resolve a requested QSA backend without silently selecting dense attention.

    ``cudnn`` is reserved for a future cuDNN contract because cuDNN's public
    DSA descriptors do not guarantee QSA's GQA2/D256 shape.  It therefore
    falls back to the selected-KV torch implementation unless strict mode is
    enabled.  ``triton`` falls back to torch on non-SM90 hosts; on H100 the
    Triton implementation is selected.
    """

    requested = (requested or "none").lower()
    if requested not in QSA_BACKENDS:
        raise ValueError(f"unsupported qsa_kernel_backend={requested!r}; choose one of {QSA_BACKENDS}")
    if requested == "none":
        if require:
            raise QSAKernelUnavailable("require_qsa_kernel=true but qsa_kernel_backend=none")
        return QSAResolvedBackend(requested, "none")
    if requested == "torch":
        return QSAResolvedBackend(requested, "torch")

    if requested == "triton":
        if _triton_sm90_available(device):
            return QSAResolvedBackend(requested, "triton")
        reason = f"Triton SM90 backend unavailable on device={device}, capability={_device_capability(device)}"
    else:
        reason = "cuDNN QSA GQA2/D256 selected-KV descriptor is not implemented"

    if require:
        raise QSAKernelUnavailable(
            f"requested qsa_kernel_backend={requested!r} is unsupported: {reason}; "
            "set require_qsa_kernel=false to use the tiled selected-KV torch backend")
    return QSAResolvedBackend(requested, "torch", reason)


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
) -> Tuple[int, int, int, int, int, int, torch.Tensor, torch.Tensor]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "QSA selected-KV expects query/key/value in [sequence,batch,heads,head_dim] layout; "
            f"got query={tuple(query.shape)}, key={tuple(key.shape)}, value={tuple(value.shape)}")
    sq, batch, num_q_heads, head_dim = query.shape
    sk, key_batch, num_kv_heads, value_dim = key.shape
    if key_batch != batch or value.shape != key.shape:
        raise ValueError(f"QSA selected-KV K/V shape mismatch: key={tuple(key.shape)}, value={tuple(value.shape)}")
    if sk <= 0 or head_dim != value_dim or num_kv_heads <= 0 or num_q_heads <= 0:
        raise ValueError(f"invalid QSA selected-KV shapes: query={tuple(query.shape)}, key={tuple(key.shape)}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"QSA GQA requires query heads divisible by KV heads, got Hq={num_q_heads}, Hkv={num_kv_heads}")
    if topk_indices.ndim != 3 or topk_indices.shape[:2] != (batch, sq):
        raise ValueError(
            f"QSA topk_indices must be [B,Sq,K] with B={batch}, Sq={sq}; got {tuple(topk_indices.shape)}")
    if topk_indices.shape[-1] <= 0:
        raise ValueError("QSA topk_indices must contain at least one slot")
    if topk_length.ndim != 2 or topk_length.shape != (batch, sq):
        raise ValueError(f"QSA topk_length must be [B,Sq]={batch,sq}; got {tuple(topk_length.shape)}")
    if not (topk_indices.dtype in (torch.int32, torch.int64) and
            topk_length.dtype in (torch.int32, torch.int64)):
        raise TypeError(
            f"QSA indices/length must be int32 or int64, got {topk_indices.dtype} and {topk_length.dtype}")
    same_device = (query.device == key.device == value.device == topk_indices.device == topk_length.device)
    if not same_device:
        raise ValueError("QSA query, K/V, topk_indices, and topk_length must be on the same device")
    return sq, sk, batch, num_q_heads, num_kv_heads, head_dim, topk_indices, topk_length


def _normalise_scale(softmax_scale: Optional[float], head_dim: int) -> float:
    if softmax_scale is None:
        return head_dim**-0.5
    scale = float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale}")
    return scale


@torch.no_grad()
def qsa_expand_block_route(
    block_indices: torch.Tensor,
    topk_length: torch.Tensor,
    query_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Materialize a compatibility token route from compact block IDs.

    Production Triton attention performs this mapping in registers.  This
    adapter exists for the torch/reference backend, external token-ID callers,
    and parity tests; using it at long context intentionally gives up the
    compact route's memory saving.  Token ordering follows ``block_indices``:
    compact routes guarantee the exact selected set, not the public indexer's
    canonical descending-score order.
    """

    block_size = int(block_size)
    if block_size <= 1:
        raise ValueError(
            f'QSA block route requires block_size > 1, got {block_size}')
    if block_indices.ndim != 3:
        raise ValueError(
            'QSA compact block route expects block_indices=[B,S,block_topk]')
    batch, sequence_length, block_slots = block_indices.shape
    if topk_length.shape != (batch, sequence_length):
        raise ValueError(
            'QSA compact block route length must have shape '
            f'{(batch, sequence_length)}, got {tuple(topk_length.shape)}')
    query_positions = query_positions.to(
        device=block_indices.device, dtype=torch.long).reshape(-1)
    if query_positions.shape != (sequence_length,):
        raise ValueError(
            'QSA compact block route query_positions must have shape '
            f'[{sequence_length}], got {tuple(query_positions.shape)}')

    lengths = topk_length.to(device=block_indices.device, dtype=torch.long)
    selected_block_counts = torch.div(
        lengths.clamp_min(0), block_size, rounding_mode='floor'
    ).clamp_max(block_slots)
    tail_lengths = lengths.clamp_min(0).remainder(block_size)
    block_offsets = torch.arange(
        block_slots, device=block_indices.device, dtype=torch.long)
    lane_offsets = torch.arange(
        block_size, device=block_indices.device, dtype=torch.long)
    valid_blocks = (
        block_offsets[None, None, :]
        < selected_block_counts[..., None]
    )
    block_tokens = (
        block_indices.to(torch.long)[..., None] * block_size
        + lane_offsets
    ).reshape(batch, sequence_length, block_slots * block_size)
    valid_block_tokens = valid_blocks[..., None].expand(
        -1, -1, -1, block_size).reshape(
            batch, sequence_length, block_slots * block_size)

    token_slots = block_slots * block_size + block_size - 1
    scratch = torch.full(
        (batch, sequence_length, token_slots + 1),
        -1,
        device=block_indices.device,
        dtype=torch.int32,
    )
    scratch[..., :block_slots * block_size] = torch.where(
        valid_block_tokens,
        block_tokens.to(torch.int32),
        torch.full_like(block_tokens, -1, dtype=torch.int32),
    )

    tail_offsets = torch.arange(
        block_size - 1, device=block_indices.device, dtype=torch.long)
    tail_valid = tail_offsets[None, None, :] < tail_lengths[..., None]
    tail_destinations = (
        selected_block_counts[..., None] * block_size + tail_offsets
    )
    tail_start = (
        torch.div(
            query_positions + 1, block_size, rounding_mode='floor'
        ) * block_size
    )
    tail_tokens = tail_start[None, :, None] + tail_offsets
    invalid_destination = torch.full_like(tail_destinations, token_slots)
    scratch.scatter_(
        -1,
        torch.where(tail_valid, tail_destinations, invalid_destination),
        torch.where(
            tail_valid,
            tail_tokens.expand(batch, -1, -1).to(torch.int32),
            torch.full_like(tail_destinations, -1, dtype=torch.int32),
        ),
    )
    return scratch[..., :token_slots].contiguous()


def _valid_selected_indices(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    seq_len_k: int,
    causal: bool,
    query_position_offset: int,
    key_position_offset: int,
    query_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return safe gather indices and the per-slot validity mask."""

    slots = indices.shape[-1]
    slot_ids = torch.arange(slots, device=indices.device, dtype=torch.int64)
    valid = slot_ids.view(1, 1, -1) < lengths.to(torch.int64).unsqueeze(-1)
    valid = valid & (indices >= 0) & (indices < seq_len_k)
    if causal:
        if query_positions is None:
            query_positions = (torch.arange(indices.shape[1], device=indices.device, dtype=torch.int64) +
                               int(query_position_offset))
        else:
            query_positions = query_positions.to(device=indices.device, dtype=torch.int64)
            if query_positions.shape != (indices.shape[1], ):
                raise ValueError(
                    f"QSA query_positions must have shape [{indices.shape[1]}], got {tuple(query_positions.shape)}")
        key_positions = indices.to(torch.int64) + int(key_position_offset)
        valid = valid & (key_positions <= query_positions.view(1, -1, 1))
    # The mask prevents an invalid -1 slot from contributing.  Clamping only
    # makes the gather memory-safe; it does not turn that slot into token 0.
    return indices.to(torch.int64).clamp(0, seq_len_k - 1), valid


def _torch_selected_kv_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    query_position_offset: int,
    key_position_offset: int,
    query_positions: torch.Tensor,
    query_tile_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tiled selected-KV forward used by correctness and backward recompute."""

    sq, batch, num_q_heads, head_dim = query.shape
    sk, _, num_kv_heads, _ = key.shape
    heads_per_kv = num_q_heads // num_kv_heads
    query_bshd = query.transpose(0, 1)
    key_bshd = key.transpose(0, 1)
    value_bshd = value.transpose(0, 1)
    output = torch.zeros((batch, sq, num_q_heads, head_dim), device=query.device, dtype=query.dtype)
    lse = torch.full((batch, num_q_heads, sq), -float("inf"), device=query.device, dtype=torch.float32)

    query_tile_size = max(1, int(query_tile_size))
    for query_start in range(0, sq, query_tile_size):
        query_end = min(sq, query_start + query_tile_size)
        q_tile = query_bshd[:, query_start:query_end].float()
        indices = topk_indices[:, query_start:query_end]
        lengths = topk_length[:, query_start:query_end]
        safe_indices, valid = _valid_selected_indices(
            indices, lengths, sk, causal, query_position_offset + query_start, key_position_offset,
            query_positions[query_start:query_end])
        batch_indices = torch.arange(batch, device=query.device, dtype=torch.long)[:, None, None]
        selected_k = key_bshd[batch_indices, safe_indices].float()
        selected_v = value_bshd[batch_indices, safe_indices].float()

        for kv_head in range(num_kv_heads):
            head_start = kv_head * heads_per_kv
            head_end = head_start + heads_per_kv
            q_group = q_tile[:, :, head_start:head_end]
            k_group = selected_k[:, :, :, kv_head]
            v_group = selected_v[:, :, :, kv_head]
            scores = torch.einsum("bqhd,bqkd->bqhk", q_group, k_group) * softmax_scale
            scores = scores.masked_fill(~valid[:, :, None, :], -float("inf"))
            group_lse = torch.logsumexp(scores, dim=-1)
            # logsumexp(-inf) is -inf for an empty row.  The selected QSA
            # contract normally has at least the causal tail, but keeping the
            # zero behavior makes malformed/partial inference metadata safe.
            finite_lse = torch.isfinite(group_lse)
            safe_lse = torch.where(finite_lse, group_lse, torch.zeros_like(group_lse))
            probabilities = torch.exp(scores - safe_lse[..., None])
            probabilities = torch.where(valid[:, :, None, :], probabilities, torch.zeros_like(probabilities))
            group_output = torch.einsum("bqhk,bqkd->bqhd", probabilities, v_group)
            output[:, query_start:query_end, head_start:head_end] = group_output.to(query.dtype)
            lse[:, head_start:head_end, query_start:query_end] = group_lse.permute(0, 2, 1)
    return output.transpose(0, 1).contiguous(), lse


def _scatter_selected_grad(
    destination: torch.Tensor,
    safe_indices: torch.Tensor,
    contribution: torch.Tensor,
    kv_head: int,
) -> None:
    """Scatter a ``[B,Q,K,D]`` contribution into one KV head."""

    batch, query_tile, slots, head_dim = contribution.shape
    seq_len_k = destination.shape[1]
    flat_destination = destination.reshape(destination.shape[0] * seq_len_k, destination.shape[2], head_dim)
    batch_offsets = torch.arange(batch, device=destination.device, dtype=torch.long).view(batch, 1, 1)
    flat_indices = (safe_indices.to(torch.long) + batch_offsets * seq_len_k).reshape(-1)
    flat_contribution = contribution.reshape(-1, head_dim)
    flat_destination[:, kv_head].index_add_(0, flat_indices, flat_contribution)


def _torch_selected_kv_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    lse: torch.Tensor,
    grad_output: Optional[torch.Tensor],
    grad_lse: Optional[torch.Tensor],
    softmax_scale: float,
    causal: bool,
    query_position_offset: int,
    key_position_offset: int,
    query_positions: torch.Tensor,
    query_tile_size: int,
    need_query: bool,
    need_key: bool,
    need_value: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Recompute selected tiles and produce dQ/dK/dV."""

    sq, batch, num_q_heads, head_dim = query.shape
    sk, _, num_kv_heads, _ = key.shape
    heads_per_kv = num_q_heads // num_kv_heads
    q_bshd = query.transpose(0, 1).float()
    key_bshd = key.transpose(0, 1)
    value_bshd = value.transpose(0, 1)
    grad_out_bshd = None if grad_output is None else grad_output.transpose(0, 1).float()
    grad_query = (torch.zeros((batch, sq, num_q_heads, head_dim), device=query.device, dtype=torch.float32)
                  if need_query else None)
    grad_key = (torch.zeros((batch, sk, num_kv_heads, head_dim), device=key.device, dtype=torch.float32)
                if need_key else None)
    grad_value = (torch.zeros((batch, sk, num_kv_heads, head_dim), device=value.device, dtype=torch.float32)
                  if need_value else None)

    for query_start in range(0, sq, max(1, int(query_tile_size))):
        query_end = min(sq, query_start + max(1, int(query_tile_size)))
        q_tile = q_bshd[:, query_start:query_end]
        indices = topk_indices[:, query_start:query_end]
        lengths = topk_length[:, query_start:query_end]
        safe_indices, valid = _valid_selected_indices(
            indices, lengths, sk, causal, query_position_offset + query_start, key_position_offset,
            query_positions[query_start:query_end])
        batch_indices = torch.arange(batch, device=query.device, dtype=torch.long)[:, None, None]
        selected_k = key_bshd[batch_indices, safe_indices].float()
        selected_v = value_bshd[batch_indices, safe_indices].float()

        for kv_head in range(num_kv_heads):
            head_start = kv_head * heads_per_kv
            head_end = head_start + heads_per_kv
            q_group = q_tile[:, :, head_start:head_end]
            k_group = selected_k[:, :, :, kv_head]
            v_group = selected_v[:, :, :, kv_head]
            scores = torch.einsum("bqhd,bqkd->bqhk", q_group, k_group) * softmax_scale
            scores = scores.masked_fill(~valid[:, :, None, :], -float("inf"))
            group_lse = lse[:, head_start:head_end, query_start:query_end].permute(0, 2, 1)
            finite_lse = torch.isfinite(group_lse)
            safe_lse = torch.where(finite_lse, group_lse, torch.zeros_like(group_lse))
            probabilities = torch.exp(scores - safe_lse[..., None])
            probabilities = torch.where(valid[:, :, None, :], probabilities, torch.zeros_like(probabilities))

            if grad_out_bshd is not None:
                grad_out_group = grad_out_bshd[:, query_start:query_end, head_start:head_end]
                grad_probabilities = torch.einsum("bqhd,bqkd->bqhk", grad_out_group, v_group)
                grad_scores = probabilities * (
                    grad_probabilities - (grad_probabilities * probabilities).sum(dim=-1, keepdim=True))
            else:
                grad_scores = torch.zeros_like(probabilities)
            if grad_lse is not None:
                grad_lse_group = grad_lse[:, head_start:head_end, query_start:query_end].permute(0, 2, 1)
                grad_scores = grad_scores + grad_lse_group[..., None] * probabilities
            grad_scores = torch.where(valid[:, :, None, :], grad_scores, torch.zeros_like(grad_scores))

            if grad_query is not None:
                grad_query[:, query_start:query_end, head_start:head_end] += (
                    torch.einsum("bqhk,bqkd->bqhd", grad_scores, k_group) * softmax_scale)
            if grad_key is not None:
                grad_key_contribution = torch.einsum("bqhk,bqhd->bqkd", grad_scores, q_group) * softmax_scale
                _scatter_selected_grad(grad_key, safe_indices, grad_key_contribution, kv_head)
            if grad_value is not None:
                if grad_out_bshd is not None:
                    grad_value_contribution = torch.einsum(
                        "bqhk,bqhd->bqkd", probabilities,
                        grad_out_bshd[:, query_start:query_end, head_start:head_end])
                else:
                    grad_value_contribution = torch.zeros_like(k_group)
                _scatter_selected_grad(grad_value, safe_indices, grad_value_contribution, kv_head)

    grad_query_out = None if grad_query is None else grad_query.transpose(0, 1).to(query.dtype).contiguous()
    grad_key_out = None if grad_key is None else grad_key.transpose(0, 1).to(key.dtype).contiguous()
    grad_value_out = None if grad_value is None else grad_value.transpose(0, 1).to(value.dtype).contiguous()
    return grad_query_out, grad_key_out, grad_value_out


class _QSASelectedKVFunction(Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor,
        softmax_scale: float,
        causal: bool,
        backend: str,
        query_position_offset: int,
        key_position_offset: int,
        query_positions: torch.Tensor,
        query_tile_size: int,
        dkv_accum_dtype: str,
        selected_token_group_size: Optional[int],
        dkv_reduction: str,
        route_block_size: int,
        precomputed_scores: Optional[torch.Tensor],
    ):
        resolved = backend
        if resolved == "triton":
            from .qsa_triton import qsa_selected_kv_forward

            output, lse = qsa_selected_kv_forward(
                query, key, value, topk_indices, topk_length, softmax_scale, causal, query_positions,
                key_position_offset, route_block_size=route_block_size,
                saved_scores=precomputed_scores)
        else:
            output, lse = _torch_selected_kv_forward(
                query, key, value, topk_indices, topk_length, softmax_scale, causal, query_position_offset,
                key_position_offset, query_positions, query_tile_size)
        ctx.segmented_metadata = None
        effective_dkv_reduction = os.environ.get(
            'MCORE_BRIDGE_QSA_DKV_REDUCTION', dkv_reduction
        ).lower()
        if effective_dkv_reduction not in {'atomic', 'segmented'}:
            raise ValueError(
                f'unsupported QSA dkv_reduction={effective_dkv_reduction!r}; '
                "choose 'atomic' or 'segmented'")
        if (resolved == 'triton' and selected_token_group_size is not None and
                effective_dkv_reduction == 'segmented'):
            from .qsa_triton import qsa_prepare_segmented_metadata

            hybrid_min_fanout = int(os.environ.get(
                'MCORE_BRIDGE_QSA_SEGMENT_HYBRID_MIN_FANOUT', '0'))
            if hybrid_min_fanout < 0:
                raise ValueError(
                    'QSA segmented hybrid fanout threshold must be non-negative')

            ctx.segmented_metadata = qsa_prepare_segmented_metadata(
                topk_indices,
                topk_length,
                query_positions,
                query.shape[0],
                key.shape[0],
                int(selected_token_group_size),
                route_block_size=route_block_size,
                owner_min_fanout=hybrid_min_fanout,
            )
        # Backward can recover the softmax correction as ``sum(output *
        # grad_output)``.  Saving the already-materialized output avoids a
        # second full selected-K/V scan without allocating another tensor.
        ctx.save_for_backward(query, key, value, topk_indices, topk_length, output, lse, query_positions)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.backend = resolved
        ctx.query_position_offset = query_position_offset
        ctx.key_position_offset = key_position_offset
        ctx.query_tile_size = query_tile_size
        ctx.dkv_accum_dtype = dkv_accum_dtype
        ctx.selected_token_group_size = selected_token_group_size
        ctx.dkv_reduction = effective_dkv_reduction
        ctx.route_block_size = route_block_size
        ctx.precomputed_scores = precomputed_scores
        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        query, key, value, topk_indices, topk_length, output, lse, query_positions = ctx.saved_tensors
        if ctx.backend == 'triton':
            from .qsa_triton import qsa_selected_kv_backward

            grad_query, grad_key, grad_value = qsa_selected_kv_backward(
                query,
                key,
                value,
                topk_indices,
                topk_length,
                lse,
                output=output,
                grad_output=grad_output,
                grad_lse=grad_lse,
                softmax_scale=ctx.softmax_scale,
                causal=ctx.causal,
                query_positions=query_positions,
                key_position_offset=ctx.key_position_offset,
                dkv_accum_dtype=ctx.dkv_accum_dtype,
                selected_token_group_size=ctx.selected_token_group_size,
                segmented_metadata=ctx.segmented_metadata,
                dkv_reduction=ctx.dkv_reduction,
                precomputed_scores=ctx.precomputed_scores,
                route_block_size=ctx.route_block_size,
            )
        else:
            grad_query, grad_key, grad_value = _torch_selected_kv_backward(
                query,
                key,
                value,
                topk_indices,
                topk_length,
                lse,
                grad_output,
                grad_lse,
                ctx.softmax_scale,
                ctx.causal,
                ctx.query_position_offset,
                ctx.key_position_offset,
                query_positions,
                ctx.query_tile_size,
                ctx.needs_input_grad[0],
                ctx.needs_input_grad[1],
                ctx.needs_input_grad[2],
            )
        return (grad_query, grad_key, grad_value, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None)


class _QSASelectedKVPackedFunction(Function):
    """Autograd bridge for the fixed-grid packed THD Triton path."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor,
        key_starts: torch.Tensor,
        key_lengths: torch.Tensor,
        query_positions: torch.Tensor,
        softmax_scale: float,
        causal: bool,
        backend: str,
        key_position_offset: int,
        dkv_accum_dtype: str,
        dkv_reduction: str,
        route_block_size: int,
    ):
        if backend != 'triton':
            raise RuntimeError('packed selected-KV autograd requires the Triton backend')
        from .qsa_triton import qsa_selected_kv_forward_packed

        output, lse = qsa_selected_kv_forward_packed(
            query,
            key,
            value,
            topk_indices,
            topk_length,
            key_starts,
            key_lengths,
            query_positions,
            softmax_scale,
            causal=causal,
            key_position_offset=key_position_offset,
            route_block_size=route_block_size,
        )
        ctx.save_for_backward(
            query,
            key,
            value,
            topk_indices,
            topk_length,
            key_starts,
            key_lengths,
            query_positions,
            output,
            lse,
        )
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.backend = backend
        ctx.key_position_offset = key_position_offset
        ctx.dkv_accum_dtype = dkv_accum_dtype
        ctx.dkv_reduction = dkv_reduction
        ctx.route_block_size = route_block_size
        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        (
            query,
            key,
            value,
            topk_indices,
            topk_length,
            key_starts,
            key_lengths,
            query_positions,
            output,
            lse,
        ) = ctx.saved_tensors
        from .qsa_triton import qsa_selected_kv_backward_packed

        grad_query, grad_key, grad_value = qsa_selected_kv_backward_packed(
            query,
            key,
            value,
            topk_indices,
            topk_length,
            lse,
            key_starts,
            key_lengths,
            query_positions,
            output=output,
            grad_output=grad_output,
            grad_lse=grad_lse,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            key_position_offset=ctx.key_position_offset,
            dkv_accum_dtype=ctx.dkv_accum_dtype,
            dkv_reduction=ctx.dkv_reduction,
            route_block_size=ctx.route_block_size,
        )
        return (
            grad_query,
            grad_key,
            grad_value,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def qsa_sparse_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    backend: str = "torch",
    query_position_offset: int = 0,
    key_position_offset: int = 0,
    query_positions: Optional[torch.Tensor] = None,
    query_tile_size: int = 16,
    require_backend: bool = False,
    dkv_accum_dtype: str = 'bf16',
    selected_token_group_size: Optional[int] = None,
    dkv_reduction: str = 'atomic',
    route_block_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run QSA selected-KV attention.

    ``backend='triton'`` is the H100/SM90 online-softmax kernel.  The
    ``torch`` backend has identical selected-index semantics and is used for
    parity tests and explicit fallback.  The function never constructs a
    dense attention mask or a full score matrix.  ``dkv_reduction='segmented'``
    enables the experimental block-owned inverse-CSR dK/dV path; the causal
    tail remains additive, while ``atomic`` is the tuned default.  Setting
    ``MCORE_BRIDGE_QSA_SEGMENT_HYBRID_MIN_FANOUT`` to a positive value makes
    only complete blocks with at least that many query occurrences owner-owned;
    the remaining blocks stay on the query-side atomic path for A/B studies.
    ``MCORE_BRIDGE_QSA_SEGMENT_COMPACT_DERIVATIVES=1`` additionally stores
    probability and d-score only for owner occurrences in CSR order; this is
    an opt-in diagnostic that uses a bounded owner-sized workspace.
    ``MCORE_BRIDGE_QSA_SAVE_FORWARD_SCORES=1`` stores the FP32 raw-QK score
    workspace for the Triton atomic backward diagnostic, avoiding backward
    QK recomputation at the cost of ``[B,S,Hq,K]`` memory and traffic.
    ``route_block_size > 1`` selects the compact complete-block metadata ABI.
    """

    sq, sk, batch, num_q_heads, num_kv_heads, head_dim, topk_indices, topk_length = _validate_inputs(
        query, key, value, topk_indices, topk_length)
    del sq, sk, batch, num_q_heads, num_kv_heads
    scale = _normalise_scale(softmax_scale, head_dim)
    requested = (backend or "torch").lower()
    resolved = resolve_qsa_backend(requested, query.device, require=require_backend)
    actual = resolved.actual
    if actual == "none":
        raise ValueError("qsa_sparse_forward cannot use backend='none'; use dense/reference attention instead")
    if resolved.fallback_reason:
        logger.warning_once(
            f"QSA backend fallback: requested={resolved.requested}, actual={resolved.actual}, "
            f"reason={resolved.fallback_reason}")
    dkv_accum_dtype = str(dkv_accum_dtype).lower()
    if dkv_accum_dtype not in {'bf16', 'fp32'}:
        raise ValueError(f"unsupported QSA dkv_accum_dtype={dkv_accum_dtype!r}; choose 'bf16' or 'fp32'")
    effective_dkv_reduction = os.environ.get(
        'MCORE_BRIDGE_QSA_DKV_REDUCTION', dkv_reduction).lower()
    if effective_dkv_reduction not in {'atomic', 'segmented'}:
        raise ValueError(
            f"unsupported QSA dkv_reduction={effective_dkv_reduction!r}; "
            "choose 'atomic' or 'segmented'")
    route_block_size = int(route_block_size)
    if route_block_size <= 0:
        raise ValueError('QSA route_block_size must be positive')
    if (route_block_size > 1 and selected_token_group_size is not None
            and int(selected_token_group_size) != route_block_size):
        raise ValueError(
            'QSA compact route block size must match selected_token_group_size')
    if route_block_size > 1 and (not causal or int(key_position_offset) != 0):
        raise ValueError(
            'QSA compact block route currently requires causal attention and '
            'key_position_offset=0')
    if topk_indices.dtype != torch.int32:
        topk_indices = topk_indices.to(torch.int32)
    if topk_length.dtype != torch.int32:
        topk_length = topk_length.to(torch.int32)
    if query_positions is None:
        query_positions = torch.arange(
            query.shape[0], device=query.device, dtype=torch.int32) + int(query_position_offset)
    else:
        query_positions = query_positions.to(device=query.device, dtype=torch.int32)
    if query_positions.shape != (query.shape[0], ):
        raise ValueError(
            f"QSA query_positions must have shape [{query.shape[0]}], got {tuple(query_positions.shape)}")
    query_positions = query_positions.contiguous()
    if route_block_size > 1 and actual != 'triton':
        topk_indices = qsa_expand_block_route(
            topk_indices,
            topk_length,
            query_positions,
            route_block_size,
        )
        route_block_size = 1
    # Triton and the gather path both benefit from compact metadata.  The
    # conversion does not touch Q/K/V and remains outside the autograd graph.
    topk_indices = topk_indices.contiguous()
    topk_length = topk_length.contiguous()
    if actual == 'triton':
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
    precomputed_scores = None
    if (actual == 'triton'
            and os.environ.get('MCORE_BRIDGE_QSA_SAVE_FORWARD_SCORES', '0') != '0'):
        if effective_dkv_reduction != 'atomic':
            raise RuntimeError(
                'QSA forward score reuse currently supports atomic dK/dV only')
        if query.shape[1] != 1:
            raise RuntimeError(
                'QSA forward score reuse currently supports batch size one only')
        score_k = (
            topk_indices.shape[-1]
            if route_block_size == 1
            else route_block_size * topk_indices.shape[-1] + route_block_size - 1
        )
        score_chunk = int(os.environ.get(
            'MCORE_BRIDGE_QSA_SAVE_FORWARD_SCORE_CHUNK', '1024'))
        if score_chunk <= 0:
            raise ValueError(
                'QSA forward score reuse chunk must be positive')
        score_chunks = (query.shape[0] + score_chunk - 1) // score_chunk
        precomputed_scores = torch.empty(
            (score_chunks, score_chunk, query.shape[2], score_k),
            device=query.device,
            dtype=torch.float32,
        )
    return _QSASelectedKVFunction.apply(
        query,
        key,
        value,
        topk_indices,
        topk_length,
        scale,
        bool(causal),
        actual,
        int(query_position_offset),
        int(key_position_offset),
        query_positions,
        int(query_tile_size),
        dkv_accum_dtype,
        selected_token_group_size,
        effective_dkv_reduction,
        route_block_size,
        precomputed_scores,
    )


def qsa_sparse_forward_packed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    softmax_scale: Optional[float] = None,
    backend: str = 'torch',
    query_tile_size: int = 16,
    require_backend: bool = False,
    dkv_accum_dtype: str = 'bf16',
    selected_token_group_size: Optional[int] = None,
    dkv_reduction: str = 'atomic',
    causal: bool = True,
    route_block_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run selected-KV attention over packed THD segments.

    ``topk_indices`` contains segment-local token IDs, or compact complete-
    block IDs when ``route_block_size > 1``, as produced by
    ``QSAIndexer.select_topk_packed``.  The Triton/atomic path uses one varlen
    grid for all segments, or two device-filtered grids when a large pack
    mixes short and long segments.  Aligned packed segmented routes reuse the
    block-owner schedule in one global launch; other segment layouts retain
    the auditable per-segment fallback.
    """

    if query.ndim == 3:
        query_thd = query.unsqueeze(1)
        key_thd = key.unsqueeze(1)
        value_thd = value.unsqueeze(1)
        squeeze_batch = True
    elif query.ndim == 4 and query.shape[1] == 1:
        query_thd, key_thd, value_thd = query, key, value
        squeeze_batch = False
    else:
        raise ValueError('QSA packed attention expects [T,H,D] or [T,1,H,D] tensors')
    if key_thd.ndim != 4 or value_thd.shape != key_thd.shape:
        raise ValueError('QSA packed attention K/V must share [T,1,H,D] geometry')
    if topk_indices.shape[0] != 1 or topk_indices.shape[1] != query_thd.shape[0]:
        raise ValueError('QSA packed attention metadata must be [1,T,K]')
    if (cu_seqlens_q.ndim != 1 or cu_seqlens_kv.ndim != 1 or
            cu_seqlens_q.numel() != cu_seqlens_kv.numel() or
            cu_seqlens_q.numel() < 2):
        raise ValueError('QSA packed attention requires matching cu_seqlens_q/cu_seqlens_kv')
    q_boundaries = _packed_boundaries(cu_seqlens_q)
    kv_boundaries = _packed_boundaries(cu_seqlens_kv)
    if q_boundaries[0] != 0 or kv_boundaries[0] != 0:
        raise ValueError('QSA packed cu_seqlens must start at zero')
    if q_boundaries[-1] != query_thd.shape[0] or kv_boundaries[-1] != key_thd.shape[0]:
        raise ValueError('QSA packed cu_seqlens must cover all packed Q/K/V tokens')
    if any(end < start for start, end in zip(q_boundaries[:-1], q_boundaries[1:])) or any(
            end < start for start, end in zip(kv_boundaries[:-1], kv_boundaries[1:])):
        raise ValueError('QSA packed cu_seqlens must be non-decreasing')

    total_q = query_thd.shape[0]
    if total_q == 0:
        empty_output = query_thd.new_empty((0, 1, query_thd.shape[2], query_thd.shape[3]))
        empty_lse = torch.empty(
            (1, query_thd.shape[2], 0), device=query_thd.device, dtype=torch.float32)
        return (empty_output.squeeze(1) if squeeze_batch else empty_output), empty_lse

    requested = (backend or 'torch').lower()
    resolved = resolve_qsa_backend(requested, query_thd.device, require=require_backend)
    if resolved.fallback_reason:
        logger.warning_once(
            f'QSA packed backend fallback: requested={resolved.requested}, '
            f'actual={resolved.actual}, reason={resolved.fallback_reason}')
    dkv_accum_dtype = str(dkv_accum_dtype).lower()
    if dkv_accum_dtype not in {'bf16', 'fp32'}:
        raise ValueError(
            f'unsupported QSA packed dkv_accum_dtype={dkv_accum_dtype!r}; choose bf16 or fp32')
    effective_dkv_reduction = os.environ.get(
        'MCORE_BRIDGE_QSA_DKV_REDUCTION', dkv_reduction).lower()
    if effective_dkv_reduction not in {'atomic', 'segmented'}:
        raise ValueError(
            f'unsupported QSA packed dkv_reduction={effective_dkv_reduction!r}; '
            "choose 'atomic' or 'segmented'")
    route_block_size = int(route_block_size)
    if route_block_size <= 0:
        raise ValueError('QSA packed route_block_size must be positive')
    if (route_block_size > 1 and selected_token_group_size is not None
            and int(selected_token_group_size) != route_block_size):
        raise ValueError(
            'QSA packed compact route block size must match selected_token_group_size')
    if route_block_size > 1 and not causal:
        raise ValueError(
            'QSA packed compact block route currently requires causal attention')

    # The production atomic path maps every query token to its Q/KV segment
    # with one bandwidth-only device launch.  The fallback remains available
    # for A/B and for backends whose attention itself still loops by segment.
    cu_q = cu_seqlens_q.to(device=query_thd.device, dtype=torch.int32).contiguous()
    cu_kv = cu_seqlens_kv.to(device=query_thd.device, dtype=torch.int32).contiguous()
    fused_attention_metadata = (
        resolved.actual == 'triton'
        and effective_dkv_reduction == 'atomic'
        and os.environ.get(
            'MCORE_BRIDGE_QSA_PACKED_ATTENTION_METADATA_FUSED', '1') != '0'
    )
    if fused_attention_metadata:
        from .qsa_triton import qsa_attention_packed_metadata_from_cu

        key_starts, key_lengths, local_positions = (
            qsa_attention_packed_metadata_from_cu(cu_q, cu_kv, total_q))
    else:
        num_segments = cu_q.numel() - 1
        segment_ids = torch.repeat_interleave(
            torch.arange(
                num_segments, device=query_thd.device, dtype=torch.long),
            (cu_q[1:] - cu_q[:-1]).to(torch.long),
        )
        if segment_ids.numel() != total_q:
            raise ValueError(
                'QSA packed cu_seqlens query lengths do not cover all query tokens')
        q_segment_starts = cu_q[:-1].to(torch.long).index_select(
            0, segment_ids)
        key_starts = cu_kv[:-1].to(torch.long).index_select(0, segment_ids)
        key_lengths = (cu_kv[1:] - cu_kv[:-1]).to(torch.long).index_select(
            0, segment_ids)
        local_positions = torch.arange(
            total_q,
            device=query_thd.device,
            dtype=torch.long,
        ) - q_segment_starts
    if route_block_size > 1 and resolved.actual != 'triton':
        topk_indices = qsa_expand_block_route(
            topk_indices,
            topk_length,
            local_positions,
            route_block_size,
        )
        route_block_size = 1
    packed_indices = topk_indices[0].to(
        device=query_thd.device, dtype=torch.int32).contiguous()
    packed_lengths = topk_length[0].to(
        device=query_thd.device, dtype=torch.int32).contiguous()
    key_starts = key_starts.to(torch.int32).contiguous()
    key_lengths = key_lengths.to(torch.int32).contiguous()
    local_positions = local_positions.to(torch.int32).contiguous()

    # A packed segmented reduction can reuse the unpacked block-owner
    # implementation when Q and KV segments have identical, ratio-aligned
    # boundaries.  In that case local token/block IDs have a unique global
    # physical mapping, and global query positions preserve the causal order.
    # The guard is structural (validated by _packed_boundaries), so it does
    # not introduce a device-to-host synchronization into graph capture.
    packed_block_owned = (
        resolved.actual == 'triton'
        and effective_dkv_reduction == 'segmented'
        and selected_token_group_size is not None
        and q_boundaries == kv_boundaries
        and all(
            boundary % int(selected_token_group_size) == 0
            for boundary in kv_boundaries
        )
    )
    if packed_block_owned:
        ratio = int(selected_token_group_size)
        global_positions = key_starts + local_positions
        if route_block_size > 1:
            global_indices = torch.where(
                packed_indices >= 0,
                packed_indices + key_starts[:, None] // ratio,
                torch.full_like(packed_indices, -1),
            )
        else:
            global_indices = torch.where(
                packed_indices >= 0,
                packed_indices + key_starts[:, None],
                torch.full_like(packed_indices, -1),
            )
        packed_output, packed_lse = qsa_sparse_forward(
            query_thd,
            key_thd,
            value_thd,
            global_indices.unsqueeze(0),
            packed_lengths.unsqueeze(0),
            softmax_scale=softmax_scale,
            causal=causal,
            backend='triton',
            query_positions=global_positions,
            require_backend=require_backend,
            dkv_accum_dtype=dkv_accum_dtype,
            selected_token_group_size=ratio,
            dkv_reduction='segmented',
            route_block_size=route_block_size,
        )
        output = packed_output
        return (output.squeeze(1) if squeeze_batch else output), packed_lse

    if resolved.actual == 'triton' and effective_dkv_reduction == 'atomic':
        scale = _normalise_scale(softmax_scale, query_thd.shape[-1])
        packed_output, packed_lse = _QSASelectedKVPackedFunction.apply(
            query_thd[:, 0].contiguous(),
            key_thd[:, 0].contiguous(),
            value_thd[:, 0].contiguous(),
            packed_indices,
            packed_lengths,
            key_starts,
            key_lengths,
            local_positions,
            scale,
            bool(causal),
            'triton',
            0,
            dkv_accum_dtype,
            effective_dkv_reduction,
            route_block_size,
        )
        packed_lse = packed_lse.unsqueeze(0)
        output = packed_output.unsqueeze(1)
        return (output.squeeze(1) if squeeze_batch else output), packed_lse

    outputs = []
    lses = []
    for q_start, q_end, kv_start, kv_end in zip(
            q_boundaries[:-1], q_boundaries[1:], kv_boundaries[:-1], kv_boundaries[1:]):
        if q_end < q_start or kv_end < kv_start:
            raise ValueError('QSA packed cu_seqlens must be non-decreasing')
        if q_end == q_start:
            continue
        if q_end > query_thd.shape[0] or kv_end > key_thd.shape[0]:
            raise ValueError('QSA packed cu_seqlens exceed the input token count')
        segment_query = query_thd[q_start:q_end]
        segment_key = key_thd[kv_start:kv_end]
        segment_value = value_thd[kv_start:kv_end]
        segment_indices = topk_indices[:, q_start:q_end]
        segment_lengths = topk_length[:, q_start:q_end]
        segment_output, segment_lse = qsa_sparse_forward(
            segment_query,
            segment_key,
            segment_value,
            segment_indices,
            segment_lengths,
            softmax_scale=softmax_scale,
            causal=causal,
            backend=resolved.actual,
            query_positions=torch.arange(
                q_end - q_start, device=query_thd.device, dtype=torch.int32),
            query_tile_size=query_tile_size,
            require_backend=require_backend,
            dkv_accum_dtype=dkv_accum_dtype,
            selected_token_group_size=selected_token_group_size,
            dkv_reduction=effective_dkv_reduction,
            route_block_size=route_block_size,
        )
        outputs.append(segment_output)
        lses.append(segment_lse)
    if not outputs:
        empty_output = query_thd.new_empty((0, 1, query_thd.shape[2], query_thd.shape[3]))
        empty_lse = torch.empty(
            (1, query_thd.shape[2], 0), device=query_thd.device, dtype=torch.float32)
        return (empty_output.squeeze(1) if squeeze_batch else empty_output), empty_lse
    output = torch.cat(outputs, dim=0)
    lse = torch.cat(lses, dim=2)
    return (output.squeeze(1) if squeeze_batch else output), lse


def qsa_sparse_forward_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    query_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Small-shape reference helper used by parity tests."""

    _, _, _, _, _, head_dim, indices, lengths = _validate_inputs(query, key, value, topk_indices, topk_length)
    scale = _normalise_scale(softmax_scale, head_dim)
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)
    if lengths.dtype != torch.int32:
        lengths = lengths.to(torch.int32)
    if query_positions is None:
        query_positions = torch.arange(query.shape[0], device=query.device, dtype=torch.int32)
    return _torch_selected_kv_forward(query, key, value, indices, lengths, scale, causal, 0, 0, query_positions,
                                      query.shape[0])


__all__ = [
    "QSA_BACKENDS",
    "QSAKernelError",
    "QSAKernelUnavailable",
    "QSAResolvedBackend",
    "qsa_expand_block_route",
    "qsa_sparse_forward",
    "qsa_sparse_forward_packed",
    "qsa_sparse_forward_reference",
    "qsa_reconstruct_cp_tensor",
    "resolve_qsa_backend",
]
