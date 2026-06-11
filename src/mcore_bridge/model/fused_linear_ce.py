# Copyright (c) ModelScope Contributors. All rights reserved.
"""Vocab-parallel fused linear cross entropy helpers.

This path keeps the logits tensor scoped to one supervised-token chunk. The
Triton kernels compute CE statistics and overwrite the chunk logits with
grad-logits in-place, so the full sequence logits are never materialized.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on environments without Triton.
    triton = None
    tl = None


_MAX_ROW_BLOCK = 32768


def _require_triton():
    if triton is None:
        raise RuntimeError('LINEAR_CE_IMPL=triton requires the triton package to be installed.')


def _row_block_size(n_cols: int) -> int:
    _require_triton()
    return min(_MAX_ROW_BLOCK, triton.next_power_of_2(n_cols))


@triton.jit
def _vp_ce_stats_kernel(
    logits_ptr,
    target_ptr,
    global_max_ptr,
    local_sum_ptr,
    local_target_logit_ptr,
    n_cols: tl.constexpr,
    vocab_start_index: tl.constexpr,
    ignore_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(target_ptr + row)
    row_logits_ptr = logits_ptr + row * n_cols
    max_val = tl.load(global_max_ptr + row)

    denom = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_logits_ptr + offsets, mask=offsets < n_cols, other=-float('inf')).to(tl.float32)
        denom += tl.sum(tl.exp(logits - max_val))

    local_target = target - vocab_start_index
    has_target = (target != ignore_index) & (local_target >= 0) & (local_target < n_cols)
    target_logit = tl.load(row_logits_ptr + local_target, mask=has_target, other=0.0).to(tl.float32)

    tl.store(local_sum_ptr + row, denom)
    tl.store(local_target_logit_ptr + row, target_logit)


@triton.jit
def _vp_ce_grad_logits_kernel(
    logits_ptr,
    target_ptr,
    grad_output_ptr,
    global_max_ptr,
    global_sum_ptr,
    n_cols: tl.constexpr,
    vocab_start_index: tl.constexpr,
    ignore_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(target_ptr + row)
    row_logits_ptr = logits_ptr + row * n_cols
    max_val = tl.load(global_max_ptr + row)
    denom = tl.load(global_sum_ptr + row)
    grad_scale = tl.load(grad_output_ptr + row).to(tl.float32)
    local_target = target - vocab_start_index
    has_target = (target != ignore_index) & (local_target >= 0) & (local_target < n_cols)

    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits = tl.load(row_logits_ptr + offsets, mask=mask, other=-float('inf')).to(tl.float32)
        grad_logits = tl.exp(logits - max_val) / denom
        grad_logits -= tl.where(has_target & (offsets == local_target), 1.0, 0.0)
        grad_logits *= grad_scale
        tl.store(row_logits_ptr + offsets, grad_logits, mask=mask)


def fused_ce_stats(
    logits: torch.Tensor,
    target: torch.Tensor,
    global_max: torch.Tensor,
    *,
    vocab_start_index: int,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local sum-exp and local target logits for TP CE.

    Args:
        logits: Contiguous ``[tokens, vocab_partition]`` chunk.
        target: Contiguous global vocab ids for the same tokens.
        global_max: Per-token max after TP max reduction.
    """

    _require_triton()
    if logits.dim() != 2:
        raise ValueError(f'logits must be [tokens, vocab_partition], got {tuple(logits.shape)}')
    if not logits.is_contiguous():
        logits = logits.contiguous()
    n_rows, n_cols = logits.shape
    local_sum = torch.empty((n_rows,), dtype=torch.float32, device=logits.device)
    local_target_logit = torch.empty((n_rows,), dtype=torch.float32, device=logits.device)
    if n_rows == 0:
        return local_sum, local_target_logit

    _vp_ce_stats_kernel[(n_rows,)](
        logits,
        target,
        global_max,
        local_sum,
        local_target_logit,
        n_cols=n_cols,
        vocab_start_index=vocab_start_index,
        ignore_index=ignore_index,
        BLOCK_SIZE=_row_block_size(n_cols),
        num_warps=32,
    )
    return local_sum, local_target_logit


def fused_ce_grad_logits_(
    logits: torch.Tensor,
    target: torch.Tensor,
    grad_output: torch.Tensor,
    global_max: torch.Tensor,
    global_sum: torch.Tensor,
    *,
    vocab_start_index: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Overwrite logits with TP-aware CE grad-logits and return it."""

    _require_triton()
    if logits.dim() != 2:
        raise ValueError(f'logits must be [tokens, vocab_partition], got {tuple(logits.shape)}')
    if not logits.is_contiguous():
        raise ValueError('logits must be contiguous for in-place fused CE gradient.')
    n_rows, n_cols = logits.shape
    if n_rows == 0:
        return logits

    _vp_ce_grad_logits_kernel[(n_rows,)](
        logits,
        target,
        grad_output,
        global_max,
        global_sum,
        n_cols=n_cols,
        vocab_start_index=vocab_start_index,
        ignore_index=ignore_index,
        BLOCK_SIZE=_row_block_size(n_cols),
        num_warps=32,
    )
    return logits


class VocabParallelFusedLinearCrossEntropy(torch.autograd.Function):
    """Chunked lm_head + TP cross entropy without full logits materialization."""

    @staticmethod
    def forward(ctx, hidden_states, output_weight, labels, tp_group, vocab_start_index, chunk_size,
                reduce_grad_input):
        if labels.dim() != 2:
            raise ValueError(f'labels must be [batch, sequence], got shape: {tuple(labels.shape)}')
        if hidden_states.dim() != 3:
            raise ValueError(f'hidden_states must be [sequence, batch, hidden], got shape: {tuple(hidden_states.shape)}')
        seq_len, batch_size, hidden_size = hidden_states.shape
        if labels.shape != (batch_size, seq_len):
            raise ValueError(
                f'labels shape must match hidden states as [batch, sequence]. Got labels={tuple(labels.shape)}, '
                f'hidden_states={tuple(hidden_states.shape)}')
        if chunk_size <= 0:
            raise ValueError(f'chunk_size must be > 0. Got: {chunk_size}')

        labels_t = labels.transpose(0, 1).contiguous()
        hidden_flat = hidden_states.contiguous().view(seq_len * batch_size, hidden_size)
        target_flat = labels_t.view(-1)
        supervised_indices = torch.nonzero(target_flat != -100, as_tuple=False).flatten()
        losses_flat = torch.zeros((seq_len * batch_size,), dtype=torch.float32, device=hidden_states.device)
        global_max_values = torch.empty((supervised_indices.numel(),), dtype=torch.float32, device=hidden_states.device)
        global_sum_values = torch.empty((supervised_indices.numel(),), dtype=torch.float32, device=hidden_states.device)

        for chunk_start in range(0, supervised_indices.numel(), chunk_size):
            chunk_end = min(supervised_indices.numel(), chunk_start + chunk_size)
            token_indices = supervised_indices[chunk_start:chunk_end]
            hidden_chunk = hidden_flat.index_select(0, token_indices)
            target = target_flat.index_select(0, token_indices).contiguous()
            logits = torch.matmul(hidden_chunk, output_weight.t()).contiguous()

            local_max = logits.float().amax(dim=-1)
            global_max = local_max
            if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
                torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

            local_sum, local_target_logit = fused_ce_stats(
                logits,
                target,
                global_max,
                vocab_start_index=vocab_start_index,
            )
            global_sum = local_sum
            target_logit = local_target_logit
            if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
                torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM, group=tp_group)
                torch.distributed.all_reduce(target_logit, op=torch.distributed.ReduceOp.SUM, group=tp_group)

            chunk_loss = torch.log(global_sum) + global_max - target_logit
            losses_flat.index_copy_(0, token_indices, chunk_loss)
            global_max_values[chunk_start:chunk_end].copy_(global_max)
            global_sum_values[chunk_start:chunk_end].copy_(global_sum)

        ctx.save_for_backward(hidden_states, output_weight, target_flat, supervised_indices, global_max_values,
                              global_sum_values)
        ctx.tp_group = tp_group
        ctx.vocab_start_index = vocab_start_index
        ctx.chunk_size = chunk_size
        ctx.reduce_grad_input = reduce_grad_input
        return losses_flat.view(seq_len, batch_size).transpose(0, 1).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, output_weight, target_flat, supervised_indices, global_max_values, global_sum_values = (
            ctx.saved_tensors)
        tp_group = ctx.tp_group
        vocab_start_index = ctx.vocab_start_index
        chunk_size = ctx.chunk_size
        reduce_grad_input = ctx.reduce_grad_input
        seq_len, batch_size, hidden_size = hidden_states.shape

        hidden_flat = hidden_states.contiguous().view(seq_len * batch_size, hidden_size)
        grad_output_flat = grad_output.transpose(0, 1).contiguous().view(-1).float()
        grad_hidden_flat = torch.zeros_like(hidden_flat) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(output_weight) if ctx.needs_input_grad[1] else None

        for chunk_start in range(0, supervised_indices.numel(), chunk_size):
            chunk_end = min(supervised_indices.numel(), chunk_start + chunk_size)
            token_indices = supervised_indices[chunk_start:chunk_end]
            hidden_chunk = hidden_flat.index_select(0, token_indices)
            target = target_flat.index_select(0, token_indices).contiguous()
            logits = torch.matmul(hidden_chunk, output_weight.t()).contiguous()
            grad_output_chunk = grad_output_flat.index_select(0, token_indices).contiguous()
            grad_logits = fused_ce_grad_logits_(
                logits,
                target,
                grad_output_chunk,
                global_max_values[chunk_start:chunk_end].contiguous(),
                global_sum_values[chunk_start:chunk_end].contiguous(),
                vocab_start_index=vocab_start_index,
            )

            if grad_hidden_flat is not None:
                grad_hidden_chunk = torch.matmul(grad_logits, output_weight)
                if reduce_grad_input and tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
                    torch.distributed.all_reduce(grad_hidden_chunk, op=torch.distributed.ReduceOp.SUM, group=tp_group)
                grad_hidden_flat.index_copy_(0, token_indices, grad_hidden_chunk.to(dtype=hidden_states.dtype))

            if grad_weight is not None:
                grad_weight_chunk = torch.matmul(grad_logits.t(), hidden_chunk)
                grad_weight.add_(grad_weight_chunk.to(dtype=grad_weight.dtype))

        grad_hidden = grad_hidden_flat.view(seq_len, batch_size, hidden_size) if grad_hidden_flat is not None else None
        return grad_hidden, grad_weight, None, None, None, None, None


def _group_rank(tp_group) -> int:
    if tp_group is None or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank(tp_group)


def _group_world_size(tp_group) -> int:
    if tp_group is None or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(tp_group)


def _group_global_rank(tp_group, group_rank: int) -> int:
    if tp_group is None:
        return group_rank
    if hasattr(torch.distributed, 'get_global_rank'):
        return torch.distributed.get_global_rank(tp_group, group_rank)
    return torch.distributed.distributed_c10d._get_process_group_ranks(tp_group)[group_rank]


def _broadcast_from_group_rank(tensor: torch.Tensor, tp_group, group_rank: int) -> torch.Tensor:
    if _group_world_size(tp_group) > 1:
        torch.distributed.broadcast(tensor, src=_group_global_rank(tp_group, group_rank), group=tp_group)
    return tensor


class VocabParallelStreamingFusedLinearCrossEntropy(torch.autograd.Function):
    """SP-aware chunked lm_head + TP CE without full hidden all-gather.

    Megatron's sequence-parallel output layer gathers the whole sequence before
    the vocab-parallel lm_head so every vocab rank can see every token. This
    variant streams one sequence-parallel owner's supervised-token chunk at a
    time through a broadcast, computes TP CE, then keeps only the owner shard's
    hidden gradient. It preserves vocab-parallel softmax semantics while avoiding
    a persistent full-sequence hidden buffer in the loss path.
    """

    @staticmethod
    def forward(ctx, hidden_states, output_weight, labels, tp_group, vocab_start_index, chunk_size):
        if labels.dim() != 2:
            raise ValueError(f'labels must be [batch, sequence], got shape: {tuple(labels.shape)}')
        if hidden_states.dim() != 3:
            raise ValueError(f'hidden_states must be [local_sequence, batch, hidden], got: {tuple(hidden_states.shape)}')
        local_seq_len, batch_size, hidden_size = hidden_states.shape
        tp_size = _group_world_size(tp_group)
        tp_rank = _group_rank(tp_group)
        full_seq_len = labels.shape[1]
        if labels.shape[0] != batch_size:
            raise ValueError(
                f'labels batch must match hidden states. Got labels={tuple(labels.shape)}, '
                f'hidden_states={tuple(hidden_states.shape)}')
        if full_seq_len != local_seq_len * tp_size:
            raise ValueError(
                f'streaming linear CE expects labels sequence length to equal local_sequence * TP size. '
                f'Got labels={tuple(labels.shape)}, hidden_states={tuple(hidden_states.shape)}, tp_size={tp_size}.')
        if chunk_size <= 0:
            raise ValueError(f'chunk_size must be > 0. Got: {chunk_size}')

        labels_t = labels.transpose(0, 1).contiguous()
        target_flat = labels_t.view(-1)
        hidden_flat = hidden_states.contiguous().view(local_seq_len * batch_size, hidden_size)
        losses_flat = torch.zeros((full_seq_len * batch_size,), dtype=torch.float32, device=hidden_states.device)

        for owner_rank in range(tp_size):
            owner_offset = owner_rank * local_seq_len * batch_size
            owner_target = target_flat[owner_offset:owner_offset + local_seq_len * batch_size]
            owner_supervised = torch.nonzero(owner_target != -100, as_tuple=False).flatten()
            for chunk_start in range(0, owner_supervised.numel(), chunk_size):
                chunk_end = min(owner_supervised.numel(), chunk_start + chunk_size)
                owner_local_indices = owner_supervised[chunk_start:chunk_end]
                n_tokens = owner_local_indices.numel()
                if owner_rank == tp_rank:
                    hidden_chunk = hidden_flat.index_select(0, owner_local_indices)
                else:
                    hidden_chunk = torch.empty((n_tokens, hidden_size), dtype=hidden_states.dtype,
                                               device=hidden_states.device)
                _broadcast_from_group_rank(hidden_chunk, tp_group, owner_rank)

                target = owner_target.index_select(0, owner_local_indices).contiguous()
                logits = torch.matmul(hidden_chunk, output_weight.t()).contiguous()
                local_max = logits.float().amax(dim=-1)
                global_max = local_max
                if tp_size > 1:
                    torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

                local_sum, local_target_logit = fused_ce_stats(
                    logits,
                    target,
                    global_max,
                    vocab_start_index=vocab_start_index,
                )
                global_sum = local_sum
                target_logit = local_target_logit
                if tp_size > 1:
                    torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM, group=tp_group)
                    torch.distributed.all_reduce(target_logit, op=torch.distributed.ReduceOp.SUM, group=tp_group)

                chunk_loss = torch.log(global_sum) + global_max - target_logit
                losses_flat.index_copy_(0, owner_offset + owner_local_indices, chunk_loss)

        ctx.save_for_backward(hidden_states, output_weight, target_flat)
        ctx.tp_group = tp_group
        ctx.vocab_start_index = vocab_start_index
        ctx.chunk_size = chunk_size
        ctx.local_seq_len = local_seq_len
        ctx.batch_size = batch_size
        return losses_flat.view(full_seq_len, batch_size).transpose(0, 1).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, output_weight, target_flat = ctx.saved_tensors
        tp_group = ctx.tp_group
        vocab_start_index = ctx.vocab_start_index
        chunk_size = ctx.chunk_size
        local_seq_len = ctx.local_seq_len
        batch_size = ctx.batch_size
        tp_size = _group_world_size(tp_group)
        tp_rank = _group_rank(tp_group)
        _, _, hidden_size = hidden_states.shape

        hidden_flat = hidden_states.contiguous().view(local_seq_len * batch_size, hidden_size)
        grad_output_flat = grad_output.transpose(0, 1).contiguous().view(-1).float()
        grad_hidden_flat = torch.zeros_like(hidden_flat) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(output_weight) if ctx.needs_input_grad[1] else None

        for owner_rank in range(tp_size):
            owner_offset = owner_rank * local_seq_len * batch_size
            owner_target = target_flat[owner_offset:owner_offset + local_seq_len * batch_size]
            owner_supervised = torch.nonzero(owner_target != -100, as_tuple=False).flatten()
            for chunk_start in range(0, owner_supervised.numel(), chunk_size):
                chunk_end = min(owner_supervised.numel(), chunk_start + chunk_size)
                owner_local_indices = owner_supervised[chunk_start:chunk_end]
                n_tokens = owner_local_indices.numel()
                if owner_rank == tp_rank:
                    hidden_chunk = hidden_flat.index_select(0, owner_local_indices)
                else:
                    hidden_chunk = torch.empty((n_tokens, hidden_size), dtype=hidden_states.dtype,
                                               device=hidden_states.device)
                _broadcast_from_group_rank(hidden_chunk, tp_group, owner_rank)

                target = owner_target.index_select(0, owner_local_indices).contiguous()
                logits = torch.matmul(hidden_chunk, output_weight.t()).contiguous()
                local_max = logits.float().amax(dim=-1)
                global_max = local_max
                if tp_size > 1:
                    torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
                local_sum, _ = fused_ce_stats(
                    logits,
                    target,
                    global_max,
                    vocab_start_index=vocab_start_index,
                )
                global_sum = local_sum
                if tp_size > 1:
                    torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM, group=tp_group)

                grad_output_chunk = grad_output_flat.index_select(0, owner_offset + owner_local_indices).contiguous()
                grad_logits = fused_ce_grad_logits_(
                    logits,
                    target,
                    grad_output_chunk,
                    global_max,
                    global_sum,
                    vocab_start_index=vocab_start_index,
                )

                if grad_hidden_flat is not None:
                    grad_hidden_chunk = torch.matmul(grad_logits, output_weight)
                    if tp_size > 1:
                        torch.distributed.all_reduce(grad_hidden_chunk, op=torch.distributed.ReduceOp.SUM,
                                                     group=tp_group)
                    if owner_rank == tp_rank:
                        grad_hidden_flat.index_copy_(0, owner_local_indices,
                                                     grad_hidden_chunk.to(dtype=hidden_states.dtype))

                if grad_weight is not None:
                    grad_weight_chunk = torch.matmul(grad_logits.t(), hidden_chunk)
                    grad_weight.add_(grad_weight_chunk.to(dtype=grad_weight.dtype))

        grad_hidden = grad_hidden_flat.view(local_seq_len, batch_size, hidden_size) \
            if grad_hidden_flat is not None else None
        return grad_hidden, grad_weight, None, None, None, None
