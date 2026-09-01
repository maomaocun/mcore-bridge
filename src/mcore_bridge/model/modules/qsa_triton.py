# Copyright (c) ModelScope Contributors. All rights reserved.
"""Small SM90 Triton building blocks used by the QSA selected-KV backend.

The kernels in this module deliberately have a narrow contract.  The vendor
GEMM still owns indexer projection; an SM90-only postprocess kernel fuses its
RMSNorm, RoPE, and block pooling.  Top-k orchestration and autograd remain in
Python so the reference path stays easy to compare with the HF model and a
missing Triton installation has an explicit, testable fallback.

The production indexer kernel follows the inference-side QSA pattern: it
computes score tiles, packs score/id keys, and applies a device-side Top-K
selection before expanding complete blocks into token IDs.  Its compatibility
score kernel still writes one key-block tile at a time for FP32/debug callers;
neither path allocates a ``[B, S, S/R]`` score tensor.  The attention kernel
assigns one program to a query/small-GQA-head tile and performs online softmax
while walking the selected token list.  An optional FlashAttention varlen CSR
experiment is kept behind an explicit environment variable; native Triton
remains default.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on installations without Triton
    triton = None
    tl = None


TRITON_AVAILABLE = triton is not None

try:
    from flash_attn_interface import flash_attn_varlen_func
except ImportError:  # pragma: no cover - optional training-image acceleration
    flash_attn_varlen_func = None


FLASH_ATTN_AVAILABLE = flash_attn_varlen_func is not None


def is_sm90(device: Optional[torch.device] = None) -> bool:
    """Return whether ``device`` is a Hopper/SM90 CUDA device."""

    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        return False
    try:
        if device is None:
            device_index = torch.cuda.current_device()
        else:
            device_obj = torch.device(device)
            if device_obj.type != 'cuda':
                return False
            device_index = device_obj.index
            if device_index is None:
                device_index = torch.cuda.current_device()
        return tuple(torch.cuda.get_device_capability(device_index)) == (9, 0)
    except (RuntimeError, TypeError, ValueError):
        return False


def _qsa_trim_causal_loop(causal: bool, sequence_extent: int, logical_k: int) -> bool:
    """Select the short-sequence loop variant without a device synchronization."""

    mode = os.environ.get(
        'MCORE_BRIDGE_QSA_TRIM_CAUSAL_LOOP',
        os.environ.get('MCORE_BRIDGE_QSA_BACKWARD_TRIM_CAUSAL_LOOP', 'auto'),
    ).lower()
    if mode not in {'auto', '0', '1'}:
        raise ValueError(
            'QSA causal-loop trimming expects one of {auto,0,1}')
    if not causal:
        return False
    return sequence_extent <= 8 * logical_k if mode == 'auto' else mode == '1'


if TRITON_AVAILABLE:

    @triton.jit
    def _qsa_indexer_fused_postprocess_kernel(
        qk_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        block_key_out_ptr,
        seq_len,
        num_blocks,
        head_dim,
        rotary_dim,
        norm_epsilon,
        stride_qks,
        stride_qkb,
        stride_qkd,
        stride_qw,
        stride_kw,
        stride_fs,
        stride_fd,
        stride_qob,
        stride_qos,
        stride_qoh,
        stride_qod,
        stride_bob,
        stride_bos,
        stride_bod,
        NUM_HEADS: tl.constexpr,
        RATIO: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fuse indexer RMSNorm, RoPE, and R-token key pooling.

        Projection remains a vendor GEMM.  One CTA owns all index heads of a
        token and, on compression-block starts, also produces the pooled key.
        Cosine and sine are supplied in the same BF16 form as the torch path,
        so this launch does not introduce a lower-precision RoPE contract.
        """

        row = tl.program_id(0)
        batch = row // seq_len
        position = row - batch * seq_len
        head_offsets = tl.arange(0, NUM_HEADS)
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim

        q_ptrs = (
            qk_ptr
            + position * stride_qks
            + batch * stride_qkb
            + (head_offsets[:, None] * head_dim + d_offsets[None, :])
            * stride_qkd
        )
        q_raw = tl.load(
            q_ptrs,
            mask=d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_inv_rms = tl.rsqrt(
            tl.sum(q_raw * q_raw, axis=1) / head_dim + norm_epsilon
        )
        q_weight = tl.load(
            q_weight_ptr + d_offsets * stride_qw,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        q_normalized = (
            q_raw * q_inv_rms[:, None] * (1.0 + q_weight[None, :])
        ).to(tl.bfloat16)

        half_rotary = rotary_dim // 2
        first_half = d_offsets < half_rotary
        partner_offsets = tl.where(
            first_half,
            d_offsets + half_rotary,
            d_offsets - half_rotary,
        )
        partner_mask = (d_offsets < rotary_dim) & (
            partner_offsets >= 0
        ) & (partner_offsets < rotary_dim)
        q_partner_ptrs = (
            qk_ptr
            + position * stride_qks
            + batch * stride_qkb
            + (
                head_offsets[:, None] * head_dim
                + partner_offsets[None, :]
            ) * stride_qkd
        )
        q_partner_raw = tl.load(
            q_partner_ptrs,
            mask=partner_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_partner_weight = tl.load(
            q_weight_ptr + partner_offsets * stride_qw,
            mask=partner_mask,
            other=0.0,
        ).to(tl.float32)
        q_partner = (
            q_partner_raw
            * q_inv_rms[:, None]
            * (1.0 + q_partner_weight[None, :])
        ).to(tl.bfloat16)
        q_partner = tl.where(first_half[None, :], -q_partner, q_partner)
        cos = tl.load(
            cos_ptr + position * stride_fs + d_offsets * stride_fd,
            mask=d_offsets < rotary_dim,
            other=0.0,
        )
        sin = tl.load(
            sin_ptr + position * stride_fs + d_offsets * stride_fd,
            mask=d_offsets < rotary_dim,
            other=0.0,
        )
        q_rope = q_normalized * cos[None, :] + q_partner * sin[None, :]
        q_value = tl.where(
            (d_offsets < rotary_dim)[None, :], q_rope, q_normalized
        )
        q_out_ptrs = (
            q_out_ptr
            + batch * stride_qob
            + position * stride_qos
            + head_offsets[:, None] * stride_qoh
            + d_offsets[None, :] * stride_qod
        )
        tl.store(q_out_ptrs, q_value, mask=d_mask[None, :])

        if position % RATIO == 0:
            block = position // RATIO
            if block < num_blocks:
                ratio_offsets = tl.arange(0, RATIO)
                key_base = NUM_HEADS * head_dim
                key_ptrs = (
                    qk_ptr
                    + (position + ratio_offsets[:, None]) * stride_qks
                    + batch * stride_qkb
                    + (key_base + d_offsets[None, :]) * stride_qkd
                )
                key_raw = tl.load(
                    key_ptrs,
                    mask=d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                pooled = (
                    tl.sum(key_raw, axis=0) / RATIO
                ).to(tl.bfloat16)
                pooled_float = pooled.to(tl.float32)
                key_inv_rms = tl.rsqrt(
                    tl.sum(pooled_float * pooled_float, axis=0)
                    / head_dim
                    + norm_epsilon
                )
                key_weight = tl.load(
                    k_weight_ptr + d_offsets * stride_kw,
                    mask=d_mask,
                    other=0.0,
                ).to(tl.float32)
                key_normalized = (
                    pooled_float * key_inv_rms * (1.0 + key_weight)
                ).to(tl.bfloat16)

                key_partner_ptrs = (
                    qk_ptr
                    + (position + ratio_offsets[:, None]) * stride_qks
                    + batch * stride_qkb
                    + (key_base + partner_offsets[None, :]) * stride_qkd
                )
                key_partner_raw = tl.load(
                    key_partner_ptrs,
                    mask=partner_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                key_partner_pooled = (
                    tl.sum(key_partner_raw, axis=0) / RATIO
                ).to(tl.bfloat16).to(tl.float32)
                key_partner_weight = tl.load(
                    k_weight_ptr + partner_offsets * stride_kw,
                    mask=partner_mask,
                    other=0.0,
                ).to(tl.float32)
                key_partner = (
                    key_partner_pooled
                    * key_inv_rms
                    * (1.0 + key_partner_weight)
                ).to(tl.bfloat16)
                key_partner = tl.where(
                    first_half, -key_partner, key_partner
                )
                key_rope = key_normalized * cos + key_partner * sin
                key_value = tl.where(
                    d_offsets < rotary_dim, key_rope, key_normalized
                )
                block_out_ptrs = (
                    block_key_out_ptr
                    + batch * stride_bob
                    + block * stride_bos
                    + d_offsets * stride_bod
                )
                tl.store(block_out_ptrs, key_value, mask=d_mask)

    @triton.jit
    def _qsa_indexer_score_kernel(
        q_ptr,
        block_key_ptr,
        out_ptr,
        seq_len,
        num_blocks,
        num_heads,
        head_dim,
        indexer_scale,
        compress_ratio,
        query_position_ptr,
        stride_qp,
        block_start,
        block_pointer_start,
        output_num_blocks,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_bb,
        stride_bs,
        stride_bd,
        stride_ob,
        stride_os,
        stride_ok,
        BLOCK_BLOCKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_HEADS: tl.constexpr,
    ):
        """Compute a contiguous tile of QSA block scores.

        ``q_ptr`` is ``[B, S, index_heads, index_dim]`` and ``block_key_ptr``
        is ``[B, S/R, index_dim]``.  The reduction is exactly the QSA
        indexer rule: ``sum_h relu(dot(q_h, pooled_k)) / sqrt(index_dim)``.
        Future/incomplete blocks are written as ``-inf`` so the host-side
        merge can use the same ordering for the torch and Triton paths.
        """

        pid = tl.program_id(0)
        blocks_per_query = tl.cdiv(output_num_blocks, BLOCK_BLOCKS)
        queries_per_batch = seq_len * blocks_per_query
        batch = pid // queries_per_batch
        rem = pid - batch * queries_per_batch
        query = rem // blocks_per_query
        tile = rem - query * blocks_per_query

        block_offsets = tile * BLOCK_BLOCKS + tl.arange(0, BLOCK_BLOCKS)
        block_ids = block_start + block_offsets
        block_mask = block_ids < num_blocks
        query_position = tl.load(query_position_ptr + query * stride_qp).to(tl.int32)
        query_complete_blocks = (query_position + 1) // compress_ratio
        block_mask = block_mask & (block_ids < query_complete_blocks)

        scores = tl.zeros((BLOCK_BLOCKS,), dtype=tl.float32)
        d_offsets = tl.arange(0, BLOCK_D)
        for head in tl.static_range(0, NUM_HEADS):
            q_ptrs = (q_ptr + batch * stride_qb + query * stride_qs + head * stride_qh + d_offsets * stride_qd)
            q = tl.load(q_ptrs, mask=d_offsets < head_dim, other=0.0).to(tl.float32)

            block_pointer_ids = block_pointer_start + block_offsets
            k_ptrs = (block_key_ptr + batch * stride_bb + block_pointer_ids[:, None] * stride_bs +
                      d_offsets[None, :] * stride_bd)
            k = tl.load(k_ptrs, mask=block_mask[:, None] & (d_offsets[None, :] < head_dim), other=0.0).to(tl.float32)
            head_score = tl.reshape(tl.dot(k, q[:, None], out_dtype=tl.float32), (BLOCK_BLOCKS,))
            scores += tl.maximum(head_score, 0.0)

        scores = tl.maximum(scores, 0.0) * indexer_scale
        scores = tl.where(block_mask, scores, -float("inf"))
        out_ptrs = out_ptr + batch * stride_ob + query * stride_os + block_offsets * stride_ok
        tl.store(out_ptrs, scores, mask=block_offsets < output_num_blocks)

    @triton.jit
    def _qsa_indexer_score_batched_kernel(
        q_ptr,
        block_key_ptr,
        out_ptr,
        query_position_ptr,
        seq_len,
        num_blocks,
        num_heads,
        head_dim,
        indexer_scale,
        compress_ratio,
        stride_qp,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_bb,
        stride_bs,
        stride_bd,
        stride_ob,
        stride_os,
        stride_ok,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Materialize one bounded QSA score slab with query-side K reuse.

        Unlike the compatibility scorer above, one CTA owns ``BLOCK_M``
        queries and reuses each compressed-key tile across them.  ReLU and
        the index-head reduction are fused before the FP32 score is written,
        so the workspace is ``[B, Q, N]`` rather than ``[B, H*Q, N]``.
        """

        query_tile = tl.program_id(0)
        block_tile = tl.program_id(1)
        batch = tl.program_id(2)
        query_offsets = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        block_offsets = block_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        query_valid = query_offsets < seq_len
        block_valid = block_offsets < num_blocks
        query_positions = tl.load(
            query_position_ptr + query_offsets * stride_qp,
            mask=query_valid,
            other=-1,
        ).to(tl.int32)
        complete_blocks = (query_positions + 1) // compress_ratio
        score_valid = (
            query_valid[:, None]
            & block_valid[None, :]
            & (block_offsets[None, :] < complete_blocks[:, None])
        )

        head_offsets = tl.arange(0, BLOCK_H)
        dim_offsets = tl.arange(0, BLOCK_D)
        q_ptrs = (
            q_ptr
            + batch * stride_qb
            + query_offsets[:, None, None] * stride_qs
            + head_offsets[None, :, None] * stride_qh
            + dim_offsets[None, None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=query_valid[:, None, None]
            & (head_offsets[None, :, None] < num_heads)
            & (dim_offsets[None, None, :] < head_dim),
            other=0.0,
        )
        q = tl.reshape(q, (BLOCK_M * BLOCK_H, BLOCK_D))
        k_ptrs = (
            block_key_ptr
            + batch * stride_bb
            + dim_offsets[:, None] * stride_bd
            + block_offsets[None, :] * stride_bs
        )
        keys = tl.load(
            k_ptrs,
            mask=(dim_offsets[:, None] < head_dim) & block_valid[None, :],
            other=0.0,
        )
        dots = tl.dot(q, keys, out_dtype=tl.float32)
        dots = tl.reshape(dots, (BLOCK_M, BLOCK_H, BLOCK_N))
        head_valid = head_offsets[None, :, None] < num_heads
        scores = tl.sum(
            tl.where(head_valid, tl.maximum(dots, 0.0), 0.0), axis=1)
        scores *= indexer_scale
        scores = tl.where(score_valid, scores, -float("inf"))
        out_ptrs = (
            out_ptr
            + batch * stride_ob
            + query_offsets[:, None] * stride_os
            + block_offsets[None, :] * stride_ok
        )
        tl.store(
            out_ptrs,
            scores,
            mask=query_valid[:, None] & block_valid[None, :],
        )

    @triton.jit
    def _qsa_pad_topk_candidates(candidates, BLOCK_TOPK: tl.constexpr, BLOCK_N: tl.constexpr):
        """Pad a score/id tile to the compile-time streaming Top-K width."""

        if BLOCK_TOPK >= 2 * BLOCK_N:
            candidates = tl.reshape(
                tl.join(candidates, tl.full((BLOCK_N,), 0, tl.int64)),
                (2 * BLOCK_N,),
            )
        if BLOCK_TOPK >= 4 * BLOCK_N:
            candidates = tl.reshape(
                tl.join(candidates, tl.full((2 * BLOCK_N,), 0, tl.int64)),
                (4 * BLOCK_N,),
            )
        if BLOCK_TOPK >= 8 * BLOCK_N:
            candidates = tl.reshape(
                tl.join(candidates, tl.full((4 * BLOCK_N,), 0, tl.int64)),
                (8 * BLOCK_N,),
            )
        if BLOCK_TOPK >= 16 * BLOCK_N:
            candidates = tl.reshape(
                tl.join(candidates, tl.full((8 * BLOCK_N,), 0, tl.int64)),
                (16 * BLOCK_N,),
            )
        if BLOCK_TOPK >= 32 * BLOCK_N:
            candidates = tl.reshape(
                tl.join(candidates, tl.full((16 * BLOCK_N,), 0, tl.int64)),
                (32 * BLOCK_N,),
            )
        return candidates

    @triton.jit
    def _qsa_indexer_fused_topk_kernel(
        q_ptr,
        block_key_ptr,
        out_ptr,
        query_position_ptr,
        seq_len,
        num_blocks,
        num_heads,
        head_dim,
        compress_ratio: tl.constexpr,
        score_scale: tl.constexpr,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_bb,
        stride_bs,
        stride_bd,
        stride_ob,
        stride_os,
        token_ids_ptr,
        BLOCK_TOPK: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_TAIL: tl.constexpr,
        INDEX_BITS: tl.constexpr,
        INDEX_MASK: tl.constexpr,
        RATIO: tl.constexpr,
        USE_TOKEN_IDS: tl.constexpr,
        FILTER_ROW_MASK: tl.constexpr,
        BATCH_ONE: tl.constexpr,
        OUTPUT_BLOCKS: tl.constexpr,
    ):
        """Fuse indexer score, causal filtering, streaming Top-K and route output.

        One program owns one ``(batch, query)`` row.  It walks compressed blocks
        in device memory and keeps only ``BLOCK_TOPK`` packed score/id keys in
        registers.  The low bits use the inverse block id, so equal scores keep
        the public QSA tie rule (lower block id first).  The kernel writes the
        final selected-token list directly; no score tile, Python merge, or
        temporary block-id tensor is needed.
        """

        pid = tl.program_id(0)
        if FILTER_ROW_MASK:
            row = pid
            if tl.load(token_ids_ptr + pid) == 0:
                return
        elif USE_TOKEN_IDS:
            row = tl.load(token_ids_ptr + pid).to(tl.int32)
        else:
            row = pid
        if BATCH_ONE:
            batch = 0
            query = row
        else:
            batch = row // seq_len
            query = row - batch * seq_len
        query_position = tl.load(query_position_ptr + query).to(tl.int32)
        complete_blocks = tl.minimum(
            num_blocks, (query_position + 1) // compress_ratio
        ).to(tl.int32)
        block_count = tl.minimum(complete_blocks, BLOCK_TOPK).to(tl.int32)

        if OUTPUT_BLOCKS:
            # Compact rows whose full causal prefix fits in the budget need no
            # scores at all.  Handle them in the same launch so production no
            # longer builds O(S) short/long row lists or synchronizes the host
            # merely to dispatch a second direct-fill kernel.
            if complete_blocks <= BLOCK_TOPK:
                block_offsets = tl.arange(0, BLOCK_TOPK)
                if BATCH_ONE:
                    out_base = out_ptr + row * stride_os
                elif USE_TOKEN_IDS:
                    out_base = out_ptr + row * stride_os
                else:
                    out_base = (
                        out_ptr
                        + batch * stride_ob
                        + query * stride_os
                    )
                tl.store(
                    out_base + block_offsets,
                    tl.where(block_offsets < block_count, block_offsets, -1),
                    mask=block_offsets < BLOCK_TOPK,
                )
                return

        head_offsets = tl.arange(0, BLOCK_H)
        dim_offsets = tl.arange(0, BLOCK_D)
        head_mask = head_offsets < num_heads
        dim_mask = dim_offsets < head_dim
        if BATCH_ONE:
            q_ptrs = (
                q_ptr
                + query * stride_qs
                + head_offsets[:, None] * stride_qh
                + dim_offsets[None, :] * stride_qd
            )
        else:
            q_ptrs = (
                q_ptr
                + batch * stride_qb
                + query * stride_qs
                + head_offsets[:, None] * stride_qh
                + dim_offsets[None, :] * stride_qd
            )
        q = tl.load(
            q_ptrs,
            mask=head_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        # Zero is reserved for an invalid packed key.  QSA indexer scores are
        # non-negative after ReLU, so raw IEEE float bits preserve ordering.
        # The signed int64 container is safe because the 32-bit score bits and
        # the block-id suffix stay below int64's sign bit.
        acc = tl.zeros((BLOCK_TOPK,), dtype=tl.int64)
        # Causal rows only have ``complete_blocks`` candidates.  Bounding the
        # loop (rather than masking future blocks after loading them) avoids
        # doing roughly half a sequence of invalid dot products on average.
        for block_start in range(0, complete_blocks, BLOCK_N):
            block_ids = block_start + tl.arange(0, BLOCK_N)
            valid = block_ids < complete_blocks
            if BATCH_ONE:
                k_ptrs = (
                    block_key_ptr
                    + block_ids[None, :] * stride_bs
                    + dim_offsets[:, None] * stride_bd
                )
            else:
                k_ptrs = (
                    block_key_ptr
                    + batch * stride_bb
                    + block_ids[None, :] * stride_bs
                    + dim_offsets[:, None] * stride_bd
                )
            keys = tl.load(
                k_ptrs,
                mask=valid[None, :] & dim_mask[:, None],
                other=0.0,
            )
            dots = tl.dot(q, keys, out_dtype=tl.float32)
            scores = tl.sum(tl.maximum(dots, 0.0), axis=0)
            scores = scores * score_scale
            score_bits = scores.to(tl.uint32, bitcast=True)
            packed = (
                ((score_bits.to(tl.int64) + 1) << INDEX_BITS)
                | (INDEX_MASK - block_ids.to(tl.int64))
            )
            packed = tl.where(valid, packed, 0)
            # Keep both operands in descending order.  Elementwise maximum of
            # two sorted rows is the exact top-k union, while the bitonic step
            # provides the layout expected by Triton's vector sort.
            acc = tl.bitonic_merge(acc)
            if tl.max(packed, axis=0) > tl.min(acc, axis=0):
                candidates = _qsa_pad_topk_candidates(
                    packed, BLOCK_TOPK, BLOCK_N
                )
                acc = tl.maximum(acc, tl.sort(candidates, descending=True))

        # The compact internal ABI consumes a set of complete-block IDs; it
        # does not require score order.  Short rows returned above, while the
        # remaining saturated rows already hold the exact Top-K in bitonic
        # layout.  Avoiding one more 512-wide sort shortens the long-row hot
        # path.  The public token route keeps canonical descending-score order
        # for compatibility/debugging.
        if not OUTPUT_BLOCKS:
            acc = tl.sort(acc, descending=True)
        block_offsets = tl.arange(0, BLOCK_TOPK)
        packed_ids = (acc & INDEX_MASK).to(tl.int32)
        block_ids = INDEX_MASK - packed_ids
        valid_blocks = (acc != 0) & (block_offsets < block_count)
        # Preserve the canonical no-truncation order used by the direct-fill
        # fast path.  When every visible complete block fits in the budget the
        # score sort changes no selected set, and local block order avoids a
        # needless reduction-order difference in the downstream attention.
        block_ids = tl.where(
            (complete_blocks <= BLOCK_TOPK) & valid_blocks,
            block_offsets,
            block_ids,
        )
        block_ids = tl.where(valid_blocks, block_ids, -1)
        if BATCH_ONE:
            out_base = out_ptr + row * stride_os
        elif USE_TOKEN_IDS:
            out_base = out_ptr + row * stride_os
        else:
            out_base = out_ptr + batch * stride_ob + query * stride_os

        if OUTPUT_BLOCKS:
            # A diagnostic split-expansion route keeps the Top-K producer's
            # live state limited to score/id selection.  The host then runs a
            # tiny bandwidth-only expansion kernel.  This is useful on SM90
            # because keeping 2K token stores in the same CTA can extend the
            # register live range around the 512-entry Top-K accumulator.
            tl.store(
                out_base + block_offsets,
                tl.where(valid_blocks, block_ids, -1),
                mask=block_offsets < BLOCK_TOPK,
            )
        else:
            # Expand selected complete blocks directly into token IDs.  Keeping
            # the expansion here removes a second global workspace at 256K.
            for token_offset in tl.static_range(0, RATIO):
                output_offsets = block_offsets * compress_ratio + token_offset
                token_ids = block_ids * compress_ratio + token_offset
                tl.store(
                    out_base + output_offsets,
                    tl.where(valid_blocks, token_ids, -1),
                    mask=output_offsets < BLOCK_TOPK * RATIO,
                )

            # The in-progress causal block is appended after the selected
            # complete blocks.  Its source position is the true frontier, not
            # the budget boundary; this is important once complete_blocks >
            # BLOCK_TOPK.
            tail_offsets = tl.arange(0, BLOCK_TAIL)
            tail_output = block_count * compress_ratio + tail_offsets
            tail_values = complete_blocks * compress_ratio + tail_offsets
            tail_valid = tail_values <= query_position
            tl.store(
                out_base + tail_output,
                tl.where(tail_valid, tail_values, -1),
                mask=tail_offsets < RATIO - 1,
            )
            # For rows with fewer than ``BLOCK_TOPK`` complete blocks, the tail
            # lives inside the block-derived prefix.  Clear the fixed-width
            # tail padding explicitly; the output tensor is intentionally
            # uninitialized to avoid a separate full-buffer fill launch.
            tl.store(
                out_base + BLOCK_TOPK * compress_ratio + tail_offsets,
                -1,
                mask=(block_count < BLOCK_TOPK) & (tail_offsets < RATIO - 1),
            )

    @triton.jit
    def _qsa_indexer_expand_block_topk_kernel(
        block_ptr,
        out_ptr,
        query_position_ptr,
        seq_len,
        num_blocks,
        compress_ratio,
        stride_bb,
        stride_bs,
        stride_ob,
        stride_os,
        BLOCK_TOPK: tl.constexpr,
        BLOCK_TAIL: tl.constexpr,
        RATIO: tl.constexpr,
    ):
        """Expand standard-BSHD block IDs into the public token list."""

        row = tl.program_id(0)
        batch = row // seq_len
        query = row - batch * seq_len
        query_position = tl.load(query_position_ptr + query).to(tl.int32)
        complete_blocks = tl.minimum(
            num_blocks, (query_position + 1) // compress_ratio
        ).to(tl.int32)
        block_count = tl.minimum(complete_blocks, BLOCK_TOPK).to(tl.int32)
        block_offsets = tl.arange(0, BLOCK_TOPK)
        block_ids = tl.load(
            block_ptr
            + batch * stride_bb
            + query * stride_bs
            + block_offsets,
        ).to(tl.int32)
        valid_blocks = (
            (block_offsets < block_count)
            & (block_ids >= 0)
            & (block_ids < num_blocks)
        )
        out_base = out_ptr + batch * stride_ob + query * stride_os
        for token_offset in tl.static_range(0, RATIO):
            output_offsets = block_offsets * compress_ratio + token_offset
            token_ids = block_ids * compress_ratio + token_offset
            tl.store(
                out_base + output_offsets,
                tl.where(valid_blocks, token_ids, -1),
                mask=output_offsets < BLOCK_TOPK * compress_ratio,
            )
        tail_offsets = tl.arange(0, BLOCK_TAIL)
        tail_output = block_count * compress_ratio + tail_offsets
        tail_values = complete_blocks * compress_ratio + tail_offsets
        tail_valid = tail_values <= query_position
        tl.store(
            out_base + tail_output,
            tl.where(tail_valid, tail_values, -1),
            mask=tail_offsets < RATIO - 1,
        )
        tl.store(
            out_base + BLOCK_TOPK * compress_ratio + tail_offsets,
            -1,
            mask=(block_count < BLOCK_TOPK) & (tail_offsets < RATIO - 1),
        )

    @triton.jit
    def _qsa_expand_compact_route_kernel(
        block_ptr,
        length_ptr,
        out_ptr,
        query_position_ptr,
        seq_len,
        stride_bb,
        stride_bs,
        stride_lb,
        stride_ls,
        stride_ob,
        stride_os,
        ROUTE_SLOTS: tl.constexpr,
        BLOCK_ROUTE: tl.constexpr,
        BLOCK_TAIL: tl.constexpr,
        RATIO: tl.constexpr,
    ):
        """Expand compact block IDs into one final-width token buffer."""

        row = tl.program_id(0)
        batch = row // seq_len
        query = row - batch * seq_len
        length = tl.load(
            length_ptr + batch * stride_lb + query * stride_ls
        ).to(tl.int32)
        max_length = ROUTE_SLOTS * RATIO + RATIO - 1
        length = tl.maximum(0, tl.minimum(length, max_length))
        selected_blocks = tl.minimum(length // RATIO, ROUTE_SLOTS)
        tail_count = length - selected_blocks * RATIO

        block_offsets = tl.arange(0, BLOCK_ROUTE)
        block_mask = block_offsets < ROUTE_SLOTS
        block_ids = tl.load(
            block_ptr
            + batch * stride_bb
            + query * stride_bs
            + block_offsets,
            mask=block_mask,
            other=-1,
        ).to(tl.int32)
        valid_blocks = (
            block_mask
            & (block_offsets < selected_blocks)
            & (block_ids >= 0)
        )
        out_base = out_ptr + batch * stride_ob + query * stride_os
        for lane in tl.static_range(0, RATIO):
            output_offsets = block_offsets * RATIO + lane
            tl.store(
                out_base + output_offsets,
                tl.where(valid_blocks, block_ids * RATIO + lane, -1),
                mask=block_mask,
            )

        tail_offsets = tl.arange(0, BLOCK_TAIL)
        tail_mask = tail_offsets < RATIO - 1
        # Initialize the fixed tail suffix before a saturated row potentially
        # writes its dynamic causal tail to those same positions.
        tl.store(
            out_base + ROUTE_SLOTS * RATIO + tail_offsets,
            -1,
            mask=tail_mask,
        )
        query_position = tl.load(query_position_ptr + query).to(tl.int32)
        tail_start = ((query_position + 1) // RATIO) * RATIO
        tail_output = selected_blocks * RATIO + tail_offsets
        tl.store(
            out_base + tail_output,
            tail_start + tail_offsets,
            mask=tail_mask & (tail_offsets < tail_count),
        )

    @triton.jit
    def _qsa_cp_request_mask_kernel(
        route_ptr,
        length_ptr,
        request_mask_ptr,
        route_numel,
        global_seq_len,
        local_key_len,
        chunk_len,
        rank,
        ROUTE_SLOTS: tl.constexpr,
        CP_SIZE: tl.constexpr,
        ZIGZAG: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Mark unique remote token requests without full owner/local tensors."""

        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        in_bounds = offsets < route_numel
        row = offsets // ROUTE_SLOTS
        slot = offsets - row * ROUTE_SLOTS
        length = tl.load(length_ptr + row, mask=in_bounds, other=0).to(tl.int32)
        token = tl.load(route_ptr + offsets, mask=in_bounds, other=-1).to(tl.int32)
        valid = (
            in_bounds
            & (slot < length)
            & (token >= 0)
            & (token < global_seq_len)
        )
        if ZIGZAG:
            chunk = token // chunk_len
            within = token - chunk * chunk_len
            owner = tl.where(chunk < CP_SIZE, chunk, 2 * CP_SIZE - 1 - chunk)
            local = tl.where(chunk < CP_SIZE, within, chunk_len + within)
        else:
            owner = token // local_key_len
            local = token - owner * local_key_len
        remote = valid & (owner != rank)
        safe_owner = tl.maximum(0, tl.minimum(owner, CP_SIZE - 1))
        safe_local = tl.maximum(0, tl.minimum(local, local_key_len - 1))
        tl.atomic_xchg(
            request_mask_ptr + safe_owner * local_key_len + safe_local,
            1,
            mask=remote,
        )

    @triton.jit
    def _qsa_cp_compact_request_mask_kernel(
        block_ptr,
        length_ptr,
        query_position_ptr,
        request_mask_ptr,
        route_numel,
        seq_len,
        global_seq_len,
        local_key_len,
        chunk_len,
        rank,
        ROUTE_SLOTS: tl.constexpr,
        CP_SIZE: tl.constexpr,
        RATIO: tl.constexpr,
        ZIGZAG: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Mark owner requests directly from compact blocks and causal tails."""

        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        in_bounds = offsets < route_numel
        row = offsets // ROUTE_SLOTS
        slot = offsets - row * ROUTE_SLOTS
        length = tl.load(length_ptr + row, mask=in_bounds, other=0).to(tl.int32)
        max_length = ROUTE_SLOTS * RATIO + RATIO - 1
        length = tl.maximum(0, tl.minimum(length, max_length))
        selected_blocks = tl.minimum(length // RATIO, ROUTE_SLOTS)
        tail_count = length - selected_blocks * RATIO
        block_id = tl.load(block_ptr + offsets, mask=in_bounds, other=-1).to(tl.int32)
        block_valid = in_bounds & (slot < selected_blocks) & (block_id >= 0)

        for lane in tl.static_range(0, RATIO):
            token = block_id * RATIO + lane
            valid = block_valid & (token >= 0) & (token < global_seq_len)
            if ZIGZAG:
                chunk = token // chunk_len
                within = token - chunk * chunk_len
                owner = tl.where(
                    chunk < CP_SIZE, chunk, 2 * CP_SIZE - 1 - chunk)
                local = tl.where(
                    chunk < CP_SIZE, within, chunk_len + within)
            else:
                owner = token // local_key_len
                local = token - owner * local_key_len
            remote = valid & (owner != rank)
            safe_owner = tl.maximum(0, tl.minimum(owner, CP_SIZE - 1))
            safe_local = tl.maximum(0, tl.minimum(local, local_key_len - 1))
            tl.atomic_xchg(
                request_mask_ptr + safe_owner * local_key_len + safe_local,
                1,
                mask=remote,
            )

        # Exactly the slot-zero lane for each row emits its incomplete causal
        # block, avoiding a separate token-route or tail kernel.
        query = row - (row // seq_len) * seq_len
        query_position = tl.load(
            query_position_ptr + query, mask=in_bounds, other=0).to(tl.int32)
        tail_start = ((query_position + 1) // RATIO) * RATIO
        tail_source = in_bounds & (slot == 0)
        for lane in tl.static_range(0, RATIO - 1):
            token = tail_start + lane
            valid = (
                tail_source
                & (lane < tail_count)
                & (token >= 0)
                & (token < global_seq_len)
            )
            if ZIGZAG:
                chunk = token // chunk_len
                within = token - chunk * chunk_len
                owner = tl.where(
                    chunk < CP_SIZE, chunk, 2 * CP_SIZE - 1 - chunk)
                local = tl.where(
                    chunk < CP_SIZE, within, chunk_len + within)
            else:
                owner = token // local_key_len
                local = token - owner * local_key_len
            remote = valid & (owner != rank)
            safe_owner = tl.maximum(0, tl.minimum(owner, CP_SIZE - 1))
            safe_local = tl.maximum(0, tl.minimum(local, local_key_len - 1))
            tl.atomic_xchg(
                request_mask_ptr + safe_owner * local_key_len + safe_local,
                1,
                mask=remote,
            )

    @triton.jit
    def _qsa_cp_remap_route_kernel(
        route_ptr,
        length_ptr,
        cache_offset_ptr,
        out_ptr,
        route_numel,
        global_seq_len,
        local_key_len,
        chunk_len,
        ROUTE_SLOTS: tl.constexpr,
        CP_SIZE: tl.constexpr,
        ZIGZAG: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Map global token IDs to owner-cache offsets in one pass."""

        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        in_bounds = offsets < route_numel
        row = offsets // ROUTE_SLOTS
        slot = offsets - row * ROUTE_SLOTS
        length = tl.load(length_ptr + row, mask=in_bounds, other=0).to(tl.int32)
        token = tl.load(route_ptr + offsets, mask=in_bounds, other=-1).to(tl.int32)
        valid = (
            in_bounds
            & (slot < length)
            & (token >= 0)
            & (token < global_seq_len)
        )
        if ZIGZAG:
            chunk = token // chunk_len
            within = token - chunk * chunk_len
            owner = tl.where(chunk < CP_SIZE, chunk, 2 * CP_SIZE - 1 - chunk)
            local = tl.where(chunk < CP_SIZE, within, chunk_len + within)
        else:
            owner = token // local_key_len
            local = token - owner * local_key_len
        safe_owner = tl.maximum(0, tl.minimum(owner, CP_SIZE - 1))
        safe_local = tl.maximum(0, tl.minimum(local, local_key_len - 1))
        mapped = tl.load(
            cache_offset_ptr + safe_owner * local_key_len + safe_local,
            mask=valid,
            other=-1,
        )
        tl.store(out_ptr + offsets, mapped, mask=in_bounds)

    @triton.jit
    def _qsa_cp_compact_remap_route_kernel(
        block_ptr,
        length_ptr,
        query_position_ptr,
        cache_offset_ptr,
        out_ptr,
        output_numel,
        seq_len,
        global_seq_len,
        local_key_len,
        chunk_len,
        ROUTE_SLOTS: tl.constexpr,
        OUTPUT_SLOTS: tl.constexpr,
        CP_SIZE: tl.constexpr,
        RATIO: tl.constexpr,
        ZIGZAG: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Write cache-local token routes directly from compact block IDs."""

        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        in_bounds = offsets < output_numel
        row = offsets // OUTPUT_SLOTS
        output_slot = offsets - row * OUTPUT_SLOTS
        length = tl.load(length_ptr + row, mask=in_bounds, other=0).to(tl.int32)
        max_length = ROUTE_SLOTS * RATIO + RATIO - 1
        length = tl.maximum(0, tl.minimum(length, max_length))
        selected_blocks = tl.minimum(length // RATIO, ROUTE_SLOTS)
        tail_count = length - selected_blocks * RATIO

        block_token_count = selected_blocks * RATIO
        is_block_token = output_slot < block_token_count
        block_slot = output_slot // RATIO
        block_lane = output_slot - block_slot * RATIO
        safe_block_slot = tl.maximum(0, tl.minimum(block_slot, ROUTE_SLOTS - 1))
        block_id = tl.load(
            block_ptr + row * ROUTE_SLOTS + safe_block_slot,
            mask=in_bounds & is_block_token,
            other=-1,
        ).to(tl.int32)
        block_token = block_id * RATIO + block_lane

        tail_lane = output_slot - block_token_count
        query = row - (row // seq_len) * seq_len
        query_position = tl.load(
            query_position_ptr + query, mask=in_bounds, other=0).to(tl.int32)
        tail_token = ((query_position + 1) // RATIO) * RATIO + tail_lane
        is_tail_token = (tail_lane >= 0) & (tail_lane < tail_count)
        token = tl.where(is_block_token, block_token, tail_token)
        valid = (
            in_bounds
            & ((is_block_token & (block_id >= 0)) | is_tail_token)
            & (token >= 0)
            & (token < global_seq_len)
        )
        if ZIGZAG:
            chunk = token // chunk_len
            within = token - chunk * chunk_len
            owner = tl.where(
                chunk < CP_SIZE, chunk, 2 * CP_SIZE - 1 - chunk)
            local = tl.where(
                chunk < CP_SIZE, within, chunk_len + within)
        else:
            owner = token // local_key_len
            local = token - owner * local_key_len
        safe_owner = tl.maximum(0, tl.minimum(owner, CP_SIZE - 1))
        safe_local = tl.maximum(0, tl.minimum(local, local_key_len - 1))
        mapped = tl.load(
            cache_offset_ptr + safe_owner * local_key_len + safe_local,
            mask=valid,
            other=-1,
        )
        tl.store(out_ptr + offsets, mapped, mask=in_bounds)

    @triton.jit
    def _qsa_indexer_fused_topk_packed_kernel(
        q_ptr,
        block_key_ptr,
        out_ptr,
        block_start_ptr,
        segment_block_count_ptr,
        query_position_ptr,
        token_ids_ptr,
        total_tokens,
        total_blocks,
        num_heads,
        head_dim,
        compress_ratio: tl.constexpr,
        score_scale: tl.constexpr,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_bs,
        stride_bd,
        stride_os,
        stride_bstart,
        stride_bcount,
        stride_qpos,
        BLOCK_TOPK: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_TAIL: tl.constexpr,
        INDEX_BITS: tl.constexpr,
        INDEX_MASK: tl.constexpr,
        RATIO: tl.constexpr,
        USE_TOKEN_IDS: tl.constexpr,
        OUTPUT_BLOCKS: tl.constexpr,
    ):
        """Single-launch packed indexer with segment-local block addressing."""

        program = tl.program_id(0)
        if USE_TOKEN_IDS:
            token = tl.load(token_ids_ptr + program).to(tl.int32)
        else:
            token = program
        query_position = tl.load(query_position_ptr + token * stride_qpos).to(tl.int32)
        segment_block_start = tl.load(
            block_start_ptr + token * stride_bstart).to(tl.int32)
        segment_block_count = tl.load(
            segment_block_count_ptr + token * stride_bcount).to(tl.int32)
        complete_blocks = tl.minimum(
            segment_block_count, (query_position + 1) // compress_ratio
        ).to(tl.int32)
        block_count = tl.minimum(complete_blocks, BLOCK_TOPK).to(tl.int32)

        if OUTPUT_BLOCKS:
            # A long packed document still has an early prefix for which all
            # complete blocks fit in the budget.  The segment-level dispatcher
            # cannot identify those rows, so bypass QK/Top-K directly here.
            if complete_blocks <= BLOCK_TOPK:
                block_offsets = tl.arange(0, BLOCK_TOPK)
                out_base = out_ptr + token * stride_os
                tl.store(
                    out_base + block_offsets,
                    tl.where(block_offsets < block_count, block_offsets, -1),
                    mask=block_offsets < BLOCK_TOPK,
                )
                return

        head_offsets = tl.arange(0, BLOCK_H)
        dim_offsets = tl.arange(0, BLOCK_D)
        head_mask = head_offsets < num_heads
        dim_mask = dim_offsets < head_dim
        q_ptrs = (
            q_ptr
            + token * stride_qt
            + head_offsets[:, None] * stride_qh
            + dim_offsets[None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=head_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        # QSA indexer scores are non-negative after ReLU, so the raw IEEE
        # float bits are monotonic.  The low bits carry inverse local block id
        # for deterministic lower-id tie breaking.
        acc = tl.zeros((BLOCK_TOPK,), dtype=tl.int64)
        # The segment may be long, but this query can only score its causal
        # complete-block prefix.  Match the TokenSpeed bounded scan and leave
        # the incomplete causal block to the token expansion below.
        for block_offset_start in range(0, complete_blocks, BLOCK_N):
            block_offsets = block_offset_start + tl.arange(0, BLOCK_N)
            local_block_ids = block_offsets
            global_block_ids = segment_block_start + local_block_ids
            valid = (
                (local_block_ids < complete_blocks)
                & (global_block_ids < total_blocks)
            )
            k_ptrs = (
                block_key_ptr
                + global_block_ids[None, :] * stride_bs
                + dim_offsets[:, None] * stride_bd
            )
            keys = tl.load(
                k_ptrs,
                mask=valid[None, :] & dim_mask[:, None],
                other=0.0,
            )
            dots = tl.dot(q, keys, out_dtype=tl.float32)
            scores = tl.sum(tl.maximum(dots, 0.0), axis=0) * score_scale
            score_bits = scores.to(tl.uint32, bitcast=True)
            packed = (
                (score_bits.to(tl.int64) + 1) << INDEX_BITS
            ) | (INDEX_MASK - local_block_ids.to(tl.int64))
            packed = tl.where(valid, packed, 0)
            # The TokenSpeed merge keeps the running set bitonic and combines
            # it with a sorted tile using elementwise maximum.  BLOCK_N is K
            # for production QSA, so no 1024-wide top-k state is formed.
            acc = tl.bitonic_merge(acc)
            if tl.max(packed, axis=0) > tl.min(acc, axis=0):
                acc = tl.maximum(acc, tl.sort(packed, descending=True))

        # Short compact rows returned above.  Compact attention treats every
        # remaining saturated route as a set, so retain its exact bitonic
        # Top-K layout.  Token-ID compatibility output retains canonical score
        # ordering.
        if not OUTPUT_BLOCKS:
            acc = tl.sort(acc, descending=True)
        block_offsets = tl.arange(0, BLOCK_TOPK)
        packed_ids = (acc & INDEX_MASK).to(tl.int32)
        valid_blocks = (acc != 0) & (block_offsets < block_count)
        local_block_ids = INDEX_MASK - packed_ids
        local_block_ids = tl.where(
            (complete_blocks <= BLOCK_TOPK) & valid_blocks,
            block_offsets,
            local_block_ids,
        )
        local_block_ids = tl.where(valid_blocks, local_block_ids, -1)
        out_base = out_ptr + token * stride_os
        if OUTPUT_BLOCKS:
            tl.store(
                out_base + block_offsets,
                tl.where(valid_blocks, local_block_ids, -1),
                mask=block_offsets < BLOCK_TOPK,
            )
        else:
            for token_offset in tl.static_range(0, RATIO):
                output_offsets = block_offsets * compress_ratio + token_offset
                token_ids = local_block_ids * compress_ratio + token_offset
                tl.store(
                    out_base + output_offsets,
                    tl.where(valid_blocks, token_ids, -1),
                    mask=output_offsets < BLOCK_TOPK * RATIO,
                )
            tail_offsets = tl.arange(0, BLOCK_TAIL)
            tail_output = block_count * compress_ratio + tail_offsets
            tail_values = complete_blocks * compress_ratio + tail_offsets
            tail_valid = tail_values <= query_position
            tl.store(
                out_base + tail_output,
                tl.where(tail_valid, tail_values, -1),
                mask=tail_offsets < RATIO - 1,
            )
            tl.store(
                out_base + BLOCK_TOPK * compress_ratio + tail_offsets,
                -1,
                mask=(block_count < BLOCK_TOPK) & (tail_offsets < RATIO - 1),
            )

    @triton.jit
    def _qsa_indexer_packed_direct_fill_kernel(
        out_ptr,
        segment_block_count_ptr,
        query_position_ptr,
        token_ids_ptr,
        total_tokens,
        compress_ratio,
        stride_os,
        stride_bcount,
        stride_qpos,
        BLOCK_TOPK: tl.constexpr,
        BLOCK_TAIL: tl.constexpr,
        RATIO: tl.constexpr,
        USE_TOKEN_IDS: tl.constexpr,
        OUTPUT_BLOCKS: tl.constexpr,
    ):
        """Fill packed indexer rows when every segment fits in the budget.

        In this regime top-k is a no-op: every complete block in the causal
        prefix is selected in local order.  Keeping this as a separate kernel
        lets the common short-segment packed path skip projected-Q loads,
        block-key loads, dot products, and the running top-k state entirely.
        """

        program = tl.program_id(0)
        if USE_TOKEN_IDS:
            token = tl.load(token_ids_ptr + program).to(tl.int32)
        else:
            token = program
        query_position = tl.load(
            query_position_ptr + token * stride_qpos
        ).to(tl.int32)
        segment_block_count = tl.load(
            segment_block_count_ptr + token * stride_bcount
        ).to(tl.int32)
        complete_blocks = tl.minimum(
            segment_block_count, (query_position + 1) // compress_ratio
        ).to(tl.int32)
        block_count = tl.minimum(complete_blocks, BLOCK_TOPK).to(tl.int32)
        block_offsets = tl.arange(0, BLOCK_TOPK)
        valid_blocks = block_offsets < block_count
        local_block_ids = tl.where(valid_blocks, block_offsets, -1)
        out_base = out_ptr + token * stride_os

        if OUTPUT_BLOCKS:
            tl.store(
                out_base + block_offsets,
                tl.where(valid_blocks, local_block_ids, -1),
                mask=block_offsets < BLOCK_TOPK,
            )
        else:
            for token_offset in tl.static_range(0, RATIO):
                output_offsets = block_offsets * compress_ratio + token_offset
                token_ids = local_block_ids * compress_ratio + token_offset
                tl.store(
                    out_base + output_offsets,
                    tl.where(valid_blocks, token_ids, -1),
                    mask=output_offsets < BLOCK_TOPK * RATIO,
                )

            # The current causal block is not part of the complete-block list.
            # Fill it immediately after the selected prefix and clear the fixed
            # width padding so the output has no uninitialized entries.
            tail_offsets = tl.arange(0, BLOCK_TAIL)
            tail_output = block_count * compress_ratio + tail_offsets
            tail_values = complete_blocks * compress_ratio + tail_offsets
            tail_valid = tail_values <= query_position
            tl.store(
                out_base + tail_output,
                tl.where(tail_valid, tail_values, -1),
                mask=tail_offsets < RATIO - 1,
            )
            tl.store(
                out_base + BLOCK_TOPK * compress_ratio + tail_offsets,
                -1,
                mask=(block_count < BLOCK_TOPK) & (tail_offsets < RATIO - 1),
            )

    @triton.jit
    def _qsa_indexer_direct_fill_kernel(
        out_ptr,
        query_position_ptr,
        token_ids_ptr,
        seq_len,
        num_blocks,
        compress_ratio,
        stride_os,
        stride_qpos,
        BLOCK_TOPK: tl.constexpr,
        BLOCK_TAIL: tl.constexpr,
        RATIO: tl.constexpr,
        OUTPUT_BLOCKS: tl.constexpr,
    ):
        """Fill standard-BSHD rows whose causal prefix fits in block Top-K.

        When ``complete_blocks <= BLOCK_TOPK`` the score ordering cannot change
        the result: every complete block is selected in local order.  The
        caller supplies flattened ``[batch, query]`` row IDs so this kernel can
        handle only those rows and leave the genuinely long rows to the fused
        score/Top-K producer.
        """

        program = tl.program_id(0)
        row = tl.load(token_ids_ptr + program).to(tl.int32)
        batch = row // seq_len
        query = row - batch * seq_len
        query_position = tl.load(
            query_position_ptr + query * stride_qpos
        ).to(tl.int32)
        complete_blocks = tl.minimum(
            num_blocks, (query_position + 1) // compress_ratio
        ).to(tl.int32)
        block_count = tl.minimum(complete_blocks, BLOCK_TOPK).to(tl.int32)
        block_offsets = tl.arange(0, BLOCK_TOPK)
        valid_blocks = block_offsets < block_count
        block_ids = tl.where(valid_blocks, block_offsets, -1)
        out_base = out_ptr + row * stride_os

        if OUTPUT_BLOCKS:
            tl.store(
                out_base + block_offsets,
                tl.where(valid_blocks, block_ids, -1),
                mask=block_offsets < BLOCK_TOPK,
            )
        else:
            for token_offset in tl.static_range(0, RATIO):
                output_offsets = block_offsets * compress_ratio + token_offset
                token_ids = block_ids * compress_ratio + token_offset
                tl.store(
                    out_base + output_offsets,
                    tl.where(valid_blocks, token_ids, -1),
                    mask=output_offsets < BLOCK_TOPK * RATIO,
                )

            tail_offsets = tl.arange(0, BLOCK_TAIL)
            tail_output = block_count * compress_ratio + tail_offsets
            tail_values = complete_blocks * compress_ratio + tail_offsets
            tail_valid = tail_values <= query_position
            tl.store(
                out_base + tail_output,
                tl.where(tail_valid, tail_values, -1),
                mask=tail_offsets < RATIO - 1,
            )
            tl.store(
                out_base + BLOCK_TOPK * compress_ratio + tail_offsets,
                -1,
                mask=(block_count < BLOCK_TOPK) & (tail_offsets < RATIO - 1),
            )

    @triton.jit
    def _qsa_indexer_radix_topk_kernel(
        q_ptr,
        block_key_ptr,
        out_ptr,
        query_position_ptr,
        seq_len,
        num_blocks,
        num_heads,
        head_dim,
        compress_ratio,
        score_scale,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_bb,
        stride_bs,
        stride_bd,
        stride_ob,
        stride_os,
        BLOCK_TOPK: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        INDEX_BITS: tl.constexpr,
        INDEX_MASK: tl.constexpr,
        BLOCK_TAIL: tl.constexpr,
        RATIO: tl.constexpr,
    ):
        """Top-K fusion using Triton's radix/bitonic selection primitive.

        The packed key is ``(ordered_float_score, inverse_block_id)``.  Each
        iteration selects the best ``BLOCK_TOPK`` values from the running set
        and one compressed-key tile, avoiding the much more expensive generic
        sort path for the production ``block_topk=512`` shape.
        """

        row = tl.program_id(0)
        batch = row // seq_len
        query = row - batch * seq_len
        query_position = tl.load(query_position_ptr + query).to(tl.int32)
        complete_blocks = tl.minimum(
            num_blocks, (query_position + 1) // compress_ratio
        ).to(tl.int32)
        block_count = tl.minimum(complete_blocks, BLOCK_TOPK).to(tl.int32)

        head_offsets = tl.arange(0, BLOCK_H)
        dim_offsets = tl.arange(0, BLOCK_D)
        head_mask = head_offsets < num_heads
        dim_mask = dim_offsets < head_dim
        q_ptrs = (
            q_ptr
            + batch * stride_qb
            + query * stride_qs
            + head_offsets[:, None] * stride_qh
            + dim_offsets[None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=head_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        # ``tl.topk`` operates on the second dimension.  Keeping a singleton
        # row dimension also avoids a host-side reshape for the packed keys.
        best = tl.zeros((1, BLOCK_TOPK), dtype=tl.int64)
        num_tiles = tl.cdiv(num_blocks, BLOCK_N)
        for tile in tl.range(0, num_tiles):
            block_start = tile * BLOCK_N
            block_offsets = tl.arange(0, BLOCK_N)
            block_ids = block_start + block_offsets
            valid = block_ids < complete_blocks
            k_ptrs = (
                block_key_ptr
                + batch * stride_bb
                + block_ids[None, :] * stride_bs
                + dim_offsets[:, None] * stride_bd
            )
            keys = tl.load(
                k_ptrs,
                mask=valid[None, :] & dim_mask[:, None],
                other=0.0,
            )
            dots = tl.dot(q, keys, out_dtype=tl.float32)
            scores = tl.sum(tl.maximum(dots, 0.0), axis=0) * score_scale
            score_bits = scores.to(tl.uint32, bitcast=True)
            ordered_bits = tl.where(
                (score_bits & 0x80000000) != 0,
                ~score_bits,
                score_bits ^ 0x80000000,
            )
            packed = (
                ((ordered_bits.to(tl.int64) + 1) << INDEX_BITS)
                | (INDEX_MASK - block_ids.to(tl.int64))
            )
            packed = tl.where(valid, packed, 0)
            candidates = tl.reshape(
                tl.join(best, packed[None, :]), (1, 2 * BLOCK_TOPK)
            )
            best = tl.topk(candidates, k=BLOCK_TOPK, dim=1)

        best = tl.reshape(best, (BLOCK_TOPK,))
        packed_ids = (best & INDEX_MASK).to(tl.int32)
        valid_blocks = (best != 0) & (tl.arange(0, BLOCK_TOPK) < block_count)
        block_ids = tl.where(valid_blocks, INDEX_MASK - packed_ids, -1)
        out_base = out_ptr + batch * stride_ob + query * stride_os
        for token_offset in tl.static_range(0, RATIO):
            output_offsets = tl.arange(0, BLOCK_TOPK) * compress_ratio + token_offset
            token_ids = block_ids * compress_ratio + token_offset
            tl.store(
                out_base + output_offsets,
                tl.where(valid_blocks, token_ids, -1),
                mask=output_offsets < BLOCK_TOPK * RATIO,
            )
        tail_offsets = tl.arange(0, BLOCK_TAIL)
        tail_output = block_count * compress_ratio + tail_offsets
        tail_values = complete_blocks * compress_ratio + tail_offsets
        tail_valid = tail_values <= query_position
        tl.store(
            out_base + tail_output,
            tl.where(tail_valid, tail_values, -1),
            mask=tail_offsets < RATIO - 1,
        )
        tl.store(
            out_base + BLOCK_TOPK * compress_ratio + tail_offsets,
            -1,
            mask=(block_count < BLOCK_TOPK) & (tail_offsets < RATIO - 1),
        )

    @triton.jit
    def _qsa_indexer_merge_sorted_128(left, right, BLOCK_M: tl.constexpr):
        """Merge two sorted 128-wide rows and return top/residual halves.

        QSA's production block budget is 512.  Chaining four of these exact
        128+128 merges avoids the 1024-wide ``tl.topk`` state that is fragile
        on SM90 when combined with the indexer dot product.  Both inputs are
        descending and the second return value is the discarded residual that
        must be carried into the next 128-wide budget chunk.
        """

        joined = tl.join(left, right)
        joined = tl.permute(joined, (0, 2, 1))
        candidates = tl.reshape(joined, (BLOCK_M, 256))
        ordered = tl.sort(candidates, descending=True)
        halves = tl.reshape(ordered, (BLOCK_M, 2, 128))
        halves = tl.permute(halves, (0, 2, 1))
        return tl.split(halves)

    @triton.jit
    def _qsa_indexer_radix_topk_batched_merge4_kernel(
        q_ptr,
        block_key_ptr,
        out_ptr,
        query_position_ptr,
        seq_len,
        num_blocks,
        num_heads,
        head_dim,
        compress_ratio,
        score_scale,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_bb,
        stride_bs,
        stride_bd,
        stride_ob,
        stride_os,
        BLOCK_M: tl.constexpr,
        BLOCK_TOPK: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_TAIL: tl.constexpr,
        INDEX_BITS: tl.constexpr,
        INDEX_MASK: tl.constexpr,
        RATIO: tl.constexpr,
    ):
        """Exact batched Top-512 using four 128-wide merge segments."""

        program = tl.program_id(0)
        queries_per_batch = tl.cdiv(seq_len, BLOCK_M)
        batch = program // queries_per_batch
        query_tile = program - batch * queries_per_batch
        row_offsets = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        row_valid = row_offsets < seq_len
        query_positions = tl.load(
            query_position_ptr + row_offsets,
            mask=row_valid,
            other=0,
        ).to(tl.int32)
        complete_blocks = tl.minimum(
            num_blocks, (query_positions + 1) // compress_ratio
        ).to(tl.int32)
        block_count = tl.minimum(complete_blocks, BLOCK_TOPK).to(tl.int32)

        head_offsets = tl.arange(0, BLOCK_H)
        dim_offsets = tl.arange(0, BLOCK_D)
        head_mask = head_offsets < num_heads
        dim_mask = dim_offsets < head_dim
        q_ptrs = (
            q_ptr
            + batch * stride_qb
            + row_offsets[:, None, None] * stride_qs
            + head_offsets[None, :, None] * stride_qh
            + dim_offsets[None, None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=row_valid[:, None, None]
            & head_mask[None, :, None]
            & dim_mask[None, None, :],
            other=0.0,
        )
        q = tl.reshape(q, (BLOCK_M * BLOCK_H, BLOCK_D))

        # The standard production configuration is BLOCK_TOPK=512 and a
        # 128-wide compressed-key tile.  Keep four independent sorted chunks;
        # a new tile is inserted through the residual chain below.
        best0 = tl.zeros((BLOCK_M, 128), dtype=tl.int64)
        best1 = tl.zeros((BLOCK_M, 128), dtype=tl.int64)
        best2 = tl.zeros((BLOCK_M, 128), dtype=tl.int64)
        best3 = tl.zeros((BLOCK_M, 128), dtype=tl.int64)
        num_tiles = tl.cdiv(num_blocks, 128)
        for tile in tl.range(0, num_tiles):
            block_start = tile * 128
            block_offsets = tl.arange(0, 128)
            block_ids = block_start + block_offsets
            key_valid = block_ids < num_blocks
            k_ptrs = (
                block_key_ptr
                + batch * stride_bb
                + block_ids[None, :] * stride_bs
                + dim_offsets[:, None] * stride_bd
            )
            keys = tl.load(
                k_ptrs,
                mask=key_valid[None, :] & dim_mask[:, None],
                other=0.0,
            )
            dots = tl.dot(q, keys, out_dtype=tl.float32)
            dots = tl.reshape(dots, (BLOCK_M, BLOCK_H, 128))
            scores = tl.sum(tl.maximum(dots, 0.0), axis=1) * score_scale
            valid = key_valid[None, :] & (block_ids[None, :] < complete_blocks[:, None])
            score_bits = scores.to(tl.uint32, bitcast=True)
            ordered_bits = tl.where(
                (score_bits & 0x80000000) != 0,
                ~score_bits,
                score_bits ^ 0x80000000,
            )
            packed = (
                ((ordered_bits.to(tl.int64) + 1) << INDEX_BITS)
                | (INDEX_MASK - block_ids[None, :].to(tl.int64))
            )
            packed = tl.where(valid, packed, 0)
            packed = tl.sort(packed, descending=True)
            best0, residual = _qsa_indexer_merge_sorted_128(best0, packed, BLOCK_M)
            best1, residual = _qsa_indexer_merge_sorted_128(best1, residual, BLOCK_M)
            best2, residual = _qsa_indexer_merge_sorted_128(best2, residual, BLOCK_M)
            best3, residual = _qsa_indexer_merge_sorted_128(best3, residual, BLOCK_M)

        # Concatenate four equal-shaped rows without a 1024-wide temporary.
        joined01 = tl.permute(tl.join(best0, best1), (0, 2, 1))
        joined23 = tl.permute(tl.join(best2, best3), (0, 2, 1))
        best01 = tl.reshape(joined01, (BLOCK_M, 256))
        best23 = tl.reshape(joined23, (BLOCK_M, 256))
        best = tl.reshape(
            tl.permute(tl.join(best01, best23), (0, 2, 1)),
            (BLOCK_M, 512),
        )
        block_offsets = tl.arange(0, BLOCK_TOPK)[None, :]
        packed_ids = (best & INDEX_MASK).to(tl.int32)
        valid_blocks = (best != 0) & (block_offsets < block_count[:, None])
        block_ids = tl.where(valid_blocks, INDEX_MASK - packed_ids, -1)
        out_base = out_ptr + batch * stride_ob + row_offsets[:, None] * stride_os
        for token_offset in tl.static_range(0, RATIO):
            output_offsets = block_offsets * compress_ratio + token_offset
            token_ids = block_ids * compress_ratio + token_offset
            tl.store(
                out_base + output_offsets,
                tl.where(valid_blocks, token_ids, -1),
                mask=row_valid[:, None] & (output_offsets < BLOCK_TOPK * RATIO),
            )
        tail_offsets = tl.arange(0, BLOCK_TAIL)[None, :]
        tail_output = block_count[:, None] * compress_ratio + tail_offsets
        tail_values = complete_blocks[:, None] * compress_ratio + tail_offsets
        tail_valid = row_valid[:, None] & (tail_offsets < RATIO - 1) & (
            tail_values <= query_positions[:, None]
        )
        tl.store(
            out_base + tail_output,
            tl.where(tail_valid, tail_values, -1),
            mask=row_valid[:, None] & (tail_offsets < RATIO - 1),
        )
        tl.store(
            out_base + BLOCK_TOPK * compress_ratio + tail_offsets,
            -1,
            mask=row_valid[:, None]
            & (block_count[:, None] < BLOCK_TOPK)
            & (tail_offsets < RATIO - 1),
        )

    @triton.jit
    def _qsa_indexer_radix_topk_batched_kernel(
        q_ptr,
        block_key_ptr,
        out_ptr,
        query_position_ptr,
        seq_len,
        num_blocks,
        num_heads,
        head_dim,
        compress_ratio,
        score_scale,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_bb,
        stride_bs,
        stride_bd,
        stride_ob,
        stride_os,
        BLOCK_M: tl.constexpr,
        BLOCK_TOPK: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_TAIL: tl.constexpr,
        INDEX_BITS: tl.constexpr,
        INDEX_MASK: tl.constexpr,
        RATIO: tl.constexpr,
    ):
        """Batched packed-key Top-K that reuses every block-key tile.

        ``BLOCK_M=4`` is the default SM90 setting.  It keeps a small number of
        independent running Top-K rows while loading each compressed-key tile
        once, which materially reduces the repeated K-cache traffic of
        one-program-per-row selection without the register footprint of larger
        query batches.
        """

        program = tl.program_id(0)
        queries_per_batch = tl.cdiv(seq_len, BLOCK_M)
        batch = program // queries_per_batch
        query_tile = program - batch * queries_per_batch
        row_offsets = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        row_valid = row_offsets < seq_len
        query_positions = tl.load(
            query_position_ptr + row_offsets,
            mask=row_valid,
            other=0,
        ).to(tl.int32)
        complete_blocks = tl.minimum(
            num_blocks, (query_positions + 1) // compress_ratio
        ).to(tl.int32)
        block_count = tl.minimum(complete_blocks, BLOCK_TOPK).to(tl.int32)

        head_offsets = tl.arange(0, BLOCK_H)
        dim_offsets = tl.arange(0, BLOCK_D)
        head_mask = head_offsets < num_heads
        dim_mask = dim_offsets < head_dim
        q_ptrs = (
            q_ptr
            + batch * stride_qb
            + row_offsets[:, None, None] * stride_qs
            + head_offsets[None, :, None] * stride_qh
            + dim_offsets[None, None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=row_valid[:, None, None] & head_mask[None, :, None] & dim_mask[None, None, :],
            other=0.0,
        )
        q = tl.reshape(q, (BLOCK_M * BLOCK_H, BLOCK_D))
        best = tl.zeros((BLOCK_M, BLOCK_TOPK), dtype=tl.int64)
        num_tiles = tl.cdiv(num_blocks, BLOCK_TOPK)
        for tile in tl.range(0, num_tiles):
            block_start = tile * BLOCK_TOPK
            block_offsets = tl.arange(0, BLOCK_TOPK)
            block_ids = block_start + block_offsets
            key_valid = block_ids < num_blocks
            k_ptrs = (
                block_key_ptr
                + batch * stride_bb
                + block_ids[None, :] * stride_bs
                + dim_offsets[:, None] * stride_bd
            )
            keys = tl.load(
                k_ptrs,
                mask=key_valid[None, :] & dim_mask[:, None],
                other=0.0,
            )
            dots = tl.dot(q, keys, out_dtype=tl.float32)
            dots = tl.reshape(dots, (BLOCK_M, BLOCK_H, BLOCK_TOPK))
            scores = tl.sum(tl.maximum(dots, 0.0), axis=1) * score_scale
            valid = key_valid[None, :] & (block_ids[None, :] < complete_blocks[:, None])
            score_bits = scores.to(tl.uint32, bitcast=True)
            ordered_bits = tl.where(
                (score_bits & 0x80000000) != 0,
                ~score_bits,
                score_bits ^ 0x80000000,
            )
            packed = (
                ((ordered_bits.to(tl.int64) + 1) << INDEX_BITS)
                | (INDEX_MASK - block_ids[None, :].to(tl.int64))
            )
            packed = tl.where(valid, packed, 0)
            candidates = tl.reshape(
                tl.join(best, packed), (BLOCK_M, 2 * BLOCK_TOPK)
            )
            best = tl.topk(candidates, k=BLOCK_TOPK, dim=1)

        block_offsets = tl.arange(0, BLOCK_TOPK)[None, :]
        packed_ids = (best & INDEX_MASK).to(tl.int32)
        valid_blocks = (best != 0) & (block_offsets < block_count[:, None])
        block_ids = tl.where(valid_blocks, INDEX_MASK - packed_ids, -1)
        out_base = out_ptr + batch * stride_ob + row_offsets[:, None] * stride_os
        for token_offset in tl.static_range(0, RATIO):
            output_offsets = block_offsets * compress_ratio + token_offset
            token_ids = block_ids * compress_ratio + token_offset
            tl.store(
                out_base + output_offsets,
                tl.where(valid_blocks, token_ids, -1),
                mask=row_valid[:, None] & (output_offsets < BLOCK_TOPK * RATIO),
            )
        tail_offsets = tl.arange(0, BLOCK_TAIL)[None, :]
        tail_output = block_count[:, None] * compress_ratio + tail_offsets
        tail_values = complete_blocks[:, None] * compress_ratio + tail_offsets
        tail_valid = row_valid[:, None] & (tail_offsets < RATIO - 1) & (tail_values <= query_positions[:, None])
        tl.store(
            out_base + tail_output,
            tl.where(tail_valid, tail_values, -1),
            mask=row_valid[:, None] & (tail_offsets < RATIO - 1),
        )
        tl.store(
            out_base + BLOCK_TOPK * compress_ratio + tail_offsets,
            -1,
            mask=row_valid[:, None] & (block_count[:, None] < BLOCK_TOPK) & (tail_offsets < RATIO - 1),
        )
    @triton.jit
    def _qsa_indexer_stream_partial_kernel(
        q_ptr,
        block_key_ptr,
        partial_ptr,
        query_position_ptr,
        seq_len,
        num_blocks,
        num_heads,
        head_dim,
        compress_ratio,
        score_scale,
        blocks_per_split,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_bb,
        stride_bs,
        stride_bd,
        stride_pb,
        stride_ps,
        stride_pp,
        stride_pk,
        BLOCK_TOPK: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        INDEX_BITS: tl.constexpr,
        INDEX_MASK: tl.constexpr,
        RATIO: tl.constexpr,
    ):
        """Produce one packed streaming Top-K partial per block split."""

        row = tl.program_id(0)
        split = tl.program_id(1)
        batch = row // seq_len
        query = row - batch * seq_len
        query_position = tl.load(query_position_ptr + query).to(tl.int32)
        complete_blocks = tl.minimum(
            num_blocks, (query_position + 1) // compress_ratio
        ).to(tl.int32)
        split_start = split * blocks_per_split
        split_end = tl.minimum(
            num_blocks, tl.minimum(split_start + blocks_per_split, complete_blocks)
        )

        head_offsets = tl.arange(0, BLOCK_H)
        dim_offsets = tl.arange(0, BLOCK_D)
        head_mask = head_offsets < num_heads
        dim_mask = dim_offsets < head_dim
        q_ptrs = (
            q_ptr
            + batch * stride_qb
            + query * stride_qs
            + head_offsets[:, None] * stride_qh
            + dim_offsets[None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=head_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        acc = tl.zeros((BLOCK_TOPK,), dtype=tl.int64)
        for block_start in tl.range(split_start, split_end, BLOCK_N):
            block_ids = block_start + tl.arange(0, BLOCK_N)
            valid = block_ids < split_end
            k_ptrs = (
                block_key_ptr
                + batch * stride_bb
                + block_ids[None, :] * stride_bs
                + dim_offsets[:, None] * stride_bd
            )
            keys = tl.load(
                k_ptrs,
                mask=valid[None, :] & dim_mask[:, None],
                other=0.0,
            )
            dots = tl.dot(q, keys, out_dtype=tl.float32)
            scores = tl.sum(tl.maximum(dots, 0.0), axis=0) * score_scale
            score_bits = scores.to(tl.uint32, bitcast=True)
            packed = (
                ((score_bits.to(tl.int64) + 1) << INDEX_BITS)
                | (INDEX_MASK - block_ids.to(tl.int64))
            )
            packed = tl.where(valid, packed, 0)
            acc = tl.bitonic_merge(acc)
            if tl.max(packed, axis=0) > tl.min(acc, axis=0):
                candidates = _qsa_pad_topk_candidates(
                    packed, BLOCK_TOPK, BLOCK_N
                )
                acc = tl.maximum(acc, tl.sort(candidates, descending=True))

        acc = tl.sort(acc, descending=True)
        out_ptrs = (
            partial_ptr
            + batch * stride_pb
            + query * stride_ps
            + split * stride_pp
            + tl.arange(0, BLOCK_TOPK) * stride_pk
        )
        tl.store(out_ptrs, acc)

    @triton.jit
    def _qsa_indexer_merge_topk_pairs(cur, ROWS: tl.constexpr, BLOCK_TOPK: tl.constexpr):
        """Merge adjacent packed partial rows while retaining the largest K."""

        odd = (tl.arange(0, ROWS) % 2 == 1)[:, None]
        ordered = tl.where(odd, -cur, cur)
        ordered = tl.bitonic_merge(ordered)
        ordered = tl.where(odd, -ordered, ordered)
        return tl.max(tl.reshape(ordered, (ROWS // 2, 2, BLOCK_TOPK)), axis=1)

    @triton.jit
    def _qsa_indexer_merge_topk_tree(cur, ROWS: tl.constexpr, BLOCK_TOPK: tl.constexpr):
        """Reduce a power-of-two partial grid to one packed Top-K row."""

        if ROWS >= 256:
            cur = _qsa_indexer_merge_topk_pairs(cur, 256, BLOCK_TOPK)
        if ROWS >= 128:
            cur = _qsa_indexer_merge_topk_pairs(cur, 128, BLOCK_TOPK)
        if ROWS >= 64:
            cur = _qsa_indexer_merge_topk_pairs(cur, 64, BLOCK_TOPK)
        if ROWS >= 32:
            cur = _qsa_indexer_merge_topk_pairs(cur, 32, BLOCK_TOPK)
        if ROWS >= 16:
            cur = _qsa_indexer_merge_topk_pairs(cur, 16, BLOCK_TOPK)
        if ROWS >= 8:
            cur = _qsa_indexer_merge_topk_pairs(cur, 8, BLOCK_TOPK)
        if ROWS >= 4:
            cur = _qsa_indexer_merge_topk_pairs(cur, 4, BLOCK_TOPK)
        if ROWS >= 2:
            cur = _qsa_indexer_merge_topk_pairs(cur, 2, BLOCK_TOPK)
        return tl.reshape(tl.bitonic_merge(cur, descending=True), (BLOCK_TOPK,))

    @triton.jit
    def _qsa_indexer_merge_topk_kernel(
        partial_ptr,
        out_ptr,
        query_position_ptr,
        seq_len,
        num_blocks,
        compress_ratio,
        blocks_per_split,
        stride_pb,
        stride_ps,
        stride_pp,
        stride_pk,
        stride_ob,
        stride_os,
        BLOCK_TOPK: tl.constexpr,
        SPLITS: tl.constexpr,
        BLOCK_TAIL: tl.constexpr,
        INDEX_MASK: tl.constexpr,
        RATIO: tl.constexpr,
        OUTPUT_BLOCKS: tl.constexpr,
    ):
        """Merge packed split Top-K rows and expand them to selected tokens."""

        row = tl.program_id(0)
        batch = row // seq_len
        query = row - batch * seq_len
        query_position = tl.load(query_position_ptr + query).to(tl.int32)
        complete_blocks = tl.minimum(
            num_blocks, (query_position + 1) // compress_ratio
        ).to(tl.int32)
        block_count = tl.minimum(complete_blocks, BLOCK_TOPK).to(tl.int32)
        split_ids = tl.arange(0, SPLITS)[:, None]
        topk_offsets = tl.arange(0, BLOCK_TOPK)[None, :]
        partials = tl.load(
            partial_ptr
            + batch * stride_pb
            + query * stride_ps
            + split_ids * stride_pp
            + topk_offsets * stride_pk,
            mask=split_ids * blocks_per_split < complete_blocks,
            other=0,
        )
        acc = _qsa_indexer_merge_topk_tree(partials, SPLITS, BLOCK_TOPK)
        block_offsets = tl.arange(0, BLOCK_TOPK)
        packed_ids = (acc & INDEX_MASK).to(tl.int32)
        valid_blocks = (acc != 0) & (block_offsets < block_count)
        block_ids = tl.where(valid_blocks, INDEX_MASK - packed_ids, -1)
        out_base = out_ptr + batch * stride_ob + query * stride_os
        if OUTPUT_BLOCKS:
            tl.store(
                out_base + block_offsets,
                tl.where(valid_blocks, block_ids, -1),
                mask=block_offsets < BLOCK_TOPK,
            )
        else:
            for token_offset in tl.static_range(0, RATIO):
                output_offsets = block_offsets * compress_ratio + token_offset
                token_ids = block_ids * compress_ratio + token_offset
                tl.store(
                    out_base + output_offsets,
                    tl.where(valid_blocks, token_ids, -1),
                    mask=output_offsets < BLOCK_TOPK * RATIO,
                )
            tail_offsets = tl.arange(0, BLOCK_TAIL)
            tail_output = block_count * compress_ratio + tail_offsets
            tail_values = complete_blocks * compress_ratio + tail_offsets
            tail_valid = tail_values <= query_position
            tl.store(
                out_base + tail_output,
                tl.where(tail_valid, tail_values, -1),
                mask=tail_offsets < RATIO - 1,
            )
            tl.store(
                out_base + BLOCK_TOPK * compress_ratio + tail_offsets,
                -1,
                mask=(block_count < BLOCK_TOPK) & (tail_offsets < RATIO - 1),
            )

    @triton.jit
    def _qsa_selected_kv_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        index_ptr,
        length_ptr,
        out_ptr,
        lse_ptr,
        seq_len_q,
        seq_len_k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        key_position_offset,
        query_position_ptr,
        token_ids_ptr,
        stride_qp,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vd,
        stride_ib,
        stride_is,
        stride_ik,
        stride_lb,
        stride_ls,
        stride_ob,
        stride_os,
        stride_oh,
        stride_od,
        stride_lseb,
        stride_lseh,
        stride_lses,
        K: tl.constexpr,
        HEADS_PER_KV: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
    ):
        """Selected-KV online-softmax forward for one ``(batch, query, head)``."""

        batch = tl.program_id(0)
        query = tl.program_id(1)
        head = tl.program_id(2)
        kv_head = head // HEADS_PER_KV

        d_offsets = tl.arange(0, BLOCK_D)
        q_base = q_ptr + batch * stride_qb + query * stride_qs + head * stride_qh
        q = tl.load(q_base + d_offsets * stride_qd, mask=d_offsets < head_dim, other=0.0).to(tl.bfloat16)

        length = tl.load(length_ptr + batch * stride_lb + query * stride_ls).to(tl.int32)
        query_position = tl.load(query_position_ptr + query * stride_qp).to(tl.int32)
        max_value = -float("inf")
        denominator = 0.0
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for key_start in tl.range(0, K, BLOCK_K):
            key_offsets = key_start + tl.arange(0, BLOCK_K)
            selected = tl.load(
                index_ptr + batch * stride_ib + query * stride_is + key_offsets * stride_ik,
                mask=key_offsets < K,
                other=-1,
            ).to(tl.int32)
            valid = (key_offsets < length) & (selected >= 0) & (selected < seq_len_k)
            if CAUSAL:
                valid = valid & (selected + key_position_offset <= query_position)
            safe_selected = tl.where(valid, selected, 0)

            k_ptrs = (k_ptr + batch * stride_kb + safe_selected[:, None] * stride_ks + kv_head * stride_kh +
                      d_offsets[None, :] * stride_kd)
            v_ptrs = (v_ptr + batch * stride_vb + safe_selected[:, None] * stride_vs + kv_head * stride_vh +
                      d_offsets[None, :] * stride_vd)
            k = tl.load(k_ptrs, mask=valid[:, None] & (d_offsets[None, :] < head_dim), other=0.0).to(tl.bfloat16)
            v = tl.load(v_ptrs, mask=valid[:, None] & (d_offsets[None, :] < head_dim), other=0.0).to(tl.bfloat16)

            scores = tl.reshape(tl.dot(k, q[:, None], out_dtype=tl.float32), (BLOCK_K,)) * softmax_scale
            scores = tl.where(valid, scores, -float("inf"))
            tile_max = tl.max(scores, axis=0)
            new_max = tl.maximum(max_value, tile_max)
            old_scale = tl.where(max_value == -float("inf"), 0.0, tl.exp(max_value - new_max))
            probabilities = tl.exp(tl.where(valid, scores - new_max, -float("inf")))
            tile_sum = tl.sum(probabilities, axis=0)
            denominator = denominator * old_scale + tile_sum
            accumulator = accumulator * old_scale + tl.sum(probabilities[:, None] * v, axis=0)
            max_value = new_max

        has_value = denominator > 0.0
        output = tl.where(has_value, accumulator / denominator, 0.0)
        log_sum_exp = tl.where(has_value, max_value + tl.log(denominator), -float("inf"))
        out_base = out_ptr + batch * stride_ob + query * stride_os + head * stride_oh
        tl.store(out_base + d_offsets * stride_od, output, mask=d_offsets < head_dim)
        lse_ptrs = lse_ptr + batch * stride_lseb + head * stride_lseh + query * stride_lses
        tl.store(lse_ptrs, log_sum_exp)

    @triton.jit
    def _qsa_selected_kv_backward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        index_ptr,
        length_ptr,
        lse_ptr,
        grad_out_ptr,
        out_ptr,
        grad_lse_ptr,
        grad_q_ptr,
        grad_k_ptr,
        grad_v_ptr,
        seq_len_q,
        seq_len_k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        key_position_offset,
        query_position_ptr,
        stride_qp,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vd,
        stride_ib,
        stride_is,
        stride_ik,
        stride_lb,
        stride_ls,
        stride_lseb,
        stride_lseh,
        stride_lses,
        stride_gob,
        stride_gos,
        stride_goh,
        stride_god,
        stride_glseb,
        stride_glseh,
        stride_glses,
        stride_dqb,
        stride_dqs,
        stride_dqh,
        stride_dqd,
        stride_dkb,
        stride_dks,
        stride_dkh,
        stride_dkd,
        stride_dvb,
        stride_dvs,
        stride_dvh,
        stride_dvd,
        K: tl.constexpr,
        HEADS_PER_KV: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
        HAS_GRAD_OUTPUT: tl.constexpr,
        HAS_GRAD_LSE: tl.constexpr,
    ):
        """Selected-KV backward with atomic dK/dV accumulation.

        A query/head program owns dQ.  K/V are potentially selected by many
        query programs, so their gradients are accumulated with FP32 atomics;
        this is the same duplicate-access rule used by the torch reference.
        """

        batch = tl.program_id(0)
        query = tl.program_id(1)
        head = tl.program_id(2)
        kv_head = head // HEADS_PER_KV

        d_offsets = tl.arange(0, BLOCK_D)
        q_base = q_ptr + batch * stride_qb + query * stride_qs + head * stride_qh
        q = tl.load(q_base + d_offsets * stride_qd, mask=d_offsets < head_dim, other=0.0).to(tl.bfloat16)
        go_base = grad_out_ptr + batch * stride_gob + query * stride_gos + head * stride_goh
        grad_output = tl.load(
            go_base + d_offsets * stride_god,
            mask=(d_offsets < head_dim) & HAS_GRAD_OUTPUT,
            other=0.0,
        ).to(tl.float32)
        query_position = tl.load(query_position_ptr + query * stride_qp).to(tl.int32)
        length = tl.load(length_ptr + batch * stride_lb + query * stride_ls).to(tl.int32)
        lse = tl.load(lse_ptr + batch * stride_lseb + head * stride_lseh + query * stride_lses).to(tl.float32)
        grad_lse = tl.load(
            grad_lse_ptr + batch * stride_glseb + head * stride_glseh + query * stride_glses,
            mask=HAS_GRAD_LSE,
            other=0.0,
        ).to(tl.float32)

        correction = 0.0
        for key_start in tl.range(0, K, BLOCK_K):
            key_offsets = key_start + tl.arange(0, BLOCK_K)
            selected = tl.load(
                index_ptr + batch * stride_ib + query * stride_is + key_offsets * stride_ik,
                mask=key_offsets < K,
                other=-1,
            ).to(tl.int32)
            valid = (key_offsets < length) & (selected >= 0) & (selected < seq_len_k)
            if CAUSAL:
                valid = valid & (selected + key_position_offset <= query_position)
            safe_selected = tl.where(valid, selected, 0)
            k_ptrs = (k_ptr + batch * stride_kb + safe_selected[:, None] * stride_ks + kv_head * stride_kh +
                      d_offsets[None, :] * stride_kd)
            v_ptrs = (v_ptr + batch * stride_vb + safe_selected[:, None] * stride_vs + kv_head * stride_vh +
                      d_offsets[None, :] * stride_vd)
            k = tl.load(k_ptrs, mask=valid[:, None] & (d_offsets[None, :] < head_dim), other=0.0).to(tl.bfloat16)
            v = tl.load(v_ptrs, mask=valid[:, None] & (d_offsets[None, :] < head_dim), other=0.0).to(tl.bfloat16)
            scores = tl.reshape(tl.dot(k, q[:, None], out_dtype=tl.float32), (BLOCK_K,)) * softmax_scale
            probabilities = tl.exp(tl.where(valid, scores - lse, -float("inf")))
            d_probability = tl.sum(v * grad_output[None, :], axis=1)
            correction += tl.sum(probabilities * d_probability, axis=0)

        grad_q = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for key_start in tl.range(0, K, BLOCK_K):
            key_offsets = key_start + tl.arange(0, BLOCK_K)
            selected = tl.load(
                index_ptr + batch * stride_ib + query * stride_is + key_offsets * stride_ik,
                mask=key_offsets < K,
                other=-1,
            ).to(tl.int32)
            valid = (key_offsets < length) & (selected >= 0) & (selected < seq_len_k)
            if CAUSAL:
                valid = valid & (selected + key_position_offset <= query_position)
            safe_selected = tl.where(valid, selected, 0)
            k_ptrs = (k_ptr + batch * stride_kb + safe_selected[:, None] * stride_ks + kv_head * stride_kh +
                      d_offsets[None, :] * stride_kd)
            v_ptrs = (v_ptr + batch * stride_vb + safe_selected[:, None] * stride_vs + kv_head * stride_vh +
                      d_offsets[None, :] * stride_vd)
            k = tl.load(k_ptrs, mask=valid[:, None] & (d_offsets[None, :] < head_dim), other=0.0).to(tl.bfloat16)
            v = tl.load(v_ptrs, mask=valid[:, None] & (d_offsets[None, :] < head_dim), other=0.0).to(tl.bfloat16)
            scores = tl.reshape(tl.dot(k, q[:, None], out_dtype=tl.float32), (BLOCK_K,)) * softmax_scale
            probabilities = tl.exp(tl.where(valid, scores - lse, -float("inf")))
            d_probability = tl.sum(v * grad_output[None, :], axis=1)
            d_score = probabilities * (d_probability - correction) + grad_lse * probabilities
            d_score = tl.where(valid, d_score, 0.0)
            grad_q += tl.sum(d_score[:, None] * k, axis=0) * softmax_scale

            grad_k_ptrs = (grad_k_ptr + batch * stride_dkb + safe_selected[:, None] * stride_dks + kv_head * stride_dkh
                           + d_offsets[None, :] * stride_dkd)
            grad_v_ptrs = (grad_v_ptr + batch * stride_dvb + safe_selected[:, None] * stride_dvs + kv_head * stride_dvh
                           + d_offsets[None, :] * stride_dvd)
            tl.atomic_add(grad_k_ptrs, d_score[:, None] * q[None, :] * softmax_scale,
                          mask=valid[:, None] & (d_offsets[None, :] < head_dim))
            tl.atomic_add(grad_v_ptrs, probabilities[:, None] * grad_output[None, :],
                          mask=valid[:, None] & (d_offsets[None, :] < head_dim))

        grad_q_base = grad_q_ptr + batch * stride_dqb + query * stride_dqs + head * stride_dqh
        tl.store(grad_q_base + d_offsets * stride_dqd, grad_q, mask=d_offsets < head_dim)

    @triton.jit
    def _qsa_load_route_tokens(
        route_base,
        key_start,
        key_offsets,
        length,
        query_position,
        stride_route,
        ROUTE_SLOTS: tl.constexpr,
        ROUTE_BLOCK_SIZE: tl.constexpr,
        KEY_TILE: tl.constexpr,
    ):
        """Map compact block IDs or public token IDs to physical KV tokens.

        A compact row stores only complete-block IDs.  ``length`` remains the
        public token count, so its quotient/remainder encode the selected
        block count and causal-tail length.  The incomplete causal block is
        derived from ``query_position`` and never materialized in metadata.
        """

        if ROUTE_BLOCK_SIZE == 1:
            selected = tl.load(
                route_base + key_offsets * stride_route,
                mask=key_offsets < ROUTE_SLOTS,
                other=-1,
            ).to(tl.int32)
            route_valid = key_offsets < length
        else:
            selected_block_count = length // ROUTE_BLOCK_SIZE
            block_token_count = selected_block_count * ROUTE_BLOCK_SIZE
            from_complete_block = key_offsets < block_token_count
            if KEY_TILE % ROUTE_BLOCK_SIZE == 0:
                # Load each block ID once, then broadcast its R token lanes in
                # registers.  Loading through ``key_offsets // R`` issues R
                # redundant scalar route loads and was measurable in both
                # forward and backward on SM90.
                route_offsets = (
                    key_start // ROUTE_BLOCK_SIZE
                    + tl.arange(0, KEY_TILE // ROUTE_BLOCK_SIZE)
                )
                compact_block_ids = tl.load(
                    route_base + route_offsets * stride_route,
                    mask=(route_offsets < selected_block_count)
                    & (route_offsets < ROUTE_SLOTS),
                    other=-1,
                ).to(tl.int32)
                token_lanes = tl.arange(0, ROUTE_BLOCK_SIZE)
                block_tokens = tl.reshape(
                    compact_block_ids[:, None] * ROUTE_BLOCK_SIZE
                    + token_lanes[None, :],
                    (KEY_TILE,),
                )
                block_ids = tl.reshape(
                    compact_block_ids[:, None]
                    + tl.zeros(
                        (KEY_TILE // ROUTE_BLOCK_SIZE, ROUTE_BLOCK_SIZE),
                        tl.int32,
                    ),
                    (KEY_TILE,),
                )
            else:
                block_offsets = key_offsets // ROUTE_BLOCK_SIZE
                block_ids = tl.load(
                    route_base + block_offsets * stride_route,
                    mask=from_complete_block & (block_offsets < ROUTE_SLOTS),
                    other=-1,
                ).to(tl.int32)
                block_tokens = (
                    block_ids * ROUTE_BLOCK_SIZE
                    + key_offsets % ROUTE_BLOCK_SIZE
                )
            tail_start = (
                (query_position + 1) // ROUTE_BLOCK_SIZE
            ) * ROUTE_BLOCK_SIZE
            tail_tokens = tail_start + key_offsets - block_token_count
            selected = tl.where(
                from_complete_block, block_tokens, tail_tokens
            )
            route_valid = (
                (key_offsets < length)
                & tl.where(from_complete_block, block_ids >= 0, True)
            )
        return selected, route_valid

    @triton.jit
    def _qsa_emit_split64_dkv(
        q,
        grad_output,
        d_score,
        probabilities,
        safe_selected,
        emit_mask,
        grad_k_base,
        grad_v_base,
        head_dim,
        softmax_scale,
        stride_dks,
        stride_dkd,
        stride_dvs,
        stride_dvd,
        HEADS_PER_KV: tl.constexpr,
        BLOCK_D: tl.constexpr,
        DKV_ACCUM_BF16: tl.constexpr,
    ):
        """Emit a 64-key derivative tile as two bounded 32-key MMA results."""

        half_k: tl.constexpr = 32
        d_offsets = tl.arange(0, BLOCK_D)
        score_halves = tl.permute(
            tl.reshape(d_score, (HEADS_PER_KV, 2, half_k)),
            (0, 2, 1),
        )
        d_score_low, d_score_high = tl.split(score_halves)
        probability_halves = tl.permute(
            tl.reshape(probabilities, (HEADS_PER_KV, 2, half_k)),
            (0, 2, 1),
        )
        probability_low, probability_high = tl.split(probability_halves)
        selected_halves = tl.permute(
            tl.reshape(safe_selected, (2, half_k)), (1, 0)
        )
        selected_low, selected_high = tl.split(selected_halves)
        mask_halves = tl.permute(
            tl.reshape(emit_mask, (2, half_k)), (1, 0)
        )
        mask_low, mask_high = tl.split(mask_halves)

        grad_k_low = tl.dot(
            tl.trans(d_score_low.to(tl.bfloat16)),
            q,
            out_dtype=tl.float32,
        ) * softmax_scale
        grad_k_low_ptrs = (
            grad_k_base
            + selected_low[:, None] * stride_dks
            + d_offsets[None, :] * stride_dkd
        )
        if DKV_ACCUM_BF16:
            tl.atomic_add(
                grad_k_low_ptrs,
                grad_k_low.to(tl.bfloat16),
                mask=mask_low[:, None] & (d_offsets[None, :] < head_dim),
                sem="relaxed",
            )
        else:
            tl.atomic_add(
                grad_k_low_ptrs,
                grad_k_low,
                mask=mask_low[:, None] & (d_offsets[None, :] < head_dim),
                sem="relaxed",
            )
        grad_k_high = tl.dot(
            tl.trans(d_score_high.to(tl.bfloat16)),
            q,
            out_dtype=tl.float32,
        ) * softmax_scale
        grad_k_high_ptrs = (
            grad_k_base
            + selected_high[:, None] * stride_dks
            + d_offsets[None, :] * stride_dkd
        )
        if DKV_ACCUM_BF16:
            tl.atomic_add(
                grad_k_high_ptrs,
                grad_k_high.to(tl.bfloat16),
                mask=mask_high[:, None] & (d_offsets[None, :] < head_dim),
                sem="relaxed",
            )
        else:
            tl.atomic_add(
                grad_k_high_ptrs,
                grad_k_high,
                mask=mask_high[:, None] & (d_offsets[None, :] < head_dim),
                sem="relaxed",
            )

        grad_v_low = tl.dot(
            tl.trans(probability_low.to(tl.bfloat16)),
            grad_output.to(tl.bfloat16),
            out_dtype=tl.float32,
        )
        grad_v_low_ptrs = (
            grad_v_base
            + selected_low[:, None] * stride_dvs
            + d_offsets[None, :] * stride_dvd
        )
        if DKV_ACCUM_BF16:
            tl.atomic_add(
                grad_v_low_ptrs,
                grad_v_low.to(tl.bfloat16),
                mask=mask_low[:, None] & (d_offsets[None, :] < head_dim),
                sem="relaxed",
            )
        else:
            tl.atomic_add(
                grad_v_low_ptrs,
                grad_v_low,
                mask=mask_low[:, None] & (d_offsets[None, :] < head_dim),
                sem="relaxed",
            )
        grad_v_high = tl.dot(
            tl.trans(probability_high.to(tl.bfloat16)),
            grad_output.to(tl.bfloat16),
            out_dtype=tl.float32,
        )
        grad_v_high_ptrs = (
            grad_v_base
            + selected_high[:, None] * stride_dvs
            + d_offsets[None, :] * stride_dvd
        )
        if DKV_ACCUM_BF16:
            tl.atomic_add(
                grad_v_high_ptrs,
                grad_v_high.to(tl.bfloat16),
                mask=mask_high[:, None] & (d_offsets[None, :] < head_dim),
                sem="relaxed",
            )
        else:
            tl.atomic_add(
                grad_v_high_ptrs,
                grad_v_high,
                mask=mask_high[:, None] & (d_offsets[None, :] < head_dim),
                sem="relaxed",
            )

    @triton.jit
    def _qsa_selected_kv_forward_grouped_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        index_ptr,
        length_ptr,
        out_ptr,
        lse_ptr,
        seq_len_q,
        seq_len_k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        key_position_offset,
        query_position_ptr,
        stride_qp,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vd,
        stride_ib,
        stride_is,
        stride_ik,
        stride_lb,
        stride_ls,
        stride_ob,
        stride_os,
        stride_oh,
        stride_od,
        stride_lseb,
        stride_lseh,
        stride_lses,
        K: tl.constexpr,
        HEADS_PER_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        NUM_HEAD_TILES: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
        TRIM_CAUSAL_LOOP: tl.constexpr,
        ROUTE_SLOTS: tl.constexpr,
        ROUTE_BLOCK_SIZE: tl.constexpr,
    ):
        """Grouped forward: one program reuses K/V for a small GQA head tile."""

        batch_query = tl.program_id(0)
        batch = batch_query // seq_len_q
        query = batch_query - batch * seq_len_q
        head_tile_program = tl.program_id(1)
        kv_head = head_tile_program // NUM_HEAD_TILES
        head_tile = head_tile_program % NUM_HEAD_TILES
        head_offsets = tl.arange(0, HEADS_PER_KV)
        heads = kv_head * GROUP_SIZE + head_tile * HEADS_PER_KV + head_offsets
        head_valid = (head_tile * HEADS_PER_KV + head_offsets < GROUP_SIZE) & (heads < num_q_heads)
        d_offsets = tl.arange(0, BLOCK_D)
        q_ptrs = (q_ptr + batch * stride_qb + query * stride_qs + heads[:, None] * stride_qh +
                  d_offsets[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=head_valid[:, None] & (d_offsets[None, :] < head_dim), other=0.0)
        q = q.to(tl.bfloat16)
        length = tl.load(length_ptr + batch * stride_lb + query * stride_ls).to(tl.int32)
        query_position = tl.load(query_position_ptr + query * stride_qp).to(tl.int32)
        max_value = tl.full((HEADS_PER_KV,), -float("inf"), tl.float32)
        denominator = tl.zeros((HEADS_PER_KV,), dtype=tl.float32)
        accumulator = tl.zeros((HEADS_PER_KV, BLOCK_D), dtype=tl.float32)

        loop_end = K
        if TRIM_CAUSAL_LOOP:
            loop_end = tl.minimum(length, K)
        for key_start in tl.range(0, loop_end, BLOCK_K):
            key_offsets = key_start + tl.arange(0, BLOCK_K)
            selected, route_valid = _qsa_load_route_tokens(
                index_ptr + batch * stride_ib + query * stride_is,
                key_start,
                key_offsets,
                length,
                query_position,
                stride_ik,
                ROUTE_SLOTS,
                ROUTE_BLOCK_SIZE,
                BLOCK_K,
            )
            valid = route_valid & (selected >= 0) & (selected < seq_len_k)
            if CAUSAL:
                valid = valid & (selected + key_position_offset <= query_position)
            safe_selected = tl.where(valid, selected, 0)
            k_ptrs = (k_ptr + batch * stride_kb + safe_selected[:, None] * stride_ks + kv_head * stride_kh +
                      d_offsets[None, :] * stride_kd)
            v_ptrs = (v_ptr + batch * stride_vb + safe_selected[:, None] * stride_vs + kv_head * stride_vh +
                      d_offsets[None, :] * stride_vd)
            k = tl.load(
                k_ptrs,
                mask=valid[:, None] & (d_offsets[None, :] < head_dim),
                other=0.0,
            ).to(tl.bfloat16)
            v = tl.load(
                v_ptrs,
                mask=valid[:, None] & (d_offsets[None, :] < head_dim),
                other=0.0,
            ).to(tl.bfloat16)
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * softmax_scale
            scores = tl.where(valid[None, :], scores, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(max_value, tile_max)
            old_scale = tl.where(
                max_value == -float("inf"),
                0.0,
                tl.exp2((max_value - new_max) * 1.4426950408889634),
            )
            probabilities = tl.exp2(
                tl.where(
                    valid[None, :],
                    (scores - new_max[:, None]) * 1.4426950408889634,
                    -float("inf"),
                )
            )
            tile_sum = tl.sum(probabilities, axis=1)
            denominator = denominator * old_scale + tile_sum
            # Express P @ V as a matrix product so SM90 can use the same
            # tensor-core path as Q @ K.  The previous broadcasted multiply
            # materialized a [HEADS_PER_KV, BLOCK_K, BLOCK_D] temporary and
            # reduced it elementwise, which inflated the live register set.
            accumulator = accumulator * old_scale[:, None] + tl.dot(
                probabilities.to(tl.bfloat16), v, out_dtype=tl.float32
            )
            max_value = new_max

        has_value = denominator > 0.0
        output = tl.where(has_value[:, None], accumulator / denominator[:, None], 0.0)
        log_sum_exp = tl.where(
            has_value,
            max_value + tl.log2(denominator) / 1.4426950408889634,
            -float("inf"),
        )
        out_ptrs = (out_ptr + batch * stride_ob + query * stride_os + heads[:, None] * stride_oh +
                    d_offsets[None, :] * stride_od)
        tl.store(out_ptrs, output, mask=head_valid[:, None] & (d_offsets[None, :] < head_dim))
        lse_ptrs = lse_ptr + batch * stride_lseb + heads * stride_lseh + query * stride_lses
        tl.store(lse_ptrs, log_sum_exp, mask=head_valid)

    @triton.jit
    def _qsa_selected_kv_forward_packed_grouped_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        index_ptr,
        length_ptr,
        key_start_ptr,
        key_length_ptr,
        query_position_ptr,
        out_ptr,
        lse_ptr,
        total_q,
        total_k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        key_position_offset,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_it,
        stride_ik,
        stride_lt,
        stride_kst,
        stride_klt,
        stride_qpt,
        stride_ot,
        stride_oh,
        stride_od,
        stride_lseh,
        stride_lset,
        K: tl.constexpr,
        HEADS_PER_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        NUM_HEAD_TILES: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
        TRIM_CAUSAL_LOOP: tl.constexpr,
        ROUTE_SLOTS: tl.constexpr,
        ROUTE_BLOCK_SIZE: tl.constexpr,
    ):
        """Packed-THD selected-KV forward with one launch for all segments."""

        token = tl.program_id(0)
        head_tile_program = tl.program_id(1)
        kv_head = head_tile_program // NUM_HEAD_TILES
        head_tile = head_tile_program % NUM_HEAD_TILES
        head_offsets = tl.arange(0, HEADS_PER_KV)
        heads = kv_head * GROUP_SIZE + head_tile * HEADS_PER_KV + head_offsets
        head_valid = (
            (head_tile * HEADS_PER_KV + head_offsets < GROUP_SIZE)
            & (heads < num_q_heads)
        )
        d_offsets = tl.arange(0, BLOCK_D)
        q_ptrs = (
            q_ptr
            + token * stride_qt
            + heads[:, None] * stride_qh
            + d_offsets[None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=head_valid[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        ).to(tl.bfloat16)
        length = tl.load(length_ptr + token * stride_lt).to(tl.int32)
        key_start = tl.load(key_start_ptr + token * stride_kst).to(tl.int32)
        key_length = tl.load(key_length_ptr + token * stride_klt).to(tl.int32)
        query_position = tl.load(query_position_ptr + token * stride_qpt).to(tl.int32)
        max_value = tl.full((HEADS_PER_KV,), -float("inf"), tl.float32)
        denominator = tl.zeros((HEADS_PER_KV,), dtype=tl.float32)
        accumulator = tl.zeros((HEADS_PER_KV, BLOCK_D), dtype=tl.float32)

        loop_end = K
        if TRIM_CAUSAL_LOOP:
            loop_end = tl.minimum(length, K)
        for key_offset_start in tl.range(0, loop_end, BLOCK_K):
            key_offsets = key_offset_start + tl.arange(0, BLOCK_K)
            selected, route_valid = _qsa_load_route_tokens(
                index_ptr + token * stride_it,
                key_offset_start,
                key_offsets,
                length,
                query_position,
                stride_ik,
                ROUTE_SLOTS,
                ROUTE_BLOCK_SIZE,
                BLOCK_K,
            )
            valid = (
                route_valid
                & (selected >= 0)
                & (selected < key_length)
            )
            if CAUSAL:
                valid = valid & (selected + key_position_offset <= query_position)
            safe_selected = key_start + tl.where(valid, selected, 0)
            k_ptrs = (
                k_ptr
                + safe_selected[:, None] * stride_kt
                + kv_head * stride_kh
                + d_offsets[None, :] * stride_kd
            )
            v_ptrs = (
                v_ptr
                + safe_selected[:, None] * stride_vt
                + kv_head * stride_vh
                + d_offsets[None, :] * stride_vd
            )
            k = tl.load(
                k_ptrs,
                mask=valid[:, None] & (d_offsets[None, :] < head_dim),
                other=0.0,
            ).to(tl.bfloat16)
            v = tl.load(
                v_ptrs,
                mask=valid[:, None] & (d_offsets[None, :] < head_dim),
                other=0.0,
            ).to(tl.bfloat16)
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * softmax_scale
            scores = tl.where(valid[None, :], scores, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(max_value, tile_max)
            old_scale = tl.where(
                max_value == -float("inf"),
                0.0,
                tl.exp2((max_value - new_max) * 1.4426950408889634),
            )
            probabilities = tl.exp2(
                tl.where(
                    valid[None, :],
                    (scores - new_max[:, None]) * 1.4426950408889634,
                    -float("inf"),
                )
            )
            tile_sum = tl.sum(probabilities, axis=1)
            denominator = denominator * old_scale + tile_sum
            accumulator = accumulator * old_scale[:, None] + tl.dot(
                probabilities.to(tl.bfloat16), v, out_dtype=tl.float32
            )
            max_value = new_max

        has_value = denominator > 0.0
        output = tl.where(has_value[:, None], accumulator / denominator[:, None], 0.0)
        log_sum_exp = tl.where(
            has_value,
            max_value + tl.log2(denominator) / 1.4426950408889634,
            -float("inf"),
        )
        out_ptrs = (
            out_ptr
            + token * stride_ot
            + heads[:, None] * stride_oh
            + d_offsets[None, :] * stride_od
        )
        tl.store(
            out_ptrs,
            output,
            mask=head_valid[:, None] & (d_offsets[None, :] < head_dim),
        )
        lse_ptrs = lse_ptr + heads * stride_lseh + token * stride_lset
        tl.store(lse_ptrs, log_sum_exp, mask=head_valid)

    @triton.jit
    def _qsa_selected_kv_backward_grouped_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        index_ptr,
        length_ptr,
        lse_ptr,
        grad_out_ptr,
        out_ptr,
        grad_lse_ptr,
        grad_q_ptr,
        grad_k_ptr,
        grad_v_ptr,
        seq_len_q,
        seq_len_k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        key_position_offset,
        query_position_ptr,
        stride_qp,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vd,
        stride_ib,
        stride_is,
        stride_ik,
        stride_lb,
        stride_ls,
        stride_lseb,
        stride_lseh,
        stride_lses,
        stride_gob,
        stride_gos,
        stride_goh,
        stride_god,
        stride_ob,
        stride_os,
        stride_oh,
        stride_od,
        stride_glseb,
        stride_glseh,
        stride_glses,
        stride_dqb,
        stride_dqs,
        stride_dqh,
        stride_dqd,
        stride_dkb,
        stride_dks,
        stride_dkh,
        stride_dkd,
        stride_dvb,
        stride_dvs,
        stride_dvh,
        stride_dvd,
        correction_ptr,
        stride_cb,
        stride_ch,
        stride_cs,
        K: tl.constexpr,
        HEADS_PER_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        NUM_HEAD_TILES: tl.constexpr,
        BLOCK_K: tl.constexpr,
        CORRECTION_BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
        HAS_GRAD_OUTPUT: tl.constexpr,
        HAS_GRAD_LSE: tl.constexpr,
        USE_OUTPUT_DELTA: tl.constexpr,
        DKV_ACCUM_BF16: tl.constexpr,
        TENSORIZE_DERIVATIVES: tl.constexpr,
        TRIM_CAUSAL_LOOP: tl.constexpr,
        ROUTE_SLOTS: tl.constexpr,
        ROUTE_BLOCK_SIZE: tl.constexpr,
        SEGMENT_BLOCK_TOPK: tl.constexpr,
        RATIO: tl.constexpr,
        STORE_CORRECTION: tl.constexpr,
        TAIL_ONLY: tl.constexpr,
        EMIT_DKV: tl.constexpr,
    ):
        """Grouped backward with one configurable atomic dK/dV update per GQA head tile."""

        batch_query = tl.program_id(0)
        batch = batch_query // seq_len_q
        query = batch_query - batch * seq_len_q
        head_tile_program = tl.program_id(1)
        kv_head = head_tile_program // NUM_HEAD_TILES
        head_tile = head_tile_program % NUM_HEAD_TILES
        head_offsets = tl.arange(0, HEADS_PER_KV)
        heads = kv_head * GROUP_SIZE + head_tile * HEADS_PER_KV + head_offsets
        head_valid = (head_tile * HEADS_PER_KV + head_offsets < GROUP_SIZE) & (heads < num_q_heads)
        d_offsets = tl.arange(0, BLOCK_D)
        q_ptrs = (q_ptr + batch * stride_qb + query * stride_qs + heads[:, None] * stride_qh +
                  d_offsets[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=head_valid[:, None] & (d_offsets[None, :] < head_dim), other=0.0)
        q = q.to(tl.bfloat16)
        grad_out_ptrs = (grad_out_ptr + batch * stride_gob + query * stride_gos + heads[:, None] * stride_goh +
                         d_offsets[None, :] * stride_god)
        grad_output = tl.load(
            grad_out_ptrs,
            mask=head_valid[:, None] & (d_offsets[None, :] < head_dim) & HAS_GRAD_OUTPUT,
            other=0.0,
        ).to(tl.float32)
        query_position = tl.load(query_position_ptr + query * stride_qp).to(tl.int32)
        length = tl.load(length_ptr + batch * stride_lb + query * stride_ls).to(tl.int32)
        lse = tl.load(
            lse_ptr + batch * stride_lseb + heads * stride_lseh + query * stride_lses,
            mask=head_valid,
            other=0.0,
        ).to(tl.float32)
        grad_lse = tl.load(
            grad_lse_ptr + batch * stride_glseb + heads * stride_glseh + query * stride_glses,
            mask=head_valid & HAS_GRAD_LSE,
            other=0.0,
        ).to(tl.float32)

        if USE_OUTPUT_DELTA:
            # For O = sum_j(P_j V_j), the softmax correction
            # sum_j(P_j * dot(dO, V_j)) is exactly dot(dO, O).  FlashAttention
            # calls this quantity delta.  Reading the BF16 forward output once
            # removes the former full K/V rescan and keeps workspace O(1).
            out_ptrs = (
                out_ptr
                + batch * stride_ob
                + query * stride_os
                + heads[:, None] * stride_oh
                + d_offsets[None, :] * stride_od
            )
            output = tl.load(
                out_ptrs,
                mask=head_valid[:, None] & (d_offsets[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            correction = tl.sum(output * grad_output, axis=1)
        else:
            correction = tl.zeros((HEADS_PER_KV,), dtype=tl.float32)
            correction_loop_end = K
            if TRIM_CAUSAL_LOOP:
                correction_loop_end = tl.minimum(length, K)
            for key_start in tl.range(
                0, correction_loop_end, CORRECTION_BLOCK_K
            ):
                key_offsets = key_start + tl.arange(0, CORRECTION_BLOCK_K)
                selected, route_valid = _qsa_load_route_tokens(
                    index_ptr + batch * stride_ib + query * stride_is,
                    key_start,
                    key_offsets,
                    length,
                    query_position,
                    stride_ik,
                    ROUTE_SLOTS,
                    ROUTE_BLOCK_SIZE,
                    CORRECTION_BLOCK_K,
                )
                valid = route_valid & (selected >= 0) & (selected < seq_len_k)
                if CAUSAL:
                    valid = valid & (selected + key_position_offset <= query_position)
                safe_selected = tl.where(valid, selected, 0)
                k_ptrs = (k_ptr + batch * stride_kb + safe_selected[:, None] * stride_ks + kv_head * stride_kh +
                          d_offsets[None, :] * stride_kd)
                v_ptrs = (v_ptr + batch * stride_vb + safe_selected[:, None] * stride_vs + kv_head * stride_vh +
                          d_offsets[None, :] * stride_vd)
                k = tl.load(
                    k_ptrs,
                    mask=valid[:, None] & (d_offsets[None, :] < head_dim),
                    other=0.0,
                ).to(tl.bfloat16)
                v = tl.load(
                    v_ptrs,
                    mask=valid[:, None] & (d_offsets[None, :] < head_dim),
                    other=0.0,
                ).to(tl.bfloat16)
                scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * softmax_scale
                probabilities = tl.exp2(tl.where(
                    valid[None, :],
                    (scores - lse[:, None]) * 1.4426950408889634,
                    -float("inf"),
                ))
                d_probability = tl.dot(
                    grad_output.to(tl.bfloat16),
                    tl.trans(v),
                    out_dtype=tl.float32,
                )
                correction += tl.sum(probabilities * d_probability, axis=1)

        if STORE_CORRECTION:
            correction_ptrs = (
                correction_ptr
                + batch * stride_cb
                + heads * stride_ch
                + query * stride_cs
            )
            tl.store(correction_ptrs, correction, mask=head_valid)

        grad_q = tl.zeros((HEADS_PER_KV, BLOCK_D), dtype=tl.float32)
        loop_end = K
        if TRIM_CAUSAL_LOOP:
            loop_end = tl.minimum(length, K)
        for key_start in tl.range(0, loop_end, BLOCK_K):
            key_offsets = key_start + tl.arange(0, BLOCK_K)
            selected, route_valid = _qsa_load_route_tokens(
                index_ptr + batch * stride_ib + query * stride_is,
                key_start,
                key_offsets,
                length,
                query_position,
                stride_ik,
                ROUTE_SLOTS,
                ROUTE_BLOCK_SIZE,
                BLOCK_K,
            )
            valid = route_valid & (selected >= 0) & (selected < seq_len_k)
            if CAUSAL:
                valid = valid & (selected + key_position_offset <= query_position)
            safe_selected = tl.where(valid, selected, 0)
            k_ptrs = (k_ptr + batch * stride_kb + safe_selected[:, None] * stride_ks + kv_head * stride_kh +
                      d_offsets[None, :] * stride_kd)
            v_ptrs = (v_ptr + batch * stride_vb + safe_selected[:, None] * stride_vs + kv_head * stride_vh +
                      d_offsets[None, :] * stride_vd)
            k = tl.load(k_ptrs, mask=valid[:, None] & (d_offsets[None, :] < head_dim), other=0.0).to(tl.bfloat16)
            v = tl.load(v_ptrs, mask=valid[:, None] & (d_offsets[None, :] < head_dim), other=0.0).to(tl.bfloat16)
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * softmax_scale
            probabilities = tl.exp2(tl.where(
                valid[None, :],
                (scores - lse[:, None]) * 1.4426950408889634,
                -float("inf"),
            ))
            d_probability = tl.dot(
                grad_output.to(tl.bfloat16),
                tl.trans(v),
                out_dtype=tl.float32,
            )
            d_score = probabilities * (d_probability - correction[:, None]) + grad_lse[:, None] * probabilities
            d_score = tl.where(
                valid[None, :] & head_valid[:, None], d_score, 0.0)
            if TENSORIZE_DERIVATIVES:
                grad_q += tl.dot(
                    d_score.to(tl.bfloat16), k, out_dtype=tl.float32
                ) * softmax_scale
            else:
                grad_q += tl.sum(
                    d_score[:, :, None] * k[None, :, :], axis=1
                ) * softmax_scale

            if EMIT_DKV:
                emit_mask = valid
                if TAIL_ONLY:
                    tail_start = tl.minimum(
                        (query_position + 1) // RATIO, SEGMENT_BLOCK_TOPK
                    ) * RATIO
                    emit_mask = emit_mask & (key_offsets >= tail_start)
                # In segmented mode EMIT_DKV is compile-time false, so the
                # complete-block dK/dV reductions are removed entirely from
                # this dQ/correction kernel.  The small causal tail has a
                # dedicated kernel below; otherwise the segmented reducer is
                # the sole producer of complete-block dK/dV.
                if TENSORIZE_DERIVATIVES and BLOCK_K == 64:
                    _qsa_emit_split64_dkv(
                        q,
                        grad_output,
                        d_score,
                        probabilities,
                        safe_selected,
                        emit_mask,
                        grad_k_ptr
                        + batch * stride_dkb
                        + kv_head * stride_dkh,
                        grad_v_ptr
                        + batch * stride_dvb
                        + kv_head * stride_dvh,
                        head_dim,
                        softmax_scale,
                        stride_dks,
                        stride_dkd,
                        stride_dvs,
                        stride_dvd,
                        HEADS_PER_KV,
                        BLOCK_D,
                        DKV_ACCUM_BF16,
                    )
                elif TENSORIZE_DERIVATIVES:
                    grad_k = tl.dot(
                        tl.trans(d_score.to(tl.bfloat16)),
                        q,
                        out_dtype=tl.float32,
                    ) * softmax_scale
                else:
                    grad_k = tl.sum(
                        d_score[:, :, None] * q[:, None, :], axis=0
                    ) * softmax_scale
                if not (TENSORIZE_DERIVATIVES and BLOCK_K == 64):
                    grad_k_ptrs = (
                        grad_k_ptr
                        + batch * stride_dkb
                        + safe_selected[:, None] * stride_dks
                        + kv_head * stride_dkh
                        + d_offsets[None, :] * stride_dkd
                    )
                    if DKV_ACCUM_BF16:
                        tl.atomic_add(
                            grad_k_ptrs,
                            grad_k.to(tl.bfloat16),
                            mask=emit_mask[:, None] & (d_offsets[None, :] < head_dim),
                            sem="relaxed",
                        )
                    else:
                        tl.atomic_add(
                            grad_k_ptrs,
                            grad_k,
                            mask=emit_mask[:, None] & (d_offsets[None, :] < head_dim),
                            sem="relaxed",
                        )

                    # Retire the dK tensor-core result before creating dV.  The
                    # two [BLOCK_K, D] FP32 accumulators otherwise overlap their
                    # live ranges and force the SM90 kernel to its 255-register
                    # ceiling even though their atomics are already serialized.
                    if TENSORIZE_DERIVATIVES:
                        grad_v = tl.dot(
                            tl.trans(probabilities.to(tl.bfloat16)),
                            grad_output.to(tl.bfloat16),
                            out_dtype=tl.float32,
                        )
                    else:
                        grad_v = tl.sum(
                            probabilities[:, :, None] * grad_output[:, None, :], axis=0
                        )
                    grad_v_ptrs = (
                        grad_v_ptr
                        + batch * stride_dvb
                        + safe_selected[:, None] * stride_dvs
                        + kv_head * stride_dvh
                        + d_offsets[None, :] * stride_dvd
                    )
                    if DKV_ACCUM_BF16:
                        tl.atomic_add(
                            grad_v_ptrs,
                            grad_v.to(tl.bfloat16),
                            mask=emit_mask[:, None] & (d_offsets[None, :] < head_dim),
                            sem="relaxed",
                        )
                    else:
                        tl.atomic_add(
                            grad_v_ptrs,
                            grad_v,
                            mask=emit_mask[:, None] & (d_offsets[None, :] < head_dim),
                            sem="relaxed",
                        )

        grad_q_ptrs = (grad_q_ptr + batch * stride_dqb + query * stride_dqs + heads[:, None] * stride_dqh +
                       d_offsets[None, :] * stride_dqd)
        tl.store(grad_q_ptrs, grad_q, mask=head_valid[:, None] & (d_offsets[None, :] < head_dim))

    @triton.jit
    def _qsa_selected_kv_backward_packed_grouped_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        index_ptr,
        length_ptr,
        lse_ptr,
        grad_out_ptr,
        out_ptr,
        grad_lse_ptr,
        grad_q_ptr,
        grad_k_ptr,
        grad_v_ptr,
        key_start_ptr,
        key_length_ptr,
        query_position_ptr,
        total_q,
        total_k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        key_position_offset,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_it,
        stride_ik,
        stride_lt,
        stride_lseh,
        stride_lset,
        stride_got,
        stride_goh,
        stride_god,
        stride_ot,
        stride_oh,
        stride_od,
        stride_glseh,
        stride_glset,
        stride_dqt,
        stride_dqh,
        stride_dqd,
        stride_dkt,
        stride_dkh,
        stride_dkd,
        stride_dvt,
        stride_dvh,
        stride_dvd,
        stride_kst,
        stride_klt,
        stride_qpt,
        K: tl.constexpr,
        HEADS_PER_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        NUM_HEAD_TILES: tl.constexpr,
        BLOCK_K: tl.constexpr,
        CORRECTION_BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
        HAS_GRAD_OUTPUT: tl.constexpr,
        HAS_GRAD_LSE: tl.constexpr,
        USE_OUTPUT_DELTA: tl.constexpr,
        DKV_ACCUM_BF16: tl.constexpr,
        TENSORIZE_DERIVATIVES: tl.constexpr,
        TRIM_CAUSAL_LOOP: tl.constexpr,
        ROUTE_SLOTS: tl.constexpr,
        ROUTE_BLOCK_SIZE: tl.constexpr,
    ):
        """Packed-THD backward with recompute and relaxed dK/dV atomics."""

        token = tl.program_id(0)
        head_tile_program = tl.program_id(1)
        kv_head = head_tile_program // NUM_HEAD_TILES
        head_tile = head_tile_program % NUM_HEAD_TILES
        head_offsets = tl.arange(0, HEADS_PER_KV)
        heads = kv_head * GROUP_SIZE + head_tile * HEADS_PER_KV + head_offsets
        head_valid = (
            (head_tile * HEADS_PER_KV + head_offsets < GROUP_SIZE)
            & (heads < num_q_heads)
        )
        d_offsets = tl.arange(0, BLOCK_D)
        q_ptrs = (
            q_ptr
            + token * stride_qt
            + heads[:, None] * stride_qh
            + d_offsets[None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=head_valid[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        ).to(tl.bfloat16)
        grad_out_ptrs = (
            grad_out_ptr
            + token * stride_got
            + heads[:, None] * stride_goh
            + d_offsets[None, :] * stride_god
        )
        grad_output = tl.load(
            grad_out_ptrs,
            mask=head_valid[:, None]
            & (d_offsets[None, :] < head_dim)
            & HAS_GRAD_OUTPUT,
            other=0.0,
        ).to(tl.float32)
        query_position = tl.load(query_position_ptr + token * stride_qpt).to(tl.int32)
        length = tl.load(length_ptr + token * stride_lt).to(tl.int32)
        key_start = tl.load(key_start_ptr + token * stride_kst).to(tl.int32)
        key_length = tl.load(key_length_ptr + token * stride_klt).to(tl.int32)
        lse = tl.load(
            lse_ptr + heads * stride_lseh + token * stride_lset,
            mask=head_valid,
            other=0.0,
        ).to(tl.float32)
        grad_lse = tl.load(
            grad_lse_ptr + heads * stride_glseh + token * stride_glset,
            mask=head_valid & HAS_GRAD_LSE,
            other=0.0,
        ).to(tl.float32)

        if USE_OUTPUT_DELTA:
            out_ptrs = (
                out_ptr
                + token * stride_ot
                + heads[:, None] * stride_oh
                + d_offsets[None, :] * stride_od
            )
            output = tl.load(
                out_ptrs,
                mask=head_valid[:, None] & (d_offsets[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            correction = tl.sum(output * grad_output, axis=1)
        else:
            correction = tl.zeros((HEADS_PER_KV,), dtype=tl.float32)
            correction_loop_end = K
            if TRIM_CAUSAL_LOOP:
                correction_loop_end = tl.minimum(length, K)
            for key_offset_start in tl.range(
                0, correction_loop_end, CORRECTION_BLOCK_K
            ):
                key_offsets = key_offset_start + tl.arange(0, CORRECTION_BLOCK_K)
                selected, route_valid = _qsa_load_route_tokens(
                    index_ptr + token * stride_it,
                    key_offset_start,
                    key_offsets,
                    length,
                    query_position,
                    stride_ik,
                    ROUTE_SLOTS,
                    ROUTE_BLOCK_SIZE,
                    CORRECTION_BLOCK_K,
                )
                valid = (
                    route_valid
                    & (selected >= 0)
                    & (selected < key_length)
                )
                if CAUSAL:
                    valid = valid & (selected + key_position_offset <= query_position)
                safe_selected = key_start + tl.where(valid, selected, 0)
                k_ptrs = (
                    k_ptr
                    + safe_selected[:, None] * stride_kt
                    + kv_head * stride_kh
                    + d_offsets[None, :] * stride_kd
                )
                v_ptrs = (
                    v_ptr
                    + safe_selected[:, None] * stride_vt
                    + kv_head * stride_vh
                    + d_offsets[None, :] * stride_vd
                )
                k = tl.load(
                    k_ptrs,
                    mask=valid[:, None] & (d_offsets[None, :] < head_dim),
                    other=0.0,
                ).to(tl.bfloat16)
                v = tl.load(
                    v_ptrs,
                    mask=valid[:, None] & (d_offsets[None, :] < head_dim),
                    other=0.0,
                ).to(tl.bfloat16)
                scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * softmax_scale
                probabilities = tl.exp2(
                    tl.where(
                        valid[None, :],
                        (scores - lse[:, None]) * 1.4426950408889634,
                        -float("inf"),
                    )
                )
                d_probability = tl.dot(
                    grad_output.to(tl.bfloat16),
                    tl.trans(v),
                    out_dtype=tl.float32,
                )
                correction += tl.sum(probabilities * d_probability, axis=1)

        grad_q = tl.zeros((HEADS_PER_KV, BLOCK_D), dtype=tl.float32)
        loop_end = K
        if TRIM_CAUSAL_LOOP:
            loop_end = tl.minimum(length, K)
        for key_offset_start in tl.range(0, loop_end, BLOCK_K):
            key_offsets = key_offset_start + tl.arange(0, BLOCK_K)
            selected, route_valid = _qsa_load_route_tokens(
                index_ptr + token * stride_it,
                key_offset_start,
                key_offsets,
                length,
                query_position,
                stride_ik,
                ROUTE_SLOTS,
                ROUTE_BLOCK_SIZE,
                BLOCK_K,
            )
            valid = (
                route_valid
                & (selected >= 0)
                & (selected < key_length)
            )
            if CAUSAL:
                valid = valid & (selected + key_position_offset <= query_position)
            safe_selected = key_start + tl.where(valid, selected, 0)
            k_ptrs = (
                k_ptr
                + safe_selected[:, None] * stride_kt
                + kv_head * stride_kh
                + d_offsets[None, :] * stride_kd
            )
            v_ptrs = (
                v_ptr
                + safe_selected[:, None] * stride_vt
                + kv_head * stride_vh
                + d_offsets[None, :] * stride_vd
            )
            k = tl.load(
                k_ptrs,
                mask=valid[:, None] & (d_offsets[None, :] < head_dim),
                other=0.0,
            ).to(tl.bfloat16)
            v = tl.load(
                v_ptrs,
                mask=valid[:, None] & (d_offsets[None, :] < head_dim),
                other=0.0,
            ).to(tl.bfloat16)
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * softmax_scale
            probabilities = tl.exp2(
                tl.where(
                    valid[None, :],
                    (scores - lse[:, None]) * 1.4426950408889634,
                    -float("inf"),
                )
            )
            d_probability = tl.dot(
                grad_output.to(tl.bfloat16),
                tl.trans(v),
                out_dtype=tl.float32,
            )
            d_score = (
                probabilities * (d_probability - correction[:, None])
                + grad_lse[:, None] * probabilities
            )
            d_score = tl.where(
                valid[None, :] & head_valid[:, None], d_score, 0.0)
            if TENSORIZE_DERIVATIVES and BLOCK_K == 64:
                grad_q += tl.dot(
                    d_score.to(tl.bfloat16), k, out_dtype=tl.float32
                ) * softmax_scale
                _qsa_emit_split64_dkv(
                    q,
                    grad_output,
                    d_score,
                    probabilities,
                    safe_selected,
                    valid,
                    grad_k_ptr + kv_head * stride_dkh,
                    grad_v_ptr + kv_head * stride_dvh,
                    head_dim,
                    softmax_scale,
                    stride_dkt,
                    stride_dkd,
                    stride_dvt,
                    stride_dvd,
                    HEADS_PER_KV,
                    BLOCK_D,
                    DKV_ACCUM_BF16,
                )
            elif TENSORIZE_DERIVATIVES:
                grad_q += tl.dot(
                    d_score.to(tl.bfloat16), k, out_dtype=tl.float32
                ) * softmax_scale
                grad_k = tl.dot(
                    tl.trans(d_score.to(tl.bfloat16)),
                    q,
                    out_dtype=tl.float32,
                ) * softmax_scale
                grad_v = tl.dot(
                    tl.trans(probabilities.to(tl.bfloat16)),
                    grad_output.to(tl.bfloat16),
                    out_dtype=tl.float32,
                )
            else:
                grad_q += tl.sum(
                    d_score[:, :, None] * k[None, :, :], axis=1
                ) * softmax_scale
                grad_k = tl.sum(
                    d_score[:, :, None] * q[:, None, :], axis=0
                ) * softmax_scale
                grad_v = tl.sum(
                    probabilities[:, :, None] * grad_output[:, None, :], axis=0
                )
            if not (TENSORIZE_DERIVATIVES and BLOCK_K == 64):
                grad_k_ptrs = (
                    grad_k_ptr
                    + safe_selected[:, None] * stride_dkt
                    + kv_head * stride_dkh
                    + d_offsets[None, :] * stride_dkd
                )
                grad_v_ptrs = (
                    grad_v_ptr
                    + safe_selected[:, None] * stride_dvt
                    + kv_head * stride_dvh
                    + d_offsets[None, :] * stride_dvd
                )
                if DKV_ACCUM_BF16:
                    tl.atomic_add(
                        grad_k_ptrs,
                        grad_k.to(tl.bfloat16),
                        mask=valid[:, None] & (d_offsets[None, :] < head_dim),
                        sem="relaxed",
                    )
                    tl.atomic_add(
                        grad_v_ptrs,
                        grad_v.to(tl.bfloat16),
                        mask=valid[:, None] & (d_offsets[None, :] < head_dim),
                        sem="relaxed",
                    )
                else:
                    tl.atomic_add(
                        grad_k_ptrs,
                        grad_k,
                        mask=valid[:, None] & (d_offsets[None, :] < head_dim),
                        sem="relaxed",
                    )
                    tl.atomic_add(
                        grad_v_ptrs,
                        grad_v,
                        mask=valid[:, None] & (d_offsets[None, :] < head_dim),
                        sem="relaxed",
                    )

        grad_q_ptrs = (
            grad_q_ptr
            + token * stride_dqt
            + heads[:, None] * stride_dqh
            + d_offsets[None, :] * stride_dqd
        )
        tl.store(
            grad_q_ptrs,
            grad_q,
            mask=head_valid[:, None] & (d_offsets[None, :] < head_dim),
        )

    @triton.jit
    def _qsa_selected_kv_backward_tail_grouped_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        index_ptr,
        length_ptr,
        lse_ptr,
        grad_out_ptr,
        grad_lse_ptr,
        correction_ptr,
        grad_k_ptr,
        grad_v_ptr,
        seq_len_q,
        seq_len_k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        key_position_offset,
        query_position_ptr,
        token_ids_ptr,
        stride_qp,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vd,
        stride_ib,
        stride_is,
        stride_ik,
        stride_lb,
        stride_ls,
        stride_lseb,
        stride_lseh,
        stride_lses,
        stride_gob,
        stride_gos,
        stride_goh,
        stride_god,
        stride_glseb,
        stride_glseh,
        stride_glses,
        stride_cb,
        stride_ch,
        stride_cs,
        stride_dkb,
        stride_dks,
        stride_dkh,
        stride_dkd,
        stride_dvb,
        stride_dvs,
        stride_dvh,
        stride_dvd,
        K: tl.constexpr,
        HEADS_PER_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        NUM_HEAD_TILES: tl.constexpr,
        BLOCK_TAIL: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SEGMENT_BLOCK_TOPK: tl.constexpr,
        RATIO: tl.constexpr,
        CAUSAL: tl.constexpr,
        HAS_GRAD_OUTPUT: tl.constexpr,
        HAS_GRAD_LSE: tl.constexpr,
        DKV_ACCUM_BF16: tl.constexpr,
        USE_TOKEN_IDS: tl.constexpr,
    ):
        """Accumulate only the causal tail left out of segmented dK/dV.

        Complete QSA blocks are reduced by the inverse-CSR kernel.  The
        at-most ``RATIO - 1`` tokens in the current causal block can overlap a
        complete block selected by a later query, so they remain additive
        atomics.  Keeping this work in a separate narrow kernel avoids
        computing full BLOCK_K dK/dV reductions in the main backward pass.
        """

        batch_query = tl.program_id(0)
        if USE_TOKEN_IDS:
            row = tl.load(token_ids_ptr + batch_query).to(tl.int32)
            batch = row // seq_len_q
            query = row - batch * seq_len_q
        else:
            batch = batch_query // seq_len_q
            query = batch_query - batch * seq_len_q
        head_tile_program = tl.program_id(1)
        kv_head = head_tile_program // NUM_HEAD_TILES
        head_tile = head_tile_program % NUM_HEAD_TILES
        head_offsets = tl.arange(0, HEADS_PER_KV)
        heads = kv_head * GROUP_SIZE + head_tile * HEADS_PER_KV + head_offsets
        head_valid = (
            (head_tile * HEADS_PER_KV + head_offsets < GROUP_SIZE)
            & (heads < num_q_heads)
        )
        d_offsets = tl.arange(0, BLOCK_D)
        q_ptrs = (
            q_ptr
            + batch * stride_qb
            + query * stride_qs
            + heads[:, None] * stride_qh
            + d_offsets[None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=head_valid[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        ).to(tl.bfloat16)
        grad_out_ptrs = (
            grad_out_ptr
            + batch * stride_gob
            + query * stride_gos
            + heads[:, None] * stride_goh
            + d_offsets[None, :] * stride_god
        )
        grad_output = tl.load(
            grad_out_ptrs,
            mask=head_valid[:, None]
            & (d_offsets[None, :] < head_dim)
            & HAS_GRAD_OUTPUT,
            other=0.0,
        ).to(tl.float32)
        query_position = tl.load(
            query_position_ptr + query * stride_qp
        ).to(tl.int32)
        length = tl.load(
            length_ptr + batch * stride_lb + query * stride_ls
        ).to(tl.int32)
        lse = tl.load(
            lse_ptr
            + batch * stride_lseb
            + heads * stride_lseh
            + query * stride_lses,
            mask=head_valid,
            other=0.0,
        ).to(tl.float32)
        correction = tl.load(
            correction_ptr
            + batch * stride_cb
            + heads * stride_ch
            + query * stride_cs,
            mask=head_valid,
            other=0.0,
        ).to(tl.float32)
        grad_lse = tl.load(
            grad_lse_ptr
            + batch * stride_glseb
            + heads * stride_glseh
            + query * stride_glses,
            mask=head_valid & HAS_GRAD_LSE,
            other=0.0,
        ).to(tl.float32)

        tail_start = tl.minimum(
            (query_position + 1) // RATIO, SEGMENT_BLOCK_TOPK
        ) * RATIO
        tail_offsets = tl.arange(0, BLOCK_TAIL)
        slots = tail_start + tail_offsets
        selected = tl.load(
            index_ptr
            + batch * stride_ib
            + query * stride_is
            + slots * stride_ik,
            mask=slots < K,
            other=-1,
        ).to(tl.int32)
        valid = (
            (slots < length)
            & (slots < K)
            & (selected >= 0)
            & (selected < seq_len_k)
        )
        if CAUSAL:
            valid = valid & (
                selected + key_position_offset <= query_position
            )
        safe_selected = tl.where(valid, selected, 0)
        k_ptrs = (
            k_ptr
            + batch * stride_kb
            + safe_selected[:, None] * stride_ks
            + kv_head * stride_kh
            + d_offsets[None, :] * stride_kd
        )
        v_ptrs = (
            v_ptr
            + batch * stride_vb
            + safe_selected[:, None] * stride_vs
            + kv_head * stride_vh
            + d_offsets[None, :] * stride_vd
        )
        k = tl.load(
            k_ptrs,
            mask=valid[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        ).to(tl.bfloat16)
        v = tl.load(
            v_ptrs,
            mask=valid[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        ).to(tl.bfloat16)
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * softmax_scale
        probabilities = tl.exp2(
            tl.where(
                valid[None, :],
                (scores - lse[:, None]) * 1.4426950408889634,
                -float("inf"),
            )
        )
        d_probability = tl.sum(
            v[None, :, :] * grad_output[:, None, :], axis=2
        )
        d_score = (
            probabilities * (d_probability - correction[:, None])
            + grad_lse[:, None] * probabilities
        )
        d_score = tl.where(valid[None, :], d_score, 0.0)
        grad_k = tl.sum(
            d_score[:, :, None] * q[:, None, :], axis=0
        ) * softmax_scale
        grad_v = tl.sum(
            probabilities[:, :, None] * grad_output[:, None, :], axis=0
        )
        grad_k_ptrs = (
            grad_k_ptr
            + batch * stride_dkb
            + safe_selected[:, None] * stride_dks
            + kv_head * stride_dkh
            + d_offsets[None, :] * stride_dkd
        )
        grad_v_ptrs = (
            grad_v_ptr
            + batch * stride_dvb
            + safe_selected[:, None] * stride_dvs
            + kv_head * stride_dvh
            + d_offsets[None, :] * stride_dvd
        )
        emit_mask = valid[:, None] & (d_offsets[None, :] < head_dim)
        if DKV_ACCUM_BF16:
            tl.atomic_add(
                grad_k_ptrs,
                grad_k.to(tl.bfloat16),
                mask=emit_mask,
                sem="relaxed",
            )
            tl.atomic_add(
                grad_v_ptrs,
                grad_v.to(tl.bfloat16),
                mask=emit_mask,
                sem="relaxed",
            )
        else:
            tl.atomic_add(
                grad_k_ptrs,
                grad_k,
                mask=emit_mask,
                sem="relaxed",
            )
            tl.atomic_add(
                grad_v_ptrs,
                grad_v,
                mask=emit_mask,
                sem="relaxed",
            )

    @triton.jit
    def _qsa_segmented_dkv_reduce_grouped_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        lse_ptr,
        grad_out_ptr,
        grad_lse_ptr,
        correction_ptr,
        occurrence_query_ptr,
        segment_start_ptr,
        grad_key_ptr,
        grad_value_ptr,
        seq_len_q,
        seq_len_k,
        num_blocks,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vd,
        stride_lseb,
        stride_lseh,
        stride_lses,
        stride_gob,
        stride_gos,
        stride_goh,
        stride_god,
        stride_glseb,
        stride_glseh,
        stride_glses,
        stride_cb,
        stride_ch,
        stride_cs,
        stride_dkb,
        stride_dks,
        stride_dkh,
        stride_dkd,
        stride_dvb,
        stride_dvs,
        stride_dvh,
        stride_dvd,
        RATIO: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_H: tl.constexpr,
        HEAD_TILE: tl.constexpr,
        NUM_HEAD_TILES: tl.constexpr,
        BLOCK_OCC: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_GRAD_OUTPUT: tl.constexpr,
        HAS_GRAD_LSE: tl.constexpr,
        DKV_ACCUM_BF16: tl.constexpr,
        BATCH_ONE: tl.constexpr,
        SPLIT_HEADS: tl.constexpr,
    ):
        """Matrix-tiled inverse-CSR reduction with one owner CTA per head tile.

        The previous reducer traversed ``token -> occurrence -> head`` and
        reloaded each occurrence's Q/grad-output once per token in the
        compression block.  This version traverses ``occurrence -> head``
        first, loads those tensors once, and computes all RATIO token
        contributions as a small ``BLOCK_OCC x RATIO`` score tile.  It keeps
        the same inverse-CSR ownership and additive tail semantics while
        making the complete-block path tensor-core friendly.

        """

        program = tl.program_id(0)
        head_work = program % (num_kv_heads * NUM_HEAD_TILES)
        kv_head = head_work // NUM_HEAD_TILES
        head_tile = head_work - kv_head * NUM_HEAD_TILES
        key_group = program // (num_kv_heads * NUM_HEAD_TILES)
        batch = key_group // num_blocks
        block = key_group - batch * num_blocks
        segment_start = tl.load(segment_start_ptr + key_group).to(tl.int32)
        segment_end = tl.load(segment_start_ptr + key_group + 1).to(tl.int32)
        if segment_start >= segment_end:
            return

        d_offsets = tl.arange(0, BLOCK_D)
        token_offsets = tl.arange(0, RATIO)
        key_positions = block * RATIO + token_offsets
        key_mask = key_positions < seq_len_k
        key_ptrs = (
            key_ptr
            + batch * stride_kb
            + key_positions[:, None] * stride_ks
            + kv_head * stride_kh
            + d_offsets[None, :] * stride_kd
        )
        key_value_raw = tl.load(
            key_ptrs,
            mask=key_mask[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        value_ptrs = (
            value_ptr
            + batch * stride_vb
            + key_positions[:, None] * stride_vs
            + kv_head * stride_vh
            + d_offsets[None, :] * stride_vd
        )
        value_value_raw = tl.load(
            value_ptrs,
            mask=key_mask[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        grad_key_acc = tl.zeros((RATIO, BLOCK_D), dtype=tl.float32)
        grad_value_acc = tl.zeros((RATIO, BLOCK_D), dtype=tl.float32)

        for occurrence_offset in tl.range(
            0, segment_end - segment_start, BLOCK_OCC
        ):
            occurrence = (
                segment_start
                + occurrence_offset
                + tl.arange(0, BLOCK_OCC)
            )
            row = tl.load(
                occurrence_query_ptr + occurrence,
                mask=occurrence < segment_end,
                other=0,
            ).to(tl.int32)
            if BATCH_ONE:
                query_batch = tl.zeros((BLOCK_OCC,), dtype=tl.int32)
                query = row
            else:
                query_batch = row // seq_len_q
                query = row - query_batch * seq_len_q
            if BATCH_ONE:
                same_batch = occurrence < segment_end
            else:
                same_batch = (query_batch == batch) & (occurrence < segment_end)
            for head_lane in tl.static_range(0, HEAD_TILE):
                head_offset = head_tile * HEAD_TILE + head_lane
                head = kv_head * GROUP_SIZE + head_offset
                head_valid = (
                    (head_offset < GROUP_SIZE) & (head < num_q_heads)
                )
                if BATCH_ONE:
                    q_ptrs = (
                        query_ptr
                        + query[:, None] * stride_qs
                        + head * stride_qh
                        + d_offsets[None, :] * stride_qd
                    )
                    grad_out_ptrs = (
                        grad_out_ptr
                        + query[:, None] * stride_gos
                        + head * stride_goh
                        + d_offsets[None, :] * stride_god
                    )
                    lse_ptrs = (
                        lse_ptr
                        + head * stride_lseh
                        + query * stride_lses
                    )
                    correction_ptrs = (
                        correction_ptr
                        + head * stride_ch
                        + query * stride_cs
                    )
                    grad_lse_ptrs = (
                        grad_lse_ptr
                        + head * stride_glseh
                        + query * stride_glses
                    )
                else:
                    q_ptrs = (
                        query_ptr
                        + query_batch[:, None] * stride_qb
                        + query[:, None] * stride_qs
                        + head * stride_qh
                        + d_offsets[None, :] * stride_qd
                    )
                    grad_out_ptrs = (
                        grad_out_ptr
                        + query_batch[:, None] * stride_gob
                        + query[:, None] * stride_gos
                        + head * stride_goh
                        + d_offsets[None, :] * stride_god
                    )
                    lse_ptrs = (
                        lse_ptr
                        + query_batch * stride_lseb
                        + head * stride_lseh
                        + query * stride_lses
                    )
                    correction_ptrs = (
                        correction_ptr
                        + query_batch * stride_cb
                        + head * stride_ch
                        + query * stride_cs
                    )
                    grad_lse_ptrs = (
                        grad_lse_ptr
                        + query_batch * stride_glseb
                        + head * stride_glseh
                        + query * stride_glses
                    )
                q_value_raw = tl.load(
                    q_ptrs,
                    mask=same_batch[:, None]
                    & head_valid
                    & (d_offsets[None, :] < head_dim),
                    other=0.0,
                )
                grad_output_raw = tl.load(
                    grad_out_ptrs,
                    mask=same_batch[:, None]
                    & head_valid
                    & (d_offsets[None, :] < head_dim)
                    & HAS_GRAD_OUTPUT,
                    other=0.0,
                )
                lse = tl.load(
                    lse_ptrs,
                    mask=same_batch & head_valid,
                    other=0.0,
                ).to(tl.float32)
                score = tl.dot(
                    q_value_raw,
                    tl.trans(key_value_raw),
                    out_dtype=tl.float32,
                ) * softmax_scale
                probability = tl.exp2(
                    tl.where(
                        same_batch[:, None] & head_valid & key_mask[None, :],
                        (score - lse[:, None]) * 1.4426950408889634,
                        -float("inf"),
                    )
                )
                correction = tl.load(
                    correction_ptrs,
                    mask=same_batch & head_valid,
                    other=0.0,
                ).to(tl.float32)
                grad_lse = tl.load(
                    grad_lse_ptrs,
                    mask=same_batch & head_valid & HAS_GRAD_LSE,
                    other=0.0,
                ).to(tl.float32)
                d_probability = tl.dot(
                    grad_output_raw,
                    tl.trans(value_value_raw),
                    out_dtype=tl.float32,
                )
                grad_score = (
                    probability * (d_probability - correction[:, None])
                    + grad_lse[:, None] * probability
                )
                grad_score = tl.where(
                    same_batch[:, None]
                    & head_valid
                    & key_mask[None, :],
                    grad_score,
                    0.0,
                )
                if DKV_ACCUM_BF16:
                    grad_key_acc += tl.dot(
                        tl.trans(grad_score.to(tl.bfloat16)),
                        q_value_raw,
                        out_dtype=tl.float32,
                    ) * softmax_scale
                    grad_value_acc += tl.dot(
                        tl.trans(probability.to(tl.bfloat16)),
                        grad_output_raw,
                        out_dtype=tl.float32,
                    )
                else:
                    q_value = q_value_raw.to(tl.float32)
                    grad_output = grad_output_raw.to(tl.float32)
                    grad_key_acc += tl.sum(
                        tl.trans(grad_score)[:, :, None]
                        * q_value[None, :, :],
                        axis=1,
                    ) * softmax_scale
                    grad_value_acc += tl.sum(
                        tl.trans(probability)[:, :, None]
                        * grad_output[None, :, :],
                        axis=1,
                    )

        key_output_ptrs = (
            grad_key_ptr
            + batch * stride_dkb
            + key_positions[:, None] * stride_dks
            + kv_head * stride_dkh
            + d_offsets[None, :] * stride_dkd
        )
        value_output_ptrs = (
            grad_value_ptr
            + batch * stride_dvb
            + key_positions[:, None] * stride_dvs
            + kv_head * stride_dvh
            + d_offsets[None, :] * stride_dvd
        )
        output_mask = key_mask[:, None] & (d_offsets[None, :] < head_dim)
        if SPLIT_HEADS:
            if DKV_ACCUM_BF16:
                tl.atomic_add(
                    key_output_ptrs,
                    grad_key_acc.to(tl.bfloat16),
                    mask=output_mask,
                    sem="relaxed",
                )
                tl.atomic_add(
                    value_output_ptrs,
                    grad_value_acc.to(tl.bfloat16),
                    mask=output_mask,
                    sem="relaxed",
                )
            else:
                tl.atomic_add(
                    key_output_ptrs,
                    grad_key_acc,
                    mask=output_mask,
                    sem="relaxed",
                )
                tl.atomic_add(
                    value_output_ptrs,
                    grad_value_acc,
                    mask=output_mask,
                    sem="relaxed",
                )
        else:
            # The tail atomic launch runs before this kernel. Add its
            # contribution before storing so a tail token shared with a later
            # complete block is not overwritten by the segmented result.
            existing_key = tl.load(
                key_output_ptrs, mask=output_mask, other=0.0
            ).to(tl.float32)
            existing_value = tl.load(
                value_output_ptrs, mask=output_mask, other=0.0
            ).to(tl.float32)
            grad_key_acc += existing_key
            grad_value_acc += existing_value
            if DKV_ACCUM_BF16:
                tl.store(
                    key_output_ptrs,
                    grad_key_acc.to(tl.bfloat16),
                    mask=output_mask,
                )
                tl.store(
                    value_output_ptrs,
                    grad_value_acc.to(tl.bfloat16),
                    mask=output_mask,
                )
            else:
                tl.store(key_output_ptrs, grad_key_acc, mask=output_mask)
                tl.store(value_output_ptrs, grad_value_acc, mask=output_mask)

    @triton.jit
    def _qsa_segmented_dkv_reduce_flattened_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        lse_ptr,
        grad_out_ptr,
        grad_lse_ptr,
        correction_ptr,
        occurrence_query_ptr,
        segment_start_ptr,
        grad_key_ptr,
        grad_value_ptr,
        seq_len_q,
        seq_len_k,
        num_blocks,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vd,
        stride_lseb,
        stride_lseh,
        stride_lses,
        stride_gob,
        stride_gos,
        stride_goh,
        stride_god,
        stride_glseb,
        stride_glseh,
        stride_glses,
        stride_cb,
        stride_ch,
        stride_cs,
        stride_dkb,
        stride_dks,
        stride_dkh,
        stride_dkd,
        stride_dvb,
        stride_dvs,
        stride_dvh,
        stride_dvd,
        RATIO: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_H: tl.constexpr,
        HEAD_TILE: tl.constexpr,
        NUM_HEAD_TILES: tl.constexpr,
        BLOCK_OCC: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_GRAD_OUTPUT: tl.constexpr,
        HAS_GRAD_LSE: tl.constexpr,
        DKV_ACCUM_BF16: tl.constexpr,
        BATCH_ONE: tl.constexpr,
        SPLIT_HEADS: tl.constexpr,
    ):
        """Reduce a block with occurrence and GQA-head rows fused into one MMA tile.

        ``RATIO`` is only four for the production route, so treating each head
        as an independent ``BLOCK_OCC x RATIO`` matrix leaves most of a Hopper
        MMA instruction empty.  Flattening ``occurrence x head`` gives QK/dP a
        wide M dimension and gives dK/dV a wide reduction dimension.  It also
        makes adjacent GQA heads of one query contiguous in the gather stream.
        """

        program = tl.program_id(0)
        head_work = program % (num_kv_heads * NUM_HEAD_TILES)
        kv_head = head_work // NUM_HEAD_TILES
        head_tile = head_work - kv_head * NUM_HEAD_TILES
        key_group = program // (num_kv_heads * NUM_HEAD_TILES)
        batch = key_group // num_blocks
        block = key_group - batch * num_blocks
        segment_start = tl.load(segment_start_ptr + key_group).to(tl.int32)
        segment_end = tl.load(segment_start_ptr + key_group + 1).to(tl.int32)
        if segment_start >= segment_end:
            return

        d_offsets = tl.arange(0, BLOCK_D)
        token_offsets = tl.arange(0, RATIO)
        key_positions = block * RATIO + token_offsets
        key_mask = key_positions < seq_len_k
        key_ptrs = (
            key_ptr
            + batch * stride_kb
            + key_positions[:, None] * stride_ks
            + kv_head * stride_kh
            + d_offsets[None, :] * stride_kd
        )
        value_ptrs = (
            value_ptr
            + batch * stride_vb
            + key_positions[:, None] * stride_vs
            + kv_head * stride_vh
            + d_offsets[None, :] * stride_vd
        )
        key_value_raw = tl.load(
            key_ptrs,
            mask=key_mask[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        value_value_raw = tl.load(
            value_ptrs,
            mask=key_mask[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        grad_key_acc = tl.zeros((RATIO, BLOCK_D), dtype=tl.float32)
        grad_value_acc = tl.zeros((RATIO, BLOCK_D), dtype=tl.float32)
        flat_offsets = tl.arange(0, BLOCK_M)
        occurrence_lanes = flat_offsets // HEAD_TILE
        head_lanes = flat_offsets - occurrence_lanes * HEAD_TILE
        head_offsets = head_tile * HEAD_TILE + head_lanes
        heads = kv_head * GROUP_SIZE + head_offsets
        head_valid = (head_offsets < GROUP_SIZE) & (heads < num_q_heads)

        for occurrence_offset in tl.range(
            0, segment_end - segment_start, BLOCK_OCC
        ):
            occurrence = segment_start + occurrence_offset + occurrence_lanes
            row = tl.load(
                occurrence_query_ptr + occurrence,
                mask=occurrence < segment_end,
                other=0,
            ).to(tl.int32)
            if BATCH_ONE:
                query_batch = tl.zeros((BLOCK_M,), dtype=tl.int32)
                query = row
                same_batch = occurrence < segment_end
            else:
                query_batch = row // seq_len_q
                query = row - query_batch * seq_len_q
                same_batch = (
                    (query_batch == batch) & (occurrence < segment_end)
                )
            row_valid = same_batch & head_valid
            q_ptrs = (
                query_ptr
                + query_batch[:, None] * stride_qb
                + query[:, None] * stride_qs
                + heads[:, None] * stride_qh
                + d_offsets[None, :] * stride_qd
            )
            grad_out_ptrs = (
                grad_out_ptr
                + query_batch[:, None] * stride_gob
                + query[:, None] * stride_gos
                + heads[:, None] * stride_goh
                + d_offsets[None, :] * stride_god
            )
            q_value_raw = tl.load(
                q_ptrs,
                mask=row_valid[:, None]
                & (d_offsets[None, :] < head_dim),
                other=0.0,
            )
            grad_output_raw = tl.load(
                grad_out_ptrs,
                mask=row_valid[:, None]
                & (d_offsets[None, :] < head_dim)
                & HAS_GRAD_OUTPUT,
                other=0.0,
            )
            lse = tl.load(
                lse_ptr
                + query_batch * stride_lseb
                + heads * stride_lseh
                + query * stride_lses,
                mask=row_valid,
                other=0.0,
            ).to(tl.float32)
            correction = tl.load(
                correction_ptr
                + query_batch * stride_cb
                + heads * stride_ch
                + query * stride_cs,
                mask=row_valid,
                other=0.0,
            ).to(tl.float32)
            grad_lse = tl.load(
                grad_lse_ptr
                + query_batch * stride_glseb
                + heads * stride_glseh
                + query * stride_glses,
                mask=row_valid & HAS_GRAD_LSE,
                other=0.0,
            ).to(tl.float32)
            score = tl.dot(
                q_value_raw,
                tl.trans(key_value_raw),
                out_dtype=tl.float32,
            ) * softmax_scale
            probability = tl.exp2(
                tl.where(
                    row_valid[:, None] & key_mask[None, :],
                    (score - lse[:, None]) * 1.4426950408889634,
                    -float("inf"),
                )
            )
            d_probability = tl.dot(
                grad_output_raw,
                tl.trans(value_value_raw),
                out_dtype=tl.float32,
            )
            grad_score = (
                probability * (d_probability - correction[:, None])
                + grad_lse[:, None] * probability
            )
            grad_score = tl.where(
                row_valid[:, None] & key_mask[None, :],
                grad_score,
                0.0,
            )
            if DKV_ACCUM_BF16:
                grad_key_acc += tl.dot(
                    tl.trans(grad_score.to(tl.bfloat16)),
                    q_value_raw,
                    out_dtype=tl.float32,
                ) * softmax_scale
                grad_value_acc += tl.dot(
                    tl.trans(probability.to(tl.bfloat16)),
                    grad_output_raw,
                    out_dtype=tl.float32,
                )
            else:
                q_value = q_value_raw.to(tl.float32)
                grad_output = grad_output_raw.to(tl.float32)
                grad_key_acc += tl.sum(
                    tl.trans(grad_score)[:, :, None]
                    * q_value[None, :, :],
                    axis=1,
                ) * softmax_scale
                grad_value_acc += tl.sum(
                    tl.trans(probability)[:, :, None]
                    * grad_output[None, :, :],
                    axis=1,
                )

        key_output_ptrs = (
            grad_key_ptr
            + batch * stride_dkb
            + key_positions[:, None] * stride_dks
            + kv_head * stride_dkh
            + d_offsets[None, :] * stride_dkd
        )
        value_output_ptrs = (
            grad_value_ptr
            + batch * stride_dvb
            + key_positions[:, None] * stride_dvs
            + kv_head * stride_dvh
            + d_offsets[None, :] * stride_dvd
        )
        output_mask = key_mask[:, None] & (d_offsets[None, :] < head_dim)
        if SPLIT_HEADS:
            if DKV_ACCUM_BF16:
                tl.atomic_add(
                    key_output_ptrs,
                    grad_key_acc.to(tl.bfloat16),
                    mask=output_mask,
                    sem="relaxed",
                )
                tl.atomic_add(
                    value_output_ptrs,
                    grad_value_acc.to(tl.bfloat16),
                    mask=output_mask,
                    sem="relaxed",
                )
            else:
                tl.atomic_add(
                    key_output_ptrs,
                    grad_key_acc,
                    mask=output_mask,
                    sem="relaxed",
                )
                tl.atomic_add(
                    value_output_ptrs,
                    grad_value_acc,
                    mask=output_mask,
                    sem="relaxed",
                )
        else:
            # Preserve the causal-tail contribution written before this launch.
            existing_key = tl.load(
                key_output_ptrs, mask=output_mask, other=0.0
            ).to(tl.float32)
            existing_value = tl.load(
                value_output_ptrs, mask=output_mask, other=0.0
            ).to(tl.float32)
            grad_key_acc += existing_key
            grad_value_acc += existing_value
            if DKV_ACCUM_BF16:
                tl.store(
                    key_output_ptrs,
                    grad_key_acc.to(tl.bfloat16),
                    mask=output_mask,
                )
                tl.store(
                    value_output_ptrs,
                    grad_value_acc.to(tl.bfloat16),
                    mask=output_mask,
                )
            else:
                tl.store(key_output_ptrs, grad_key_acc, mask=output_mask)
                tl.store(value_output_ptrs, grad_value_acc, mask=output_mask)

    @triton.jit
    def _qsa_segment_count_blocks_kernel(
        index_ptr,
        length_ptr,
        query_position_ptr,
        counts_ptr,
        seq_len,
        seq_len_k,
        num_blocks,
        stride_ib,
        stride_is,
        stride_ik,
        stride_lb,
        stride_ls,
        BLOCK_TOPK: tl.constexpr,
        RATIO: tl.constexpr,
    ):
        """Count complete selected QSA blocks per ``(batch, key block)``."""

        row = tl.program_id(0)
        batch = row // seq_len
        query = row - batch * seq_len
        query_position = tl.load(query_position_ptr + query).to(tl.int32)
        complete_blocks = tl.minimum(
            num_blocks, (query_position + 1) // RATIO
        ).to(tl.int32)
        block_count = tl.minimum(complete_blocks, BLOCK_TOPK).to(tl.int32)
        slots = tl.arange(0, BLOCK_TOPK)
        token_starts = slots * RATIO
        first = tl.load(
            index_ptr
            + batch * stride_ib
            + query * stride_is
            + token_starts * stride_ik,
        ).to(tl.int32)
        length = tl.load(length_ptr + batch * stride_lb + query * stride_ls).to(tl.int32)
        block_ids = first // RATIO
        valid = (
            (slots < block_count)
            & (token_starts + RATIO <= length)
            & (first >= 0)
            & (first < seq_len_k)
            & (block_ids >= 0)
            & (block_ids < num_blocks)
        )
        for token_offset in tl.static_range(0, RATIO):
            token = tl.load(
                index_ptr
                + batch * stride_ib
                + query * stride_is
                + (token_starts + token_offset) * stride_ik,
            ).to(tl.int32)
            valid = valid & (token == block_ids * RATIO + token_offset)
        keys = batch * num_blocks + block_ids
        tl.atomic_add(counts_ptr + keys, 1, mask=valid)

    @triton.jit
    def _qsa_segment_fill_blocks_kernel(
        index_ptr,
        length_ptr,
        query_position_ptr,
        cursor_ptr,
        occurrence_query_ptr,
        seq_len,
        seq_len_k,
        num_blocks,
        stride_ib,
        stride_is,
        stride_ik,
        stride_lb,
        stride_ls,
        BLOCK_TOPK: tl.constexpr,
        RATIO: tl.constexpr,
    ):
        """Fill inverse block CSR lists with flattened query-row owners."""

        row = tl.program_id(0)
        batch = row // seq_len
        query = row - batch * seq_len
        query_position = tl.load(query_position_ptr + query).to(tl.int32)
        complete_blocks = tl.minimum(
            num_blocks, (query_position + 1) // RATIO
        ).to(tl.int32)
        block_count = tl.minimum(complete_blocks, BLOCK_TOPK).to(tl.int32)
        slots = tl.arange(0, BLOCK_TOPK)
        token_starts = slots * RATIO
        first = tl.load(
            index_ptr
            + batch * stride_ib
            + query * stride_is
            + token_starts * stride_ik,
        ).to(tl.int32)
        length = tl.load(length_ptr + batch * stride_lb + query * stride_ls).to(tl.int32)
        block_ids = first // RATIO
        valid = (
            (slots < block_count)
            & (token_starts + RATIO <= length)
            & (first >= 0)
            & (first < seq_len_k)
            & (block_ids >= 0)
            & (block_ids < num_blocks)
        )
        for token_offset in tl.static_range(0, RATIO):
            token = tl.load(
                index_ptr
                + batch * stride_ib
                + query * stride_is
                + (token_starts + token_offset) * stride_ik,
            ).to(tl.int32)
            valid = valid & (token == block_ids * RATIO + token_offset)
        keys = batch * num_blocks + block_ids
        destinations = tl.atomic_add(cursor_ptr + keys, 1, mask=valid)
        tl.store(
            occurrence_query_ptr + destinations,
            row,
            mask=valid,
        )

    @triton.jit
    def _qsa_segmented_dkv_reduce_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        lse_ptr,
        grad_out_ptr,
        grad_lse_ptr,
        correction_ptr,
        occurrence_query_ptr,
        segment_start_ptr,
        grad_key_ptr,
        grad_value_ptr,
        seq_len_q,
        seq_len_k,
        num_blocks,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vd,
        stride_lseb,
        stride_lseh,
        stride_lses,
        stride_gob,
        stride_gos,
        stride_goh,
        stride_god,
        stride_glseb,
        stride_glseh,
        stride_glses,
        stride_cb,
        stride_ch,
        stride_cs,
        stride_dkb,
        stride_dks,
        stride_dkh,
        stride_dkd,
        stride_dvb,
        stride_dvs,
        stride_dvh,
        stride_dvd,
        RATIO: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_OCC: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_GRAD_OUTPUT: tl.constexpr,
        HAS_GRAD_LSE: tl.constexpr,
        DKV_ACCUM_BF16: tl.constexpr,
    ):
        """Reduce every selected block segment without dK/dV atomics."""

        program = tl.program_id(0)
        kv_head = program % num_kv_heads
        key_group = program // num_kv_heads
        batch = key_group // num_blocks
        block = key_group - batch * num_blocks
        segment_start = tl.load(segment_start_ptr + key_group).to(tl.int32)
        segment_end = tl.load(segment_start_ptr + key_group + 1).to(tl.int32)
        if segment_start >= segment_end:
            return

        d_offsets = tl.arange(0, BLOCK_D)
        for token_offset in tl.static_range(0, RATIO):
            key_position = block * RATIO + token_offset
            key_mask = key_position < seq_len_k
            key_ptrs = (
                key_ptr
                + batch * stride_kb
                + key_position * stride_ks
                + kv_head * stride_kh
                + d_offsets * stride_kd
            )
            value_ptrs = (
                value_ptr
                + batch * stride_vb
                + key_position * stride_vs
                + kv_head * stride_vh
                + d_offsets * stride_vd
            )
            key_value_raw = tl.load(
                key_ptrs,
                mask=key_mask & (d_offsets < head_dim),
                other=0.0,
            )
            value_value_raw = tl.load(
                value_ptrs,
                mask=key_mask & (d_offsets < head_dim),
                other=0.0,
            )
            key_value = key_value_raw.to(tl.float32)
            value_value = value_value_raw.to(tl.float32)
            grad_key_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
            grad_value_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for occurrence_offset in tl.range(
                0, segment_end - segment_start, BLOCK_OCC
            ):
                occurrence = segment_start + occurrence_offset + tl.arange(0, BLOCK_OCC)
                row = tl.load(
                    occurrence_query_ptr + occurrence,
                    mask=occurrence < segment_end,
                    other=0,
                ).to(tl.int32)
                query_batch = row // seq_len_q
                query = row - query_batch * seq_len_q
                # ``query_batch`` must equal the segment owner.  The check is
                # kept in the mask rather than a host-side validation so the
                # inverse map remains usable with arbitrary batch sizes.
                same_batch = (query_batch == batch) & (occurrence < segment_end)
                for head_offset in tl.static_range(0, GROUP_SIZE):
                    head = kv_head * GROUP_SIZE + head_offset
                    q_ptrs = (
                        query_ptr
                        + query_batch[:, None] * stride_qb
                        + query[:, None] * stride_qs
                        + head * stride_qh
                        + d_offsets[None, :] * stride_qd
                    )
                    grad_out_ptrs = (
                        grad_out_ptr
                        + query_batch[:, None] * stride_gob
                        + query[:, None] * stride_gos
                        + head * stride_goh
                        + d_offsets[None, :] * stride_god
                    )
                    q_value_raw = tl.load(
                        q_ptrs,
                        mask=same_batch[:, None] & (head < num_q_heads) &
                        (d_offsets[None, :] < head_dim),
                        other=0.0,
                    )
                    grad_output_raw = tl.load(
                        grad_out_ptrs,
                        mask=same_batch[:, None] & (head < num_q_heads) &
                        (d_offsets[None, :] < head_dim) & HAS_GRAD_OUTPUT,
                        other=0.0,
                    )
                    q_value = q_value_raw.to(tl.float32)
                    grad_output = grad_output_raw.to(tl.float32)
                    score = tl.reshape(
                        tl.dot(q_value_raw, key_value_raw[:, None], out_dtype=tl.float32),
                        (BLOCK_OCC,),
                    ) * softmax_scale
                    lse = tl.load(
                        lse_ptr
                        + query_batch * stride_lseb
                        + head * stride_lseh
                        + query * stride_lses,
                        mask=same_batch,
                        other=0.0,
                    ).to(tl.float32)
                    probability = tl.exp2((score - lse) * 1.4426950408889634)
                    d_probability = tl.reshape(
                        tl.dot(grad_output_raw, value_value_raw[:, None], out_dtype=tl.float32),
                        (BLOCK_OCC,),
                    )
                    correction = tl.load(
                        correction_ptr
                        + query_batch * stride_cb
                        + head * stride_ch
                        + query * stride_cs,
                        mask=same_batch,
                        other=0.0,
                    ).to(tl.float32)
                    grad_lse = tl.load(
                        grad_lse_ptr
                        + query_batch * stride_glseb
                        + head * stride_glseh
                        + query * stride_glses,
                        mask=same_batch & HAS_GRAD_LSE,
                        other=0.0,
                    ).to(tl.float32)
                    grad_score = probability * (d_probability - correction) + grad_lse * probability
                    grad_score = tl.where(same_batch & (head < num_q_heads) & key_mask, grad_score, 0.0)
                    if DKV_ACCUM_BF16:
                        grad_key_acc += tl.reshape(
                            tl.dot(
                                grad_score.to(tl.bfloat16)[None, :],
                                q_value_raw,
                                out_dtype=tl.float32,
                            ),
                            (BLOCK_D,),
                        ) * softmax_scale
                        grad_value_acc += tl.reshape(
                            tl.dot(
                                probability.to(tl.bfloat16)[None, :],
                                grad_output_raw,
                                out_dtype=tl.float32,
                            ),
                            (BLOCK_D,),
                        )
                    else:
                        grad_key_acc += tl.sum(
                            grad_score[:, None] * q_value, axis=0
                        ) * softmax_scale
                        grad_value_acc += tl.sum(
                            probability[:, None] * grad_output, axis=0
                        )

            key_output_ptrs = (
                grad_key_ptr
                + batch * stride_dkb
                + key_position * stride_dks
                + kv_head * stride_dkh
                + d_offsets * stride_dkd
            )
            value_output_ptrs = (
                grad_value_ptr
                + batch * stride_dvb
                + key_position * stride_dvs
                + kv_head * stride_dvh
                + d_offsets * stride_dvd
            )
            output_mask = key_mask & (d_offsets < head_dim)
            # The causal tail is deliberately accumulated by the small atomic
            # path.  Early-row tail tokens can share physical positions with
            # a complete block selected by later rows, so segmented output
            # must add to (rather than overwrite) those already-present terms.
            existing_key = tl.load(key_output_ptrs, mask=output_mask, other=0.0).to(tl.float32)
            existing_value = tl.load(value_output_ptrs, mask=output_mask, other=0.0).to(tl.float32)
            grad_key_acc += existing_key
            grad_value_acc += existing_value
            if DKV_ACCUM_BF16:
                tl.store(key_output_ptrs, grad_key_acc.to(tl.bfloat16), mask=output_mask)
                tl.store(value_output_ptrs, grad_value_acc.to(tl.bfloat16), mask=output_mask)
            else:
                tl.store(key_output_ptrs, grad_key_acc, mask=output_mask)
                tl.store(value_output_ptrs, grad_value_acc, mask=output_mask)


def qsa_indexer_fused_postprocess(
    qk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    num_heads: int,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse production indexer normalization, RoPE, and key pooling."""

    if not TRITON_AVAILABLE:
        raise RuntimeError('QSA fused indexer postprocess requires Triton')
    if not qk.is_cuda or qk.dtype != torch.bfloat16:
        raise ValueError('QSA fused indexer postprocess requires CUDA BF16 projection output')
    if qk.ndim != 3:
        raise ValueError('QSA fused indexer postprocess expects qk=[S,B,(H+1)D]')
    seq_len, batch, projected_dim = qk.shape
    num_heads = int(num_heads)
    ratio = int(ratio)
    if num_heads <= 0 or ratio <= 1 or projected_dim % (num_heads + 1):
        raise ValueError('QSA fused indexer postprocess received incompatible head/ratio shape')
    head_dim = projected_dim // (num_heads + 1)
    if head_dim != 128 or num_heads != 4 or ratio != 4:
        raise ValueError('QSA fused indexer postprocess currently specializes H=4,D=128,R=4')
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError('QSA fused indexer postprocess norm weights must match head_dim')
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != seq_len:
        raise ValueError('QSA fused indexer postprocess expects cos/sin=[S,rotary_dim]')
    rotary_dim = cos.shape[1]
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError('QSA fused indexer postprocess requires an even rotary_dim <= head_dim')
    if any(t.device != qk.device or t.dtype != qk.dtype for t in (q_weight, k_weight, cos, sin)):
        raise ValueError('QSA fused indexer postprocess inputs must share CUDA device and BF16 dtype')

    qk = qk.contiguous()
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    num_blocks = seq_len // ratio
    q = torch.empty(
        (batch, seq_len, num_heads, head_dim),
        device=qk.device,
        dtype=qk.dtype,
    )
    block_keys = torch.empty(
        (batch, num_blocks, head_dim),
        device=qk.device,
        dtype=qk.dtype,
    )
    # Keep four warps fixed: it matches the torch FP32 reduction tree before
    # the BF16 cast.  Other warp counts are slightly faster at long sequence
    # lengths but change BF16 least-significant bits and therefore are not a
    # valid production tuning dimension.
    _qsa_indexer_fused_postprocess_kernel[(batch * seq_len,)](
        qk,
        q_weight,
        k_weight,
        cos,
        sin,
        q,
        block_keys,
        seq_len,
        num_blocks,
        head_dim,
        rotary_dim,
        float(norm_epsilon),
        qk.stride(0),
        qk.stride(1),
        qk.stride(2),
        q_weight.stride(0),
        k_weight.stride(0),
        cos.stride(0),
        cos.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        block_keys.stride(0),
        block_keys.stride(1),
        block_keys.stride(2),
        NUM_HEADS=num_heads,
        RATIO=ratio,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return q, block_keys


def qsa_expand_compact_route_triton(
    block_indices: torch.Tensor,
    topk_length: torch.Tensor,
    query_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Expand compact blocks with one bandwidth-only Triton launch."""

    if not TRITON_AVAILABLE:
        raise RuntimeError('QSA compact route expansion requires Triton')
    if not block_indices.is_cuda:
        raise ValueError('QSA Triton compact route expansion requires CUDA')
    if block_indices.ndim != 3:
        raise ValueError(
            'QSA compact route expansion expects block_indices=[B,S,Kb]')
    batch, seq_len, route_slots = block_indices.shape
    if route_slots <= 0:
        raise ValueError('QSA compact route expansion requires route slots')
    if topk_length.shape != (batch, seq_len):
        raise ValueError('QSA compact route expansion route/length shape mismatch')
    block_size = int(block_size)
    if block_size <= 1:
        raise ValueError('QSA compact route expansion requires block_size > 1')
    query_positions = query_positions.to(
        device=block_indices.device, dtype=torch.int32).reshape(-1).contiguous()
    if query_positions.shape != (seq_len,):
        raise ValueError(
            f'QSA compact route query_positions must have shape [{seq_len}]')
    block_indices = block_indices.to(dtype=torch.int32).contiguous()
    topk_length = topk_length.to(
        device=block_indices.device, dtype=torch.int32).contiguous()
    token_slots = route_slots * block_size + block_size - 1
    output = torch.empty(
        (batch, seq_len, token_slots),
        device=block_indices.device,
        dtype=torch.int32,
    )
    _qsa_expand_compact_route_kernel[(batch * seq_len,)](
        block_indices,
        topk_length,
        output,
        query_positions,
        seq_len,
        block_indices.stride(0),
        block_indices.stride(1),
        topk_length.stride(0),
        topk_length.stride(1),
        output.stride(0),
        output.stride(1),
        ROUTE_SLOTS=route_slots,
        BLOCK_ROUTE=triton.next_power_of_2(route_slots),
        BLOCK_TAIL=max(1, triton.next_power_of_2(block_size - 1)),
        RATIO=block_size,
        num_warps=1,
        num_stages=1,
    )
    return output


def _qsa_cp_partition_args(
    global_seq_len: int,
    cp_size: int,
    rank: int,
    partition_mode: str,
):
    global_seq_len = int(global_seq_len)
    cp_size = int(cp_size)
    rank = int(rank)
    if cp_size <= 1 or not 0 <= rank < cp_size:
        raise ValueError(
            f'QSA CP planner expects cp_size>1 and rank in range, got '
            f'cp_size={cp_size}, rank={rank}')
    if global_seq_len <= 0 or global_seq_len % cp_size:
        raise ValueError(
            'QSA CP planner requires global sequence divisible by CP size')
    local_key_len = global_seq_len // cp_size
    if partition_mode == 'contiguous':
        return global_seq_len, cp_size, rank, local_key_len, local_key_len, False
    if partition_mode != 'zigzag':
        raise ValueError(f'unsupported QSA CP partition mode: {partition_mode!r}')
    if global_seq_len % (2 * cp_size):
        raise ValueError(
            'QSA zigzag CP requires global sequence divisible by 2*cp_size')
    return (
        global_seq_len,
        cp_size,
        rank,
        local_key_len,
        global_seq_len // (2 * cp_size),
        True,
    )


def qsa_cp_request_mask_triton(
    token_indices: torch.Tensor,
    topk_length: torch.Tensor,
    global_seq_len: int,
    cp_size: int,
    rank: int,
    partition_mode: str,
) -> torch.Tensor:
    """Build a byte owner-request mask without full-width mapping tensors."""

    if not TRITON_AVAILABLE or not token_indices.is_cuda:
        raise RuntimeError('QSA CP Triton planner requires CUDA and Triton')
    if token_indices.ndim != 3 or topk_length.shape != token_indices.shape[:2]:
        raise ValueError('QSA CP Triton planner route/length shape mismatch')
    if token_indices.dtype != torch.int32:
        raise ValueError('QSA CP Triton planner requires int32 token indices')
    token_indices = token_indices.contiguous()
    topk_length = topk_length.to(
        device=token_indices.device, dtype=torch.int32).contiguous()
    (global_seq_len, cp_size, rank, local_key_len,
     chunk_len, zigzag) = _qsa_cp_partition_args(
         global_seq_len, cp_size, rank, partition_mode)
    request_mask_i32 = torch.zeros(
        (cp_size, local_key_len),
        device=token_indices.device,
        dtype=torch.int32,
    )
    block = 256
    route_numel = token_indices.numel()
    _qsa_cp_request_mask_kernel[(triton.cdiv(route_numel, block),)](
        token_indices,
        topk_length,
        request_mask_i32,
        route_numel,
        global_seq_len,
        local_key_len,
        chunk_len,
        rank,
        ROUTE_SLOTS=token_indices.shape[-1],
        CP_SIZE=cp_size,
        ZIGZAG=zigzag,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )
    return request_mask_i32.to(torch.uint8)


def qsa_cp_compact_request_mask_triton(
    block_indices: torch.Tensor,
    topk_length: torch.Tensor,
    query_positions: torch.Tensor,
    block_size: int,
    global_seq_len: int,
    cp_size: int,
    rank: int,
    partition_mode: str,
) -> torch.Tensor:
    """Build owner requests directly from compact blocks and causal tails."""

    if not TRITON_AVAILABLE or not block_indices.is_cuda:
        raise RuntimeError('QSA compact CP planner requires CUDA and Triton')
    if block_indices.ndim != 3 or topk_length.shape != block_indices.shape[:2]:
        raise ValueError('QSA compact CP planner route/length shape mismatch')
    if block_indices.dtype != torch.int32:
        raise ValueError('QSA compact CP planner requires int32 block indices')
    batch, seq_len, route_slots = block_indices.shape
    if batch <= 0 or seq_len <= 0 or route_slots <= 0:
        raise ValueError('QSA compact CP planner requires non-empty routes')
    block_size = int(block_size)
    if block_size <= 1:
        raise ValueError('QSA compact CP planner requires block_size > 1')
    block_indices = block_indices.contiguous()
    topk_length = topk_length.to(
        device=block_indices.device, dtype=torch.int32).contiguous()
    query_positions = query_positions.to(
        device=block_indices.device, dtype=torch.int32).reshape(-1).contiguous()
    if query_positions.shape != (seq_len,):
        raise ValueError(
            f'QSA compact CP planner query_positions must have shape '
            f'[{seq_len}]')
    (global_seq_len, cp_size, rank, local_key_len,
     chunk_len, zigzag) = _qsa_cp_partition_args(
         global_seq_len, cp_size, rank, partition_mode)
    request_mask_i32 = torch.zeros(
        (cp_size, local_key_len),
        device=block_indices.device,
        dtype=torch.int32,
    )
    block = 128
    route_numel = block_indices.numel()
    _qsa_cp_compact_request_mask_kernel[(
        triton.cdiv(route_numel, block),
    )](
        block_indices,
        topk_length,
        query_positions,
        request_mask_i32,
        route_numel,
        seq_len,
        global_seq_len,
        local_key_len,
        chunk_len,
        rank,
        ROUTE_SLOTS=route_slots,
        CP_SIZE=cp_size,
        RATIO=block_size,
        ZIGZAG=zigzag,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )
    return request_mask_i32.to(torch.uint8)


def qsa_cp_remap_route_triton(
    token_indices: torch.Tensor,
    topk_length: torch.Tensor,
    cache_offsets: torch.Tensor,
    global_seq_len: int,
    cp_size: int,
    rank: int,
    partition_mode: str,
    in_place: bool = False,
) -> torch.Tensor:
    """Map global token IDs to final owner-cache offsets in one launch."""

    if not TRITON_AVAILABLE or not token_indices.is_cuda:
        raise RuntimeError('QSA CP Triton remap requires CUDA and Triton')
    if token_indices.ndim != 3 or topk_length.shape != token_indices.shape[:2]:
        raise ValueError('QSA CP Triton remap route/length shape mismatch')
    if token_indices.dtype != torch.int32:
        raise ValueError('QSA CP Triton remap requires int32 token indices')
    if in_place and not token_indices.is_contiguous():
        raise ValueError('QSA CP in-place remap requires a contiguous route')
    token_indices = token_indices.contiguous()
    topk_length = topk_length.to(
        device=token_indices.device, dtype=torch.int32).contiguous()
    (global_seq_len, cp_size, rank, local_key_len,
     chunk_len, zigzag) = _qsa_cp_partition_args(
         global_seq_len, cp_size, rank, partition_mode)
    if cache_offsets.shape != (cp_size, local_key_len):
        raise ValueError('QSA CP cache-offset table shape mismatch')
    cache_offsets = cache_offsets.to(
        device=token_indices.device, dtype=torch.int32).contiguous()
    output = token_indices if in_place else torch.empty_like(token_indices)
    block = 256
    route_numel = token_indices.numel()
    _qsa_cp_remap_route_kernel[(triton.cdiv(route_numel, block),)](
        token_indices,
        topk_length,
        cache_offsets,
        output,
        route_numel,
        global_seq_len,
        local_key_len,
        chunk_len,
        ROUTE_SLOTS=token_indices.shape[-1],
        CP_SIZE=cp_size,
        ZIGZAG=zigzag,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )
    return output


def qsa_cp_compact_remap_route_triton(
    block_indices: torch.Tensor,
    topk_length: torch.Tensor,
    query_positions: torch.Tensor,
    cache_offsets: torch.Tensor,
    block_size: int,
    global_seq_len: int,
    cp_size: int,
    rank: int,
    partition_mode: str,
) -> torch.Tensor:
    """Write the final owner-cache token route directly from compact blocks."""

    if not TRITON_AVAILABLE or not block_indices.is_cuda:
        raise RuntimeError('QSA compact CP remap requires CUDA and Triton')
    if block_indices.ndim != 3 or topk_length.shape != block_indices.shape[:2]:
        raise ValueError('QSA compact CP remap route/length shape mismatch')
    if block_indices.dtype != torch.int32:
        raise ValueError('QSA compact CP remap requires int32 block indices')
    batch, seq_len, route_slots = block_indices.shape
    if batch <= 0 or seq_len <= 0 or route_slots <= 0:
        raise ValueError('QSA compact CP remap requires non-empty routes')
    block_size = int(block_size)
    if block_size <= 1:
        raise ValueError('QSA compact CP remap requires block_size > 1')
    block_indices = block_indices.contiguous()
    topk_length = topk_length.to(
        device=block_indices.device, dtype=torch.int32).contiguous()
    query_positions = query_positions.to(
        device=block_indices.device, dtype=torch.int32).reshape(-1).contiguous()
    if query_positions.shape != (seq_len,):
        raise ValueError(
            f'QSA compact CP remap query_positions must have shape [{seq_len}]')
    (global_seq_len, cp_size, rank, local_key_len,
     chunk_len, zigzag) = _qsa_cp_partition_args(
         global_seq_len, cp_size, rank, partition_mode)
    if cache_offsets.shape != (cp_size, local_key_len):
        raise ValueError('QSA compact CP cache-offset table shape mismatch')
    cache_offsets = cache_offsets.to(
        device=block_indices.device, dtype=torch.int32).contiguous()
    token_slots = route_slots * block_size + block_size - 1
    output = torch.empty(
        (batch, seq_len, token_slots),
        device=block_indices.device,
        dtype=torch.int32,
    )
    block = 1024
    output_numel = output.numel()
    _qsa_cp_compact_remap_route_kernel[(
        triton.cdiv(output_numel, block),
    )](
        block_indices,
        topk_length,
        query_positions,
        cache_offsets,
        output,
        output_numel,
        seq_len,
        global_seq_len,
        local_key_len,
        chunk_len,
        ROUTE_SLOTS=route_slots,
        OUTPUT_SLOTS=token_slots,
        CP_SIZE=cp_size,
        RATIO=block_size,
        ZIGZAG=zigzag,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )
    return output


def qsa_indexer_score_slab_with_ratio(
    q: torch.Tensor,
    block_keys: torch.Tensor,
    query_positions: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """Materialize a bounded FP32 score slab with batched-query K reuse."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("QSA Triton kernels require triton to be installed")
    if not q.is_cuda or not block_keys.is_cuda:
        raise ValueError("QSA Triton indexer requires CUDA tensors")
    if q.ndim != 4 or block_keys.ndim != 3:
        raise ValueError(
            f"unexpected QSA indexer shapes: q={tuple(q.shape)}, "
            f"block_keys={tuple(block_keys.shape)}")
    batch, seq_len, num_heads, head_dim = q.shape
    block_count = block_keys.shape[1]
    if block_keys.shape[0] != batch or block_keys.shape[-1] != head_dim:
        raise ValueError(
            f"QSA indexer shape mismatch: q={tuple(q.shape)}, "
            f"block_keys={tuple(block_keys.shape)}")
    if q.dtype != torch.bfloat16 or block_keys.dtype != torch.bfloat16:
        raise ValueError("QSA batched score slab currently requires BF16 inputs")
    if compress_ratio < 2:
        raise ValueError("QSA score slab requires compress_ratio >= 2")
    if block_count == 0:
        return q.new_empty((batch, seq_len, 0), dtype=torch.float32)
    query_positions = query_positions.to(
        device=q.device, dtype=torch.int32).contiguous()
    if query_positions.shape != (seq_len,):
        raise ValueError(
            f"QSA indexer query_positions must have shape [{seq_len}], "
            f"got {tuple(query_positions.shape)}")
    block_m = int(os.environ.get('MCORE_BRIDGE_QSA_INDEXER_SLAB_BLOCK_M', '16'))
    block_n = int(os.environ.get('MCORE_BRIDGE_QSA_INDEXER_SLAB_BLOCK_N', '128'))
    if block_m not in {4, 8, 16, 32} or block_n not in {32, 64, 128, 256}:
        raise ValueError(
            'QSA indexer score slab expects BLOCK_M in {4,8,16,32} and '
            'BLOCK_N in {32,64,128,256}')
    num_warps = int(os.environ.get('MCORE_BRIDGE_QSA_INDEXER_SLAB_WARPS', '4'))
    num_stages = int(os.environ.get('MCORE_BRIDGE_QSA_INDEXER_SLAB_STAGES', '1'))
    if num_warps not in {4, 8} or num_stages not in {1, 2, 3, 4}:
        raise ValueError(
            'QSA indexer score slab expects warps in {4,8} and stages in {1,2,3,4}')
    maxnreg = int(os.environ.get('MCORE_BRIDGE_QSA_INDEXER_SLAB_MAXNREG', '128'))
    if maxnreg < 0:
        raise ValueError('QSA indexer score slab maxnreg must be non-negative')
    launch_options = {
        'num_warps': num_warps,
        'num_stages': num_stages,
    }
    if maxnreg:
        launch_options['maxnreg'] = maxnreg
    output = torch.empty(
        (batch, seq_len, block_count), device=q.device, dtype=torch.float32)
    grid = (
        triton.cdiv(seq_len, block_m),
        triton.cdiv(block_count, block_n),
        batch,
    )
    _qsa_indexer_score_batched_kernel[grid](
        q,
        block_keys,
        output,
        query_positions,
        seq_len,
        block_count,
        num_heads,
        head_dim,
        head_dim**-0.5,
        compress_ratio,
        query_positions.stride(0),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        block_keys.stride(0),
        block_keys.stride(1),
        block_keys.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_H=max(1, triton.next_power_of_2(num_heads)),
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        **launch_options,
    )
    return output


def qsa_indexer_slab_topk_with_ratio(
    q: torch.Tensor,
    block_keys: torch.Tensor,
    query_positions: torch.Tensor,
    compress_ratio: int,
    block_topk: int,
    max_score_bytes: int = 512 * 1024 * 1024,
    boundary_guard: float = 0.0,
    validate_positions: bool = True,
) -> torch.Tensor:
    """Select compact block routes through bounded score slabs.

    The hot path computes ``BLOCK_M`` queries together so every compressed K
    tile is reused across query rows, then delegates the row selection to
    CUDA's float Top-K.  At most ``max_score_bytes`` of FP32 scores are live.
    Rows whose K/K+1 boundary is numerically ambiguous are recomputed by the
    deterministic streaming selector, preserving the established packed-key
    tie rule without taxing ordinary rows with int64 Top-K.

    This path deliberately accepts only dense, zero-based causal positions.
    Packed documents and CP shards retain their segment-aware streaming
    kernels.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("QSA Triton kernels require triton to be installed")
    if q.ndim != 4 or block_keys.ndim != 3:
        raise ValueError(
            f"unexpected QSA slab Top-K shapes: q={tuple(q.shape)}, "
            f"block_keys={tuple(block_keys.shape)}")
    if not q.is_cuda or not block_keys.is_cuda:
        raise ValueError("QSA slab Top-K requires CUDA tensors")
    batch, seq_len, num_heads, head_dim = q.shape
    num_blocks = block_keys.shape[1]
    if block_keys.shape != (batch, num_blocks, head_dim):
        raise ValueError(
            f"QSA slab Top-K shape mismatch: q={tuple(q.shape)}, "
            f"block_keys={tuple(block_keys.shape)}")
    if q.dtype != torch.bfloat16 or block_keys.dtype != torch.bfloat16:
        raise ValueError("QSA slab Top-K currently requires BF16 inputs")
    if compress_ratio < 2 or block_topk < 1:
        raise ValueError("QSA slab Top-K requires positive K and compress_ratio >= 2")
    if num_blocks <= block_topk:
        raise ValueError("QSA slab Top-K is only needed when blocks exceed the budget")
    if max_score_bytes <= 0:
        raise ValueError("QSA slab Top-K max_score_bytes must be positive")
    if boundary_guard < 0:
        raise ValueError("QSA slab Top-K boundary_guard must be non-negative")
    query_positions = query_positions.to(
        device=q.device, dtype=torch.int32).contiguous()
    if query_positions.shape != (seq_len,):
        raise ValueError(
            "QSA slab Top-K requires one query position per sequence row")
    if validate_positions and not torch.equal(
            query_positions,
            torch.arange(seq_len, device=q.device, dtype=torch.int32)):
        raise ValueError(
            "QSA slab Top-K requires contiguous zero-based query positions")

    selected = torch.empty(
        (batch, seq_len, block_topk), device=q.device, dtype=torch.int32)
    slots = torch.arange(block_topk, device=q.device, dtype=torch.int32)
    # visible_blocks first exceeds K at zero-based position (K+1)*R-1.
    sparse_start = min(seq_len, (block_topk + 1) * compress_ratio - 1)
    if sparse_start:
        direct_counts = ((query_positions[:sparse_start] + 1) // compress_ratio).clamp_max(
            block_topk)
        selected[:, :sparse_start] = torch.where(
            slots[None, None, :] < direct_counts[None, :, None],
            slots[None, None, :],
            -1,
        )
    if sparse_start == seq_len:
        return selected

    block_m = int(os.environ.get('MCORE_BRIDGE_QSA_INDEXER_SLAB_BLOCK_M', '16'))
    if block_m not in {4, 8, 16, 32}:
        raise ValueError(
            'QSA indexer score slab BLOCK_M expects one of {4,8,16,32}')
    max_rows_by_workspace = max_score_bytes // max(
        batch * num_blocks * torch.empty((), dtype=torch.float32).element_size(), 1)
    if max_rows_by_workspace < block_m:
        raise RuntimeError(
            "QSA slab Top-K workspace limit is too small for one query tile: "
            f"need at least {batch * num_blocks * 4 * block_m} bytes")
    desired_rows = min(8192, max(2048, seq_len // 4))
    slab_rows = min(desired_rows, max_rows_by_workspace)
    slab_rows = max(block_m, (slab_rows // block_m) * block_m)
    ambiguous = torch.zeros(
        (batch, seq_len), device=q.device, dtype=torch.bool)
    for query_start in range(sparse_start, seq_len, slab_rows):
        query_end = min(seq_len, query_start + slab_rows)
        # Positions are contiguous, so the last row gives the largest causal
        # block prefix without a device-to-host scalar synchronization.
        visible_blocks = min(num_blocks, query_end // compress_ratio)
        scores = qsa_indexer_score_slab_with_ratio(
            q[:, query_start:query_end],
            block_keys[:, :visible_blocks],
            query_positions[query_start:query_end],
            compress_ratio,
        )
        values, indices = torch.topk(
            scores, block_topk + 1, dim=-1, largest=True, sorted=False)
        next_value, drop_offset = torch.min(values, dim=-1)
        output_offsets = torch.arange(
            block_topk, device=q.device, dtype=drop_offset.dtype)
        keep_offsets = output_offsets + (
            output_offsets < drop_offset[..., None]).logical_not()
        selected[:, query_start:query_end] = torch.gather(
            indices, -1, keep_offsets).to(torch.int32)
        if boundary_guard == 0:
            boundary_ambiguous = torch.sum(
                values == next_value[..., None], dim=-1) > 1
        else:
            candidate_offsets = torch.arange(
                block_topk + 1, device=q.device, dtype=drop_offset.dtype)
            kth = torch.where(
                candidate_offsets == drop_offset[..., None],
                float('inf'),
                values,
            ).amin(dim=-1)
            tolerance = kth.abs().clamp_min(1.0) * boundary_guard
            boundary_ambiguous = kth - next_value <= tolerance
        ambiguous[:, query_start:query_end] = boundary_ambiguous

    # Keep the tie path entirely on-device.  Launching one tiny predicate CTA
    # per row is cheaper than synchronizing the host to materialize nonzero
    # row IDs, and remains compatible with CUDA Graph capture.  Ordinary rows
    # return before loading Q/K; exact-boundary rows run the deterministic
    # packed-key selector and overwrite their provisional CUDA Top-K result.
    index_bits = max(1, (num_blocks - 1).bit_length())
    index_mask = (1 << index_bits) - 1
    stream_block_n = int(os.environ.get(
        'MCORE_BRIDGE_QSA_INDEXER_STREAM_BLOCK_N', str(block_topk)))
    if (stream_block_n < 1 or (stream_block_n & (stream_block_n - 1))
            or stream_block_n > block_topk
            or block_topk % stream_block_n):
        raise ValueError(
            'QSA fused indexer stream BLOCK_N must be a positive '
            'power-of-two not larger than block_topk and divide block_topk')
    indexer_num_warps = int(os.environ.get(
        'MCORE_BRIDGE_QSA_INDEXER_WARPS', '8'))
    indexer_num_stages = int(os.environ.get(
        'MCORE_BRIDGE_QSA_INDEXER_STAGES', '1'))
    _qsa_indexer_fused_topk_kernel[(batch * seq_len,)](
        q,
        block_keys,
        selected,
        query_positions,
        seq_len,
        num_blocks,
        num_heads,
        head_dim,
        compress_ratio,
        head_dim**-0.5,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        block_keys.stride(0),
        block_keys.stride(1),
        block_keys.stride(2),
        selected.stride(0),
        selected.stride(1),
        ambiguous,
        BLOCK_TOPK=block_topk,
        BLOCK_N=stream_block_n,
        BLOCK_H=max(1, triton.next_power_of_2(num_heads)),
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        BLOCK_TAIL=max(
            1, triton.next_power_of_2(compress_ratio - 1)),
        INDEX_BITS=index_bits,
        INDEX_MASK=index_mask,
        RATIO=compress_ratio,
        USE_TOKEN_IDS=False,
        FILTER_ROW_MASK=True,
        BATCH_ONE=batch == 1,
        OUTPUT_BLOCKS=True,
        num_warps=indexer_num_warps,
        num_stages=indexer_num_stages,
    )
    return selected


def qsa_indexer_score_tile_with_ratio(
    q: torch.Tensor,
    block_keys: torch.Tensor,
    query_positions: torch.Tensor,
    block_start: int,
    total_num_blocks: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Launch the Triton indexer scorer for one global block tile.

    ``block_keys`` contains only the tile, while ``block_start`` is its global
    block id.  The kernel receives separate pointer and logical offsets so a
    tile at the end of the sequence cannot accidentally read past its local
    allocation.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("QSA Triton kernels require triton to be installed")
    if not q.is_cuda or not block_keys.is_cuda:
        raise ValueError("QSA Triton indexer requires CUDA tensors")
    batch, seq_len, num_heads, head_dim = q.shape
    if q.ndim != 4 or block_keys.ndim != 3:
        raise ValueError(f"unexpected QSA indexer shapes: q={tuple(q.shape)}, block_keys={tuple(block_keys.shape)}")
    batch, seq_len, num_heads, head_dim = q.shape
    block_count = block_keys.shape[1]
    if block_keys.shape[0] != batch or block_keys.shape[-1] != head_dim:
        raise ValueError(f"QSA indexer shape mismatch: q={tuple(q.shape)}, block_keys={tuple(block_keys.shape)}")
    if block_count == 0:
        return q.new_empty((batch, seq_len, 0), dtype=torch.float32)
    query_positions = query_positions.to(device=q.device, dtype=torch.int32).contiguous()
    if query_positions.shape != (seq_len, ):
        raise ValueError(f"QSA indexer query_positions must have shape [{seq_len}], got {tuple(query_positions.shape)}")
    tile_size = min(64, max(16, triton.next_power_of_2(min(block_count, 64))))
    output = torch.empty((batch, seq_len, block_count), device=q.device, dtype=torch.float32)
    grid = (batch * seq_len * triton.cdiv(block_count, tile_size), )
    _qsa_indexer_score_kernel[grid](
        q,
        block_keys,
        output,
        seq_len,
        total_num_blocks,
        num_heads,
        head_dim,
        head_dim**-0.5,
        compress_ratio,
        query_positions,
        query_positions.stride(0),
        block_start,
        0,
        block_count,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        block_keys.stride(0),
        block_keys.stride(1),
        block_keys.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_BLOCKS=tile_size,
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        NUM_HEADS=num_heads,
        num_warps=4,
    )
    return output


def qsa_indexer_fused_topk_with_ratio(
    q: torch.Tensor,
    block_keys: torch.Tensor,
    query_positions: torch.Tensor,
    compress_ratio: int,
    block_topk: int,
    max_partial_bytes: int = 64 * 1024 * 1024,
    return_block_ids: bool = False,
) -> torch.Tensor:
    """Fuse QSA indexer scoring and streaming block/token Top-K.

    The kernel is deliberately restricted to the production BF16 QSA shape
    family.  The existing tiled scorer remains available for FP32/debug and
    non-power-of-two diagnostic configurations.  ``q`` is ``[B,S,H,D]`` and
    ``block_keys`` is ``[B,N,D]``.  By default the returned token IDs are
    ``[B,S,K]`` with the same block-first plus causal-tail layout as
    :class:`QSAIndexer`.  ``return_block_ids=True`` retains only the compact
    ``[B,S,block_topk]`` complete-block route; attention derives token lanes
    and the causal tail directly from that representation.  Compact output is
    an exact deterministic Top-K set, but its block order is kernel-internal;
    only the public token-ID output promises canonical score order.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("QSA Triton kernels require triton to be installed")
    if not (q.is_cuda and block_keys.is_cuda):
        raise ValueError("QSA fused indexer requires CUDA tensors")
    if q.dtype != torch.bfloat16 or block_keys.dtype != torch.bfloat16:
        raise ValueError("QSA fused indexer currently requires BF16 projected tensors")
    if q.ndim != 4 or block_keys.ndim != 3:
        raise ValueError(
            f"unexpected QSA fused indexer shapes: q={tuple(q.shape)}, "
            f"block_keys={tuple(block_keys.shape)}")
    batch, seq_len, num_heads, head_dim = q.shape
    if block_keys.shape[0] != batch or block_keys.shape[2] != head_dim:
        raise ValueError(
            f"QSA fused indexer shape mismatch: q={tuple(q.shape)}, "
            f"block_keys={tuple(block_keys.shape)}")
    if compress_ratio < 2:
        raise ValueError("QSA fused indexer requires compress_ratio >= 2")
    if block_topk < 1 or (block_topk & (block_topk - 1)):
        raise ValueError("QSA fused indexer requires a power-of-two block_topk")
    num_blocks = block_keys.shape[1]
    if num_blocks <= block_topk:
        raise ValueError("QSA fused indexer is only needed when blocks exceed the budget")
    query_positions = query_positions.to(device=q.device, dtype=torch.int32).contiguous()
    if query_positions.shape != (seq_len,):
        raise ValueError(
            f"QSA fused indexer query_positions must have shape [{seq_len}], "
            f"got {tuple(query_positions.shape)}")
    output_width = (
        block_topk
        if return_block_ids
        else block_topk * compress_ratio + compress_ratio - 1
    )
    output = torch.empty(
        (batch, seq_len, output_width),
        device=q.device,
        dtype=torch.int32,
    )
    index_bits = max(1, (num_blocks - 1).bit_length())
    # 128 keeps the score tile small enough for the BF16 tensor-core path;
    # 512-wide tiles overfill the fused Top-K register state on SM90.
    block_n = min(128, block_topk)
    indexer_block_m = int(os.environ.get('MCORE_BRIDGE_QSA_INDEXER_BLOCK_M', '4'))
    if indexer_block_m not in {1, 2, 4}:
        raise ValueError('QSA fused indexer BLOCK_M expects one of {1,2,4}')
    indexer_num_warps = int(os.environ.get('MCORE_BRIDGE_QSA_INDEXER_WARPS', '8'))
    if indexer_num_warps not in {2, 4, 8}:
        raise ValueError('QSA fused indexer num_warps expects one of {2,4,8}')
    indexer_num_stages = int(
        os.environ.get('MCORE_BRIDGE_QSA_INDEXER_STAGES', '1'))
    if indexer_num_stages not in {1, 2, 3, 4}:
        raise ValueError('QSA fused indexer stages expects one of {1,2,3,4}')
    index_mask = (1 << index_bits) - 1
    # A single-row program minimizes workspace at long context.  For shorter
    # sequences, split the block scan into a small power-of-two grid so the
    # large Top-K register state does not serialize the whole indexer.  The
    # partial workspace is explicitly capped and is released after the merge.
    bytes_per_split = max(batch * seq_len * block_topk * 8, 1)
    split_budget = max(1, int(max_partial_bytes) // bytes_per_split)
    possible_splits = max(1, triton.cdiv(num_blocks, block_n))
    # A whole-row scan is faster while the compressed context is only a few
    # Top-K widths.  Splitting such rows creates a large partial Top-K tensor
    # without enough independent work to amortize the merge launch.
    if num_blocks <= 4 * block_topk:
        split_budget = 1
    splits = 1
    while splits * 2 <= min(split_budget, possible_splits):
        splits *= 2

    # The Qwen4-Exp production shape is block_topk=512.  Follow the
    # TokenSpeed-style running merge there: a tile is exactly one K-wide
    # sorted list, and ``maximum(bitonic_merge(acc), sorted(tile))`` avoids
    # the fragile 1024-wide tl.topk state.  The same route is also used by the
    # smaller power-of-two diagnostic budgets (K >= 8), keeping fallback
    # behavior on the same launch/ordering path as production K=512.
    streaming_enabled = (
        block_topk in {8, 16, 32, 64, 128, 256, 512}
        and (block_topk & (block_topk - 1)) == 0
        and os.environ.get('MCORE_BRIDGE_QSA_INDEXER_STREAMING', '1') != '0'
    )
    if return_block_ids and not streaming_enabled:
        raise RuntimeError(
            'QSA compact block route requires the streaming fused indexer')
    stream_block_n = int(os.environ.get(
        'MCORE_BRIDGE_QSA_INDEXER_STREAM_BLOCK_N', str(block_topk)))
    if (stream_block_n < 1 or (stream_block_n & (stream_block_n - 1))
            or stream_block_n > block_topk or block_topk % stream_block_n):
        raise ValueError(
            'QSA fused indexer stream BLOCK_N must be a positive power-of-two '
            'not larger than block_topk and divide block_topk')
    batch_one = (
        batch == 1
        and os.environ.get('MCORE_BRIDGE_QSA_INDEXER_BATCH_ONE', '1') != '0')
    split_expansion_default = '1' if batch == 1 and seq_len >= 65536 else '0'
    if (streaming_enabled and splits == 1
            and os.environ.get(
                'MCORE_BRIDGE_QSA_INDEXER_SPLIT_EXPANSION',
                split_expansion_default) == '1'):
        # Keep the 512-entry Top-K producer free of the 2K token expansion
        # stores.  This is an A/B route for compiler register allocation; it
        # uses one compact int32 block-id buffer and a bandwidth-only second
        # launch, while preserving the public token-list contract exactly.
        selected_blocks = (
            output
            if return_block_ids
            else torch.empty(
                (batch, seq_len, block_topk),
                device=q.device,
                dtype=torch.int32,
            )
        )
        _qsa_indexer_fused_topk_kernel[(batch * seq_len,)](
            q,
            block_keys,
            selected_blocks,
            query_positions,
            seq_len,
            num_blocks,
            num_heads,
            head_dim,
            compress_ratio,
            head_dim**-0.5,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            block_keys.stride(0),
            block_keys.stride(1),
            block_keys.stride(2),
            selected_blocks.stride(0),
            selected_blocks.stride(1),
            query_positions,
            BLOCK_TOPK=block_topk,
            BLOCK_N=stream_block_n,
            BLOCK_H=max(1, triton.next_power_of_2(num_heads)),
            BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
            BLOCK_TAIL=max(1, triton.next_power_of_2(compress_ratio - 1)),
            INDEX_BITS=index_bits,
            INDEX_MASK=index_mask,
            RATIO=compress_ratio,
            USE_TOKEN_IDS=False,
            FILTER_ROW_MASK=False,
            BATCH_ONE=batch_one,
            OUTPUT_BLOCKS=True,
            num_warps=indexer_num_warps,
            num_stages=indexer_num_stages,
        )
        if return_block_ids:
            return selected_blocks
        _qsa_indexer_expand_block_topk_kernel[(batch * seq_len,)](
            selected_blocks,
            output,
            query_positions,
            seq_len,
            num_blocks,
            compress_ratio,
            selected_blocks.stride(0),
            selected_blocks.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_TOPK=block_topk,
            BLOCK_TAIL=max(1, triton.next_power_of_2(compress_ratio - 1)),
            RATIO=compress_ratio,
            num_warps=1,
            num_stages=1,
        )
        return output
    if (streaming_enabled and splits == 1
            and os.environ.get('MCORE_BRIDGE_QSA_INDEXER_PARTIAL', '0') == '1'):
        # TokenSpeed's two-stage layout is useful as a diagnostic: keep the
        # score/top-k registers separate from token expansion.  It is
        # deliberately opt-in because the one-split int64 partial consumes
        # roughly B*S*block_topk*8 bytes.
        blocks_per_split = num_blocks
        partial = torch.empty(
            (batch, seq_len, 1, block_topk),
            device=q.device,
            dtype=torch.int64,
        )
        _qsa_indexer_stream_partial_kernel[(batch * seq_len, 1)](
            q,
            block_keys,
            partial,
            query_positions,
            seq_len,
            num_blocks,
            num_heads,
            head_dim,
            compress_ratio,
            head_dim**-0.5,
            blocks_per_split,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            block_keys.stride(0),
            block_keys.stride(1),
            block_keys.stride(2),
            partial.stride(0),
            partial.stride(1),
            partial.stride(2),
            partial.stride(3),
            BLOCK_TOPK=block_topk,
            BLOCK_N=stream_block_n,
            BLOCK_H=max(1, triton.next_power_of_2(num_heads)),
            BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
            INDEX_BITS=index_bits,
            INDEX_MASK=index_mask,
            RATIO=compress_ratio,
            num_warps=indexer_num_warps,
            num_stages=indexer_num_stages,
        )
        _qsa_indexer_merge_topk_kernel[(batch * seq_len,)](
            partial,
            output,
            query_positions,
            seq_len,
            num_blocks,
            compress_ratio,
            blocks_per_split,
            partial.stride(0),
            partial.stride(1),
            partial.stride(2),
            partial.stride(3),
            output.stride(0),
            output.stride(1),
            BLOCK_TOPK=block_topk,
            SPLITS=1,
            BLOCK_TAIL=max(1, triton.next_power_of_2(compress_ratio - 1)),
            INDEX_MASK=index_mask,
            RATIO=compress_ratio,
            OUTPUT_BLOCKS=return_block_ids,
            num_warps=8,
            num_stages=1,
        )
        return output
    direct_max_multiple = int(os.environ.get(
        'MCORE_BRIDGE_QSA_INDEXER_DIRECT_MAX_MULTIPLE', '8'))
    if direct_max_multiple < 1:
        raise ValueError(
            'QSA fused indexer direct-fill max multiple must be positive')
    if (streaming_enabled and splits == 1
            and not return_block_ids
            and num_blocks <= direct_max_multiple * block_topk
            and os.environ.get('MCORE_BRIDGE_QSA_INDEXER_MIXED_DIRECT', '1') != '0'
            and os.environ.get('MCORE_BRIDGE_QSA_INDEXER_PARTIAL', '0') != '1'):
        # Short causal rows need no score computation: all complete blocks fit
        # in the budget and are selected in local order.  Dispatch them to a
        # compact fill kernel and keep the real Top-K work for the remaining
        # rows.  The heuristic is limited to short contexts, where the extra
        # row-list launch is amortized by the skipped score tiles.
        complete_blocks = torch.minimum(
            torch.full_like(
                query_positions, num_blocks, dtype=torch.int64),
            (query_positions.to(torch.int64) + 1) // compress_ratio,
        )
        short_mask = complete_blocks <= block_topk
        if bool(short_mask.any().item()):
            row_ids = torch.arange(
                batch * seq_len, device=q.device, dtype=torch.int32)
            short_mask_flat = short_mask.unsqueeze(0).expand(
                batch, -1).reshape(-1)
            short_ids = row_ids[short_mask_flat].contiguous()
            long_ids = row_ids[~short_mask_flat].contiguous()
            _qsa_indexer_direct_fill_kernel[(short_ids.numel(),)](
                output,
                query_positions,
                short_ids,
                seq_len,
                num_blocks,
                compress_ratio,
                output.stride(1),
                query_positions.stride(0),
                BLOCK_TOPK=block_topk,
                BLOCK_TAIL=max(1, triton.next_power_of_2(compress_ratio - 1)),
                RATIO=compress_ratio,
                OUTPUT_BLOCKS=return_block_ids,
                num_warps=1,
                num_stages=1,
            )
            if long_ids.numel():
                _qsa_indexer_fused_topk_kernel[(long_ids.numel(),)](
                    q,
                    block_keys,
                    output,
                    query_positions,
                    seq_len,
                    num_blocks,
                    num_heads,
                    head_dim,
                    compress_ratio,
                    head_dim**-0.5,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    block_keys.stride(0),
                    block_keys.stride(1),
                    block_keys.stride(2),
                    output.stride(0),
                    output.stride(1),
                    long_ids,
                    BLOCK_TOPK=block_topk,
                    BLOCK_N=stream_block_n,
                    BLOCK_H=max(1, triton.next_power_of_2(num_heads)),
                    BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
                    BLOCK_TAIL=max(1, triton.next_power_of_2(compress_ratio - 1)),
                    INDEX_BITS=index_bits,
                    INDEX_MASK=index_mask,
                    RATIO=compress_ratio,
                    USE_TOKEN_IDS=True,
                    FILTER_ROW_MASK=False,
                    BATCH_ONE=batch_one,
                    OUTPUT_BLOCKS=return_block_ids,
                    num_warps=indexer_num_warps,
                    num_stages=indexer_num_stages,
                )
            return output
    if streaming_enabled and splits == 1:
        _qsa_indexer_fused_topk_kernel[(batch * seq_len,)](
            q,
            block_keys,
            output,
            query_positions,
            seq_len,
            num_blocks,
            num_heads,
            head_dim,
            compress_ratio,
            head_dim**-0.5,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            block_keys.stride(0),
            block_keys.stride(1),
            block_keys.stride(2),
            output.stride(0),
            output.stride(1),
            query_positions,
            BLOCK_TOPK=block_topk,
            BLOCK_N=stream_block_n,
            BLOCK_H=max(1, triton.next_power_of_2(num_heads)),
            BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
            INDEX_BITS=index_bits,
            INDEX_MASK=index_mask,
            BLOCK_TAIL=max(1, triton.next_power_of_2(compress_ratio - 1)),
            RATIO=compress_ratio,
            USE_TOKEN_IDS=False,
            FILTER_ROW_MASK=False,
            BATCH_ONE=batch_one,
            OUTPUT_BLOCKS=return_block_ids,
            num_warps=indexer_num_warps,
            num_stages=indexer_num_stages,
        )
        return output

    if streaming_enabled and splits > 1:
        # A large partial budget enables block-scan parallelism.  Each row is
        # split over disjoint compressed-key ranges, and one final merge keeps
        # the same packed score/id ordering as the one-row route.  This is an
        # opt-in long-context path because the int64 partial workspace is
        # intentionally bounded by max_partial_bytes.
        blocks_per_split = triton.cdiv(num_blocks, splits)
        partial = torch.empty(
            (batch, seq_len, splits, block_topk),
            device=q.device,
            dtype=torch.int64,
        )
        _qsa_indexer_stream_partial_kernel[(batch * seq_len, splits)](
            q,
            block_keys,
            partial,
            query_positions,
            seq_len,
            num_blocks,
            num_heads,
            head_dim,
            compress_ratio,
            head_dim**-0.5,
            blocks_per_split,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            block_keys.stride(0),
            block_keys.stride(1),
            block_keys.stride(2),
            partial.stride(0),
            partial.stride(1),
            partial.stride(2),
            partial.stride(3),
            BLOCK_TOPK=block_topk,
            # Keep the same tile width as the tuned single-row producer.
            # The old split-only BLOCK_N=128 multiplied the number of
            # streaming Top-K merges by four for the production K=512 shape.
            BLOCK_N=stream_block_n,
            BLOCK_H=max(1, triton.next_power_of_2(num_heads)),
            BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
            INDEX_BITS=index_bits,
            INDEX_MASK=index_mask,
            RATIO=compress_ratio,
            num_warps=4,
            num_stages=2,
        )
        _qsa_indexer_merge_topk_kernel[(batch * seq_len,)](
            partial,
            output,
            query_positions,
            seq_len,
            num_blocks,
            compress_ratio,
            blocks_per_split,
            partial.stride(0),
            partial.stride(1),
            partial.stride(2),
            partial.stride(3),
            output.stride(0),
            output.stride(1),
            BLOCK_TOPK=block_topk,
            SPLITS=splits,
            BLOCK_TAIL=max(1, triton.next_power_of_2(compress_ratio - 1)),
            INDEX_MASK=index_mask,
            RATIO=compress_ratio,
            OUTPUT_BLOCKS=return_block_ids,
            num_warps=8,
            num_stages=1,
        )
        return output

    if (block_topk == 512 and block_n == 128 and
            os.environ.get('MCORE_BRIDGE_QSA_INDEXER_MERGE4', '1') != '0'):
        _qsa_indexer_radix_topk_batched_merge4_kernel[
            (batch * triton.cdiv(seq_len, indexer_block_m),)
        ](
            q,
            block_keys,
            output,
            query_positions,
            seq_len,
            num_blocks,
            num_heads,
            head_dim,
            compress_ratio,
            head_dim**-0.5,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            block_keys.stride(0),
            block_keys.stride(1),
            block_keys.stride(2),
            output.stride(0),
            output.stride(1),
            BLOCK_M=indexer_block_m,
            BLOCK_TOPK=block_topk,
            BLOCK_H=max(1, triton.next_power_of_2(num_heads)),
            BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
            INDEX_BITS=index_bits,
            INDEX_MASK=index_mask,
            BLOCK_TAIL=max(1, triton.next_power_of_2(compress_ratio - 1)),
            RATIO=compress_ratio,
            num_warps=indexer_num_warps,
            num_stages=1,
        )
        return output

    if splits == 1:
        _qsa_indexer_radix_topk_batched_kernel[
            (batch * triton.cdiv(seq_len, indexer_block_m),)
        ](
            q,
            block_keys,
            output,
            query_positions,
            seq_len,
            num_blocks,
            num_heads,
            head_dim,
            compress_ratio,
            head_dim**-0.5,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            block_keys.stride(0),
            block_keys.stride(1),
            block_keys.stride(2),
            output.stride(0),
            output.stride(1),
            BLOCK_M=indexer_block_m,
            BLOCK_TOPK=block_topk,
            BLOCK_H=max(1, triton.next_power_of_2(num_heads)),
            BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
            INDEX_BITS=index_bits,
            INDEX_MASK=index_mask,
            BLOCK_TAIL=max(1, triton.next_power_of_2(compress_ratio - 1)),
            RATIO=compress_ratio,
            num_warps=indexer_num_warps,
            num_stages=1,
        )
        return output

    blocks_per_split = triton.cdiv(num_blocks, splits)
    partial = torch.empty(
        (batch, seq_len, splits, block_topk),
        device=q.device,
        dtype=torch.int64,
    )
    _qsa_indexer_stream_partial_kernel[(batch * seq_len, splits)](
        q,
        block_keys,
        partial,
        query_positions,
        seq_len,
        num_blocks,
        num_heads,
        head_dim,
        compress_ratio,
        head_dim**-0.5,
        blocks_per_split,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        block_keys.stride(0),
        block_keys.stride(1),
        block_keys.stride(2),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        partial.stride(3),
        BLOCK_TOPK=block_topk,
        BLOCK_N=block_n,
        BLOCK_H=max(1, triton.next_power_of_2(num_heads)),
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        INDEX_BITS=index_bits,
        INDEX_MASK=index_mask,
        RATIO=compress_ratio,
        num_warps=4,
        num_stages=2,
    )
    _qsa_indexer_merge_topk_kernel[(batch * seq_len,)](
        partial,
        output,
        query_positions,
        seq_len,
        num_blocks,
        compress_ratio,
        blocks_per_split,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        partial.stride(3),
        output.stride(0),
        output.stride(1),
        BLOCK_TOPK=block_topk,
        SPLITS=splits,
        BLOCK_TAIL=max(1, triton.next_power_of_2(compress_ratio - 1)),
        INDEX_MASK=index_mask,
        RATIO=compress_ratio,
        OUTPUT_BLOCKS=False,
        num_warps=8,
        num_stages=1,
    )
    return output


def qsa_indexer_fused_topk_packed(
    query: torch.Tensor,
    block_keys: torch.Tensor,
    block_starts: torch.Tensor,
    segment_block_counts: torch.Tensor,
    query_positions: torch.Tensor,
    compress_ratio: int,
    block_topk: int,
    return_block_ids: bool = False,
) -> torch.Tensor:
    """Run TokenSpeed-style indexer Top-K once over packed query segments.

    ``return_block_ids`` keeps segment-local complete-block IDs and omits the
    expanded token lanes/tail from persistent metadata.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("QSA Triton kernels require triton to be installed")
    if not (query.is_cuda and block_keys.is_cuda):
        raise ValueError("QSA packed indexer requires CUDA tensors")
    if query.ndim != 3 or block_keys.ndim != 2:
        raise ValueError("QSA packed indexer expects query [T,H,D], blocks [N,D]")
    if query.dtype != torch.bfloat16 or block_keys.dtype != torch.bfloat16:
        raise ValueError("QSA packed indexer currently requires BF16 projected tensors")
    total_tokens, num_heads, head_dim = query.shape
    total_blocks, block_dim = block_keys.shape
    if block_dim != head_dim or compress_ratio < 2:
        raise ValueError("QSA packed indexer has incompatible ratio/head dimension")
    if block_topk < 1 or (block_topk & (block_topk - 1)):
        raise ValueError("QSA packed indexer requires a power-of-two block_topk")
    for name, tensor in (
        ('block_starts', block_starts),
        ('segment_block_counts', segment_block_counts),
        ('query_positions', query_positions),
    ):
        if tensor.shape != (total_tokens,):
            raise ValueError(f"QSA packed indexer {name} must be [{total_tokens}]")
    if total_tokens == 0:
        return torch.empty(
            (
                0,
                block_topk if return_block_ids
                else block_topk * compress_ratio + compress_ratio - 1,
            ),
            device=query.device,
            dtype=torch.int32,
        )
    block_starts = block_starts.to(device=query.device, dtype=torch.int32).contiguous()
    segment_block_counts = segment_block_counts.to(
        device=query.device, dtype=torch.int32).contiguous()
    query_positions = query_positions.to(device=query.device, dtype=torch.int32).contiguous()
    query = query.contiguous()
    block_keys = block_keys.contiguous()
    output_width = (
        block_topk
        if return_block_ids
        else block_topk * compress_ratio + compress_ratio - 1
    )
    output = torch.empty(
        (total_tokens, output_width),
        device=query.device,
        dtype=torch.int32,
    )
    max_segment_blocks = max(1, int(segment_block_counts.max().item()))
    if max_segment_blocks <= block_topk:
        # All complete blocks fit in the selected-token budget, so the
        # indexer result is deterministic local block expansion.  This is a
        # particularly important case for packed 1K-4K sequences, where a
        # regular top-k launch would otherwise repeat the same score work for
        # every independent segment.
        _qsa_indexer_packed_direct_fill_kernel[(total_tokens,)](
            output,
            segment_block_counts,
            query_positions,
            segment_block_counts,
            total_tokens,
            compress_ratio,
            output.stride(0),
            segment_block_counts.stride(0),
            query_positions.stride(0),
            BLOCK_TOPK=block_topk,
            BLOCK_TAIL=max(1, triton.next_power_of_2(compress_ratio - 1)),
            RATIO=compress_ratio,
            USE_TOKEN_IDS=False,
            OUTPUT_BLOCKS=return_block_ids,
            num_warps=1,
            num_stages=1,
        )
        return output
    # Mixed packed batches can contain both short segments (all blocks fit in
    # the budget) and long segments (real score/top-k required).  Dispatch
    # those token subsets independently so short documents do not pay for a
    # score scan.  The metadata lists are O(T) and are only materialized in
    # this mixed case.
    short_mask = segment_block_counts <= block_topk
    if (os.environ.get('MCORE_BRIDGE_QSA_INDEXER_MIXED_DIRECT', '1') != '0'
            and bool(short_mask.any().item())):
        short_token_ids = torch.nonzero(short_mask, as_tuple=False).flatten().to(torch.int32)
        long_token_ids = torch.nonzero(~short_mask, as_tuple=False).flatten().to(torch.int32)
        _qsa_indexer_packed_direct_fill_kernel[(short_token_ids.numel(),)](
            output,
            segment_block_counts,
            query_positions,
            short_token_ids,
            total_tokens,
            compress_ratio,
            output.stride(0),
            segment_block_counts.stride(0),
            query_positions.stride(0),
            BLOCK_TOPK=block_topk,
            BLOCK_TAIL=max(1, triton.next_power_of_2(compress_ratio - 1)),
            RATIO=compress_ratio,
            USE_TOKEN_IDS=True,
            OUTPUT_BLOCKS=return_block_ids,
            num_warps=1,
            num_stages=1,
        )
        index_bits = max(1, (max_segment_blocks - 1).bit_length())
        index_mask = (1 << index_bits) - 1
        num_warps = int(os.environ.get('MCORE_BRIDGE_QSA_INDEXER_WARPS', '8'))
        if num_warps not in {2, 4, 8}:
            raise ValueError('QSA packed indexer num_warps expects one of {2,4,8}')
        num_stages = int(
            os.environ.get('MCORE_BRIDGE_QSA_INDEXER_STAGES', '2'))
        if num_stages not in {1, 2, 3, 4}:
            raise ValueError('QSA packed indexer stages expects one of {1,2,3,4}')
        stream_block_n = int(os.environ.get(
            'MCORE_BRIDGE_QSA_INDEXER_STREAM_BLOCK_N', str(block_topk)))
        if (stream_block_n < 1 or (stream_block_n & (stream_block_n - 1))
                or stream_block_n > block_topk or block_topk % stream_block_n):
            raise ValueError(
                'QSA packed indexer stream BLOCK_N must be a positive '
                'power-of-two not larger than block_topk and divide block_topk')
        if long_token_ids.numel():
            _qsa_indexer_fused_topk_packed_kernel[(long_token_ids.numel(),)](
                query,
                block_keys,
                output,
                block_starts,
                segment_block_counts,
                query_positions,
                long_token_ids,
                total_tokens,
                total_blocks,
                num_heads,
                head_dim,
                compress_ratio,
                head_dim**-0.5,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                block_keys.stride(0),
                block_keys.stride(1),
                output.stride(0),
                block_starts.stride(0),
                segment_block_counts.stride(0),
                query_positions.stride(0),
                BLOCK_TOPK=block_topk,
                BLOCK_N=stream_block_n,
                BLOCK_H=max(1, triton.next_power_of_2(num_heads)),
                BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
                BLOCK_TAIL=max(1, triton.next_power_of_2(compress_ratio - 1)),
                INDEX_BITS=index_bits,
                INDEX_MASK=index_mask,
                RATIO=compress_ratio,
                USE_TOKEN_IDS=True,
                OUTPUT_BLOCKS=return_block_ids,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        return output
    index_bits = max(1, (max_segment_blocks - 1).bit_length())
    index_mask = (1 << index_bits) - 1
    num_warps = int(os.environ.get('MCORE_BRIDGE_QSA_INDEXER_WARPS', '8'))
    if num_warps not in {2, 4, 8}:
        raise ValueError('QSA packed indexer num_warps expects one of {2,4,8}')
    num_stages = int(
        os.environ.get('MCORE_BRIDGE_QSA_INDEXER_STAGES', '2'))
    if num_stages not in {1, 2, 3, 4}:
        raise ValueError('QSA packed indexer stages expects one of {1,2,3,4}')
    stream_block_n = int(os.environ.get(
        'MCORE_BRIDGE_QSA_INDEXER_STREAM_BLOCK_N', str(block_topk)))
    if (stream_block_n < 1 or (stream_block_n & (stream_block_n - 1))
            or stream_block_n > block_topk or block_topk % stream_block_n):
        raise ValueError(
            'QSA packed indexer stream BLOCK_N must be a positive power-of-two '
            'not larger than block_topk and divide block_topk')
    _qsa_indexer_fused_topk_packed_kernel[(total_tokens,)](
        query,
        block_keys,
        output,
        block_starts,
        segment_block_counts,
        query_positions,
        segment_block_counts,
        total_tokens,
        total_blocks,
        num_heads,
        head_dim,
        compress_ratio,
        head_dim**-0.5,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        block_keys.stride(0),
        block_keys.stride(1),
        output.stride(0),
        block_starts.stride(0),
        segment_block_counts.stride(0),
        query_positions.stride(0),
        BLOCK_TOPK=block_topk,
        BLOCK_N=stream_block_n,
        BLOCK_H=max(1, triton.next_power_of_2(num_heads)),
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        BLOCK_TAIL=max(1, triton.next_power_of_2(compress_ratio - 1)),
        INDEX_BITS=index_bits,
        INDEX_MASK=index_mask,
        RATIO=compress_ratio,
        USE_TOKEN_IDS=False,
        OUTPUT_BLOCKS=return_block_ids,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def qsa_selected_kv_forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                            topk_indices: torch.Tensor, topk_length: torch.Tensor,
                            softmax_scale: float, causal: bool = True,
                            query_positions: torch.Tensor = None, key_position_offset: int = 0,
                            route_block_size: int = 1) -> tuple:
    """Launch the Triton selected-KV forward kernel.

    The public attention wrapper validates and makes all tensors contiguous
    before calling this function.  The returned LSE layout is ``[B, Hq, Sq]``.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("QSA Triton kernels require triton to be installed")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise ValueError("QSA Triton attention requires CUDA tensors")
    sq, batch, num_q_heads, head_dim = query.shape
    sk, key_batch, num_kv_heads, value_dim = key.shape
    if key_batch != batch or value.shape != key.shape:
        raise ValueError(f"QSA selected-KV K/V shape mismatch: key={tuple(key.shape)}, value={tuple(value.shape)}")
    if value_dim != head_dim or num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"QSA selected-KV head shape is unsupported: query={tuple(query.shape)}, key={tuple(key.shape)}")
    if topk_indices.shape != (batch, sq, topk_indices.shape[-1]) or topk_length.shape != (batch, sq):
        raise ValueError(
            f"QSA selected-KV index shape mismatch: query={tuple(query.shape)}, "
            f"indices={tuple(topk_indices.shape)}, length={tuple(topk_length.shape)}")
    route_slots = topk_indices.shape[-1]
    route_block_size = int(route_block_size)
    if route_slots <= 0:
        raise ValueError("QSA selected-KV requires at least one index slot")
    if route_block_size <= 0:
        raise ValueError("QSA selected-KV route_block_size must be positive")
    if route_block_size > 1 and (not causal or int(key_position_offset) != 0):
        raise ValueError(
            "QSA compact block route requires causal attention and key_position_offset=0")
    k_slots = (
        route_slots
        if route_block_size == 1
        else route_slots * route_block_size + route_block_size - 1
    )

    output = torch.empty_like(query)
    lse = torch.empty((batch, num_q_heads, sq), device=query.device, dtype=torch.float32)
    if query_positions is None:
        query_positions = torch.arange(sq, device=query.device, dtype=torch.int32)
    query_positions = query_positions.to(device=query.device, dtype=torch.int32).contiguous()
    if query_positions.shape != (sq, ):
        raise ValueError(f"QSA query_positions must have shape [{sq}], got {tuple(query_positions.shape)}")
    group_size = num_q_heads // num_kv_heads
    # Cover one complete GQA group with one power-of-two tile.  On the
    # production Hq/Hkv=24/2 shape this removes two duplicate K/V scans per
    # KV head; TP2/4/8 naturally dispatch to 16/8/4 heads respectively.
    default_head_tile = min(16, 1 << max(0, group_size - 1).bit_length())
    head_tile_size = int(os.environ.get(
        'MCORE_BRIDGE_QSA_FORWARD_HEAD_TILE', str(default_head_tile)))
    if head_tile_size not in {1, 2, 4, 8, 16}:
        raise ValueError('QSA forward head tile expects one of {1,2,4,8,16}')
    num_head_tiles = triton.cdiv(group_size, head_tile_size)
    # Keep every CUDA grid dimension below its device limit.  In particular,
    # a 65,536-row sequence cannot be placed directly on grid.y (max 65,535).
    grid = (batch * sq, num_kv_heads * num_head_tiles)
    block_d = max(16, triton.next_power_of_2(head_dim))
    # The wider SM90 tile amortizes compact-route decode and online-softmax
    # bookkeeping while still fitting three 4-warp CTAs per SM (166 registers
    # and about 76 KiB shared memory on the production D=256 shape).
    default_block_k = (
        64
        if head_tile_size == 16 and head_dim == 256 and is_sm90(query.device)
        else 16
    )
    block_k = int(os.environ.get(
        'MCORE_BRIDGE_QSA_FORWARD_BLOCK_K', str(default_block_k)))
    if block_k not in {8, 16, 32, 64, 128}:
        raise ValueError('QSA forward BLOCK_K expects one of {8,16,32,64,128}')
    default_num_warps = min(4, max(1, head_tile_size // 4))
    forward_num_warps = int(os.environ.get(
        'MCORE_BRIDGE_QSA_FORWARD_WARPS', str(default_num_warps)))
    forward_num_stages = int(os.environ.get('MCORE_BRIDGE_QSA_FORWARD_STAGES', '2'))
    if forward_num_warps not in {1, 2, 4, 8} or forward_num_stages not in {1, 2, 3, 4}:
        raise ValueError(
            'QSA forward tuning expects warps in {1,2,4,8} and stages in {1,2,3,4}')
    default_forward_maxnreg = (
        88
        if (head_tile_size == 16 and head_dim == 256 and block_k == 16
            and is_sm90(query.device))
        else 0
    )
    forward_maxnreg = int(os.environ.get(
        'MCORE_BRIDGE_QSA_FORWARD_MAXNREG', str(default_forward_maxnreg)))
    if forward_maxnreg < 0:
        raise ValueError('QSA forward maxnreg must be non-negative')
    launch_options = {
        'num_warps': forward_num_warps,
        'num_stages': forward_num_stages,
    }
    if forward_maxnreg:
        launch_options['maxnreg'] = forward_maxnreg
    trim_causal_loop = _qsa_trim_causal_loop(causal, sq, k_slots)
    _qsa_selected_kv_forward_grouped_kernel[grid](
        query,
        key,
        value,
        topk_indices,
        topk_length,
        output,
        lse,
        sq,
        sk,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        key_position_offset,
        query_positions,
        query_positions.stride(0),
        query.stride(1),
        query.stride(0),
        query.stride(2),
        query.stride(3),
        key.stride(1),
        key.stride(0),
        key.stride(2),
        key.stride(3),
        value.stride(1),
        value.stride(0),
        value.stride(2),
        value.stride(3),
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        topk_length.stride(0),
        topk_length.stride(1),
        output.stride(1),
        output.stride(0),
        output.stride(2),
        output.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        K=k_slots,
        HEADS_PER_KV=head_tile_size,
        GROUP_SIZE=group_size,
        NUM_HEAD_TILES=num_head_tiles,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        CAUSAL=causal,
        TRIM_CAUSAL_LOOP=trim_causal_loop,
        ROUTE_SLOTS=route_slots,
        ROUTE_BLOCK_SIZE=route_block_size,
        **launch_options,
    )
    return output, lse


def qsa_selected_kv_forward_packed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    key_starts: torch.Tensor,
    key_lengths: torch.Tensor,
    query_positions: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
    key_position_offset: int = 0,
    route_block_size: int = 1,
) -> tuple:
    """Launch one selected-KV forward kernel over all packed THD segments."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("QSA Triton kernels require triton to be installed")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise ValueError("QSA packed Triton attention requires CUDA tensors")
    if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("QSA packed Triton attention expects Q/K/V as [T,H,D]")
    total_q, num_q_heads, head_dim = query.shape
    total_k, num_kv_heads, value_dim = key.shape
    if value_dim != head_dim or num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        raise ValueError("QSA packed Triton attention has incompatible Q/KV heads")
    if topk_indices.ndim != 2 or topk_indices.shape[0] != total_q:
        raise ValueError("QSA packed Triton indices must be [T,K]")
    if topk_length.shape != (total_q,):
        raise ValueError("QSA packed Triton lengths must be [T]")
    route_slots = topk_indices.shape[1]
    route_block_size = int(route_block_size)
    if route_slots <= 0 or route_block_size <= 0:
        raise ValueError("QSA packed Triton route slots/block size must be positive")
    if route_block_size > 1 and (not causal or int(key_position_offset) != 0):
        raise ValueError(
            "QSA packed compact block route requires causal attention and key_position_offset=0")
    logical_k = (
        route_slots
        if route_block_size == 1
        else route_slots * route_block_size + route_block_size - 1
    )
    for name, tensor in (
        ('key_starts', key_starts),
        ('key_lengths', key_lengths),
        ('query_positions', query_positions),
    ):
        if tensor.shape != (total_q,):
            raise ValueError(f"QSA packed Triton {name} must be [{total_q}]")
    if total_q == 0:
        return query.new_empty((0, num_q_heads, head_dim)), torch.empty(
            (num_q_heads, 0), device=query.device, dtype=torch.float32)
    topk_indices = topk_indices.to(device=query.device, dtype=torch.int32).contiguous()
    topk_length = topk_length.to(device=query.device, dtype=torch.int32).contiguous()
    key_starts = key_starts.to(device=query.device, dtype=torch.int32).contiguous()
    key_lengths = key_lengths.to(device=query.device, dtype=torch.int32).contiguous()
    query_positions = query_positions.to(device=query.device, dtype=torch.int32).contiguous()
    output = torch.empty_like(query)
    lse = torch.empty((num_q_heads, total_q), device=query.device, dtype=torch.float32)
    group_size = num_q_heads // num_kv_heads
    default_head_tile = min(16, 1 << max(0, group_size - 1).bit_length())
    head_tile_size = int(os.environ.get(
        'MCORE_BRIDGE_QSA_FORWARD_HEAD_TILE', str(default_head_tile)))
    if head_tile_size not in {1, 2, 4, 8, 16}:
        raise ValueError('QSA packed forward head tile expects one of {1,2,4,8,16}')
    num_head_tiles = triton.cdiv(group_size, head_tile_size)
    block_d = max(16, triton.next_power_of_2(head_dim))
    default_block_k = (
        64
        if head_tile_size == 16 and head_dim == 256 and is_sm90(query.device)
        else 16
    )
    block_k = int(os.environ.get(
        'MCORE_BRIDGE_QSA_FORWARD_BLOCK_K', str(default_block_k)))
    if block_k not in {8, 16, 32, 64, 128}:
        raise ValueError('QSA packed forward BLOCK_K expects one of {8,16,32,64,128}')
    default_num_warps = min(4, max(1, head_tile_size // 4))
    num_warps = int(os.environ.get(
        'MCORE_BRIDGE_QSA_FORWARD_WARPS', str(default_num_warps)))
    num_stages = int(os.environ.get('MCORE_BRIDGE_QSA_FORWARD_STAGES', '2'))
    if num_warps not in {1, 2, 4, 8} or num_stages not in {1, 2, 3, 4}:
        raise ValueError('QSA packed forward tuning has invalid warps/stages')
    default_forward_maxnreg = (
        88
        if (head_tile_size == 16 and head_dim == 256 and block_k == 16
            and is_sm90(query.device))
        else 0
    )
    forward_maxnreg = int(os.environ.get(
        'MCORE_BRIDGE_QSA_FORWARD_MAXNREG', str(default_forward_maxnreg)))
    if forward_maxnreg < 0:
        raise ValueError('QSA packed forward maxnreg must be non-negative')
    launch_options = {
        'num_warps': num_warps,
        'num_stages': num_stages,
    }
    if forward_maxnreg:
        launch_options['maxnreg'] = forward_maxnreg
    # As in packed backward, total_q is conservative and avoids reading a
    # GPU-resident maximum document length onto the host.
    trim_causal_loop = _qsa_trim_causal_loop(causal, total_q, logical_k)
    _qsa_selected_kv_forward_packed_grouped_kernel[
        (total_q, num_kv_heads * num_head_tiles)
    ](
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
        total_q,
        total_k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        key_position_offset,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_length.stride(0),
        key_starts.stride(0),
        key_lengths.stride(0),
        query_positions.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        K=logical_k,
        HEADS_PER_KV=head_tile_size,
        GROUP_SIZE=group_size,
        NUM_HEAD_TILES=num_head_tiles,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        CAUSAL=causal,
        TRIM_CAUSAL_LOOP=trim_causal_loop,
        ROUTE_SLOTS=route_slots,
        ROUTE_BLOCK_SIZE=route_block_size,
        **launch_options,
    )
    return output, lse


def _flash_selected_kv_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    grad_output: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    query_positions: torch.Tensor,
    key_position_offset: int,
    query_tile_size: int,
) -> tuple:
    """Use FlashAttention varlen as a bounded CSR backward kernel.

    Each query row is represented as a one-token Q segment and its selected
    K/V rows form the corresponding variable-length segment.  The packed
    gradients are scattered back once per query tile, so the implementation
    avoids the scalar dK/dV atomic update for every head lane in the native
    Triton fallback.  LSE gradients intentionally use the native path because
    the model does not consume the diagnostic LSE output.
    """

    if not FLASH_ATTN_AVAILABLE:
        raise RuntimeError("FlashAttention varlen is not available")
    sq, batch, num_q_heads, head_dim = query.shape
    sk, _, num_kv_heads, _ = key.shape
    k_slots = topk_indices.shape[-1]
    if k_slots <= 0:
        raise ValueError("QSA selected-KV requires at least one index slot")
    query_tile_size = max(1, int(query_tile_size))
    heads_per_kv = num_q_heads // num_kv_heads
    query_bshd = query.transpose(0, 1).contiguous()
    key_bshd = key.transpose(0, 1).contiguous()
    value_bshd = value.transpose(0, 1).contiguous()
    grad_output_bshd = grad_output.transpose(0, 1).contiguous()
    grad_query_bshd = torch.zeros_like(query_bshd) if query.requires_grad else None
    grad_key_bshd = torch.zeros_like(key_bshd) if key.requires_grad else None
    grad_value_bshd = torch.zeros_like(value_bshd) if value.requires_grad else None
    batch_key_stride = sk
    key_flat = None if grad_key_bshd is None else grad_key_bshd.view(batch * sk, num_kv_heads, head_dim)
    value_flat = None if grad_value_bshd is None else grad_value_bshd.view(batch * sk, num_kv_heads, head_dim)
    slot_offsets = torch.arange(k_slots, device=query.device, dtype=torch.long)
    for query_start in range(0, sq, query_tile_size):
        query_end = min(sq, query_start + query_tile_size)
        tile_queries = query_end - query_start
        indices = topk_indices[:, query_start:query_end].to(torch.long)
        lengths = topk_length[:, query_start:query_end].to(torch.long).clamp(min=0, max=k_slots)
        safe_indices = indices.clamp(min=0, max=sk - 1)
        valid = slot_offsets.view(1, 1, -1) < lengths.unsqueeze(-1)
        valid = valid & (indices >= 0) & (indices < sk)
        if causal:
            valid = valid & (
                indices + int(key_position_offset)
                <= query_positions[query_start:query_end].view(1, -1, 1)
            )
        empty_rows = ~valid.any(dim=-1)
        packed_valid = valid.clone()
        if bool(empty_rows.any()):
            packed_valid[..., 0] = packed_valid[..., 0] | empty_rows
        packed_valid_flat = packed_valid.reshape(batch * tile_queries, k_slots)
        packed_rows, packed_slots = torch.nonzero(packed_valid_flat, as_tuple=True)
        packed_lengths = packed_valid_flat.sum(dim=-1, dtype=torch.int32)
        cu_k = torch.cat(
            (
                torch.zeros(1, device=query.device, dtype=torch.int32),
                packed_lengths.cumsum(0).to(torch.int32),
            )
        )
        total_selected = int(cu_k[-1].item())
        if total_selected >= 2**31:
            raise RuntimeError("QSA FlashAttention CSR tile exceeds int32 cu_seqlens capacity")
        flat_indices = safe_indices.reshape(batch * tile_queries, k_slots)
        flat_batch_rows = packed_rows // tile_queries
        packed_token_ids = flat_indices[packed_rows, packed_slots]
        packed_batch_rows = flat_batch_rows
        packed_q = query_bshd[:, query_start:query_end].reshape(batch * tile_queries, num_q_heads, head_dim)
        packed_k = key_bshd[packed_batch_rows, packed_token_ids]
        packed_v = value_bshd[packed_batch_rows, packed_token_ids]
        packed_q = packed_q.detach().requires_grad_(query.requires_grad)
        packed_k = packed_k.detach().requires_grad_(key.requires_grad)
        packed_v = packed_v.detach().requires_grad_(value.requires_grad)
        cu_q = torch.arange(batch * tile_queries + 1, device=query.device, dtype=torch.int32)
        max_k = int(packed_lengths.max().item())
        with torch.enable_grad():
            packed_out = flash_attn_varlen_func(
                packed_q,
                packed_k,
                packed_v,
                cu_q,
                cu_k,
                1,
                max_k,
                softmax_scale=softmax_scale,
                causal=False,
                pack_gqa=True,
            )
            grad_out = grad_output_bshd[:, query_start:query_end].reshape(
                batch * tile_queries, num_q_heads, head_dim
            )
            if bool(empty_rows.any()):
                grad_out = grad_out.clone()
                grad_out[empty_rows.reshape(-1)] = 0
            grad_inputs = []
            if query.requires_grad:
                grad_inputs.append(packed_q)
            if key.requires_grad:
                grad_inputs.append(packed_k)
            if value.requires_grad:
                grad_inputs.append(packed_v)
            grads = torch.autograd.grad(
                packed_out,
                grad_inputs,
                grad_outputs=grad_out,
                allow_unused=True,
            ) if grad_inputs else ()
        grad_iter = iter(grads)
        if grad_query_bshd is not None:
            grad_query_bshd[:, query_start:query_end] = next(grad_iter).reshape(
                batch, tile_queries, num_q_heads, head_dim
            )
        if key_flat is not None:
            grad_key_flat = next(grad_iter)
            flat_key_ids = packed_batch_rows * batch_key_stride + packed_token_ids
            key_flat.index_add_(0, flat_key_ids, grad_key_flat)
        if value_flat is not None:
            grad_value_flat = next(grad_iter)
            flat_value_ids = packed_batch_rows * batch_key_stride + packed_token_ids
            value_flat.index_add_(0, flat_value_ids, grad_value_flat)

    grad_query_out = None if grad_query_bshd is None else grad_query_bshd.transpose(0, 1).contiguous()
    grad_key_out = None if grad_key_bshd is None else grad_key_bshd.transpose(0, 1).contiguous()
    grad_value_out = None if grad_value_bshd is None else grad_value_bshd.transpose(0, 1).contiguous()
    return grad_query_out, grad_key_out, grad_value_out


def qsa_prepare_segmented_metadata(
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    query_positions: torch.Tensor,
    seq_len_q: int,
    seq_len_k: int,
    ratio: int,
):
    """Build the reusable inverse CSR map for QSA complete blocks."""

    if ratio < 2 or (ratio & (ratio - 1)):
        raise ValueError(
            "QSA segmented dK/dV reduction requires a power-of-two ratio >= 2")
    batch, sq, k_slots = topk_indices.shape
    block_topk_numerator = k_slots - (ratio - 1)
    if block_topk_numerator <= 0 or block_topk_numerator % ratio:
        raise ValueError(
            "QSA segmented dK/dV reduction requires token slots to equal "
            "block_topk*ratio + ratio - 1")
    block_topk = block_topk_numerator // ratio
    if block_topk <= 0 or (block_topk & (block_topk - 1)):
        raise ValueError(
            "QSA segmented dK/dV reduction currently requires a power-of-two block_topk")
    num_blocks = triton.cdiv(seq_len_k, ratio)
    counts = torch.zeros(
        (batch * num_blocks,), device=topk_indices.device, dtype=torch.int32
    )
    _qsa_segment_count_blocks_kernel[(batch * sq,)](
        topk_indices,
        topk_length,
        query_positions,
        counts,
        seq_len_q,
        seq_len_k,
        num_blocks,
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        topk_length.stride(0),
        topk_length.stride(1),
        BLOCK_TOPK=block_topk,
        RATIO=ratio,
        num_warps=4,
        num_stages=1,
    )
    # QSA training shapes stay below int32 occurrence capacity.  Keeping the
    # offsets in int32 avoids the int64 sort workspace that this inverse map
    # was introduced to remove.
    starts = torch.empty(
        (counts.numel() + 1,), device=topk_indices.device, dtype=torch.int32
    )
    starts[0] = 0
    starts[1:] = counts.cumsum(0).to(torch.int32)
    cursor = starts[:-1].clone()
    occurrences = torch.empty(
        (batch * sq * block_topk,), device=topk_indices.device, dtype=torch.int32
    )
    _qsa_segment_fill_blocks_kernel[(batch * sq,)](
        topk_indices,
        topk_length,
        query_positions,
        cursor,
        occurrences,
        seq_len_q,
        seq_len_k,
        num_blocks,
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        topk_length.stride(0),
        topk_length.stride(1),
        BLOCK_TOPK=block_topk,
        RATIO=ratio,
        num_warps=4,
        num_stages=1,
    )
    return occurrences, starts, block_topk, num_blocks


def qsa_segmented_dkv_tail(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor,
    correction: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    softmax_scale: float,
    query_positions: torch.Tensor,
    ratio: int,
    block_topk: int,
) -> None:
    """Atomically add only the incomplete causal block for segmented dK/dV."""

    sq, batch, num_q_heads, head_dim = query.shape
    _, _, num_kv_heads, _ = key.shape
    group_size = num_q_heads // num_kv_heads
    head_tile_size = int(os.environ.get('MCORE_BRIDGE_QSA_BACKWARD_HEAD_TILE', '4'))
    if head_tile_size not in {1, 2, 4, 8, 16}:
        raise ValueError('QSA segmented tail head tile expects one of {1,2,4,8,16}')
    num_head_tiles = triton.cdiv(group_size, head_tile_size)
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_tail = max(1, triton.next_power_of_2(ratio - 1))
    num_warps = int(os.environ.get('MCORE_BRIDGE_QSA_BACKWARD_WARPS', '1'))
    num_stages = int(os.environ.get('MCORE_BRIDGE_QSA_BACKWARD_STAGES', '1'))
    use_tail_dispatch = os.environ.get(
        'MCORE_BRIDGE_QSA_SEGMENT_TAIL_DISPATCH', '0') != '0'
    if use_tail_dispatch:
        # Only rows with a non-empty incomplete causal block can contribute to
        # this launch.  Avoiding the 25% empty rows for R=4 removes unnecessary
        # Q/grad-output loads from the segmented path while keeping the
        # complete block reducer unchanged.
        tail_mask = (query_positions.to(torch.long) + 1).remainder(ratio) > 0
        if not bool(tail_mask.any().item()):
            return
        row_ids = torch.arange(
            batch * sq, device=query.device, dtype=torch.int32).reshape(batch, sq)
        tail_token_ids = row_ids[
            tail_mask.unsqueeze(0).expand(batch, -1)
        ].contiguous()
        tail_grid_rows = tail_token_ids.numel()
        tail_token_ids_arg = tail_token_ids
    else:
        tail_grid_rows = batch * sq
        # The no-dispatch diagnostic uses the original flattened launch; the
        # pointer is compile-time ignored when USE_TOKEN_IDS is false.
        tail_token_ids_arg = query_positions
    _qsa_selected_kv_backward_tail_grouped_kernel[
        (tail_grid_rows, num_kv_heads * num_head_tiles)
    ](
        query,
        key,
        value,
        topk_indices,
        topk_length,
        lse,
        grad_output,
        grad_lse,
        correction,
        grad_key,
        grad_value,
        sq,
        key.shape[0],
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        0,
        query_positions,
        tail_token_ids_arg,
        query_positions.stride(0),
        query.stride(1),
        query.stride(0),
        query.stride(2),
        query.stride(3),
        key.stride(1),
        key.stride(0),
        key.stride(2),
        key.stride(3),
        value.stride(1),
        value.stride(0),
        value.stride(2),
        value.stride(3),
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        topk_length.stride(0),
        topk_length.stride(1),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        grad_output.stride(1),
        grad_output.stride(0),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_lse.stride(0),
        grad_lse.stride(1),
        grad_lse.stride(2),
        correction.stride(0),
        correction.stride(1),
        correction.stride(2),
        grad_key.stride(1),
        grad_key.stride(0),
        grad_key.stride(2),
        grad_key.stride(3),
        grad_value.stride(1),
        grad_value.stride(0),
        grad_value.stride(2),
        grad_value.stride(3),
        K=topk_indices.shape[-1],
        HEADS_PER_KV=head_tile_size,
        GROUP_SIZE=group_size,
        NUM_HEAD_TILES=num_head_tiles,
        BLOCK_TAIL=block_tail,
        BLOCK_D=block_d,
        SEGMENT_BLOCK_TOPK=block_topk,
        RATIO=ratio,
        CAUSAL=True,
        HAS_GRAD_OUTPUT=True,
        HAS_GRAD_LSE=grad_lse is not lse,
        DKV_ACCUM_BF16=grad_key.dtype == torch.bfloat16,
        USE_TOKEN_IDS=use_tail_dispatch,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def qsa_segmented_dkv_reduce(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor,
    correction: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    softmax_scale: float,
    query_positions: torch.Tensor,
    ratio: int,
    segmented_metadata=None,
) -> None:
    """Build a block-level inverse CSR map and reduce dK/dV by segment.

    QSA selects complete ``ratio``-token blocks.  Inverting those block IDs
    reduces the occurrence metadata and the reduction launch domain by
    ``ratio`` compared with sorting every selected token.  The caller handles
    the at-most ``ratio - 1`` causal-tail tokens with the regular atomic path.
    This function writes into already-zeroed ``grad_key`` and ``grad_value``.
    """

    if ratio < 2 or (ratio & (ratio - 1)):
        raise ValueError(
            "QSA segmented dK/dV reduction requires a power-of-two ratio >= 2")
    sq, batch, num_q_heads, head_dim = query.shape
    sk, key_batch, num_kv_heads, value_dim = key.shape
    k_slots = topk_indices.shape[-1]
    block_topk_numerator = k_slots - (ratio - 1)
    if block_topk_numerator <= 0 or block_topk_numerator % ratio:
        raise ValueError(
            "QSA segmented dK/dV reduction requires token slots to equal "
            "block_topk*ratio + ratio - 1")
    block_topk = block_topk_numerator // ratio
    if block_topk <= 0 or (block_topk & (block_topk - 1)):
        raise ValueError(
            "QSA segmented dK/dV reduction currently requires a power-of-two block_topk")
    if key_batch != batch or value.shape != key.shape or value_dim != head_dim:
        raise ValueError("QSA segmented dK/dV reduction received incompatible Q/K/V shapes")
    if topk_indices.shape[:2] != (batch, sq) or topk_length.shape != (batch, sq):
        raise ValueError("QSA segmented dK/dV reduction received incompatible index shapes")
    if segmented_metadata is None:
        segmented_metadata = qsa_prepare_segmented_metadata(
            topk_indices, topk_length, query_positions, sq, sk, ratio
        )
    occurrences, starts, block_topk, num_blocks = segmented_metadata
    group_size = num_q_heads // num_kv_heads
    block_d = max(16, triton.next_power_of_2(head_dim))
    # Real score-selected routes have a heavier occurrence tail than the
    # uniform synthetic fixture; BLOCK_OCC=16 is the stable SM90 compromise.
    block_occ = int(os.environ.get('MCORE_BRIDGE_QSA_SEGMENT_BLOCK_OCC', '16'))
    if block_occ not in {16, 32, 64}:
        raise ValueError('QSA segmented BLOCK_OCC expects one of {16,32,64}')
    flatten_default = (
        is_sm90(query.device)
        and grad_key.dtype == torch.bfloat16
        and ratio == 4
        and head_dim == 256
        and group_size >= 2
    )
    flatten_heads = os.environ.get(
        'MCORE_BRIDGE_QSA_SEGMENT_FLATTEN_HEADS',
        '1' if flatten_default else '0',
    ) != '0'
    default_segment_warps = 2 if flatten_heads else 4
    segmented_num_warps = int(
        os.environ.get(
            'MCORE_BRIDGE_QSA_SEGMENT_WARPS', str(default_segment_warps)))
    if segmented_num_warps not in {1, 2, 4, 8}:
        raise ValueError('QSA segmented reducer warps expects one of {1,2,4,8}')
    default_segment_head_tile = 2 if flatten_heads else group_size
    segment_head_tile = int(os.environ.get(
        'MCORE_BRIDGE_QSA_SEGMENT_HEAD_TILE', str(default_segment_head_tile)))
    if segment_head_tile not in {1, 2, 3, 4, 6, 8, 12, 16}:
        raise ValueError(
            'QSA segmented head tile expects one of {1,2,3,4,6,8,12,16}')
    num_segment_head_tiles = triton.cdiv(group_size, segment_head_tile)
    if flatten_heads and segment_head_tile not in {1, 2, 4, 8, 16}:
        raise ValueError(
            'QSA flattened segmented reducer requires a power-of-two head tile')
    if flatten_heads and grad_key.dtype != torch.bfloat16:
        raise ValueError(
            'QSA flattened segmented reducer currently requires BF16 dK/dV accumulation')
    grid = (batch * num_blocks * num_kv_heads * num_segment_head_tiles,)
    kernel_args = (
        query,
        key,
        value,
        lse,
        grad_output,
        grad_lse,
        correction,
        occurrences,
        starts,
        grad_key,
        grad_value,
        sq,
        sk,
        num_blocks,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        query.stride(1),
        query.stride(0),
        query.stride(2),
        query.stride(3),
        key.stride(1),
        key.stride(0),
        key.stride(2),
        key.stride(3),
        value.stride(1),
        value.stride(0),
        value.stride(2),
        value.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        grad_output.stride(1),
        grad_output.stride(0),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_lse.stride(0),
        grad_lse.stride(1),
        grad_lse.stride(2),
        correction.stride(0),
        correction.stride(1),
        correction.stride(2),
        grad_key.stride(1),
        grad_key.stride(0),
        grad_key.stride(2),
        grad_key.stride(3),
        grad_value.stride(1),
        grad_value.stride(0),
        grad_value.stride(2),
        grad_value.stride(3),
    )
    kernel_options = {
        'RATIO': ratio,
        'GROUP_SIZE': group_size,
        'BLOCK_H': max(1, triton.next_power_of_2(group_size)),
        'HEAD_TILE': segment_head_tile,
        'NUM_HEAD_TILES': num_segment_head_tiles,
        'BLOCK_OCC': block_occ,
        'BLOCK_D': block_d,
        'HAS_GRAD_OUTPUT': True,
        'HAS_GRAD_LSE': grad_lse is not lse,
        'DKV_ACCUM_BF16': grad_key.dtype == torch.bfloat16,
        'BATCH_ONE': batch == 1,
        'SPLIT_HEADS': num_segment_head_tiles > 1,
        'num_warps': segmented_num_warps,
        'num_stages': 1,
    }
    if flatten_heads:
        _qsa_segmented_dkv_reduce_flattened_kernel[grid](
            *kernel_args,
            BLOCK_M=block_occ * segment_head_tile,
            **kernel_options,
        )
    else:
        _qsa_segmented_dkv_reduce_grouped_kernel[grid](
            *kernel_args,
            **kernel_options,
        )


def qsa_selected_kv_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor = None,
    grad_lse: torch.Tensor = None,
    softmax_scale: float = 1.0,
    causal: bool = True,
    query_positions: torch.Tensor = None,
    key_position_offset: int = 0,
    query_tile_size: int = 1024,
    dkv_accum_dtype: str = 'bf16',
    selected_token_group_size: Optional[int] = None,
    segmented_metadata=None,
    dkv_reduction: str = 'atomic',
    output: torch.Tensor = None,
    route_block_size: int = 1,
) -> tuple:
    """Launch the Triton selected-KV backward kernel.

    dK/dV accumulation is explicit: ``bf16`` reduces atomic traffic for the
    BF16 training path, while ``fp32`` retains the higher-precision reference.
    When ``output`` is supplied, the default path obtains the softmax
    correction from ``dot(grad_output, output)``; set
    ``MCORE_BRIDGE_QSA_BACKWARD_OUTPUT_DELTA=0`` to recompute it from K/V.
    Additive dK/dV atomics use relaxed memory ordering because no consumer
    observes a partial gradient before the kernel completes.
    ``MCORE_BRIDGE_QSA_DKV_REDUCTION=segmented`` enables the QSA production
    block-level inverse-CSR reduction when ``selected_token_group_size`` is
    supplied; the causal tail remains on a small atomic path.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("QSA Triton kernels require triton to be installed")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise ValueError("QSA Triton attention requires CUDA tensors")
    sq, batch, num_q_heads, head_dim = query.shape
    sk, key_batch, num_kv_heads, value_dim = key.shape
    if key_batch != batch or value.shape != key.shape or value_dim != head_dim:
        raise ValueError(
            f"QSA selected-KV backward shape mismatch: query={tuple(query.shape)}, key={tuple(key.shape)}, "
            f"value={tuple(value.shape)}")
    if num_q_heads % num_kv_heads:
        raise ValueError(f"QSA GQA requires Hq divisible by Hkv, got {num_q_heads} and {num_kv_heads}")
    if topk_indices.shape[:2] != (batch, sq) or topk_length.shape != (batch, sq):
        raise ValueError(
            f"QSA selected-KV backward index shape mismatch: indices={tuple(topk_indices.shape)}, "
            f"length={tuple(topk_length.shape)}, query={tuple(query.shape)}")
    route_slots = topk_indices.shape[-1]
    route_block_size = int(route_block_size)
    if route_slots <= 0 or route_block_size <= 0:
        raise ValueError("QSA backward route slots/block size must be positive")
    if route_block_size > 1 and (not causal or int(key_position_offset) != 0):
        raise ValueError(
            "QSA compact block route backward requires causal attention and key_position_offset=0")
    logical_k = (
        route_slots
        if route_block_size == 1
        else route_slots * route_block_size + route_block_size - 1
    )
    if lse.shape != (batch, num_q_heads, sq):
        raise ValueError(f"QSA LSE shape must be {(batch, num_q_heads, sq)}, got {tuple(lse.shape)}")
    if query_positions is None:
        query_positions = torch.arange(sq, device=query.device, dtype=torch.int32)
    query_positions = query_positions.to(device=query.device, dtype=torch.int32).contiguous()
    if query_positions.shape != (sq, ):
        raise ValueError(f"QSA query_positions must have shape [{sq}], got {tuple(query_positions.shape)}")
    topk_indices = topk_indices.to(torch.int32).contiguous()
    topk_length = topk_length.to(torch.int32).contiguous()
    dkv_accum_dtype = str(dkv_accum_dtype).lower()
    if dkv_accum_dtype not in {'bf16', 'fp32'}:
        raise ValueError(f'unsupported QSA dkv_accum_dtype={dkv_accum_dtype!r}; choose bf16 or fp32')
    dkv_reduction = str(dkv_reduction).lower()
    if dkv_reduction not in {'atomic', 'segmented'}:
        raise ValueError(
            f'unsupported QSA dkv_reduction={dkv_reduction!r}; choose atomic or segmented')
    # The CSR FlashAttention route is useful for experiments, but its packed
    # gather/scatter must be measured against the native kernel per image.
    # Keep native Triton as the production default; opt in explicitly when
    # profiling a FlashAttention-enabled training image.
    use_flash_backward = os.environ.get("MCORE_BRIDGE_QSA_FLASH_BACKWARD", "0") == "1"
    if (use_flash_backward and route_block_size == 1 and FLASH_ATTN_AVAILABLE and dkv_accum_dtype == 'bf16'
            and grad_output is not None and grad_lse is None):
        return _flash_selected_kv_backward(
            query,
            key,
            value,
            topk_indices,
            topk_length,
            grad_output,
            softmax_scale,
            causal,
            query_positions,
            key_position_offset,
            query_tile_size,
        )
    grad_output_present = grad_output is not None
    grad_lse_present = grad_lse is not None
    use_output_delta = (
        output is not None
        and os.environ.get('MCORE_BRIDGE_QSA_BACKWARD_OUTPUT_DELTA', '1') != '0'
    )
    if output is not None and output.shape != query.shape:
        raise ValueError(
            f'QSA backward output must match query shape {tuple(query.shape)}, got {tuple(output.shape)}')
    if output is None:
        # The pointer is never read when USE_OUTPUT_DELTA is false.
        output = query
    else:
        output = output.contiguous()
    segment_reduction_requested = os.environ.get(
        "MCORE_BRIDGE_QSA_DKV_REDUCTION", dkv_reduction
    ).lower() == "segmented"
    if route_block_size > 1 and segment_reduction_requested:
        raise RuntimeError(
            'QSA compact block route currently requires atomic dK/dV reduction')
    segment_ratio = None
    use_segmented_reduction = False
    if segment_reduction_requested and selected_token_group_size is not None:
        segment_ratio = int(selected_token_group_size)
        block_topk_numerator = topk_indices.shape[-1] - (segment_ratio - 1)
        supported = (
            causal
            and key_position_offset == 0
            and grad_output_present
            and query.dtype == torch.bfloat16
            and key.dtype == torch.bfloat16
            and value.dtype == torch.bfloat16
            and segment_ratio >= 2
            and (segment_ratio & (segment_ratio - 1)) == 0
            and block_topk_numerator > 0
            and block_topk_numerator % segment_ratio == 0
            and (
                (block_topk_numerator // segment_ratio) &
                ((block_topk_numerator // segment_ratio) - 1)
            ) == 0
        )
        if not supported:
            raise RuntimeError(
                "QSA segmented dK/dV reduction requires causal BF16 Q/K/V, "
                "key_position_offset=0, a supplied grad_output, a "
                "power-of-two compression ratio, and a power-of-two "
                "block_topk token layout"
            )
        use_segmented_reduction = True
    if grad_output is None:
        # The constexpr mask prevents any reads from this dummy pointer.
        grad_output = query
    else:
        grad_output = grad_output.contiguous()
    if grad_lse is None:
        grad_lse = lse
    else:
        grad_lse = grad_lse.contiguous()
    grad_query = torch.empty_like(query, dtype=torch.float32)
    # dQ and the per-program reductions remain FP32.  The output is cast back
    # to the input dtype before returning through autograd.
    grad_key = torch.zeros_like(key) if dkv_accum_dtype == 'bf16' else torch.zeros_like(key, dtype=torch.float32)
    grad_value = torch.zeros_like(value) if dkv_accum_dtype == 'bf16' else torch.zeros_like(value, dtype=torch.float32)
    correction = (
        torch.empty((batch, num_q_heads, sq), device=query.device, dtype=torch.float32)
        if use_segmented_reduction else lse
    )
    group_size = num_q_heads // num_kv_heads
    tensorized_default = (
        group_size >= 5
        and head_dim >= 64
        and dkv_accum_dtype == 'bf16'
    )
    default_head_tile = 16 if tensorized_default else min(
        4, 1 << max(0, group_size - 1).bit_length())
    head_tile_size = int(os.environ.get(
        'MCORE_BRIDGE_QSA_BACKWARD_HEAD_TILE', str(default_head_tile)))
    if head_tile_size not in {1, 2, 4, 8, 16}:
        raise ValueError('QSA backward head tile expects one of {1,2,4,8,16}')
    tensorized_tile = (
        head_tile_size >= 16
        and head_dim >= 64
        and dkv_accum_dtype == 'bf16'
    )
    num_head_tiles = triton.cdiv(group_size, head_tile_size)
    grid = (batch * sq, num_kv_heads * num_head_tiles)
    block_d = max(16, triton.next_power_of_2(head_dim))
    trim_causal_loop = _qsa_trim_causal_loop(causal, sq, logical_k)
    hopper_short_compact = (
        tensorized_tile
        and head_tile_size == 16
        and group_size == 12
        and head_dim == 256
        and route_block_size == 4
        and trim_causal_loop
        and sq <= 8 * logical_k
        and is_sm90(query.device)
    )
    # K32 is the stable Hopper derivative tile.  The former short-sequence
    # K64 path reduced loop overhead but required four warps and generated a
    # wider burst of reductions into hot KV rows.  A two-warp K32 launch is
    # faster when the causal runtime bound is active and retains K32's
    # bounded derivative/atomic tile.  Token-route compatibility and other
    # shapes retain the previously tuned K64 short-prefix dispatch.
    default_block_k = (
        32
        if hopper_short_compact
        else 64
        if (
            tensorized_tile
            and head_dim == 256
            and is_sm90(query.device)
            and sq <= 8 * logical_k
        )
        else 32 if tensorized_tile else 8
    )
    block_k = int(os.environ.get(
        'MCORE_BRIDGE_QSA_BACKWARD_BLOCK_K', str(default_block_k)))
    if block_k not in {4, 8, 16, 32, 64, 128}:
        raise ValueError('QSA backward BLOCK_K expects one of {4,8,16,32,64,128}')
    correction_block_k = int(os.environ.get(
        'MCORE_BRIDGE_QSA_BACKWARD_CORRECTION_BLOCK_K', '32'))
    if correction_block_k not in {4, 8, 16, 32, 64, 128}:
        raise ValueError(
            'QSA backward correction BLOCK_K expects one of {4,8,16,32,64,128}')
    tensorize_derivatives = tensorized_tile and block_k >= 16
    hopper_short_k32 = (
        hopper_short_compact
        and tensorize_derivatives
        and block_k == 32
    )
    default_num_warps = (
        2 if hopper_short_k32 else 4 if tensorize_derivatives else 1
    )
    backward_num_warps = int(os.environ.get(
        'MCORE_BRIDGE_QSA_BACKWARD_WARPS', str(default_num_warps)))
    default_num_stages = 2 if tensorize_derivatives else 1
    backward_num_stages = int(os.environ.get(
        'MCORE_BRIDGE_QSA_BACKWARD_STAGES', str(default_num_stages)))
    if backward_num_warps not in {1, 2, 4, 8} or backward_num_stages not in {1, 2, 3, 4}:
        raise ValueError(
            'QSA backward tuning expects warps in {1,2,4,8} and stages in {1,2,3,4}')
    _qsa_selected_kv_backward_grouped_kernel[grid](
        query,
        key,
        value,
        topk_indices,
        topk_length,
        lse,
        grad_output,
        output,
        grad_lse,
        grad_query,
        grad_key,
        grad_value,
        sq,
        sk,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        key_position_offset,
        query_positions,
        query_positions.stride(0),
        query.stride(1),
        query.stride(0),
        query.stride(2),
        query.stride(3),
        key.stride(1),
        key.stride(0),
        key.stride(2),
        key.stride(3),
        value.stride(1),
        value.stride(0),
        value.stride(2),
        value.stride(3),
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        topk_length.stride(0),
        topk_length.stride(1),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        grad_output.stride(1),
        grad_output.stride(0),
        grad_output.stride(2),
        grad_output.stride(3),
        output.stride(1),
        output.stride(0),
        output.stride(2),
        output.stride(3),
        grad_lse.stride(0),
        grad_lse.stride(1),
        grad_lse.stride(2),
        grad_query.stride(1),
        grad_query.stride(0),
        grad_query.stride(2),
        grad_query.stride(3),
        grad_key.stride(1),
        grad_key.stride(0),
        grad_key.stride(2),
        grad_key.stride(3),
        grad_value.stride(1),
        grad_value.stride(0),
        grad_value.stride(2),
        grad_value.stride(3),
        correction,
        correction.stride(0),
        correction.stride(1),
        correction.stride(2),
        K=logical_k,
        HEADS_PER_KV=head_tile_size,
        GROUP_SIZE=group_size,
        NUM_HEAD_TILES=num_head_tiles,
        BLOCK_K=block_k,
        CORRECTION_BLOCK_K=correction_block_k,
        BLOCK_D=block_d,
        CAUSAL=causal,
        HAS_GRAD_OUTPUT=grad_output_present,
        HAS_GRAD_LSE=grad_lse_present,
        USE_OUTPUT_DELTA=use_output_delta,
        DKV_ACCUM_BF16=dkv_accum_dtype == 'bf16',
        SEGMENT_BLOCK_TOPK=(
            (topk_indices.shape[-1] - (segment_ratio - 1)) // segment_ratio
            if use_segmented_reduction else 1
        ),
        RATIO=segment_ratio if use_segmented_reduction else 2,
        STORE_CORRECTION=use_segmented_reduction,
        TAIL_ONLY=use_segmented_reduction,
        # Segmented mode computes complete-block dK/dV exactly once in the
        # inverse-CSR reducer.  Its tiny causal tail is emitted by the
        # dedicated tail launch below; keeping EMIT_DKV false here removes
        # the otherwise duplicated complete-block reductions.
        EMIT_DKV=not use_segmented_reduction,
        TENSORIZE_DERIVATIVES=tensorize_derivatives,
        TRIM_CAUSAL_LOOP=trim_causal_loop,
        ROUTE_SLOTS=route_slots,
        ROUTE_BLOCK_SIZE=route_block_size,
        num_warps=backward_num_warps,
        num_stages=backward_num_stages,
    )
    if use_segmented_reduction:
        qsa_segmented_dkv_tail(
            query,
            key,
            value,
            topk_indices,
            topk_length,
            lse,
            grad_output,
            grad_lse,
            correction,
            grad_key,
            grad_value,
            softmax_scale,
            query_positions,
            segment_ratio,
            (topk_indices.shape[-1] - (segment_ratio - 1)) // segment_ratio,
        )
        qsa_segmented_dkv_reduce(
            query,
            key,
            value,
            topk_indices,
            topk_length,
            lse,
            grad_output,
            grad_lse,
            correction,
            grad_key,
            grad_value,
            softmax_scale,
            query_positions,
            segment_ratio,
            segmented_metadata=segmented_metadata,
        )
    return grad_query.to(query.dtype), grad_key.to(key.dtype), grad_value.to(value.dtype)


def qsa_selected_kv_backward_packed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    lse: torch.Tensor,
    key_starts: torch.Tensor,
    key_lengths: torch.Tensor,
    query_positions: torch.Tensor,
    grad_output: torch.Tensor = None,
    grad_lse: torch.Tensor = None,
    softmax_scale: float = 1.0,
    causal: bool = True,
    key_position_offset: int = 0,
    dkv_accum_dtype: str = 'bf16',
    dkv_reduction: str = 'atomic',
    output: torch.Tensor = None,
    route_block_size: int = 1,
) -> tuple:
    """Launch one packed-THD selected-KV backward with relaxed dK/dV atomics.

    ``output`` enables the same allocation-free output-delta correction used
    by the unpacked launcher.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("QSA Triton kernels require triton to be installed")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise ValueError("QSA packed Triton attention requires CUDA tensors")
    if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("QSA packed Triton backward expects Q/K/V as [T,H,D]")
    total_q, num_q_heads, head_dim = query.shape
    total_k, num_kv_heads, value_dim = key.shape
    if value_dim != head_dim or num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        raise ValueError("QSA packed Triton backward has incompatible Q/KV heads")
    if topk_indices.ndim != 2 or topk_indices.shape[0] != total_q:
        raise ValueError("QSA packed Triton backward indices must be [T,K]")
    if topk_length.shape != (total_q,):
        raise ValueError("QSA packed Triton backward lengths must be [T]")
    route_slots = topk_indices.shape[1]
    route_block_size = int(route_block_size)
    if route_slots <= 0 or route_block_size <= 0:
        raise ValueError(
            "QSA packed backward route slots/block size must be positive")
    if route_block_size > 1 and (not causal or int(key_position_offset) != 0):
        raise ValueError(
            "QSA packed compact route backward requires causal attention and key_position_offset=0")
    logical_k = (
        route_slots
        if route_block_size == 1
        else route_slots * route_block_size + route_block_size - 1
    )
    if lse.shape != (num_q_heads, total_q):
        raise ValueError("QSA packed Triton backward LSE must be [H,T]")
    for name, tensor in (
        ('key_starts', key_starts),
        ('key_lengths', key_lengths),
        ('query_positions', query_positions),
    ):
        if tensor.shape != (total_q,):
            raise ValueError(f"QSA packed Triton backward {name} must be [{total_q}]")
    if str(dkv_reduction).lower() != 'atomic':
        raise RuntimeError(
            "QSA packed varlen backward currently supports only dkv_reduction='atomic'")
    dkv_accum_dtype = str(dkv_accum_dtype).lower()
    if dkv_accum_dtype not in {'bf16', 'fp32'}:
        raise ValueError('QSA packed backward dkv_accum_dtype expects bf16 or fp32')
    if total_q == 0:
        return (
            query.new_empty(query.shape),
            key.new_zeros(key.shape) if dkv_accum_dtype == 'bf16' else key.new_zeros(key.shape, dtype=torch.float32),
            value.new_zeros(value.shape) if dkv_accum_dtype == 'bf16' else value.new_zeros(value.shape, dtype=torch.float32),
        )
    topk_indices = topk_indices.to(device=query.device, dtype=torch.int32).contiguous()
    topk_length = topk_length.to(device=query.device, dtype=torch.int32).contiguous()
    key_starts = key_starts.to(device=query.device, dtype=torch.int32).contiguous()
    key_lengths = key_lengths.to(device=query.device, dtype=torch.int32).contiguous()
    query_positions = query_positions.to(device=query.device, dtype=torch.int32).contiguous()
    lse = lse.contiguous()
    grad_output_present = grad_output is not None
    grad_lse_present = grad_lse is not None
    use_output_delta = (
        output is not None
        and os.environ.get('MCORE_BRIDGE_QSA_BACKWARD_OUTPUT_DELTA', '1') != '0'
    )
    if output is not None and output.shape != query.shape:
        raise ValueError(
            f'QSA packed backward output must match query shape {tuple(query.shape)}, got {tuple(output.shape)}')
    if output is None:
        output = query
    else:
        output = output.contiguous()
    if grad_output is None:
        grad_output = query
    else:
        grad_output = grad_output.contiguous()
    if grad_lse is None:
        grad_lse = lse
    else:
        grad_lse = grad_lse.contiguous()
    grad_query = torch.empty_like(query, dtype=torch.float32)
    grad_key = (
        torch.zeros_like(key)
        if dkv_accum_dtype == 'bf16'
        else torch.zeros_like(key, dtype=torch.float32)
    )
    grad_value = (
        torch.zeros_like(value)
        if dkv_accum_dtype == 'bf16'
        else torch.zeros_like(value, dtype=torch.float32)
    )
    group_size = num_q_heads // num_kv_heads
    tensorized_default = (
        group_size >= 5 and head_dim >= 64 and dkv_accum_dtype == 'bf16'
    )
    default_head_tile = 16 if tensorized_default else min(
        4, 1 << max(0, group_size - 1).bit_length())
    head_tile_size = int(os.environ.get(
        'MCORE_BRIDGE_QSA_BACKWARD_HEAD_TILE', str(default_head_tile)))
    if head_tile_size not in {1, 2, 4, 8, 16}:
        raise ValueError('QSA packed backward head tile expects one of {1,2,4,8,16}')
    tensorized_tile = (
        head_tile_size >= 16
        and head_dim >= 64
        and dkv_accum_dtype == 'bf16'
    )
    num_head_tiles = triton.cdiv(group_size, head_tile_size)
    block_d = max(16, triton.next_power_of_2(head_dim))
    default_block_k = 32 if tensorized_tile else 8
    block_k = int(os.environ.get(
        'MCORE_BRIDGE_QSA_BACKWARD_BLOCK_K', str(default_block_k)))
    if block_k not in {4, 8, 16, 32, 64, 128}:
        raise ValueError('QSA packed backward BLOCK_K expects one of {4,8,16,32,64,128}')
    correction_block_k = int(os.environ.get(
        'MCORE_BRIDGE_QSA_BACKWARD_CORRECTION_BLOCK_K', '32'))
    if correction_block_k not in {4, 8, 16, 32, 64, 128}:
        raise ValueError(
            'QSA packed backward correction BLOCK_K expects one of {4,8,16,32,64,128}')
    tensorize_derivatives = tensorized_tile and block_k >= 16
    default_num_warps = 4 if tensorize_derivatives else 1
    num_warps = int(os.environ.get(
        'MCORE_BRIDGE_QSA_BACKWARD_WARPS', str(default_num_warps)))
    default_num_stages = 2 if tensorize_derivatives else 1
    num_stages = int(os.environ.get(
        'MCORE_BRIDGE_QSA_BACKWARD_STAGES', str(default_num_stages)))
    if num_warps not in {1, 2, 4, 8} or num_stages not in {1, 2, 3, 4}:
        raise ValueError('QSA packed backward tuning has invalid warps/stages')
    # total_q is a conservative proxy for the longest packed document.  It
    # enables the profitable short-document variant without synchronizing on
    # GPU-resident per-token key lengths.
    trim_causal_loop = _qsa_trim_causal_loop(causal, total_q, logical_k)
    _qsa_selected_kv_backward_packed_grouped_kernel[
        (total_q, num_kv_heads * num_head_tiles)
    ](
        query,
        key,
        value,
        topk_indices,
        topk_length,
        lse,
        grad_output,
        output,
        grad_lse,
        grad_query,
        grad_key,
        grad_value,
        key_starts,
        key_lengths,
        query_positions,
        total_q,
        total_k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        key_position_offset,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_length.stride(0),
        lse.stride(0),
        lse.stride(1),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        grad_lse.stride(0),
        grad_lse.stride(1),
        grad_query.stride(0),
        grad_query.stride(1),
        grad_query.stride(2),
        grad_key.stride(0),
        grad_key.stride(1),
        grad_key.stride(2),
        grad_value.stride(0),
        grad_value.stride(1),
        grad_value.stride(2),
        key_starts.stride(0),
        key_lengths.stride(0),
        query_positions.stride(0),
        K=logical_k,
        HEADS_PER_KV=head_tile_size,
        GROUP_SIZE=group_size,
        NUM_HEAD_TILES=num_head_tiles,
        BLOCK_K=block_k,
        CORRECTION_BLOCK_K=correction_block_k,
        BLOCK_D=block_d,
        CAUSAL=causal,
        HAS_GRAD_OUTPUT=grad_output_present,
        HAS_GRAD_LSE=grad_lse_present,
        USE_OUTPUT_DELTA=use_output_delta,
        DKV_ACCUM_BF16=dkv_accum_dtype == 'bf16',
        TENSORIZE_DERIVATIVES=tensorize_derivatives,
        TRIM_CAUSAL_LOOP=trim_causal_loop,
        ROUTE_SLOTS=route_slots,
        ROUTE_BLOCK_SIZE=route_block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return grad_query.to(query.dtype), grad_key.to(key.dtype), grad_value.to(value.dtype)


__all__ = [
    "TRITON_AVAILABLE",
    "is_sm90",
    "qsa_cp_compact_remap_route_triton",
    "qsa_cp_compact_request_mask_triton",
    "qsa_cp_remap_route_triton",
    "qsa_cp_request_mask_triton",
    "qsa_expand_compact_route_triton",
    "qsa_indexer_fused_postprocess",
    "qsa_indexer_fused_topk_with_ratio",
    "qsa_indexer_fused_topk_packed",
    "qsa_indexer_score_slab_with_ratio",
    "qsa_indexer_score_tile_with_ratio",
    "qsa_indexer_slab_topk_with_ratio",
    "qsa_prepare_segmented_metadata",
    "qsa_segmented_dkv_reduce",
    "qsa_selected_kv_backward",
    "qsa_selected_kv_backward_packed",
    "qsa_selected_kv_forward",
    "qsa_selected_kv_forward_packed",
]
