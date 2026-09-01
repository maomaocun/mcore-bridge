# Copyright (c) ModelScope Contributors. All rights reserved.
"""Qwen4-Exp QSA indexer and selected-token metadata producer."""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
from torch import nn


class Qwen4ExpTextRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, dtype=None, sequence_parallel: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=dtype))
        # Replicated across TP; reduce grads across TP when SP is on.
        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        # zero-centered: (1 + w), and Qwen4ExpText casts after scaling
        return (out * (1.0 + self.weight.float())).type_as(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class QSAIndexer(nn.Module):
    """Produce QSA selected-KV routes without constructing an S-by-S mask.

    ``select_topk`` is the supported selected-KV API.  On the production BF16
    path it dispatches to a device-side packed-key Top-K kernel; ``select_mask`` remains
    as an intentionally expensive compatibility adapter for
    ``qsa_kernel_backend=none`` and old callers.  Both methods share the same
    block score and tie-breaking convention: descending score, then ascending
    block id.  The public token route preserves that canonical order for
    reproducible index dumps.  The compact production route guarantees the
    same exact block set but may retain a deterministic kernel-internal
    permutation because selected-KV attention is invariant to route order.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.index_n_heads = config.indexer_n_heads
        self.index_kv_heads = config.indexer_kv_heads
        self.index_head_dim = config.indexer_head_dim
        self.compress_ratio = config.indexer_compress_ratio
        self.token_budget = config.indexer_budget
        self.block_topk = math.ceil(self.token_budget / self.compress_ratio)
        if self.block_topk <= 0:
            raise ValueError(
                f'QSA indexer_budget must contain at least one complete block, got '
                f'budget={self.token_budget}, compress_ratio={self.compress_ratio}')
        # A query can receive ``block_topk * R`` complete-block tokens and up
        # to R-1 causal tail tokens.  The configured Qwen shape is divisible,
        # but deriving this from block_topk also keeps the invariant true for
        # a diagnostic non-divisible configuration.
        self.selected_k = self.block_topk * self.compress_ratio + self.compress_ratio - 1
        # Replicated projection (reference uses ReplicatedLinear).
        self.index_qk_proj = nn.Linear(
            config.hidden_size, (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            dtype=config.params_dtype)
        self.q_layernorm = Qwen4ExpTextRMSNorm(
            self.index_head_dim,
            eps=config.layernorm_epsilon,
            dtype=config.params_dtype,
            sequence_parallel=config.sequence_parallel)
        self.k_layernorm = Qwen4ExpTextRMSNorm(
            self.index_head_dim,
            eps=config.layernorm_epsilon,
            dtype=config.params_dtype,
            sequence_parallel=config.sequence_parallel)
        setattr(self.index_qk_proj.weight, 'sequence_parallel', config.sequence_parallel)

    def forward(self, *args, **kwargs):
        raise RuntimeError('QSAIndexer performs selection via `select_topk`, not `forward`.')

    def _materialize_freqs(self, freqs: torch.Tensor, sequence_length: int, dtype: torch.dtype):
        if isinstance(freqs, tuple):
            # The mcore caller supplies one angle tensor for self-attention;
            # accepting a tuple makes the adapter convenient for HF-style test
            # fixtures without changing the mcore path.
            freqs = freqs[0]
        if freqs is None:
            raise ValueError('QSA indexer requires rotary frequencies')
        mscale = getattr(self.config, 'attention_scaling', 1.0) or 1.0
        f = freqs.reshape(freqs.shape[0], -1)[:sequence_length]
        cos = (torch.cos(f) * mscale).to(dtype)
        sin = (torch.sin(f) * mscale).to(dtype)
        return cos, sin

    @staticmethod
    def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        rot = cos.shape[-1]
        t_rope, t_pass = t[..., :rot], t[..., rot:]
        t_rope = (t_rope * cos) + (_rotate_half(t_rope) * sin)
        return torch.cat((t_rope, t_pass), dim=-1)

    @torch.no_grad()
    def _project_and_pool(self, hidden_states: torch.Tensor, freqs: torch.Tensor):
        if hidden_states.ndim != 3:
            raise ValueError(f'QSA indexer expects hidden_states [S,B,H], got {tuple(hidden_states.shape)}')
        sequence_length, batch, _ = hidden_states.shape
        ratio = self.compress_ratio
        num_blocks = sequence_length // ratio
        device = hidden_states.device

        qk = self.index_qk_proj(hidden_states)
        if self.index_kv_heads != 1:
            raise ValueError(f'Qwen4-Exp QSA requires indexer_kv_heads=1, got {self.index_kv_heads}')
        cos, sin = self._materialize_freqs(freqs, sequence_length, qk.dtype)
        use_fused_postprocess = (
            os.environ.get('MCORE_BRIDGE_QSA_INDEXER_FUSED_POSTPROCESS', '1') != '0'
            and qk.is_cuda
            and qk.dtype == torch.bfloat16
            and self.index_n_heads == 4
            and self.index_head_dim == 128
            and self.compress_ratio == 4
            and self.q_layernorm.weight.device == qk.device
            and self.q_layernorm.weight.dtype == qk.dtype
            and self.k_layernorm.weight.device == qk.device
            and self.k_layernorm.weight.dtype == qk.dtype
            and cos.ndim == 2
            and 0 < cos.shape[1] <= self.index_head_dim
            and cos.shape[1] % 2 == 0
        )
        if use_fused_postprocess:
            from .qsa_triton import is_sm90, qsa_indexer_fused_postprocess

            if is_sm90(qk.device):
                return qsa_indexer_fused_postprocess(
                    qk,
                    self.q_layernorm.weight,
                    self.k_layernorm.weight,
                    cos,
                    sin,
                    self.config.layernorm_epsilon,
                    self.index_n_heads,
                    self.compress_ratio,
                )
        q, token_k = torch.split(
            qk,
            [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
            dim=-1)
        q = q.reshape(sequence_length, batch, self.index_n_heads, self.index_head_dim).permute(1, 0, 2, 3)
        raw_keys = token_k.reshape(
            sequence_length, batch, self.index_kv_heads, self.index_head_dim).permute(1, 0, 2, 3)
        raw_keys = raw_keys.squeeze(2)
        q = self.q_layernorm(q)
        q = self._apply_rope(q, cos[None, :, None, :], sin[None, :, None, :]).contiguous()

        usable = num_blocks * ratio
        if num_blocks:
            key_groups = raw_keys[:, :usable].reshape(batch, num_blocks, ratio, self.index_head_dim)
            pooled = key_groups.float().mean(dim=2).to(raw_keys.dtype)
            pooled = self.k_layernorm(pooled)
            starts = torch.arange(num_blocks, device=device, dtype=torch.long) * ratio
            block_keys = self._apply_rope(pooled, cos[starts][None], sin[starts][None]).contiguous()
        else:
            block_keys = raw_keys.new_empty((batch, 0, self.index_head_dim))
        return q, block_keys

    @torch.no_grad()
    def _project_and_pool_packed(self, hidden_states: torch.Tensor, freqs: torch.Tensor,
                                 boundaries):
        """Project packed THD tokens and concatenate segment-local block keys."""

        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError('QSA packed indexer expects hidden_states [T,1,H]')
        total_length = hidden_states.shape[0]
        ratio = self.compress_ratio
        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk,
            [self.index_n_heads * self.index_head_dim,
             self.index_kv_heads * self.index_head_dim],
            dim=-1)
        q = q.reshape(total_length, 1, self.index_n_heads, self.index_head_dim)
        q = q.permute(1, 0, 2, 3)
        raw_keys = token_k.reshape(
            total_length, 1, self.index_kv_heads, self.index_head_dim)
        raw_keys = raw_keys.permute(1, 0, 2, 3).squeeze(2)
        q = self.q_layernorm(q)

        q_cos_chunks = []
        q_sin_chunks = []
        block_key_chunks = []
        block_cos_chunks = []
        block_sin_chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            length = end - start
            if length == 0:
                continue
            segment_freqs = self._slice_packed_freqs(
                freqs, start, end, length, total_length)
            cos, sin = self._materialize_freqs(segment_freqs, length, q.dtype)
            q_cos_chunks.append(cos)
            q_sin_chunks.append(sin)
            usable = (length // ratio) * ratio
            if usable:
                block_key_chunks.append(
                    raw_keys[0, start:start + usable].reshape(
                        usable // ratio, ratio, self.index_head_dim).float().mean(1).to(raw_keys.dtype))
                block_cos_chunks.append(cos[:usable:ratio])
                block_sin_chunks.append(sin[:usable:ratio])

        q_cos = torch.cat(q_cos_chunks, dim=0)
        q_sin = torch.cat(q_sin_chunks, dim=0)
        q = self._apply_rope(q, q_cos[None, :, None, :], q_sin[None, :, None, :]).contiguous()
        if block_key_chunks:
            pooled = self.k_layernorm(torch.cat(block_key_chunks, dim=0))
            block_cos = torch.cat(block_cos_chunks, dim=0)
            block_sin = torch.cat(block_sin_chunks, dim=0)
            block_keys = self._apply_rope(pooled, block_cos, block_sin).contiguous()
        else:
            block_keys = raw_keys.new_empty((0, self.index_head_dim))

        device = hidden_states.device
        lengths_q = torch.tensor(
            [end - start for start, end in zip(boundaries[:-1], boundaries[1:])],
            device=device, dtype=torch.long)
        block_counts = torch.tensor(
            [(end - start) // ratio for start, end in zip(boundaries[:-1], boundaries[1:])],
            device=device, dtype=torch.long)
        segment_ids = torch.repeat_interleave(
            torch.arange(len(boundaries) - 1, device=device, dtype=torch.long), lengths_q)
        q_starts = torch.tensor(boundaries[:-1], device=device, dtype=torch.long)
        block_starts = torch.cat((
            torch.zeros(1, device=device, dtype=torch.long),
            block_counts.cumsum(0)[:-1],
        ))
        per_query_block_starts = block_starts.index_select(0, segment_ids)
        per_query_block_counts = block_counts.index_select(0, segment_ids)
        local_positions = (
            torch.arange(total_length, device=device, dtype=torch.long)
            - q_starts.index_select(0, segment_ids)
        )
        return (
            q[0],
            block_keys,
            per_query_block_starts.to(torch.int32).contiguous(),
            per_query_block_counts.to(torch.int32).contiguous(),
            local_positions.to(torch.int32).contiguous(),
        )

    @staticmethod
    def _stable_merge_topk(
        best_scores: torch.Tensor,
        best_blocks: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_blocks: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge a block tile with deterministic score/id ordering."""

        scores = torch.cat((best_scores, candidate_scores), dim=-1)
        blocks = torch.cat((best_blocks, candidate_blocks), dim=-1)
        # Candidate tiles are visited in ascending block order and the old
        # list is already canonical.  A stable descending sort therefore
        # resolves equal scores in favor of the lower block id.
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]
        return scores.gather(-1, order), blocks.gather(-1, order)

    def _score_block_tile(
        self,
        q_tile: torch.Tensor,
        block_keys_tile: torch.Tensor,
        query_positions: torch.Tensor,
        block_start: int,
        total_blocks: int,
        backend: str,
    ) -> torch.Tensor:
        if backend == 'triton':
            from .qsa_triton import qsa_indexer_score_tile_with_ratio

            return qsa_indexer_score_tile_with_ratio(
                q_tile.contiguous(), block_keys_tile.contiguous(), query_positions, block_start, total_blocks,
                self.compress_ratio)
        # [B,Q,H,D] x [B,K,D] -> [B,Q,H,K].  The tile is released after the
        # merge; unlike the old implementation this is never [B,S,S/R].
        scores = torch.einsum('bqhd,bkd->bqhk', q_tile.float(), block_keys_tile.float())
        return torch.relu(scores).sum(dim=2) / math.sqrt(self.index_head_dim)

    @torch.no_grad()
    def _select_from_projected(
        self,
        q: torch.Tensor,
        block_keys: torch.Tensor,
        query_positions: torch.Tensor,
        backend: str,
        query_tile_size: int,
        key_tile_size: int,
        return_block_ids: bool = False,
        dense_zero_based: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k merge for projected queries and globally indexed block keys.

        The compatibility result contains expanded token IDs.  Production
        callers can request only complete-block IDs; token lanes and the
        incomplete causal tail are then reconstructed inside attention.
        """

        if backend not in {'torch', 'triton'}:
            raise ValueError(f'unsupported QSA indexer backend={backend!r}; choose torch or triton')
        if q.ndim != 4 or block_keys.ndim != 3 or query_positions.ndim != 1:
            raise ValueError(
                f'QSA projected shapes must be q=[B,S,H,D], blocks=[B,N,D], positions=[S]; '
                f'got q={tuple(q.shape)}, blocks={tuple(block_keys.shape)}, positions={tuple(query_positions.shape)}')
        batch, sequence_length, _, _ = q.shape
        if block_keys.shape[0] != batch or block_keys.shape[2] != self.index_head_dim:
            raise ValueError(
                f'QSA projected batch/head shape mismatch: q={tuple(q.shape)}, blocks={tuple(block_keys.shape)}')
        if query_positions.shape[0] != sequence_length:
            raise ValueError(
                f'QSA query position length mismatch: q={sequence_length}, positions={query_positions.shape[0]}')
        query_tile_size = max(1, int(query_tile_size))
        key_tile_size = max(1, int(key_tile_size))
        ratio = self.compress_ratio
        num_blocks = block_keys.shape[1]
        block_topk = min(self.block_topk, num_blocks)
        query_positions = query_positions.to(device=q.device, dtype=torch.long)
        complete_blocks = torch.div(query_positions + 1, ratio, rounding_mode='floor')
        block_counts = complete_blocks.clamp_max(block_topk)
        tail_lengths = query_positions + 1 - complete_blocks * ratio
        topk_length = (block_counts * ratio + tail_lengths).to(torch.int32).expand(batch, -1).clone()
        if num_blocks <= self.block_topk:
            if return_block_ids:
                block_slots = torch.arange(
                    self.block_topk, device=q.device, dtype=torch.int32)
                valid_blocks = (
                    block_slots[None, None, :]
                    < block_counts[None, :, None]
                )
                selected_blocks = torch.where(
                    valid_blocks,
                    block_slots[None, None, :],
                    -torch.ones_like(block_slots)[None, None, :],
                ).expand(batch, -1, -1).clone()
                return selected_blocks, topk_length
            selected = torch.full(
                (batch, sequence_length, self.selected_k), -1,
                dtype=torch.int32, device=q.device)
            slots = torch.arange(self.selected_k, device=q.device, dtype=torch.long)
            causal = slots[None, :] <= query_positions[:, None]
            full = torch.where(causal[None], slots[None, None, :], -torch.ones_like(slots)[None, None, :])
            selected.copy_(full.to(torch.int32).expand(batch, -1, -1))
            return selected, topk_length

        # Production BF16 QSA uses a single device-side streaming kernel for
        # score -> causal filter -> block Top-K -> optional token expansion.  Besides
        # removing the Python ``query_tile x key_tile`` launch/merge loop, the
        # kernel writes the final route directly and therefore does not need
        # a second temporary ``[B,S,block_topk]`` tensor.  Keep the tiled path
        # below for FP32 and deliberately irregular diagnostic shapes.
        if (backend == 'triton' and q.dtype == torch.bfloat16 and
                block_topk == self.block_topk and
                block_topk > 0 and (block_topk & (block_topk - 1)) == 0 and
                ratio >= 2 and
                os.environ.get('MCORE_BRIDGE_QSA_INDEXER_FUSED', '1') != '0'):
            from .qsa_triton import (
                is_sm90,
                qsa_indexer_fused_topk_with_ratio,
                qsa_indexer_slab_topk_with_ratio,
            )

            # The production Qwen shape benefits from scoring query rows in
            # tensor-core tiles and selecting each bounded FP32 slab with
            # CUDA Top-K.  Packed documents and CP use non-contiguous logical
            # positions, so they retain the segment-aware streaming kernels.
            use_score_slabs = (
                return_block_ids
                and block_topk == 512
                and ratio == 4
                and q.shape[2:] == (4, 128)
                and dense_zero_based
                and is_sm90(q.device)
                and os.environ.get(
                    'MCORE_BRIDGE_QSA_INDEXER_SCORE_SLAB', '1') != '0'
            )
            if use_score_slabs:
                max_score_mb = int(os.environ.get(
                    'MCORE_BRIDGE_QSA_INDEXER_SCORE_SLAB_MB', '512'))
                if max_score_mb <= 0:
                    raise ValueError(
                        'MCORE_BRIDGE_QSA_INDEXER_SCORE_SLAB_MB must be positive')
                boundary_guard = float(os.environ.get(
                    'MCORE_BRIDGE_QSA_INDEXER_SCORE_BOUNDARY_GUARD', '0'))
                selected_blocks = qsa_indexer_slab_topk_with_ratio(
                    q.contiguous(),
                    block_keys.contiguous(),
                    query_positions,
                    ratio,
                    block_topk,
                    max_score_bytes=max_score_mb * 2**20,
                    boundary_guard=boundary_guard,
                    validate_positions=False,
                )
                return selected_blocks, topk_length

            max_partial_mb = int(os.environ.get(
                'MCORE_BRIDGE_QSA_INDEXER_MAX_PARTIAL_MB', '64'))
            if max_partial_mb <= 0:
                raise ValueError(
                    'MCORE_BRIDGE_QSA_INDEXER_MAX_PARTIAL_MB must be positive')
            fused_selected = qsa_indexer_fused_topk_with_ratio(
                q.contiguous(),
                block_keys.contiguous(),
                query_positions,
                ratio,
                block_topk,
                max_partial_bytes=max_partial_mb * 2**20,
                return_block_ids=return_block_ids,
            )
            return fused_selected, topk_length

        # The fused route returns its own final token list.  Allocate the
        # fallback workspace only after that route is ruled out; otherwise a
        # 256K run transiently holds two full [B,S,K] int32 buffers.
        selected = torch.full(
            (
                batch,
                sequence_length,
                self.block_topk if return_block_ids else self.selected_k,
            ), -1,
            dtype=torch.int32, device=q.device)
        for query_start in range(0, sequence_length, query_tile_size):
            query_end = min(sequence_length, query_start + query_tile_size)
            q_tile = q[:, query_start:query_end]
            query_block_counts = complete_blocks[query_start:query_end]
            query_pos_tile = query_positions[query_start:query_end]
            best_scores = torch.full(
                (batch, query_end - query_start, block_topk), -float('inf'), dtype=torch.float32, device=q.device)
            best_blocks = torch.full(
                (batch, query_end - query_start, block_topk), -1, dtype=torch.long, device=q.device)

            for block_start in range(0, num_blocks, key_tile_size):
                block_end = min(num_blocks, block_start + key_tile_size)
                candidate_blocks = torch.arange(block_start, block_end, device=q.device, dtype=torch.long)
                candidate_scores = self._score_block_tile(
                    q_tile, block_keys[:, block_start:block_end], query_pos_tile, block_start, num_blocks, backend)
                valid = candidate_blocks[None, None, :] < query_block_counts[None, :, None]
                candidate_scores = candidate_scores.masked_fill(~valid, -float('inf'))
                candidate_blocks = candidate_blocks[None, None, :].expand(batch, query_end - query_start, -1)
                best_scores, best_blocks = self._stable_merge_topk(
                    best_scores, best_blocks, candidate_scores, candidate_blocks, block_topk)

            if return_block_ids:
                valid_selected_blocks = (
                    torch.arange(block_topk, device=q.device)[None, None, :]
                    < query_block_counts[None, :, None]
                )
                selected[:, query_start:query_end, :block_topk] = torch.where(
                    valid_selected_blocks,
                    best_blocks,
                    torch.full_like(best_blocks, -1),
                ).to(torch.int32)
                continue

            tile_queries = query_end - query_start
            scratch = torch.full(
                (batch, tile_queries, self.selected_k + 1), -1, dtype=torch.int32, device=q.device)
            block_offsets = torch.arange(ratio, device=q.device, dtype=torch.long)
            block_positions = torch.arange(block_topk, device=q.device, dtype=torch.long)[:, None] * ratio
            block_positions = block_positions + block_offsets[None, :]
            block_valid = torch.arange(block_topk, device=q.device)[None, None, :] < query_block_counts[
                None, :, None]
            block_tokens = (best_blocks[..., None] * ratio + block_offsets[None, None, None, :]).reshape(
                batch, tile_queries, block_topk * ratio).to(torch.int32)
            block_positions = block_positions.reshape(1, 1, -1).expand(batch, tile_queries, -1)
            block_valid = block_valid[..., None].expand(batch, tile_queries, block_topk, ratio).reshape(
                batch, tile_queries, -1)
            block_scatter = torch.where(block_valid, block_positions, torch.full_like(block_positions, self.selected_k))
            scratch.scatter_(-1, block_scatter, torch.where(block_valid, block_tokens,
                                                            torch.full_like(block_tokens, -1)))

            tail_values = query_block_counts[None, :, None] * ratio + torch.arange(
                ratio - 1, device=q.device, dtype=torch.long)[None, None, :]
            tail_valid = tail_values < (query_pos_tile[None, :, None] + 1)
            tail_offsets = block_counts[query_start:query_end][None, :, None] * ratio + torch.arange(
                ratio - 1, device=q.device, dtype=torch.long)[None, None, :]
            tail_scatter = torch.where(tail_valid, tail_offsets, torch.full_like(tail_offsets, self.selected_k))
            tail_values = tail_values.expand(batch, -1, -1).to(torch.int32)
            tail_valid = tail_valid.expand(batch, -1, -1)
            scratch.scatter_(-1, tail_scatter.expand(batch, -1, -1), torch.where(
                tail_valid, tail_values, torch.full_like(tail_values, -1)))
            selected[:, query_start:query_end] = scratch[..., :self.selected_k]
        return selected, topk_length

    @torch.no_grad()
    def select_topk(
        self,
        hidden_states: torch.Tensor,
        freqs: torch.Tensor,
        backend: str = 'torch',
        query_tile_size: Optional[int] = None,
        key_tile_size: Optional[int] = None,
        return_block_ids: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return route IDs and token lengths for selected-KV attention.

        ``return_block_ids=False`` preserves the public, score-ordered token-ID
        contract.  The compact production form uses ``return_block_ids=True``,
        has shape ``[B,S,block_topk]``, and guarantees the exact selected set;
        its block order is deliberately not part of the public ABI.
        """

        if hidden_states.ndim != 3:
            raise ValueError(f'QSA indexer expects hidden_states [S,B,H], got {tuple(hidden_states.shape)}')
        sequence_length = hidden_states.shape[0]
        q, block_keys = self._project_and_pool(hidden_states, freqs)
        query_positions = torch.arange(sequence_length, device=hidden_states.device, dtype=torch.long)
        return self._select_from_projected(
            q, block_keys, query_positions, backend,
            query_tile_size or getattr(self.config, 'qsa_indexer_query_tile_size', 128),
            key_tile_size or getattr(self.config, 'qsa_indexer_key_tile_size', 512),
            return_block_ids=return_block_ids,
            dense_zero_based=True)

    @staticmethod
    def _slice_packed_freqs(freqs, start: int, end: int, length: int, total_length: int):
        """Select one packed segment from either flattened or max-length RoPE."""

        if isinstance(freqs, tuple):
            return tuple(
                QSAIndexer._slice_packed_freqs(item, start, end, length, total_length)
                for item in freqs
            )
        if freqs is None:
            return None
        if freqs.shape[0] == total_length:
            return freqs[start:end]
        if freqs.shape[0] >= length:
            # Packed THD RoPE commonly reuses one max-length position table
            # for every sample; positions reset at each segment boundary.
            return freqs[:length]
        raise ValueError(
            f'QSA packed rotary length {freqs.shape[0]} is shorter than segment length {length}')

    @torch.no_grad()
    def select_topk_packed(
        self,
        hidden_states: torch.Tensor,
        freqs: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        backend: str = 'torch',
        query_tile_size: Optional[int] = None,
        key_tile_size: Optional[int] = None,
        return_block_ids: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select QSA tokens independently for each packed/THD segment.

        The returned indices are local to each segment.  This deliberate
        segment-local contract lets the packed attention adapter rebase keys
        inside one varlen launch without ever exposing tokens from a neighboring
        document.  Triton/BF16 supported budgets use the fused packed path;
        torch and irregular diagnostic shapes retain the segment loop.
        """

        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError(
                'QSA packed indexer expects hidden_states [T,1,H] (THD dummy batch)')
        if cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() < 2:
            raise ValueError('QSA packed indexer requires cu_seqlens_q [num_segments+1]')
        total_length = hidden_states.shape[0]
        boundaries = cu_seqlens_q.to(device='cpu', dtype=torch.long).tolist()
        if boundaries[0] != 0 or boundaries[-1] != total_length:
            raise ValueError(
                f'QSA packed cu_seqlens must start at 0 and end at T={total_length}, '
                f'got {boundaries[0]}..{boundaries[-1]}')
        if any(end < start for start, end in zip(boundaries[:-1], boundaries[1:])):
            raise ValueError('QSA packed cu_seqlens must be non-decreasing')
        if total_length == 0:
            return (
                torch.empty(
                    (
                        1,
                        0,
                        self.block_topk if return_block_ids else self.selected_k,
                    ),
                    device=hidden_states.device,
                    dtype=torch.int32,
                ),
                torch.empty(
                    (1, 0), device=hidden_states.device, dtype=torch.int32))

        # The production packed path projects/pools all documents once and
        # runs the segment-local streaming Top-K kernel in a single launch.
        # Keep torch/irregular debug shapes on the auditable segment loop.
        if (backend == 'triton' and hidden_states.dtype == torch.bfloat16 and
                self.block_topk in {128, 256, 512} and
                (self.block_topk & (self.block_topk - 1)) == 0 and
                os.environ.get('MCORE_BRIDGE_QSA_INDEXER_FUSED', '1') != '0'):
            from .qsa_triton import qsa_indexer_fused_topk_packed

            q, block_keys, block_starts, block_counts, local_positions = (
                self._project_and_pool_packed(hidden_states, freqs, boundaries))
            selected = qsa_indexer_fused_topk_packed(
                q,
                block_keys,
                block_starts,
                block_counts,
                local_positions,
                self.compress_ratio,
                self.block_topk,
                return_block_ids=return_block_ids,
            )
            complete_blocks = torch.minimum(
                block_counts.to(torch.long),
                (local_positions.to(torch.long) + 1) // self.compress_ratio,
            )
            selected_block_counts = complete_blocks.clamp_max(self.block_topk)
            tail_lengths = (
                local_positions.to(torch.long) + 1
                - complete_blocks * self.compress_ratio
            )
            topk_lengths = (
                selected_block_counts * self.compress_ratio + tail_lengths
            ).to(torch.int32)
            return selected.unsqueeze(0), topk_lengths.unsqueeze(0)

        selected = torch.full(
            (
                1,
                total_length,
                self.block_topk if return_block_ids else self.selected_k,
            ),
            -1,
            device=hidden_states.device,
            dtype=torch.int32,
        )
        lengths = torch.zeros(
            (1, total_length), device=hidden_states.device, dtype=torch.int32
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end == start:
                continue
            segment_length = end - start
            segment_freqs = self._slice_packed_freqs(
                freqs, start, end, segment_length, total_length)
            segment_selected, segment_lengths = self.select_topk(
                hidden_states[start:end],
                segment_freqs,
                backend=backend,
                query_tile_size=query_tile_size,
                key_tile_size=key_tile_size,
                return_block_ids=return_block_ids,
            )
            selected[:, start:end] = segment_selected
            lengths[:, start:end] = segment_lengths
        return selected, lengths

    @torch.no_grad()
    def select_topk_cp(
        self,
        hidden_states: torch.Tensor,
        freqs: torch.Tensor,
        query_positions: torch.Tensor,
        cp_group,
        partition_mode: str = 'zigzag',
        backend: str = 'torch',
        query_tile_size: Optional[int] = None,
        key_tile_size: Optional[int] = None,
        return_block_ids: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CP indexer path: all-gather block keys, never full hidden states.

        Queries are projected only on their local rank.  Each rank pools its
        local CP segments, gathers the small indexer block-key tensor, restores
        global block order, and performs top-k only for local query positions.
        """

        if hidden_states.ndim != 3 or query_positions.ndim != 1:
            raise ValueError('QSA CP indexer expects hidden_states=[S_local,B,H] and query_positions=[S_local]')
        cp_size = cp_group.size()
        if cp_size <= 1:
            return self.select_topk(
                hidden_states,
                freqs,
                backend,
                query_tile_size,
                key_tile_size,
                return_block_ids,
            )
        local_sequence, batch, _ = hidden_states.shape
        if query_positions.shape[0] != local_sequence:
            raise ValueError('QSA CP query position length must equal local hidden sequence length')
        global_sequence = local_sequence * cp_size
        ratio = self.compress_ratio
        if global_sequence % cp_size or local_sequence % ratio:
            raise ValueError('QSA CP requires local sequence length divisible by compress_ratio')

        if isinstance(freqs, tuple):
            freqs = tuple(
                freq.index_select(0, query_positions.to(torch.long)) if freq.shape[0] != local_sequence else freq
                for freq in freqs)
            freq_for_query = freqs[0]
        else:
            freq_for_query = (freqs.index_select(0, query_positions.to(torch.long))
                              if freqs.shape[0] != local_sequence else freqs)
        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk, [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim], dim=-1)
        q = q.reshape(local_sequence, batch, self.index_n_heads, self.index_head_dim).permute(1, 0, 2, 3)
        raw_keys = token_k.reshape(
            local_sequence, batch, self.index_kv_heads, self.index_head_dim).permute(1, 0, 2, 3).squeeze(2)
        q = self.q_layernorm(q)
        cos, sin = self._materialize_freqs(freq_for_query, local_sequence, q.dtype)
        q = self._apply_rope(q, cos[None, :, None, :], sin[None, :, None, :]).contiguous()

        if partition_mode == 'contiguous':
            segment_lengths = (local_sequence, )
        elif partition_mode == 'zigzag':
            if global_sequence % (2 * cp_size):
                raise ValueError('QSA zigzag CP requires global sequence divisible by 2*cp_size')
            segment_lengths = (global_sequence // (2 * cp_size), ) * 2
        else:
            raise ValueError(f'unsupported QSA CP partition mode: {partition_mode!r}')
        local_blocks = []
        segment_start = 0
        for segment_length in segment_lengths:
            if segment_length % ratio:
                raise ValueError('QSA CP segment length must be divisible by compress_ratio')
            segment_blocks = segment_length // ratio
            keys = raw_keys[:, segment_start:segment_start + segment_length]
            keys = keys.reshape(batch, segment_blocks, ratio, self.index_head_dim)
            pooled = self.k_layernorm(keys.float().mean(dim=2).to(raw_keys.dtype))
            block_starts = segment_start + torch.arange(segment_blocks, device=q.device, dtype=torch.long) * ratio
            local_blocks.append(self._apply_rope(pooled, cos[block_starts][None], sin[block_starts][None]))
            segment_start += segment_length
        local_blocks = torch.cat(local_blocks, dim=1).contiguous()
        local_block_count = local_blocks.shape[1]
        gathered = [torch.empty_like(local_blocks) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered, local_blocks, group=cp_group)
        gathered = torch.stack(gathered, dim=0)

        global_block_positions = torch.arange(0, global_sequence, ratio, device=q.device, dtype=torch.long)
        if partition_mode == 'contiguous':
            owner = global_block_positions // local_sequence
            slot = (global_block_positions % local_sequence) // ratio
        else:
            chunk_length = global_sequence // (2 * cp_size)
            chunk = global_block_positions // chunk_length
            within = global_block_positions % chunk_length
            owner = torch.where(chunk < cp_size, chunk, 2 * cp_size - 1 - chunk)
            local_offset = torch.where(chunk < cp_size, within, chunk_length + within)
            slot = local_offset // ratio
        global_blocks = gathered[owner, :, slot, :].permute(1, 0, 2).contiguous()
        return self._select_from_projected(
            q,
            global_blocks,
            query_positions,
            backend,
            query_tile_size or getattr(self.config, 'qsa_indexer_query_tile_size', 128),
            key_tile_size or getattr(self.config, 'qsa_indexer_key_tile_size', 512),
            return_block_ids=return_block_ids,
        )

    @torch.no_grad()
    def select_mask(self, hidden_states: torch.Tensor, freqs: torch.Tensor) -> Optional[torch.Tensor]:
        """Compatibility adapter returning the legacy arbitrary mask.

        This method is intentionally not used by the selected-KV path.  It is
        retained so ``qsa_kernel_backend=none`` and existing callers preserve
        their behavior.
        """

        sequence_length, batch, _ = hidden_states.shape
        if sequence_length // self.compress_ratio <= self.block_topk:
            return None
        topk_indices, topk_length = self.select_topk(hidden_states, freqs, backend='torch')
        selected = torch.zeros(
            (batch, sequence_length, sequence_length + 1), dtype=torch.bool, device=hidden_states.device)
        safe = topk_indices.to(torch.long).clamp_min(0)
        valid = torch.arange(self.selected_k, device=hidden_states.device)[None, None, :] < topk_length[..., None]
        scatter_indices = torch.where(valid, safe, torch.full_like(safe, sequence_length))
        selected.scatter_(-1, scatter_indices, True)
        allowed = selected[..., :sequence_length]
        # ``select_topk`` is causal by construction; retaining this explicit
        # invariant protects the compatibility adapter from future changes.
        positions = torch.arange(sequence_length, device=hidden_states.device)
        allowed &= positions[None, None, :] <= positions[None, :, None]
        return ~allowed.unsqueeze(1)


__all__ = ['QSAIndexer', 'Qwen4ExpTextRMSNorm']
