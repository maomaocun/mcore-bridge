# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import os
import torch
import transformer_engine.pytorch as te
from contextlib import contextmanager
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
    apply_rotary_pos_emb,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from typing import Optional

from mcore_bridge.bridge import GPTBridge
from mcore_bridge.utils import Fp8Dequantizer, fp4_to_fp8

from ..constant import ModelType
from ..gpt_model import GPTModel
from ..modules.compressor import Compressor, CSAIndexer
from ..register import ModelLoader, ModelMeta, register_model
from ..rope import get_rope_inv_freq


def _dsv4_log_bridge_kv(phase, layer_number, tensor):
    """Log KV projection finiteness for an explicitly selected debug rank."""
    if os.environ.get('DSV4_LOG_BRIDGE_KV', '0') != '1':
        return
    try:
        if int(layer_number) != int(os.environ.get('DSV4_LOG_BRIDGE_KV_LAYER', '1')):
            return
    except ValueError:
        return
    rank = os.environ.get('RANK', 'unknown')
    selected = {
        item.strip()
        for item in os.environ.get('DSV4_LOG_BRIDGE_KV_RANKS', '3,7').split(',')
        if item.strip()
    }
    if rank not in selected or tensor is None:
        return
    value = tensor.detach()
    if value.numel() == 0:
        print(f'[DSV4 Bridge KV] phase={phase} rank={rank} shape={tuple(value.shape)} empty', flush=True)
        return
    finite = torch.isfinite(value)
    print(
        f'[DSV4 Bridge KV] phase={phase} rank={rank} shape={tuple(value.shape)} '
        f'dtype={value.dtype} finite={bool(finite.all().item())} '
        f'nan={int(torch.isnan(value).sum().item())} '
        f'inf={int(torch.isinf(value).sum().item())} '
        f'max_abs={float(value.float().abs().amax().item())}',
        flush=True,
    )

try:
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import \
        FineGrainedActivationOffloadingInterface as off_interface
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import \
        DSv4HybridSelfAttention as McoreDSv4HybridSelfAttention
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import _q_rms_norm
    from megatron.core.typed_torch import apply_module
except ImportError:
    McoreDSv4HybridSelfAttention = object
    _q_rms_norm = None
    apply_module = None
    off_interface = None

try:
    from megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention import \
        copy_attention_chunk
except ImportError:
    copy_attention_chunk = None


class _MergeRotaryOutput(torch.autograd.Function):
    """Merge inverse-RoPE output into the original attention buffer.

    This is used only with ``DSV4_CSA_RECOMPUTE_OUT=1``. The sparse-attention
    backward then recomputes its original output, while this node routes the
    content gradient directly and the rotary gradient through the independent
    rotary slice without allocating another full hidden-state tensor.
    """

    @staticmethod
    def forward(ctx, core_output, rotated_output, content_dim):
        ctx.content_dim = int(content_dim)
        ctx.rotary_dim = int(rotated_output.shape[-1])
        core_output.data.narrow(-1, ctx.content_dim, ctx.rotary_dim).copy_(rotated_output.data)
        return core_output

    @staticmethod
    def backward(ctx, grad_output):
        rotary_grad = grad_output.narrow(-1, ctx.content_dim, ctx.rotary_dim).contiguous()
        # The raw-storage update avoids a second full-size gradient buffer;
        # this node owns the incoming gradient and returns it only once.
        grad_output.data.narrow(-1, ctx.content_dim, ctx.rotary_dim).zero_()
        return grad_output, rotary_grad, None


def merge_rotary_output(core_output, rotated_output, content_dim):
    """Apply the memory-bounded inverse-RoPE merge autograd node."""
    return _MergeRotaryOutput.apply(core_output, rotated_output, content_dim)


@contextmanager
def _patch_YarnRotaryEmbedding(config):
    """Temporarily patch missing rope scaling attrs on config for YarnRotaryEmbedding init.

    YarnRotaryEmbedding requires beta_fast/beta_slow/mscale/mscale_all_dim on config,
    but DeepSeek-V4 HF config may not include them. This context manager sets defaults
    on entry and removes them on exit, keeping the config clean (the resulting
    YarnRotaryEmbedding module will be deleted later anyway).
    """
    defaults = {
        'original_max_position_embeddings': 4096,
        'beta_fast': 32.0,
        'beta_slow': 1.0,
        'mscale': 1.0,
        'mscale_all_dim': 0.0,
    }
    added = []
    for attr, value in defaults.items():
        if getattr(config, attr, None) is None:
            setattr(config, attr, value)
            added.append(attr)
    try:
        yield config
    finally:
        # Restore: remove attrs that were temporarily added
        for attr in added:
            delattr(config, attr)


class DSv4HybridSelfAttention(McoreDSv4HybridSelfAttention):

    def __init__(self, config, *args, **kwargs):
        assert McoreDSv4HybridSelfAttention is not object, (
            'Please install the Megatron-Core dev branch: '
            '`pip install git+https://github.com/NVIDIA/Megatron-LM@dev`')
        with _patch_YarnRotaryEmbedding(config):
            super().__init__(config, *args, **kwargs)
        self.layer_type = self.config.hf_config.layer_types[self.layer_number - 1]
        self.rope_layer_type = 'main' if self.layer_type == 'sliding_attention' else 'compress'
        if config.fp8_param:
            group_proj_in_size = self.query_projection_size // config.o_groups
            del self.linear_o_group_proj
            if config.o_groups % self.tp_size != 0:
                raise ValueError(
                    "o_groups must be divisible by tensor model parallel size for FP8 grouped output: "
                    f"{config.o_groups} % {self.tp_size} != 0"
                )
            self.linear_o_group_proj = te.GroupedLinear(
                num_gemms=self.o_local_groups,
                in_features=group_proj_in_size,
                out_features=config.o_lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
            )
            self._o_group_proj_is_grouped_linear = True
        else:
            self._o_group_proj_is_grouped_linear = False

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        rotary_pos_emb=None,
        *,
        inference_params=None,
        boundary_hidden=None,
        boundary_rotary_pos_emb=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert (hidden_states.ndim == 3), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"
        if packed_seq_params is not None:
            assert (packed_seq_params.local_cp_size
                    is None), 'dynamic_context_parallel is not supported with MLA yet and is planned for future. \
            Please disable dynamic_context_parallel.'

        assert (inference_context is None
                and inference_params is None), 'Inference is not supported for DSv4HybridSelfAttention.'

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        # q_compressed: [s, b, q_lora_rank]
        q_compressed, _ = self.linear_q_down_proj(hidden_states)

        kv_compressed = hidden_states

        if packed_seq_params is not None:
            # If sequence packing, TE expect [t, h, d] shaped qkv input.
            # In Megatron-Core, the qkv shape is [t, 1, h, d].
            # So we need to reshape qkv from [t, 1, h, d] to [t, h, d].
            q_compressed = q_compressed.squeeze(1)

        # =========================================
        # Apply norm
        # =========================================

        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = apply_module(self.q_layernorm)(q_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        try:
            qkv_up_proj_chunk_size = int(
                os.environ.get('DSV4_QKV_UP_PROJ_CHUNK_SIZE', '0') or 0
            )
        except ValueError:
            qkv_up_proj_chunk_size = 0

        def run_chunked_projection(projection, projection_input):
            """Run a token-wise TE projection in bounded row chunks.

            Q/KV up projections retain their full logical output for the CSA
            interface, but a single 256K TE GEMM can request a multi-GiB
            temporary during activation recompute.  The copy node keeps each
            chunk's autograd edge without introducing a second full output.
            """
            if (
                qkv_up_proj_chunk_size <= 0
                or projection_input.size(0) <= qkv_up_proj_chunk_size
                or copy_attention_chunk is None
            ):
                return projection(projection_input)

            output = None
            bias = None
            total_rows = projection_input.size(0)
            for start in range(0, total_rows, qkv_up_proj_chunk_size):
                end = min(start + qkv_up_proj_chunk_size, total_rows)
                output_chunk, bias_chunk = projection(projection_input[start:end])
                input_chunk_rows = end - start
                direct_rows = input_chunk_rows
                gathered_rows = input_chunk_rows * self.tp_size
                if output_chunk.size(0) not in (direct_rows, gathered_rows):
                    raise RuntimeError(
                        'DSV4 QKV up projection chunking received an unsupported '
                        'sequence layout: '
                        f'input_rows={input_chunk_rows}, output_rows={output_chunk.size(0)}, '
                        f'tp={self.tp_size}'
                    )
                if output is None:
                    output_rows = total_rows if output_chunk.size(0) == direct_rows else total_rows * self.tp_size
                    output = torch.empty(
                        (output_rows, *output_chunk.shape[1:]),
                        dtype=output_chunk.dtype,
                        device=output_chunk.device,
                    )
                    bias = bias_chunk
                if output_chunk.size(0) == direct_rows:
                    output = copy_attention_chunk(output, output_chunk, start)
                else:
                    # TE sequence-parallel column projections return the
                    # gathered rows in rank-major order.  The local input
                    # chunk at offset ``start`` therefore contributes to
                    # global positions ``rank * total_rows + start``.
                    for rank in range(self.tp_size):
                        rank_chunk = output_chunk.narrow(
                            0, rank * input_chunk_rows, input_chunk_rows
                        )
                        output = copy_attention_chunk(
                            output,
                            rank_chunk,
                            rank * total_rows + start,
                        )
            return output, bias

        def qkv_up_proj_and_rope_apply(q_compressed,
                                       kv_compressed,
                                       rotary_pos_emb,
                                       boundary_kv_compressed=None,
                                       boundary_rotary_pos_emb=None):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [s, b, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [s, b, ...] or [t, ...] for two cases.
            """
            # q_compressed: [num_tokens, q_lora_rank]
            # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
            query_chunk_provider = None
            indexer_qr_chunk_provider = None
            stream_query = (
                os.environ.get('DSV4_STREAM_QKV_QUERY', '0').strip() == '1'
                and os.environ.get('DSV4_STREAM_CORE_OUTPUT', '0').strip() == '1'
                and packed_seq_params is not None
                and packed_seq_params.qkv_format == 'thd'
                and self.tp_size > 1
                and self.config.sequence_parallel
                and boundary_kv_compressed is None
                and not self.recompute_up_proj
            )

            if stream_query:
                local_q_rows = q_compressed.size(0)
                global_q_rows = local_q_rows * self.tp_size
                q_cu_seqlens = packed_seq_params.cu_seqlens_q

                def compressed_query_chunk_provider(start, end):
                    if start < 0 or end > global_q_rows or end <= start:
                        raise RuntimeError(
                            f'Invalid DSV4 streamed compressed-query chunk: '
                            f'start={start}, end={end}, local_rows={local_q_rows}, '
                            f'global_rows={global_q_rows}'
                        )
                    pieces = []
                    piece_start = start
                    while piece_start < end:
                        owner = piece_start // local_q_rows
                        piece_end = min(end, (owner + 1) * local_q_rows)
                        piece_rows = piece_end - piece_start
                        local_start = piece_start - owner * local_q_rows
                        qr_local = q_compressed.narrow(0, local_start, piece_rows)
                        qr_global = tensor_parallel.gather_from_sequence_parallel_region(
                            qr_local, group=self.pg_collection.tp
                        )
                        expected_rows = piece_rows * self.tp_size
                        if qr_global.size(0) != expected_rows:
                            raise RuntimeError(
                                'DSV4 streamed compressed-query gather returned '
                                f'an unexpected row count: got={qr_global.size(0)}, '
                                f'expected={expected_rows}'
                            )
                        pieces.append(
                            qr_global.narrow(0, owner * piece_rows, piece_rows)
                        )
                        piece_start = piece_end
                    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

                def query_chunk_provider(start, end):
                    if (
                        start < 0
                        or end > global_q_rows
                        or end <= start
                    ):
                        raise RuntimeError(
                            f'Invalid DSV4 streamed Q chunk: start={start}, end={end}, '
                            f'local_rows={local_q_rows}, global_rows={global_q_rows}'
                        )
                    pieces = []
                    piece_start = start
                    while piece_start < end:
                        owner = piece_start // local_q_rows
                        piece_end = min(end, (owner + 1) * local_q_rows)
                        local_start = piece_start - owner * local_q_rows
                        local_end = piece_end - owner * local_q_rows
                        projection_chunk_size = (
                            qkv_up_proj_chunk_size
                            if qkv_up_proj_chunk_size > 0
                            else local_end - local_start
                        )
                        for projection_start in range(
                            local_start, local_end, projection_chunk_size
                        ):
                            projection_end = min(
                                projection_start + projection_chunk_size, local_end
                            )
                            projection_rows = projection_end - projection_start
                            q_local, _ = self.linear_q_up_proj(
                                q_compressed.narrow(0, projection_start, projection_rows)
                            )
                            if q_local.size(0) == projection_rows:
                                q_local = tensor_parallel.gather_from_sequence_parallel_region(
                                    q_local, group=self.pg_collection.tp
                                )
                            expected_rows = projection_rows * self.tp_size
                            if q_local.size(0) != expected_rows:
                                raise RuntimeError(
                                    'DSV4 streamed Q projection returned an unexpected '
                                    f'row count: got={q_local.size(0)}, expected={expected_rows}'
                                )
                            q_chunk = q_local.narrow(
                                0, owner * projection_rows, projection_rows
                            )
                            q_chunk = q_chunk.view(
                                projection_rows,
                                self.num_attention_heads_per_partition,
                                self.q_head_dim,
                            )
                            q_chunk = _q_rms_norm(q_chunk, self.config.layernorm_epsilon)
                            pos_dim = self.config.qk_pos_emb_head_dim
                            q_no_pe, q_pos_emb = torch.split(
                                q_chunk, [q_chunk.size(-1) - pos_dim, pos_dim], dim=-1
                            )
                            global_start = owner * local_q_rows + projection_start
                            global_end = global_start + projection_rows
                            global_rows = torch.arange(
                                global_start,
                                global_end,
                                device=q_chunk.device,
                                dtype=q_cu_seqlens.dtype,
                            )
                            seq_ids = torch.bucketize(
                                global_rows,
                                q_cu_seqlens[1:],
                                out_int32=True,
                                right=True,
                            ).clamp_max(q_cu_seqlens.numel() - 2)
                            positions = global_rows - q_cu_seqlens[seq_ids]
                            positions = positions.clamp_min(0).clamp_max(rotary_pos_emb.size(0) - 1)
                            freqs_chunk = rotary_pos_emb.index_select(0, positions.long())
                            q_pos_emb = _apply_rotary_pos_emb_bshd(
                                q_pos_emb.contiguous(),
                                freqs_chunk,
                                rotary_interleaved=self.config.rotary_interleaved,
                                mla_rotary_interleaved=True,
                                mla_output_remove_interleaving=True,
                            )
                            pieces.append(torch.cat([q_no_pe, q_pos_emb], dim=-1).contiguous())
                        piece_start = piece_end
                    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

                query_chunk_provider.total_q = global_q_rows
                query_chunk_provider.num_heads = self.num_attention_heads_per_partition
                query_chunk_provider.head_dim = self.q_head_dim
                query_chunk_provider.device = q_compressed.device
                query_chunk_provider.dtype = q_compressed.dtype
                indexer_qr_chunk_provider = compressed_query_chunk_provider
                q = None
            else:
                q, _ = run_chunked_projection(self.linear_q_up_proj, q_compressed)
                if (
                    self.tp_size > 1
                    and self.config.sequence_parallel
                    and q.size(0) == q_compressed.size(0)
                ):
                    q = tensor_parallel.gather_from_sequence_parallel_region(
                        q, group=self.pg_collection.tp
                    )

                # q: [num_tokens, n, q_head_dim]
                q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)
                q = _q_rms_norm(q, self.config.layernorm_epsilon)

            # The Bridge path owns its RoPE application instead of calling the
            # MCore DSv4 implementation.  With sequence parallelism, q-up and
            # kv projection expose the TP-local sequence length only after the
            # projections; slice the global position table at that point.
            local_rotary_pos_emb = rotary_pos_emb
            local_kv_rotary_pos_emb = None
            if self.tp_size > 1 and self.config.sequence_parallel and packed_seq_params is None:
                tp_rank = torch.distributed.get_rank(group=self.pg_collection.tp)

                def _slice_tp_rope(tensor, local_seq_len):
                    if tensor is None or tensor.size(0) == local_seq_len:
                        return tensor
                    start = tp_rank * local_seq_len
                    if start + local_seq_len > tensor.size(0):
                        raise RuntimeError(
                            "DSv4 Bridge TP RoPE table is shorter than the local sequence slice: "
                            f"table={tensor.size(0)}, start={start}, local={local_seq_len}"
                        )
                    return tensor.narrow(0, start, local_seq_len)

                if q is not None:
                    local_rotary_pos_emb = _slice_tp_rope(local_rotary_pos_emb, q.size(0))

            boundary_rows = 0
            if boundary_kv_compressed is not None:
                boundary_rows = boundary_kv_compressed.shape[0]
                kv_projection_input = torch.cat([boundary_kv_compressed, kv_compressed], dim=0)
                kv_rotary_pos_emb = torch.cat([boundary_rotary_pos_emb, rotary_pos_emb], dim=0)
            else:
                kv_projection_input = kv_compressed
                kv_rotary_pos_emb = rotary_pos_emb
            _dsv4_log_bridge_kv('projection_input', self.layer_number, kv_projection_input)

            if boundary_kv_compressed is not None:
                # Boundary rows are a CP-only prefix.  Keep this uncommon
                # path on its original whole-tensor implementation until a
                # prefix-aware sequence-parallel gather is added.
                kv, _ = self.linear_kv_proj(kv_projection_input)
                kv_up_projection_is_gathered = False
            else:
                kv, _ = run_chunked_projection(self.linear_kv_proj, kv_projection_input)
                kv_up_projection_is_gathered = (
                    self.tp_size > 1
                    and self.config.sequence_parallel
                    and kv.size(0) == kv_compressed.size(0) * self.tp_size
                )
            _dsv4_log_bridge_kv('after_linear_kv_proj', self.layer_number, kv)
            kv = self.kv_layernorm(kv)
            _dsv4_log_bridge_kv('after_kv_layernorm', self.layer_number, kv)
            if self.tp_size > 1 and self.config.sequence_parallel:
                # q-up is a standard SP column-parallel projection and
                # therefore sees the complete CP-local sequence. The V4 KV
                # projection is duplicated, so gather its local output to the
                # same sequence space before CSA/attention.
                if boundary_rows:
                    boundary_kv_part = kv[:boundary_rows]
                    local_kv_part = tensor_parallel.gather_from_sequence_parallel_region(
                        kv[boundary_rows:], group=self.pg_collection.tp
                    )
                    kv = torch.cat([boundary_kv_part, local_kv_part], dim=0)
                elif not kv_up_projection_is_gathered:
                    kv = tensor_parallel.gather_from_sequence_parallel_region(
                        kv, group=self.pg_collection.tp
                    )
            _dsv4_log_bridge_kv('after_tp_kv_gather', self.layer_number, kv)
            local_kv_rotary_pos_emb = kv_rotary_pos_emb
            if self.tp_size > 1 and self.config.sequence_parallel and packed_seq_params is None:
                local_kv_rotary_pos_emb = _slice_tp_rope(local_kv_rotary_pos_emb, kv.size(0))
            boundary_kv = None
            pos_dim = self.config.qk_pos_emb_head_dim

            if q is not None:
                # q_no_pe: [num_tokens, n, qk_head_dim]
                # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                q_no_pe, q_pos_emb = torch.split(q, [q.shape[-1] - pos_dim, pos_dim], dim=-1)

                # RoPE and query (shared for wkv and latent)
                # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                q_pos_emb = apply_rotary_pos_emb(
                    q_pos_emb,
                    local_rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    cp_group=self.pg_collection.cp,
                    mla_rotary_interleaved=True,
                    mla_output_remove_interleaving=True,
                )
                # query: [num_tokens, n, (qk_head_dim + v_head_dim)]
                query = torch.cat([q_no_pe, q_pos_emb], dim=-1)
            else:
                query = None

            kv_no_pe, k_pos_emb = torch.split(kv, [kv.size(-1) - pos_dim, pos_dim], dim=-1)

            # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                local_kv_rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                cp_group=self.pg_collection.cp,
                mla_rotary_interleaved=True,
                mla_output_remove_interleaving=True,
            )

            # Single head: key = value = [num_tokens, 1, v_head_dim]
            kv = torch.cat([kv_no_pe, k_pos_emb], dim=-1).unsqueeze(-2)
            _dsv4_log_bridge_kv('after_kv_rope', self.layer_number, kv)
            if boundary_kv_compressed is not None:
                boundary_kv = kv[:boundary_rows]
                kv = kv[boundary_rows:]
            key = kv
            value = kv

            if query is not None:
                query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            if boundary_kv is not None:
                boundary_kv = boundary_kv.contiguous()
            if boundary_kv is None:
                result = (query, key, value)
            else:
                result = (query, key, value, boundary_kv)
            if query_chunk_provider is not None:
                result = result + (query_chunk_provider, )
            if indexer_qr_chunk_provider is not None:
                result = result + (indexer_qr_chunk_provider, )
            return result

        query_chunk_provider = None
        indexer_qr_chunk_provider = None
        if self.recompute_up_proj:
            quantization = self.config.fp8 or self.config.fp4
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=quantization)
            if boundary_hidden is None:
                query, key, value = self.qkv_up_checkpoint.checkpoint(qkv_up_proj_and_rope_apply, q_compressed,
                                                                      kv_compressed, rotary_pos_emb)
                boundary_kv = None
            else:
                query, key, value, boundary_kv = self.qkv_up_checkpoint.checkpoint(qkv_up_proj_and_rope_apply,
                                                                                   q_compressed, kv_compressed,
                                                                                   rotary_pos_emb, boundary_hidden,
                                                                                   boundary_rotary_pos_emb)
        else:
            if boundary_hidden is None:
                qkv_result = qkv_up_proj_and_rope_apply(
                    q_compressed, kv_compressed, rotary_pos_emb
                )
                query, key, value = qkv_result[:3]
                if len(qkv_result) > 3:
                    query_chunk_provider = qkv_result[3]
                if len(qkv_result) > 4:
                    indexer_qr_chunk_provider = qkv_result[4]
                boundary_kv = None
            else:
                query, key, value, boundary_kv = qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, rotary_pos_emb,
                                                                            boundary_hidden, boundary_rotary_pos_emb)

        result = (query, key, value, q_compressed, kv_compressed)
        if boundary_kv is not None:
            result = result + (boundary_kv, )
        if query_chunk_provider is not None:
            result = result + (query_chunk_provider, )
        if indexer_qr_chunk_provider is not None:
            result = result + (indexer_qr_chunk_provider, )
        return result

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """Forward pass for DeepSeek-v4 Hybrid Attention"""
        rotary_pos_emb = rotary_pos_emb[self.rope_layer_type]
        assert (attention_bias is None), 'Attention bias should not be passed into DSv4HybridAttention.'
        assert (rotary_pos_cos is None
                and rotary_pos_sin is None), 'DSv4HybridAttention does not support Flash Decoding'
        assert (not rotary_pos_cos_sin), 'Flash-infer rope has not been tested with DSv4HybridAttention.'
        assert (inference_context is None
                and inference_params is None), 'Inference is not supported for DSv4HybridAttention.'

        # Select this microbatch's dynamic CP group. QKV captures it explicitly
        # for recompute; the rest of this forward reads it from pg_collection.
        # Restore the static group before returning.
        cp_group = self.pg_collection.cp
        cp_size = cp_group.size()
        qkv_format = packed_seq_params.qkv_format if packed_seq_params is not None else None
        if cp_size > 1 and qkv_format != 'thd':
            raise ValueError("DSv4 Hybrid with CP requires qkv_format='thd'.")
        use_thd_cp = cp_size > 1 and qkv_format == 'thd'
        if use_thd_cp and packed_seq_params.cp_partition_mode != 'contiguous':
            raise ValueError('DSv4 THD CP requires a contiguous CP partition.')

        sequence_parallel_local_length = hidden_states.size(0)
        core_hidden_states = hidden_states
        if self.tp_size > 1 and self.config.sequence_parallel:
            # q-up uses TE's standard SP all-gather. The duplicated KV path
            # and CSA compressor need the same CP-local sequence explicitly.
            core_hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=self.pg_collection.tp
            )

        boundary_hidden = None
        boundary_rotary_pos_emb = None
        if use_thd_cp:
            # Keep Bridge on the same CP utility implementation as the V4 CSA
            # core.  The old top-level ``csa_cp_utils`` module imports the
            # removed dsa_kernels.indexer_topk symbol on the current MCore
            # backend; the maintained utility lives under csa_utils.cp_utils.
            from megatron.core.transformer.experimental_attention_variant.csa_utils import (
                cp_utils,
            )
            boundary_hidden = cp_utils.exchange_cp_boundary_hidden(
                core_hidden_states,
                self._dsv4_compress_ratio,
                self.config.csa_window_size,
                self.pg_collection.cp,
            )
            boundary_rotary_pos_emb = cp_utils.exchange_cp_boundary_hidden(
                rotary_pos_emb,
                self._dsv4_compress_ratio,
                self.config.csa_window_size,
                self.pg_collection.cp,
            )
        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        qkv = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            rotary_pos_emb=rotary_pos_emb,
            inference_context=inference_context,
            boundary_hidden=boundary_hidden,
            boundary_rotary_pos_emb=boundary_rotary_pos_emb,
        )
        # The streamed providers are returned only by the non-CP TP path.  A
        # TP×CP layout still needs these locals initialized before the common
        # sequence-parallel gather below; otherwise entering CP through the
        # `use_thd_cp` branch raises UnboundLocalError before attention.
        query_chunk_provider = None
        indexer_qr_chunk_provider = None
        if use_thd_cp:
            query, key, value, q_compressed, kv_compressed, boundary_kv = qkv
        else:
            query, key, value, q_compressed, kv_compressed = qkv[:5]
            query_chunk_provider = qkv[5] if len(qkv) > 5 else None
            indexer_qr_chunk_provider = qkv[6] if len(qkv) > 6 else None
            boundary_kv = None

        core_q_compressed = q_compressed
        if self.tp_size > 1 and self.config.sequence_parallel:
            # q-down is duplicated and remains TP-local, while CSA's learned
            # indexer consumes the same complete CP-local sequence as the
            # gathered hidden states. Keep the local q-compressed tensor for
            # q-up, and provide a gathered copy to the core-attention path.
            if indexer_qr_chunk_provider is None:
                core_q_compressed = tensor_parallel.gather_from_sequence_parallel_region(
                    q_compressed, group=self.pg_collection.tp
                )

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        if query is not None:
            query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        stream_core_output = (
            os.environ.get('DSV4_STREAM_CORE_OUTPUT', '0').strip() == '1'
            and self.tp_size > 1
            and self.config.sequence_parallel
            and packed_seq_params is not None
            and packed_seq_params.qkv_format == 'thd'
            and not use_thd_cp
            and core_hidden_states.size(0) == sequence_parallel_local_length * self.tp_size
            and not self.offload_core_attention
            and not self.offload_attn_proj
            and not self.recompute_up_proj
            and not self._o_group_proj_is_grouped_linear
            and copy_attention_chunk is not None
        )
        stream_output = None
        stream_bias = None

        if stream_core_output:
            # The CSA path normally returns one global [T, heads*D] buffer.
            # At 256K that buffer is itself a multi-GiB allocation even after
            # query/top-k chunking.  Consume each chunk through the exact
            # downstream projection chain while it is still small, retaining
            # only this rank's sequence-parallel output rows.
            stream_cu_seqlens = (
                packed_seq_params.cu_seqlens_kv_padded
                if packed_seq_params.cu_seqlens_kv_padded is not None
                else packed_seq_params.cu_seqlens_kv
            )
            stream_wo_a_weight = self.linear_o_group_proj.view(
                self.o_local_groups, self.config.o_lora_rank, -1
            )

            def consume_core_output(start, end, core_chunk):
                nonlocal stream_output, stream_bias
                chunk_rows = end - start
                if core_chunk.ndim == 2:
                    core_chunk = core_chunk.unsqueeze(1)
                if core_chunk.ndim != 3:
                    raise RuntimeError(
                        'DSV4 streamed CSA output must be [T, 1, H] or [T, H], '
                        f'got {tuple(core_chunk.shape)}'
                    )

                n_heads = self.num_attention_heads_per_partition
                pos_dim = self.config.qk_pos_emb_head_dim
                chunk_view = core_chunk.view(chunk_rows, core_chunk.size(1), n_heads, -1)
                content_chunk, rot_chunk = torch.split(
                    chunk_view,
                    [chunk_view.size(-1) - pos_dim, pos_dim],
                    dim=-1,
                )

                # Build document-relative positions for this global chunk.  A
                # packed batch can contain multiple documents; slicing the
                # frequency table by [start:end] would be wrong after a reset.
                global_rows = torch.arange(
                    start,
                    end,
                    device=rot_chunk.device,
                    dtype=stream_cu_seqlens.dtype,
                )
                seq_ids = torch.bucketize(
                    global_rows,
                    stream_cu_seqlens[1:],
                    out_int32=True,
                    right=True,
                ).clamp_max(stream_cu_seqlens.numel() - 2)
                positions = global_rows - stream_cu_seqlens[seq_ids]
                positions = positions.clamp_min(0).clamp_max(rotary_pos_emb.size(0) - 1)
                freqs_chunk = rotary_pos_emb.index_select(0, positions.long())

                rot_chunk = _apply_rotary_pos_emb_bshd(
                    rot_chunk.squeeze(1).contiguous(),
                    freqs_chunk,
                    rotary_interleaved=self.config.rotary_interleaved,
                    mla_rotary_interleaved=True,
                    inverse=True,
                    mla_output_remove_interleaving=True,
                ).unsqueeze(1)
                projected_input = torch.cat(
                    [content_chunk, rot_chunk], dim=-1
                ).reshape(chunk_rows, core_chunk.size(1), -1)
                projected_input = projected_input.view(
                    chunk_rows, core_chunk.size(1), self.o_local_groups, -1
                )
                projected_input = torch.einsum(
                    '...gd,grd->...gr', projected_input, stream_wo_a_weight
                ).reshape(chunk_rows, core_chunk.size(1), -1)
                # Every TP rank sees the same global chunk.  Disable SP just
                # for this short row-parallel call so TE returns the full
                # chunk after the TP reduction; slicing an SP reduce-scatter
                # result by ``start // tp`` would be wrong for chunks before
                # this rank's global sequence interval.
                sequence_parallel_attrs = []
                for module in self.linear_proj.modules():
                    if hasattr(module, 'sequence_parallel'):
                        sequence_parallel_attrs.append((module, module.sequence_parallel))
                        module.sequence_parallel = False
                try:
                    chunk_output, chunk_bias = self.linear_proj(projected_input)
                finally:
                    for module, value in sequence_parallel_attrs:
                        module.sequence_parallel = value

                if chunk_output.size(0) != chunk_rows:
                    raise RuntimeError(
                        'Streamed TP projection must return the full global chunk '
                        'when sequence_parallel is disabled: '
                        f'got={chunk_output.size(0)}, expected={chunk_rows}'
                    )
                local_global_start = self.pg_collection.tp.rank() * sequence_parallel_local_length
                local_global_end = local_global_start + sequence_parallel_local_length
                copy_start = max(start, local_global_start)
                copy_end = min(end, local_global_end)
                if copy_start >= copy_end:
                    return
                chunk_output = chunk_output.narrow(0, copy_start - start, copy_end - copy_start)
                destination_start = copy_start - local_global_start
                if stream_output is None:
                    stream_output = torch.empty(
                        (sequence_parallel_local_length, *chunk_output.shape[1:]),
                        dtype=chunk_output.dtype,
                        device=chunk_output.device,
                    )
                    stream_bias = chunk_bias
                stream_output = copy_attention_chunk(
                    stream_output, chunk_output, destination_start
                )

        core_attn_manager = off_interface(self.offload_core_attention and self.training, query, 'core_attn')
        with core_attn_manager as query:
            core_attn_kwargs = {}
            if boundary_hidden is not None:
                core_attn_kwargs['boundary_hidden'] = boundary_hidden
                core_attn_kwargs['boundary_kv'] = boundary_kv
            if stream_core_output:
                core_attn_kwargs['stream_output_consumer'] = consume_core_output
            if query_chunk_provider is not None:
                core_attn_kwargs['query_chunk_provider'] = query_chunk_provider
            if indexer_qr_chunk_provider is not None:
                core_attn_kwargs['indexer_qr_chunk_provider'] = indexer_qr_chunk_provider
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                x=core_hidden_states,
                qr=core_q_compressed,
                **core_attn_kwargs,
            )
        forced_released_tensors = [query, key, value]
        if boundary_kv is not None:
            forced_released_tensors.append(boundary_kv)
        if core_attn_out is not None:
            core_attn_out = core_attn_manager.group_offload(
                core_attn_out, forced_released_tensors=forced_released_tensors
            )
        else:
            # The streaming consumer has already run the inverse-RoPE and
            # output-projection chain for every chunk.  Still release the QKV
            # group through the manager before returning.
            core_attn_manager.group_offload(
                None, forced_released_tensors=forced_released_tensors
            )

        if stream_core_output:
            if stream_output is None:
                raise RuntimeError('DSV4 CSA output streaming produced no output chunks.')
            return stream_output, stream_bias

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # inverse RoPE on last qk_pos_emb_head_dim of each head
        seq_len = core_attn_out.size(0)
        n_heads = self.num_attention_heads_per_partition
        pos_dim = self.config.qk_pos_emb_head_dim
        core_attn_out = core_attn_out.view(seq_len, core_attn_out.size(1), n_heads, -1)
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if packed_seq:
            cu_seqlens_kv = (
                packed_seq_params.cu_seqlens_kv_padded
                if packed_seq_params.cu_seqlens_kv_padded is not None else packed_seq_params.cu_seqlens_kv)
        else:
            cu_seqlens_kv = None

        content_part, rot_part = torch.split(core_attn_out, [core_attn_out.size(-1) - pos_dim, pos_dim], dim=-1)
        use_recompute_rope_merge = (
            os.environ.get('DSV4_CSA_RECOMPUTE_OUT', '0').strip() == '1'
        )
        if packed_seq:
            rot_part_in = rot_part.squeeze(1)
        else:
            rot_part_in = rot_part
        if use_recompute_rope_merge:
            # The fused RoPE backward must retain the pre-rotation values
            # after the original attention buffer is overwritten.
            rot_part_in = rot_part_in.contiguous()
        rot_part_out = apply_rotary_pos_emb(
            rot_part_in,
            rotary_pos_emb,
            self.config,
            cu_seqlens=cu_seqlens_kv,
            cp_group=self.pg_collection.cp,
            mla_rotary_interleaved=True,
            inverse=True,
            mla_output_remove_interleaving=True,
        )
        if packed_seq:
            rot_part = rot_part_out.unsqueeze(1)
        else:
            rot_part = rot_part_out
        if use_recompute_rope_merge:
            core_attn_out = merge_rotary_output(
                core_attn_out, rot_part, content_part.size(-1)
            )
        else:
            core_attn_out = torch.cat([content_part, rot_part], dim=-1)
        core_attn_out = core_attn_out.view(seq_len, core_attn_out.size(1), -1)

        # Grouped output
        if self._o_group_proj_is_grouped_linear:
            s, b = core_attn_out.size(0), core_attn_out.size(1)
            # [s, b, G*D] -> [G, s*b, D] -> [G*s*b, D]
            core_attn_out = core_attn_out.view(s, b, self.o_local_groups, -1)
            core_attn_out = core_attn_out.permute(2, 0, 1, 3).contiguous()
            core_attn_out = core_attn_out.reshape(-1, core_attn_out.size(-1))
            m_splits = [s * b] * self.o_local_groups
            core_attn_out = self.linear_o_group_proj(core_attn_out, m_splits)
            # [G*s*b, R] -> [G, s, b, R] -> [s, b, G*R]
            core_attn_out = core_attn_out.view(self.o_local_groups, s, b, -1)
            core_attn_out = core_attn_out.permute(1, 2, 0, 3).contiguous()
            core_attn_out = core_attn_out.reshape(s, b, -1)
        else:
            core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), self.o_local_groups, -1)
            wo_a_weight = self.linear_o_group_proj.view(self.o_local_groups, self.config.o_lora_rank, -1)
            try:
                output_chunk_size = int(os.environ.get('DSV4_TP_O_PROJ_CHUNK_SIZE', '0') or 0)
            except ValueError:
                output_chunk_size = 0
            try:
                linear_proj_chunk_hint = int(
                    os.environ.get('DSV4_TP_LINEAR_PROJ_CHUNK_SIZE', '0') or 0
                )
            except ValueError:
                linear_proj_chunk_hint = 0
            defer_wo_a_projection = (
                output_chunk_size > 0
                and linear_proj_chunk_hint > 0
                and core_attn_out.size(0) > output_chunk_size
                and self.tp_size > 1
                and self.config.sequence_parallel
            )
            if output_chunk_size > 0 and core_attn_out.size(0) > output_chunk_size:
                if copy_attention_chunk is None:
                    raise RuntimeError(
                        'DSV4_TP_O_PROJ_CHUNK_SIZE requires the MCore copy_attention_chunk helper.'
                    )
                if not defer_wo_a_projection:
                    projected = None
                    for start in range(0, core_attn_out.size(0), output_chunk_size):
                        end = min(start + output_chunk_size, core_attn_out.size(0))
                        projected_chunk = torch.einsum(
                            '...gd,grd->...gr', core_attn_out[start:end], wo_a_weight
                        )
                        if projected is None:
                            projected = torch.empty(
                                (core_attn_out.size(0), projected_chunk.size(1), projected_chunk.size(2),
                                 projected_chunk.size(3)),
                                dtype=projected_chunk.dtype,
                                device=projected_chunk.device,
                            )
                        projected = copy_attention_chunk(projected, projected_chunk, start)
                    core_attn_out = projected
            else:
                core_attn_out = torch.einsum('...gd,grd->...gr', core_attn_out, wo_a_weight)
            if not defer_wo_a_projection:
                core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)

        # =================
        # Output. [sq, b, h]
        # =================
        attn_proj_manager = off_interface(self.offload_attn_proj, core_attn_out, 'attn_proj')
        with attn_proj_manager as core_attn_out:
            try:
                linear_proj_chunk_size = int(
                    os.environ.get('DSV4_TP_LINEAR_PROJ_CHUNK_SIZE', '0') or 0
                )
            except ValueError:
                linear_proj_chunk_size = 0

            # TE's row-parallel projection performs the TP reduce-scatter in one
            # call.  At very long sequence lengths its temporary collective
            # buffer can be larger than the remaining headroom even when the
            # preceding MLA workspaces have been chunked.  Run the same module
            # on sequence-aligned pieces and copy the local sequence outputs
            # into one autograd-connected buffer.  This is opt-in because it
            # trades one large GEMM/collective for several smaller ones.
            should_chunk_linear_proj = (
                linear_proj_chunk_size > 0
                and core_attn_out.size(0) > linear_proj_chunk_size
                and self.tp_size > 1
                and self.config.sequence_parallel
            )
            if defer_wo_a_projection:
                if copy_attention_chunk is None:
                    raise RuntimeError(
                        'DSV4_TP_LINEAR_PROJ_CHUNK_SIZE requires the MCore '
                        'copy_attention_chunk helper.'
                    )
                # Fuse the low-rank wo_a projection with the row-parallel
                # projection.  Keeping only one sequence piece alive is what
                # removes the otherwise unavoidable full [S, B, G, R] buffer.
                chunk_size = min(linear_proj_chunk_size, output_chunk_size)
                total_rows = core_attn_out.size(0)
                remainder = total_rows % chunk_size
                if chunk_size % self.tp_size != 0 or total_rows % self.tp_size != 0 or (
                    remainder and remainder % self.tp_size != 0
                ):
                    raise ValueError(
                        'TP output projection chunks must produce TP-aligned pieces: '
                        f'chunk={chunk_size}, sequence={total_rows}, tp={self.tp_size}'
                    )

                output = None
                bias = None
                output_start = 0
                for start in range(0, total_rows, chunk_size):
                    end = min(start + chunk_size, total_rows)
                    projected_chunk = torch.einsum(
                        '...gd,grd->...gr', core_attn_out[start:end], wo_a_weight
                    ).reshape(end - start, core_attn_out.size(1), -1)
                    chunk_output, chunk_bias = self.linear_proj(projected_chunk)
                    if output is None:
                        if (total_rows * chunk_output.size(0)) % (end - start) != 0:
                            raise ValueError(
                                'TP output projection shape is not integral for '
                                f'chunk={chunk_size}, input={total_rows}, '
                                f'first_output_rows={chunk_output.size(0)}'
                            )
                        output = torch.empty(
                            (
                                total_rows * chunk_output.size(0) // (end - start),
                                *chunk_output.shape[1:],
                            ),
                            dtype=chunk_output.dtype,
                            device=chunk_output.device,
                        )
                        bias = chunk_bias
                    output = copy_attention_chunk(output, chunk_output, output_start)
                    output_start += chunk_output.size(0)
                assert output is not None
                assert output_start == output.size(0)
            elif should_chunk_linear_proj:
                if copy_attention_chunk is None:
                    raise RuntimeError(
                        'DSV4_TP_LINEAR_PROJ_CHUNK_SIZE requires the MCore '
                        'copy_attention_chunk helper.'
                    )
                total_rows = core_attn_out.size(0)
                # Sequence-parallel reduce-scatter requires every input piece
                # to be divisible by TP.  The final tail may be shorter than
                # the requested chunk, but it must remain TP-aligned.
                remainder = total_rows % linear_proj_chunk_size
                if linear_proj_chunk_size % self.tp_size != 0 or total_rows % self.tp_size != 0 or (
                    remainder and remainder % self.tp_size != 0
                ):
                    raise ValueError(
                        'DSV4_TP_LINEAR_PROJ_CHUNK_SIZE must produce TP-aligned '
                        f'pieces: chunk={linear_proj_chunk_size}, '
                        f'sequence={total_rows}, tp={self.tp_size}'
                    )

                output = None
                bias = None
                output_start = 0
                for start in range(0, total_rows, linear_proj_chunk_size):
                    end = start + linear_proj_chunk_size
                    chunk_output, chunk_bias = self.linear_proj(core_attn_out[start:end])
                    if output is None:
                        if (total_rows * chunk_output.size(0)) % linear_proj_chunk_size != 0:
                            raise ValueError(
                                'TP linear projection output shape is not integral for '
                                f'chunk={linear_proj_chunk_size}, input={total_rows}, '
                                f'first_output_rows={chunk_output.size(0)}'
                            )
                        output = torch.empty(
                            (
                                total_rows * chunk_output.size(0) // linear_proj_chunk_size,
                                *chunk_output.shape[1:],
                            ),
                            dtype=chunk_output.dtype,
                            device=chunk_output.device,
                        )
                        bias = chunk_bias
                    output = copy_attention_chunk(output, chunk_output, output_start)
                    output_start += chunk_output.size(0)
                assert output is not None
                assert output_start == output.size(0)
            else:
                output, bias = self.linear_proj(core_attn_out)
        output = attn_proj_manager.group_offload(output, forced_released_tensors=[core_attn_out])

        # Some TE versions reduce the row-parallel output across TP but leave
        # the sequence dimension gathered. Restore the standard
        # sequence-parallel module contract (local sequence on return) when
        # that backend behavior is observed; do not double-scatter versions
        # that already return the local shape.
        if (
            self.tp_size > 1
            and self.config.sequence_parallel
            and output.size(0) != sequence_parallel_local_length
            and output.size(0) % self.tp_size == 0
        ):
            output = tensor_parallel.scatter_to_sequence_parallel_region(
                output, group=self.pg_collection.tp
            )

        return output, bias


class DeepseekV4GPTModel(GPTModel):

    def _init_mla_softmax_scale(self, config):
        pass

    def _get_rotary_pos_emb(self, decoder_input, position_ids, packed_seq_params, inference_context=None):
        rotary_seq_len = RotaryEmbedding.get_rotary_seq_len(self, inference_context, self.decoder, decoder_input,
                                                            self.config, packed_seq_params)
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        compress_rotary_pos_emb = self.compress_rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        rotary_pos_emb = {'main': rotary_pos_emb, 'compress': compress_rotary_pos_emb}
        return rotary_pos_emb, None, None

    def _set_inv_freq(self):
        rope_scaling = self.config.rope_scaling
        self.config.rope_scaling = rope_scaling['main']
        new_inv_freq, attention_scaling = get_rope_inv_freq(self.config)
        self.rotary_pos_emb.inv_freq = new_inv_freq.to(self.rotary_pos_emb.inv_freq.device)
        self.config.attention_scaling = attention_scaling
        # compress
        self.compress_rotary_pos_emb = copy.copy(self.rotary_pos_emb)
        self.config.rope_scaling = rope_scaling['compress']
        new_inv_freq, attention_scaling = get_rope_inv_freq(self.config)
        self.compress_rotary_pos_emb.inv_freq = new_inv_freq
        self.config.compress_attention_scaling = attention_scaling

        self.config.rope_scaling = rope_scaling


class DeepseekV4Loader(ModelLoader):
    model_cls = DeepseekV4GPTModel

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import \
            get_transformer_block_with_experimental_attention_variant_spec
        transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(self.config, vp_stage)
        for layer_spec in transformer_layer_spec.layer_specs:
            layer_spec.submodules.self_attention.module = DSv4HybridSelfAttention
            core_attention_submodules = layer_spec.submodules.self_attention.submodules.core_attention.submodules
            if getattr(core_attention_submodules, 'compressor', None) is not None:
                core_attention_submodules.compressor.module = Compressor
            if getattr(core_attention_submodules, 'indexer', None) is not None:
                core_attention_submodules.indexer.module = CSAIndexer
                core_attention_submodules.indexer.submodules.compressor.module = Compressor
        return transformer_layer_spec


class DeepseekV4Bridge(GPTBridge):
    hf_mtp_prefix = 'model.mtp'
    hf_embed_key = 'model.embed.weight'
    hf_attn_prefix = 'attn'
    hf_mlp_prefix = 'ffn'
    hf_lm_head_key = 'model.head.weight'
    hf_score_key = 'model.score.weight'
    hf_input_layernorm_key = 'attn_norm.weight'
    hf_post_attention_layernorm_key = 'ffn_norm.weight'
    hf_expert_bias_key = 'gate.bias'

    def _set_o_group_proj_grouped(self, mg_attn, hf_state_dict, to_mcore):
        """Handle GroupedLinear state dict for linear_o_group_proj in fp8 mode.

        HF stores a single wo_a.weight of shape [G*R, D].
        GroupedLinear stores per-gemm weight{i} each of shape [R, D].
        """
        o_groups = self.config.o_groups
        local_groups = o_groups // self.tp_size
        if to_mcore:
            hf_weight = hf_state_dict['wo_a.weight'].load()
            hf_scale_inv = None
            if 'wo_a.weight_scale_inv' in hf_state_dict:
                hf_scale_inv = hf_state_dict['wo_a.weight_scale_inv'].load()
            weights = hf_weight.chunk(o_groups, dim=0)
            scale_invs = hf_scale_inv.chunk(o_groups, dim=0) if hf_scale_inv is not None else [None] * o_groups
            start = self.tp_rank * local_groups
            for i, (w, s) in enumerate(zip(weights[start:start + local_groups],
                                             scale_invs[start:start + local_groups])):
                param = getattr(mg_attn.linear_o_group_proj, f'weight{i}')
                self._set_param(param, w, s)
        else:
            if mg_attn is None:
                mg_weight = None
            else:
                mg_weight = [getattr(mg_attn.linear_o_group_proj, f'weight{i}') for i in range(local_groups)]
            weight, scale_inv = self._get_weight(mg_weight, 'linear_o_group_proj.weight0')
            if weight is not None:
                hf_state_dict['wo_a.weight'] = weight
            if scale_inv is not None:
                hf_state_dict['wo_a.weight_scale_inv'] = scale_inv

    def _convert_hf_state_dict(self, hf_state_dict, to_mcore):
        res = super()._convert_hf_state_dict(hf_state_dict, to_mcore)
        if to_mcore:
            res = self._add_prefix(res, 'model.')
            new_res = {}
            for k, v in res.items():
                if k.endswith('.scale'):
                    k = k[:-len('.scale')] + '.weight_scale_inv'
                new_res[k] = v
            res = new_res
        else:
            res = self._remove_prefix(res, 'model.')
            new_res = {}
            for k, v in res.items():
                if k.endswith('.weight_scale_inv'):
                    k = k[:-len('.weight_scale_inv')] + '.scale'
                new_res[k] = v
            res = new_res
        return res

    def _set_moe_state(
        self,
        mg_mlp,
        hf_state_dict,
        hf_prefix: str,
        layer_idx: int,
        to_mcore: bool,
        is_mtp: bool = False,
    ):
        if to_mcore:
            hf_state_dict = {
                k.replace('.w1.', '.gate_proj.').replace('.w3.', '.up_proj.').replace('.w2.', '.down_proj.'): v
                for k, v in hf_state_dict.items()
            }
        hf_state_dict = super()._set_moe_state(mg_mlp, hf_state_dict, hf_prefix, layer_idx, to_mcore, is_mtp)
        if not to_mcore:
            hf_state_dict = {
                k.replace('.gate_proj.', '.w1.').replace('.up_proj.', '.w3.').replace('.down_proj.', '.w2.'): v
                for k, v in hf_state_dict.items()
            }
        return hf_state_dict

    def _set_mla_attn_state(
        self,
        mg_attn,
        hf_state_dict,
        hf_prefix: str,
        layer_idx: int,
        to_mcore: bool,
    ):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        self._set_state_dict(mg_attn, 'linear_proj.weight', hf_state_dict, 'wo_b.weight', to_mcore)
        if self.config.fp8_param:
            self._set_o_group_proj_grouped(mg_attn, hf_state_dict, to_mcore)
        else:
            self._set_state_dict(mg_attn, 'linear_o_group_proj', hf_state_dict, 'wo_a.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_q_down_proj.weight', hf_state_dict, 'wq_a.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_q_up_proj.weight', hf_state_dict, 'wq_b.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_kv_proj.weight', hf_state_dict, 'wkv.weight', to_mcore)
        self._set_state_dict(mg_attn, 'core_attention.attn_sink', hf_state_dict, 'attn_sink', to_mcore)
        if self.config.qk_layernorm:
            self._set_state_dict(mg_attn, 'q_layernorm.weight', hf_state_dict, 'q_norm.weight', to_mcore)
            self._set_state_dict(mg_attn, 'kv_layernorm.weight', hf_state_dict, 'kv_norm.weight', to_mcore)
        has_compressor = False if mg_attn is None else mg_attn.core_attention.compressor is not None
        has_indexer = False if mg_attn is None else mg_attn.core_attention.indexer is not None
        has_compressor = self._reduce_tensor_pp_group(has_compressor, to_mcore)
        has_indexer = self._reduce_tensor_pp_group(has_indexer, to_mcore)
        if has_compressor:
            for mg_key, hf_key in zip(['ape', 'linear_wkv.weight', 'linear_wgate.weight', 'norm.weight'],
                                      ['ape', 'wkv.weight', 'wgate.weight', 'norm.weight']):
                self._set_state_dict(mg_attn, f'core_attention.compressor.{mg_key}', hf_state_dict,
                                     f'compressor.{hf_key}', to_mcore)
        if has_indexer:
            for mg_key, hf_key in zip(['linear_wq_b.weight', 'linear_weights_proj.weight'],
                                      ['wq_b.weight', 'weights_proj.weight']):
                self._set_state_dict(mg_attn, f'core_attention.indexer.{mg_key}', hf_state_dict, f'indexer.{hf_key}',
                                     to_mcore)
            for mg_key, hf_key in zip(['ape', 'linear_wkv.weight', 'linear_wgate.weight', 'norm.weight'],
                                      ['ape', 'wkv.weight', 'wgate.weight', 'norm.weight']):
                self._set_state_dict(mg_attn, f'core_attention.indexer.compressor.{mg_key}', hf_state_dict,
                                     f'indexer.compressor.{hf_key}', to_mcore)

        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_final_layernorm(self, lm_model, hf_state_dict, to_mcore):
        super()._set_final_layernorm(lm_model, hf_state_dict, to_mcore)
        for key in ['hc_head_base', 'hc_head_fn', 'hc_head_scale']:
            self._set_state_dict(lm_model, f'decoder.{key}', hf_state_dict, f'model.{key}', to_mcore)

    def _set_router(self, mg_mlp, hf_state_dict, to_mcore, **kwargs):
        is_hash_layer = False if mg_mlp is None else mg_mlp.router.is_hash_layer
        is_hash_layer = self._reduce_tensor_pp_group(is_hash_layer, to_mcore)
        if is_hash_layer:
            self._set_state_dict(mg_mlp, 'router.tid2eid', hf_state_dict, 'gate.tid2eid', to_mcore)
            kwargs['moe_router_enable_expert_bias'] = False
        super()._set_router(mg_mlp, hf_state_dict, to_mcore, **kwargs)

    def _convert_mtp_extra(self, mtp_layer, hf_state_dict, to_mcore, origin_hf_state_dict):
        for key in ['enorm.weight', 'hnorm.weight', 'e_proj.weight', 'h_proj.weight']:
            self._set_state_dict(mtp_layer, key, hf_state_dict, key, to_mcore)
        self._set_state_dict(mtp_layer, 'final_layernorm.weight', hf_state_dict, 'norm.weight', to_mcore)
        for key in ['hc_head_base', 'hc_head_fn', 'hc_head_scale']:
            self._set_state_dict(mtp_layer, key, hf_state_dict, key, to_mcore)

    def _convert_mtp_embeds(self, lm_model, hf_state_dict, to_mcore):
        if not to_mcore:
            self._set_state_dict(lm_model, 'embedding.word_embeddings.weight', hf_state_dict, 'emb.tok_emb.weight',
                                 to_mcore)
            if self.config.untie_embeddings_and_output_weights:
                self._set_state_dict(lm_model, 'output_layer.weight', hf_state_dict, 'head.weight', to_mcore)

    def _set_param(self, param, tensor, scale_inv):
        is_fp4 = tensor.dtype == torch.int8 and tensor.shape[-1] * 2 == param.shape[-1]
        if not is_fp4:
            return super()._set_param(param, tensor, scale_inv)
        tensor = fp4_to_fp8(tensor)
        tensor = tensor.reshape(*param.shape)
        scale_inv = scale_inv.reshape(-1, scale_inv.shape[-1])
        tensor = Fp8Dequantizer(block_size='auto').convert(tensor, scale_inv)
        if self._is_fp8_param(param):
            param._high_precision_init_val.copy_(tensor)
        param.data.copy_(tensor)


register_model(
    ModelMeta(
        ModelType.deepseek_v4,
        ['deepseek_v4'],
        bridge_cls=DeepseekV4Bridge,
        loader=DeepseekV4Loader,
    ))
