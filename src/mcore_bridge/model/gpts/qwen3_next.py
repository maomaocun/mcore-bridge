# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from copy import deepcopy
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, _get_extra_te_kwargs
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.huggingface import HuggingFaceModule as _HuggingFaceModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_tensor_model_parallel_rank
from megatron.core.tensor_parallel import (all_gather_last_dim_from_tensor_parallel_region,
                                           gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region)
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.utils import deprecate_inference_params, is_fa_min_version
from transformers.utils import is_torch_npu_available
from typing import Optional, Tuple, Union

from mcore_bridge.bridge import GPTBridge
from mcore_bridge.config import ModelConfig
from mcore_bridge.utils import get_env_args, get_local_layer_specs, get_logger

from ..constant import ModelType
from ..modules.qsa_attention import qsa_sparse_forward
from ..register import ModelLoader, ModelMeta, register_model

try:
    from flashattn_hopper.flash_attn_interface import _flash_attn_forward
    from flashattn_hopper.flash_attn_interface import flash_attn_with_kvcache as flash_attn3_with_kvcache

    HAVE_FA3 = True
except Exception:
    HAVE_FA3 = False

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet as _Qwen3NextGatedDeltaNet
except ImportError:
    _Qwen3NextGatedDeltaNet = object

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None

logger = get_logger()


def resolve_gdn_attention_mask(kwargs) -> Optional[torch.Tensor]:
    if is_torch_npu_available():
        attention_mask = kwargs.get('attention_mask_2d')
        if attention_mask is not None:
            return attention_mask.to(torch.bool)
    attention_mask = kwargs.get('attention_mask')
    if attention_mask is None:
        return None
    return (~attention_mask).sum(dim=(1, 2)) > 0


class Qwen3NextRMSNorm(torch.nn.Module):
    """
    Zero-Centered RMSNorm for Qwen3-Next.
    Uses (1 + weight) scaling to match HuggingFace implementation exactly.
    This eliminates the need for +1/-1 offset during weight conversion.

    Interface matches TENorm for compatibility with Megatron-Core build_module.
    """

    def __init__(self, config: ModelConfig, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.config = config
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size, dtype=config.params_dtype))
        # Mark weight for SP gradient AllReduce across TP domain (consistent with TENorm/MCoreRMSNorm)
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, hidden_states):
        output = self._norm(hidden_states.float())
        # Zero-Centered: use (1 + weight) instead of weight
        # This matches HuggingFace's Qwen3NextRMSNorm exactly
        output = output * (1.0 + self.weight.float())
        return output.type_as(hidden_states)


class Qwen3NextSelfAttention(SelfAttention):

    def __init__(self, config: ModelConfig, submodules: SelfAttentionSubmodules, *args, **kwargs):
        super(SelfAttention, self).__init__(config, submodules, *args, attention_type='self', **kwargs)
        kwargs = {}
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            2 * self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
            tp_group=self.pg_collection.tp,
            **kwargs,
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    # Code borrowed from NVIDIA/Megatron-LM
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        qsa_topk_indices: Optional[torch.Tensor] = None,
        qsa_topk_length: Optional[torch.Tensor] = None,
        qsa_kernel_backend: Optional[str] = None,
        qsa_query_position_offset: int = 0,
        qsa_key_position_offset: int = 0,
        qsa_query_positions: Optional[torch.Tensor] = None,
        qsa_global_kv: bool = False,
        qsa_cp_exchange: bool = False,
        qsa_global_seq_len: Optional[int] = None,
        qsa_route_block_size: int = 1,
        qsa_tp_sp_hidden_states: Optional[torch.Tensor] = None,
        qsa_tp_sp_rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """
        try:
            from megatron.core.utils import nvtx_range_pop, nvtx_range_push
        except ImportError:

            def nvtx_range_pop(*args, **kwargs):
                return

            def nvtx_range_push(*args, **kwargs):
                return

        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        if hasattr(self.config, 'no_rope_freq'):
            no_rope = (self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False)
            if no_rope:
                rotary_pos_emb = None

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert HAVE_FA3 or is_fa_min_version(
                '2.7.3'), 'flash attn verion v2.7.3 and above is required for dynamic batching.'

        # hidden_states: [sq, b, h]
        if self.config.flash_decode and not self.training and inference_context is not None:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb, ) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        if (qsa_topk_indices is not None and self.config.sequence_parallel
                and self.config.tensor_model_parallel_size > 1):
            if packed_seq_params is not None:
                raise ValueError('QSA TP+SP phase-1 path supports SBHD/unpacked input only')
            # TP+SP phase-1 correctness path. Sequence parallelism gives the
            # layer a local sequence shard while TP keeps local Q/KV heads.
            # Gather the sequence only inside the QSA attention layer, so the
            # surrounding MLP/GDN and the output projection remain
            # sequence-sharded. A later owner-exchange path can remove this
            # full-sequence temporary without changing the route contract.
            local_sequence = hidden_states.shape[0]
            target_sequence = int(qsa_topk_indices.shape[1])
            if qsa_tp_sp_hidden_states is not None:
                hidden_states = qsa_tp_sp_hidden_states
                if qsa_tp_sp_rotary_pos_emb is not None:
                    rotary_pos_emb = qsa_tp_sp_rotary_pos_emb
                    if not isinstance(rotary_pos_emb, tuple):
                        rotary_pos_emb = (rotary_pos_emb,) * 2
                target_sequence = hidden_states.shape[0]
            if qsa_tp_sp_hidden_states is None and local_sequence != target_sequence:
                if local_sequence * self.config.tensor_model_parallel_size != target_sequence:
                    raise ValueError(
                        'QSA TP+SP route/activation sequence mismatch: '
                        f'local={local_sequence}, route={target_sequence}, '
                        f'tp={self.config.tensor_model_parallel_size}')
                hidden_states = gather_from_sequence_parallel_region(
                    hidden_states,
                    tensor_parallel_output_grad=False,
                    group=self.pg_collection.tp,
                )
            if rotary_pos_emb is not None:
                def gather_rope(rope):
                    if rope is None or rope.shape[0] == target_sequence:
                        return rope
                    if rope.shape[0] * self.config.tensor_model_parallel_size != target_sequence:
                        raise ValueError(
                            'QSA TP+SP rope/route sequence mismatch: '
                            f'rope={rope.shape[0]}, route={target_sequence}, '
                            f'tp={self.config.tensor_model_parallel_size}')
                    return gather_from_sequence_parallel_region(
                        rope,
                        tensor_parallel_output_grad=False,
                        group=self.pg_collection.tp,
                    )
                if isinstance(rotary_pos_emb, tuple):
                    rotary_pos_emb = tuple(gather_rope(rope) for rope in rotary_pos_emb)
                else:
                    rotary_pos_emb = gather_rope(rotary_pos_emb)
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        nvtx_range_push(suffix='qkv')
        qsa_tp_sp_full_sequence = (
            qsa_topk_indices is not None
            and qsa_tp_sp_hidden_states is not None
            and self.config.sequence_parallel
            and self.config.tensor_model_parallel_size > 1
        )
        saved_qkv_sequence_parallel = None
        if qsa_tp_sp_full_sequence and hasattr(self.linear_qkv, 'sequence_parallel'):
            # TE ColumnParallelLinear assumes a sequence-parallel input when
            # this flag is true and performs an internal sequence gather. The
            # QSA phase-1 adapter has already supplied the full sequence, so
            # disable only this projection's extra gather. Its backward still
            # uses the regular TP dgrad reduction; linear_proj below retains
            # sequence_parallel=True and reduce-scatters the final output.
            saved_qkv_sequence_parallel = self.linear_qkv.sequence_parallel
            self.linear_qkv.sequence_parallel = False
        try:
            query, key, value, gate = self.get_query_key_value_tensors(hidden_states, key_value_states)
        finally:
            if saved_qkv_sequence_parallel is not None:
                self.linear_qkv.sequence_parallel = saved_qkv_sequence_parallel
        nvtx_range_pop(suffix='qkv')

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        in_decode_mode = (inference_context is not None and inference_context.is_decode_only() and not self.training)

        # This branch only runs in the decode phase of flash decoding and returns after the linear
        # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
        nvtx_range_push(suffix='adjust_key_value')
        if in_decode_mode and self.config.flash_decode:
            assert self.layer_number in inference_context.key_value_memory_dict
            assert inference_context.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[self.layer_number]
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
                rotary_interleaved=self.config.rotary_interleaved,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            output, bias = self.linear_proj(context_layer)
            return output, bias

        if (in_decode_mode and self.config.enable_cuda_graph and inference_context.is_static_batching()):
            raise ValueError('CUDA graphs must use flash decode with static batching!')

        query, key, value, rotary_pos_emb, attn_mask_type, block_table = self._adjust_key_value_for_inference(
            inference_context,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        nvtx_range_pop(suffix='adjust_key_value')

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        nvtx_range_push(suffix='rotary_pos_emb')
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                # TODO VIJAY: simplify
                if inference_context is None or inference_context.is_static_batching():
                    query = apply_rotary_pos_emb(
                        query,
                        q_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        cp_group=self.pg_collection.cp,
                    )
                else:
                    query = inference_context.apply_rotary_emb_query(
                        query, q_pos_emb, self.config, cu_seqlens_q, cp_group=self.pg_collection.cp)
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key,
                    k_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    cp_group=self.pg_collection.cp,
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        nvtx_range_pop(suffix='rotary_pos_emb')

        # ==================================
        # core attention computation
        # ==================================

        nvtx_range_push(suffix='core_attention')
        qsa_selected_kv = qsa_topk_indices is not None or qsa_topk_length is not None
        if qsa_selected_kv:
            if qsa_topk_indices is None or qsa_topk_length is None:
                raise ValueError('QSA selected-KV attention requires both qsa_topk_indices and qsa_topk_length')
            # SelfAttention normally derives this scale inside DotProductAttention.
            # Keep the same value when MCore exposes it, while retaining a
            # version-independent fallback for older Megatron-Core releases.
            softmax_scale = getattr(self, 'softmax_scale', None)
            if softmax_scale is None:
                softmax_scale = getattr(self.core_attention, 'scale', None)
            if softmax_scale is None:
                softmax_scale = self.hidden_size_per_attention_head**-0.5
            if packed_seq_params is not None:
                from ..modules.qsa_attention import qsa_sparse_forward_packed

                core_attn_out, _ = qsa_sparse_forward_packed(
                    query,
                    key,
                    value,
                    qsa_topk_indices,
                    qsa_topk_length,
                    packed_seq_params.cu_seqlens_q,
                    packed_seq_params.cu_seqlens_kv,
                    softmax_scale=softmax_scale,
                    backend=qsa_kernel_backend or 'torch',
                    query_tile_size=getattr(self.config, 'qsa_attention_query_tile_size', 16),
                    require_backend=getattr(self.config, 'require_qsa_kernel', False),
                    dkv_accum_dtype=getattr(self.config, 'qsa_dkv_accum_dtype', 'bf16'),
                    dkv_reduction=getattr(self.config, 'qsa_dkv_reduction', 'atomic'),
                    selected_token_group_size=getattr(self.config, 'indexer_compress_ratio', None),
                    route_block_size=qsa_route_block_size,
                )
            else:
                if qsa_global_kv:
                    from ..modules.qsa_attention import qsa_reconstruct_cp_tensor

                    key = qsa_reconstruct_cp_tensor(
                        key, self.pg_collection.cp, getattr(self.config, 'cp_partition_mode', 'zigzag'))
                    value = qsa_reconstruct_cp_tensor(
                        value, self.pg_collection.cp, getattr(self.config, 'cp_partition_mode', 'zigzag'))
                elif qsa_cp_exchange:
                    from ..modules.qsa_cp_exchange import qsa_exchange_selected_kv

                    key, value, qsa_topk_indices = qsa_exchange_selected_kv(
                        key,
                        value,
                        qsa_topk_indices,
                        qsa_topk_length,
                        qsa_global_seq_len or query.shape[0],
                        self.pg_collection.cp,
                        getattr(self.config, 'cp_partition_mode', 'zigzag'),
                    )
                core_attn_out, _ = qsa_sparse_forward(
                    query,
                    key,
                    value,
                    qsa_topk_indices,
                    qsa_topk_length,
                    softmax_scale=softmax_scale,
                    causal=not qsa_cp_exchange,
                    backend=qsa_kernel_backend or 'torch',
                    query_position_offset=qsa_query_position_offset,
                    key_position_offset=qsa_key_position_offset,
                    query_positions=qsa_query_positions,
                    query_tile_size=getattr(self.config, 'qsa_attention_query_tile_size', 16),
                    require_backend=getattr(self.config, 'require_qsa_kernel', False),
                    dkv_accum_dtype=getattr(self.config, 'qsa_dkv_accum_dtype', 'bf16'),
                    dkv_reduction=(
                        'atomic' if qsa_cp_exchange else getattr(self.config, 'qsa_dkv_reduction', 'atomic')
                    ),
                    # Owner exchange packs arbitrary remote tokens, so its local
                    # cache no longer preserves the ratio-sized contiguous block
                    # layout required by segmented dK/dV reduction.
                    selected_token_group_size=(
                        None if qsa_cp_exchange else getattr(self.config, 'indexer_compress_ratio', None)
                    ),
                    route_block_size=qsa_route_block_size,
                )
        elif self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                # Static batching attention kernel.
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )

            else:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, kv_lengths_decode_only, max_seqlen_k = (inference_context.cu_kv_lengths())

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    kv_lengths_decode_only,
                    block_table,
                )
                core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix='core_attention')

        # =================
        # Output. [sq, b, h]
        # =================

        # MCore's dot-product attention returns a flattened [sq, b, h] tensor,
        # while the selected-KV contract deliberately returns
        # [sq, b, heads, head_dim].  Flatten both branches at this boundary so
        # the TP-sharded output projection sees the same local feature width.
        if core_attn_out.ndim == 4:
            core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        gate = gate.reshape(gate.shape[0], gate.shape[1], -1)
        core_attn_out = core_attn_out * torch.sigmoid(gate.reshape_as(core_attn_out))
        nvtx_range_push(suffix='linear_proj')
        output, bias = self.linear_proj(core_attn_out)
        nvtx_range_pop(suffix='linear_proj')

        return output, bias

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        assert self.config.num_query_groups is not None
        if getattr(self, 'world_size', None) is not None and self.config.num_query_groups < self.world_size:
            # Note that weights are interleaved in the following manner:
            # q1 q2 k1 v1 | q3 q4 k2 v2 | q5 q6 k3 v3 | ...
            # When tp_size > num_kv_heads, we split "q1 q2 k1 v1" over multiple
            # ranks, so a rank does not have a clean partitioning of just the q_heads
            # it needs. Instead, we perform the following steps:
            # 1. Assemble the full "q1 q2 k1 v1 | q3 q4 k2 v2 | q5 q6 k3 v3 | ..."
            #    through an AG.
            # 2. Pull out the right slice (e.g., "q1 q2 k1 v1" or "q3 q4 k2 v2").
            # 3. Split q_heads (e.g., q1, q2), k_heads (e.g., k1), v_heads (e.g., v1).
            # 4. Further index into query to get only the q_heads that this rank is
            #    responsible for (e.g., q1).
            # The block of code below performs steps 1 and 2.
            mixed_qkv = all_gather_last_dim_from_tensor_parallel_region(mixed_qkv)
            idx = get_tensor_model_parallel_rank() // (self.world_size // self.config.num_query_groups)
            size = mixed_qkv.size()[-1] // self.config.num_query_groups
            mixed_qkv = mixed_qkv[:, :, idx * size:(idx + 1) * size]

        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            ((self.num_attention_heads_per_partition // self.num_query_groups_per_partition * 2 + 2)
             * self.hidden_size_per_attention_head),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        split_arg_list = [
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition
             * self.hidden_size_per_attention_head * 2),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        if getattr(self, 'world_size', None) is not None and self.config.num_query_groups < self.world_size:
            # query above corresponds to (num_q_heads / num_kv_heads) q_heads.
            # Index appropriately into query to get (num_q_heads / tp_size) q_heads.
            # This is step 4 in the list of steps above.
            idx = get_tensor_model_parallel_rank() % (self.world_size // self.config.num_query_groups)
            size = query.shape[2] // (self.world_size // self.config.num_query_groups)
            query = query[:, :, idx * size:(idx + 1) * size, :]
        query, gate = query[:, :, ::2], query[:, :, 1::2]
        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value, gate


class Qwen3NextGatedDeltaNet(_HuggingFaceModule, _Qwen3NextGatedDeltaNet):

    def __init__(self, config: ModelConfig, submodules: SelfAttentionSubmodules, layer_number: int, **kwargs):
        assert config.context_parallel_size == 1, 'Qwen3Next currently does not support context parallel.'
        assert _Qwen3NextGatedDeltaNet is not object, 'please update the `transformers` version.'
        _Qwen3NextGatedDeltaNet.__init__(self, config, layer_number)
        self.config = config
        extra_kwargs = _get_extra_te_kwargs(config)
        self.to(dtype=extra_kwargs['params_dtype'], device=extra_kwargs['device'])

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        config = self.config
        if config.sequence_parallel and config.tensor_model_parallel_size > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, tensor_parallel_output_grad=False)
        seq_len = hidden_states.shape[0]
        packed_seq_params = kwargs.get('packed_seq_params')
        thd_format = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        # Note: for packed inputs, we do not perform padding_free unpadding.
        # Doing so would allow different sequences to see each other; for efficiency we keep this implementation.
        if thd_format:
            max_seqlen_q = int(packed_seq_params.max_seqlen_q)
            new_hidden_states = hidden_states.new_zeros(
                (packed_seq_params.num_samples, max_seqlen_q, hidden_states.shape[-1]))
            attention_mask = hidden_states.new_zeros((packed_seq_params.num_samples, max_seqlen_q), dtype=torch.bool)
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            for i in range(packed_seq_params.num_samples):
                start, end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
                attention_mask[i, :end - start] = True
                new_hidden_states[i, :end - start] = hidden_states[start:end, 0]
            hidden_states = new_hidden_states
        else:
            hidden_states = hidden_states.transpose(0, 1)
            attention_mask = resolve_gdn_attention_mask(kwargs)
        with get_cuda_rng_tracker().fork('data-parallel-rng'):
            res = super().forward(hidden_states=hidden_states, attention_mask=attention_mask)
        if thd_format:
            res = res[attention_mask][:, None]
            res = torch.concat([res, res.new_zeros(seq_len - res.shape[0], 1, res.shape[2])])
        else:
            res = res.transpose(0, 1).contiguous()
        if config.sequence_parallel and config.tensor_model_parallel_size > 1:
            res = scatter_to_sequence_parallel_region(res)
        return res, None


class Qwen3NextBridge(GPTBridge):
    hf_mtp_prefix = 'mtp.layers'

    # NOTE: No offset needed for layernorm weights because we use Qwen3NextRMSNorm
    # which implements Zero-Centered RMSNorm (1 + weight) matching HuggingFace exactly.

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        is_linear_attention = self.config.linear_attention_freq[layer_idx]
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        if is_linear_attention:
            hf_state_dict.update(self._set_module(mg_attn, hf_state_dict, 'linear_attn.', to_mcore))
        else:
            hf_state_dict.update(self._set_attn_state(mg_attn, hf_state_dict, 'self_attn.', layer_idx, to_mcore))
        self._set_state_dict(mg_layer, 'input_layernorm.weight', hf_state_dict, self.hf_input_layernorm_key, to_mcore)
        return hf_state_dict

    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool, is_mtp: bool = False):
        if self.model_type != 'qwen3_5':
            return super()._set_layer_mlp(mg_layer, hf_state_dict, layer_idx, to_mcore, is_mtp=is_mtp)
        # dense
        mg_mlp = None if mg_layer is None else mg_layer.mlp
        hf_state_dict.update(
            self._set_mlp_state(mg_mlp, hf_state_dict, f'{self.hf_mlp_prefix}.', layer_idx, to_mcore, is_mtp=is_mtp))
        self._set_state_dict(mg_layer, 'pre_mlp_layernorm.weight', hf_state_dict, self.hf_post_attention_layernorm_key,
                             to_mcore)
        return hf_state_dict

    def _convert_mtp_extra(self, mtp_layer, hf_state_dict, to_mcore, origin_hf_state_dict):
        hf_state_dict = self._remove_prefix(origin_hf_state_dict, 'mtp.')
        for mg_key, key in zip(['enorm.weight', 'hnorm.weight', 'eh_proj.weight'],
                               ['pre_fc_norm_embedding.weight', 'pre_fc_norm_hidden.weight', 'fc.weight']):
            self._set_state_dict(mtp_layer, mg_key, hf_state_dict, key, to_mcore)
        self._fp8_skip_modules.update({'mtp.fc'})
        self._set_state_dict(mtp_layer, 'final_layernorm.weight', hf_state_dict, 'norm.weight', to_mcore)
        if not to_mcore:
            origin_hf_state_dict.update(self._add_prefix(hf_state_dict, 'mtp.'))


class Qwen3NextLoader(ModelLoader):
    gated_delta_net = Qwen3NextGatedDeltaNet

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        config = self.config
        config.hetereogenous_dist_checkpoint = True
        # compat Qwen3NextGatedDeltaNet
        config.hidden_act = 'silu'
        config.rms_norm_eps = config.layernorm_epsilon
        config.dtype = config.params_dtype

        # Use Zero-Centered RMSNorm to match HuggingFace exactly (no +1/-1 conversion needed)
        layer_norm_impl = Qwen3NextRMSNorm
        moe_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            use_kitchen=config.use_kitchen,
        )
        layer_specs = []
        for is_linear_attention in self.config.linear_attention_freq:
            layer_spec = deepcopy(moe_layer_spec)
            if is_linear_attention:
                layer_spec.submodules.self_attention.module = self.gated_delta_net
            else:
                layer_spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
                layer_spec.submodules.self_attention.module = Qwen3NextSelfAttention
            # Replace ALL layernorms with Qwen3NextRMSNorm (Zero-Centered)
            layer_spec.submodules.input_layernorm = layer_norm_impl
            if hasattr(layer_spec.submodules, 'pre_mlp_layernorm'):
                layer_spec.submodules.pre_mlp_layernorm = layer_norm_impl
            # qwen3.5 dense
            if config.hf_model_type == 'qwen3_5':
                layer_spec.submodules.mlp.submodules.linear_fc1 = TEColumnParallelLinear
            # Replace qk_layernorm if present
            if hasattr(layer_spec.submodules.self_attention.submodules, 'q_layernorm'):
                layer_spec.submodules.self_attention.submodules.q_layernorm = layer_norm_impl
            if hasattr(layer_spec.submodules.self_attention.submodules, 'k_layernorm'):
                layer_spec.submodules.self_attention.submodules.k_layernorm = layer_norm_impl
            layer_specs.append(layer_spec)

        local_layer_specs = get_local_layer_specs(config, layer_specs, vp_stage=vp_stage)
        block_spec = TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=layer_norm_impl)

        return block_spec

    def get_mtp_block_spec(self, *args, **kwargs):
        mtp_block_spec = super().get_mtp_block_spec(*args, **kwargs)
        if mtp_block_spec is not None:
            for layer_spec in mtp_block_spec.layer_specs:
                layer_spec.submodules.enorm = Qwen3NextRMSNorm
                layer_spec.submodules.hnorm = Qwen3NextRMSNorm
                layer_spec.submodules.layer_norm = Qwen3NextRMSNorm
        return mtp_block_spec


use_mcore_gdn = get_env_args('USE_MCORE_GDN', bool, True)

if not use_mcore_gdn:
    register_model(
        ModelMeta(
            ModelType.qwen3_next,
            ['qwen3_next'],
            bridge_cls=Qwen3NextBridge,
            loader=Qwen3NextLoader,
        ))
