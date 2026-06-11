# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import math
import megatron.core
import os
import torch
import torch.nn.functional as F
from collections import OrderedDict
from megatron.core import mpu, parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings import rope_utils
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt import GPTModel as McoreGPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import (gather_from_sequence_parallel_region,
                                                    gather_from_tensor_model_parallel_region,
                                                    scatter_to_sequence_parallel_region)
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import WrappedTensor, deprecate_inference_params
from packaging import version
from typing import Optional, Tuple

from mcore_bridge.config import ModelConfig
from mcore_bridge.utils import get_logger, roll_tensor, split_cp_inputs

from .fused_linear_ce import VocabParallelFusedLinearCrossEntropy
from .rope import dynamic_rope_update, get_rope_inv_freq

logger = get_logger()

mcore_016 = version.parse(megatron.core.__version__) >= version.parse('0.16.0rc0')


def _parse_linear_ce_chunk_size() -> int:
    raw_value = os.environ.get('LINEAR_CE_CHUNK_SIZE', '').strip().lower()
    if raw_value in {'', '0', 'false', 'none', 'off'}:
        return 0
    multiplier = 1
    if raw_value.endswith('k'):
        multiplier = 1024
        raw_value = raw_value[:-1]
    elif raw_value.endswith('m'):
        multiplier = 1024 * 1024
        raw_value = raw_value[:-1]
    try:
        chunk_size = int(float(raw_value) * multiplier)
    except ValueError as exc:
        raise ValueError(
            f'LINEAR_CE_CHUNK_SIZE must be an integer token count, e.g. 2048 or 2k. Got: '
            f'{os.environ.get("LINEAR_CE_CHUNK_SIZE")!r}'
        ) from exc
    if chunk_size < 0:
        raise ValueError(f'LINEAR_CE_CHUNK_SIZE must be >= 0. Got: {chunk_size}')
    return chunk_size


def _parse_linear_ce_impl() -> str:
    raw_value = os.environ.get('LINEAR_CE_IMPL', 'torch').strip().lower()
    aliases = {
        '': 'torch',
        'chunked': 'torch',
        'python': 'torch',
        'torch': 'torch',
        'triton': 'triton',
        'fused': 'triton',
        'fused_linear': 'triton',
    }
    if raw_value not in aliases:
        raise ValueError(f'LINEAR_CE_IMPL must be one of torch,triton. Got: {raw_value!r}')
    return aliases[raw_value]


def _tp_group_size(tp_group) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    if tp_group is None:
        return 1
    return torch.distributed.get_world_size(tp_group)


def _tp_all_reduce(tensor: torch.Tensor, op: torch.distributed.ReduceOp, tp_group) -> torch.Tensor:
    if _tp_group_size(tp_group) > 1:
        torch.distributed.all_reduce(tensor, op=op, group=tp_group)
    return tensor


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    """Compute LM loss only for supervised tokens without materializing full-sequence logits."""

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
        partition_vocab_size = output_weight.shape[0]
        vocab_end_index = vocab_start_index + partition_vocab_size
        losses_flat = torch.zeros((seq_len * batch_size,), dtype=torch.float32, device=hidden_states.device)

        for chunk_start in range(0, supervised_indices.numel(), chunk_size):
            chunk_end = min(supervised_indices.numel(), chunk_start + chunk_size)
            token_indices = supervised_indices[chunk_start:chunk_end]
            target = target_flat.index_select(0, token_indices)
            logits = torch.matmul(hidden_flat.index_select(0, token_indices), output_weight.t()).float()

            local_max = logits.max(dim=-1).values
            global_max = _tp_all_reduce(local_max, torch.distributed.ReduceOp.MAX, tp_group)
            exp_logits = torch.exp(logits - global_max.unsqueeze(-1))
            global_sum = _tp_all_reduce(exp_logits.sum(dim=-1), torch.distributed.ReduceOp.SUM, tp_group)

            target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
            local_target = (target - vocab_start_index).masked_fill(target_mask, 0)
            target_logits = torch.gather(logits, dim=-1, index=local_target.unsqueeze(-1)).squeeze(-1)
            target_logits = target_logits.masked_fill(target_mask, 0.0)
            target_logits = _tp_all_reduce(target_logits, torch.distributed.ReduceOp.SUM, tp_group)

            chunk_loss = torch.log(global_sum) + global_max - target_logits
            losses_flat.index_copy_(0, token_indices, chunk_loss)

        ctx.save_for_backward(hidden_states, output_weight, target_flat, supervised_indices)
        ctx.tp_group = tp_group
        ctx.vocab_start_index = vocab_start_index
        ctx.chunk_size = chunk_size
        ctx.reduce_grad_input = reduce_grad_input
        return losses_flat.view(seq_len, batch_size).transpose(0, 1).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, output_weight, target_flat, supervised_indices = ctx.saved_tensors
        tp_group = ctx.tp_group
        vocab_start_index = ctx.vocab_start_index
        chunk_size = ctx.chunk_size
        reduce_grad_input = ctx.reduce_grad_input
        partition_vocab_size = output_weight.shape[0]
        vocab_end_index = vocab_start_index + partition_vocab_size
        seq_len, batch_size, hidden_size = hidden_states.shape

        hidden_flat = hidden_states.contiguous().view(seq_len * batch_size, hidden_size)
        grad_output_flat = grad_output.transpose(0, 1).contiguous().view(-1).float()
        grad_hidden_flat = torch.zeros_like(hidden_flat) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(output_weight) if ctx.needs_input_grad[1] else None

        for chunk_start in range(0, supervised_indices.numel(), chunk_size):
            chunk_end = min(supervised_indices.numel(), chunk_start + chunk_size)
            token_indices = supervised_indices[chunk_start:chunk_end]
            hidden_chunk = hidden_flat.index_select(0, token_indices)
            target = target_flat.index_select(0, token_indices)
            logits = torch.matmul(hidden_chunk, output_weight.t()).float()

            local_max = logits.max(dim=-1).values
            global_max = _tp_all_reduce(local_max, torch.distributed.ReduceOp.MAX, tp_group)
            exp_logits = torch.exp(logits - global_max.unsqueeze(-1))
            global_sum = _tp_all_reduce(exp_logits.sum(dim=-1), torch.distributed.ReduceOp.SUM, tp_group)
            grad_logits = exp_logits / global_sum.unsqueeze(-1)

            target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
            local_target = (target - vocab_start_index).masked_fill(target_mask, 0)
            subtract = (~target_mask).to(dtype=grad_logits.dtype).unsqueeze(-1)
            grad_logits.scatter_add_(dim=-1, index=local_target.unsqueeze(-1), src=-subtract)
            grad_logits.mul_(grad_output_flat.index_select(0, token_indices).unsqueeze(-1))

            if grad_hidden_flat is not None:
                grad_hidden_chunk = torch.matmul(grad_logits, output_weight.float())
                if reduce_grad_input:
                    _tp_all_reduce(grad_hidden_chunk, torch.distributed.ReduceOp.SUM, tp_group)
                grad_hidden_flat.index_copy_(0, token_indices, grad_hidden_chunk.to(dtype=hidden_states.dtype))

            if grad_weight is not None:
                grad_weight_chunk = torch.matmul(grad_logits.t(), hidden_chunk.float())
                grad_weight.add_(grad_weight_chunk.to(dtype=grad_weight.dtype))

        grad_hidden = grad_hidden_flat.view(seq_len, batch_size, hidden_size) if grad_hidden_flat is not None else None
        return grad_hidden, grad_weight, None, None, None, None, None


def _chunked_linear_cross_entropy_loss(model, hidden_states, output_weight, labels, chunk_size):
    if output_weight is None:
        output_weight = model.output_layer.weight
    if output_weight is None:
        raise ValueError('Unable to locate output layer weight for LINEAR_CE_CHUNK_SIZE.')

    if getattr(model.output_layer, 'sequence_parallel', False):
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=True, group=model.pg_collection.tp)
        reduce_grad_input = False
    else:
        reduce_grad_input = _tp_group_size(model.pg_collection.tp) > 1

    vocab_start_index = torch.distributed.get_rank(model.pg_collection.tp) * output_weight.shape[0] \
        if _tp_group_size(model.pg_collection.tp) > 1 else 0
    linear_ce_impl = _parse_linear_ce_impl()
    if linear_ce_impl == 'triton':
        return VocabParallelFusedLinearCrossEntropy.apply(
            hidden_states, output_weight, labels, model.pg_collection.tp, vocab_start_index, chunk_size,
            reduce_grad_input)
    return _ChunkedLinearCrossEntropy.apply(
        hidden_states, output_weight, labels, model.pg_collection.tp, vocab_start_index, chunk_size, reduce_grad_input)


class OutputLayerLinear(TELinear):

    def forward(self, hidden_states, *args, **kwargs):
        return super().forward(hidden_states)

    def sharded_state_dict(
            self,
            prefix: str = '',
            sharded_offsets: Tuple[Tuple[int, int, int]] = (),
            metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        res = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        for k, v in res.items():
            if k.endswith('._extra_state'):
                if v.data is not None and v.data.numel() == 0:
                    v.data = None
        return res


class GPTModel(McoreGPTModel):
    config: ModelConfig

    def __init__(
        self,
        config: ModelConfig,
        transformer_layer_spec: ModuleSpec,
        pre_process: bool = True,
        post_process: bool = True,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
    ):
        vocab_size = math.ceil(
            config.padded_vocab_size / config.tensor_model_parallel_size) * config.tensor_model_parallel_size
        hf_rope_scaling = config.rope_scaling
        if config.multi_latent_attention:
            config.rope_type = 'rope'  # use transformers implementation
            # Set default value, the following content will not be used. (dummy)
            config.mscale_all_dim = 0.
            config.cache_mla_latents = False
            config.rotary_scaling_factor = 40
            if hf_rope_scaling and hf_rope_scaling['rope_type'] == 'yarn':
                # softmax_scale
                config.mscale = hf_rope_scaling['mscale']
                config.mscale_all_dim = hf_rope_scaling['mscale_all_dim']
                config.rotary_scaling_factor = hf_rope_scaling['factor']
        self.hf_rope_scaling = hf_rope_scaling
        super().__init__(
            config,
            transformer_layer_spec,
            vocab_size,
            config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=not config.untie_embeddings_and_output_weights,
            position_embedding_type=config.position_embedding_type,
            rotary_base=config.rotary_base,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )
        if config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=config.qk_pos_emb_head_dim,
                rotary_percent=1,
                rotary_interleaved=config.rotary_interleaved,
                rotary_base=config.rotary_base,
                use_cpu_initialization=config.use_cpu_initialization,
            )
            # save memory
            for i in range(len(self.decoder.layers)):
                if hasattr(self.decoder.layers[i].self_attention, 'rotary_pos_emb'):
                    del self.decoder.layers[i].self_attention.rotary_pos_emb
        self.attention_scaling = 1.
        new_inv_freq, self.attention_scaling = get_rope_inv_freq(config)
        self.rotary_pos_emb.inv_freq = new_inv_freq.to(self.rotary_pos_emb.inv_freq.device)
        if self.config.task_type == 'seq_cls' and self.post_process:
            self.output_layer = OutputLayerLinear(
                config.hidden_size,
                self.config.num_labels,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                parallel_mode='duplicated',
                skip_weight_param_allocation=False,
            )
        elif self.config.task_type == 'embedding' and self.post_process:
            self.output_layer = None

        if self.attention_scaling != 1 and config.apply_rope_fusion:
            config.apply_rope_fusion = False
            logger.warning(f'`apply_rope_fusion` does not support `attention_scaling`. '
                           f'Setting `config.apply_rope_fusion`: {config.apply_rope_fusion}')
        if not config.apply_rope_fusion:
            self._patch_apply_rotary_pos_emb()
        if getattr(self, 'mtp', None) is not None:
            for layer in self.mtp.layers:
                # compat megatron-core main branch
                if not hasattr(layer, 'transformer_layer'):

                    def _value(self):
                        return getattr(self, 'mtp_model_layer')

                    setattr(layer.__class__, 'transformer_layer', property(_value))
                attention = layer.transformer_layer.self_attention
                attention.config = copy.copy(attention.config)
                attention.config.apply_rope_fusion = False

    def _patch_apply_rotary_pos_emb(self):
        if hasattr(rope_utils, '_origin_apply_rotary_pos_emb_bshd'):
            return
        _origin_apply_rotary_pos_emb_bshd = rope_utils._apply_rotary_pos_emb_bshd

        def _apply_rotary_pos_emb_bshd(
            t: torch.Tensor,
            freqs: torch.Tensor,
            rotary_interleaved: bool = False,
            multi_latent_attention: bool = False,  # not use
            mscale: float = 1.0,
            **kwargs,
        ) -> torch.Tensor:
            """Apply rotary positional embedding to input tensor T.

            check https://kexue.fm/archives/8265 for detailed formulas

            Args:
                t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
                freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

            Returns:
                Tensor: The input tensor after applying RoPE
            """
            mscale = self.attention_scaling
            rot_dim = freqs.shape[-1]

            # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
            t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
            if multi_latent_attention:
                x1 = t[..., 0::2]
                x2 = t[..., 1::2]
                t = torch.cat((x1, x2), dim=-1)

            # first part is cosine component
            # second part is sine component, need to change signs with _rotate_half method
            cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
            sin_ = (torch.sin(freqs) * mscale).to(t.dtype)

            t = (t * cos_) + (rope_utils._rotate_half(t, rotary_interleaved) * sin_)
            return torch.cat((t, t_pass), dim=-1)

        rope_utils._apply_rotary_pos_emb_bshd = _apply_rotary_pos_emb_bshd
        rope_utils._origin_apply_rotary_pos_emb_bshd = _origin_apply_rotary_pos_emb_bshd

    def _preprocess(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        decoder_input: torch.Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Preprocesses inputs for the transformer decoder.

        Applies embeddings to input tokens, or uses `decoder_input` from a previous
        pipeline stage. Also sets up rotary positional embeddings.
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
        in_inference_mode = inference_context is not None and not self.training

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        if decoder_input is not None and self.training and torch.is_grad_enabled() and not decoder_input.requires_grad:
            # fix LoRA incompatibility with gradient checkpointing
            decoder_input = decoder_input.requires_grad_(True)

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if self.position_embedding_type in {'rope', 'mrope'}:
            if not self.training and self.config.flash_decode and inference_context:
                assert (inference_context.is_static_batching()
                        ), 'GPTModel currently only supports static inference batching.'
                # Flash decoding uses precomputed cos and sin for RoPE
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                    inference_context.max_sequence_length,
                    self.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
                )
            else:
                rotary_seq_len = RotaryEmbedding.get_rotary_seq_len(self, inference_context, self.decoder,
                                                                    decoder_input, self.config, packed_seq_params)
                if self.hf_rope_scaling is not None:
                    attention_scaling = dynamic_rope_update(self, self.rotary_pos_emb.inv_freq, rotary_seq_len)
                    if attention_scaling is not None and attention_scaling != self.attention_scaling:
                        raise ValueError('Currently does not support changing attention_scaling during training. '
                                         f'self.attention_scaling: {self.attention_scaling}, '
                                         f'current_attention_scaling: {attention_scaling}.')
                if self.position_embedding_type == 'mrope':
                    rotary_pos_emb = self.rotary_pos_emb(
                        position_ids,
                        mrope_section=self.mrope_section,
                        mrope_interleaved=self.config.mrope_interleaved,
                    )
                else:
                    packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
                    rotary_pos_emb = self.rotary_pos_emb(
                        rotary_seq_len,
                        packed_seq=packed_seq,
                    )

        if (in_inference_mode and ((self.config.enable_cuda_graph and self.config.cuda_graph_scope != 'full_iteration')
                                   or self.config.flash_decode) and rotary_pos_cos is not None
                and inference_context.is_static_batching()):
            current_batch_size = input_ids.shape[0]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
            )
        else:
            sequence_len_offset = None

        # Wrap decoder_input to allow the decoder (TransformerBlock) to delete the
        # reference held by this caller function, enabling early garbage collection for
        # inference. Skip wrapping if decoder_input is logged after decoder completion.
        if in_inference_mode and not has_config_logger_enabled(self.config):
            decoder_input = WrappedTensor(decoder_input)

        return decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset

    # Code borrowed from NVIDIA/Megatron-LM
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
        if self.config.position_embedding_type == 'mrope' and position_ids.ndim == 2:  # qwen3_asr
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        inference_context = deprecate_inference_params(inference_context, inference_params)
        # There is a difference in whether rotary_pos_emb can be fused between the decoder and MTP.
        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = (
            self._preprocess(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=decoder_input,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            ))
        decoder_rotary_pos_emb = rotary_pos_emb
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.position_embedding_type == 'rope' and packed_seq and not self.config.apply_rope_fusion:
            assert position_ids.shape[0] == 1, f'position_ids.shape: {position_ids.shape}'
            decoder_rotary_pos_emb = rotary_pos_emb[position_ids[0]]

        mtp_decoder_input = decoder_input
        if self.config.is_multimodal and self.config.mtp_num_layers and decoder_input is None:
            input_tensor = self.get_input_tensor()
            input_tensor, mtp_decoder_input = input_tensor.chunk(2, dim=0)
            self.set_input_tensor(input_tensor)
        kwargs = {}
        if mcore_016 and attention_mask is not None:
            assert packed_seq_params is None
            padding_mask = ~((~attention_mask).sum(dim=(1, 2)) > 0)
            if self.config.context_parallel_size > 1:
                padding_mask = split_cp_inputs(padding_mask, None, 1)
            tp_size = self.config.tensor_model_parallel_size
            if self.config.sequence_parallel and tp_size > 1:
                assert padding_mask.shape[1] % tp_size == 0, f'padding_mask.shape: {padding_mask.shape}'
                padding_mask = torch.chunk(padding_mask, tp_size, dim=1)[mpu.get_tensor_model_parallel_rank()]
            kwargs['padding_mask'] = padding_mask.contiguous()
        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=decoder_rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
            **kwargs,
        )

        # MTP: https://github.com/NVIDIA/Megatron-LM/issues/1661
        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            loss_mask=loss_mask,
            decoder_input=mtp_decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )

    def _postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
    ):
        """Postprocesses decoder hidden states to generate logits or compute loss.

        Applies Multi-Token Prediction if enabled, generates output logits through
        the output layer, and computes language model loss when labels are provided.
        """
        if not self.post_process:
            if self.config.is_multimodal and self.config.mtp_num_layers:
                return torch.concat([hidden_states, decoder_input], dim=0)
            else:
                return hidden_states
        labels = labels if self.config.task_type == 'causal_lm' else None
        in_inference_mode = inference_context is not None and not self.training
        if in_inference_mode:
            assert runtime_gather_output, 'Inference must always gather TP logits'

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        if self.config.is_multimodal and self.config.context_parallel_size > 1 and input_ids is not None:
            # input_ids is required by MTP.
            input_ids = split_cp_inputs(input_ids, getattr(packed_seq_params, 'cu_seqlens_q', None), 1)

        if self.mtp_process and labels is not None:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=self.embedding,
                decoder_input=decoder_input if self.config.is_multimodal else None,
                **(extra_block_kwargs or {}),
            )
            mtp_labels = labels.clone()
            hidden_states_list = torch.chunk(hidden_states, 1 + self.config.mtp_unroll_steps, dim=0)
            hidden_states = hidden_states_list[0]
            if loss_mask is None:
                # if loss_mask is not provided, use all ones as loss_mask
                loss_mask = torch.ones_like(mtp_labels)
            for mtp_layer_number in range(self.config.mtp_unroll_steps):
                # output
                mtp_logits, _ = self.output_layer(
                    hidden_states_list[mtp_layer_number + 1],
                    weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                )
                # Calc loss for the current Multi-Token Prediction (MTP) layers.
                mtp_labels, _ = roll_tensor(
                    mtp_labels,
                    shifts=-1,
                    dims=-1,
                    cp_group=self.cp_group,
                    packed_seq_params=packed_seq_params,
                )
                loss_mask, _ = roll_tensor(
                    loss_mask,
                    shifts=-1,
                    dims=-1,
                    cp_group=self.cp_group,
                    packed_seq_params=packed_seq_params,
                )
                mtp_loss = self.compute_language_model_loss(mtp_labels, mtp_logits)
                loss_mask_ = (loss_mask & (mtp_labels != -100))
                num_tokens = loss_mask_.sum()
                mtp_loss = loss_mask_ * mtp_loss
                if self.training:
                    mtp_loss_for_log = (
                        torch.sum(mtp_loss) / num_tokens if num_tokens > 0 else mtp_loss.new_tensor(0.0))
                    MTPLossLoggingHelper.save_loss_to_tracker(
                        mtp_loss_for_log,
                        mtp_layer_number,
                        self.config.mtp_unroll_steps,
                        avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                    )
                mtp_loss_scale = self.config.mtp_loss_scaling_factor / self.config.mtp_unroll_steps
                if self.config.calculate_per_token_loss:
                    hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss)
                else:
                    hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss / num_tokens)
        sequence_parallel_override = False
        if in_inference_mode and inference_context.materialize_only_last_token_logits:
            if inference_context.is_static_batching():
                hidden_states = hidden_states[-1:, :, :]
            else:
                if self.output_layer.sequence_parallel:
                    # Perform the sequence parallel gather here instead of after the output layer
                    # because we need to slice the last token logits from the full view of the
                    # packed logits across all requests.
                    # TODO(ksanthanam): Make the equivalent change in the `MambaModel` code after
                    # merging in !3722.
                    hidden_states = gather_from_sequence_parallel_region(hidden_states, group=self.pg_collection.tp)
                    self.output_layer.sequence_parallel = False
                    sequence_parallel_override = True

                # Reshape [B, 1, H] to [1, B, H] → extract each sample’s true last‐token hidden
                # state ([B, H]) → unsqueeze back to [1, B, H]
                # (so that the output layer, which expects S×B×H, receives only the final token)
                hidden_states = inference_context.last_token_logits(hidden_states.squeeze(1).unsqueeze(0)).unsqueeze(1)

        linear_ce_chunk_size = _parse_linear_ce_chunk_size()
        if (linear_ce_chunk_size > 0 and labels is not None and self.config.task_type == 'causal_lm'
                and not in_inference_mode):
            if runtime_gather_output:
                raise ValueError('LINEAR_CE_CHUNK_SIZE requires vocab-parallel output; runtime_gather_output must be false.')
            if getattr(self.config, 'use_mup', False):
                raise ValueError('LINEAR_CE_CHUNK_SIZE currently does not support MuP output scaling.')
            if (os.environ.get('LINEAR_CE_DEBUG', '').lower() in {'1', 'true', 'yes', 'on'}
                    and not getattr(self, '_linear_ce_chunk_size_logged', False)):
                logger.info(
                    f'Using supervised-token chunked linear CE loss; '
                    f'impl={_parse_linear_ce_impl()}, chunk_size={linear_ce_chunk_size}.')
                self._linear_ce_chunk_size_logged = True
            return _chunked_linear_cross_entropy_loss(self, hidden_states, output_weight, labels, linear_ce_chunk_size)

        if self.config.task_type == 'embedding':
            logits = F.normalize(hidden_states, p=2, dim=-1)
        else:
            logits, _ = self.output_layer(
                hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)
            if self.config.task_type == 'generative_reranker':
                logits = gather_from_tensor_model_parallel_region(logits)
                positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
                negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')
                positive_token_id = self.tokenizer.convert_tokens_to_ids(positive_token)
                negative_token_id = self.tokenizer.convert_tokens_to_ids(negative_token)
                logits = (logits[..., positive_token_id] - logits[..., negative_token_id])[..., None]
        if self.config.task_type in {'seq_cls', 'embedding'
                                     } and self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1:
            logits = gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)

        # Restore sequence parallel execution to the output layer if necessary.
        if sequence_parallel_override:
            assert (in_inference_mode and inference_context.is_dynamic_batching()
                    and inference_context.materialize_only_last_token_logits)
            self.output_layer.sequence_parallel = True

        if has_config_logger_enabled(self.config):
            payload = OrderedDict({
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'decoder_input': decoder_input,
                'logits': logits,
            })
            log_config_to_disk(self.config, payload, prefix='input_and_logits')

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        return loss

    def get_input_tensor(self):
        return self.decoder.input_tensor
