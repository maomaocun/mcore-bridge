# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import megatron.core
import os
import peft
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from contextlib import contextmanager, nullcontext
from importlib import metadata
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import (TEColumnParallelGroupedLinear, TEColumnParallelLinear,
                                                         TEGroupedLinear, TELayerNormColumnParallelLinear, TELinear,
                                                         TERowParallelGroupedLinear, TERowParallelLinear)
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.parallel_state import get_expert_tensor_parallel_world_size, get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.router import TopKRouter
from packaging import version
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils.other import transpose
from transformers.utils import is_torch_npu_available
from typing import Any, List, Optional, Tuple

from mcore_bridge.utils import get_current_device

from .npu_lora import NpuGroupedLoraLinear, is_expert_layer
from .utils import tuners_sharded_state_dict

mcore_016 = version.parse(megatron.core.__version__) >= version.parse('0.16.0rc0')
peft_019 = version.parse(peft.__version__) >= version.parse('0.19.0')
MINDSPEED_015 = version.parse('0.15.0')


def _get_mindspeed_version():
    try:
        return version.parse(metadata.version('mindspeed'))
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _use_legacy_npu_local_linear() -> bool:
    if not is_torch_npu_available():
        return False
    mindspeed_version = _get_mindspeed_version()
    if mindspeed_version is None:
        # Fall back to the conservative path when the version is unknown so we
        # do not force an older NPU stack onto the 0.15 TE semantics.
        return True
    return mindspeed_version < MINDSPEED_015


def _build_local_te_linear(input_size: int, output_size: int, bias: bool, **kwargs):
    if _use_legacy_npu_local_linear():
        return nn.Linear(
            in_features=input_size,
            out_features=output_size,
            bias=bias,
        )
    local_kwargs = dict(kwargs)
    local_kwargs.pop('tp_group', None)
    return TELinear(
        input_size=input_size,
        output_size=output_size,
        bias=bias,
        parallel_mode='duplicated',
        skip_weight_param_allocation=False,
        **local_kwargs,
    )


def _get_tensor_parallel_group_for_lora(base_layer):
    """Resolve the tensor-parallel group across TE and MindSpeed TE variants.

    Megatron's TE layers expose ``tp_group`` directly, but MindSpeed 0.15.x
    replaces some TE classes (for example
    ``MindSpeedTELayerNormColumnParallelLinear``) with implementations that keep
    the same tensor-parallel semantics under ``parallel_group`` instead. LoRA
    still needs to forward the right group into the newly created parallel
    adapter layers, otherwise adapter injection fails before training starts.
    """
    tp_group = getattr(base_layer, 'tp_group', None)
    if tp_group is not None:
        return tp_group
    return getattr(base_layer, 'parallel_group', None)


class _InplaceScaledAdd(torch.autograd.Function):
    """Add one LoRA output slice into a non-leaf base output in place.

    TE grouped linears return tensors backed by custom autograd Functions, so
    modifying a ``result.narrow(...)`` view directly is rejected by PyTorch.
    This small autograd bridge marks the base output dirty and supplies the
    corresponding sliced gradient for the LoRA branch without allocating a
    full-size ``result + delta`` tensor.
    """

    @staticmethod
    def forward(ctx, base, delta, start, alpha):
        start = int(start)
        num_tokens = delta.shape[0]
        ctx.start = start
        ctx.num_tokens = num_tokens
        ctx.alpha = float(alpha)
        # TE grouped-linear outputs are custom autograd outputs.  Even a
        # detached narrow view increments their version counter and is
        # rejected by the view/in-place guard.  This adapter owns the backward
        # rule explicitly, so use the raw storage update and avoid creating a
        # tracked view or version-counter edge here.
        base.data.narrow(0, start, num_tokens).add_(delta.data, alpha=ctx.alpha)
        return base

    @staticmethod
    def backward(ctx, grad_output):
        delta_grad = grad_output.narrow(0, ctx.start, ctx.num_tokens).mul(ctx.alpha)
        return grad_output, delta_grad, None, None


class _InplaceScaledAddStrided(torch.autograd.Function):
    """Accumulate a TP sequence-gathered LoRA chunk into global rows."""

    @staticmethod
    def forward(ctx, base, delta, local_start, local_num_tokens, local_total_tokens, alpha):
        local_start = int(local_start)
        local_num_tokens = int(local_num_tokens)
        local_total_tokens = int(local_total_tokens)
        replica_count = base.shape[0] // local_total_tokens
        ctx.local_start = local_start
        ctx.local_num_tokens = local_num_tokens
        ctx.local_total_tokens = local_total_tokens
        ctx.replica_count = replica_count
        ctx.alpha = float(alpha)
        for replica in range(replica_count):
            base_start = replica * local_total_tokens + local_start
            delta_start = replica * local_num_tokens
            base.data.narrow(0, base_start, local_num_tokens).add_(
                delta.data.narrow(0, delta_start, local_num_tokens), alpha=ctx.alpha
            )
        return base

    @staticmethod
    def backward(ctx, grad_output):
        delta_grad = torch.cat(
            [
                grad_output.narrow(
                    0,
                    replica * ctx.local_total_tokens + ctx.local_start,
                    ctx.local_num_tokens,
                )
                for replica in range(ctx.replica_count)
            ],
            dim=0,
        ).mul(ctx.alpha)
        return grad_output, delta_grad, None, None, None, None


class LoraParallelLinear(MegatronModule, LoraLayer):

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ):
        config = base_layer.config
        super().__init__(config=config)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            LoraLayer.__init__(self, base_layer=base_layer)

        if use_dora:
            raise ValueError(f'{self.__class__.__name__} does not support DoRA yet, please set it to False')

        self.is_parallel_a = isinstance(base_layer, (TERowParallelLinear, TERowParallelGroupedLinear))
        self.is_grouped = isinstance(base_layer, TEGroupedLinear)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.is_expert = is_expert_layer(base_layer)
        self.sequence_parallel = getattr(base_layer, 'sequence_parallel', False)
        if self.is_expert:
            self.tp_size = get_expert_tensor_parallel_world_size()
            # TODO: For TEGroupedLinear under ETP, initialization must use different random seeds
            #       across EP ranks and identical random seeds across ETP ranks.
            # Additionally, a parameter-averaging all_reduce across the ETP group is required.
            # Note that TEGroupedLinear here excludes TEColumnParallelGroupedLinear and TERowParallelGroupedLinear.
            if self.tp_size > 1:
                raise ValueError('Currently, LoRA does not support ETP.')
        else:
            self.tp_size = get_tensor_model_parallel_world_size()
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            lora_bias=lora_bias,
        )

        self.is_target_conv_1d_layer = False
        self._dsv4_lora_debug_logged = False
        self._dsv4_lora_chunk_debug_count = 0

    @staticmethod
    def _grouped_lora_token_counts(args, kwargs):
        """Return grouped-linear token counts as a Python list."""
        token_arg = args[0] if args else kwargs.get('tokens_per_expert')
        if token_arg is None:
            return None
        if torch.is_tensor(token_arg):
            token_arg = token_arg.detach().cpu().tolist()
        return [int(count) for count in token_arg]

    @staticmethod
    def _grouped_lora_call_args(args, kwargs, token_counts):
        """Replace ``tokens_per_expert`` while preserving other TE arguments."""
        if args:
            return (token_counts, *args[1:]), kwargs
        grouped_kwargs = dict(kwargs)
        grouped_kwargs['tokens_per_expert'] = token_counts
        return args, grouped_kwargs

    def _forward_grouped_lora_chunked(self, result, x, lora_A, lora_B, dropout, scaling, args, kwargs):
        """Apply grouped LoRA in bounded token chunks.

        MoE dispatch lays out rows contiguously by expert.  Splitting that
        layout and accumulating each LoRA output directly into ``result``
        avoids materializing the full ``lora_B(lora_A(x))`` delta and the
        additional scaled/add result tensors at once.
        """
        try:
            chunk_size = int(os.environ.get('DSV4_LORA_GROUPED_CHUNK_SIZE', '0') or 0)
        except ValueError:
            chunk_size = 0
        debug = os.environ.get('DSV4_LORA_DEBUG_SHAPES', '0') == '1'
        if debug and result.shape[0] >= 16384 and self._dsv4_lora_chunk_debug_count < 40:
            print(
                '[dsv4-lora-debug] grouped_enter base={} x_shape={} result_shape={} '
                'args_len={} lora_a={} lora_b={} chunk_size={}'.format(
                    type(self.base_layer).__name__,
                    tuple(x.shape),
                    tuple(result.shape),
                    len(args),
                    type(lora_A).__name__,
                    type(lora_B).__name__,
                    chunk_size,
                ),
                flush=True,
            )
            self._dsv4_lora_chunk_debug_count += 1
        if chunk_size <= 0:
            return None

        token_counts = self._grouped_lora_token_counts(args, kwargs)
        num_gemms = getattr(lora_A, 'num_gemms', None)
        if token_counts is None or num_gemms is None or len(token_counts) != num_gemms:
            if debug and result.shape[0] >= 16384 and self._dsv4_lora_chunk_debug_count < 40:
                print(
                    '[dsv4-lora-debug] grouped_skip_counts token_counts_len={} '
                    'num_gemms={} token_sum={} x_rows={} result_rows={}'.format(
                        None if token_counts is None else len(token_counts),
                        num_gemms,
                        None if token_counts is None else sum(token_counts),
                        x.shape[0],
                        result.shape[0],
                    ),
                    flush=True,
                )
                self._dsv4_lora_chunk_debug_count += 1
            return None
        if getattr(lora_B, 'num_gemms', num_gemms) != num_gemms:
            return None
        total_tokens = sum(token_counts)
        if total_tokens <= chunk_size or total_tokens != x.shape[0] or total_tokens != result.shape[0]:
            if debug and result.shape[0] >= 16384 and self._dsv4_lora_chunk_debug_count < 40:
                print(
                    '[dsv4-lora-debug] grouped_skip_rows token_sum={} x_rows={} '
                    'result_rows={} chunk_size={}'.format(
                        total_tokens, x.shape[0], result.shape[0], chunk_size
                    ),
                    flush=True,
                )
                self._dsv4_lora_chunk_debug_count += 1
            return None

        # The rank-sized A output is cheap to keep even for a long dispatched
        # sequence.  Compute it once so chunking only changes the large B
        # output; this also preserves the original dropout/RNG behavior and
        # avoids compiling a second grouped-A variant for every chunk.
        if isinstance(lora_A, (TEGroupedLinear, NpuGroupedLoraLinear)):
            lora_a_result = lora_A(dropout(x), *args, **kwargs)
        else:
            lora_a_result = lora_A(dropout(x))
        if isinstance(lora_a_result, tuple):
            lora_a_result = lora_a_result[0]

        def apply_chunk(start, num_tokens, counts):
            nonlocal result
            chunk_args, chunk_kwargs = self._grouped_lora_call_args(args, kwargs, counts)
            chunk_lora_a = lora_a_result.narrow(0, start, num_tokens)
            if isinstance(lora_B, (TEGroupedLinear, NpuGroupedLoraLinear)):
                chunk_result = lora_B(chunk_lora_a, *chunk_args, **chunk_kwargs)
            else:
                chunk_result = lora_B(chunk_lora_a)
            if isinstance(chunk_result, tuple):
                chunk_result = chunk_result[0]
            result = _InplaceScaledAdd.apply(result, chunk_result, start, scaling)

        input_offset = 0
        pending_start = None
        pending_tokens = 0
        pending_counts = [0] * num_gemms
        for expert_idx, expert_tokens in enumerate(token_counts):
            remaining = expert_tokens
            while remaining > 0:
                if pending_start is None:
                    pending_start = input_offset
                available = chunk_size - pending_tokens
                take = min(remaining, available)
                pending_counts[expert_idx] += take
                pending_tokens += take
                input_offset += take
                remaining -= take
                if pending_tokens == chunk_size:
                    apply_chunk(pending_start, pending_tokens, pending_counts)
                    pending_start = None
                    pending_tokens = 0
                    pending_counts = [0] * num_gemms
        if pending_tokens:
            apply_chunk(pending_start, pending_tokens, pending_counts)
        return result

    def _forward_lora_linear_chunked(self, result, x, lora_A, lora_B, dropout, scaling):
        """Apply a regular (non-grouped) LoRA branch in row chunks.

        Long V4 Indexer projections can produce a multi-GiB LoRA B output even
        though the rank-sized A output is small.  Compute A once, then bound B
        by the first (sequence/token) dimension and accumulate each slice with
        the same explicit autograd bridge used by grouped experts.
        """
        try:
            chunk_size = int(os.environ.get('DSV4_LORA_LINEAR_CHUNK_SIZE', '0') or 0)
        except ValueError:
            chunk_size = 0
        debug_chunk = os.environ.get('DSV4_LORA_DEBUG_SHAPES', '0') == '1'
        debug_large = result.numel() > 2**28
        if debug_chunk and debug_large and self._dsv4_lora_chunk_debug_count < 20:
            print(
                '[dsv4-lora-debug] linear_chunk_enter x_shape={} result_shape={} '
                'chunk_size={}'.format(tuple(x.shape), tuple(result.shape), chunk_size),
                flush=True,
            )
            self._dsv4_lora_chunk_debug_count += 1
        if chunk_size <= 0 or result.shape[0] <= chunk_size:
            if debug_chunk and debug_large and self._dsv4_lora_chunk_debug_count < 20:
                print('[dsv4-lora-debug] linear_chunk_skip early_dim0', flush=True)
                self._dsv4_lora_chunk_debug_count += 1
            return None

        lora_a_result = lora_A(dropout(x))
        if isinstance(lora_a_result, tuple):
            lora_a_result = lora_a_result[0]
        local_total_tokens = lora_a_result.shape[0]
        if debug_chunk and debug_large and self._dsv4_lora_chunk_debug_count < 20:
            print(
                '[dsv4-lora-debug] linear_chunk_a_shape={} result_rows={} '.format(
                    tuple(lora_a_result.shape), result.shape[0]
                ),
                flush=True,
            )
            self._dsv4_lora_chunk_debug_count += 1
        if local_total_tokens <= 0 or result.shape[0] % local_total_tokens != 0:
            if debug_chunk and debug_large and self._dsv4_lora_chunk_debug_count < 20:
                print(
                    '[dsv4-lora-debug] linear_chunk_skip a_shape={} result_shape={} '.format(
                        tuple(lora_a_result.shape), tuple(result.shape)
                    ),
                    flush=True,
                )
            return None
        replica_count = result.shape[0] // local_total_tokens
        local_chunk_size = max(1, chunk_size // replica_count)
        if os.environ.get('DSV4_LORA_DEBUG_SHAPES', '0') == '1' and result.numel() > 2**28:
            print(
                '[dsv4-lora-debug] linear_chunk_use a_shape={} result_shape={} replicas={} '
                'local_chunk={} '.format(
                    tuple(lora_a_result.shape),
                    tuple(result.shape),
                    replica_count,
                    local_chunk_size,
                ),
                flush=True,
            )
        for start in range(0, local_total_tokens, local_chunk_size):
            end = min(start + local_chunk_size, local_total_tokens)
            chunk_result = lora_B(lora_a_result.narrow(0, start, end - start))
            if isinstance(chunk_result, tuple):
                chunk_result = chunk_result[0]
            if chunk_result.shape[0] != replica_count * (end - start):
                raise RuntimeError(
                    'DSV4_LORA_LINEAR_CHUNK_SIZE expected sequence-parallel LoRA output '
                    f'with {replica_count * (end - start)} rows, got {chunk_result.shape[0]}'
                )
            result = _InplaceScaledAddStrided.apply(
                result, chunk_result, start, end - start, local_total_tokens, scaling
            )
        return result

    def update_layer(self, adapter_name, r, *, lora_alpha, **kwargs):
        if peft_019 and 'config' in kwargs:
            config = kwargs['config']
            lora_dropout, init_lora_weights, use_rslora, lora_bias = (config.lora_dropout, config.init_lora_weights,
                                                                      config.use_rslora, config.lora_bias)
        else:
            lora_dropout, init_lora_weights, use_rslora, lora_bias = (kwargs['lora_dropout'],
                                                                      kwargs['init_lora_weights'], kwargs['use_rslora'],
                                                                      kwargs['lora_bias'])
        if r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer

        # lora needs to be forced to upgrade to 32-bit precision, otherwise it will overflow
        kwargs = {
            'skip_bias_add': False,
            'init_method': self.config.init_method,
            'config': self.config,
            'is_expert': self.is_expert,
        }
        if not (mcore_016 and self.is_grouped):
            tp_group = _get_tensor_parallel_group_for_lora(self.base_layer)
            if tp_group is not None:
                kwargs['tp_group'] = tp_group
        if isinstance(self.base_layer, TopKRouter):
            router_shape = self.base_layer.weight.shape
            lora_a = _build_local_te_linear(router_shape[1], r, lora_bias, **kwargs)
            lora_b = _build_local_te_linear(r, router_shape[0], lora_bias, **kwargs)
        elif self.is_parallel_a:
            in_features = self.in_features * self.tp_size
            if self.is_grouped:
                if is_torch_npu_available():
                    lora_a = NpuGroupedLoraLinear(
                        self.base_layer.num_gemms,
                        in_features,
                        r,
                        config=self.config,
                        bias=False,
                        is_expert=self.is_expert)
                    lora_b = NpuGroupedLoraLinear(
                        self.base_layer.num_gemms,
                        r,
                        self.out_features,
                        config=self.config,
                        bias=lora_bias,
                        is_expert=self.is_expert)
                else:
                    lora_a = TERowParallelGroupedLinear(
                        num_gemms=self.base_layer.num_gemms,
                        input_size=in_features,
                        output_size=r,
                        bias=False,
                        **kwargs,
                    )
                    lora_b = TEGroupedLinear(
                        num_gemms=self.base_layer.num_gemms,
                        input_size=r,
                        output_size=self.out_features,
                        bias=lora_bias,
                        parallel_mode=None,
                        **kwargs,
                    )
            else:
                lora_a = TERowParallelLinear(
                    input_size=in_features,
                    output_size=r,
                    bias=False,
                    input_is_parallel=True,
                    **kwargs,
                )
                lora_b = _build_local_te_linear(r, self.out_features, lora_bias, **kwargs)
                lora_a.parallel_mode = self.base_layer.parallel_mode  # fix moe_shared_expert_overlap
        else:
            if is_torch_npu_available():
                out_features = self.out_features
            else:
                out_features = self.out_features * self.tp_size
            if self.is_grouped:
                if is_torch_npu_available():
                    lora_a = NpuGroupedLoraLinear(
                        self.base_layer.num_gemms,
                        self.in_features,
                        r,
                        config=self.config,
                        bias=lora_bias,
                        is_expert=self.is_expert)
                    lora_b = NpuGroupedLoraLinear(
                        self.base_layer.num_gemms,
                        r,
                        out_features,
                        config=self.config,
                        bias=lora_bias,
                        is_expert=self.is_expert)
                else:
                    lora_a = TEGroupedLinear(
                        num_gemms=self.base_layer.num_gemms,
                        input_size=self.in_features,
                        output_size=r,
                        bias=lora_bias,
                        parallel_mode=None,
                        **kwargs)
                    lora_b = TEColumnParallelGroupedLinear(
                        num_gemms=self.base_layer.num_gemms,
                        input_size=r,
                        output_size=out_features,
                        bias=lora_bias,
                        **kwargs,
                    )
            else:
                lora_a = _build_local_te_linear(self.in_features, r, lora_bias, **kwargs)
                lora_b = TEColumnParallelLinear(
                    input_size=r,
                    output_size=out_features,
                    bias=lora_bias,
                    gather_output=False,
                    **kwargs,
                )
                lora_b.parallel_mode = self.base_layer.parallel_mode  # fix moe_shared_expert_overlap
        for lora in [lora_a, lora_b]:
            # When parallel_mode is set to None by moe_shared_expert_overlap,
            # disable UB comm overlap; the corresponding collectives are driven
            # externally by the framework.
            if isinstance(lora, (TERowParallelLinear, TEColumnParallelLinear)) and lora.parallel_mode is None:
                lora.ub_overlap_rs_fprop = False
                lora.ub_overlap_ag_dgrad = False
                lora.ub_overlap_ag_fprop = False
                lora.ub_overlap_rs_dgrad = False

        self.lora_A[adapter_name] = lora_a
        self.lora_B[adapter_name] = lora_b
        if hasattr(self, 'lora_bias'):
            self.lora_bias[adapter_name] = lora_bias
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / (r**0.5)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def _get_rng_context(self, lora):
        if not get_cuda_rng_tracker().is_initialized():
            return nullcontext()
        parallel_mode = getattr(lora, 'parallel_mode', None)
        if self.is_expert:
            rng_context = get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name())
        elif parallel_mode in {'duplicated', None}:
            rng_context = nullcontext()
        else:
            rng_context = get_cuda_rng_tracker().fork()
        return rng_context

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            lora_a = self.lora_A[adapter_name]
            lora_b = self.lora_B[adapter_name]
            if isinstance(lora_a, (TEGroupedLinear, NpuGroupedLoraLinear)):
                weights_a = [getattr(lora_a, f'weight{i}') for i in range(lora_a.num_gemms)]
            else:
                weights_a = [lora_a.weight]
            if isinstance(lora_b, (TEGroupedLinear, NpuGroupedLoraLinear)):
                weights_b = [getattr(lora_b, f'weight{i}') for i in range(lora_b.num_gemms)]
            else:
                weights_b = [lora_b.weight]
            with self._get_rng_context(lora_a):
                for weight_a in weights_a:
                    if init_lora_weights is True:
                        # initialize A the same way as the default for nn.Linear and B to zero
                        # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                        nn.init.kaiming_uniform_(weight_a, a=math.sqrt(5))
                    elif init_lora_weights.lower() == 'gaussian':
                        nn.init.normal_(weight_a, std=1 / self.r[adapter_name])
                    else:
                        raise ValueError(f'Unknown initialization {init_lora_weights=}')
            for weight_b in weights_b:
                nn.init.zeros_(weight_b)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    @contextmanager
    def _patch_router_gating(self):
        origin_gating = self.base_layer.__class__.gating

        def gating(_self, x):
            result = origin_gating(_self, x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(result.dtype)

                lora_result = F.linear(dropout(x), lora_A.weight.to(result.dtype))
                if isinstance(lora_result, tuple):
                    lora_result = lora_result[0]
                lora_result = F.linear(lora_result, lora_B.weight.to(result.dtype))
                if isinstance(lora_result, tuple):
                    lora_result = lora_result[0]
                lora_result = lora_result * scaling

                result = result + lora_result
            return result

        self.base_layer.__class__.gating = gating
        try:
            yield
        finally:
            self.base_layer.__class__.gating = origin_gating

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        previous_dtype = x.dtype
        if self.disable_adapters and self.merged:
            self.unmerge()

        if isinstance(self.base_layer, TELayerNormColumnParallelLinear):
            if self.disable_adapters or self.merged:
                self.base_layer.return_layernorm_output = False
                result, bias = self.base_layer(x, *args, **kwargs)
            else:
                self.base_layer.return_layernorm_output = True
                if is_torch_npu_available():
                    # NPU: base_layer only returns (output, bias); it does not expose the LayerNorm/RMSNorm output.
                    inp = x  # Keep the original pre-norm input.
                    result, bias = self.base_layer(inp, *args, **kwargs)

                    # Key: For LoRA we need the same "x" as in the non-NPU branch, i.e. the post-norm activation
                    # (LayerNorm/RMSNorm output, which is the actual input to the fused linear).
                    if hasattr(self.base_layer, 'config') and (hasattr(self.base_layer, '_layernorm')
                                                               or hasattr(self.base_layer, '_rmsnorm')):
                        norm_type = getattr(self.base_layer.config, 'normalization', None)

                        if norm_type == 'LayerNorm':
                            if not hasattr(self.base_layer, '_layernorm'):
                                raise RuntimeError(
                                    'NPU LoRA path expects base_layer to provide `_layernorm`, but it is missing. '
                                    'Cannot reconstruct the post-LayerNorm activation for LoRA.')
                            x = self.base_layer._layernorm(inp)
                        else:
                            # Default to RMSNorm path when normalization is not LayerNorm.
                            if not hasattr(self.base_layer, '_rmsnorm'):
                                raise RuntimeError(
                                    'NPU LoRA path expects base_layer to provide `_rmsnorm`, but it is missing. '
                                    'Cannot reconstruct the post-RMSNorm activation for LoRA.')
                            x = self.base_layer._rmsnorm(inp)
                    else:
                        raise RuntimeError('NPU LoRA path requires base_layer to expose post-norm activations '
                                           '(LayerNorm/RMSNorm output). Expected base_layer to have `config` '
                                           'and either `_layernorm` or `_rmsnorm`. '
                                           f'Got base_layer type: {type(self.base_layer)}. ')
                else:
                    (result, x), bias = self.base_layer(x, *args, **kwargs)
        elif isinstance(self.base_layer, (TELinear, TEGroupedLinear)):
            result, bias = self.base_layer(x, *args, **kwargs)
        elif isinstance(self.base_layer, TopKRouter):
            with self._patch_router_gating():
                result, bias = self.base_layer(x, *args, **kwargs)
        else:
            raise ValueError(f'Unsupported base layer type: {type(self.base_layer)}')
        if not isinstance(self.base_layer, TopKRouter) and not self.disable_adapters and not self.merged:
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                dtype = lora_A.weight0.dtype if isinstance(lora_A, (TEGroupedLinear,
                                                                    NpuGroupedLoraLinear)) else lora_A.weight.dtype
                x = x.to(dtype)

                if (os.environ.get('DSV4_LORA_DEBUG_SHAPES', '0') == '1'
                        and not self._dsv4_lora_debug_logged and result.numel() > 2**28):
                    print(
                        '[dsv4-lora-debug] rank={} base={} is_grouped={} x_shape={} '
                        'result_shape={} lora_a={} lora_b={} linear_chunk={} grouped_chunk={} '.format(
                            os.environ.get('RANK', '?'),
                            type(self.base_layer).__name__,
                            self.is_grouped,
                            tuple(x.shape),
                            tuple(result.shape),
                            type(lora_A).__name__,
                            type(lora_B).__name__,
                            os.environ.get('DSV4_LORA_LINEAR_CHUNK_SIZE', '0'),
                            os.environ.get('DSV4_LORA_GROUPED_CHUNK_SIZE', '0'),
                        ),
                        flush=True,
                    )
                    if not isinstance(lora_A, (TEGroupedLinear, NpuGroupedLoraLinear)):
                        print(
                            '[dsv4-lora-debug] linear_candidate={} result_gt_chunk={} '
                            'result_mod_x={} '.format(
                                not self.is_grouped,
                                result.shape[0] > int(
                                    os.environ.get('DSV4_LORA_LINEAR_CHUNK_SIZE', '0') or 0
                                ),
                                (result.shape[0] % x.shape[0]) if x.shape[0] else 'zero',
                            ),
                            flush=True,
                        )
                    self._dsv4_lora_debug_logged = True

                # Use the adapter's actual grouped type instead of the base
                # layer's class hierarchy.  Some TE wrappers expose a
                # sequence-parallel column-linear base under a grouped-looking
                # wrapper, while their LoRA adapters are ordinary linears.
                if isinstance(lora_A, (TEGroupedLinear, NpuGroupedLoraLinear)):
                    chunked_result = self._forward_grouped_lora_chunked(
                        result, x, lora_A, lora_B, dropout, scaling, args, kwargs)
                    if chunked_result is not None:
                        result = chunked_result
                        continue
                    if (os.environ.get('DSV4_LORA_DEBUG_SHAPES', '0') == '1'
                            and result.numel() > 2**28):
                        print(
                            '[dsv4-lora-debug] grouped_chunk_fallback base={} '
                            'x_shape={} result_shape={} args_len={} lora_a={} lora_b={}'.format(
                                type(self.base_layer).__name__,
                                tuple(x.shape),
                                tuple(result.shape),
                                len(args),
                                type(lora_A).__name__,
                                type(lora_B).__name__,
                            ),
                            flush=True,
                        )
                else:
                    chunked_result = self._forward_lora_linear_chunked(
                        result, x, lora_A, lora_B, dropout, scaling)
                    if chunked_result is not None:
                        result = chunked_result
                        continue
                    if (os.environ.get('DSV4_LORA_DEBUG_SHAPES', '0') == '1'
                            and result.numel() > 2**28):
                        print(
                            '[dsv4-lora-debug] linear_chunk_fallback base={} '
                            'x_shape={} result_shape={} lora_a={} lora_b={}'.format(
                                type(self.base_layer).__name__,
                                tuple(x.shape),
                                tuple(result.shape),
                                type(lora_A).__name__,
                                type(lora_B).__name__,
                            ),
                            flush=True,
                        )

                lora_result = lora_A(dropout(x), *args, **kwargs) if isinstance(
                    lora_A, (TEGroupedLinear, NpuGroupedLoraLinear)) else lora_A(dropout(x))
                if isinstance(lora_result, tuple):
                    lora_result = lora_result[0]
                lora_result = lora_B(lora_result, *args, **kwargs) if isinstance(
                    lora_B, (TEGroupedLinear, NpuGroupedLoraLinear)) else lora_B(lora_result)
                if isinstance(lora_result, tuple):
                    lora_result = lora_result[0]
                if lora_result.shape != result.shape:
                    raise RuntimeError(
                        'DSV4 LoRA fallback received mismatched base and adapter outputs: '
                        f'base={tuple(result.shape)}, adapter={tuple(lora_result.shape)}'
                    )
                # The ordinary fallback used to allocate one full output for
                # scaling and another for ``result + lora_result``.  Long TP
                # streamed-Q chunks make that pair of copies large enough to
                # OOM even though the adapter rank is tiny.  Reuse the same
                # explicit autograd bridge as the chunked path; it preserves
                # gradients for both branches without a full-size temporary.
                result = _InplaceScaledAdd.apply(result, lora_result, 0, scaling)

        result = result.to(previous_dtype)
        return result, bias

    def sharded_state_dict(
            self,
            prefix: str = '',
            sharded_offsets: Tuple[Tuple[int, int, int]] = (),
            metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        sharded_state_dict = tuners_sharded_state_dict(self, prefix, sharded_offsets, metadata)
        if prefix.endswith('linear_fc1.'):
            if isinstance(self.base_layer, TEGroupedLinear) and self.config.gated_linear_unit:
                num_global_experts = (parallel_state.get_expert_model_parallel_world_size() * self.base_layer.num_gemms)
                local_expert_indices_offset = (
                    parallel_state.get_expert_model_parallel_rank() * self.base_layer.num_gemms)
                ep_axis = len(sharded_offsets)
                for i in range(self.base_layer.num_gemms):
                    new_sharded_offsets = (
                        *sharded_offsets,
                        (ep_axis, local_expert_indices_offset + i, num_global_experts),
                    )
                    for k in (f'{prefix}base_layer.weight{i}', f'{prefix}base_layer.bias{i}'):
                        if k in sharded_state_dict:
                            sharded_state_dict[k] = apply_swiglu_sharded_factory(sharded_state_dict[k],
                                                                                 new_sharded_offsets)
            else:
                for k, v in sharded_state_dict.items():
                    if k in [f'{prefix}base_layer.weight', f'{prefix}base_layer.bias']:
                        sharded_state_dict[k] = apply_swiglu_sharded_factory(sharded_state_dict[k], sharded_offsets)
        return sharded_state_dict

    def get_delta_weights(self, adapter) -> List[torch.Tensor]:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        lora_A = self.lora_A[adapter]
        lora_B = self.lora_B[adapter]
        if self.is_grouped:
            weight_A = [getattr(lora_A, f'weight{i}') for i in range(lora_A.num_gemms)]
            weight_B = [getattr(lora_B, f'weight{i}') for i in range(lora_B.num_gemms)]
        else:
            weight_A = [self.lora_A[adapter].weight]
            weight_B = [self.lora_B[adapter].weight]
        output_tensor = []
        assert len(weight_A) == len(weight_B)
        for i in range(len(weight_B)):
            output_tensor.append(transpose(weight_B[i] @ weight_A[i], self.fan_in_fan_out) * self.scaling[adapter])

        return output_tensor

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        base_layer = self.get_base_layer()
        origin_device = base_layer.weight0.device if self.is_grouped else base_layer.weight.device
        if origin_device.type == 'cpu':
            self.to(device=get_current_device())
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                if self.is_grouped:
                    orig_weights = [getattr(base_layer, f'weight{i}') for i in range(base_layer.num_gemms)]
                else:
                    orig_weights = [base_layer.weight]
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = [weight.data.clone() for weight in orig_weights]
                    delta_weights = self.get_delta_weights(active_adapter)
                    for orig_weight, delta_weight in zip(orig_weights, delta_weights):
                        orig_weight.data += delta_weight
                    if not all(torch.isfinite(orig_weights[i]).all() for i in range(len(orig_weights))):
                        raise ValueError(
                            f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                    if self.is_grouped:
                        for i in range(base_layer.num_gemms):
                            weight = getattr(base_layer, f'weight{i}')
                            weight.data = orig_weights[i]
                    else:
                        base_layer.weight.data = orig_weights[0]
                else:
                    delta_weights = self.get_delta_weights(active_adapter)
                    for orig_weight, delta_weight in zip(orig_weights, delta_weights):
                        orig_weight.data += delta_weight
                self.merged_adapters.append(active_adapter)
        if origin_device.type == 'cpu':
            self.to(device=origin_device)

    def unmerge(self) -> None:
        """
        Unmerge all merged adapter weights from the base weights.

        This method reverses the merge operation by subtracting the LoRA delta weights
        from the base layer weights, restoring the original base weights.
        """
        if not self.merged:
            # No adapters to unmerge
            return

        base_layer = self.get_base_layer()
        origin_device = base_layer.weight0.device if self.is_grouped else base_layer.weight.device
        if origin_device.type == 'cpu':
            self.to(device=get_current_device())

        for active_adapter in self.merged_adapters:
            if active_adapter in self.lora_A.keys():
                if self.is_grouped:
                    orig_weights = [getattr(base_layer, f'weight{i}') for i in range(base_layer.num_gemms)]
                else:
                    orig_weights = [base_layer.weight]

                delta_weights = self.get_delta_weights(active_adapter)
                for orig_weight, delta_weight in zip(orig_weights, delta_weights):
                    # Subtract the delta weight to unmerge
                    orig_weight.data -= delta_weight

        # Clear the merged adapters list
        self.merged_adapters = []

        if origin_device.type == 'cpu':
            self.to(device=origin_device)
