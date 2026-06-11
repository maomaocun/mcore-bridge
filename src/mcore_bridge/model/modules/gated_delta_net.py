# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import torch
import torch.nn.functional as F
import transformer_engine
from contextlib import nullcontext
from functools import lru_cache
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import all_to_all_hp2sp, all_to_all_sp2hp
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint, sharded_state_dict_default
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push
from typing import List, Optional

from mcore_bridge.config import ModelConfig

try:
    from fla.modules.convolution import causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    causal_conv1d = None
    l2norm = None
    chunk_gated_delta_rule = None

try:
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet as _GatedDeltaNet
    from megatron.core.ssm.gated_delta_net import GatedDeltaNetSubmodules, torch_chunk_gated_delta_rule
except ImportError:
    _GatedDeltaNet = object

try:
    from megatron.core.ssm.gated_delta_net import tensor_a2a_cp2hp as _mcore_tensor_a2a_cp2hp
    from megatron.core.ssm.gated_delta_net import tensor_a2a_hp2cp as _mcore_tensor_a2a_hp2cp
except ImportError:
    _mcore_tensor_a2a_cp2hp = None
    _mcore_tensor_a2a_hp2cp = None


# Code borrowed from NVIDIA/Megatron-LM
def _unpack_sequence(x, cu_seqlens, dim=1):
    unpacked_x = []
    num_seqs = cu_seqlens.shape[0] - 1
    for i in range(num_seqs):
        idx_start = cu_seqlens[i].item()
        idx_end = cu_seqlens[i + 1].item()
        chunked_index = [slice(None)] * dim + [slice(idx_start, idx_end)]
        unpacked_x.append(x[tuple(chunked_index)])
    return unpacked_x


# Code borrowed from NVIDIA/Megatron-LM
# Avoid the warning caused by `param[slices]`
def get_parameter_local_cp(
    param: torch.Tensor,
    dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """Get the local parameter for the current context parallel rank.

    Args:
        param (torch.Tensor): The entire parameter to get the local parameter for.
        dim (int): The dimension to split the parameter along. Usually the dimension of head.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None,
            first split the parameter along the dimension dim into sections,
            then get the local hidden parallel weights separately,
            finally concatenate the local hidden parallel weights along the dimension dim.

    Returns:
        torch.Tensor: The local parameter for the current context parallel rank.
    """

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    # No need to split if CP size is 1.
    if cp_size == 1:
        return param

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(param, split_sections, dim=dim)
        outputs = []
        for p in inputs:
            p = get_parameter_local_cp(p, dim, cp_group)
            outputs.append(p)
        return torch.cat(outputs, dim=dim)

    # Slice the parameter.
    slices = [slice(None)] * param.dim()
    dim_size = param.size(dim=dim)
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    param = param[tuple(slices)]
    return param


def _all_to_all_cp2hp(input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup) -> torch.Tensor:
    assert input_.dim() == 3, 'all_to_all_cp2hp assumes 3-d input shape.'
    s_in, b_in, h_in = input_.shape
    s_out, h_out = s_in * cp_group.size(), h_in // cp_group.size()
    output = all_to_all_sp2hp(input_, group=cp_group)
    return output.reshape(s_out, b_in, h_out)


def _all_to_all_hp2cp(input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup) -> torch.Tensor:
    assert input_.dim() == 3, 'all_to_all_hp2cp assumes 3-d input shape.'
    s_in, b_in, h_in = input_.shape
    s_out, h_out = s_in // cp_group.size(), h_in * cp_group.size()
    output = all_to_all_hp2sp(input_, group=cp_group)
    return output.reshape(s_out, b_in, h_out)


def _undo_attention_load_balancing(input_: torch.Tensor, cp_size: int) -> torch.Tensor:
    num_chunks = 2 * cp_size
    chunks = torch.chunk(input_, chunks=num_chunks, dim=0)
    order = [2 * i for i in range(cp_size)] + [num_chunks - 2 * i - 1 for i in range(cp_size)]
    return torch.cat([chunks[i] for i in order], dim=0)


def _redo_attention_load_balancing(input_: torch.Tensor, cp_size: int) -> torch.Tensor:
    num_chunks = 2 * cp_size
    chunks = torch.chunk(input_, chunks=num_chunks, dim=0)
    order = [None] * num_chunks
    order[::2] = range(cp_size)
    order[1::2] = reversed(range(cp_size, num_chunks))
    return torch.cat([chunks[i] for i in order], dim=0)


def tensor_a2a_cp2hp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
    undo_attention_load_balancing: bool = True,
):
    if _mcore_tensor_a2a_cp2hp is not None:
        return _mcore_tensor_a2a_cp2hp(
            tensor,
            seq_dim=seq_dim,
            head_dim=head_dim,
            cp_group=cp_group,
            split_sections=split_sections,
            undo_attention_load_balancing=undo_attention_load_balancing,
        )

    cp_size = cp_group.size()
    if cp_size == 1:
        return tensor
    assert seq_dim == 0, f'tensor_a2a_cp2hp only supports seq_dim == 0 for now, but got {seq_dim=}'
    assert head_dim == -1 or head_dim == 2, f'tensor_a2a_cp2hp only supports head_dim == -1 or 2, got {head_dim=}'
    assert tensor.dim() == 3, f'tensor_a2a_cp2hp only supports 3-d input tensor, got {tensor.dim()=}'

    if split_sections is not None:
        outputs = [
            tensor_a2a_cp2hp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                undo_attention_load_balancing=False,
            ) for x in torch.split(tensor, split_sections, dim=head_dim)
        ]
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_cp2hp(tensor, cp_group)
    if undo_attention_load_balancing:
        tensor = _undo_attention_load_balancing(tensor, cp_size)
    return tensor


def tensor_a2a_hp2cp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
    redo_attention_load_balancing: bool = True,
):
    if _mcore_tensor_a2a_hp2cp is not None:
        return _mcore_tensor_a2a_hp2cp(
            tensor,
            seq_dim=seq_dim,
            head_dim=head_dim,
            cp_group=cp_group,
            split_sections=split_sections,
            redo_attention_load_balancing=redo_attention_load_balancing,
        )

    cp_size = cp_group.size()
    if cp_size == 1:
        return tensor
    assert seq_dim == 0, f'tensor_a2a_hp2cp only supports seq_dim == 0 for now, but got {seq_dim=}'
    assert head_dim == -1 or head_dim == 2, f'tensor_a2a_hp2cp only supports head_dim == -1 or 2, got {head_dim=}'
    assert tensor.dim() == 3, f'tensor_a2a_hp2cp only supports 3-d input tensor, got {tensor.dim()=}'

    if redo_attention_load_balancing:
        tensor = _redo_attention_load_balancing(tensor, cp_size)
    if split_sections is not None:
        outputs = [
            tensor_a2a_hp2cp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                redo_attention_load_balancing=False,
            ) for x in torch.split(tensor, split_sections, dim=head_dim)
        ]
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_hp2cp(tensor, cp_group)
    return tensor


def _build_thd_cp_a2a_perm(cu_seqlens: torch.Tensor, cp_size: int, t_global: int):
    cu = cu_seqlens.to(dtype=torch.long)
    seq_lens = torch.diff(cu)
    if (seq_lens % (2 * cp_size) != 0).any():
        raise ValueError(
            f'GDN CP with THD format requires each packed sequence length to be divisible by '
            f'2*cp_size={2 * cp_size}, but got lengths: {seq_lens.tolist()}')
    t_local = t_global // cp_size
    positions = torch.arange(t_global, device=cu.device)
    seq_idx = torch.bucketize(positions, cu[1:], right=True)
    halves = seq_lens // (2 * cp_size)
    local_starts = cu[:-1] // cp_size
    global_starts = cu[:-1]
    half_i = halves[seq_idx]
    pos_in_seq = positions - global_starts[seq_idx]
    natural_chunk = pos_in_seq // half_i
    offset = pos_in_seq - natural_chunk * half_i
    lb_chunk = torch.where(natural_chunk < cp_size, 2 * natural_chunk, 4 * cp_size - 2 * natural_chunk - 1)
    rank = lb_chunk // 2
    half_within_rank = lb_chunk - 2 * rank
    k = half_within_rank * half_i + offset
    idx = rank * t_local + local_starts[seq_idx] + k
    inv = torch.empty_like(idx)
    inv[idx] = positions
    return idx, inv


@lru_cache(maxsize=8)
def _build_head_perm_for_split_sections(split_sections, cp_size: int, device: torch.device) -> torch.Tensor:
    assert all(s % cp_size == 0 for s in split_sections), (
        f'split_sections {split_sections} must be divisible by cp_size {cp_size} for GDN')
    offset = 0
    parts = []
    for size in split_sections:
        parts.append(torch.arange(offset, offset + size, device=device, dtype=torch.long).view(cp_size, -1))
        offset += size
    return torch.cat(parts, dim=-1).view(-1)


class GatedDeltaNet(_GatedDeltaNet):

    def __init__(self, config: ModelConfig, submodules: 'GatedDeltaNetSubmodules', *args, **kwargs):
        if config.linear_decoupled_in_proj:
            in_proj = submodules.in_proj
            submodules.in_proj = IdentityOp
        if 'cp_comm_type' not in inspect.signature(_GatedDeltaNet).parameters:
            kwargs.pop('cp_comm_type', None)
        try:
            super().__init__(config, submodules, *args, **kwargs)
        finally:
            if config.linear_decoupled_in_proj:
                submodules.in_proj = in_proj
        self.cp_size = self.pg_collection.cp.size()
        self.qk_dim_local_tp = self.qk_dim // self.tp_size
        self.v_dim_local_tp = self.v_dim // self.tp_size
        self.conv_dim_local_tp = self.conv_dim // self.tp_size
        if not config.linear_decoupled_in_proj:
            return
        self.in_proj_qkvz_dim = self.qk_dim * 2 + self.v_dim * 2
        self.in_proj_ba_dim = self.num_value_heads * 2
        del self.in_proj
        self.in_proj_qkvz = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_qkvz_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='fc1',
            tp_group=self.pg_collection.tp,
        )
        if config.fp8_param:
            fp8_context = transformer_engine.pytorch.fp8_model_init(enabled=False)
        else:
            fp8_context = nullcontext()
        with fp8_context:
            self.in_proj_ba = build_module(
                submodules.in_proj,
                self.hidden_size,
                self.in_proj_ba_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='fc1_ba',
                tp_group=self.pg_collection.tp,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        """
        Perform a forward pass through the GDN module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) GDN output and bias.

        """
        # TODO: Deal with attention_mask (There is an issue when left padding is used.)
        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        cp_size = self.config.context_parallel_size
        seq_len = seq_len * self.sp_size * cp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()), 'GDN does not currently support dynamic inference batching.'
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError('GDN does not support inference for now.')

        if not self.config.linear_decoupled_in_proj:
            return self._forward_merged_in_proj(hidden_states, packed_seq_params)

        cu_seqlens = None if packed_seq_params is None else packed_seq_params.cu_seqlens_q
        # Input projection
        num_key_heads_per_device = self.num_key_heads // self.tp_size // cp_size
        nvtx_range_push(suffix='in_proj')
        if self.config.linear_decoupled_in_proj:
            qkvz, _ = self.in_proj_qkvz(hidden_states)
            if self.config.fp8_param:
                fp8_context = transformer_engine.pytorch.fp8_autocast(enabled=False)
            else:
                fp8_context = nullcontext()
            with fp8_context:
                ba, _ = self.in_proj_ba(hidden_states)
            qkvz = qkvz.view(qkvz.shape[:-1] + (num_key_heads_per_device, qkvz.shape[-1] // num_key_heads_per_device))
            ba = ba.view(ba.shape[:-1] + (num_key_heads_per_device, ba.shape[-1] // num_key_heads_per_device))
            qkvzba = torch.concat([qkvz, ba], dim=-1).view(*qkvz.shape[:2], -1)
        else:
            qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix='in_proj')

        if cp_size > 1:
            from megatron.core.ssm.gated_delta_net import tensor_a2a_cp2hp, tensor_a2a_hp2cp
            if cu_seqlens is not None:
                unpacked_qkvzba = _unpack_sequence(qkvzba, cu_seqlens // self.cp_size, dim=0)
                outputs = []
                for qkvzba_i in unpacked_qkvzba:
                    qkvzba_i = tensor_a2a_cp2hp(
                        qkvzba_i,
                        seq_dim=0,
                        head_dim=-1,
                        cp_group=self.pg_collection.cp,
                    )
                    outputs.append(qkvzba_i)
                qkvzba = torch.cat(outputs, dim=0)
            else:
                # CP All to All: CP to HP
                qkvzba = tensor_a2a_cp2hp(
                    qkvzba,
                    seq_dim=0,
                    head_dim=-1,
                    cp_group=self.pg_collection.cp,
                )
        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        qkvzba = qkvzba.view(qkvzba.shape[:-1]
                             + (num_key_heads_per_device, qkvzba.shape[-1] // num_key_heads_per_device))
        qkvzba = qkvzba.transpose(0, 1)
        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (self.qk_dim * 2 + self.v_dim) // self.num_key_heads,
                self.v_dim // self.num_key_heads,
                self.num_value_heads // self.num_key_heads,
                self.num_value_heads // self.num_key_heads,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)
        alpha = alpha.reshape(batch, seq_len, -1)
        qkv = qkv.reshape(batch, seq_len, -1)

        # Convolution on qkv
        nvtx_range_push(suffix='conv1d')
        if cp_size > 1:
            conv1d_weight = get_parameter_local_cp(
                self.conv1d.weight,
                dim=0,
                cp_group=self.pg_collection.cp,
            )
            conv1d_bias = (
                get_parameter_local_cp(
                    self.conv1d.bias,
                    dim=0,
                    cp_group=self.pg_collection.cp,
                ) if self.conv_bias else None)
        else:
            conv1d_weight = self.conv1d.weight
            conv1d_bias = self.conv1d.bias

        if (causal_conv1d is None) or self.config.deterministic_mode:
            assert cu_seqlens is None, 'Packed sequences are not supported when fla is not available.'
            qkv = qkv.transpose(1, 2).contiguous()  # b, s, d -> b, d, s
            conv_out = F.conv1d(
                input=qkv,
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
            )
            qkv = self.act_fn(conv_out[..., :seq_len])
            qkv = qkv.transpose(1, 2)  # b, d, s -> b, s, d
        else:
            assert self.activation in ['silu', 'swish']
            qkv = causal_conv1d(
                x=qkv,
                weight=conv1d_weight.squeeze(1),  # d, 1, w -> d, w
                bias=conv1d_bias,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
            )[0]
        nvtx_range_pop(suffix='conv1d')
        # Split qkv into query, key, and value
        qkv = qkv.view(qkv.shape[:-1] + (num_key_heads_per_device, qkv.shape[-1] // num_key_heads_per_device))
        query, key, value = torch.split(
            qkv,
            [self.qk_dim // self.num_key_heads, self.qk_dim // self.num_key_heads, self.v_dim // self.num_key_heads],
            dim=-1,
        )
        query = query.reshape(batch, seq_len, -1, self.key_head_dim)
        key = key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)
        # Apply L2 norm to query and key
        if self.use_qk_l2norm:
            query = l2norm(query.contiguous())
            key = l2norm(key.contiguous())
        if self.num_value_heads // self.num_key_heads > 1:
            query = query.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
            key = key.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)

        # Make contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        # Calculate g and beta
        nvtx_range_push(suffix='g_and_beta')
        if cp_size > 1:
            A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
            dt_bias_local_cp = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=self.pg_collection.cp)
        else:
            A_log_local_cp, dt_bias_local_cp = self.A_log, self.dt_bias
        g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)  # In fp32
        beta = beta.sigmoid()
        nvtx_range_pop(suffix='g_and_beta')

        nvtx_range_push(suffix='gated_delta_rule')
        if self.config.deterministic_mode:
            assert cu_seqlens is None, ('cu_seqlens is not supported for torch_chunk_gated_delta_rule for now.')
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=cu_seqlens,
            )
        nvtx_range_pop(suffix='gated_delta_rule')

        # RMSNorm
        nvtx_range_push(suffix='gated_norm')
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix='gated_norm')

        # Transpose: b s x --> s b x
        # From bshd back to sbhd format
        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()
        if cp_size > 1:
            if cu_seqlens is not None:
                unpacked_norm_out = _unpack_sequence(norm_out, cu_seqlens, dim=0)
                outputs = []
                for norm_out_i in unpacked_norm_out:
                    norm_out_i = tensor_a2a_hp2cp(norm_out_i, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp)
                    outputs.append(norm_out_i)
                norm_out = torch.cat(outputs, dim=0)
            else:
                norm_out = tensor_a2a_hp2cp(norm_out, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp)

        # Output projection
        nvtx_range_push(suffix='out_proj')
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix='out_proj')

        return out, out_bias

    def _forward_merged_in_proj(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        seq_len, batch, _ = hidden_states.shape
        cp_size = self.cp_size
        seq_len = seq_len * self.sp_size * cp_size
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        thd_cp_a2a_inv = None

        if packed_seq:
            assert batch == 1, 'Packed sequence expects batch dimension to be 1.'
            if self.config.deterministic_mode:
                raise NotImplementedError('GDN CP with packed sequence does not support deterministic mode.')
            cu_seqlens_q = (packed_seq_params.cu_seqlens_q_padded
                            if packed_seq_params.cu_seqlens_q_padded is not None else packed_seq_params.cu_seqlens_q)
            cu_seqlens_kv = (packed_seq_params.cu_seqlens_kv_padded if packed_seq_params.cu_seqlens_kv_padded
                             is not None else packed_seq_params.cu_seqlens_kv)
            if cu_seqlens_q[-1].item() != seq_len:
                raise ValueError(
                    f'GDN packed cu_seqlens_q[-1]={cu_seqlens_q[-1].item()} does not match total seq_len={seq_len}.')
            if not torch.equal(cu_seqlens_q, cu_seqlens_kv):
                raise ValueError('GDN currently requires cu_seqlens_q and cu_seqlens_kv to match.')
        else:
            cu_seqlens_q = None

        nvtx_range_push(suffix='in_proj')
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix='in_proj')

        if cp_size > 1:
            head_perm = _build_head_perm_for_split_sections(
                (
                    self.qk_dim_local_tp,
                    self.qk_dim_local_tp,
                    self.v_dim_local_tp,
                    self.v_dim_local_tp,
                    self.num_value_heads // self.tp_size,
                    self.num_value_heads // self.tp_size,
                ),
                cp_size,
                qkvzba.device,
            )
            qkvzba = qkvzba.index_select(-1, head_perm)

        if packed_seq:
            qkvzba = tensor_a2a_cp2hp(
                qkvzba,
                seq_dim=0,
                head_dim=-1,
                cp_group=self.pg_collection.cp,
                undo_attention_load_balancing=False,
            )
            if cp_size > 1:
                thd_cp_a2a_idx, thd_cp_a2a_inv = _build_thd_cp_a2a_perm(cu_seqlens_q, cp_size, seq_len)
                qkvzba = qkvzba.index_select(0, thd_cp_a2a_idx)
        else:
            qkvzba = tensor_a2a_cp2hp(qkvzba, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp)

        qkvzba = qkvzba.transpose(0, 1)
        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // cp_size,
                self.v_dim_local_tp // cp_size,
                self.num_value_heads // self.tp_size // cp_size,
                self.num_value_heads // self.tp_size // cp_size,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)
        alpha = alpha.reshape(batch, seq_len, -1)

        nvtx_range_push(suffix='conv1d')
        qkv_channels_split_sections = [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp]
        conv1d_weight = get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=self.pg_collection.cp,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=self.pg_collection.cp,
                split_sections=qkv_channels_split_sections,
            ) if self.conv_bias else None)
        if (causal_conv1d is None) or self.config.deterministic_mode:
            assert cu_seqlens_q is None, 'Packed sequences are not supported when fla is not available.'
            qkv = qkv.transpose(1, 2).contiguous()
            conv_out = F.conv1d(
                input=qkv,
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
                groups=self.conv_dim_local_tp // cp_size,
            )
            qkv = self.act_fn(conv_out[..., :seq_len])
            qkv = qkv.transpose(1, 2)
        else:
            assert self.activation in ['silu', 'swish']
            qkv = causal_conv1d(
                x=qkv,
                weight=conv1d_weight.squeeze(1),
                bias=conv1d_bias,
                activation=self.activation,
                cu_seqlens=cu_seqlens_q,
            )[0]
        nvtx_range_pop(suffix='conv1d')

        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // cp_size, self.v_dim_local_tp // cp_size],
            dim=-1,
        )
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)
        if self.use_qk_l2norm:
            query_key = l2norm(query_key.contiguous())
        num_query_key_heads_per_device = self.qk_dim_local_tp // self.key_head_dim // cp_size
        query, key = torch.split(query_key, [num_query_key_heads_per_device, num_query_key_heads_per_device], dim=2)
        if self.num_value_heads // self.num_key_heads > 1:
            repeat_factor = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        nvtx_range_push(suffix='g_and_beta')
        A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
        dt_bias_local_cp = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=self.pg_collection.cp)
        g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)
        beta = beta.sigmoid()
        nvtx_range_pop(suffix='g_and_beta')

        nvtx_range_push(suffix='gated_delta_rule')
        if self.config.deterministic_mode:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=cu_seqlens_q,
            )
        nvtx_range_pop(suffix='gated_delta_rule')

        nvtx_range_push(suffix='gated_norm')
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix='gated_norm')

        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()
        if packed_seq:
            if cp_size > 1:
                norm_out = norm_out.index_select(0, thd_cp_a2a_inv)
            norm_out = tensor_a2a_hp2cp(
                norm_out,
                seq_dim=0,
                head_dim=-1,
                cp_group=self.pg_collection.cp,
                redo_attention_load_balancing=False,
            )
        else:
            norm_out = tensor_a2a_hp2cp(norm_out, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp)

        nvtx_range_push(suffix='out_proj')
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix='out_proj')

        return out, out_bias

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None, tp_group=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group

        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        # Parameters
        self._save_to_state_dict(sharded_state_dict, '', keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={
                'A_log': 0,
                'dt_bias': 0,
            },  # parameters sharded across TP
            sharded_offsets=sharded_offsets,
            tp_group=(tp_group if tp_group is not None else self.pg_collection.tp),
            dp_cp_group=metadata['dp_cp_group'],
        )
        # Submodules
        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        for name, module in self.named_children():
            if name == 'conv1d':
                # Add TP sharding for Conv1d
                module_sd = module.state_dict(prefix='', keep_vars=True)
                tp_sharding_map = {'weight': 0}
                if self.conv_bias:
                    tp_sharding_map['bias'] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f'{prefix}{name}.',
                    tp_sharding_map,
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f'{prefix}{name}.', sharded_offsets, metadata, tp_group=tp_group)

            sharded_state_dict.update(module_sharded_sd)

        return sharded_state_dict
