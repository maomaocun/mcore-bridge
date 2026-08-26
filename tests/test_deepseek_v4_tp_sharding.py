from types import SimpleNamespace

import torch

from mcore_bridge.bridge.gpt_bridge import GPTBridge


def _bridge(tp_size=2, tp_rank=1):
    bridge = GPTBridge.__new__(GPTBridge)
    bridge.config = SimpleNamespace(task_type='causal_lm')
    bridge.tp_size = tp_size
    bridge.tp_rank = tp_rank
    bridge.etp_size = 1
    bridge.etp_rank = 0
    return bridge


def test_dsv4_tp_axes_are_explicit():
    bridge = _bridge()
    assert bridge._get_tp_split_dim('linear_o_group_proj') == 0
    assert bridge._get_tp_split_dim('core_attention.attn_sink') == 0
    assert bridge._get_tp_split_dim('linear_kv_proj.weight') is None
    assert bridge._get_tp_split_dim('linear_q_up_proj.weight') == 0
    assert bridge._get_tp_split_dim('linear_proj.weight') == 1


def test_dsv4_grouped_output_splits_contiguous_groups():
    bridge = _bridge(tp_size=2, tp_rank=1)
    weight = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    shard = bridge._split_tp(weight, bridge._get_tp_split_dim('linear_o_group_proj'), False, False)
    assert shard.shape == (4, 4)
    assert torch.equal(shard, weight[4:])


def test_dsv4_attention_sink_splits_heads():
    bridge = _bridge(tp_size=8, tp_rank=3)
    sink = torch.arange(64, dtype=torch.float32)
    shard = bridge._split_tp(sink, bridge._get_tp_split_dim('core_attention.attn_sink'), False, False)
    assert shard.shape == (8,)
    assert torch.equal(shard, sink[24:32])
