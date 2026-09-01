# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU/reference tests for the QSA selected-KV contract.

The same tests can be run on a Hopper worker with ``pytest``; the public
Triton parity test is intentionally separate so a CPU-only checkout still
verifies the index semantics and custom backward.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from mcore_bridge.model.modules.qsa_attention import (qsa_expand_block_route, qsa_sparse_forward,
                                                      qsa_sparse_forward_packed, qsa_sparse_forward_reference,
                                                      resolve_qsa_backend)
from mcore_bridge.model.modules.qsa_cp_exchange import _build_owner_plan, qsa_exchange_selected_kv
from mcore_bridge.model.modules.qsa_indexer import QSAIndexer
from mcore_bridge.model.modules.qsa_triton import (
    qsa_attention_packed_metadata_from_cu,
    qsa_expand_compact_route_triton,
    qsa_indexer_fused_topk_packed,
    qsa_indexer_fused_topk_with_ratio,
    qsa_indexer_packed_metadata_from_cu,
    qsa_indexer_slab_topk_with_ratio,
)
from mcore_bridge.model.gpts.qwen4_exp import Qwen4ExpLayer


def _dense_selected_attention(query, key, value, indices, lengths, scale):
    """Independent small-shape reference in SBHD layout."""

    sq, batch, num_q_heads, dim = query.shape
    sk, _, num_kv_heads, _ = key.shape
    group_size = num_q_heads // num_kv_heads
    output = torch.zeros_like(query)
    lse = torch.full((batch, num_q_heads, sq), -float('inf'), dtype=torch.float32)
    for b in range(batch):
        for q_pos in range(sq):
            length = int(lengths[b, q_pos].item())
            row = indices[b, q_pos, :length].to(torch.long)
            row = row[(row >= 0) & (row < sk) & (row <= q_pos)]
            for q_head in range(num_q_heads):
                kv_head = q_head // group_size
                scores = query[q_pos, b, q_head].float() @ key[row, b, kv_head].float().transpose(0, 1)
                scores = scores * scale
                row_lse = torch.logsumexp(scores, dim=-1)
                probs = torch.softmax(scores, dim=-1)
                output[q_pos, b, q_head] = (probs[:, None] * value[row, b, kv_head].float()).sum(0).to(query.dtype)
                lse[b, q_head, q_pos] = row_lse
    return output, lse


def _sorted_valid_route(indices, lengths):
    """Canonicalize route sets without treating compact order as an ABI."""

    slots = torch.arange(indices.shape[-1], device=indices.device)
    valid = slots.reshape(*([1] * (indices.ndim - 1)), -1) < lengths.long().unsqueeze(-1)
    return torch.where(valid, indices, torch.full_like(indices, -1)).sort(dim=-1).values


def test_selected_kv_forward_matches_independent_gqa2_reference():
    torch.manual_seed(7)
    sq, batch, hq, hkv, dim, slots = 6, 2, 4, 2, 5, 5
    query = torch.randn(sq, batch, hq, dim, dtype=torch.float32)
    key = torch.randn(sq, batch, hkv, dim, dtype=torch.float32)
    value = torch.randn(sq, batch, hkv, dim, dtype=torch.float32)
    indices = torch.tensor([
        [[0, -1, -1, -1, -1], [0, 1, -1, -1, -1], [1, 1, 2, -1, -1], [0, 2, 3, 3, -1], [4, 0, 4, 2, -1],
         [1, 5, 2, 5, -1]],
        [[0, -1, -1, -1, -1], [1, 0, -1, -1, -1], [2, 0, 2, -1, -1], [3, 1, 3, 0, -1], [0, 4, 2, 4, -1],
         [5, 1, 5, 3, -1]],
    ], dtype=torch.int32)
    lengths = torch.tensor([[1, 2, 3, 4, 4, 4], [1, 2, 3, 4, 4, 4]], dtype=torch.int32)
    scale = 1.0 / math.sqrt(dim)

    expected, expected_lse = _dense_selected_attention(query, key, value, indices, lengths, scale)
    actual, actual_lse = qsa_sparse_forward(
        query, key, value, indices, lengths, softmax_scale=scale, backend='torch')
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_lse, expected_lse, atol=1e-6, rtol=1e-6)


def test_selected_kv_supports_tp_local_three_to_one_gqa_shape():
    torch.manual_seed(8)
    query = torch.randn(4, 1, 3, 4)
    key = torch.randn(4, 1, 1, 4)
    value = torch.randn(4, 1, 1, 4)
    indices = torch.tensor([[[0, -1, -1], [0, 1, -1], [1, 2, 0], [3, 1, 3]]], dtype=torch.int32)
    lengths = torch.tensor([[1, 2, 3, 3]], dtype=torch.int32)
    expected, expected_lse = _dense_selected_attention(query, key, value, indices, lengths, 0.5)
    actual, actual_lse = qsa_sparse_forward(query, key, value, indices, lengths, 0.5, backend='torch')
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_lse, expected_lse, atol=1e-6, rtol=1e-6)


def test_selected_kv_packed_segments_are_isolated():
    torch.manual_seed(82)
    first, second = 5, 7
    total, hq, hkv, dim, slots = first + second, 4, 2, 3, 5
    query = torch.randn(total, hq, dim)
    key = torch.randn(total, hkv, dim)
    value = torch.randn_like(key)
    indices = torch.full((1, total, slots), -1, dtype=torch.int32)
    lengths = torch.zeros((1, total), dtype=torch.int32)
    for start, length in ((0, first), (first, second)):
        for offset in range(length):
            count = min(offset + 1, slots)
            indices[0, start + offset, :count] = torch.arange(count)
            lengths[0, start + offset] = count
    cu = torch.tensor([0, first, total], dtype=torch.int32)
    actual, actual_lse = qsa_sparse_forward_packed(
        query, key, value, indices, lengths, cu, cu, backend='torch')
    expected_outputs = []
    expected_lse = []
    for start, length in ((0, first), (first, second)):
        output, lse = qsa_sparse_forward(
            query[start:start + length].unsqueeze(1),
            key[start:start + length].unsqueeze(1),
            value[start:start + length].unsqueeze(1),
            indices[:, start:start + length],
            lengths[:, start:start + length],
            backend='torch')
        expected_outputs.append(output.squeeze(1))
        expected_lse.append(lse)
    assert torch.allclose(actual, torch.cat(expected_outputs), atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_lse, torch.cat(expected_lse, dim=2), atol=1e-6, rtol=1e-6)


def test_packed_boundary_validation_cache_tracks_inplace_mutation():
    total, hq, hkv, dim, slots = 4, 2, 1, 3, 2
    query = torch.randn(total, hq, dim)
    key = torch.randn(total, hkv, dim)
    value = torch.randn_like(key)
    indices = torch.zeros((1, total, slots), dtype=torch.int32)
    lengths = torch.ones((1, total), dtype=torch.int32)
    cu = torch.tensor([0, 2, total], dtype=torch.int32)
    qsa_sparse_forward_packed(
        query, key, value, indices, lengths, cu, cu, backend='torch')
    cu[1] = total + 1
    with pytest.raises(ValueError, match='non-decreasing'):
        qsa_sparse_forward_packed(
            query, key, value, indices, lengths, cu, cu, backend='torch')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_packed_varlen_matches_segment_loop_on_sm90():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(76)
    device = 'cuda'
    segments = (17, 11, 8)
    total = sum(segments)
    query = torch.randn(total, 4, 16, device=device, dtype=torch.bfloat16)
    key = torch.randn(total, 2, 16, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    cu = torch.tensor([0, 17, 28, 36], device=device, dtype=torch.int32)
    slots = 9
    indices = torch.full((1, total, slots), -1, device=device, dtype=torch.int32)
    lengths = torch.zeros((1, total), device=device, dtype=torch.int32)
    offset = 0
    for segment_length in segments:
        for row in range(segment_length):
            count = min(row + 1, slots)
            indices[0, offset + row, :count] = torch.arange(
                count, device=device, dtype=torch.int32)
            lengths[0, offset + row] = count
        offset += segment_length
    actual, actual_lse = qsa_sparse_forward_packed(
        query, key, value, indices, lengths, cu, cu,
        softmax_scale=16 ** -0.5, backend='triton')
    expected_outputs = []
    expected_lse = []
    offset = 0
    for segment_length in segments:
        output, lse = qsa_sparse_forward(
            query[offset:offset + segment_length].unsqueeze(1),
            key[offset:offset + segment_length].unsqueeze(1),
            value[offset:offset + segment_length].unsqueeze(1),
            indices[:, offset:offset + segment_length],
            lengths[:, offset:offset + segment_length],
            softmax_scale=16 ** -0.5,
            backend='torch',
            query_positions=torch.arange(segment_length, device=device, dtype=torch.int32),
        )
        expected_outputs.append(output.squeeze(1))
        expected_lse.append(lse)
        offset += segment_length
    assert torch.allclose(
        actual.float(), torch.cat(expected_outputs).float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(
        actual_lse, torch.cat(expected_lse, dim=2), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_packed_varlen_backward_matches_torch_on_sm90():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(761)
    device = 'cuda'
    segments = (17, 11, 8)
    total = sum(segments)
    hq, hkv, dim, slots = 4, 2, 16, 9
    query_base = torch.randn(total, hq, dim, device=device, dtype=torch.bfloat16)
    key_base = torch.randn(total, hkv, dim, device=device, dtype=torch.bfloat16)
    value_base = torch.randn_like(key_base)
    cu = torch.tensor([0, 17, 28, 36], device=device, dtype=torch.int32)
    indices = torch.full((1, total, slots), -1, device=device, dtype=torch.int32)
    lengths = torch.zeros((1, total), device=device, dtype=torch.int32)
    offset = 0
    for segment_length in segments:
        for row in range(segment_length):
            count = min(row + 1, slots)
            values = torch.arange(count, device=device, dtype=torch.int32)
            if count >= 5:
                values[-2:] = values[1]
            indices[0, offset + row, :count] = values
            lengths[0, offset + row] = count
        offset += segment_length
    grad_output = torch.randn_like(query_base)
    grad_lse = torch.randn(1, hq, total, device=device, dtype=torch.float32)

    def run(backend):
        query = query_base.detach().clone().requires_grad_()
        key = key_base.detach().clone().requires_grad_()
        value = value_base.detach().clone().requires_grad_()
        output, lse = qsa_sparse_forward_packed(
            query,
            key,
            value,
            indices,
            lengths,
            cu,
            cu,
            backend=backend,
            dkv_accum_dtype='fp32',
        )
        ((output.float() * grad_output.float()).sum() + (lse * grad_lse).sum()).backward()
        return output.detach(), lse.detach(), query.grad, key.grad, value.grad

    torch_result = run('torch')
    triton_result = run('triton')
    assert torch.allclose(triton_result[0].float(), torch_result[0].float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(triton_result[1], torch_result[1], atol=2e-2, rtol=2e-2)
    for triton_grad, torch_grad in zip(triton_result[2:], torch_result[2:]):
        assert torch.isfinite(triton_grad).all()
        assert torch.allclose(triton_grad.float(), torch_grad.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_packed_split64_backward_matches_torch_on_sm90(monkeypatch):
    """Cover the explicit packed K64 derivative-subtile tuning path."""

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    monkeypatch.setenv('MCORE_BRIDGE_QSA_BACKWARD_HEAD_TILE', '16')
    monkeypatch.setenv('MCORE_BRIDGE_QSA_BACKWARD_BLOCK_K', '64')

    torch.manual_seed(762)
    device = 'cuda'
    segments = (17, 11)
    total = sum(segments)
    hq, hkv, dim, slots = 24, 2, 256, 67
    query_base = torch.randn(
        total, hq, dim, device=device, dtype=torch.bfloat16)
    key_base = torch.randn(
        total, hkv, dim, device=device, dtype=torch.bfloat16)
    value_base = torch.randn_like(key_base)
    cu = torch.tensor([0, segments[0], total], device=device, dtype=torch.int32)
    indices = torch.full(
        (1, total, slots), -1, device=device, dtype=torch.int32)
    lengths = torch.zeros((1, total), device=device, dtype=torch.int32)
    offset = 0
    for segment_length in segments:
        for row in range(segment_length):
            count = row + 1
            values = torch.arange(count, device=device, dtype=torch.int32)
            if count >= 5:
                values[-2:] = values[1]
            indices[0, offset + row, :count] = values
            lengths[0, offset + row] = count
        offset += segment_length
    grad_output = torch.randn_like(query_base)
    grad_lse = torch.randn(
        1, hq, total, device=device, dtype=torch.float32)

    def run(backend):
        query = query_base.detach().clone().requires_grad_()
        key = key_base.detach().clone().requires_grad_()
        value = value_base.detach().clone().requires_grad_()
        output, lse = qsa_sparse_forward_packed(
            query,
            key,
            value,
            indices,
            lengths,
            cu,
            cu,
            backend=backend,
            dkv_accum_dtype='bf16',
        )
        ((output.float() * grad_output.float()).sum()
         + (lse * grad_lse).sum()).backward()
        return output.detach(), lse.detach(), query.grad, key.grad, value.grad

    torch_result = run('torch')
    triton_result = run('triton')
    assert torch.allclose(
        triton_result[0].float(), torch_result[0].float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(
        triton_result[1], torch_result[1], atol=2e-2, rtol=2e-2)
    for triton_grad, torch_grad in zip(triton_result[2:], torch_result[2:]):
        assert torch.isfinite(triton_grad).all()
        assert torch.allclose(
            triton_grad.float(), torch_grad.float(), atol=0.125, rtol=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize(
    'boundaries',
    ((0, 9), (0, 3, 8), (0, 7, 7, 16), (0, 1, 2, 3, 9)),
)
def test_triton_packed_metadata_from_cu_resets_segments_on_sm90(boundaries):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    ratio = 4
    cu = torch.tensor(boundaries, device='cuda', dtype=torch.int32)
    actual = qsa_indexer_packed_metadata_from_cu(
        cu, boundaries[-1], ratio)
    expected_starts = torch.empty(
        boundaries[-1], device='cuda', dtype=torch.int32)
    expected_counts = torch.empty_like(expected_starts)
    expected_positions = torch.empty_like(expected_starts)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        expected_starts[start:end] = start // ratio
        expected_counts[start:end] = (end - start) // ratio
        expected_positions[start:end] = torch.arange(
            end - start, device='cuda', dtype=torch.int32)
    expected = (expected_starts, expected_counts, expected_positions)
    assert all(torch.equal(got, want) for got, want in zip(actual, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', (torch.int32, torch.int64))
def test_triton_attention_packed_metadata_supports_distinct_q_kv_on_sm90(dtype):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    # The empty middle query segment exercises duplicate q boundaries; its KV
    # segment may still be non-empty. Query and KV lengths are intentionally
    # different so this cannot accidentally reuse self-attention offsets.
    cu_q = torch.tensor([0, 3, 3, 8], device='cuda', dtype=dtype)
    cu_kv = torch.tensor([0, 5, 7, 9], device='cuda', dtype=dtype)
    actual = qsa_attention_packed_metadata_from_cu(cu_q, cu_kv, 8)
    expected = (
        torch.tensor(
            [0, 0, 0, 7, 7, 7, 7, 7], device='cuda', dtype=torch.int32),
        torch.tensor(
            [5, 5, 5, 2, 2, 2, 2, 2], device='cuda', dtype=torch.int32),
        torch.tensor(
            [0, 1, 2, 0, 1, 2, 3, 4], device='cuda', dtype=torch.int32),
    )
    assert all(torch.equal(got, want) for got, want in zip(actual, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_packed_indexer_direct_fill_matches_segment_contract_on_sm90():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(77)
    device = 'cuda'
    ratio = 4
    block_topk = 8
    segments = (17, 11)
    total = sum(segments)
    query = torch.randn(total, 4, 16, device=device, dtype=torch.bfloat16)
    # The direct-fill route must not depend on score contents, but keep a
    # correctly shaped block-key tensor so the production API is exercised.
    block_keys = torch.randn(8, 16, device=device, dtype=torch.bfloat16)
    block_starts = torch.empty(total, device=device, dtype=torch.int32)
    block_counts = torch.empty(total, device=device, dtype=torch.int32)
    query_positions = torch.empty(total, device=device, dtype=torch.int32)
    offset = 0
    block_start = 0
    for segment_length in segments:
        segment_blocks = (segment_length + ratio - 1) // ratio
        block_starts[offset:offset + segment_length] = block_start
        block_counts[offset:offset + segment_length] = segment_blocks
        query_positions[offset:offset + segment_length] = torch.arange(
            segment_length, device=device, dtype=torch.int32)
        offset += segment_length
        block_start += segment_blocks
    actual = qsa_indexer_fused_topk_packed(
        query,
        block_keys,
        block_starts,
        block_counts,
        query_positions,
        ratio,
        block_topk,
    )
    compact = qsa_indexer_fused_topk_packed(
        query,
        block_keys,
        block_starts,
        block_counts,
        query_positions,
        ratio,
        block_topk,
        return_block_ids=True,
    )
    complete = torch.minimum(
        block_counts.to(torch.long),
        (query_positions.to(torch.long) + 1) // ratio,
    )
    lengths = (
        complete.clamp_max(block_topk) * ratio
        + query_positions.to(torch.long) + 1
        - complete * ratio
    ).to(torch.int32).unsqueeze(0)
    expanded = qsa_expand_block_route(
        compact.unsqueeze(0), lengths, query_positions, ratio)[0]
    assert torch.equal(
        _sorted_valid_route(expanded, lengths[0]),
        _sorted_valid_route(actual, lengths[0]),
    )
    expected = torch.full_like(actual, -1)
    for row in range(total):
        position = int(query_positions[row].item())
        complete = min(int(block_counts[row].item()), (position + 1) // ratio)
        for block in range(complete):
            expected[row, block * ratio:(block + 1) * ratio] = torch.arange(
                block * ratio,
                (block + 1) * ratio,
                device=device,
                dtype=torch.int32,
            )
        tail_start = complete * ratio
        for tail_offset in range(ratio - 1):
            token = tail_start + tail_offset
            if token <= position:
                expected[row, tail_start + tail_offset] = token
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_packed_indexer_mixed_short_long_dispatch_matches_sm90():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(79)
    device = 'cuda'
    ratio = 4
    block_topk = 8
    segments = (17, 37)
    total = sum(segments)
    query = torch.randn(total, 4, 16, device=device, dtype=torch.bfloat16)
    total_blocks = sum((n + ratio - 1) // ratio for n in segments)
    block_keys = torch.randn(total_blocks, 16, device=device, dtype=torch.bfloat16)
    block_starts = torch.empty(total, device=device, dtype=torch.int32)
    block_counts = torch.empty(total, device=device, dtype=torch.int32)
    query_positions = torch.empty(total, device=device, dtype=torch.int32)
    offset = 0
    block_start = 0
    for segment_length in segments:
        segment_blocks = (segment_length + ratio - 1) // ratio
        block_starts[offset:offset + segment_length] = block_start
        block_counts[offset:offset + segment_length] = segment_blocks
        query_positions[offset:offset + segment_length] = torch.arange(
            segment_length, device=device, dtype=torch.int32)
        offset += segment_length
        block_start += segment_blocks
    actual = qsa_indexer_fused_topk_packed(
        query,
        block_keys,
        block_starts,
        block_counts,
        query_positions,
        ratio,
        block_topk,
    )
    compact = qsa_indexer_fused_topk_packed(
        query,
        block_keys,
        block_starts,
        block_counts,
        query_positions,
        ratio,
        block_topk,
        return_block_ids=True,
    )
    complete = torch.minimum(
        block_counts.to(torch.long),
        (query_positions.to(torch.long) + 1) // ratio,
    )
    lengths = (
        complete.clamp_max(block_topk) * ratio
        + query_positions.to(torch.long) + 1
        - complete * ratio
    ).to(torch.int32).unsqueeze(0)
    expanded = qsa_expand_block_route(
        compact.unsqueeze(0), lengths, query_positions, ratio)[0]
    assert torch.equal(
        _sorted_valid_route(expanded, lengths[0]),
        _sorted_valid_route(actual, lengths[0]),
    )
    expected = torch.full_like(actual, -1)
    short_length = segments[0]
    for row in range(short_length):
        complete = min(int(block_counts[row]), (row + 1) // ratio)
        expected[row, :complete * ratio] = torch.arange(
            complete * ratio, device=device, dtype=torch.int32)
        tail_start = complete * ratio
        for tail_offset in range(ratio - 1):
            if tail_start + tail_offset <= row:
                expected[row, tail_start + tail_offset] = tail_start + tail_offset
    long_start = short_length
    long_blocks_start = int(block_starts[long_start])
    long_blocks = int(block_counts[long_start])
    expected[long_start:] = qsa_indexer_fused_topk_with_ratio(
        query[long_start:].unsqueeze(0),
        block_keys[long_blocks_start:long_blocks_start + long_blocks].unsqueeze(0),
        query_positions[long_start:],
        ratio,
        block_topk,
    )[0]
    assert torch.equal(actual, expected)

    # Compact production dispatch must not synchronize through max/any or
    # materialize dynamic nonzero token lists.  Mixed short/long rows replay
    # in one fixed-size launch under CUDA Graph capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qsa_indexer_fused_topk_packed(
            query,
            block_keys,
            block_starts,
            block_counts,
            query_positions,
            ratio,
            block_topk,
            return_block_ids=True,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, compact)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_segmented_dkv_matches_atomic_gradients_on_sm90():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(78)
    device = 'cuda'
    seq_len = 24
    ratio = 4
    block_topk = 4
    slots = block_topk * ratio + ratio - 1
    query = torch.randn(seq_len, 1, 4, 8, device=device, dtype=torch.bfloat16)
    key = torch.randn(seq_len, 1, 2, 8, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    indices = torch.full((1, seq_len, slots), -1, device=device, dtype=torch.int32)
    lengths = torch.zeros((1, seq_len), device=device, dtype=torch.int32)
    num_blocks = (seq_len + ratio - 1) // ratio
    for position in range(seq_len):
        complete = min(num_blocks, (position + 1) // ratio)
        selected_blocks = min(complete, block_topk)
        tail_length = position + 1 - complete * ratio
        selected_length = selected_blocks * ratio + tail_length
        if selected_blocks:
            indices[0, position, :selected_blocks * ratio] = torch.arange(
                selected_blocks * ratio, device=device, dtype=torch.int32)
        if tail_length:
            tail_start = selected_blocks * ratio
            indices[0, position, tail_start:tail_start + tail_length] = torch.arange(
                complete * ratio,
                complete * ratio + tail_length,
                device=device,
                dtype=torch.int32,
            )
        lengths[0, position] = selected_length

    def run(reduction):
        q = query.detach().clone().requires_grad_(True)
        k = key.detach().clone().requires_grad_(True)
        v = value.detach().clone().requires_grad_(True)
        output, lse = qsa_sparse_forward(
            q,
            k,
            v,
            indices,
            lengths,
            softmax_scale=8 ** -0.5,
            backend='triton',
            selected_token_group_size=ratio,
            dkv_reduction=reduction,
        )
        output.float().square().mean().backward()
        return output.detach(), lse.detach(), q.grad, k.grad, v.grad

    atomic = run('atomic')
    segmented = run('segmented')
    assert torch.allclose(segmented[0], atomic[0], atol=2e-2, rtol=2e-2)
    assert torch.allclose(segmented[1], atomic[1], atol=2e-2, rtol=2e-2)
    for segmented_grad, atomic_grad in zip(segmented[2:], atomic[2:]):
        assert torch.isfinite(segmented_grad).all()
        assert torch.allclose(segmented_grad, atomic_grad, atol=0.1, rtol=0.1)


def test_selected_kv_tp_head_shards_reconstruct_full_attention():
    torch.manual_seed(81)
    sq, batch, hq, hkv, dim, slots = 7, 1, 4, 2, 4, 5
    query = torch.randn(sq, batch, hq, dim)
    key = torch.randn(sq, batch, hkv, dim)
    value = torch.randn_like(key)
    indices = torch.tensor(
        [[[0, -1, -1, -1, -1], [0, 1, -1, -1, -1], [1, 2, 0, -1, -1],
          [0, 2, 3, 3, -1], [4, 0, 4, 2, -1], [5, 1, 5, 3, -1], [6, 2, 0, 4, 1]]],
        dtype=torch.int32,
    )
    lengths = torch.tensor([[1, 2, 3, 4, 4, 4, 5]], dtype=torch.int32)
    full, full_lse = qsa_sparse_forward(query, key, value, indices, lengths, backend='torch')
    shard_outputs = []
    shard_lse = []
    for tp_rank in range(2):
        head_slice = slice(tp_rank * (hq // 2), (tp_rank + 1) * (hq // 2))
        output, lse = qsa_sparse_forward(
            query[:, :, head_slice], key[:, :, tp_rank:tp_rank + 1], value[:, :, tp_rank:tp_rank + 1],
            indices, lengths, backend='torch')
        shard_outputs.append(output)
        shard_lse.append(lse)
    assert torch.allclose(torch.cat(shard_outputs, dim=2), full, atol=1e-6, rtol=1e-6)
    assert torch.allclose(torch.cat(shard_lse, dim=1), full_lse, atol=1e-6, rtol=1e-6)


def test_selected_kv_backward_reduces_duplicate_kv_accesses():
    torch.manual_seed(11)
    sq, batch, hq, hkv, dim, slots = 5, 2, 4, 2, 3, 5
    query = torch.randn(sq, batch, hq, dim, requires_grad=True)
    key = torch.randn(sq, batch, hkv, dim, requires_grad=True)
    value = torch.randn(sq, batch, hkv, dim, requires_grad=True)
    indices = torch.tensor([
        [[0, -1, -1, -1, -1], [0, 1, -1, -1, -1], [1, 1, 2, -1, -1], [0, 2, 3, 3, -1], [4, 0, 4, 2, -1]],
        [[0, -1, -1, -1, -1], [1, 0, -1, -1, -1], [2, 0, 2, -1, -1], [3, 1, 3, 0, -1], [0, 4, 2, 4, -1]],
    ], dtype=torch.int32)
    lengths = torch.tensor([[1, 2, 3, 4, 4], [1, 2, 3, 4, 4]], dtype=torch.int32)
    scale = dim**-0.5

    out, lse = qsa_sparse_forward(query, key, value, indices, lengths, scale, backend='torch')
    grad_out = torch.randn_like(out)
    grad_lse = torch.randn_like(lse)
    (out * grad_out).sum().backward(retain_graph=True)
    # Run a second graph for the independent autograd reference so the custom
    # Function's gradients are compared with exactly the same upstream values.
    custom_grads = (query.grad.detach().clone(), key.grad.detach().clone(), value.grad.detach().clone())

    query.grad = key.grad = value.grad = None
    reference_out, reference_lse = qsa_sparse_forward_reference(query, key, value, indices, lengths, scale)
    ((reference_out * grad_out).sum() + (reference_lse * grad_lse).sum()).backward()
    reference_grads = (query.grad, key.grad, value.grad)
    # Include LSE in the custom graph in a fresh invocation for a direct check.
    query.grad = key.grad = value.grad = None
    out, lse = qsa_sparse_forward(query, key, value, indices, lengths, scale, backend='torch')
    ((out * grad_out).sum() + (lse * grad_lse).sum()).backward()
    custom_grads_with_lse = (query.grad, key.grad, value.grad)
    for custom, reference, custom_lse in zip(custom_grads_with_lse, reference_grads, custom_grads):
        assert torch.allclose(custom, reference, atol=2e-5, rtol=2e-5)
    # A duplicate index must contribute more than once; this also guards
    # against accidentally replacing index_add_ with assignment.
    assert not torch.equal(custom_grads_with_lse[1], torch.zeros_like(custom_grads_with_lse[1]))


def test_selected_kv_honors_global_query_positions_for_cp_order():
    torch.manual_seed(19)
    query_positions = torch.tensor([0, 4, 1], dtype=torch.int32)
    query = torch.randn(3, 1, 4, 3)
    key = torch.randn(5, 1, 2, 3)
    value = torch.randn(5, 1, 2, 3)
    # The local query order is [global 0, global 4, global 1], as can happen
    # after a CP zigzag split.  Include a future token in the second row to
    # ensure the kernel uses query_positions rather than the local row id.
    indices = torch.tensor([[[0, 1, -1, -1], [4, 3, 0, 2], [1, 0, 4, -1]]], dtype=torch.int32)
    lengths = torch.tensor([[1, 4, 3]], dtype=torch.int32)
    actual, actual_lse = qsa_sparse_forward(
        query, key, value, indices, lengths, backend='torch', query_positions=query_positions)
    expected, expected_lse = qsa_sparse_forward_reference(
        query, key, value, indices, lengths, query_positions=query_positions)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_lse, expected_lse, atol=1e-6, rtol=1e-6)


class _FakeCPGroup:
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def test_cp_owner_plan_remaps_zigzag_selected_tokens_once():
    group = _FakeCPGroup(size=2, rank=0)
    indices = torch.tensor([[[0, 2, 6, 2, -1]]], dtype=torch.int32)
    lengths = torch.tensor([[4]], dtype=torch.int32)
    plan, mapped = _build_owner_plan(indices, lengths, 8, group, 'zigzag')
    # Rank 0 owns global [0,1,6,7] and rank 1 owns [2,3,4,5].  The repeated
    # remote token 2 is requested once and shares one compact cache slot.
    assert mapped.tolist() == [[[0, 4, 2, 4, -1]]]
    assert plan.send_splits == [0, 1]
    assert plan.request_mask.tolist() == [[0, 0, 0, 0], [1, 0, 0, 0]]


def test_cp_owner_plan_consumes_compact_blocks_without_global_token_route():
    group = _FakeCPGroup(size=2, rank=0)
    block_size = 2
    query_positions = torch.tensor([2, 13], dtype=torch.int32)
    blocks = torch.tensor(
        [[[0, -1, -1], [0, 3, 5]]], dtype=torch.int32)
    lengths = torch.tensor([[3, 6]], dtype=torch.int32)

    compact_plan, compact_mapped = _build_owner_plan(
        blocks,
        lengths,
        16,
        group,
        'zigzag',
        route_block_size=block_size,
        query_positions=query_positions,
    )
    expanded = qsa_expand_block_route(
        blocks, lengths, query_positions, block_size)
    token_plan, token_mapped = _build_owner_plan(
        expanded, lengths, 16, group, 'zigzag')

    assert torch.equal(compact_mapped, token_mapped)
    assert compact_mapped.tolist() == [
        [[0, 1, 2, -1, -1, -1, -1],
         [0, 1, 8, 9, 10, 11, -1]]
    ]
    assert compact_plan.send_splits == token_plan.send_splits == [0, 4]
    assert torch.equal(compact_plan.request_mask, token_plan.request_mask)
    assert compact_plan.request_mask.tolist() == [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
    ]


def test_cp1_owner_exchange_preserves_token_output_contract_for_compact_input():
    group = _FakeCPGroup(size=1, rank=0)
    key = torch.randn(8, 1, 2, 4)
    value = torch.randn_like(key)
    positions = torch.tensor([2, 7], dtype=torch.int32)
    blocks = torch.tensor(
        [[[0, -1], [0, 2]]], dtype=torch.int32)
    lengths = torch.tensor([[3, 5]], dtype=torch.int32)

    actual_key, actual_value, actual_route = qsa_exchange_selected_kv(
        key,
        value,
        blocks,
        lengths,
        8,
        group,
        route_block_size=2,
        query_positions=positions,
    )
    expected_route = qsa_expand_block_route(
        blocks, lengths, positions, block_size=2)

    assert actual_key is key
    assert actual_value is value
    assert torch.equal(actual_route, expected_route)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_cp_owner_plan_matches_cpu_compact_zigzag():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    group = _FakeCPGroup(size=2, rank=0)
    query_positions = torch.tensor([2, 13], dtype=torch.int32)
    blocks = torch.tensor(
        [[[0, -1, -1], [0, 3, 5]]], dtype=torch.int32)
    lengths = torch.tensor([[3, 6]], dtype=torch.int32)
    expected_plan, expected_mapped = _build_owner_plan(
        blocks,
        lengths,
        16,
        group,
        'zigzag',
        route_block_size=2,
        query_positions=query_positions,
    )
    actual_plan, actual_mapped = _build_owner_plan(
        blocks.cuda(),
        lengths.cuda(),
        16,
        group,
        'zigzag',
        route_block_size=2,
        query_positions=query_positions.cuda(),
    )
    assert torch.equal(actual_mapped.cpu(), expected_mapped)
    assert torch.equal(
        actual_plan.request_mask.cpu(), expected_plan.request_mask)
    assert actual_plan.send_splits == expected_plan.send_splits


def test_backend_resolution_never_hides_strict_triton_fallback():
    resolved = resolve_qsa_backend('triton', torch.device('cpu'))
    assert resolved.actual == 'torch'
    assert resolved.fallback_reason
    with pytest.raises(RuntimeError, match='qsa_kernel_backend'):
        resolve_qsa_backend('triton', torch.device('cpu'), require=True)


def _indexer_config():
    class Config:
        hidden_size = 8
        indexer_n_heads = 2
        indexer_kv_heads = 1
        indexer_head_dim = 4
        indexer_budget = 8
        indexer_compress_ratio = 2
        layernorm_epsilon = 1e-6
        params_dtype = torch.float32
        sequence_parallel = False
        attention_scaling = 1.0
        qsa_indexer_query_tile_size = 3
        qsa_indexer_key_tile_size = 2

    return Config()


def test_indexer_returns_causal_int32_indices_and_tail_without_s_square_mask():
    config = _indexer_config()
    indexer = QSAIndexer(config)
    hidden = torch.randn(11, 2, config.hidden_size)
    freqs = torch.zeros(11, 1, 1, config.indexer_head_dim)
    indices, lengths = indexer.select_topk(hidden, freqs, backend='torch')
    assert indices.dtype == torch.int32
    assert lengths.dtype == torch.int32
    assert indices.shape == (2, 11, config.indexer_budget + config.indexer_compress_ratio - 1)
    assert lengths.shape == (2, 11)
    for batch in range(indices.shape[0]):
        for query in range(indices.shape[1]):
            length = int(lengths[batch, query])
            row = indices[batch, query]
            assert torch.all(row[:length] >= 0)
            assert torch.all(row[:length] <= query)
            assert torch.all(row[length:] == -1)
    # At query 10 there is a one-token causal tail after five complete blocks.
    assert 10 in indices[0, 10, :int(lengths[0, 10])]


def test_compact_block_route_expands_to_public_token_contract():
    config = _indexer_config()
    indexer = QSAIndexer(config)
    hidden = torch.randn(23, 2, config.hidden_size)
    freqs = torch.zeros(23, 1, 1, config.indexer_head_dim)
    token_indices, token_lengths = indexer.select_topk(
        hidden, freqs, backend='torch')
    block_indices, block_lengths = indexer.select_topk(
        hidden, freqs, backend='torch', return_block_ids=True)
    expanded = qsa_expand_block_route(
        block_indices,
        block_lengths,
        torch.arange(hidden.shape[0]),
        config.indexer_compress_ratio,
    )
    assert block_indices.shape == (2, 23, indexer.block_topk)
    assert torch.equal(block_lengths, token_lengths)
    assert torch.equal(expanded, token_indices)

    query = torch.randn(23, 2, 4, 8)
    key = torch.randn(23, 2, 2, 8)
    value = torch.randn_like(key)
    expected = qsa_sparse_forward(
        query, key, value, token_indices, token_lengths, backend='torch')
    actual = qsa_sparse_forward(
        query,
        key,
        value,
        block_indices,
        block_lengths,
        backend='torch',
        selected_token_group_size=config.indexer_compress_ratio,
        route_block_size=config.indexer_compress_ratio,
    )
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_compact_route_expansion_matches_torch_with_padding():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    device = 'cuda'
    block_size = 4
    positions = torch.arange(7, device=device, dtype=torch.int32)
    lengths = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 0]],
        device=device,
        dtype=torch.int32,
    )
    blocks = torch.tensor(
        [[[2, -1, -1], [1, -1, -1], [0, -1, -1],
          [2, -1, -1], [1, -1, -1], [0, -1, -1], [2, -1, -1]],
         [[1, -1, -1], [2, -1, -1], [0, -1, -1],
          [1, -1, -1], [2, -1, -1], [0, -1, -1], [-1, -1, -1]]],
        device=device,
        dtype=torch.int32,
    )
    expected = qsa_expand_block_route(
        blocks, lengths, positions, block_size)
    actual = qsa_expand_compact_route_triton(
        blocks, lengths, positions, block_size)
    assert torch.equal(actual, expected)


def test_compact_block_route_accepts_segmented_override(monkeypatch):
    query = torch.randn(5, 1, 4, 8)
    key = torch.randn(5, 1, 2, 8)
    value = torch.randn_like(key)
    blocks = torch.tensor([
        [[-1], [-1], [-1], [0], [0]],
    ], dtype=torch.int32)
    lengths = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32)
    monkeypatch.setenv('MCORE_BRIDGE_QSA_DKV_REDUCTION', 'segmented')
    output, lse = qsa_sparse_forward(
        query,
        key,
        value,
        blocks,
        lengths,
        backend='torch',
        selected_token_group_size=4,
        route_block_size=4,
    )
    assert output.shape == query.shape
    assert lse.shape == (1, query.shape[2], query.shape[0])


@pytest.mark.parametrize(
    ('causal', 'key_position_offset'),
    ((False, 0), (True, 4)),
)
def test_compact_block_route_rejects_unreconstructable_tail(
        causal, key_position_offset):
    query = torch.randn(5, 1, 4, 8)
    key = torch.randn(5, 1, 2, 8)
    value = torch.randn_like(key)
    blocks = torch.tensor([
        [[-1], [-1], [-1], [0], [0]],
    ], dtype=torch.int32)
    lengths = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32)
    with pytest.raises(ValueError, match='causal attention.*key_position_offset=0'):
        qsa_sparse_forward(
            query,
            key,
            value,
            blocks,
            lengths,
            backend='torch',
            causal=causal,
            key_position_offset=key_position_offset,
            selected_token_group_size=4,
            route_block_size=4,
        )


def test_indexer_route_lengths_cover_non_aligned_remainders():
    config = _indexer_config()
    indexer = QSAIndexer(config)
    for sequence_length in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65):
        hidden = torch.randn(sequence_length, 2, config.hidden_size)
        freqs = torch.zeros(
            sequence_length, 1, 1, config.indexer_head_dim)
        indices, lengths = indexer.select_topk(hidden, freqs, backend='torch')
        positions = torch.arange(sequence_length)
        complete = (positions + 1) // config.indexer_compress_ratio
        expected = (
            complete.clamp_max(indexer.block_topk)
            * config.indexer_compress_ratio
            + (positions + 1) % config.indexer_compress_ratio
        ).to(torch.int32)
        assert torch.equal(lengths, expected.expand(2, -1))
        for batch in range(2):
            for query in range(sequence_length):
                length = int(lengths[batch, query])
                row = indices[batch, query]
                assert torch.all(row[:length] >= 0)
                assert torch.all(row[:length] <= query)
                assert torch.all(row[length:] == -1)


def test_indexer_tie_break_is_lower_block_id_first():
    config = _indexer_config()
    indexer = QSAIndexer(config)
    with torch.no_grad():
        indexer.index_qk_proj.weight.zero_()
        indexer.q_layernorm.weight.zero_()
        indexer.k_layernorm.weight.zero_()
    hidden = torch.randn(11, 1, config.hidden_size)
    freqs = torch.zeros(11, 1, 1, config.indexer_head_dim)
    indices, lengths = indexer.select_topk(hidden, freqs, backend='torch')
    # Five complete blocks compete for four slots; equal scores use block ids
    # 0,1,2,3, followed by the causal tail token 10.
    assert indices[0, 10, :9].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 10]
    assert int(lengths[0, 10]) == 9


def test_indexer_packed_segments_reset_local_positions():
    config = _indexer_config()
    indexer = QSAIndexer(config)
    hidden = torch.randn(12, 1, config.hidden_size)
    freqs = torch.zeros(12, 1, 1, config.indexer_head_dim)
    indices, lengths = indexer.select_topk_packed(
        hidden, freqs, torch.tensor([0, 5, 12], dtype=torch.int32), backend='torch')
    assert indices.shape == (1, 12, config.indexer_budget + config.indexer_compress_ratio - 1)
    for start, end in ((0, 5), (5, 12)):
        for row in range(start, end):
            length = int(lengths[0, row])
            local_position = row - start
            assert torch.all(indices[0, row, :length] <= local_position)
            assert torch.all(indices[0, row, length:] == -1)


def test_qsa_unpacked_padding_mask_is_right_tail_and_not_loss_mask():
    padding = torch.tensor(
        [[False, False, True, True], [False, True, True, True]], dtype=torch.bool)
    resolved = Qwen4ExpLayer._qsa_resolve_right_padding_mask(
        {'padding_mask': padding}, batch=2, sequence_length=4)
    assert torch.equal(resolved, padding)

    attention_mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
    attention_mask[0, :, :, 2:] = True
    attention_mask[1, :, :, 1:] = True
    resolved_from_attention = Qwen4ExpLayer._qsa_resolve_right_padding_mask(
        {'attention_mask': attention_mask}, batch=2, sequence_length=4)
    assert torch.equal(resolved_from_attention, padding)


def test_qsa_packed_padding_mask_preserves_document_boundaries():
    packed = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 5, 12, 16], dtype=torch.int32),
        seq_lens=torch.tensor([3, 7], dtype=torch.int32),
    )
    resolved = Qwen4ExpLayer._qsa_resolve_packed_padding_mask(
        packed, total_tokens=16, device=torch.device('cpu'))
    expected = torch.tensor(
        [False, False, False, True, True,
         False, False, False, False, False, False, False,
         True, True, True, True], dtype=torch.bool)
    assert torch.equal(resolved, expected)


def test_qsa_selection_zeroes_packed_alignment_tail_without_labels():
    class FakeIndexer:
        @staticmethod
        def _result(hidden_states):
            total = hidden_states.shape[0]
            indices = torch.arange(5, dtype=torch.int32).view(1, 1, 5)
            indices = indices.expand(1, total, -1).clone()
            lengths = torch.full((1, total), 5, dtype=torch.int32)
            return indices, lengths

        @torch.no_grad()
        def select_topk(self, hidden_states, freqs, **kwargs):
            return self._result(hidden_states)

        @torch.no_grad()
        def select_topk_packed(self, hidden_states, freqs, cu_seqlens, **kwargs):
            return self._result(hidden_states)

    layer = object.__new__(Qwen4ExpLayer)
    layer.self_attention = SimpleNamespace(indexer=FakeIndexer())
    layer.config = SimpleNamespace(
        qsa_kernel_backend='torch', csa_dense_mode=False,
        require_qsa_kernel=False, qsa_dense_fallback_max_seq_len=4096,
        tensor_model_parallel_size=1, sequence_parallel=False,
        hidden_size=4, context_parallel_size=1, qsa_cp_mode='disabled',
        cp_partition_mode='zigzag', qsa_indexer_query_tile_size=8,
        qsa_indexer_key_tile_size=8, qsa_dkv_reduction='atomic')
    packed = SimpleNamespace(
        qkv_format='thd', pad_between_seqs=False,
        cu_seqlens_q=torch.tensor([0, 5, 12, 16], dtype=torch.int32),
        seq_lens=torch.tensor([3, 7], dtype=torch.int32))
    hidden = torch.randn(16, 1, 4)
    selected = layer._qsa_select_topk(
        hidden,
        {'packed_seq_params': packed, 'rotary_pos_emb': torch.zeros(16, 1, 1, 1),
         'attention_mask': None, 'inference_context': None,
         'attention_bias': None})
    indices, lengths = selected[:2]
    assert torch.equal(lengths[0, 3:5], torch.zeros(2, dtype=torch.int32))
    assert torch.equal(lengths[0, 12:], torch.zeros(4, dtype=torch.int32))
    assert torch.equal(indices[0, 3:5], torch.full((2, 5), -1, dtype=torch.int32))
    assert torch.equal(indices[0, 12:], torch.full((4, 5), -1, dtype=torch.int32))


@pytest.mark.parametrize('reduction', ('atomic', 'segmented'))
def test_qwen4_exp_selection_uses_compact_route_for_triton_reduction(
        monkeypatch, reduction):
    class FakeIndexer:
        compress_ratio = 4
        return_block_ids = None

        @torch.no_grad()
        def select_topk(self, hidden_states, freqs, **kwargs):
            self.return_block_ids = kwargs['return_block_ids']
            slots = 2 if self.return_block_ids else 11
            indices = torch.zeros(
                (1, hidden_states.shape[0], slots), dtype=torch.int32)
            lengths = torch.ones(
                (1, hidden_states.shape[0]), dtype=torch.int32)
            return indices, lengths

    monkeypatch.setattr(
        'mcore_bridge.model.gpts.qwen4_exp.resolve_qsa_backend',
        lambda requested, device, require: SimpleNamespace(
            requested=requested,
            actual='triton',
            fallback_reason=None,
        ),
    )
    indexer = FakeIndexer()
    layer = object.__new__(Qwen4ExpLayer)
    layer.self_attention = SimpleNamespace(indexer=indexer)
    layer.config = SimpleNamespace(
        qsa_kernel_backend='triton', csa_dense_mode=False,
        require_qsa_kernel=True, qsa_dense_fallback_max_seq_len=4096,
        tensor_model_parallel_size=1, sequence_parallel=False,
        hidden_size=4, context_parallel_size=1, qsa_cp_mode='disabled',
        cp_partition_mode='zigzag', qsa_indexer_query_tile_size=8,
        qsa_indexer_key_tile_size=8, qsa_dkv_reduction=reduction,
        qsa_compact_block_route=True)
    hidden = torch.randn(16, 1, 4)
    selected = layer._qsa_select_topk(
        hidden,
        {'packed_seq_params': None, 'rotary_pos_emb': torch.zeros(16, 1, 1, 1),
         'attention_mask': None, 'inference_context': None,
         'attention_bias': None})
    assert indexer.return_block_ids is True
    assert selected[0].shape == (1, 16, 2)
    assert selected[-1] == 4


def test_selected_kv_empty_route_rows_are_zero_and_do_not_backpropagate():
    torch.manual_seed(241)
    query = torch.randn(5, 2, 4, 3, requires_grad=True)
    key = torch.randn(5, 2, 2, 3, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    indices = torch.full((2, 5, 5), -1, dtype=torch.int32)
    lengths = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]], dtype=torch.int32)
    indices[0, 0, 0] = 0
    indices[0, 1, :2] = torch.tensor([0, 1])
    indices[0, 2, :3] = torch.tensor([0, 1, 2])
    indices[1, 0, 0] = 0
    indices[1, 1, :2] = torch.tensor([0, 1])
    output, _ = qsa_sparse_forward(
        query, key, value, indices, lengths, backend='torch')
    assert torch.equal(output[0, 3:], torch.zeros_like(output[0, 3:]))
    assert torch.equal(output[1, 2:], torch.zeros_like(output[1, 2:]))
    output.square().sum().backward()
    assert torch.equal(query.grad[0, 3:], torch.zeros_like(query.grad[0, 3:]))
    assert torch.equal(query.grad[1, 2:], torch.zeros_like(query.grad[1, 2:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_selected_kv_matches_torch_on_sm90():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(17)
    device = 'cuda'
    query = torch.randn(9, 1, 4, 16, device=device, dtype=torch.bfloat16)
    key = torch.randn(9, 1, 2, 16, device=device, dtype=torch.bfloat16)
    value = torch.randn(9, 1, 2, 16, device=device, dtype=torch.bfloat16)
    indices = torch.tensor([[list(range(i + 1)) + [-1] * (9 - i - 1) for i in range(9)]], device=device,
                           dtype=torch.int32)
    lengths = torch.arange(1, 10, device=device, dtype=torch.int32)[None]
    torch_out, torch_lse = qsa_sparse_forward(query, key, value, indices, lengths, backend='torch')
    triton_out, triton_lse = qsa_sparse_forward(query, key, value, indices, lengths, backend='triton')
    assert torch.allclose(triton_out.float(), torch_out.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(triton_lse, torch_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_selected_kv_right_padding_rows_are_zero_on_sm90():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(238)
    device = 'cuda'
    query = torch.randn(9, 2, 4, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(9, 2, 2, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    indices = torch.full((2, 9, 9), -1, device=device, dtype=torch.int32)
    lengths = torch.zeros((2, 9), device=device, dtype=torch.int32)
    for batch, valid_rows in enumerate((9, 5)):
        for query_position in range(valid_rows):
            count = query_position + 1
            indices[batch, query_position, :count] = torch.arange(
                count, device=device, dtype=torch.int32)
            lengths[batch, query_position] = count
    output, lse = qsa_sparse_forward(
        query, key, value, indices, lengths, backend='triton', require_backend=True)
    assert torch.equal(output[:, 5:, :, :][1], torch.zeros_like(output[1, 5:]))
    assert torch.equal(lse[1, :, 5:], torch.full_like(lse[1, :, 5:], -float('inf')))
    (output.float().square().sum()).backward()
    assert torch.equal(query.grad[1, 5:], torch.zeros_like(query.grad[1, 5:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_fused_indexer_postprocess_is_bitwise_exact_on_sm90(monkeypatch):
    """Keep the default Hopper fusion identical to the torch BF16 contract."""

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(72)
    config = _indexer_config()
    config.hidden_size = 80
    config.params_dtype = torch.bfloat16
    config.indexer_n_heads = 4
    config.indexer_kv_heads = 1
    config.indexer_head_dim = 128
    config.indexer_budget = 2048
    config.indexer_compress_ratio = 4
    indexer = QSAIndexer(config).cuda()
    indexer.q_layernorm.weight.data.uniform_(-0.3, 0.3)
    indexer.k_layernorm.weight.data.uniform_(-0.3, 0.3)
    hidden = torch.randn(
        67, 2, config.hidden_size, device='cuda', dtype=torch.bfloat16)
    freqs = torch.randn(
        67, 1, 1, config.indexer_head_dim,
        device='cuda', dtype=torch.bfloat16)

    monkeypatch.setenv('MCORE_BRIDGE_QSA_INDEXER_FUSED_POSTPROCESS', '0')
    expected_q, expected_keys = indexer._project_and_pool(hidden, freqs)
    monkeypatch.delenv(
        'MCORE_BRIDGE_QSA_INDEXER_FUSED_POSTPROCESS', raising=False)
    actual_q, actual_keys = indexer._project_and_pool(hidden, freqs)

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_keys, expected_keys)


@pytest.mark.parametrize('freq_layout', ('max_length', 'flattened'))
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_fused_packed_indexer_postprocess_is_bitwise_exact_on_sm90(
        monkeypatch, freq_layout):
    """Packed fusion must preserve segment-local RoPE and block boundaries."""

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(74)
    config = _indexer_config()
    config.hidden_size = 80
    config.params_dtype = torch.bfloat16
    config.indexer_n_heads = 4
    config.indexer_kv_heads = 1
    config.indexer_head_dim = 128
    config.indexer_budget = 2048
    config.indexer_compress_ratio = 4
    indexer = QSAIndexer(config).cuda()
    indexer.q_layernorm.weight.data.uniform_(-0.3, 0.3)
    indexer.k_layernorm.weight.data.uniform_(-0.3, 0.3)
    hidden = torch.randn(
        16, 1, config.hidden_size, device='cuda', dtype=torch.bfloat16)
    freq_length = 9 if freq_layout == 'max_length' else hidden.shape[0]
    freqs = torch.randn(
        freq_length, 1, 1, config.indexer_head_dim,
        device='cuda', dtype=torch.bfloat16)
    # Include an empty document and tails that must not be pooled into the
    # following document.  Both non-empty segments reset RoPE position to 0.
    boundaries = [0, 7, 7, 16]

    monkeypatch.setenv('MCORE_BRIDGE_QSA_INDEXER_FUSED_POSTPROCESS', '0')
    monkeypatch.setenv('MCORE_BRIDGE_QSA_PACKED_METADATA_FUSED', '0')
    expected = indexer._project_and_pool_packed(
        hidden, freqs, boundaries)
    monkeypatch.delenv(
        'MCORE_BRIDGE_QSA_INDEXER_FUSED_POSTPROCESS', raising=False)
    monkeypatch.delenv(
        'MCORE_BRIDGE_QSA_PACKED_METADATA_FUSED', raising=False)
    actual = indexer._project_and_pool_packed(hidden, freqs, boundaries)

    assert all(torch.equal(actual_tensor, expected_tensor)
               for actual_tensor, expected_tensor in zip(actual, expected))


@pytest.mark.parametrize('freq_layout', ('max_length', 'flattened'))
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_packed_device_cu_matches_host_path_and_captures_on_sm90(
        monkeypatch, freq_layout):
    """Guard static slab holes, long Top-K, and full selector graph replay."""

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(76)
    config = _indexer_config()
    config.hidden_size = 80
    config.params_dtype = torch.bfloat16
    config.indexer_n_heads = 4
    config.indexer_kv_heads = 1
    config.indexer_head_dim = 128
    config.indexer_budget = 2048
    config.indexer_compress_ratio = 4
    indexer = QSAIndexer(config).cuda()
    indexer.q_layernorm.weight.data.uniform_(-0.3, 0.3)
    indexer.k_layernorm.weight.data.uniform_(-0.3, 0.3)
    # Two three-token documents accumulate enough discarded tail to make the
    # static block slab start the long document at slot one.  The old compact
    # host path starts it at slot zero, so route parity exercises hole mapping.
    boundaries = (0, 3, 6, 6, 2319)
    cu = torch.tensor(boundaries, device='cuda', dtype=torch.int32)
    hidden = torch.randn(
        boundaries[-1], 1, config.hidden_size,
        device='cuda', dtype=torch.bfloat16)
    max_segment_length = max(end - start for start, end in zip(
        boundaries[:-1], boundaries[1:]))
    freq_length = (
        max_segment_length
        if freq_layout == 'max_length' else boundaries[-1]
    )
    freqs = torch.randn(
        freq_length,
        1, 1, config.indexer_head_dim,
        device='cuda', dtype=torch.bfloat16)

    monkeypatch.setenv('MCORE_BRIDGE_QSA_PACKED_METADATA_FUSED', '0')
    expected_indices, expected_lengths = indexer.select_topk_packed(
        hidden, freqs, cu, backend='triton', return_block_ids=True)
    monkeypatch.delenv(
        'MCORE_BRIDGE_QSA_PACKED_METADATA_FUSED', raising=False)
    actual_indices, actual_lengths = indexer.select_topk_packed(
        hidden, freqs, cu, backend='triton', return_block_ids=True)

    assert torch.equal(actual_lengths, expected_lengths)
    assert int(actual_lengths.max()) <= indexer.selected_k
    assert torch.equal(
        torch.sort(actual_indices, dim=-1).values,
        torch.sort(expected_indices, dim=-1).values,
    )

    if freq_layout == 'max_length':
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_indices, captured_lengths = indexer.select_topk_packed(
                hidden, freqs, cu, backend='triton', return_block_ids=True)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured_lengths, actual_lengths)
        assert torch.equal(
            torch.sort(captured_indices, dim=-1).values,
            torch.sort(actual_indices, dim=-1).values,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_fused_indexer_topk_matches_torch_sets_on_sm90():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(73)
    config = _indexer_config()
    config.params_dtype = torch.bfloat16
    config.indexer_n_heads = 2
    config.indexer_head_dim = 16
    config.indexer_budget = 16
    config.indexer_compress_ratio = 2
    indexer = QSAIndexer(config).cuda()
    q = torch.randn(2, 64, 2, 16, device='cuda', dtype=torch.bfloat16)
    block_keys = torch.randn(2, 64, 16, device='cuda', dtype=torch.bfloat16)
    positions = torch.arange(64, device='cuda', dtype=torch.long)
    expected, expected_lengths = indexer._select_from_projected(
        q, block_keys, positions, backend='torch', query_tile_size=8, key_tile_size=16)
    actual = qsa_indexer_fused_topk_with_ratio(q, block_keys, positions, 2, 8)
    compact = qsa_indexer_fused_topk_with_ratio(
        q, block_keys, positions, 2, 8, return_block_ids=True)
    assert compact.shape == (2, 64, 8)
    expanded = qsa_expand_block_route(compact, expected_lengths, positions, 2)
    assert torch.equal(
        _sorted_valid_route(expanded, expected_lengths),
        _sorted_valid_route(actual, expected_lengths),
    )
    assert torch.equal(expected_lengths, (
        ((positions + 1) // 2).clamp_max(8) * 2 + (positions + 1) % 2
    ).to(torch.int32).expand(2, -1))
    for batch in range(2):
        for query in range(64):
            actual_row = actual[batch, query]
            expected_row = expected[batch, query]
            assert torch.equal(
                actual_row[actual_row >= 0].sort().values,
                expected_row[expected_row >= 0].sort().values,
            )

    # The compact route can retain the fused kernel's bitonic permutation.
    # Exercise that representation through attention so set equivalence is
    # also guarded at the operator boundary, not only in index metadata.
    attention_q = torch.randn(
        64, 2, 4, 16, device='cuda', dtype=torch.bfloat16)
    attention_k = torch.randn(
        64, 2, 2, 16, device='cuda', dtype=torch.bfloat16)
    attention_v = torch.randn_like(attention_k)
    token_output, token_lse = qsa_sparse_forward(
        attention_q,
        attention_k,
        attention_v,
        actual,
        expected_lengths,
        backend='triton',
        require_backend=True,
        query_positions=positions,
    )
    compact_output, compact_lse = qsa_sparse_forward(
        attention_q,
        attention_k,
        attention_v,
        compact,
        expected_lengths,
        backend='triton',
        require_backend=True,
        query_positions=positions,
        selected_token_group_size=2,
        route_block_size=2,
    )
    assert torch.allclose(
        compact_output.float(), token_output.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(compact_lse, token_lse, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_fused_indexer_large_budget_prefix_on_sm90():
    """Cover the production K=512 stream and its causal-prefix rows."""

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(75)
    device = 'cuda'
    seq_len, num_blocks, block_topk, ratio = 2048, 1024, 512, 2
    q = torch.randn(2, seq_len, 4, 16, device=device, dtype=torch.bfloat16)
    block_keys = torch.randn(2, num_blocks, 16, device=device, dtype=torch.bfloat16)
    positions = torch.arange(seq_len, device=device, dtype=torch.long)
    actual = qsa_indexer_fused_topk_with_ratio(
        q, block_keys, positions, ratio, block_topk)
    scores = torch.einsum('bshd,bnd->bshn', q.float(), block_keys.float())
    scores = scores.relu().sum(2) / (16 ** 0.5)
    block_ids = torch.arange(num_blocks, device=device).view(1, 1, -1)
    scores = scores.masked_fill(
        block_ids >= ((positions + 1) // ratio).view(1, seq_len, 1), -float('inf'))
    expected = torch.topk(scores, k=block_topk, dim=-1, sorted=False).indices
    for batch in range(q.shape[0]):
        for row in (3, 4, 7, 511, 512, 1023, 2047):
            visible = min((row + 1) // ratio, block_topk)
            got = (actual[batch, row, :visible * ratio:ratio] // ratio).sort().values
            ref = expected[batch, row, :visible].sort().values
            assert torch.equal(got, ref)
            assert torch.equal(
                actual[batch, row, visible * ratio:visible * ratio +
                       (row + 1 - ((row + 1) // ratio) * ratio)],
                torch.arange((row + 1) // ratio * ratio, row + 1,
                             device=device, dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_indexer_score_slab_matches_streaming_sets_on_sm90():
    """Guard the production query-tiled score/Top-K route and tie fallback."""

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(241)
    device = 'cuda'
    seq_len, ratio, block_topk = 4096, 4, 512
    batch = 2
    q = torch.randn(
        batch, seq_len, 4, 128, device=device, dtype=torch.bfloat16)
    block_keys = torch.randn(
        batch, seq_len // ratio, 128, device=device, dtype=torch.bfloat16)
    positions = torch.arange(seq_len, device=device, dtype=torch.int32)
    expected = qsa_indexer_fused_topk_with_ratio(
        q, block_keys, positions, ratio, block_topk, return_block_ids=True)
    actual = qsa_indexer_slab_topk_with_ratio(
        q, block_keys, positions, ratio, block_topk)
    assert torch.equal(
        torch.sort(actual, dim=-1).values,
        torch.sort(expected, dim=-1).values,
    )

    # The production path must not synchronize through ``.item()`` or build a
    # host-side ambiguous-row list: both would make capture fail.  Replaying
    # the captured score/Top-K/tie-mask sequence must retain the exact set.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qsa_indexer_slab_topk_with_ratio(
            q, block_keys, positions, ratio, block_topk,
            validate_positions=False,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(
        torch.sort(captured, dim=-1).values,
        torch.sort(expected, dim=-1).values,
    )

    # Equal scores cross the K/K+1 boundary on every saturated row.  Those
    # rows must take the deterministic packed-key fallback and keep the lower
    # block IDs, rather than inheriting CUDA Top-K's unspecified tie order.
    tied = qsa_indexer_slab_topk_with_ratio(
        torch.zeros_like(q),
        torch.zeros_like(block_keys),
        positions,
        ratio,
        block_topk,
    )
    expected_tie_ids = torch.arange(
        block_topk, device=device, dtype=torch.int32).expand(batch, -1)
    assert torch.equal(torch.sort(tied[:, -1]).values, expected_tie_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_fused_indexer_split_partial_matches_torch_on_sm90(monkeypatch):
    """Keep the opt-in multi-partial route exact after tile tuning."""

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(239)
    device = 'cuda'
    ratio, block_topk, seq_len = 2, 8, 80
    q = torch.randn(1, seq_len, 2, 16, device=device, dtype=torch.bfloat16)
    block_keys = torch.randn(1, seq_len // ratio, 16, device=device, dtype=torch.bfloat16)
    positions = torch.arange(seq_len, device=device, dtype=torch.long)
    config = _indexer_config()
    config.indexer_head_dim = 16
    config.indexer_budget = block_topk * ratio
    config.indexer_compress_ratio = ratio
    indexer = QSAIndexer(config)
    expected, expected_lengths = indexer._select_from_projected(
        q, block_keys, positions, backend='torch', query_tile_size=16, key_tile_size=16)

    # 1 MiB is intentionally large enough for the bounded partial metadata,
    # but forces four partials for this 40-block/8-block-topk fixture.
    monkeypatch.setenv('MCORE_BRIDGE_QSA_INDEXER_MAX_PARTIAL_MB', '1')
    monkeypatch.setenv('MCORE_BRIDGE_QSA_INDEXER_SPLIT_EXPANSION', '0')
    monkeypatch.delenv('MCORE_BRIDGE_QSA_INDEXER_STREAM_BLOCK_N', raising=False)
    actual = qsa_indexer_fused_topk_with_ratio(
        q, block_keys, positions, ratio, block_topk, max_partial_bytes=1 << 20)
    assert actual.shape == expected.shape
    assert torch.equal(expected_lengths, (
        ((positions + 1) // ratio).clamp_max(block_topk) * ratio
        + (positions + 1) % ratio
    ).to(torch.int32).view(1, -1))
    for row in range(seq_len):
        got = actual[0, row]
        ref = expected[0, row]
        assert torch.equal(got[got >= 0].sort().values, ref[ref >= 0].sort().values)
        tail_start = min((row + 1) // ratio, block_topk) * ratio
        tail_len = (row + 1) - ((row + 1) // ratio) * ratio
        assert torch.equal(got[tail_start:tail_start + tail_len],
                           ref[tail_start:tail_start + tail_len])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_segmented_dkv_matches_relaxed_atomic_on_sm90(monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    # Exercise the actual SM90 D=256 default independently from the grouped
    # fallback test above; inherited tuning variables must not select it.
    for name in (
        'MCORE_BRIDGE_QSA_SEGMENT_FLATTEN_HEADS',
        'MCORE_BRIDGE_QSA_SEGMENT_HEAD_TILE',
        'MCORE_BRIDGE_QSA_SEGMENT_WARPS',
    ):
        monkeypatch.delenv(name, raising=False)
    torch.manual_seed(74)
    # group_size=3 also exercises the masked final two-head tile used by
    # narrow TP-local shapes.
    sq, batch, hq, hkv, dim, ratio, block_topk = 64, 2, 6, 2, 256, 4, 4
    slots = block_topk * ratio + ratio - 1
    positions = torch.arange(sq, device='cuda', dtype=torch.int32)
    indices = torch.full((batch, sq, slots), -1, device='cuda', dtype=torch.int32)
    lengths = torch.zeros((batch, sq), device='cuda', dtype=torch.int32)
    for query in range(sq):
        complete = (query + 1) // ratio
        selected_blocks = min(complete, block_topk)
        tail = (query + 1) - complete * ratio
        values = sum((list(range(block * ratio, block * ratio + ratio))
                      for block in range(selected_blocks)), [])
        values.extend(range(complete * ratio, complete * ratio + tail))
        if values:
            selected = torch.tensor(
                values, device='cuda', dtype=torch.int32)
            indices[:, query, :len(values)] = selected
        lengths[:, query] = len(values)
    q0 = torch.randn(sq, batch, hq, dim, device='cuda', dtype=torch.bfloat16)
    k0 = torch.randn(sq, batch, hkv, dim, device='cuda', dtype=torch.bfloat16)
    v0 = torch.randn_like(k0)
    grad_out = torch.randn_like(q0)

    def run(reduction):
        q = q0.detach().clone().requires_grad_()
        k = k0.detach().clone().requires_grad_()
        v = v0.detach().clone().requires_grad_()
        out, lse = qsa_sparse_forward(
            q, k, v, indices, lengths, backend='triton', query_positions=positions,
            selected_token_group_size=ratio, dkv_reduction=reduction)
        (out.float() * grad_out.float()).sum().backward()
        return out.detach(), lse.detach(), q.grad, k.grad, v.grad

    atomic = run('atomic')
    segmented = run('segmented')
    assert torch.equal(segmented[0], atomic[0])
    assert torch.equal(segmented[1], atomic[1])
    assert torch.equal(segmented[2], atomic[2])
    assert torch.allclose(segmented[3].float(), atomic[3].float(), atol=0.125, rtol=0.05)
    assert torch.allclose(segmented[4].float(), atomic[4].float(), atol=0.125, rtol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_hybrid_owner_mask_preserves_compact_gradients_on_sm90(monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(2026)
    device = 'cuda'
    sq, batch, hq, hkv, dim, ratio, block_topk = 64, 1, 24, 2, 256, 4, 4
    positions = torch.arange(sq, device=device, dtype=torch.int32)
    indices = torch.full(
        (batch, sq, block_topk), -1, device=device, dtype=torch.int32)
    lengths = torch.zeros((batch, sq), device=device, dtype=torch.int32)
    for query in range(sq):
        complete = min((query + 1) // ratio, block_topk)
        tail = (query + 1) - ((query + 1) // ratio) * ratio
        if complete:
            block_ids = torch.arange(
                complete, device=device, dtype=torch.int32)
            if query % 2:
                block_ids = block_ids.flip(0)
            if complete >= 3 and query % 3 == 0:
                block_ids[-1] = 0
            indices[0, query, :complete] = block_ids
        lengths[0, query] = complete * ratio + tail
    q0 = torch.randn(sq, batch, hq, dim, device=device, dtype=torch.bfloat16)
    k0 = torch.randn(sq, batch, hkv, dim, device=device, dtype=torch.bfloat16)
    v0 = torch.randn_like(k0)
    grad_out = torch.randn_like(q0)
    grad_lse = torch.randn(batch, hq, sq, device=device, dtype=torch.float32) * 0.01

    def run(threshold=None, reuse=False, compact=False, fuse=False, tiled=False,
            listed=False, persistent=False, grouped_blocks=None,
            grouped_union=False, table_scan=False, table_recompute=False,
            split_recompute=False):
        monkeypatch.setenv('MCORE_BRIDGE_QSA_DKV_REDUCTION', 'segmented')
        monkeypatch.delenv('MCORE_BRIDGE_QSA_SEGMENT_FUSE_HEAD_TILES', raising=False)
        monkeypatch.delenv('MCORE_BRIDGE_QSA_SEGMENT_FUSE_HEAD_TILES_TILED', raising=False)
        monkeypatch.delenv('MCORE_BRIDGE_QSA_SEGMENT_COMPACT_DERIVATIVES', raising=False)
        monkeypatch.delenv('MCORE_BRIDGE_QSA_SEGMENT_COMPACT_BLOCK_LIST', raising=False)
        monkeypatch.delenv('MCORE_BRIDGE_QSA_SEGMENT_PERSISTENT_OWNER', raising=False)
        monkeypatch.delenv('MCORE_BRIDGE_QSA_SEGMENT_PERSISTENT_CTAS', raising=False)
        monkeypatch.delenv('MCORE_BRIDGE_QSA_SEGMENT_GROUP_BLOCKS', raising=False)
        monkeypatch.delenv('MCORE_BRIDGE_QSA_SEGMENT_GROUP_UNION', raising=False)
        monkeypatch.delenv('MCORE_BRIDGE_QSA_SEGMENT_TABLE_SCAN', raising=False)
        monkeypatch.delenv(
            'MCORE_BRIDGE_QSA_SEGMENT_TABLE_RECOMPUTE_DERIVATIVES',
            raising=False)
        monkeypatch.delenv(
            'MCORE_BRIDGE_QSA_SEGMENT_SPLIT_RECOMPUTE_DKV', raising=False)
        monkeypatch.delenv(
            'MCORE_BRIDGE_QSA_SEGMENT_COSELECT_ORDER', raising=False)
        monkeypatch.delenv('MCORE_BRIDGE_QSA_SEGMENT_FLATTEN_HEADS', raising=False)
        if fuse:
            monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_FUSE_HEAD_TILES', '1')
        if tiled:
            monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_FUSE_HEAD_TILES_TILED', '1')
        if compact:
            monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_COMPACT_DERIVATIVES', '1')
        if listed:
            monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_COMPACT_BLOCK_LIST', '1')
        if persistent:
            monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_PERSISTENT_OWNER', '1')
            monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_PERSISTENT_CTAS', '16')
        if grouped_blocks is not None:
            monkeypatch.setenv(
                'MCORE_BRIDGE_QSA_SEGMENT_GROUP_BLOCKS', str(grouped_blocks))
            monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_FLATTEN_HEADS', '1')
        if grouped_union:
            monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_GROUP_UNION', '1')
        if table_scan:
            monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_TABLE_SCAN', '1')
        if table_recompute:
            monkeypatch.setenv(
                'MCORE_BRIDGE_QSA_SEGMENT_TABLE_RECOMPUTE_DERIVATIVES', '1')
        if split_recompute:
            monkeypatch.setenv(
                'MCORE_BRIDGE_QSA_SEGMENT_SPLIT_RECOMPUTE_DKV', '1')
        if threshold is None:
            monkeypatch.delenv('MCORE_BRIDGE_QSA_SEGMENT_HYBRID_MIN_FANOUT', raising=False)
        else:
            monkeypatch.setenv(
                'MCORE_BRIDGE_QSA_SEGMENT_HYBRID_MIN_FANOUT', str(threshold))
        if reuse:
            monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_REUSE_DERIVATIVES', '1')
            monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_SCORE_DTYPE', 'bf16')
            monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_D_SCORE_DTYPE', 'bf16')
        else:
            monkeypatch.delenv('MCORE_BRIDGE_QSA_SEGMENT_REUSE_DERIVATIVES', raising=False)
            monkeypatch.delenv('MCORE_BRIDGE_QSA_SEGMENT_SCORE_DTYPE', raising=False)
            monkeypatch.delenv('MCORE_BRIDGE_QSA_SEGMENT_D_SCORE_DTYPE', raising=False)
        q = q0.detach().clone().requires_grad_()
        k = k0.detach().clone().requires_grad_()
        v = v0.detach().clone().requires_grad_()
        output, lse = qsa_sparse_forward(
            q, k, v, indices, lengths, backend='triton',
            query_positions=positions, selected_token_group_size=ratio,
            dkv_reduction='segmented', route_block_size=ratio)
        ((output.float() * grad_out.float()).sum()
         + (lse * grad_lse).sum()).backward()
        result = (
            output.detach(), lse.detach(), q.grad.detach(),
            k.grad.detach(), v.grad.detach())
        del q, k, v, output, lse
        return result

    reference = run()
    hybrid = run(threshold=2)
    hybrid_saved = run(threshold=2, reuse=True)
    fused_owner = run(threshold=2, reuse=True, compact=True, fuse=True)
    tiled_owner = run(threshold=2, reuse=True, compact=True, fuse=True, tiled=True)
    listed_owner = run(threshold=2, reuse=True, compact=True, listed=True)
    persistent_owner = run(threshold=2, reuse=True, compact=True, persistent=True)
    grouped_owner2 = run(threshold=2, reuse=True, compact=True, grouped_blocks=2)
    grouped_owner4 = run(threshold=2, reuse=True, compact=True, grouped_blocks=4)
    grouped_union_duplicate = run(
        threshold=2, reuse=True, compact=True, grouped_blocks=4,
        grouped_union=True)
    table_scan_duplicate = run(
        threshold=2, reuse=True, compact=True, grouped_blocks=4,
        table_scan=True)
    for actual in (hybrid, hybrid_saved, fused_owner, tiled_owner, listed_owner,
                   persistent_owner, grouped_owner2, grouped_owner4,
                   grouped_union_duplicate, table_scan_duplicate):
        assert torch.equal(actual[0], reference[0])
        assert torch.equal(actual[1], reference[1])
        assert torch.equal(actual[2], reference[2])
        for actual_grad, reference_grad in zip(actual[3:], reference[3:]):
            assert torch.isfinite(actual_grad).all()
            assert torch.allclose(
                actual_grad.float(), reference_grad.float(), atol=0.125, rtol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_split_dkv_backward_matches_fused_on_sm90(monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(2027)
    device = 'cuda'
    sq, batch, hq, hkv, dim, ratio, block_topk = 64, 1, 24, 2, 256, 4, 4
    positions = torch.arange(sq, device=device, dtype=torch.int32)
    indices = torch.full(
        (batch, sq, block_topk), -1, device=device, dtype=torch.int32)
    lengths = torch.zeros((batch, sq), device=device, dtype=torch.int32)
    for query in range(sq):
        complete = min((query + 1) // ratio, block_topk)
        tail = (query + 1) - ((query + 1) // ratio) * ratio
        if complete:
            indices[0, query, :complete] = torch.arange(
                complete, device=device, dtype=torch.int32)
        lengths[0, query] = complete * ratio + tail
    q0 = torch.randn(sq, batch, hq, dim, device=device, dtype=torch.bfloat16)
    k0 = torch.randn(sq, batch, hkv, dim, device=device, dtype=torch.bfloat16)
    v0 = torch.randn_like(k0)
    grad_out = torch.randn_like(q0)
    grad_lse = torch.randn(batch, hq, sq, device=device, dtype=torch.float32) * 0.01

    def run(split):
        monkeypatch.delenv('MCORE_BRIDGE_QSA_DKV_REDUCTION', raising=False)
        monkeypatch.delenv('MCORE_BRIDGE_QSA_BACKWARD_SPLIT_DKV', raising=False)
        if split:
            monkeypatch.setenv('MCORE_BRIDGE_QSA_BACKWARD_SPLIT_DKV', '1')
        q = q0.detach().clone().requires_grad_()
        k = k0.detach().clone().requires_grad_()
        v = v0.detach().clone().requires_grad_()
        output, lse = qsa_sparse_forward(
            q, k, v, indices, lengths, backend='triton',
            query_positions=positions, selected_token_group_size=ratio,
            route_block_size=ratio)
        ((output.float() * grad_out.float()).sum()
         + (lse * grad_lse).sum()).backward()
        result = (q.grad.detach(), k.grad.detach(), v.grad.detach())
        del q, k, v, output, lse
        return result

    fused = run(False)
    split = run(True)
    assert torch.equal(fused[0], split[0])
    for split_grad, fused_grad in zip(split[1:], fused[1:]):
        assert torch.isfinite(split_grad).all()
        assert torch.allclose(
            split_grad.float(), fused_grad.float(), atol=0.5, rtol=0.05)


@pytest.mark.parametrize('output_delta', ('1', '0'))
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_selected_kv_backward_matches_torch_on_sm90(monkeypatch, output_delta):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    monkeypatch.setenv('MCORE_BRIDGE_QSA_BACKWARD_OUTPUT_DELTA', output_delta)
    torch.manual_seed(18)
    device = 'cuda'
    sq, hq, hkv, dim, slots = 32, 4, 2, 16, 9
    query_base = torch.randn(sq, 1, hq, dim, device=device, dtype=torch.bfloat16)
    key_base = torch.randn(sq, 1, hkv, dim, device=device, dtype=torch.bfloat16)
    value_base = torch.randn_like(key_base)
    indices = torch.randint(0, sq, (1, sq, slots), device=device, dtype=torch.int32)
    positions = torch.arange(sq, device=device, dtype=torch.int32)
    indices = torch.minimum(indices, positions.view(1, sq, 1))
    lengths = torch.randint(1, slots + 1, (1, sq), device=device, dtype=torch.int32)
    grad_out = torch.randn(sq, 1, hq, dim, device=device, dtype=torch.bfloat16)

    def run(backend):
        query = query_base.detach().clone().requires_grad_()
        key = key_base.detach().clone().requires_grad_()
        value = value_base.detach().clone().requires_grad_()
        output, lse = qsa_sparse_forward(
            query, key, value, indices, lengths, backend=backend, query_positions=positions)
        (output * grad_out).sum().backward()
        return output.detach(), lse.detach(), query.grad, key.grad, value.grad

    torch_result = run('torch')
    triton_result = run('triton')
    assert torch.allclose(triton_result[0].float(), torch_result[0].float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(triton_result[1], torch_result[1], atol=2e-2, rtol=2e-2)
    for triton_grad, torch_grad in zip(triton_result[2:], torch_result[2:]):
        assert torch.allclose(triton_grad.float(), torch_grad.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_selected_kv_production_compact_default_dispatch_matches_torch(monkeypatch):
    """Guard the SM90 D=256/K=2051 compact two-warp dispatch."""

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    for name in (
        'MCORE_BRIDGE_QSA_FORWARD_HEAD_TILE',
        'MCORE_BRIDGE_QSA_FORWARD_BLOCK_K',
        'MCORE_BRIDGE_QSA_FORWARD_WARPS',
        'MCORE_BRIDGE_QSA_BACKWARD_HEAD_TILE',
        'MCORE_BRIDGE_QSA_BACKWARD_BLOCK_K',
        'MCORE_BRIDGE_QSA_BACKWARD_WARPS',
    ):
        monkeypatch.delenv(name, raising=False)

    torch.manual_seed(314159)
    sq, hq, hkv, dim, block_slots = 32, 24, 2, 256, 512
    device = 'cuda'
    query_base = torch.randn(sq, 1, hq, dim, device=device, dtype=torch.bfloat16)
    key_base = torch.randn(sq, 1, hkv, dim, device=device, dtype=torch.bfloat16)
    value_base = torch.randn_like(key_base)
    positions = torch.arange(sq, device=device, dtype=torch.int32)
    indices = torch.full(
        (1, sq, block_slots), -1, device=device, dtype=torch.int32)
    lengths = torch.empty((1, sq), device=device, dtype=torch.int32)
    for position in range(sq):
        # Vary complete-block counts and retain duplicate block IDs to exercise
        # atomic accumulation without reducing compile-time logical K=2051.
        complete_blocks = (position + 1) // 4
        block_count = (
            (position * 17) % complete_blocks + 1
            if complete_blocks else 0
        )
        if block_count:
            indices[0, position, :block_count] = torch.randint(
                0,
                complete_blocks,
                (block_count,),
                device=device,
                dtype=torch.int32,
            )
        lengths[0, position] = block_count * 4 + (position + 1) % 4
    grad_out = torch.randn_like(query_base)

    def run(backend):
        query = query_base.detach().clone().requires_grad_()
        key = key_base.detach().clone().requires_grad_()
        value = value_base.detach().clone().requires_grad_()
        output, lse = qsa_sparse_forward(
            query,
            key,
            value,
            indices,
            lengths,
            backend=backend,
            require_backend=backend == 'triton',
            query_positions=positions,
            selected_token_group_size=4,
            route_block_size=4,
        )
        (output.float() * grad_out.float()).sum().backward()
        return output.detach(), lse.detach(), query.grad, key.grad, value.grad

    reference = run('torch')
    actual = run('triton')
    assert torch.allclose(actual[0].float(), reference[0].float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(actual[1], reference[1], atol=2e-5, rtol=2e-5)
    assert torch.allclose(actual[2].float(), reference[2].float(), atol=5e-2, rtol=5e-2)
    for actual_grad, reference_grad in zip(actual[3:], reference[3:]):
        difference = actual_grad.float() - reference_grad.float()
        assert difference.abs().mean().item() < 2e-2
        assert difference.norm().item() / reference_grad.float().norm().item() < 2e-2
        assert difference.abs().max().item() <= 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize(
    'dkv_accum_dtype,save_forward_scores,save_score_dtype',
    (('bf16', False, 'fp32'), ('fp32', False, 'fp32'),
     ('bf16', True, 'bf16'), ('bf16', True, 'fp32')),
)
def test_triton_compact_block_route_forward_backward_matches_token_route(
        monkeypatch, dkv_accum_dtype, save_forward_scores, save_score_dtype):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    monkeypatch.setenv(
        'MCORE_BRIDGE_QSA_BACKWARD_TENSORIZE_FP32_ATOMIC',
        '1' if dkv_accum_dtype == 'fp32' else '0')
    monkeypatch.setenv(
        'MCORE_BRIDGE_QSA_SAVE_FORWARD_SCORES',
        '1' if save_forward_scores else '0')
    monkeypatch.setenv(
        'MCORE_BRIDGE_QSA_SAVE_FORWARD_SCORE_DTYPE', save_score_dtype)
    torch.manual_seed(2718)
    device = 'cuda'
    sq, hq, hkv, dim, ratio, block_topk = 35, 12, 2, 64, 4, 8
    positions = torch.arange(sq, device=device, dtype=torch.int32)
    blocks = torch.full(
        (1, sq, block_topk), -1, device=device, dtype=torch.int32)
    lengths = torch.zeros((1, sq), device=device, dtype=torch.int32)
    for position in range(sq):
        complete = (position + 1) // ratio
        selected_count = min(complete, block_topk)
        if selected_count:
            blocks[0, position, :selected_count] = torch.arange(
                selected_count - 1, -1, -1,
                device=device,
                dtype=torch.int32,
            )
        lengths[0, position] = (
            selected_count * ratio + (position + 1 - complete * ratio))
    tokens = qsa_expand_block_route(blocks, lengths, positions, ratio)
    assert blocks.shape[-1] * ratio + ratio - 1 == tokens.shape[-1]
    assert blocks.numel() < tokens.numel()

    query_base = torch.randn(
        sq, 1, hq, dim, device=device, dtype=torch.bfloat16)
    key_base = torch.randn(
        sq, 1, hkv, dim, device=device, dtype=torch.bfloat16)
    value_base = torch.randn_like(key_base)
    grad_output = torch.randn_like(query_base)
    grad_lse = torch.randn(
        1, hq, sq, device=device, dtype=torch.float32) * 0.01

    def run(route, backend, block_size):
        query = query_base.detach().clone().requires_grad_()
        key = key_base.detach().clone().requires_grad_()
        value = value_base.detach().clone().requires_grad_()
        output, lse = qsa_sparse_forward(
            query,
            key,
            value,
            route,
            lengths,
            backend=backend,
            require_backend=backend == 'triton',
            query_positions=positions,
            dkv_accum_dtype=dkv_accum_dtype,
            selected_token_group_size=ratio,
            route_block_size=block_size,
        )
        ((output.float() * grad_output.float()).sum()
         + (lse * grad_lse).sum()).backward()
        return output.detach(), lse.detach(), query.grad, key.grad, value.grad

    reference = run(tokens, 'torch', 1)
    actual = run(blocks, 'triton', ratio)
    assert torch.allclose(actual[0].float(), reference[0].float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(actual[1], reference[1], atol=2e-5, rtol=2e-5)
    assert torch.allclose(actual[2].float(), reference[2].float(), atol=5e-2, rtol=5e-2)
    grad_atol = 1.25e-1 if dkv_accum_dtype == 'fp32' else 2e-2
    grad_rtol = 1e-1 if dkv_accum_dtype == 'fp32' else 2e-2
    for actual_grad, reference_grad in zip(actual[3:], reference[3:]):
        difference = actual_grad.float() - reference_grad.float()
        assert difference.abs().mean().item() < grad_atol
        assert difference.norm().item() / reference_grad.float().norm().item() < grad_rtol


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_packed_compact_block_route_matches_token_route():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(3141)
    device = 'cuda'
    segments = (19, 13)
    total = sum(segments)
    hq, hkv, dim, ratio, block_topk = 12, 2, 64, 4, 4
    cu = torch.tensor(
        [0, segments[0], total], device=device, dtype=torch.int32)
    local_positions = torch.cat(tuple(
        torch.arange(length, device=device, dtype=torch.int32)
        for length in segments
    ))
    blocks = torch.full(
        (1, total, block_topk), -1, device=device, dtype=torch.int32)
    lengths = torch.zeros((1, total), device=device, dtype=torch.int32)
    for token, position in enumerate(local_positions.tolist()):
        complete = (position + 1) // ratio
        selected_count = min(complete, block_topk)
        if selected_count:
            blocks[0, token, :selected_count] = torch.arange(
                selected_count, device=device, dtype=torch.int32)
        lengths[0, token] = (
            selected_count * ratio + (position + 1 - complete * ratio))
    tokens = qsa_expand_block_route(
        blocks, lengths, local_positions, ratio)

    query = torch.randn(
        total, hq, dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(
        total, hkv, dim, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    reference = qsa_sparse_forward_packed(
        query, key, value, tokens, lengths, cu, cu, backend='torch')
    actual = qsa_sparse_forward_packed(
        query,
        key,
        value,
        blocks,
        lengths,
        cu,
        cu,
        backend='triton',
        require_backend=True,
        selected_token_group_size=ratio,
        route_block_size=ratio,
    )
    assert torch.allclose(actual[0].float(), reference[0].float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(actual[1], reference[1], atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize(
    'transpose_dkv,group_union,table_scan,table_scan_full,table_recompute,split_recompute,coselect_order',
    (
        (False, False, False, False, False, False, False),
        (True, False, False, False, False, False, False),
        (False, True, False, False, False, False, False),
        (True, True, False, False, False, False, False),
        (False, False, True, False, False, False, False),
        (False, False, True, True, False, False, False),
        (False, False, True, False, True, False, False),
        (False, False, True, False, True, True, False),
        (False, False, True, False, False, False, True),
        (False, False, True, False, True, False, True),
    ),
)
def test_triton_packed_compact_block_owned_backward_matches_token_route_on_sm90(
        monkeypatch, transpose_dkv, group_union, table_scan, table_scan_full,
        table_recompute, split_recompute, coselect_order):
    """Exercise packed block ownership on aligned multi-segment routes."""

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    monkeypatch.setenv('MCORE_BRIDGE_QSA_DKV_REDUCTION', 'segmented')
    monkeypatch.setenv(
        'MCORE_BRIDGE_QSA_SEGMENT_REUSE_DERIVATIVES',
        '0' if table_recompute else '1')
    monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_SCORE_DTYPE', 'bf16')
    monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_D_SCORE_DTYPE', 'bf16')
    monkeypatch.setenv(
        'MCORE_BRIDGE_QSA_SEGMENT_TRANSPOSE_DKV',
        '1' if transpose_dkv else '0')
    monkeypatch.setenv(
        'MCORE_BRIDGE_QSA_SEGMENT_GROUP_BLOCKS',
        '4' if (group_union or table_scan) else '1')
    monkeypatch.setenv(
        'MCORE_BRIDGE_QSA_SEGMENT_GROUP_UNION',
        '1' if group_union else '0')
    monkeypatch.setenv(
        'MCORE_BRIDGE_QSA_SEGMENT_HYBRID_MIN_FANOUT',
        '0' if table_scan_full else '1' if (group_union or table_scan) else '0')
    monkeypatch.setenv('MCORE_BRIDGE_QSA_SEGMENT_FLATTEN_HEADS', '1')
    monkeypatch.setenv(
        'MCORE_BRIDGE_QSA_SEGMENT_COMPACT_DERIVATIVES',
        '1' if (group_union or table_scan) and not table_recompute else '0')
    monkeypatch.setenv(
        'MCORE_BRIDGE_QSA_SEGMENT_TABLE_SCAN',
        '1' if table_scan else '0')
    monkeypatch.setenv(
        'MCORE_BRIDGE_QSA_SEGMENT_TABLE_RECOMPUTE_DERIVATIVES',
        '1' if table_recompute else '0')
    monkeypatch.setenv(
        'MCORE_BRIDGE_QSA_SEGMENT_SPLIT_RECOMPUTE_DKV',
        '1' if split_recompute else '0')
    monkeypatch.setenv(
        'MCORE_BRIDGE_QSA_SEGMENT_COSELECT_ORDER',
        '1' if coselect_order else '0')
    torch.manual_seed(3142)
    device = 'cuda'
    segments = (20, 16)
    total = sum(segments)
    hq, hkv, dim, ratio, block_topk = 12, 2, 64, 4, 4
    cu = torch.tensor(
        [0, segments[0], total], device=device, dtype=torch.int32)
    local_positions = torch.cat(tuple(
        torch.arange(length, device=device, dtype=torch.int32)
        for length in segments
    ))
    blocks = torch.full(
        (1, total, block_topk), -1, device=device, dtype=torch.int32)
    lengths = torch.zeros((1, total), device=device, dtype=torch.int32)
    for token, position in enumerate(local_positions.tolist()):
        complete = (position + 1) // ratio
        selected_count = min(complete, block_topk)
        if selected_count:
            blocks[0, token, :selected_count] = torch.arange(
                selected_count - 1, -1, -1, device=device, dtype=torch.int32)
        lengths[0, token] = (
            selected_count * ratio + (position + 1 - complete * ratio))
    tokens = qsa_expand_block_route(
        blocks, lengths, local_positions, ratio)

    query_base = torch.randn(
        total, hq, dim, device=device, dtype=torch.bfloat16)
    key_base = torch.randn(
        total, hkv, dim, device=device, dtype=torch.bfloat16)
    value_base = torch.randn_like(key_base)
    grad_output = torch.randn_like(query_base)
    grad_lse = torch.randn(
        1, hq, total, device=device, dtype=torch.float32) * 0.01

    def run(route, backend, route_size):
        query = query_base.detach().clone().requires_grad_()
        key = key_base.detach().clone().requires_grad_()
        value = value_base.detach().clone().requires_grad_()
        output, lse = qsa_sparse_forward_packed(
            query,
            key,
            value,
            route,
            lengths,
            cu,
            cu,
            backend=backend,
            require_backend=backend == 'triton',
            selected_token_group_size=ratio,
            dkv_reduction='segmented',
            route_block_size=route_size,
        )
        ((output.float() * grad_output.float()).sum()
         + (lse * grad_lse).sum()).backward()
        return output.detach(), lse.detach(), query.grad, key.grad, value.grad

    reference = run(tokens, 'torch', 1)
    actual = run(blocks, 'triton', ratio)
    assert torch.allclose(actual[0].float(), reference[0].float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(actual[1], reference[1], atol=2e-2, rtol=2e-2)
    assert torch.allclose(actual[2].float(), reference[2].float(), atol=0.1, rtol=0.1)
    for actual_grad, reference_grad in zip(actual[3:], reference[3:]):
        difference = actual_grad.float() - reference_grad.float()
        assert difference.abs().mean().item() < 2e-2
        assert difference.norm().item() / reference_grad.float().norm().item() < 3e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_packed_segment_dispatch_matches_single_grid_on_sm90(monkeypatch):
    """Guard device-side short/long routing in a large mixed THD pack."""

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(1618)
    device = 'cuda'
    # logical K=19 gives the auto-trim boundary 8*K=152.  The total pack and
    # second document exceed it, while the first document stays below it, so
    # both filtered launches own real rows.
    segments = (31, 160)
    total = sum(segments)
    hq, hkv, dim, ratio, block_topk = 12, 2, 64, 4, 4
    cu = torch.tensor(
        [0, segments[0], total], device=device, dtype=torch.int32)
    local_positions = torch.cat(tuple(
        torch.arange(length, device=device, dtype=torch.int32)
        for length in segments
    ))
    blocks = torch.full(
        (1, total, block_topk), -1, device=device, dtype=torch.int32)
    lengths = torch.zeros((1, total), device=device, dtype=torch.int32)
    for token, position in enumerate(local_positions.tolist()):
        complete = (position + 1) // ratio
        selected_count = min(complete, block_topk)
        if selected_count:
            blocks[0, token, :selected_count] = torch.arange(
                selected_count, device=device, dtype=torch.int32)
        lengths[0, token] = (
            selected_count * ratio + (position + 1 - complete * ratio))

    query_base = torch.randn(
        total, hq, dim, device=device, dtype=torch.bfloat16)
    key_base = torch.randn(
        total, hkv, dim, device=device, dtype=torch.bfloat16)
    value_base = torch.randn_like(key_base)
    grad_output = torch.randn_like(query_base)
    grad_lse = torch.randn(
        1, hq, total, device=device, dtype=torch.float32) * 0.01

    def run(enabled):
        monkeypatch.setenv(
            'MCORE_BRIDGE_QSA_PACKED_SEGMENT_DISPATCH', str(int(enabled)))
        query = query_base.detach().clone().requires_grad_()
        key = key_base.detach().clone().requires_grad_()
        value = value_base.detach().clone().requires_grad_()
        output, lse = qsa_sparse_forward_packed(
            query,
            key,
            value,
            blocks,
            lengths,
            cu,
            cu,
            backend='triton',
            require_backend=True,
            selected_token_group_size=ratio,
            route_block_size=ratio,
        )
        ((output.float() * grad_output.float()).sum()
         + (lse * grad_lse).sum()).backward()
        return output.detach(), lse.detach(), query.grad, key.grad, value.grad

    single_grid = run(False)
    dispatched = run(True)
    assert torch.equal(dispatched[0], single_grid[0])
    assert torch.equal(dispatched[1], single_grid[1])
    assert torch.equal(dispatched[2], single_grid[2])
    for actual_grad, reference_grad in zip(dispatched[3:], single_grid[3:]):
        difference = actual_grad.float() - reference_grad.float()
        assert difference.abs().mean().item() < 2e-2
        assert difference.norm().item() / reference_grad.float().norm().item() < 2e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_triton_packed_training_is_cuda_graph_capturable_on_sm90():
    """Prevent host metadata synchronization from returning to packed THD."""

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires H100/SM90')
    torch.manual_seed(20260902)
    device = 'cuda'
    segments = (31, 160)
    total = sum(segments)
    hq, hkv, dim, slots = 4, 2, 16, 19
    query = torch.randn(
        total, hq, dim, device=device, dtype=torch.bfloat16,
        requires_grad=True)
    key = torch.randn(
        total, hkv, dim, device=device, dtype=torch.bfloat16,
        requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    grad_output = torch.randn_like(query)
    grad_lse = torch.randn(
        1, hq, total, device=device, dtype=torch.float32) * 0.01
    cu = torch.tensor(
        [0, segments[0], total], device=device, dtype=torch.int32)
    indices = torch.full(
        (1, total, slots), -1, device=device, dtype=torch.int32)
    lengths = torch.zeros((1, total), device=device, dtype=torch.int32)
    for start, segment_length in (
            (0, segments[0]), (segments[0], segments[1])):
        for position in range(segment_length):
            count = min(position + 1, slots)
            indices[0, start + position, :count] = torch.arange(
                count, device=device, dtype=torch.int32)
            lengths[0, start + position] = count

    def step():
        output, lse = qsa_sparse_forward_packed(
            query,
            key,
            value,
            indices,
            lengths,
            cu,
            cu,
            backend='triton',
            require_backend=True,
        )
        loss = ((output.float() * grad_output.float()).sum()
                + (lse * grad_lse).sum())
        loss.backward()
        return output, lse, loss

    current_stream = torch.cuda.current_stream()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream):
        for _ in range(2):
            query.grad = key.grad = value.grad = None
            eager_output, eager_lse, eager_loss = step()
    current_stream.wait_stream(warmup_stream)
    torch.cuda.synchronize()
    eager_grads = tuple(
        tensor.grad.detach().clone() for tensor in (query, key, value))
    del eager_output, eager_lse, eager_loss
    query.grad.zero_()
    key.grad.zero_()
    value.grad.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        query.grad.zero_()
        key.grad.zero_()
        value.grad.zero_()
        graph_output, graph_lse, graph_loss = step()
    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(graph_loss)
    assert torch.isfinite(graph_output).all()
    assert torch.isfinite(graph_lse).all()
    graph_grads = (query.grad, key.grad, value.grad)
    assert torch.equal(graph_grads[0], eager_grads[0])
    for actual, expected in zip(graph_grads[1:], eager_grads[1:]):
        difference = actual.float() - expected.float()
        assert difference.abs().mean().item() < 2e-2
        assert difference.norm().item() / expected.float().norm().item() < 2e-2
