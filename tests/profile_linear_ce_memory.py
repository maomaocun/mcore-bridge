import importlib.util
import os
from pathlib import Path

os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce


_FUSED_LINEAR_CE_PATH = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/fused_linear_ce.py'
_FUSED_LINEAR_CE_SPEC = importlib.util.spec_from_file_location('mcore_bridge_fused_linear_ce', _FUSED_LINEAR_CE_PATH)
_FUSED_LINEAR_CE = importlib.util.module_from_spec(_FUSED_LINEAR_CE_SPEC)
assert _FUSED_LINEAR_CE_SPEC.loader is not None
_FUSED_LINEAR_CE_SPEC.loader.exec_module(_FUSED_LINEAR_CE)
VocabParallelStreamingFusedLinearCrossEntropy = _FUSED_LINEAR_CE.VocabParallelStreamingFusedLinearCrossEntropy


def _init_distributed():
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    return local_rank, dist.get_rank(), dist.get_world_size()


def _make_labels(seq_len, batch_size, vocab_size, ignore_tokens):
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
    if ignore_tokens > 0:
        labels[:, :ignore_tokens] = -100
    return labels


def _print_peak(case, loss):
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    stats = torch.tensor([allocated, reserved, float(loss.detach())], dtype=torch.float64, device='cuda')
    gathered = [torch.empty_like(stats) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, stats)
    if dist.get_rank() == 0:
        max_allocated = max(t[0].item() for t in gathered)
        max_reserved = max(t[1].item() for t in gathered)
        losses = [round(t[2].item(), 6) for t in gathered]
        print(
            f'{case}: max_allocated_gib={max_allocated:.4f} '
            f'max_reserved_gib={max_reserved:.4f} losses={losses}',
            flush=True,
        )


def main():
    _, rank, world_size = _init_distributed()
    if parallel_state._GLOBAL_MEMORY_BUFFER is None:
        parallel_state._set_global_memory_buffer()
    case = os.environ.get('LINEAR_CE_PROFILE_CASE', 'native_mcore').strip().lower()
    seq_len = int(os.environ.get('PROFILE_SEQ_LEN', '8192'))
    batch_size = int(os.environ.get('PROFILE_BATCH_SIZE', '1'))
    hidden_size = int(os.environ.get('PROFILE_HIDDEN_SIZE', '5120'))
    vocab_size = int(os.environ.get('PROFILE_VOCAB_SIZE', '248320'))
    chunk_size = int(os.environ.get('PROFILE_CHUNK_SIZE', '2048'))
    ignore_tokens = int(os.environ.get('PROFILE_IGNORE_TOKENS', str(seq_len // 4)))
    dtype = torch.bfloat16 if os.environ.get('PROFILE_DTYPE', 'bf16').lower() == 'bf16' else torch.float16

    if vocab_size % world_size != 0:
        raise AssertionError(f'vocab_size={vocab_size} must be divisible by world_size={world_size}')
    if seq_len % world_size != 0:
        raise AssertionError(f'seq_len={seq_len} must be divisible by world_size={world_size}')

    torch.manual_seed(20260611 + rank)
    local_seq_len = seq_len // world_size
    vocab_partition = vocab_size // world_size
    vocab_start = rank * vocab_partition
    hidden = (torch.randn(local_seq_len, batch_size, hidden_size, device='cuda', dtype=dtype) * 0.01)
    hidden.requires_grad_(True)
    weight = (torch.randn(vocab_partition, hidden_size, device='cuda', dtype=dtype) * 0.01)
    weight.requires_grad_(True)
    labels = _make_labels(seq_len, batch_size, vocab_size, ignore_tokens)
    supervised = (labels != -100).sum().float()

    # Keep setup allocations out of the measured peak.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if case == 'native_mcore':
        logits = linear_with_grad_accumulation_and_async_allreduce(
            hidden,
            weight,
            None,
            gradient_accumulation_fusion=False,
            allreduce_dgrad=False,
            sequence_parallel=True,
            grad_output_buffer=None,
            wgrad_deferral_limit=None,
            tp_group=dist.group.WORLD,
        )
        losses = fused_vocab_parallel_cross_entropy(
            logits, labels.transpose(0, 1).contiguous(), dist.group.WORLD)
    elif case == 'streaming':
        losses = VocabParallelStreamingFusedLinearCrossEntropy.apply(
            hidden, weight, labels, dist.group.WORLD, vocab_start, chunk_size)
    else:
        raise ValueError(f'Unknown LINEAR_CE_PROFILE_CASE={case!r}')

    loss = losses.sum() / supervised.clamp_min(1.0)
    loss.backward()
    _print_peak(case, loss)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
