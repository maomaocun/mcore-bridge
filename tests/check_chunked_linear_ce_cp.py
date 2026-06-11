import os

os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mcore_bridge.model.gpt_model import _ChunkedLinearCrossEntropy


def _init_distributed():
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    return local_rank, dist.get_rank(), dist.get_world_size()


def _make_groups(rank, world_size):
    tp_size = int(os.environ.get('TP_SIZE', '2'))
    if world_size % tp_size != 0:
        raise AssertionError(f'world_size={world_size} must be divisible by TP_SIZE={tp_size}')
    cp_size = world_size // tp_size
    if cp_size < 2:
        raise AssertionError('check_chunked_linear_ce_cp.py requires CP size >= 2')

    tp_group = None
    cp_group = None
    for cp_rank in range(cp_size):
        ranks = [cp_rank * tp_size + tp_rank for tp_rank in range(tp_size)]
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            tp_group = group
    for tp_rank in range(tp_size):
        ranks = [cp_rank * tp_size + tp_rank for cp_rank in range(cp_size)]
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            cp_group = group

    return tp_size, cp_size, rank % tp_size, rank // tp_size, tp_group, cp_group


def _split_cp_reference(inputs: torch.Tensor, cp_size: int, cp_rank: int, dim: int) -> torch.Tensor:
    if dim < 0:
        dim = (dim + inputs.ndim) % inputs.ndim
    inputs = inputs.view(
        *inputs.shape[:dim],
        2 * cp_size,
        inputs.shape[dim] // (2 * cp_size),
        *inputs.shape[dim + 1:],
    )
    index = torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], device=inputs.device)
    return inputs.index_select(dim, index).reshape(*inputs.shape[:dim], -1, *inputs.shape[dim + 2:])


def _assert_close(name, actual, expected, atol=2e-5, rtol=2e-5):
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max()
    denom = expected.float().abs().clamp_min(1e-6)
    max_rel = (diff / denom).max()
    stats = torch.stack([max_abs, max_rel])
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(f'{name}: max_abs={stats[0].item():.8f}, max_rel={stats[1].item():.8f}', flush=True)
    if stats[0].item() > atol and stats[1].item() > rtol:
        raise AssertionError(
            f'{name} mismatch: max_abs={stats[0].item():.8f}, max_rel={stats[1].item():.8f}, '
            f'atol={atol}, rtol={rtol}')


def _make_case(seq_len, batch, hidden_size, vocab_size):
    torch.manual_seed(20260611)
    hidden = (torch.randn(seq_len, batch, hidden_size, device='cuda') * 0.07).requires_grad_(True)
    weight = (torch.randn(vocab_size, hidden_size, device='cuda') * 0.05).requires_grad_(True)
    labels = torch.randint(0, vocab_size, (batch, seq_len), device='cuda')

    seq_idx = torch.arange(seq_len, device='cuda').unsqueeze(0)
    batch_idx = torch.arange(batch, device='cuda').unsqueeze(1)
    labels = labels.masked_fill((seq_idx + 2 * batch_idx) % 5 == 0, -100)
    return hidden, weight, labels


def _reference_losses(hidden, weight, labels):
    seq_len, batch, _ = hidden.shape
    logits = torch.matmul(hidden, weight.t()).float()
    losses = F.cross_entropy(
        logits.view(seq_len * batch, weight.shape[0]),
        labels.transpose(0, 1).contiguous().view(-1),
        reduction='none',
        ignore_index=-100,
    )
    return losses.view(seq_len, batch).transpose(0, 1).contiguous()


def _run_case(name, tp_rank, cp_rank, tp_size, cp_size, tp_group, cp_group, reduce_grad_input):
    if dist.get_rank() == 0:
        print(f'checking {name}', flush=True)

    seq_len = 24
    batch = 2
    hidden_size = 8
    vocab_size = 24
    chunk_size = 3
    hidden, weight, labels = _make_case(seq_len, batch, hidden_size, vocab_size)

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    ref_losses = _reference_losses(ref_hidden, ref_weight, labels)

    torch.manual_seed(1234)
    grad_out = torch.randn(batch, seq_len, device='cuda') * 0.13
    grad_out = grad_out.masked_fill(labels == -100, 0.0)
    (ref_losses * grad_out).sum().backward()

    local_hidden = _split_cp_reference(hidden.detach(), cp_size, cp_rank, dim=0).contiguous().requires_grad_(True)
    local_labels = _split_cp_reference(labels, cp_size, cp_rank, dim=1).contiguous()
    local_grad_out = _split_cp_reference(grad_out, cp_size, cp_rank, dim=1).contiguous()
    local_ref_losses = _split_cp_reference(ref_losses.detach(), cp_size, cp_rank, dim=1).contiguous()
    local_ref_hidden_grad = _split_cp_reference(ref_hidden.grad.detach(), cp_size, cp_rank, dim=0).contiguous()

    partition_vocab_size = vocab_size // tp_size
    vocab_start = tp_rank * partition_vocab_size
    vocab_end = vocab_start + partition_vocab_size
    local_weight = weight.detach()[vocab_start:vocab_end].contiguous().requires_grad_(True)

    losses = _ChunkedLinearCrossEntropy.apply(
        local_hidden,
        local_weight,
        local_labels,
        tp_group,
        vocab_start,
        chunk_size,
        reduce_grad_input,
    )
    _assert_close(f'{name} loss', losses, local_ref_losses)

    (losses * local_grad_out).sum().backward()

    hidden_grad = local_hidden.grad.detach().clone()
    if not reduce_grad_input:
        dist.all_reduce(hidden_grad, op=dist.ReduceOp.SUM, group=tp_group)
    _assert_close(f'{name} hidden grad', hidden_grad, local_ref_hidden_grad)

    weight_grad = local_weight.grad.detach().clone()
    dist.all_reduce(weight_grad, op=dist.ReduceOp.SUM, group=cp_group)
    _assert_close(f'{name} weight grad', weight_grad, ref_weight.grad[vocab_start:vocab_end])


def main():
    _, rank, world_size = _init_distributed()
    tp_size, cp_size, tp_rank, cp_rank, tp_group, cp_group = _make_groups(rank, world_size)
    if tp_size < 2:
        raise AssertionError('check_chunked_linear_ce_cp.py requires TP size >= 2')

    _run_case(
        'CP local CE with internal TP grad reduce',
        tp_rank,
        cp_rank,
        tp_size,
        cp_size,
        tp_group,
        cp_group,
        reduce_grad_input=True,
    )
    _run_case(
        'CP local CE with external sequence-parallel grad reduce',
        tp_rank,
        cp_rank,
        tp_size,
        cp_size,
        tp_group,
        cp_group,
        reduce_grad_input=False,
    )

    if rank == 0:
        print(f'chunked linear CE CP checks OK with tp_size={tp_size}, cp_size={cp_size}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
