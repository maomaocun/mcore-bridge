import os
import importlib.util
from pathlib import Path

import torch
import torch.distributed as dist


def _init_distributed():
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    print(f'[local_rank={local_rank}] set cuda device', flush=True)
    torch.cuda.set_device(local_rank)
    print(f'[local_rank={local_rank}] init process group', flush=True)
    dist.init_process_group('nccl')
    print(f'[rank={dist.get_rank()}] process group ready', flush=True)
    return local_rank, dist.get_rank(), dist.get_world_size()


def _load_gdn_helpers():
    module_path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/modules/gated_delta_net.py'
    print(f'[pid={os.getpid()}] loading helpers from {module_path}', flush=True)
    spec = importlib.util.spec_from_file_location('mcore_bridge_gdn_for_a2a_check', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f'[pid={os.getpid()}] helpers loaded', flush=True)
    return (
        module._build_head_perm_for_split_sections,
        module.tensor_a2a_cp2hp,
        module.tensor_a2a_hp2cp,
    )


def _split_cp_inputs_reference(inputs: torch.Tensor, cp_size: int, cp_rank: int, dim: int) -> torch.Tensor:
    view_shape = (*inputs.shape[:dim], 2 * cp_size, inputs.shape[dim] // (2 * cp_size), *inputs.shape[dim + 1:])
    inputs = inputs.view(view_shape)
    index = torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], device=inputs.device)
    return inputs.index_select(dim, index).reshape(*inputs.shape[:dim], -1, *inputs.shape[dim + 2:])


def main():
    _, rank, world_size = _init_distributed()
    _build_head_perm_for_split_sections, tensor_a2a_cp2hp, tensor_a2a_hp2cp = _load_gdn_helpers()
    if world_size < 2:
        raise AssertionError('check_gdn_cp_a2a.py requires at least 2 ranks.')

    cp_group = dist.group.WORLD
    cp_size = world_size
    seq_len = 8 * cp_size
    batch = 2
    q = 4 * cp_size
    k = 4 * cp_size
    v = 12 * cp_size
    z = 12 * cp_size
    beta = 3 * cp_size
    alpha = 3 * cp_size
    full_hidden = q + k + v + z + beta + alpha

    full = torch.arange(seq_len * batch * full_hidden, device='cuda', dtype=torch.float32).view(
        seq_len, batch, full_hidden)
    local_cp = _split_cp_inputs_reference(full, cp_size, rank, dim=0).contiguous()

    head_perm = _build_head_perm_for_split_sections((q, k, v, z, beta, alpha), cp_size, torch.device('cuda'))
    local_cp = local_cp.index_select(-1, head_perm)
    hp = tensor_a2a_cp2hp(local_cp, seq_dim=0, head_dim=-1, cp_group=cp_group)
    roundtrip = tensor_a2a_hp2cp(hp, seq_dim=0, head_dim=-1, cp_group=cp_group)

    if not torch.equal(roundtrip, local_cp):
        diff = (roundtrip - local_cp).abs().max().item()
        raise AssertionError(f'GDN CP A2A roundtrip mismatch on rank {rank}: max_diff={diff}')

    if rank == 0:
        print(f'GDN CP A2A roundtrip OK with cp_size={cp_size}, shape={tuple(local_cp.shape)}')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
