# Portions adapted and modified for STOMP from DINO (Apache-2.0).
# Copyright (c) Facebook, Inc. and its affiliates.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

import os
import pickle
import torch
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def reduce(tensor, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
    return dist.reduce(tensor, dst, op, group, async_op)

def reduce_mean_scalar(val):
    if isinstance(val, torch.Tensor):
        t = val.detach().clone()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t = torch.tensor(val, dtype=torch.float32, device=device)

    if is_dist_avail_and_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= get_world_size()

    return t.item()

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if get_world_size() <= 1:
        return
    dist.barrier()

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def gather_varlen_tensors(x_loc: torch.Tensor):
    """
    Gather tensors with variable length along dim 0 from all distributed ranks.

    This helper first gathers the per-rank shapes, allocates correctly sized
    output tensors (preserving `dtype` and `device`), then uses
    `torch.distributed.all_gather` to collect the uneven tensors.
    On the main process (rank 0) it concatenates the results along dim 0
    (or stacks scalars), preserving rank order: rank 0 rows first, then rank 1,
    and so on.

    If distributed is unavailable or uninitialized, the input is returned
    unchanged. Non-main ranks return `None`.

    Args:
        x_loc (torch.Tensor): Local tensor whose length along dim 0 may differ
            across ranks (e.g., logits of shape (N_r, C) or targets of shape (N_r,)).

    Returns:
        torch.Tensor | None:
            - Rank 0: concatenated tensor of shape (sum_r N_r, *x_loc.shape[1:]).
              For 0-dim inputs (scalars), returns a 1-D tensor of length `world_size`
              via `torch.stack`.
            - Other ranks: `None`.
            - Non-distributed execution: returns `x_loc`.

    Notes:
        - All non-first dimensions (e.g., `C` in (N_r, C)) must be identical across ranks.
        - Order is preserved across separate calls (e.g., call once for logits and once
          for targets) as long as you build the local tensors in the same per-sample order.
        - Ranks with zero local samples are supported; they still must call this function.
        - No explicit barriers are required; the collectives synchronize internally.

    Example:
        >>> g_logits  = gather_varlen_tensors(local_logits)   # (N, C) on rank 0
        >>> g_targets = gather_varlen_tensors(local_targets)  # (N,)   on rank 0
        >>> if is_main_process():
        ...     loss = F.cross_entropy(g_logits, g_targets)
        ...     acc1 = (g_logits.argmax(1) == g_targets).float().mean().item()
    """
    if is_dist_avail_and_initialized() is False:
        return x_loc
    
    device = x_loc.device
    world_size = get_world_size()

    shape_loc = torch.tensor(x_loc.shape, device=device, dtype=torch.int64)
    shape_glo = [torch.empty_like(shape_loc) for _ in range(world_size)]
    dist.all_gather(shape_glo, shape_loc)

    shapes = [tuple(int(v) for v in s.tolist()) for s in shape_glo]
    x_glo = [x_loc.new_empty(s) for s in shapes]
    dist.all_gather(x_glo, x_loc)

    if is_main_process():
        if x_loc.dim() == 0:
            return torch.stack(x_glo, dim=0)
        else:
            return torch.cat(x_glo)
    else:
        return None

def _print_on_master_only(is_master: bool):
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print 

def init_distributed_mode():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    device = torch.device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=device,
    )
    torch.cuda.set_device(device)
    print(f'initialized distributed (rank {rank} | local_rank {local_rank} | world_size {world_size})', flush=True)
    dist.barrier()

    # disable printing when not in master proc
    is_master = rank == 0
    _print_on_master_only(is_master)