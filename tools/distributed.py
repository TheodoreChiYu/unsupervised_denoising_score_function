import os

import torch as th
import torch.distributed as dist


def setup_distributed(rank: int, world_size: int, master_config: dict) -> None:
    """
    Set up a distributed process group.

    :param rank: int.
    :param world_size: int, total number of subprocesses.
    :param master_config: it should contain two keys `addr` and `port`.
    """
    os.environ["MASTER_ADDR"] = master_config["addr"]
    os.environ["MASTER_PORT"] = master_config["port"]
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def get_curr_device():
    """
    Get the current device to use for torch.distributed.
    """
    if not th.cuda.is_available():
        return th.device("cpu")

    if dist.is_initialized():
        return th.device(dist.get_rank())
    else:
        return th.device("cuda")


def sync_params(params) -> None:
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)
