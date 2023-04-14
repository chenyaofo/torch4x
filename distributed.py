import os
import logging
import pathlib
import shutil
import typing

import torch
import torch.distributed as dist

_logger = logging.getLogger(__name__)


def is_dist_avail_and_init() -> bool:
    """

    Returns:
        bool: True if distributed mode is initialized correctly, False otherwise.
    """
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    """

    Returns:
        int: The rank of the current node in distributed system, return 0 if distributed 
        mode is not initialized.
    """
    rank = os.environ.get("RANK", None)
    if rank is None:
        rank = dist.get_rank() if is_dist_avail_and_init() else 0
    return int(rank)


def local_rank() -> int:
    """

    Returns:
        int: The local rank of the current node in distributed system, return 0 if distributed 
        mode is not initialized.
    """
    rank = os.environ.get("LOCAL_RANK", None)
    if rank is None:
        rank = 0
    return int(rank)


def world_size() -> int:
    """

    Returns:
        int: The world size of the  distributed system, return 1 if distributed mode is not 
        initialized.
    """
    world_size = os.environ.get("WORLD_SIZE", None)
    if world_size is None:
        world_size = dist.get_world_size() if is_dist_avail_and_init() else 1
    return int(world_size)


def is_rank_0() -> bool:
    """

    Returns:
        int: True if the rank current node is euqal to 0. Thus it will always return True if 
        distributed mode is not initialized.
    """
    return rank() == 0


def torchsave(obj: typing.Any, f: str, **kwargs) -> None:
    """A simple warp of torch.save. This function is only performed when the current node is the
    master. It will do nothing otherwise. 

    Args:
        obj (typing.Any): The object to save.
        f (str): The output file path.
    """
    if is_rank_0():
        f: pathlib.Path = pathlib.Path(f)
        tmp_name = f.with_name("tmp.pt")
        torch.save(obj, tmp_name,  **kwargs)
        shutil.move(tmp_name, f)
