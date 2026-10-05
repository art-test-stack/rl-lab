from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch import nn
from dataclasses import dataclass
import os, warnings
from functools import lru_cache

import logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------
# distributed utils
# -----------------------------------------------------------

@lru_cache()
def get_device_type():
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.set_per_process_memory_fraction(0.)
        device_type = "mps"
    elif torch.xpu.is_available():
        device_type = "xpu"
    else:
        device_type = "cpu"
    
    if device_type in ("xpu", "tpu"):
        warnings.warn(f"Device type {device_type!r} is not fully tested. May exhibit unexpected behavior.")
    return device_type

@lru_cache()
def is_ddp_requested() -> bool:
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    return dist.is_initialized() and dist.is_available()

@lru_cache()
def get_base_dist_info():
    is_requested = is_ddp_requested()
    if is_requested:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        world_size = 1
        rank = 0
        local_rank = 0
    tp_size = 1 # TODO: implement tensor parallelism support

    if is_ddp_initialized():
        # sanity check that env vars match dist info
        assert world_size == dist.get_world_size(), f"World size from environment variable WORLD_SIZE={world_size} does not match dist.get_world_size()={dist.get_world_size()}"
        assert rank == dist.get_rank(), f"Rank from environment variable RANK={rank} does not match dist.get_rank()={dist.get_rank()}"
        assert torch.cuda.current_device() == local_rank

    return is_requested, world_size, rank, local_rank, tp_size

def init_distributed(
    device_type: str | None = None,
    random_seed: int = 42,
) -> DistributedContext:
    device_type = device_type or get_device_type()

    assert device_type in ("cuda", "mps", "cpu"), (
        f"Unsupported device type: {device_type!r}."
    )

    # Reproducibility
    torch.manual_seed(random_seed)

    if device_type == "cuda":
        torch.cuda.manual_seed(random_seed)
        torch.set_float32_matmul_precision("high")

    elif device_type == "mps":
        torch.mps.manual_seed(random_seed)

    # Single process by default
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Device placement
    if device_type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device(device_type)

    # No torch.distributed initialization for normal single-device execution
    if world_size == 1:
        return DistributedContext(device=device)

    assert all(
        key in os.environ
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    ), "RANK, LOCAL_RANK and WORLD_SIZE must be set for distributed execution."

    if device_type == "mps":
        raise ValueError("Distributed execution requires CPU/Gloo or CUDA/NCCL")
    if not dist.is_initialized():
        # Select the requested backend instead of auto-detecting host accelerators.
        dist.init_process_group(backend="nccl" if device_type == "cuda" else "gloo")

    mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(world_size,),
        mesh_dim_names=("dp",),
    )

    return DistributedContext(
        device=device,
        mesh=mesh,
    )


@dataclass(frozen=True)
class DistributedContext:
    device: torch.device
    mesh: DeviceMesh | None = None

    @property
    def rank(self) -> int:
        return self.mesh.get_rank() if self.mesh is not None else 0

    @property
    def world_size(self) -> int:
        return self.mesh.size() if self.mesh is not None else 1

    @property
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def is_distributed(self) -> bool:
        return self.mesh is not None

    @property
    def dp_rank(self) -> int:
        if self.mesh is None:
            return 0
        return self.mesh["dp"].get_rank()

    @property
    def dp_mesh(self) -> DeviceMesh | None:
        if self.mesh is None:
            return
        return self.mesh["dp"]


def cleanup_dist_groups():
    if is_ddp_initialized():
        dist.destroy_process_group()
