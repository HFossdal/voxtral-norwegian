"""Data-parallel fine-tuning on the GPUs of one machine.

Started with torchrun, which runs one process per GPU and sets `RANK`,
`LOCAL_RANK` and `WORLD_SIZE`:

    python -m torch.distributed.run --standalone --nproc_per_node 4 scripts/finetune.py ...

Every process holds the whole model. Each optimizer step's batch is split
between the processes, and their LoRA gradients are summed with one
all-reduce before the step, so all copies take the same update. Without
torchrun, `init()` returns `SINGLE` and every helper here does nothing.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as torch_dist

# Generous: the other processes wait at a barrier while the main process
# downloads the data and the model before them.
TIMEOUT = timedelta(hours=2)


@dataclass(frozen=True)
class Dist:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


SINGLE = Dist()


def init() -> Dist:
    """Join the process group if started by torchrun; otherwise `SINGLE`."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return SINGLE
    dist = Dist(int(os.environ["RANK"]), world_size, int(os.environ["LOCAL_RANK"]))
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch_dist.init_process_group(backend, timeout=TIMEOUT)
    return dist


def close(dist: Dist) -> None:
    if dist.enabled:
        torch_dist.destroy_process_group()


def process_device(device: torch.device, dist: Dist) -> torch.device:
    """This process's GPU: `cuda` becomes `cuda:<local rank>`."""
    if dist.enabled and device.type == "cuda":
        return torch.device("cuda", dist.local_rank)
    return device


def shard(n: int, dist: Dist) -> tuple[int, int]:
    """This process's contiguous part of `n` items, as (start, stop).

    Rounded up, so the main process gets an item whenever there is one.
    """
    w = dist.world_size
    return (n * dist.rank + w - 1) // w, (n * (dist.rank + 1) + w - 1) // w


def barrier(dist: Dist) -> None:
    if dist.enabled:
        torch_dist.barrier()


@contextmanager
def main_first(dist: Dist) -> Iterator[None]:
    """The main process runs the block first (downloads, cache files), then the others."""
    if not dist.is_main:
        barrier(dist)
    yield
    if dist.is_main:
        barrier(dist)


def broadcast_object(obj: Any, dist: Dist) -> Any:
    """The main process's `obj`, on every process."""
    if not dist.enabled:
        return obj
    box = [obj]
    torch_dist.broadcast_object_list(box, src=0)
    return box[0]


def sum_scalar(value: float, dist: Dist, device: torch.device) -> float:
    if not dist.enabled:
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    torch_dist.all_reduce(tensor)
    return tensor.item()


def sum_gradients(params: Sequence[torch.nn.Parameter], dist: Dist) -> None:
    """Replace each gradient by its sum over all processes (one all-reduce per dtype).

    A process without a gradient (no clips in its part of a short batch)
    contributes zeros.
    """
    if not dist.enabled:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    for dtype in {p.dtype for p in params}:
        group = [p for p in params if p.dtype == dtype]
        flat = torch.cat([p.grad.reshape(-1) for p in group])
        torch_dist.all_reduce(flat)
        offset = 0
        for p in group:
            p.grad.copy_(flat[offset : offset + p.numel()].view_as(p))
            offset += p.numel()


def broadcast_parameters(params: Sequence[torch.nn.Parameter], dist: Dist) -> None:
    """Give every process the main process's values (LoRA starts random)."""
    if not dist.enabled:
        return
    with torch.no_grad():
        for p in params:
            torch_dist.broadcast(p.data, src=0)


def step_seed(seed: int, step: int, dist: Dist) -> int:
    """Torch seed for one step of one process: dropout differs between processes
    and does not depend on where a run was interrupted."""
    return int(np.random.SeedSequence([seed, step, dist.rank]).generate_state(1)[0])
