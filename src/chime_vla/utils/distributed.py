"""DDP gather helpers (CODE_STANDARDS §5.6)."""

from __future__ import annotations

import torch
import torch.distributed as dist


def is_dist_initialized() -> bool:
    """True iff torch.distributed has been initialized for this process."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """world_size, or 1 outside DDP."""
    return dist.get_world_size() if is_dist_initialized() else 1


def get_rank() -> int:
    """current rank, or 0 outside DDP."""
    return dist.get_rank() if is_dist_initialized() else 0


def all_gather_concat(t: torch.Tensor) -> torch.Tensor:
    """Gather variable-length tensors from every rank along dim 0.

    Single-rank fast path returns the input unchanged.  Across ranks the
    tensor must already be padded to a common length on dim 0 — this helper
    intentionally does **not** handle ragged gather; callers responsible for
    pre-padding (per-task val MSE only carries fixed-shape scalars / per-ep
    metric values).
    """
    if not is_dist_initialized():
        return t
    world = get_world_size()
    bufs = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(bufs, t.contiguous())
    return torch.cat(bufs, dim=0)
