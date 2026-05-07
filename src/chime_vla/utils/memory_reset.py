"""Episode-boundary memory reset hook (CODE_STANDARDS §1.2).

Single entry point for resetting all memory containers (M_work / M_geo /
M_sem) for a given subset of batch rows.  Called by the LightningModule
on episode transitions detected at the dataloader / batch-collate stage.

Expected ``state`` shape:

    state.M_work : :class:`chime_vla.perception.fifo_buffer.WorkBuffer`
    state.M_geo  : :class:`chime_vla.memory.geo_grid.GeoGrid`
    state.M_sem  : :class:`chime_vla.memory.sem_bank.SemBank`
"""

from __future__ import annotations

from typing import Optional, Protocol

import torch
from torch import Tensor


class _ChimeStateLike(Protocol):
    """Structural type — anything with M_work / M_geo / M_sem attributes."""

    M_work: object
    M_geo: object
    M_sem: object


def reset_memory(
    state: _ChimeStateLike,
    batch_indices: Optional[Tensor] = None,
    *,
    regen_sem_keys: bool = True,
) -> None:
    """Reset all memory containers for selected batch rows (or all of them).

    Args:
        state:          object exposing ``M_work``, ``M_geo``, ``M_sem``.
        batch_indices:  ``(B',)`` long tensor of episodes to reset, or
                        ``None`` for full reset (every episode in the batch).
        regen_sem_keys: forwarded to :meth:`SemBank.reset` — when True the
                        episode boundary draws fresh frozen random keys.

    Returns:
        None — every container is mutated in place.
    """
    raise NotImplementedError("utils.memory_reset.reset_memory — M0 stub")
