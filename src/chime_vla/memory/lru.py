"""LRU eviction strategies for [C7] SemBank.

Two policies (selected by ``C7Config.evict_strategy``):

* ``timestamp`` (MVP / M0-M2) — evict the slot whose ``timestamp`` is the
  smallest, i.e. the least-recently-written.
* ``csm_lru`` (M3+, requires [C12] CSM importance weights) — evict the
  slot with the smallest combined ``CSM_importance × timestamp_decay``
  score.

Both strategies operate in-place on a :class:`chime_vla.memory.sem_bank.SemBank`.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from chime_vla.memory.sem_bank import SemBank


# --------------------------------------------------------------------------- #
# Functional helpers                                                          #
# --------------------------------------------------------------------------- #
def find_oldest_occupied_slot(
    sem_bank: SemBank, batch_idx: int
) -> Optional[int]:
    """Return the index of the oldest occupied slot for ``batch_idx``.

    Free slots are masked out (their timestamp is forced to ``int64.max`` so
    they never win the ``argmin``).  Returns ``None`` if every slot is free.
    """
    occupied = ~sem_bank.slot_free[batch_idx]  # (K_s,) bool
    if not bool(occupied.any()):
        return None
    timestamps = sem_bank.timestamp[batch_idx].clone()
    timestamps[~occupied] = torch.iinfo(timestamps.dtype).max
    return int(timestamps.argmin().item())


# --------------------------------------------------------------------------- #
# Evictor classes                                                             #
# --------------------------------------------------------------------------- #
class TimestampLRUEvictor:
    """Evict the slot with the lowest ``timestamp`` (least recently written)."""

    def __init__(self) -> None:
        pass

    def select(self, m_sem: SemBank, batch_idx: int) -> int:
        """Return the slot index to evict for ``batch_idx``.

        Returns the smallest-timestamp slot among slots with ``slot_free=False``;
        if every slot is free this returns ``0`` by convention.
        """
        idx = find_oldest_occupied_slot(m_sem, batch_idx)
        if idx is None:
            return 0
        return idx

    def __call__(self, m_sem: SemBank, batch_idx: int) -> int:
        return self.select(m_sem, batch_idx)


class CSMLRUEvictor:
    """Combined importance × recency eviction (M3+, requires [C12] CSM).

    Score: ``importance[i] * exp(-decay * (now - timestamp[i]))``.
    The slot with the smallest score is evicted.

    M3 stub: this remains a placeholder until the CSM importance pipeline
    lands ([C12]).  ``select`` is implemented in fp32 ``argmin`` form so it
    can be unit-tested with synthetic importance vectors.
    """

    def __init__(self, decay: float = 0.01) -> None:
        self.decay = float(decay)

    def select(
        self,
        m_sem: SemBank,
        batch_idx: int,
        importance: Tensor,
        now: int,
    ) -> int:
        """Return slot index to evict.

        Args:
            m_sem:      :class:`SemBank` instance.
            batch_idx:  which episode in the batch.
            importance: ``(K_s,)`` per-slot importance from [C12].
            now:        current step.
        """
        occupied = ~m_sem.slot_free[batch_idx]
        if not bool(occupied.any()):
            return 0
        ts = m_sem.timestamp[batch_idx].to(torch.float32)
        recency = torch.exp(-self.decay * (float(now) - ts))
        score = importance.to(torch.float32) * recency
        # Mask free slots so they never win the argmin.
        score = score.masked_fill(~occupied, float("inf"))
        return int(score.argmin().item())

    def __call__(
        self,
        m_sem: SemBank,
        batch_idx: int,
        importance: Tensor,
        now: int,
    ) -> int:
        return self.select(m_sem, batch_idx, importance, now)


# --------------------------------------------------------------------------- #
# High-level driver                                                           #
# --------------------------------------------------------------------------- #
def evict_inplace(
    m_sem: SemBank,
    evictor: TimestampLRUEvictor | CSMLRUEvictor,
    batch_idx: int,
    *,
    importance: Optional[Tensor] = None,
    now: Optional[int] = None,
) -> int:
    """Pick a slot via ``evictor`` and call ``m_sem.evict``.

    Returns the evicted slot index (for logging).  When every slot is
    already free we short-circuit with index ``-1`` and skip ``evict``.
    """
    if isinstance(evictor, CSMLRUEvictor):
        if importance is None or now is None:
            raise ValueError(
                "CSMLRUEvictor requires `importance` and `now` kwargs."
            )
        slot_idx = evictor(m_sem, batch_idx, importance, now)
    else:
        slot_idx = evictor(m_sem, batch_idx)

    # Guard: bank fully free → nothing to evict.
    if bool(m_sem.slot_free[batch_idx].all()):
        return -1
    m_sem.evict(batch_idx, slot_idx)
    return slot_idx
