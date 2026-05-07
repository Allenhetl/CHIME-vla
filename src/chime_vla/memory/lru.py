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


class TimestampLRUEvictor:
    """Evict the slot with the lowest ``timestamp`` (least recently written).

    M0: stub.
    """

    def __init__(self) -> None:
        pass

    def select(self, m_sem: SemBank, batch_idx: int) -> int:
        """Return the slot index to evict for ``batch_idx``.

        Returns the smallest-timestamp slot among slots with ``slot_free=False``;
        if every slot is free this returns 0 by convention.
        """
        raise NotImplementedError("TimestampLRUEvictor.select — M0 stub")

    def __call__(self, m_sem: SemBank, batch_idx: int) -> int:
        return self.select(m_sem, batch_idx)


class CSMLRUEvictor:
    """Combined importance × recency eviction (M3+, requires [C12] CSM).

    Score: ``importance[i] * exp(-decay * (now - timestamp[i]))``.
    The slot with the smallest score is evicted.

    M0: stub.
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
        raise NotImplementedError("CSMLRUEvictor.select — M0 stub")

    def __call__(
        self,
        m_sem: SemBank,
        batch_idx: int,
        importance: Tensor,
        now: int,
    ) -> int:
        return self.select(m_sem, batch_idx, importance, now)


def evict_inplace(
    m_sem: SemBank,
    evictor: TimestampLRUEvictor | CSMLRUEvictor,
    batch_idx: int,
    *,
    importance: Optional[Tensor] = None,
    now: Optional[int] = None,
) -> int:
    """High-level helper: pick a slot via ``evictor`` and call ``m_sem.evict``.

    Returns the evicted slot index (for logging).  M0: stub.
    """
    raise NotImplementedError("memory.lru.evict_inplace — M0 stub")
