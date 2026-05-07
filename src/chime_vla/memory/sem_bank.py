"""[C7] M_sem — slot-bank with frozen random keys + slot_free mask
(CODE_STRUCTURE §3.5, CODE_STANDARDS §1.9).

Component map: C7 (memory, deploy + train).  Holds:
* ``k`` : (B, K_s, d_s) fp32 — per-episode frozen *random* keys (regenerated
  on episode reset)
* ``v`` : (B, K_s, d_s) fp32 — delta-rule accumulator (zero at episode start)
* ``slot_free`` : (B, K_s) bool — episode-scoped free-slot mask
* ``timestamp`` : (B, K_s) int32 — last write step (for LRU eviction)

Slot lifecycle invariant (CODE_STANDARDS §1.9, ``tests/test_slot_lifecycle.py``):
    * episode start: ``slot_free`` all True; ``v = 0``; ``k = randn(...)``
    * write to slot i: ``v_i ← v_i + Δv``; ``slot_free[i] ← False``
    * evict slot i:  ``v_i ← 0``; ``slot_free[i] ← True``; ``k_i unchanged``
    * softmax routing must apply ``logit -= 1e9 * slot_free`` to mask free slots.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from chime_vla.config import C7Config


class SemBank:
    """Slot bank with frozen random keys + slot_free mask.

    Not an ``nn.Module``: keys are randomised per-episode (frozen for the
    duration of the episode), values are delta-rule accumulated.  Eviction
    via :meth:`evict` zeros the value but preserves the key (so the slot
    can be re-used with the same surface but a fresh accumulator).

    M0: stub — :meth:`reset` and :meth:`evict` raise.
    """

    def __init__(self, cfg: C7Config, batch_size: int, device: torch.device | str):
        self.cfg = cfg
        self.K_s: int = cfg.K_s
        self.d_s: int = cfg.d_s
        self.evict_strategy: str = cfg.evict_strategy
        self.B: int = batch_size
        self.device: torch.device = torch.device(device)

        # Allocate eagerly with conservative defaults; reset() randomises k.
        self.k: Tensor = torch.zeros(
            (self.B, self.K_s, self.d_s), dtype=torch.float32, device=self.device
        )
        self.v: Tensor = torch.zeros(
            (self.B, self.K_s, self.d_s), dtype=torch.float32, device=self.device
        )
        self.slot_free: Tensor = torch.ones(
            (self.B, self.K_s), dtype=torch.bool, device=self.device
        )
        self.timestamp: Tensor = torch.zeros(
            (self.B, self.K_s), dtype=torch.int32, device=self.device
        )

    def reset(
        self,
        batch_indices: Optional[Tensor] = None,
        regen_keys: bool = True,
    ) -> None:
        """Reset selected episodes: zero v, set slot_free=True, optionally re-randomise k.

        Args:
            batch_indices: ``(B',)`` long tensor or None (= reset all).
            regen_keys: if True, re-draw frozen random keys for the reset rows.
        """
        raise NotImplementedError("[C7] SemBank.reset — M0 stub")

    def evict(self, batch_idx: int, slot_idx: int) -> None:
        """Evict one slot for one batch row: ``v_i ← 0``, ``slot_free[i] ← True``.

        Key ``k_i`` is **not** modified — slot retains its surface for
        future reuse with a fresh value accumulator (CODE_STANDARDS §1.9).
        """
        raise NotImplementedError("[C7] SemBank.evict — M0 stub")
