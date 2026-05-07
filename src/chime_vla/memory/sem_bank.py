"""[C7] M_sem — slot-bank with frozen random keys + slot_free mask
(CODE_STRUCTURE §3.5, CODE_STANDARDS §1.9).

Component map: C7 (memory, deploy + train).  Holds:
* ``k`` : (B, K_s, d_s) fp32 — per-episode frozen *random* keys (regenerated
  on episode reset)
* ``v`` : (B, K_s, d_s) fp32 — delta-rule accumulator (zero at episode start)
* ``slot_free`` : (B, K_s) bool — episode-scoped free-slot mask
* ``timestamp`` : (B, K_s) int64 — last write step (for LRU eviction)

Slot lifecycle invariant (CODE_STANDARDS §1.9, ``tests/test_slot_lifecycle.py``):
    * episode start: ``slot_free`` all True; ``v = 0``; ``k = randn(...)``
    * write to slot i: ``v_i ← v_i + Δv``; ``slot_free[i] ← False``
    * evict slot i:  ``v_i ← 0``; ``slot_free[i] ← True``; ``k_i unchanged``
    * softmax routing must apply ``logit -= 1e9 * slot_free`` to mask free slots.
"""

from __future__ import annotations

import math
import warnings
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
    """

    def __init__(self, cfg: C7Config, batch_size: int, device: torch.device | str):
        self.cfg = cfg
        self.K_s: int = cfg.K_s
        self.d_s: int = cfg.d_s
        self.evict_strategy: str = cfg.evict_strategy
        self.B: int = batch_size
        self.device: torch.device = torch.device(device)

        # Frozen random keys (unit-variance), per-episode.  Detached, no grad.
        # `torch.randn(...) / sqrt(d_s)` ⇒ each entry ~ N(0, 1/d_s),
        # vectors have unit-variance norm ~ 1.
        self.k: Tensor = (
            torch.randn(
                (self.B, self.K_s, self.d_s),
                dtype=torch.float32,
                device=self.device,
            )
            / math.sqrt(self.d_s)
        ).detach()

        # Value tensor — fp32 (CODE_STANDARDS §1.7) for delta-rule
        # accumulation across T~200 steps without bf16 underflow.
        self.v: Tensor = torch.zeros(
            (self.B, self.K_s, self.d_s), dtype=torch.float32, device=self.device
        )

        # All slots start free.
        self.slot_free: Tensor = torch.ones(
            (self.B, self.K_s), dtype=torch.bool, device=self.device
        )

        # Last-write step (int64 to match torch defaults & avoid overflow on
        # long episodes).
        self.timestamp: Tensor = torch.zeros(
            (self.B, self.K_s), dtype=torch.int64, device=self.device
        )

    # ------------------------------------------------------------------ #
    # Episode-boundary reset                                             #
    # ------------------------------------------------------------------ #
    def reset(
        self,
        batch_indices: Optional[Tensor] = None,
        regen_keys: bool = True,
    ) -> None:
        """Reset selected episodes: zero v, set slot_free=True, optionally
        re-randomise k, and zero timestamp.

        Args:
            batch_indices: ``(B',)`` long tensor, or ``None`` to reset every
                row in the batch.
            regen_keys: if True, re-draw frozen random keys for the reset rows.
                Per architecture v2.1 §H Trade-off 2 we abandon cross-episode
                key sharing — each episode gets fresh frozen keys.
        """
        if batch_indices is None:
            self.v.zero_()
            self.slot_free.fill_(True)
            self.timestamp.zero_()
            if regen_keys:
                new_k = (
                    torch.randn_like(self.k) / math.sqrt(self.d_s)
                ).detach()
                self.k.copy_(new_k)
            return

        # Per-row reset.
        idx = batch_indices.to(self.device).long()
        self.v[idx] = 0
        self.slot_free[idx] = True
        self.timestamp[idx] = 0
        if regen_keys:
            new_rows = (
                torch.randn(
                    (idx.numel(), self.K_s, self.d_s),
                    dtype=self.k.dtype,
                    device=self.device,
                )
                / math.sqrt(self.d_s)
            ).detach()
            self.k[idx] = new_rows

    # ------------------------------------------------------------------ #
    # LRU eviction primitive                                             #
    # ------------------------------------------------------------------ #
    def evict(self, batch_idx: int, slot_idx: int) -> None:
        """Evict one slot for one batch row: ``v_i ← 0``, ``slot_free[i] ← True``.

        Key ``k_i`` is **not** modified — slot retains its surface for
        future reuse with a fresh value accumulator (CODE_STANDARDS §1.9).

        Evicting an already-free slot is a no-op (with a warning) — caller
        ought to check ``slot_free`` first, but we tolerate the redundancy.
        """
        if bool(self.slot_free[batch_idx, slot_idx]):
            warnings.warn(
                f"SemBank.evict({batch_idx}, {slot_idx}): slot is already free; "
                "no-op",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self.v[batch_idx, slot_idx] = 0
        self.slot_free[batch_idx, slot_idx] = True
        # k_i is intentionally untouched (CODE_STANDARDS §1.9).
