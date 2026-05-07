"""[C4] Semantic write head — slot-routed delta-rule into M_sem.

Component map: C4 (heads, deploy + train).  Projects ``h_t`` to a query and
a value, routes the value into one of the K_s slots of :class:`SemBank`
via softmax over slot keys, and applies a delta-rule update gated by
``γ_sem``.

Slot lifecycle (CODE_STANDARDS §1.9, architecture v2.1 [C7] §D5):
    * Softmax routing logits MUST be penalised by
      ``logit -= 1e9 * slot_free`` so that *free* slots are not written
      (they are reserved for explicit allocate-on-first-write).
    * **First-write fallback**: if every slot is free (e.g. the very first
      write into a fresh bank), we drop the penalty and route to the
      cosine-best slot — otherwise the masked softmax would be 0/0.
    * Slots that receive ``routing_prob > mark_occupied_threshold`` flip
      ``slot_free → False`` and update ``timestamp``.

SG-1 contract (CODE_STANDARDS §1.3):
    Caller MUST pass ``sg(γ_sem)`` — same reason as [C3].

dtype path:
    * Projections (``MLP_q``, ``MLP_v``) run in autocast bf16, but the
      delta is cast to fp32 before the out-of-place add (CODE_STANDARDS §1.7).
    * ``m_sem.v`` is fp32 (delta-rule accumulator) and is REASSIGNED on
      every write — see the "Autograd-friendly write path" note in
      ``geo_write.py`` for the same rationale (M3+ grad flow).
    * ``m_sem.k`` is **frozen** — never touched by the write path.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from chime_vla.config import C4Config
from chime_vla.memory.sem_bank import SemBank


# Threshold above which a routing probability counts as "occupying" a slot
# (flips slot_free → False and bumps timestamp).  Hard-coded since C4Config
# doesn't expose it; safe default per architecture v2.1 [C4] §D.5.
_MARK_OCCUPIED_THRESHOLD: float = 0.1


def _make_mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Module:
    """Two-layer MLP with GELU non-linearity (matches [C3] convention)."""
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, d_out),
    )


class SemWriteHead(nn.Module):
    """Slot-routed delta-rule write into the semantic slot bank.

    Mutates ``m_sem.v`` (and ``m_sem.slot_free`` / ``m_sem.timestamp``) in
    place; :meth:`forward` returns ``None``.
    """

    def __init__(
        self,
        cfg: C4Config,
        d_h: int,
        d_s: int,
        K_s: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_h: int = d_h
        self.d_s: int = d_s
        self.K_s: int = K_s
        self.qv_proj_hidden: int = cfg.qv_proj_hidden
        self.softmax_temp: float = cfg.softmax_temp

        # Routing query + value projections.  pool(h_t) ∈ R^{d_h} → R^{d_s}.
        self.mlp_q = _make_mlp(d_h, self.qv_proj_hidden, d_s)
        self.mlp_v = _make_mlp(d_h, self.qv_proj_hidden, d_s)

        self.mark_occupied_threshold: float = _MARK_OCCUPIED_THRESHOLD

    # ------------------------------------------------------------------ #
    def forward(
        self,
        h_t: Tensor,
        gamma_sem: Tensor,
        m_sem: SemBank,
        step: int = 0,
    ) -> None:
        """Slot-route + delta-rule write the current frame's semantic content.

        Args:
            h_t:       ``(B, N, d_h)`` bf16 (under autocast) or fp32.
            gamma_sem: ``(B,)`` fp32 in ``[0, 1]``.  **Caller MUST pass
                       ``sg(γ_sem)`` per SG-1.**
            m_sem:     :class:`SemBank`; ``v`` / ``slot_free`` / ``timestamp``
                       mutated in place.
            step:      current global step (used to update ``timestamp``).

        Returns:
            None.
        """
        B, N, d_h = h_t.shape
        assert d_h == self.d_h, (
            f"SemWriteHead: expected d_h={self.d_h}, got {d_h}"
        )
        assert gamma_sem.shape == (B,), (
            f"SemWriteHead: gamma_sem must be (B,), got {tuple(gamma_sem.shape)}"
        )

        # ---- Step 1: pool tokens -------------------------------------- #
        # Architecture §D.1: pool 256 tokens to a single 1152-d vector.
        pooled = h_t.mean(dim=1)  # (B, d_h)

        # ---- Step 2: project to (q, v_new) ---------------------------- #
        # Match MLP parameter dtype to support callers that pass bf16
        # tensors outside an autocast region (e.g. unit smoke tests).
        param_dtype = next(self.mlp_q.parameters()).dtype
        if pooled.dtype != param_dtype:
            pooled = pooled.to(param_dtype)
        q = self.mlp_q(pooled)        # (B, d_s)
        v_new = self.mlp_v(pooled)    # (B, d_s)

        # Cast to fp32 for the routing softmax + delta-rule write — mixed
        # precision flows through MLPs but the memory write must be fp32
        # (CODE_STANDARDS §1.7).
        q_f32 = q.to(torch.float32)
        v_new_f32 = v_new.to(torch.float32)

        # ---- Step 3: cosine-style routing softmax --------------------- #
        # logits_{b, i} = (q · k_i) / softmax_temp
        # k is fp32 frozen, q already fp32.
        logits = (
            torch.einsum("bd,bkd->bk", q_f32, m_sem.k) / self.softmax_temp
        )  # (B, K_s)

        slot_free = m_sem.slot_free  # (B, K_s) bool
        # First-write fallback: if every slot in a row is free, do not
        # subtract the penalty — otherwise softmax over all -1e9 ⇒ NaN.
        all_free_per_row = slot_free.all(dim=-1, keepdim=True)  # (B, 1)
        penalty = torch.where(
            all_free_per_row,
            torch.zeros_like(slot_free, dtype=torch.float32),
            slot_free.to(torch.float32) * 1e9,
        )  # (B, K_s)
        logits = logits - penalty
        routing_probs = F.softmax(logits, dim=-1)  # (B, K_s)

        # ---- Step 4: delta-rule accumulation -------------------------- #
        # delta_{b, i, d} = γ_sem_b · w_{b, i} · v_new_{b, d}
        # Architecture §D simplification: φ(q) ⊗ v with φ=ELU+1 reduces to
        # the same outer-product up to a positive scale; CODE_STRUCTURE
        # signature uses scalar weights · v_new (matching prompt spec).
        gamma_f32 = gamma_sem.to(torch.float32)  # (B,)
        delta = (
            gamma_f32.view(B, 1, 1)
            * routing_probs.unsqueeze(-1)  # (B, K_s, 1)
            * v_new_f32.unsqueeze(1)       # (B, 1,   d_s)
        )  # (B, K_s, d_s) fp32

        # Out-of-place add + reassignment (autograd-friendly path; see
        # module docstring).  The previous in-place ``m_sem.v.add_`` would
        # bump the version counter on a tensor that is later read by [C8],
        # invalidating saved forward state for L_main / L_PRH backward.
        m_sem.v = m_sem.v + delta.to(m_sem.v.dtype)

        # ---- Step 5: update slot_free + timestamp --------------------- #
        # γ_sem ≈ 0 ⇒ no meaningful write happened ⇒ skip metadata bumps.
        # We tolerate γ being exactly 0; the metadata update is gated by
        # both `gamma > 0` and `routing_prob > threshold`.
        # M3 fix: use argmax (top-1) instead of threshold-based mask.
        # With K_s=64 the softmax is too flat (~1/64=0.0156) for any slot
        # to cross threshold 0.1, so slot_free never flipped → M_sem
        # occupancy stayed 0% → C8 read found no occupied slots → C4
        # received no gradient. Argmax guarantees the dominant slot per
        # write is marked, breaking the deadlock without requiring sharp
        # softmax tuning.
        with torch.no_grad():
            gamma_active = gamma_f32 > 0  # (B,)
            if bool(gamma_active.any()):
                top_slot = routing_probs.argmax(dim=-1)  # (B,)
                rows = torch.arange(B, device=top_slot.device)[gamma_active]
                cols = top_slot[gamma_active]
                m_sem.slot_free[rows, cols] = False
                m_sem.timestamp[rows, cols] = int(step)
