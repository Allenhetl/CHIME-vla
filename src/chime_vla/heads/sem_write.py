"""[C4] Semantic write head — slot-routed delta-rule into M_sem.

Component map: C4 (heads, deploy + train).  Projects ``h_t`` to a query and
a value, routes the value into one of the K_s slots of :class:`SemBank`
via softmax over slot keys, and applies a delta-rule update gated by
``γ_sem``.

Slot lifecycle (CODE_STANDARDS §1.9):
    Softmax routing logits MUST be penalised by
    ``logit -= 1e9 * slot_free`` so that *free* slots are not written
    (free slots are reserved for fresh allocations triggered by the
    "explicit allocate-on-first-write" rule, architecture v2.1 §D.5).

SG-1 contract (CODE_STANDARDS §1.3):
    Caller MUST pass ``sg(γ_sem)`` — same reason as [C3].

dtype path: same as [C3] — projections fp32, in-place writes fp32.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from chime_vla.config import C4Config
from chime_vla.memory.sem_bank import SemBank


class SemWriteHead(nn.Module):
    """Slot-routed delta-rule write into the semantic slot bank.

    Mutates ``m_sem.v`` (and ``m_sem.slot_free`` / ``m_sem.timestamp``) in
    place; :meth:`forward` returns ``None``.

    M0: stub.
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

    def forward(
        self,
        h_t: Tensor,
        gamma_sem: Tensor,
        m_sem: SemBank,
    ) -> None:
        """Slot-route + delta-rule write the current frame's semantic content.

        Args:
            h_t:       ``(B, N, d_h)`` bf16.
            gamma_sem: ``(B,)`` fp32 in ``[0, 1]``.  **Caller MUST pass
                       ``sg(γ_sem)`` per SG-1.**
            m_sem:     :class:`SemBank`; ``v`` / ``slot_free`` / ``timestamp``
                       mutated in place.

        Returns:
            None.
        """
        raise NotImplementedError("[C4] SemWriteHead.forward — M0 stub")
