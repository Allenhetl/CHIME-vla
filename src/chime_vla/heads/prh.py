"""[C11] Predictive Read Head (training-only).

Component map: C11 (heads, training-only — never invoked at deploy).
Takes the post-readout ``m_t = c_t.mean(dim=1)`` (after :meth:`sg`),
projects through small MLPs, and predicts:

    For each k in cfg.horizons:
        ô_{t+k} : (B, d_h)            — future obs/observation hidden
        â_{t+k} : (B, action_dim=8)   — future action

L_PRH (training/losses.py) compares against teacher-forced future
embeddings (and ground-truth actions weighted by α_a).

SG-2 contract (CODE_STANDARDS §1.3, ``docs/grad_flow_contract.md``):
    Caller MUST pass ``sg(m_t)`` — gradient must NOT propagate from
    L_PRH through ``c_t / read interface / perception / write heads``.
    PRH only fits a probe over the read-out manifold.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from chime_vla.config import C11Config


class PRH(nn.Module):
    """Per-horizon (ô_{t+k}, â_{t+k}) prediction probe.

    Topology: one shared trunk MLP + 2 prediction heads per horizon
    (so ``len(cfg.horizons) * 2`` linear-final heads in total).

    M0: stub.
    """

    def __init__(self, cfg: C11Config, d_h: int, action_dim: int):
        super().__init__()
        self.cfg = cfg
        self.d_h: int = d_h
        self.action_dim: int = action_dim
        self.horizons: list[int] = list(cfg.horizons)
        self.alpha_a: float = cfg.alpha_a
        self.pred_mlp_hidden: int = cfg.pred_mlp_hidden

    def forward(self, m_t_sg: Tensor) -> dict[int, tuple[Tensor, Tensor]]:
        """Predict per-horizon ``(ô, â)`` from a stop-grad'd readout vector.

        Args:
            m_t_sg: ``(B, d_h)`` bf16. **Caller MUST pass ``sg(m_t)`` per
                    SG-2** — ``mean`` over c_t is computed upstream.

        Returns:
            ``{k: (o_hat[B, d_h], a_hat[B, action_dim])}`` for each
            ``k in cfg.horizons``.
        """
        raise NotImplementedError("[C11] PRH.forward — M0 stub")
