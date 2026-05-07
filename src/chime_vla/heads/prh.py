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

Topology (per architecture v2.1 [C11] §):
    Six independent 2-layer MLPs — one (obs, act) pair per horizon.
    obs head: Linear(d_h, hidden) → GELU → Linear(hidden, d_h)
    act head: Linear(d_h, hidden) → GELU → Linear(hidden, action_dim)

dtype: PRH internals run in fp32 (loss-precision rule, CODE_STANDARDS
§1.7).  ``forward`` upcasts the (typically bf16) input to fp32 and
returns fp32 predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from chime_vla.config import C11Config


class PRH(nn.Module):
    """Per-horizon ``(ô_{t+k}, â_{t+k})`` prediction probe.

    Six 2-layer MLPs total (``len(cfg.horizons)`` × {obs, act}) indexed
    by ``self.heads[f"obs_{k}"]`` / ``self.heads[f"act_{k}"]``.
    """

    def __init__(self, cfg: C11Config, d_h: int, action_dim: int):
        super().__init__()
        self.cfg = cfg
        self.d_h: int = int(d_h)
        self.action_dim: int = int(action_dim)
        self.horizons: list[int] = list(cfg.horizons)
        self.alpha_a: float = float(cfg.alpha_a)
        self.pred_mlp_hidden: int = int(cfg.pred_mlp_hidden)

        heads: dict[str, nn.Module] = {}
        for k in self.horizons:
            heads[f"obs_{k}"] = nn.Sequential(
                nn.Linear(self.d_h, self.pred_mlp_hidden),
                nn.GELU(),
                nn.Linear(self.pred_mlp_hidden, self.d_h),
            )
            heads[f"act_{k}"] = nn.Sequential(
                nn.Linear(self.d_h, self.pred_mlp_hidden),
                nn.GELU(),
                nn.Linear(self.pred_mlp_hidden, self.action_dim),
            )
        self.heads = nn.ModuleDict(heads)

        # Force fp32 parameters explicitly (defensive — some downstream
        # autocast contexts could otherwise instantiate bf16 params).
        self.to(torch.float32)

    def forward(self, m_t_sg: Tensor) -> dict[int, tuple[Tensor, Tensor]]:
        """Predict per-horizon ``(ô, â)`` from a stop-grad'd readout vector.

        Args:
            m_t_sg: ``(B, d_h)`` (typically bf16). **Caller MUST pass
                ``sg(m_t)`` per SG-2** — ``mean`` over c_t is computed
                upstream.

        Returns:
            ``{k: (o_hat[B, d_h], a_hat[B, action_dim])}`` for each
            ``k in cfg.horizons`` — both tensors fp32.
        """
        if m_t_sg.dim() != 2:
            raise ValueError(
                f"PRH.forward expected (B, d_h); got {tuple(m_t_sg.shape)}"
            )
        if m_t_sg.size(-1) != self.d_h:
            raise ValueError(
                f"PRH.forward expected last-dim={self.d_h}; got {m_t_sg.size(-1)}"
            )

        m32 = m_t_sg.to(torch.float32)

        out: dict[int, tuple[Tensor, Tensor]] = {}
        for k in self.horizons:
            o_hat = self.heads[f"obs_{k}"](m32)
            a_hat = self.heads[f"act_{k}"](m32)
            out[int(k)] = (o_hat, a_hat)
        return out
