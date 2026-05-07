"""[C5] Episodic Salience Predictor with Contrast (ESPC).

Component map: C5 (heads, deploy + train).  Reads M_work^{t-1} (the FIFO
**before** the current frame's append, per CODE_STANDARDS §1.3) and emits
two scalar gates per batch row:

    γ_geo, γ_sem ∈ [0, 1]

These gate the write strength of [C3] / [C4] respectively, with the
caller wrapping them in ``sg(.)`` before passing them to write heads
(SG-1, see ``docs/grad_flow_contract.md``).

Architecture (CODE_STRUCTURE §3.3, architecture v2.1 §C):
    1. ψ encoder over (h_t, m_work):
         MVP: ``GRU`` over time (cfg.use_gru=True, cfg.psi_layers=1)
         Full: ``1-layer Transformer``
    2. ``geo_proj`` and ``sem_proj`` MLPs to ``cfg.d_proj=64``
    3. EMA-normalised contrast: ``z = (x - μ_ema) / (σ_ema + eps)``
       (μ_ema, σ_ema updated post-step via :meth:`update_ema`; warmup
       ``cfg.ema_warmup_steps`` returns identity normalisation)
    4. ``γ = sigmoid(z / cfg.sigmoid_temp)``

dtype path (CODE_STANDARDS §1.7):
    inputs bf16 → cast to fp32 internally → γ returned as fp32.
    EMA running stats kept in fp32.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from chime_vla.config import C5Config


class ESPC(nn.Module):
    """ψ encoder + EMA contrast + γ_geo / γ_sem heads.

    Caller-side SG contract:
        Outputs of :meth:`forward` are the *raw* γ values; the train_step
        wraps them in ``sg(.)`` before handing them to [C3] / [C4]
        (SG-1).  The L_HCS BCE target also uses ``sg(.)`` on the
        γ̂ Hindsight target, never on the predicted γ — the gradient
        flowing into ψ is the only learning signal for [C5].

    M0: stub.
    """

    def __init__(self, cfg: C5Config, d_h: int):
        super().__init__()
        self.cfg = cfg
        self.d_h: int = d_h
        self.K_w: int = cfg.K_w
        self.psi_layers: int = cfg.psi_layers
        self.use_gru: bool = cfg.use_gru
        self.d_proj: int = cfg.d_proj
        self.ema_coeff: float = cfg.ema_coeff
        self.ema_warmup_steps: int = cfg.ema_warmup_steps
        self.sigmoid_temp: float = cfg.sigmoid_temp

        # EMA running stats (fp32, per CODE_STANDARDS §1.7) for the two channels.
        self.register_buffer(
            "ema_mu", torch.zeros(2, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "ema_sigma", torch.ones(2, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "ema_step", torch.zeros((), dtype=torch.long), persistent=True
        )

    def forward(self, h_t: Tensor, m_work: Tensor) -> tuple[Tensor, Tensor]:
        """Compute ``(γ_geo, γ_sem)`` for the current frame.

        Args:
            h_t:    ``(B, N, d_h)`` bf16 — current frame tokens.
            m_work: ``(B, K_w, N, d_h)`` bf16 — FIFO **before** appending h_t
                    (per CODE_STANDARDS §1.3 invariant).

        Returns:
            ``(γ_geo, γ_sem)`` each ``(B,)`` fp32 in ``[0, 1]``.
            Caller wraps in ``sg(.)`` before passing to [C3] / [C4] (SG-1).
        """
        raise NotImplementedError("[C5] ESPC.forward — M0 stub")

    def update_ema(self) -> None:
        """Push the most recent batch's pre-sigmoid statistics into the EMA.

        Called *after* loss.backward + optimizer.step to keep the contrast
        baseline up to date.  During warmup (step < ema_warmup_steps) the
        update accumulates statistics but the forward returns identity z.
        """
        raise NotImplementedError("[C5] ESPC.update_ema — M0 stub")
