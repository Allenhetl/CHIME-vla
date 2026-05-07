"""[C9] Action Expert — π0 flow-matching head + LoRA.

Component map: C9 (action, deploy + train).  Consumes the readout
``c_t`` plus a CLS-pooled current-frame embedding, and emits a 8-DoF
action via flow matching ODE.

Modes:
    * MVP / cfg.one_step_distill=True — single-step (distilled) action head
      for fast inference and stable training.
    * Full / cfg.one_step_distill=False — 4-8 ODE steps with EulerMaruyama
      sampler.

LoRA / freeze contract:
    * cfg.freeze_base=True — freeze base π0 head, train only LoRA adapters
      (rank cfg.lora_r=16) plus the post-LoRA projection.
    * :meth:`freeze` — freezes *every* parameter (used by [C12] CSM and
      L_PRH-only training paths to obtain a "frozen expert" snapshot).

dtype path (CODE_STANDARDS §1.7):
    inputs bf16 → forward bf16 under autocast → output fp32 action.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from chime_vla.config import C9Config


class ActionExpert(nn.Module):
    """π0 flow-matching action expert with LoRA adapters.

    M0: stub.
    """

    def __init__(self, cfg: C9Config, d_h: int, action_dim: int):
        super().__init__()
        self.cfg = cfg
        self.d_h: int = d_h
        self.action_dim: int = action_dim
        self.head: str = cfg.head
        self.lora_r: int = cfg.lora_r
        self.one_step_distill: bool = cfg.one_step_distill
        self.freeze_base: bool = cfg.freeze_base

    def forward(self, c_t: Tensor, h_t_cls: Tensor) -> Tensor:
        """Predict the 8-DoF action.

        Args:
            c_t:    ``(B, N_q + K_w, d_h)`` bf16 — readout context.
            h_t_cls: ``(B, d_h)`` bf16 — pooled current frame
                     (typically ``h_t.mean(dim=1)``; see
                     ``chime_train_step`` pseudocode in CODE_STRUCTURE §7).

        Returns:
            action: ``(B, action_dim=8)`` fp32 — predicted action vector.
            (Velocity vector form when one_step_distill=False; the actual
            integrated action otherwise.)
        """
        raise NotImplementedError("[C9] ActionExpert.forward — M0 stub")

    def freeze(self) -> None:
        """Freeze every parameter (for CSM probe / PRH-only training)."""
        for p in self.parameters():
            p.requires_grad = False
