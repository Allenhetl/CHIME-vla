"""[C1] VLM Backbone — SigLIP-ViT visual encoder + LoRA.

Component map: C1 (perception, deploy + train).  Wraps a frozen SigLIP-ViT
(``siglip_vit_{s,b,l}``) and exposes per-frame token features ``h_t``.  LoRA
adapters (rank ``cfg.lora_r``) are trainable when ``cfg.freeze_backbone``
is False; otherwise the entire vision tower is frozen and only the
projection / proprio fusion layers learn.

dtype path (CODE_STANDARDS §1.7):
    rgb: float32 in [0,1] (B, 3, 224, 224)
    proprio: float32 (B, P=8)
    forward in bf16 under autocast → returns bf16 (B, N=256, d_h=1152).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from chime_vla.config import C1Config


class VLMBackbone(nn.Module):
    """SigLIP-ViT visual encoder fused with proprio token (CODE_STRUCTURE §3.1).

    The fusion strategy (concat / cross-attn / FiLM) is selected by
    ``cfg.backbone`` family; the public forward signature is identical
    across choices.

    M0: stub — see ``raise NotImplementedError`` in :meth:`forward`.
    """

    def __init__(self, cfg: C1Config):
        super().__init__()
        self.cfg = cfg
        self.backbone_name: str = cfg.backbone
        self.lora_r: int = cfg.lora_r
        self.freeze_backbone: bool = cfg.freeze_backbone
        # Output dims (per architecture v2.1; SigLIP2-base = 768, doubled by
        # pooler ⊕ spatial_mean fusion).  M0 stub: declarative only.
        self.d_h: int = 1152
        self.N: int = 256

    def forward(self, rgb: Tensor, proprio: Tensor) -> Tensor:
        """Encode one frame.

        Args:
            rgb:     ``(B, 3, 224, 224)`` float32 in ``[0, 1]``.
            proprio: ``(B, P=8)`` float32.

        Returns:
            ``h_t`` : ``(B, N=256, d_h=1152)`` bf16 token sequence.
        """
        raise NotImplementedError("[C1] VLMBackbone.forward — M0 stub")
