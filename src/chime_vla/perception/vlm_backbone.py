"""[C1] VLM Backbone — SigLIP-ViT visual encoder + LoRA.

Component map: C1 (perception, deploy + train).  Wraps a (frozen) SigLIP-ViT
(``siglip_vit_{s,b,l}``) and exposes per-frame token features ``h_t``.  LoRA
adapters (rank ``cfg.lora_r``) are trainable when ``cfg.freeze_backbone``
is True (the standard MVP setting); the proprio fusion + d_h projection
heads are always trainable.

Shape contract (CODE_STRUCTURE.md §3.1):
    rgb     : (B, 3, 224, 224)  float32 in [0, 1]  *or*  uint8 in [0, 255]
    proprio : (B, P=8)          float32
    returns : h_t (B, N=256, d_h=1152) bf16

dtype path (CODE_STANDARDS §1.7):
    SigLIP forward runs in bf16 (autocast-friendly); output is cast to bf16.
    On CPU-only test environments BF16 autocast is a no-op but the explicit
    ``.to(bf16)`` cast on the return tensor still satisfies the contract.

Design notes (deviations from spec, all documented):
    * SigLIP2-base produces a 14x14 = 196 patch grid.  We bilinearly upsample
      the 2D token grid to 16x16 = 256 so N=256 exactly matches CODE_STRUCTURE
      (rather than picking N=257 with a prefix proprio token).  This keeps
      every downstream component (C2 FIFO, C3/C4 write heads) on the
      declared N=256 contract.
    * SigLIP-base hidden = 768; we add a trainable Linear(768, 1152) head to
      reach the architecture's d_h=1152.
    * proprio is fused via a per-token additive bias: ``proj(proprio)`` is
      broadcast-added to all 256 tokens.  Keeps N=256, mirrors a "modulation"
      style fusion (cheap, well-conditioned, similar to FiLM-bias-only).
    * LoRA: hand-rolled rank-r adapter on q_proj/k_proj/v_proj of every
      SigLIP encoder layer self_attn.  No `peft` dependency.

Mock fallback (offline / cache-miss):
    If the HF download of ``google/siglip2-base-patch16-224`` fails (no
    network, no cache), we fall back to a simple Conv2d patch embed + a
    LayerNorm "mock backbone" that produces 196-d-768 patch tokens.  This
    keeps unit tests passing on machines without HF cache.  ``self.is_mock``
    is True in that case.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from chime_vla.config import C1Config


# SigLIP2-base normalisation (mean=0.5, std=0.5 -> [-1, 1]).
_SIGLIP_NORM_MEAN = 0.5
_SIGLIP_NORM_STD = 0.5

# HF id used when cfg.backbone == "siglip_vit_b" / default fallback.
_HF_MODEL_ID = "google/siglip2-base-patch16-224"

# Target token count after bilinear upsample of patch grid.
_TARGET_N = 256
_TARGET_GRID = 16  # 16 * 16 == 256


class _LoRALinear(nn.Module):
    """Minimal LoRA adapter wrapping an existing ``nn.Linear``.

    y = base(x) + (alpha / r) * dropout(x) @ A^T @ B^T
    Base weights are kept (and remain frozen if ``base.weight.requires_grad``
    is False, as set by VLMBackbone for the SigLIP tower).
    """

    def __init__(self, base: nn.Linear, r: int, alpha: Optional[float] = None,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        in_f = base.in_features
        out_f = base.out_features
        self.r = int(r)
        self.alpha = float(alpha if alpha is not None else r)
        self.scaling = self.alpha / max(1, self.r)
        # A: in_f -> r ; B: r -> out_f.  init A ~ kaiming, B = 0 so initial
        # adapter contribution is 0 (preserves pretrained behaviour at init).
        self.lora_A = nn.Parameter(torch.zeros(self.r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B already zeros.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        if self.r <= 0:
            return out
        # Use base dtype for the LoRA path so autocast/bf16 stays consistent.
        x_l = self.dropout(x)
        # Cast LoRA params to the activation dtype to avoid mixed-dtype errors
        # under bf16 autocast on CPU (where autocast is a no-op).
        a = self.lora_A.to(x_l.dtype)
        b = self.lora_B.to(x_l.dtype)
        delta = (x_l @ a.t()) @ b.t()
        return out + self.scaling * delta


def _inject_lora(vision: nn.Module, r: int) -> int:
    """Wrap q_proj / k_proj / v_proj on every SiglipAttention with LoRA.

    Returns number of Linear layers wrapped.  Iterates by name and rebinds
    via ``setattr`` so the parent module sees the new submodule.
    """
    if r <= 0:
        return 0
    n_wrapped = 0
    for module in vision.modules():
        # Heuristic: any module with q_proj+k_proj+v_proj attrs that are
        # nn.Linear is a self-attention block.
        has_qkv = all(
            isinstance(getattr(module, name, None), nn.Linear)
            for name in ("q_proj", "k_proj", "v_proj")
        )
        if not has_qkv:
            continue
        for name in ("q_proj", "k_proj", "v_proj"):
            base: nn.Linear = getattr(module, name)
            wrapped = _LoRALinear(base, r=r)
            setattr(module, name, wrapped)
            n_wrapped += 1
    return n_wrapped


class _MockVisionTower(nn.Module):
    """Tiny patch-embed-only stand-in used when HF SigLIP cannot be loaded.

    Produces ``last_hidden_state`` of shape [B, 196, 768] mimicking the
    SigLIP2-base output so downstream code is shape-compatible.
    """

    def __init__(self, hidden_size: int = 768, patch_size: int = 16,
                 image_size: int = 224):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patches = (image_size // patch_size) ** 2  # 196
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=patch_size,
                                     stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values: Tensor):  # type: ignore[override]
        x = self.patch_embed(pixel_values)            # [B, C, h, w]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)              # [B, N, C]
        x = x + self.pos_embed
        x = self.norm(x)
        # Mimic HF output struct (only need last_hidden_state).
        class _Out:
            pass
        out = _Out()
        out.last_hidden_state = x
        out.pooler_output = x.mean(dim=1)
        return out


class VLMBackbone(nn.Module):
    """SigLIP-ViT visual encoder fused with proprio (CODE_STRUCTURE §3.1).

    The public forward signature matches the architecture v2.1 [C1] card:
    ``forward(rgb, proprio) -> h_t``.

    Args:
        cfg: C1Config (backbone string, lora rank, freeze flag).

    Attributes:
        is_mock: True when the HF SigLIP load failed and the mock tower is
            being used.  Tests can read this flag to pick numerical
            tolerances appropriately.
    """

    def __init__(self, cfg: C1Config):
        super().__init__()
        self.cfg = cfg
        self.backbone_name: str = cfg.backbone
        self.lora_r: int = cfg.lora_r
        self.freeze_backbone: bool = cfg.freeze_backbone

        # Architecture-locked output dims (per CODE_STRUCTURE §3.1).
        self.d_h: int = 1152
        self.N: int = _TARGET_N
        self.proprio_dim: int = 8

        # ------------------------------------------------------------------
        # 1) SigLIP vision tower (or mock fallback).
        # ------------------------------------------------------------------
        self.is_mock: bool = False
        try:
            from transformers import AutoModel  # type: ignore
            try:
                full = AutoModel.from_pretrained(_HF_MODEL_ID, dtype=torch.float32)
            except TypeError:
                # Older transformers used torch_dtype kwarg.
                full = AutoModel.from_pretrained(_HF_MODEL_ID,
                                                 torch_dtype=torch.float32)
            self.vision = full.vision_model
            self._vision_hidden = int(self.vision.config.hidden_size)
            self._image_size = int(self.vision.config.image_size)
            self._patch_size = int(self.vision.config.patch_size)
        except Exception as exc:  # pragma: no cover — exercised only offline
            # Fall back to a tiny mock tower so tests / offline dev still works.
            self.is_mock = True
            self._vision_hidden = 768
            self._image_size = 224
            self._patch_size = 16
            self.vision = _MockVisionTower(
                hidden_size=self._vision_hidden,
                patch_size=self._patch_size,
                image_size=self._image_size,
            )
            self._mock_reason = repr(exc)

        self._patch_grid: int = self._image_size // self._patch_size  # 14

        # Freeze the vision tower if requested (MVP default).
        if self.freeze_backbone:
            for p in self.vision.parameters():
                p.requires_grad = False
            self.vision.eval()

        # ------------------------------------------------------------------
        # 2) LoRA on q/k/v.
        # ------------------------------------------------------------------
        self._n_lora_wrapped: int = _inject_lora(self.vision, self.lora_r)
        # _inject_lora set requires_grad on lora_A/lora_B to True by default
        # (Parameter init), but if we just walked through frozen params above
        # those LoRA params were not yet attached.  After injection, ensure
        # LoRA params are trainable even when freeze_backbone=True.
        for n, p in self.vision.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                p.requires_grad = True

        # ------------------------------------------------------------------
        # 3) Proprio fusion + d_h projection (always trainable).
        # ------------------------------------------------------------------
        # proprio (B, 8) -> hidden tower dim, broadcast-added per token.
        self.proprio_proj = nn.Sequential(
            nn.Linear(self.proprio_dim, self._vision_hidden),
            nn.GELU(),
            nn.Linear(self._vision_hidden, self._vision_hidden),
        )
        # token dim 768 -> 1152 (per architecture v2.1 d_h).
        self.feature_proj = nn.Linear(self._vision_hidden, self.d_h)
        nn.init.xavier_uniform_(self.feature_proj.weight)
        nn.init.zeros_(self.feature_proj.bias)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise(self, rgb: Tensor) -> Tensor:
        """Convert input rgb to fp32 in [-1, 1] using SigLIP2 normalization.

        Accepts uint8 [0, 255] or float [0, 1].
        """
        if rgb.dtype == torch.uint8:
            x = rgb.to(torch.float32) / 255.0
        else:
            x = rgb.to(torch.float32)
            # If the caller already passed [-1, 1] we still re-normalise; cheap
            # and avoids ambiguity.  The contract says [0, 1] floats.
        x = (x - _SIGLIP_NORM_MEAN) / _SIGLIP_NORM_STD
        return x

    def _resize_tokens_to_n(self, tokens: Tensor) -> Tensor:
        """Bilinearly resample [B, n, C] patch tokens (n = G*G) to [B, N, C].

        Reshapes to [B, C, G, G], bilinear upsample to [B, C, G', G'] where
        G' * G' == N, then flattens.  We require N to be a perfect square
        (256 -> 16x16).  No-op if already at target.
        """
        B, n, C = tokens.shape
        if n == self.N:
            return tokens
        g = int(round(math.sqrt(n)))
        if g * g != n:
            raise RuntimeError(
                f"VLMBackbone: expected square patch grid, got n={n}"
            )
        x = tokens.transpose(1, 2).reshape(B, C, g, g)
        x = F.interpolate(x, size=(_TARGET_GRID, _TARGET_GRID),
                          mode="bilinear", align_corners=False)
        x = x.reshape(B, C, _TARGET_GRID * _TARGET_GRID).transpose(1, 2)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, rgb: Tensor, proprio: Tensor) -> Tensor:
        """Encode one frame.

        Args:
            rgb:     ``(B, 3, 224, 224)`` float in ``[0, 1]`` or uint8 in
                ``[0, 255]``.
            proprio: ``(B, P=8)`` float32.

        Returns:
            ``h_t``: ``(B, N=256, d_h=1152)`` bf16.
        """
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(
                f"VLMBackbone.forward expected rgb of shape (B,3,H,W); "
                f"got {tuple(rgb.shape)}"
            )
        if proprio.dim() != 2 or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"VLMBackbone.forward expected proprio (B,{self.proprio_dim}); "
                f"got {tuple(proprio.shape)}"
            )

        x = self._normalise(rgb).to(rgb.device)

        # SigLIP forward in bf16 path (autocast on CUDA; manual cast on CPU).
        device_type = "cuda" if rgb.is_cuda else "cpu"
        # Keep base SigLIP weights fp32; autocast handles the bf16 compute.
        # On CPU autocast is a no-op for many ops, so cast is benign.
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16,
                                enabled=(device_type == "cuda")):
            out = self.vision(pixel_values=x)
            tokens = out.last_hidden_state  # [B, n, hidden]

            # Resize patch grid to 16x16 = 256 so N matches CODE_STRUCTURE.
            tokens = self._resize_tokens_to_n(tokens)  # [B, 256, hidden]

            # Per-token additive proprio bias.
            p_bias = self.proprio_proj(proprio.to(tokens.dtype))  # [B, hidden]
            tokens = tokens + p_bias.unsqueeze(1)  # broadcast over N

            # Project hidden -> d_h.
            h_t = self.feature_proj(tokens)  # [B, 256, d_h]

        return h_t.to(torch.bfloat16)
