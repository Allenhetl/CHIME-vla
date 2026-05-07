"""Frozen SigLIP2 wrapper for Phase A cache (PLAN §2.2, CODE_STANDARDS §3.2).

Pipeline (per frame):
    [H, W, 3] uint8 (PyAV decode, on CPU)
        ↓ uint8/255 + HWC->CHW + move to GPU
        ↓ letterbox 224 with fill=0.5  (run on GPU — 5-10x faster than CPU)
        ↓ SigLIP2 normalize (mean=0.5, std=0.5 → [-1, 1])
        ↓ frozen SigLIP2 vision_model forward in bf16
        ↓ pooler_output (768) ⊕ last_hidden_state.mean(dim=1) (768)
        ↓ cast fp16 → write to .pt

Why GPU letterbox + skip processor: HF processor accepts CPU tensors only and
adds list/tensor coercion overhead per frame; with batch=128 that's ~30% of
wall time.  Doing the resize/pad/normalize on GPU keeps the input pipeline
saturated and idle GPU < 5%.

Why bf16 forward: Google's SigLIP2 was trained in bf16; discrepancy with fp32 < 1%
and we run ~3x faster.  Output is cast to fp16 before disk write to halve cache size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = "google/siglip2-base-patch16-224"


def _gpu_letterbox(imgs: torch.Tensor, size: int, fill: float) -> torch.Tensor:
    """Batched letterbox on the input device (CPU or GPU).

    Args:
        imgs: ``[B, 3, H, W]`` float in [0, 1].

    Returns:
        ``[B, 3, size, size]`` float in [0, 1] padded with ``fill``.
    """
    B, _, H, W = imgs.shape
    scale = size / max(H, W)
    new_h = max(1, min(int(round(H * scale)), size))
    new_w = max(1, min(int(round(W * scale)), size))
    imgs = F.interpolate(imgs, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h = size - new_h
    pad_w = size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    imgs = F.pad(imgs, (left, right, top, bottom), mode="constant", value=fill)
    return imgs


class SigLIPWrapper(nn.Module):
    """Wrap HF SigLIP2 with GPU-side preprocessing.

    Args:
        model_name:  HF repo id.
        device:      torch device for the underlying model.
        dtype:       compute dtype (default bf16).
        letterbox_fill: 0.5 = neutral grey *after* SigLIP2 normalize (PLAN §1.4).
    """

    SIGLIP_NORM_MEAN = 0.5
    SIGLIP_NORM_STD = 0.5

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        letterbox_fill: float = 0.5,
    ):
        super().__init__()
        from transformers import AutoModel

        self.model_name = model_name
        self.dtype = dtype
        self.device = torch.device(device)
        self.letterbox_fill = float(letterbox_fill)

        full = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        self.vision = full.vision_model
        self.vision.eval()
        self.vision.to(self.device)
        for p in self.vision.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward_batch(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        """Encode a batch of raw uint8 frames into 1536-d features.

        Args:
            frames_uint8: ``[B, H, W, 3]`` torch.uint8 (PyAV output, CPU tensor).

        Returns:
            ``[B, 1536]`` fp16 tensor on CPU (pooler ⊕ spatial_mean).
        """
        if frames_uint8.dtype != torch.uint8:
            raise ValueError(f"expected uint8 frames, got {frames_uint8.dtype}")
        if frames_uint8.dim() != 4 or frames_uint8.size(-1) != 3:
            raise ValueError(f"expected [B, H, W, 3], got {tuple(frames_uint8.shape)}")

        # Move to GPU first, then everything stays on device.
        x = frames_uint8.to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).contiguous().to(torch.float32) / 255.0  # [B, 3, H, W]
        x = _gpu_letterbox(x, size=224, fill=self.letterbox_fill)
        x = (x - self.SIGLIP_NORM_MEAN) / self.SIGLIP_NORM_STD  # SigLIP2 normalize
        x = x.to(self.dtype)

        out = self.vision(pixel_values=x, output_hidden_states=False)
        pooler = out.pooler_output  # [B, 768]
        spatial = out.last_hidden_state.mean(dim=1)  # [B, 768]
        feat = torch.cat([pooler, spatial], dim=-1)  # [B, 1536]
        return feat.to(torch.float16).cpu()
