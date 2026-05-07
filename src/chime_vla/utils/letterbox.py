"""Letterbox preprocessing (CODE_STANDARDS §3.2).

Pipeline (per frame):
    [3, H, W] float in [0,1]   (decord uint8 / 255, HWC->CHW done by caller)
        ↓ bilinear resize so the longer side becomes ``size``
        ↓ pad shorter side symmetrically with constant ``fill`` (default 0.5)
        ↓ [3, size, size] float in [0,1]

The fill value 0.5 is intentional: SigLIP2 normalize uses mean=std=0.5, so
0.5 maps to 0 — the most neutral grey — once normalisation runs.  Do **not**
use ImageNet (114/255).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def letterbox(img: torch.Tensor, size: int = 224, fill: float = 0.5) -> torch.Tensor:
    """Letterbox a single image to a square ``size`` while preserving aspect ratio.

    Args:
        img:  ``[3, H, W]`` float tensor in ``[0, 1]``.
        size: target side length.
        fill: constant value used for padding (default 0.5 ≡ neutral grey post SigLIP norm).

    Returns:
        ``[3, size, size]`` float tensor in ``[0, 1]``.

    Raises:
        ValueError: if ``img`` is not 3-D ``CHW``.
    """
    if img.dim() != 3 or img.size(0) != 3:
        raise ValueError(f"letterbox expects [3, H, W]; got shape {tuple(img.shape)}")

    _, h, w = img.shape
    if h <= 0 or w <= 0:
        raise ValueError(f"letterbox got non-positive H/W: {h}x{w}")

    scale = size / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    new_h = max(1, min(new_h, size))
    new_w = max(1, min(new_w, size))

    img = F.interpolate(
        img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)

    pad_h = size - new_h
    pad_w = size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    # F.pad on CHW: pad order is (left, right, top, bottom)
    img = F.pad(img, (left, right, top, bottom), mode="constant", value=fill)
    return img


def letterbox_batch(imgs: torch.Tensor, size: int = 224, fill: float = 0.5) -> torch.Tensor:
    """Apply :func:`letterbox` to a batch.

    Args:
        imgs: ``[B, 3, H, W]`` float in ``[0, 1]``.

    Returns:
        ``[B, 3, size, size]`` float in ``[0, 1]``.
    """
    if imgs.dim() != 4 or imgs.size(1) != 3:
        raise ValueError(f"letterbox_batch expects [B, 3, H, W]; got {tuple(imgs.shape)}")
    return torch.stack([letterbox(im, size=size, fill=fill) for im in imgs], dim=0)
