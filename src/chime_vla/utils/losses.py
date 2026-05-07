"""Masked MSE — only valid frames contribute (PLAN §3.3.1, CODE_STANDARDS §5.4).

Mask convention is the codebase-wide rule: ``True = real frame, False = padding``.
"""

from __future__ import annotations

import torch


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean-squared error averaged over real frames only.

    Args:
        pred:   ``[B, T, D]`` predictions.
        target: ``[B, T, D]`` targets.
        mask:   ``[B, T]`` bool, ``True`` = real frame.  ``None`` ≡ all real.

    Returns:
        Scalar loss.  Equals ``(diff² × mask) / (mask.sum() × D)``; zero
        denominator returns 0 to avoid NaN propagating into the optimiser
        when an entire batch is padding (should not happen but defensive).
    """
    diff_sq = (pred - target).pow(2)
    if mask is None:
        return diff_sq.mean()

    if mask.dim() != diff_sq.dim() - 1:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} incompatible with pred {tuple(pred.shape)}"
        )

    mask_f = mask.unsqueeze(-1).to(diff_sq.dtype)
    num = (diff_sq * mask_f).sum()
    denom = mask_f.sum() * diff_sq.size(-1)
    if denom <= 0:
        return diff_sq.new_zeros(())
    return num / denom
