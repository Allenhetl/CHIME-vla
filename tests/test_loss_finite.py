"""4-step toy batch — all five losses must be finite, non-NaN, non-Inf.

Contract (CODE_STANDARDS §4): xfail at M0 (loss bodies not yet wired); strict
from M1.
"""

from __future__ import annotations

import pytest
import torch


M0_XFAIL = pytest.mark.xfail(
    reason="M0: loss bodies are stubs; will green at M1",
    strict=False,
)


@M0_XFAIL
def test_5_losses_finite_on_4_step_batch(mock_chime_config, synthetic_batch_with_gamma):
    """Run `chime_train_step` for 4 frames and check each loss term."""
    pytest.importorskip("chime_vla")
    from chime_vla.training.train_step import chime_train_step
    from chime_vla.training.lightning_module import ChimeVlaLightning

    cfg = mock_chime_config
    cfg.hindsight.enabled = True
    cfg.loss.lambda_1_target = 0.3
    cfg.loss.lambda_2 = 0.5
    cfg.loss.lambda_3 = 0.1
    model = ChimeVlaLightning(cfg)

    # Truncate batch to T=4
    batch = {k: (v[:, :4] if v.dim() >= 2 and k != "episode_id" else v)
             for k, v in synthetic_batch_with_gamma.items()}
    out = chime_train_step(batch, model, cfg, step=0)

    for key in ("L_main", "L_HCS", "L_PRH", "L_CSM", "L_aux", "total"):
        v = out[key]
        assert torch.isfinite(v).all(), f"{key} not finite: {v}"
