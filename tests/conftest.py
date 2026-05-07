"""Shared pytest fixtures for CHIME-VLA.

Fixtures here are intentionally lightweight: they construct synthetic batches
and minimal config / model instances so component contracts can be tested
without LIBERO data on disk and without GPU.

Until `src/chime_vla/` lands in full, fixtures that need real component classes
(`mock_chime_model`) gracefully `pytest.importorskip` so the suite still loads.
See CODE_STRUCTURE.md §8 (M0 deliverable checklist).
"""

from __future__ import annotations

import torch
import pytest


# -----------------------------------------------------------------------------
# Synthetic batch (no GPU, no LIBERO files needed)
# -----------------------------------------------------------------------------

@pytest.fixture
def synthetic_batch() -> dict[str, torch.Tensor]:
    """Tiny synthetic batch matching the schema from CODE_STRUCTURE.md §5.

    Shapes:
        rgb         : (B=2, T=8, 3, 224, 224) float32 in [0, 1]
        proprio     : (B=2, T=8, 8) float32
        action      : (B=2, T=8, 8) float32
        sub_task_id : (B=2, T=8) int32
        valid_mask  : (B=2, T=8) bool, all True
        episode_id  : (B=2,) int64
    """
    B, T = 2, 8
    g = torch.Generator().manual_seed(0)
    return {
        "rgb": torch.rand(B, T, 3, 224, 224, generator=g),
        "proprio": torch.randn(B, T, 8, generator=g),
        "action": torch.randn(B, T, 8, generator=g),
        "sub_task_id": torch.zeros(B, T, dtype=torch.int32),
        "valid_mask": torch.ones(B, T, dtype=torch.bool),
        "episode_id": torch.tensor([0, 1], dtype=torch.int64),
    }


@pytest.fixture
def synthetic_batch_with_gamma(synthetic_batch) -> dict[str, torch.Tensor]:
    """Same as `synthetic_batch` plus γ̂_geo / γ̂_sem labels (Hindsight enabled)."""
    B, T = synthetic_batch["rgb"].shape[:2]
    g = torch.Generator().manual_seed(1)
    out = dict(synthetic_batch)
    out["gamma_hat_geo"] = torch.rand(B, T, generator=g)
    out["gamma_hat_sem"] = torch.rand(B, T, generator=g)
    return out


# -----------------------------------------------------------------------------
# Config / model fixtures (skip if src/ not yet present)
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_chime_config():
    """Minimal `ChimeConfig` instance with all dataclass defaults.

    Skips the test if `chime_vla` is not importable yet (e.g., during the
    early M0 implementation window where `src/chime_vla/` is being filled in
    by a sibling agent).
    """
    chime_vla = pytest.importorskip("chime_vla")
    from chime_vla.config import ChimeConfig  # noqa: WPS433 — lazy import on purpose
    return ChimeConfig()


@pytest.fixture
def mock_chime_model(mock_chime_config):
    """Instantiate the 13 component stubs against a minimal config.

    Returns a `SimpleNamespace` whose attributes are the component instances
    (rather than a fully-wired Lightning module) so tests can hit components
    one at a time.  Falls back to `pytest.importorskip` if the stubs do not
    yet exist on disk.
    """
    pytest.importorskip("chime_vla")

    from types import SimpleNamespace

    cfg = mock_chime_config

    # Lazy imports so a missing stub only skips the dependent test.
    try:
        from chime_vla.perception.vlm_backbone import VLMBackbone
        from chime_vla.perception.fifo_buffer import WorkBuffer
        from chime_vla.heads.geo_write import GeoWriteHead
        from chime_vla.heads.sem_write import SemWriteHead
        from chime_vla.heads.espc import ESPC
        from chime_vla.memory.geo_grid import GeoGrid
        from chime_vla.memory.sem_bank import SemBank
        from chime_vla.readout.read_interface import ReadInterface
        from chime_vla.action.action_expert import ActionExpert
        from chime_vla.heads.prh import PRH
        from chime_vla.heads.csm import CSM
    except ImportError as exc:
        pytest.skip(f"chime_vla stubs not yet available: {exc}")

    device = torch.device("cpu")
    B = 2

    return SimpleNamespace(
        cfg=cfg,
        c1=VLMBackbone(cfg.c1),
        c2=WorkBuffer(cfg.c2, batch_size=B, device=device),
        c5=ESPC(cfg.c5, d_h=cfg.c2.d_h),
        c3=GeoWriteHead(cfg.c3, d_h=cfg.c2.d_h, d_g=cfg.c6.d_g, alpha_l=cfg.c6.alpha_l),
        c4=SemWriteHead(cfg.c4, d_h=cfg.c2.d_h, d_s=cfg.c7.d_s, K_s=cfg.c7.K_s),
        c6=GeoGrid(cfg.c6, batch_size=B, d_g=cfg.c6.d_g, device=device),
        c7=SemBank(cfg.c7, batch_size=B, device=device),
        c8=ReadInterface(
            cfg.c8, d_h=cfg.c2.d_h, d_s=cfg.c7.d_s, K_w=cfg.c2.K_w, K_s=cfg.c7.K_s
        ),
        c9=ActionExpert(cfg.c9, d_h=cfg.c2.d_h, action_dim=cfg.data.action_dim),
        c11=PRH(cfg.c11, d_h=cfg.c2.d_h, action_dim=cfg.data.action_dim),
        c12=CSM(cfg.c12),
    )
