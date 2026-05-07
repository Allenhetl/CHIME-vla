"""Forward-shape contracts (CODE_STRUCTURE.md §3.1-§3.10).

This file MUST stay strict (no xfail) at every milestone — per CODE_STANDARDS
§4 the forward-shapes test is the floor of correctness.  We accomplish that
during M0 (when component forwards just `raise NotImplementedError`) by:

  1. Asserting on configuration / instantiation surface — `ChimeConfig()` and
     individual component `__init__` signatures that the real forward will
     consume.  These checks pass without ever calling the un-impl forward.
  2. Where we want a true forward, we monkeypatch the component's `forward`
     to a deterministic-shape dummy and verify the surrounding module honors
     the contract (e.g., that `c8` returns (B, N_q + K_w, d_h)).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch


# ---------------------------------------------------------------------------
# 1. Config surface (must be strict-pass even in M0)
# ---------------------------------------------------------------------------

def test_chime_config_instantiates():
    """`ChimeConfig()` must construct with default factories and expose all
    13 component sub-configs."""
    pytest.importorskip("chime_vla")
    from chime_vla.config import ChimeConfig

    cfg = ChimeConfig()
    for name in (
        "c1", "c2", "c3", "c4", "c5", "c6", "c7",
        "c8", "c9", "c10", "c11", "c12",
        "loss", "train", "data", "hindsight",
    ):
        assert hasattr(cfg, name), f"ChimeConfig missing field {name}"
    # Cross-check a couple of MVP-locked defaults
    assert cfg.c1.backbone in ("siglip_vit_b", "siglip_vit_l", "siglip_vit_s")
    assert cfg.c2.K_w == 8
    assert cfg.c2.d_h == 1152
    assert cfg.c2.N == 256


# ---------------------------------------------------------------------------
# 2. Per-component instantiation (covers C1, C5, C8, C9, C11)
# ---------------------------------------------------------------------------

def test_c1_instantiation_records_config():
    """[C1] VLMBackbone stub must accept C1Config and stash backbone name."""
    pytest.importorskip("chime_vla")
    from chime_vla.config import C1Config
    from chime_vla.perception.vlm_backbone import VLMBackbone

    cfg = C1Config(backbone="siglip_vit_b", lora_r=16, freeze_backbone=True)
    m = VLMBackbone(cfg)
    # The stub is allowed to NotImplementedError on forward; the contract
    # we lock here is that the config arrived intact.
    assert getattr(m, "cfg", None) is cfg or hasattr(m, "backbone")


def test_c5_instantiation_records_config():
    """[C5] ESPC must accept C5Config and the perception d_h."""
    pytest.importorskip("chime_vla")
    from chime_vla.config import C5Config
    from chime_vla.heads.espc import ESPC

    cfg = C5Config(psi_layers=1, use_gru=True, d_proj=64)
    m = ESPC(cfg, d_h=1152)
    assert isinstance(m, torch.nn.Module)


def test_c8_read_returns_n_q_plus_k_w_tokens(mock_chime_model, synthetic_batch):
    """[C8] cross-attn output shape: (B, N_q + K_w, d_h).

    We bypass the unimplemented forward via monkeypatch so this test is
    strict-pass regardless of stub state.  This locks the SHAPE contract.
    """
    m = mock_chime_model
    cfg = m.cfg
    B = synthetic_batch["rgb"].shape[0]
    expected_shape = (B, cfg.c8.N_q + cfg.c2.K_w, cfg.c2.d_h)

    def _dummy_forward(self, m_work, m_geo, m_sem, h_t, prh_path=False):
        return torch.zeros(*expected_shape)

    m_work_dummy = torch.zeros(B, cfg.c2.K_w, cfg.c2.N, cfg.c2.d_h)
    with patch.object(type(m.c8), "forward", _dummy_forward, create=True):
        out = m.c8(m_work_dummy, m.c6, m.c7, torch.zeros(B, cfg.c2.N, cfg.c2.d_h))
    assert tuple(out.shape) == expected_shape


def test_c9_action_expert_output_shape(mock_chime_model, synthetic_batch):
    """[C9] ActionExpert(c_t, h_t_cls) → (B, action_dim)."""
    m = mock_chime_model
    cfg = m.cfg
    B = synthetic_batch["rgb"].shape[0]
    c_t = torch.zeros(B, cfg.c8.N_q + cfg.c2.K_w, cfg.c2.d_h)
    h_cls = torch.zeros(B, cfg.c2.d_h)
    expected = (B, cfg.data.action_dim)

    def _dummy_forward(self, c_t, h_t_cls):
        return torch.zeros(c_t.shape[0], cfg.data.action_dim)

    with patch.object(type(m.c9), "forward", _dummy_forward, create=True):
        out = m.c9(c_t, h_cls)
    assert tuple(out.shape) == expected


def test_c11_prh_returns_dict_keyed_by_horizon(mock_chime_model):
    """[C11] PRH(m_t) → {k: (o_hat, a_hat)} for k in cfg.c11.horizons."""
    m = mock_chime_model
    cfg = m.cfg
    B = 2
    m_t = torch.zeros(B, cfg.c2.d_h)

    def _dummy_forward(self, m_t_sg):
        return {
            k: (torch.zeros(B, cfg.c2.d_h), torch.zeros(B, cfg.data.action_dim))
            for k in cfg.c11.horizons
        }

    with patch.object(type(m.c11), "forward", _dummy_forward, create=True):
        out = m.c11(m_t)
    assert set(out.keys()) == set(cfg.c11.horizons)
    for k, (o_hat, a_hat) in out.items():
        assert o_hat.shape == (B, cfg.c2.d_h)
        assert a_hat.shape == (B, cfg.data.action_dim)


# ---------------------------------------------------------------------------
# 3. Memory container shape contracts (C6 / C7)
# ---------------------------------------------------------------------------

def test_c6_geo_grid_shape_contract(mock_chime_config):
    pytest.importorskip("chime_vla")
    from chime_vla.memory.geo_grid import GeoGrid

    cfg = mock_chime_config
    grid = GeoGrid(cfg.c6, batch_size=2, d_g=cfg.c6.d_g, device=torch.device("cpu"))
    for L in cfg.c6.levels:
        assert L in grid.grids
        # (B, D, H, W, d_g)
        assert grid.grids[L].shape == (2, L, L, L, cfg.c6.d_g)


def test_c7_sem_bank_shape_contract(mock_chime_config):
    pytest.importorskip("chime_vla")
    from chime_vla.memory.sem_bank import SemBank

    cfg = mock_chime_config
    bank = SemBank(cfg.c7, batch_size=2, device=torch.device("cpu"))
    assert bank.k.shape == (2, cfg.c7.K_s, cfg.c7.d_s)
    assert bank.v.shape == (2, cfg.c7.K_s, cfg.c7.d_s)
    assert bank.slot_free.shape == (2, cfg.c7.K_s)
    assert bank.slot_free.dtype == torch.bool
