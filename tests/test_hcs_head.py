"""Unit tests for the full [C10] HCS-H module.

Covers:
  - ``GradCamDecomposer.decompose`` shape contract
  - ``RudderLSTM`` forward + backward + per-frame contribution
  - ``HCSHead.compute`` end-to-end smoke against a tiny stub base policy

The stub base policy mimics the surface used by ``HCSHead._forward_segment``
(``c1, c2, c3, c4, c5, c6, c7, c8, c9, cfg``) but is far cheaper than the
real CHIME backbone — keeps the test under 5 s on CPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. GradCamDecomposer
# ---------------------------------------------------------------------------


def test_gradcam_decomposer_shapes():
    from chime_vla.training.hcs_head import GradCamDecomposer

    J = torch.randn(5, 3, 8, 8).abs()
    geo, sem = GradCamDecomposer.decompose(J)
    assert geo.shape == (5,)
    assert sem.shape == (5,)
    assert geo.dtype == torch.float32
    assert sem.dtype == torch.float32
    # Both pooling streams must be non-negative on a non-negative input.
    assert (geo >= 0).all()
    assert (sem >= 0).all()


def test_gradcam_decomposer_rejects_wrong_rank():
    from chime_vla.training.hcs_head import GradCamDecomposer

    bad = torch.randn(5, 3, 8)        # missing W
    with pytest.raises(ValueError):
        GradCamDecomposer.decompose(bad)


def test_gradcam_decomposer_geo_responds_to_localised_peak():
    """A frame with a single bright pixel should have higher J_geo
    than a frame with a uniform dim background of equal mean."""
    from chime_vla.training.hcs_head import GradCamDecomposer

    J = torch.zeros(2, 3, 8, 8)
    # frame 0: one bright pixel
    J[0, :, 0, 0] = 10.0
    # frame 1: same total mass spread uniformly  (mass = 30 across 64 cells)
    J[1] = 30.0 / (3 * 8 * 8)
    geo, _ = GradCamDecomposer.decompose(J)
    assert geo[0] > geo[1]


# ---------------------------------------------------------------------------
# 2. RudderLSTM
# ---------------------------------------------------------------------------


def test_rudder_lstm_forward_shape_and_backward():
    from chime_vla.training.hcs_head import RudderLSTM

    rudder = RudderLSTM(d_feat=64, d_hidden=32)
    feat = torch.randn(2, 20, 64, requires_grad=False)
    out = rudder(feat)
    assert out.shape == (2, 20)
    # Backward path through BCE-with-logits target.
    target = torch.zeros(2, 20)
    target[:, -1] = 1.0
    loss = F.binary_cross_entropy_with_logits(out, target)
    loss.backward()
    # Sanity: head's weight got a non-zero grad.
    assert rudder.head.weight.grad is not None
    assert rudder.head.weight.grad.abs().sum().item() > 0


def test_rudder_per_frame_contribution_shape_and_range():
    from chime_vla.training.hcs_head import RudderLSTM

    rudder = RudderLSTM(d_feat=16, d_hidden=16)
    feat = torch.randn(3, 12, 16)
    delta = rudder.per_frame_contribution(feat)
    assert delta.shape == (3, 12)
    # |Δp| where p ∈ [0,1] ⇒ in [0, 1].
    assert (delta >= 0).all()
    assert (delta <= 1).all()


def test_rudder_lstm_trains_to_separate_terminal_signal():
    """Smoke-train RUDDER for 30 epochs on synthetic data: loss must drop."""
    from chime_vla.training.hcs_head import RudderLSTM

    torch.manual_seed(0)
    rudder = RudderLSTM(d_feat=16, d_hidden=32)
    feat = torch.randn(4, 16, 16)
    target = torch.zeros(4, 16)
    target[:, -1] = 1.0  # success only at terminal frame, cumulative.
    opt = torch.optim.AdamW(rudder.parameters(), lr=1e-2)
    losses: list[float] = []
    for _ in range(30):
        pred = rudder(feat)
        loss = F.binary_cross_entropy_with_logits(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0], (
        f"RUDDER LSTM failed to train: first={losses[0]:.4f}, "
        f"last={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. HCSHead — end-to-end smoke against a minimal stub policy
# ---------------------------------------------------------------------------


class _StubC1(nn.Module):
    """Tiny ViT-stand-in: maps RGB+proprio to (B, N, d_h) tokens."""

    def __init__(self, d_h: int = 16, n_tokens: int = 4):
        super().__init__()
        self.d_h = d_h
        self.n_tokens = n_tokens
        # 3*8*8 spatial pool to get a small feature.  AvgPool to 4x4 → 48 -> proj.
        self.proj = nn.Linear(3 * 4 * 4 + 8, d_h * n_tokens)

    def forward(self, rgb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        # rgb: (B, 3, H, W).  Adaptive pool to (B, 3, 4, 4) for any input H, W.
        x = F.adaptive_avg_pool2d(rgb, (4, 4)).flatten(1)   # (B, 48)
        x = torch.cat([x, proprio], dim=1)                  # (B, 48 + 8)
        out = self.proj(x).reshape(rgb.shape[0], self.n_tokens, self.d_h)
        return out


class _StubC5(nn.Module):
    """ESPC stub: returns (γ_geo, γ_sem) of shape (B,) ∈ (0,1)."""

    def __init__(self, d_h: int = 16):
        super().__init__()
        self.gate = nn.Linear(d_h, 2)

    def forward(self, h_t, m_work_prev):
        # h_t: (B, N, d_h)
        x = h_t.mean(dim=1)
        g = torch.sigmoid(self.gate(x))
        return g[:, 0], g[:, 1]


class _StubMemoryHead(nn.Module):
    """No-op write head: receives (h, gamma, mem) but mutates nothing."""

    def forward(self, h, gamma, mem, step: int = 0):  # noqa: D401 — stub
        return None

    def __call__(self, *args, **kwargs):  # mirror nn.Module call semantics
        return self.forward(*args, **kwargs)


class _StubC8(nn.Module):
    """Read interface stub: returns context vector (B, N, d_h)."""

    def __init__(self, d_h: int = 16):
        super().__init__()
        self.proj = nn.Linear(d_h, d_h)

    def forward(self, m_work, m_geo, m_sem, h_t):
        # Just project h_t through a linear; no actual memory read in stub.
        return self.proj(h_t)


class _StubC9(nn.Module):
    """Action expert stub: pools context → 8-dim action."""

    def __init__(self, d_h: int = 16, action_dim: int = 8):
        super().__init__()
        self.head = nn.Linear(d_h, action_dim)

    def forward(self, c_t, h_cls):
        return self.head(c_t.mean(dim=1) + h_cls)


class _StubMemory:
    """Bare-bones placeholder that has just enough surface to be passed
    through the stub heads (which ignore it)."""

    def __init__(self):
        pass


class _StubBasePolicy(nn.Module):
    """Mimics ChimeVlaLightning's outward attributes used by HCSHead."""

    def __init__(self, d_h: int = 16, n_tokens: int = 4):
        super().__init__()
        self.c1 = _StubC1(d_h=d_h, n_tokens=n_tokens)
        self.c5 = _StubC5(d_h=d_h)
        self.c3 = _StubMemoryHead()
        self.c4 = _StubMemoryHead()
        self.c8 = _StubC8(d_h=d_h)
        self.c9 = _StubC9(d_h=d_h, action_dim=8)
        # Minimal config the segment-forward expects to read.
        self.cfg = SimpleNamespace(
            c2=SimpleNamespace(),
            c6=SimpleNamespace(d_g=8),
            c7=SimpleNamespace(),
        )


class _MockWorkBuffer:
    """Drop-in stand-in for ``WorkBuffer`` used by the segment forward.

    Mirrors the attributes that ``HCSHead._forward_segment`` reads:
    ``buffer`` (the FIFO tensor), ``K_w`` (length), ``_n_appended``, and
    a ``snapshot()`` method.  No real ringing — we only need
    ``snapshot()`` to return a tensor the stub C5 can ignore.
    """

    def __init__(self, cfg, batch_size: int = 1, device=None):
        del cfg
        self.K_w = 4
        self.buffer = torch.zeros(batch_size, self.K_w, 4, 16, device=device)
        self._n_appended = torch.zeros(batch_size, dtype=torch.long, device=device)

    def snapshot(self):
        return self.buffer


@pytest.fixture
def patched_segment_helpers(monkeypatch):
    """Patch the lazy imports inside HCSHead._forward_segment so the test
    doesn't pull in the real GeoGrid / SemBank / WorkBuffer (heavy)."""
    import sys
    # Stub modules that mirror the imports inside HCSHead._forward_segment.
    fake_fifo = SimpleNamespace(WorkBuffer=_MockWorkBuffer)
    fake_geo = SimpleNamespace(GeoGrid=lambda *a, **kw: _StubMemory())
    fake_sem = SimpleNamespace(SemBank=lambda *a, **kw: _StubMemory())
    monkeypatch.setitem(sys.modules, "chime_vla.perception.fifo_buffer", fake_fifo)
    monkeypatch.setitem(sys.modules, "chime_vla.memory.geo_grid", fake_geo)
    monkeypatch.setitem(sys.modules, "chime_vla.memory.sem_bank", fake_sem)
    yield


def test_hcs_head_compute_smoke(patched_segment_helpers):
    from chime_vla.training.hcs_head import HCSHead, RudderLSTM

    torch.manual_seed(0)
    base = _StubBasePolicy(d_h=16, n_tokens=4)
    base.eval()

    T = 8
    rgb = torch.rand(T, 3, 16, 16)
    proprio = torch.randn(T, 8)
    action = torch.randn(T, 8)
    rewards = torch.zeros(T)
    rewards[-1] = 1.0  # cumulative success target — used by RUDDER

    rudder = RudderLSTM(d_feat=16, d_hidden=16)
    head = HCSHead(
        base_policy=base,
        deltas=(2, 4),     # < T=8 — must satisfy T > delta_max
        rudder=rudder,
        alpha_J=1.0,
        alpha_R=0.5,
        device="cpu",
    )
    out = head.compute(rgb, proprio, action, rewards)

    # Schema contract — all required keys present.
    for key in (
        "gamma_geo",
        "gamma_sem",
        "J_geo_raw",
        "J_sem_raw",
        "rudder_delta",
        "meta",
    ):
        assert key in out, f"missing key {key!r}"

    assert out["gamma_geo"].shape == (T,)
    assert out["gamma_sem"].shape == (T,)
    assert out["J_geo_raw"].shape == (T,)
    assert out["J_sem_raw"].shape == (T,)
    assert out["rudder_delta"].shape == (T,)
    # Range — gamma_* are sigmoid outputs.
    assert (out["gamma_geo"] >= 0).all() and (out["gamma_geo"] <= 1).all()
    assert (out["gamma_sem"] >= 0).all() and (out["gamma_sem"] <= 1).all()
    # Meta keys per docs/hindsight_contract.md §3.
    assert out["meta"]["delta_set"] == [2, 4]
    assert out["meta"]["rudder_attached"] is True
    assert out["meta"]["n_pairs"] > 0


def test_hcs_head_compute_without_rudder(patched_segment_helpers):
    """Without RUDDER attached, rudder_delta is all-zero and γ̂ depends on J only."""
    from chime_vla.training.hcs_head import HCSHead

    torch.manual_seed(0)
    base = _StubBasePolicy(d_h=16, n_tokens=4)
    base.eval()
    T = 6
    head = HCSHead(
        base_policy=base, deltas=(2,), rudder=None,
        alpha_J=1.0, alpha_R=0.0, device="cpu",
    )
    out = head.compute(
        torch.rand(T, 3, 16, 16),
        torch.randn(T, 8),
        torch.randn(T, 8),
        reward_seq=None,
    )
    assert (out["rudder_delta"] == 0).all()
    assert out["meta"]["rudder_attached"] is False


def test_hcs_head_rejects_short_trajectory():
    from chime_vla.training.hcs_head import HCSHead

    base = _StubBasePolicy(d_h=16, n_tokens=4)
    head = HCSHead(base_policy=base, deltas=(4, 16), rudder=None, device="cpu")
    # T=8 < delta_max=16 should raise.
    with pytest.raises(ValueError, match="must exceed max delta"):
        head.compute(
            torch.rand(8, 3, 16, 16),
            torch.randn(8, 8),
            torch.randn(8, 8),
            torch.zeros(8),
        )


# ---------------------------------------------------------------------------
# 4. fit_rudder smoke (synthetic only — F7-Phase-2 will run on real LIBERO)
# ---------------------------------------------------------------------------


def test_hcs_head_fit_rudder_synthetic():
    from chime_vla.training.hcs_head import HCSHead, RudderLSTM

    torch.manual_seed(0)
    base = _StubBasePolicy(d_h=16, n_tokens=4)
    rudder = RudderLSTM(d_feat=16, d_hidden=32)
    head = HCSHead(base_policy=base, deltas=(2,), rudder=rudder, device="cpu")

    # 4 synthetic episodes of variable length T_i, all with sparse terminal reward.
    feats: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    for T_i in [10, 12, 14, 12]:
        feats.append(torch.randn(T_i, 16))
        r = torch.zeros(T_i)
        r[-1] = 1.0
        # cumulative
        rewards.append(r.cumsum(0).clamp(max=1.0))

    stats = head.fit_rudder(feats, rewards, epochs=20, lr=1e-2)
    assert stats["loss_last"] < stats["loss_first"]
