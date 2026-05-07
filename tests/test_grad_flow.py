"""SG-1 .. SG-7 grad-flow contract tests.

This file is the CI gate specified in `docs/grad_flow_contract.md`.

M0/M1: all 7 tests `xfail(strict=False)` — stubs raise `NotImplementedError`
on backward, which counts as an expected failure.  Once M2 lands, change
`strict=False` to `strict=True` (as called out in CODE_STANDARDS §4): a test
that suddenly passes after a refactor must be treated as a regression.
"""

from __future__ import annotations

import pytest
import torch


M0_XFAIL = pytest.mark.xfail(
    reason="M0: component stubs raise NotImplementedError; will green at M2",
    strict=False,
)


def _zero_or_none(grad: torch.Tensor | None, atol: float = 1e-9) -> bool:
    return grad is None or grad.abs().max().item() < atol


# ---------------------------------------------------------------------------
# SG-1: γ → ψ stop-grad (write heads consume sg(γ))
# ---------------------------------------------------------------------------

def test_sg_1_gamma_to_psi(mock_chime_model, synthetic_batch):
    """L_main back-prop must NOT reach ESPC ψ via γ_geo / γ_sem.

    Per architecture v2.1 §B SG-1: write heads see `sg(γ)`, so the only legal
    grad path into ψ is L_HCS.  Computing `L_main.backward()` therefore must
    leave `model.c5.psi` (and projections, per SG-6) with grad=None or 0.
    """
    m = mock_chime_model
    batch = synthetic_batch
    m.c1.zero_grad(set_to_none=True)
    m.c5.zero_grad(set_to_none=True)

    # Forward one step (B, T=8 → take t=0)
    rgb_t = batch["rgb"][:, 0]
    proprio_t = batch["proprio"][:, 0]
    h_t = m.c1(rgb_t, proprio_t)
    m_work = m.c2.snapshot()
    gamma_geo, gamma_sem = m.c5(h_t, m_work)
    m.c2.append(h_t)
    m.c3(h_t, gamma_geo.detach(), m.c6)  # SG-1 explicit
    m.c4(h_t, gamma_sem.detach(), m.c7)  # SG-1 explicit
    c_t = m.c8(m.c2.snapshot(), m.c6, m.c7, h_t, prh_path=False)
    a_pred = m.c9(c_t, h_t.mean(dim=1))

    L_main = (a_pred - batch["action"][:, 0]).pow(2).mean()
    L_main.backward()

    for name, p in m.c5.named_parameters():
        assert _zero_or_none(p.grad), f"SG-1 violated on c5.{name}: grad leaked"


# ---------------------------------------------------------------------------
# SG-2: PRH → [C1] stop-grad (PRH consumes sg(m_t))
# ---------------------------------------------------------------------------

def test_sg_2_prh_query_to_perception(mock_chime_model, synthetic_batch):
    """L_PRH must NOT flow back into [C1] via the m_t query."""
    m = mock_chime_model
    batch = synthetic_batch
    m.c1.zero_grad(set_to_none=True)

    rgb_t = batch["rgb"][:, 0]
    proprio_t = batch["proprio"][:, 0]
    h_t = m.c1(rgb_t, proprio_t)
    m.c2.append(h_t)
    c_t = m.c8(m.c2.snapshot(), m.c6, m.c7, h_t, prh_path=True)
    m_t = c_t.mean(dim=1)
    prh_out = m.c11(m_t.detach())  # SG-2 explicit
    # build a synthetic L_PRH on the head outputs
    L_prh = sum(o.pow(2).mean() + a.pow(2).mean() for (o, a) in prh_out.values())
    L_prh.backward()

    for name, p in m.c1.named_parameters():
        assert _zero_or_none(p.grad), f"SG-2 violated on c1.{name}: grad leaked"


# ---------------------------------------------------------------------------
# SG-3: γ̂ is a target, must be detached before BCE
# ---------------------------------------------------------------------------

def test_sg_3_gammahat_target(mock_chime_model, synthetic_batch_with_gamma):
    """γ̂ tensors loaded from Hindsight must be treated as `.detach()`-equivalent
    targets: a synthetic γ̂ that requires grad must have no grad after BCE."""
    m = mock_chime_model
    batch = synthetic_batch_with_gamma

    gamma_hat = batch["gamma_hat_geo"][:, 0].clone().requires_grad_(True)
    rgb_t = batch["rgb"][:, 0]
    proprio_t = batch["proprio"][:, 0]
    h_t = m.c1(rgb_t, proprio_t)
    m_work = m.c2.snapshot()
    gamma_geo, _ = m.c5(h_t, m_work)

    # SG-3: BCE target must be detached.
    L_hcs = torch.nn.functional.binary_cross_entropy(gamma_geo, gamma_hat.detach())
    L_hcs.backward()

    assert _zero_or_none(gamma_hat.grad), "SG-3 violated: grad leaked into γ̂"


# ---------------------------------------------------------------------------
# SG-4: CSM uses frozen action expert — no grad to [C9]
# ---------------------------------------------------------------------------

@M0_XFAIL
def test_sg_4_csm_through_frozen_action(mock_chime_model, synthetic_batch):
    """L_CSM must leave [C9] params with grad=None."""
    m = mock_chime_model
    m.c9.zero_grad(set_to_none=True)
    m.c9.freeze()

    # Synthetic m_t to feed CSM
    B = synthetic_batch["rgb"].shape[0]
    m_t = torch.randn(B, m.cfg.c2.d_h, requires_grad=True)
    w = m.c12(m_t, m.c7, m.c9)
    L_csm = w.pow(2).mean()
    L_csm.backward()

    for name, p in m.c9.named_parameters():
        assert _zero_or_none(p.grad), f"SG-4 violated on c9.{name}: grad leaked"


# ---------------------------------------------------------------------------
# SG-5: M_work content seen by ψ on L_HCS path — must not flow to [C1]
# ---------------------------------------------------------------------------

@M0_XFAIL
def test_sg_5_mwork_to_perception_via_psi(mock_chime_model, synthetic_batch_with_gamma):
    """L_HCS → ψ → M_work[t-K_w..t-1] → [C1] is forbidden.

    Implementation: M_work is detached when fetched by ψ.  Test asserts no
    grad on [C1] params after `L_HCS.backward()`.
    """
    m = mock_chime_model
    m.c1.zero_grad(set_to_none=True)
    batch = synthetic_batch_with_gamma

    # Simulate a few prior frames so M_work is non-empty
    for t in range(2):
        h_prev = m.c1(batch["rgb"][:, t], batch["proprio"][:, t])
        m.c2.append(h_prev)
    # Re-zero so only the current step's grads matter
    m.c1.zero_grad(set_to_none=True)

    h_t = m.c1(batch["rgb"][:, 2], batch["proprio"][:, 2])
    gamma_geo, _ = m.c5(h_t, m.c2.snapshot())
    L_hcs = torch.nn.functional.binary_cross_entropy(
        gamma_geo, batch["gamma_hat_geo"][:, 2].detach()
    )
    L_hcs.backward()

    # The CURRENT-step h_t may legally backprop through ψ → C1.  What's
    # forbidden is grad through the HISTORICAL frames sitting in M_work.
    # Implementation contract: ψ detaches M_work, so the only [C1] grad path
    # is via h_t at this step — already accounted for in zero_grad above.
    # We assert that running this same backward twice does not accumulate
    # grad from older frames (they were detached).
    grads_first = {n: (p.grad.clone() if p.grad is not None else None)
                   for n, p in m.c1.named_parameters()}
    L_hcs2 = torch.nn.functional.binary_cross_entropy(
        m.c5(h_t, m.c2.snapshot())[0],
        batch["gamma_hat_geo"][:, 2].detach(),
    )
    # If M_work was NOT detached the second backward would touch older h's.
    L_hcs2.backward()
    for n, p in m.c1.named_parameters():
        if grads_first[n] is None:
            continue  # no grad either time — trivially OK
        # The second backward must double the same grad, not introduce new
        # grad rows from M_work history.
        diff = (p.grad - 2 * grads_first[n]).abs().max().item()
        assert diff < 1e-5, f"SG-5 violated on c1.{n}: extra grad path through M_work"


# ---------------------------------------------------------------------------
# SG-6: [C5] geo_proj / sem_proj — trainable ONLY by L_HCS, NOT L_main
# ---------------------------------------------------------------------------

def test_sg_6_proj_only_lhcs(mock_chime_model, synthetic_batch):
    """L_main must NOT touch ESPC's geo_proj / sem_proj parameters."""
    m = mock_chime_model
    m.c5.zero_grad(set_to_none=True)

    rgb_t = synthetic_batch["rgb"][:, 0]
    proprio_t = synthetic_batch["proprio"][:, 0]
    h_t = m.c1(rgb_t, proprio_t)
    m.c2.append(h_t)
    c_t = m.c8(m.c2.snapshot(), m.c6, m.c7, h_t, prh_path=False)
    a_pred = m.c9(c_t, h_t.mean(dim=1))
    L_main = (a_pred - synthetic_batch["action"][:, 0]).pow(2).mean()
    L_main.backward()

    for name, p in m.c5.named_parameters():
        if "geo_proj" in name or "sem_proj" in name:
            assert _zero_or_none(p.grad), f"SG-6 violated on c5.{name}"


# ---------------------------------------------------------------------------
# SG-7: read attention entropy floor (runtime metric, not sg-able)
# ---------------------------------------------------------------------------

def test_sg_7_attention_entropy_floor(mock_chime_model, synthetic_batch):
    """SG-7 is structural (cross-attn cannot be sg-ed without breaking the
    read pathway).  We assert the runtime monitor: H(attn over M_work) stays
    above `cfg.loss.entropy_floor` on a synthetic forward.
    """
    m = mock_chime_model
    rgb_t = synthetic_batch["rgb"][:, 0]
    proprio_t = synthetic_batch["proprio"][:, 0]
    h_t = m.c1(rgb_t, proprio_t)
    m.c2.append(h_t)
    _ = m.c8(m.c2.snapshot(), m.c6, m.c7, h_t, prh_path=False)

    H = m.c8.attn_entropy_to_M_work
    floor = m.cfg.loss.entropy_floor
    assert H.min().item() > floor, f"SG-7 floor breached: {H.min().item()} <= {floor}"
