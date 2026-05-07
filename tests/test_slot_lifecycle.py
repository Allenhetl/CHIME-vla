"""SemBank slot-lifecycle invariants (CODE_STANDARDS §1.9).

All tests xfail at M0 (SemBank stub has no real evict path yet); they go
strict at M2.
"""

from __future__ import annotations

import pytest
import torch


M0_XFAIL = pytest.mark.xfail(
    reason="M0: SemBank stub not impl; lifecycle invariants land at M2",
    strict=False,
)


@M0_XFAIL
def test_initial_state_all_free(mock_chime_config):
    """A fresh SemBank must have slot_free=True for every slot, every batch."""
    pytest.importorskip("chime_vla")
    from chime_vla.memory.sem_bank import SemBank

    bank = SemBank(mock_chime_config.c7, batch_size=2, device=torch.device("cpu"))
    assert bank.slot_free.dtype == torch.bool
    assert bank.slot_free.shape == (2, mock_chime_config.c7.K_s)
    assert bool(bank.slot_free.all()), "fresh SemBank must have all slots free"


@M0_XFAIL
def test_evict_zeros_value_keeps_key(mock_chime_config):
    """evict(b, i): v_i ← 0, slot_free[i] ← 1, k_i unchanged."""
    pytest.importorskip("chime_vla")
    from chime_vla.memory.sem_bank import SemBank

    bank = SemBank(mock_chime_config.c7, batch_size=2, device=torch.device("cpu"))
    # Pretend slot (0, 3) was written
    bank.v[0, 3] = torch.randn_like(bank.v[0, 3])
    bank.slot_free[0, 3] = False
    k_before = bank.k[0, 3].clone()

    bank.evict(batch_idx=0, slot_idx=3)

    assert bank.v[0, 3].abs().max() < 1e-9, "evict must zero v_i"
    assert bool(bank.slot_free[0, 3]), "evict must set slot_free=1"
    assert torch.allclose(bank.k[0, 3], k_before), "evict must NOT touch k_i"


@M0_XFAIL
def test_softmax_logit_penalty_isolates_free_slots(mock_chime_config):
    """logit_i -= 1e9 * slot_free[i] should make the softmax over free slots
    yield ~0 probability mass.

    This is the routing-side invariant from CODE_STANDARDS §1.9.
    """
    K = mock_chime_config.c7.K_s
    logits = torch.zeros(2, K)
    slot_free = torch.zeros(2, K, dtype=torch.bool)
    slot_free[:, : K // 2] = True
    masked = logits - 1e9 * slot_free.float()
    probs = torch.softmax(masked, dim=-1)
    assert probs[:, : K // 2].max().item() < 1e-6, "free slots leaked attention"
    # And the occupied half should sum to ~1
    assert torch.allclose(probs[:, K // 2:].sum(dim=-1), torch.ones(2), atol=1e-5)
