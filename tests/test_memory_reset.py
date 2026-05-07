"""Episode-boundary memory reset (CODE_STANDARDS §1.2)."""

from __future__ import annotations

import pytest
import torch


M0_XFAIL = pytest.mark.xfail(
    reason="M0: memory reset hook not impl; lands at M1",
    strict=False,
)


def test_reset_zeros_all_memory(mock_chime_config):
    """`reset_memory(state, batch_indices)` clears M_geo, M_sem.v and re-flips
    slot_free to all-True for the named batch slots.  M_work is also flushed.
    """
    pytest.importorskip("chime_vla")
    from chime_vla.memory.geo_grid import GeoGrid
    from chime_vla.memory.sem_bank import SemBank
    from chime_vla.perception.fifo_buffer import WorkBuffer

    cfg = mock_chime_config
    device = torch.device("cpu")
    B = 2
    geo = GeoGrid(cfg.c6, batch_size=B, d_g=cfg.c6.d_g, device=device)
    sem = SemBank(cfg.c7, batch_size=B, device=device)
    work = WorkBuffer(cfg.c2, batch_size=B, device=device)

    # Pollute the state.
    for L, grid in geo.grids.items():
        grid.add_(1.0)
    sem.v.add_(1.0)
    sem.slot_free.fill_(False)
    work.append(torch.randn(B, cfg.c2.N, cfg.c2.d_h))

    # Reset only batch index 0.
    geo.reset(batch_indices=torch.tensor([0]))
    sem.reset(batch_indices=torch.tensor([0]))
    work.reset(batch_indices=torch.tensor([0]))

    for L, grid in geo.grids.items():
        assert grid[0].abs().max() < 1e-9, f"M_geo[0] not cleared at level {L}"
        assert grid[1].abs().max() > 0.5, f"M_geo[1] (untouched) wrongly cleared at level {L}"
    assert sem.v[0].abs().max() < 1e-9
    assert bool(sem.slot_free[0].all()), "slot_free not reset to True"
    assert not bool(sem.slot_free[1].any()), "untouched batch lane should still be occupied"
