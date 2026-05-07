"""[C6] M_geo — multi-resolution voxel grid container (CODE_STRUCTURE §3.5).

Component map: C6 (memory, deploy + train).  No learnable parameters; this
is a pure data container holding per-level voxel grids and per-voxel
``last_write_step`` timestamps for LRU / staleness diagnostics.

dtype path (CODE_STANDARDS §1.7):
    grids[level]      : (B, D, H, W, d_g) **fp32** — delta-rule writes
                        accumulate over T~200 steps; bf16 would lose precision.
    timestamp[level]  : (B, D, H, W) int32 — last write step.

Batch contract (CODE_STANDARDS §1.2): always (B, …, d_g).

Episode reset is the caller's responsibility — see
``chime_vla.utils.memory_reset.reset_memory``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from chime_vla.config import C6Config


class GeoGrid:
    """Multi-resolution voxel grid storing per-cell features.

    Attributes:
        levels:    list of cubic resolutions (e.g. ``[16]`` MVP, ``[8, 16, 32]`` full).
        grids:     ``{level: (B, D, H, W, d_g) fp32}`` voxel features.
        timestamp: ``{level: (B, D, H, W) int32}`` last-write step.
        alpha_l:   per-level write-strength schedule (mirrors levels).
        workspace_bounds: ``(x_min, x_max, y_min, y_max, z_min, z_max)`` in metres.

    M0: stub — see ``raise NotImplementedError`` in :meth:`reset` /
    :meth:`occupancy_pct`.
    """

    def __init__(
        self,
        cfg: C6Config,
        batch_size: int,
        d_g: int,
        device: torch.device | str,
    ):
        self.cfg = cfg
        self.B: int = batch_size
        self.d_g: int = d_g
        self.device: torch.device = torch.device(device)
        self.levels: list[int] = list(cfg.levels)
        self.alpha_l: list[float] = list(cfg.alpha_l)
        self.workspace_bounds: list[float] = list(cfg.workspace_bounds)

        # Per-level grids; allocated eagerly so callers can write immediately.
        self.grids: dict[int, Tensor] = {
            L: torch.zeros((self.B, L, L, L, d_g), dtype=torch.float32, device=self.device)
            for L in self.levels
        }
        self.timestamp: dict[int, Tensor] = {
            L: torch.zeros((self.B, L, L, L), dtype=torch.int32, device=self.device)
            for L in self.levels
        }

    def reset(self, batch_indices: Optional[Tensor] = None) -> None:
        """Zero out grids + timestamps for selected episodes (or all of them).

        Args:
            batch_indices: long tensor of batch slots to reset, or None for all.
        """
        raise NotImplementedError("[C6] GeoGrid.reset — M0 stub")

    def occupancy_pct(self) -> dict[int, float]:
        """Per-level fraction of voxels with non-zero norm (sparse-write monitor).

        Returns:
            ``{level: fraction in [0, 1]}`` — averaged across batch.
        """
        raise NotImplementedError("[C6] GeoGrid.occupancy_pct — M0 stub")
