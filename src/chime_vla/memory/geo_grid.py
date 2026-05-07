"""[C6] M_geo — multi-resolution voxel grid container (CODE_STRUCTURE §3.5).

Component map: C6 (memory, deploy + train).  No learnable parameters; this
is a pure data container holding per-level voxel grids and per-voxel
``last_write_step`` timestamps for LRU / staleness diagnostics.

dtype path (CODE_STANDARDS §1.7):
    grids[level]      : (B, D, H, W, d_g) **fp32** — delta-rule writes
                        accumulate over T~200 steps; bf16 would lose precision.
    timestamp[level]  : (B, D, H, W) int64 — last write step (LRU eviction
                        target at M3+).

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
        timestamp: ``{level: (B, D, H, W) int64}`` last-write step.
        alpha_l:   per-level write-strength schedule (mirrors levels).
        workspace_bounds: ``(x_min, x_max, y_min, y_max, z_min, z_max)`` in metres.
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
        # fp32 — delta-rule writes accumulate over T~200 steps; bf16 would
        # lose precision (CODE_STANDARDS §1.7).
        self.grids: dict[int, Tensor] = {
            L: torch.zeros((self.B, L, L, L, d_g), dtype=torch.float32, device=self.device)
            for L in self.levels
        }
        # int64 last-write step per voxel — used by LRU eviction (M3+).
        self.timestamp: dict[int, Tensor] = {
            L: torch.zeros((self.B, L, L, L), dtype=torch.int64, device=self.device)
            for L in self.levels
        }

    def reset(self, batch_indices: Optional[Tensor] = None) -> None:
        """Zero out grids + timestamps for selected episodes (or all of them).

        ``workspace_bounds`` is metadata and unaffected.

        Args:
            batch_indices: long tensor of batch slots to reset, or ``None``
                for all.
        """
        if batch_indices is None:
            for L in self.levels:
                self.grids[L].zero_()
                self.timestamp[L].zero_()
            return

        idx = batch_indices.to(self.device).long()
        for L in self.levels:
            self.grids[L][idx] = 0.0
            self.timestamp[L][idx] = 0

    def occupancy_pct(self) -> dict[int, float]:
        """Per-level fraction of voxels with non-zero norm (sparse-write monitor).

        A voxel counts as "occupied" iff its feature vector has any non-zero
        component (``|v|.sum(-1) > 0``).  Averaged over batch lanes and
        spatial dims, returned as a scalar in ``[0, 1]`` per level.

        Returns:
            ``{level: fraction in [0, 1]}``.
        """
        out: dict[int, float] = {}
        for L in self.levels:
            grid = self.grids[L]  # (B, L, L, L, d_g)
            occupied = (grid.abs().sum(dim=-1) > 0).float()  # (B, L, L, L)
            out[L] = float(occupied.mean().item())
        return out
