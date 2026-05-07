"""[C3] Geometric write head — delta-rule scatter into M_geo.

Component map: C3 (heads, deploy + train).  Projects ``h_t`` to a per-voxel
delta and scatters it into the multi-resolution :class:`GeoGrid` weighted
by ``γ_geo``.

SG-1 contract (CODE_STANDARDS §1.3, ``docs/grad_flow_contract.md``):
    Caller MUST pass ``sg(γ_geo)`` — the gradient flow from L_main into
    [C5] via the geo channel is *blocked here* and arrives only through
    L_HCS.  See ``tests/test_grad_flow.py::test_sg_1``.

dtype path (CODE_STANDARDS §1.7):
    h_t bf16 → projections in fp32 → out-of-place scatter into GeoGrid (fp32).

Pipeline (architecture v2.1 §B / §C, lines 477-538):
    1. token_to_voxel MLP: h_t → voxel_pos ∈ [0, 1]^3
    2. per-level value projection: h_t → v_proj (B, N, d_g)
    3. discretise voxel_pos to level-L coords, scatter-add the
       α_l · γ_geo · v_proj delta into m_geo.grids[L]
    4. stamp m_geo.timestamp[L] at every touched voxel with ``step``.

Autograd-friendly write path (M3+):
    The data-channel update uses ``torch.index_put`` (out-of-place) and
    REASSIGNS ``m_geo.grids[L]``.  This avoids version-counter conflicts
    when L_main / L_PRH drive the readout through the *same* tensor that
    [C3] writes — in-place ``index_put_`` would invalidate the saved
    forward state of any subgraph that read the pre-write tensor.  The
    timestamp channel remains an in-place ``index_put_`` because it has
    no autograd consumers (int64 LRU bookkeeping, never differentiated).

Sparse-write invariants (architecture lines 511-538):
    * γ_geo = 0 ⇒ no write (algebraically exact, since the scatter value is 0).
    * Multiple tokens hitting the same voxel sum naturally (delta-rule).
    * ≤ N voxels touched per (batch, frame).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from chime_vla.config import C3Config
from chime_vla.memory.geo_grid import GeoGrid


class GeoWriteHead(nn.Module):
    """Delta-rule write into multi-resolution voxel memory.

    Mutates ``m_geo`` **in place** — :meth:`forward` returns ``None`` by
    contract (CODE_STRUCTURE §3.4).  All projections live here:

        * ``token_to_voxel`` — shared 2-layer MLP producing
          ``(B, N, 3)`` voxel positions in ``[0, 1]^3``.
        * ``value_proj[level_idx]`` — per-level ``Linear(d_h, d_g)``
          producing the per-voxel delta value.
    """

    def __init__(
        self,
        cfg: C3Config,
        d_h: int,
        d_g: int,
        alpha_l: list[float],
    ):
        super().__init__()
        self.cfg = cfg
        self.d_h: int = d_h
        self.d_g: int = d_g
        self.alpha_l: list[float] = list(alpha_l)
        self.write_levels: list[int] = list(cfg.write_levels)
        self.voxel_proj_hidden: int = cfg.voxel_proj_hidden

        if len(self.alpha_l) != len(self.write_levels):
            raise ValueError(
                f"alpha_l length {len(self.alpha_l)} != write_levels length "
                f"{len(self.write_levels)}; pass C6.alpha_l and C3.write_levels "
                "with matching cardinality."
            )

        # Shared MLP: h_t (B, N, d_h) -> voxel_pos (B, N, 3) in [0, 1]^3
        # via sigmoid (architecture v2.1 §B, line 477-509).
        self.token_to_voxel = nn.Sequential(
            nn.Linear(d_h, self.voxel_proj_hidden),
            nn.ReLU(),
            nn.Linear(self.voxel_proj_hidden, 3),
            nn.Sigmoid(),
        )

        # Per-level value projection — independent linear maps to d_g.
        self.value_proj = nn.ModuleList(
            [nn.Linear(d_h, d_g) for _ in self.write_levels]
        )

    def forward(
        self,
        h_t: Tensor,
        gamma_geo: Tensor,
        m_geo: GeoGrid,
        step: int = 0,
    ) -> None:
        """Write the current frame's geometric content into M_geo.

        Args:
            h_t:       ``(B, N, d_h)`` bf16 token tensor (current frame).
            gamma_geo: ``(B,)`` fp32 in ``[0, 1]``.  **Caller MUST pass
                       ``sg(γ_geo)`` per SG-1** — this layer does not call
                       ``detach`` itself.
            m_geo:     :class:`GeoGrid` mutated in place (delta-rule scatter
                       at every level in ``cfg.write_levels``).
            step:      current global step; stamped into ``m_geo.timestamp``
                       at each touched voxel.

        Returns:
            None — side-effect only.
        """
        B, N, d_h = h_t.shape
        if d_h != self.d_h:
            raise ValueError(f"h_t last dim {d_h} != configured d_h {self.d_h}")

        # Promote h_t to fp32 for projections — bf16 would lose precision over
        # T~200 accumulation steps (CODE_STANDARDS §1.7).
        h_t_f32 = h_t.float()

        # voxel_pos: (B, N, 3) in [0, 1]^3
        voxel_pos = self.token_to_voxel(h_t_f32)

        # gamma_geo: (B,) -> (B, 1) for broadcasting against (B, N, d_g)
        gamma_b1 = gamma_geo.to(dtype=torch.float32, device=h_t_f32.device).view(B, 1)

        # Flat batch indices (B, N) — used by index_put_ for scatter-add.
        batch_idx = (
            torch.arange(B, device=h_t_f32.device)
            .view(B, 1)
            .expand(B, N)
            .reshape(-1)
        )

        for level_idx, L in enumerate(self.write_levels):
            if L not in m_geo.grids:
                raise KeyError(
                    f"GeoWriteHead.write_levels contains {L} but GeoGrid has no "
                    f"such level (m_geo.levels={m_geo.levels})."
                )

            # value projection -> (B, N, d_g)
            v_proj = self.value_proj[level_idx](h_t_f32)

            # delta = α_l · γ_geo · v_proj  (B, N, d_g)
            alpha = float(self.alpha_l[level_idx])
            delta = alpha * gamma_b1.unsqueeze(-1) * v_proj  # (B, N, d_g)

            # discretise voxel_pos in [0, 1]^3 to integer voxel coords in [0, L-1].
            coord = (voxel_pos * L).long().clamp(0, L - 1)  # (B, N, 3)
            cx = coord[..., 0].reshape(-1)
            cy = coord[..., 1].reshape(-1)
            cz = coord[..., 2].reshape(-1)

            # scatter-add delta into m_geo.grids[L] using OUT-OF-PLACE
            # index_put + reassignment (autograd-friendly path).  See module
            # docstring "Autograd-friendly write path" for rationale.
            # delta_flat: (B*N, d_g); m_geo.grids[L]: (B, L, L, L, d_g) fp32.
            delta_flat = delta.reshape(B * N, self.d_g)
            grid = m_geo.grids[L]
            new_grid = grid.index_put(
                (batch_idx, cx, cy, cz),
                delta_flat,
                accumulate=True,
            )
            m_geo.grids[L] = new_grid

            # Timestamp: stamp every touched (b, x, y, z) with step, but
            # only for batch lanes with γ > 0.  This preserves the invariant
            # "γ=0 ⇒ no write" (architecture lines 511-538) for the LRU
            # timestamp channel as well as the data channel.
            ts = m_geo.timestamp[L]
            active_mask = (gamma_b1.view(B) > 0).view(B, 1).expand(B, N).reshape(-1)
            if active_mask.any():
                stamp_vals = torch.full(
                    (int(active_mask.sum().item()),),
                    int(step),
                    dtype=ts.dtype,
                    device=ts.device,
                )
                ts.index_put_(
                    (
                        batch_idx[active_mask],
                        cx[active_mask],
                        cy[active_mask],
                        cz[active_mask],
                    ),
                    stamp_vals,
                    accumulate=False,
                )
