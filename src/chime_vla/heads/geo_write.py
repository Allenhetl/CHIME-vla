"""[C3] Geometric write head — delta-rule scatter into M_geo.

Component map: C3 (heads, deploy + train).  Projects ``h_t`` to a per-voxel
delta and scatters it into the multi-resolution :class:`GeoGrid` weighted
by ``γ_geo``.

SG-1 contract (CODE_STANDARDS §1.3, ``docs/grad_flow_contract.md``):
    Caller MUST pass ``sg(γ_geo)`` — the gradient flow from L_main into
    [C5] via the geo channel is *blocked here* and arrives only through
    L_HCS.  See ``tests/test_grad_flow.py::test_sg_1``.

dtype path (CODE_STANDARDS §1.7):
    h_t bf16 → projections in fp32 → in-place scatter into GeoGrid (fp32).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from chime_vla.config import C3Config
from chime_vla.memory.geo_grid import GeoGrid


class GeoWriteHead(nn.Module):
    """Delta-rule write into multi-resolution voxel memory.

    Mutates ``m_geo`` **in place** — :meth:`forward` returns ``None`` by
    contract (CODE_STRUCTURE §3.4).  All projections live here (per-level
    voxel projection MLPs).

    M0: stub.
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

    def forward(
        self,
        h_t: Tensor,
        gamma_geo: Tensor,
        m_geo: GeoGrid,
    ) -> None:
        """Write the current frame's geometric content into M_geo.

        Args:
            h_t:       ``(B, N, d_h)`` bf16 token tensor (current frame).
            gamma_geo: ``(B,)`` fp32 in ``[0, 1]``.  **Caller MUST pass
                       ``sg(γ_geo)`` per SG-1** — this layer does not call
                       ``detach`` itself.
            m_geo:     :class:`GeoGrid` mutated in place (delta-rule scatter
                       at every level in ``cfg.write_levels``).

        Returns:
            None — side-effect only.
        """
        raise NotImplementedError("[C3] GeoWriteHead.forward — M0 stub")
