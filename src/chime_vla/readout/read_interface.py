"""[C8] Read interface — cross-attention over (M_work, M_sem) + trilinear M_geo.

Component map: C8 (readout, deploy + train).  Concatenates three
information sources into a single context tensor for [C9]:

    * cross-attn over M_work flattened ``(B, K_w * N, d_h)`` → ``N_q`` tokens
    * cross-attn over M_sem ``(B, K_s, d_s)``    → (folded into above N_q via shared queries)
    * trilinear sampling of M_geo (per-level)    → ``N_geo_q`` tokens

Output shape: ``(B, N_q + K_w, d_h)`` per CODE_STRUCTURE §3.6.
The ``+ K_w`` half is the raw FIFO concatenated for the action expert
(skip path through working memory).

Slot-mask contract (CODE_STANDARDS §1.9):
    Cross-attn over M_sem MUST apply ``logit -= 1e9 * slot_free`` so free
    slots do not contribute to the readout.

prh_path SG-2 contract (CODE_STANDARDS §1.3, §1.1):
    When ``prh_path=True``, query projections must be sg-isolated so that
    PRH-side gradients do not leak through the perception backbone /
    write heads.  This flag does *not* gate which attention path runs;
    it only signals SG topology.

L_aux + SG-7 monitor (CODE_STANDARDS §1.5, §1.1):
    The last forward's attention entropy over M_work is exposed via
    :attr:`attn_entropy_to_M_work` (scalar tensor, ``(B,)``) for L_aux
    (``-λ_ent · entropy.mean()``) and for the entropy-floor monitor.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from chime_vla.config import C8Config
from chime_vla.memory.geo_grid import GeoGrid
from chime_vla.memory.sem_bank import SemBank


class ReadInterface(nn.Module):
    """Cross-attention readout assembling ``c_t`` for the action expert.

    Modules expected to live here:
        * learnable query bank ``(N_q, d_h)``
        * Q/K/V projections for M_work attention (multi-head)
        * Q/K/V projections for M_sem attention (multi-head, slot-mask aware)
        * trilinear voxel sampler (no learnable params; bilinear-interp
          MLP after sampling for projection back to d_h)

    M0: stub.
    """

    def __init__(
        self,
        cfg: C8Config,
        d_h: int,
        d_s: int,
        K_w: int,
        K_s: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_h: int = d_h
        self.d_s: int = d_s
        self.K_w: int = K_w
        self.K_s: int = K_s
        self.N_q: int = cfg.N_q
        self.N_geo_q: int = cfg.N_geo_q
        self.use_kv_cache: bool = cfg.use_kv_cache

        # Internal cache for L_aux + SG-7 monitor — set by forward.
        self._last_attn_entropy_M_work: Optional[Tensor] = None

    def forward(
        self,
        m_work: Tensor,
        m_geo: GeoGrid,
        m_sem: SemBank,
        h_t: Tensor,
        prh_path: bool = False,
    ) -> Tensor:
        """Assemble the readout context ``c_t``.

        Args:
            m_work:   ``(B, K_w, N, d_h)`` bf16 — FIFO ring (post-append).
            m_geo:    :class:`GeoGrid` — read-only.
            m_sem:    :class:`SemBank` — read-only; ``slot_free`` mask must
                      be applied to attention logits per CODE_STANDARDS §1.9.
            h_t:      ``(B, N, d_h)`` bf16 — current frame tokens (for
                      query conditioning).
            prh_path: if True, signal SG-2: query projections will be
                      sg-isolated.  Caller is the train_step (PRH path).

        Returns:
            ``c_t`` : ``(B, N_q + K_w, d_h)`` bf16.
        """
        raise NotImplementedError("[C8] ReadInterface.forward — M0 stub")

    @property
    def attn_entropy_to_M_work(self) -> Tensor:
        """Last forward's attention entropy over M_work (for L_aux + SG-7 monitor).

        Shape: ``(B,)`` fp32.  Raises ``RuntimeError`` if forward has
        never been called (cache empty).
        """
        if self._last_attn_entropy_M_work is None:
            raise RuntimeError(
                "[C8] ReadInterface.attn_entropy_to_M_work accessed before "
                "any forward pass populated the cache."
            )
        return self._last_attn_entropy_M_work
