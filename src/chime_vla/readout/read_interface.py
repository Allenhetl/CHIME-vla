"""[C8] Read interface — cross-attention over (M_work, M_sem) + trilinear M_geo.

Component map: C8 (readout, deploy + train).  Concatenates three
information sources into a single context tensor for [C9]:

    * cross-attn over M_work flattened ``(B, K_w * N, d_h)`` → ``N_q`` tokens
    * cross-attn over M_sem ``(B, K_s, d_s)``    → (folded into above N_q via shared queries)
    * trilinear sampling of M_geo (per-level)    → ``N_geo_q`` tokens, fused
      additively into the cross-attn output (so output dim stays N_q).

Output shape: ``(B, N_q + K_w, d_h)`` per CODE_STRUCTURE §3.6.
The ``+ K_w`` half is the raw FIFO concatenated for the action expert
(skip path through working memory): per-frame mean-pool of M_work.

Slot-mask contract (CODE_STANDARDS §1.9):
    Cross-attn over M_sem MUST apply ``logit -= 1e9 * slot_free`` so free
    slots do not contribute to the readout.

prh_path SG-2 contract (CODE_STANDARDS §1.3, §1.1):
    When ``prh_path=True``, the query input fed to the projection is
    ``detach()``-ed so PRH-side gradients cannot flow back through the
    query path to the perception backbone / write heads.  Other gradient
    paths (key/value into M_work, the trilinear sampler params) remain
    open: SG-2 is *query-side only*.

L_aux + SG-7 monitor (CODE_STANDARDS §1.5, §1.1):
    The last forward's attention entropy over M_work is exposed via
    :attr:`attn_entropy_to_M_work` (shape ``(B,)`` fp32) for L_aux
    (``-λ_ent · entropy.mean()``) and for the entropy-floor monitor.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from chime_vla.config import C8Config
from chime_vla.memory.geo_grid import GeoGrid
from chime_vla.memory.sem_bank import SemBank


class ReadInterface(nn.Module):
    """Cross-attention readout assembling ``c_t`` for the action expert.

    Internal modules:
        * learnable query bank ``queries_kv (N_q, d_h)`` — shared between
          M_work and M_sem (fused KV) cross-attn
        * learnable spatial query bank ``queries_geo (N_geo_q, d_h)`` —
          drives the trilinear sampler (projects to a (x, y, z) coord)
        * Q/K/V projections + output proj for the (M_work + M_sem) MHA
        * ``sem_kv_proj`` projects ``d_s → d_h`` so M_sem keys/values can
          live in the same KV space as M_work
        * ``geo_coord_proj`` (d_h → 3) and ``geo_feat_proj`` (d_g → d_h)
          for the trilinear sampler
    """

    # Number of attention heads.  Kept hard-coded for MVP; could be added
    # to C8Config later.  d_h=1152 is divisible by 8 and 16.
    NUM_HEADS: int = 8

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

        if d_h % self.NUM_HEADS != 0:
            raise ValueError(
                f"[C8] d_h={d_h} must be divisible by NUM_HEADS={self.NUM_HEADS}"
            )
        self.head_dim: int = d_h // self.NUM_HEADS

        # ---- learnable queries ---- (B-free, expanded at forward)
        # queries_kv: shared queries over M_work + M_sem.
        # Initialised small so attention starts roughly uniform.
        self.queries_kv = nn.Parameter(torch.randn(self.N_q, d_h) * 0.02)
        # queries_geo: spatial queries → 3D coords for trilinear sampling.
        self.queries_geo = nn.Parameter(torch.randn(self.N_geo_q, d_h) * 0.02)

        # ---- (M_work + M_sem) MHA projections ----
        # We roll our own MHA so we can (a) easily apply the slot_free
        # mask only on the M_sem suffix of KV, and (b) extract the raw
        # attention weights to compute entropy over the M_work prefix.
        self.q_proj = nn.Linear(d_h, d_h, bias=False)
        self.k_proj = nn.Linear(d_h, d_h, bias=False)
        self.v_proj = nn.Linear(d_h, d_h, bias=False)
        self.o_proj = nn.Linear(d_h, d_h, bias=False)

        # ---- M_sem (d_s) → KV space (d_h) ----
        # M_sem stores (k, v) at d_s; we lift both to d_h before they
        # join the attention KV stack.
        self.sem_k_lift = nn.Linear(d_s, d_h, bias=False)
        self.sem_v_lift = nn.Linear(d_s, d_h, bias=False)

        # ---- trilinear sampler projections ----
        # spatial query → 3D coord ∈ [0, 1]^3 (sigmoid bounded)
        self.geo_coord_proj = nn.Linear(d_h, 3)
        # sampled per-level voxel feature (d_g) → d_h
        # We project from d_g (read off m_geo at forward, since C6 is not
        # an nn.Module and d_g is fixed via cfg).  Lazy-init: the first
        # forward call instantiates the projection at the right d_g.
        self.geo_feat_proj: Optional[nn.Linear] = None
        # cache d_g for sanity-check on subsequent forwards
        self._geo_d_g: Optional[int] = None

        # Internal cache for L_aux + SG-7 monitor — set by forward.
        self._last_attn_entropy_M_work: Optional[Tensor] = None

    # -------------------------------------------------------------- #
    # forward                                                        #
    # -------------------------------------------------------------- #
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
            h_t:      ``(B, N, d_h)`` bf16 — current frame tokens (unused
                      by the cross-attn body for MVP; learnable queries
                      already condition; kept in the signature for forward
                      compatibility with v2.2 query-conditioning).
            prh_path: if True, signal SG-2: query inputs are detached so
                      PRH-side gradients do not flow into shared params.

        Returns:
            ``c_t`` : ``(B, N_q + K_w, d_h)`` bf16.
        """
        B, K_w, N, d_h = m_work.shape
        if K_w != self.K_w or d_h != self.d_h:
            raise ValueError(
                f"[C8] m_work shape mismatch: got K_w={K_w} d_h={d_h}, "
                f"expected K_w={self.K_w} d_h={self.d_h}"
            )

        # Lazy-init geo_feat_proj on first forward (need d_g from m_geo).
        if self.geo_feat_proj is None:
            self._geo_d_g = m_geo.d_g
            self.geo_feat_proj = nn.Linear(m_geo.d_g, self.d_h, bias=False).to(
                device=m_work.device,
                dtype=torch.float32,  # params live in fp32; autocast handles bf16 ops
            )

        # All the heavy ops (linear / matmul / grid sample) under bf16
        # autocast per CODE_STANDARDS §1.7.  Entropy computation is then
        # promoted back to fp32 (numerically safer log).
        autocast_ctx = torch.amp.autocast(
            device_type=m_work.device.type, dtype=torch.bfloat16
        )

        with autocast_ctx:
            # ----- Build queries (B, N_q, d_h) -----
            # Expand learnable queries to batch.
            q_kv = self.queries_kv.unsqueeze(0).expand(B, -1, -1)  # (B, N_q, d_h)
            q_geo = self.queries_geo.unsqueeze(0).expand(B, -1, -1)  # (B, N_geo_q, d_h)

            # SG-2: detach the query *input* on PRH path.  We detach the
            # tensor handed to q_proj so PRH grad doesn't flow back via Q;
            # K/V (perception) and trilinear sampler params remain trainable.
            if prh_path:
                q_kv_in = q_kv.detach()
                q_geo_in = q_geo.detach()
            else:
                q_kv_in = q_kv
                q_geo_in = q_geo

            # ----- Cross-attn over (M_work + M_sem) -----
            attn_out, attn_w_mwork = self._cross_attn_mwork_msem(
                q_kv_in, m_work, m_sem
            )
            # attn_out: (B, N_q, d_h); attn_w_mwork: (B, num_heads, N_q, K_w*N)

            # ----- Trilinear sample of M_geo -----
            geo_sample = self._trilinear_sample(q_geo_in, m_geo)  # (B, N_geo_q, d_h)

            # ----- Fuse geo into attn_out -----
            # spec writes c_t ∈ R^{(N_q + K_w)·d_h}; N_geo_q is folded into
            # N_q via additive fusion (mean-pool of geo features broadcast
            # over the N_q queries).  This keeps the output shape contract.
            geo_fused = geo_sample.mean(dim=1, keepdim=True)  # (B, 1, d_h)
            attn_out = attn_out + geo_fused.expand_as(attn_out)

            # ----- Skip path: per-frame pool of M_work -----
            # mean-pool over the N tokens of each frame → (B, K_w, d_h)
            m_work_pool = m_work.mean(dim=2)

            # ----- concat → (B, N_q + K_w, d_h) -----
            c_t = torch.cat([attn_out, m_work_pool], dim=1)

        # ----- Entropy over M_work attention (fp32) -----
        # attn_w_mwork: (B, num_heads, N_q, K_w*N); we treat the M_work
        # slice as a probability distribution per (B, head, query).  The
        # softmax inside _cross_attn_mwork_msem already covers the *full*
        # KV (M_work + M_sem) — the M_work slice we extract is therefore
        # only a *partial* distribution; renormalise it per (B, head, q)
        # before taking entropy so the SG-7 floor stays meaningful even
        # when slots dominate (re-norm over M_work alone).
        attn_w_mwork_fp32 = attn_w_mwork.float()
        denom = attn_w_mwork_fp32.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        p = attn_w_mwork_fp32 / denom
        # H = -Σ p log p, in nats; per (B, head, query); average heads & queries.
        log_p = torch.log(p.clamp(min=1e-12))
        ent_per_q = -(p * log_p).sum(dim=-1)  # (B, num_heads, N_q)
        ent_per_b = ent_per_q.mean(dim=(1, 2))  # (B,)
        self._last_attn_entropy_M_work = ent_per_b.detach()

        return c_t

    # -------------------------------------------------------------- #
    # internals                                                      #
    # -------------------------------------------------------------- #
    def _cross_attn_mwork_msem(
        self,
        q: Tensor,  # (B, N_q, d_h) — already SG-handled
        m_work: Tensor,  # (B, K_w, N, d_h)
        m_sem: SemBank,
    ) -> tuple[Tensor, Tensor]:
        """Multi-head cross-attn with KV = [M_work_flat ; sem_lifted].

        Returns:
            * attn_out: ``(B, N_q, d_h)`` post-output-proj
            * attn_w_mwork: ``(B, num_heads, N_q, K_w*N)`` — softmax weights
              restricted to the M_work prefix (for entropy / SG-7).
        """
        B, K_w, N, d_h = m_work.shape
        K_s = self.K_s

        # KV stack: M_work flattened, then M_sem lifted into d_h.
        kv_work = m_work.reshape(B, K_w * N, d_h)  # (B, K_w*N, d_h)

        # M_sem.k / .v are fp32; cast to current autocast dtype implicitly
        # by the lift Linear layers.
        k_sem = self.sem_k_lift(m_sem.k)  # (B, K_s, d_h)
        v_sem = self.sem_v_lift(m_sem.v)  # (B, K_s, d_h)

        # Concat along KV-length dim.
        # Both K and V share the same source-token layout
        # ([M_work | M_sem]) so the slot mask only needs the K_s suffix.
        k_full = torch.cat([self.k_proj(kv_work), k_sem], dim=1)  # (B, L, d_h)
        v_full = torch.cat([self.v_proj(kv_work), v_sem], dim=1)  # (B, L, d_h)
        L = K_w * N + K_s

        q_proj = self.q_proj(q)  # (B, N_q, d_h)

        # split heads
        def split(x: Tensor) -> Tensor:
            # (B, S, d_h) → (B, num_heads, S, head_dim)
            B_, S, _ = x.shape
            return x.view(B_, S, self.NUM_HEADS, self.head_dim).transpose(1, 2)

        qh = split(q_proj)  # (B, h, N_q, hd)
        kh = split(k_full)  # (B, h, L, hd)
        vh = split(v_full)  # (B, h, L, hd)

        # scaled dot-product
        scale = 1.0 / math.sqrt(self.head_dim)
        logits = torch.matmul(qh, kh.transpose(-2, -1)) * scale  # (B, h, N_q, L)

        # ---- slot_free mask on the M_sem suffix (D5 / §1.9) ----
        # slot_free: (B, K_s) bool — True ⇒ free ⇒ subtract 1e9 from logit
        # so it cannot contribute after softmax.
        # Build a full-length additive mask (B, 1, 1, L) so it broadcasts
        # over heads and queries.
        slot_free = m_sem.slot_free.to(dtype=logits.dtype)  # (B, K_s)
        # zeros for M_work prefix, large-negative for free slots.
        sem_penalty = -1.0e9 * slot_free  # (B, K_s)
        mwork_zero = torch.zeros(
            B, K_w * N, dtype=logits.dtype, device=logits.device
        )
        full_mask = torch.cat([mwork_zero, sem_penalty], dim=1)  # (B, L)
        logits = logits + full_mask.view(B, 1, 1, L)

        # softmax over KV
        attn = torch.softmax(logits, dim=-1)  # (B, h, N_q, L)

        # head outputs
        out_h = torch.matmul(attn, vh)  # (B, h, N_q, hd)
        out = (
            out_h.transpose(1, 2)
            .contiguous()
            .view(B, q_proj.shape[1], d_h)
        )
        out = self.o_proj(out)

        # M_work slice of attention weights for entropy.
        attn_w_mwork = attn[..., : K_w * N]
        return out, attn_w_mwork

    def _trilinear_sample(
        self,
        q_geo: Tensor,  # (B, N_geo_q, d_h)
        m_geo: GeoGrid,
    ) -> Tensor:
        """Trilinear sample of M_geo at learnable spatial coordinates.

        Implementation: hand-rolled linear interpolation (8-neighbour
        weighted sum).  Picked over ``F.grid_sample`` because:
          * grid_sample expects ``(N, C, D, H, W)`` 5-D input and the
            ``(z, y, x)`` axis convention forces a transpose dance for
            ``(B, L, L, L, d_g)`` storage.
          * for L=16, K=N_geo_q=16, hand-rolled 8-corner gather is a
            handful of advanced-indexing ops at ~zero overhead.
          * the multi-level case (L ∈ {8, 16, 32}, full version) reuses
            the same code with no axis-order pitfalls.

        Multi-level fusion: weighted by ``m_geo.alpha_l[i]`` per level,
        then mean over levels.  All levels project through the same
        ``geo_feat_proj`` since d_g is shared.
        """
        B, N_geo_q, _ = q_geo.shape

        # 1. project queries → coord ∈ [0, 1]^3
        coord = torch.sigmoid(self.geo_coord_proj(q_geo))  # (B, N_geo_q, 3)

        per_level: list[Tensor] = []
        for i, L in enumerate(m_geo.levels):
            grid = m_geo.grids[L]  # (B, L, L, L, d_g) fp32
            d_g = grid.shape[-1]
            if self._geo_d_g is not None and d_g != self._geo_d_g:
                raise RuntimeError(
                    f"[C8] level {L} has d_g={d_g}, expected {self._geo_d_g}"
                )

            # 2. scale coord to [0, L-1]
            scaled = coord * (L - 1)  # (B, N_geo_q, 3)
            # 3. 8-corner indices + interp weights.
            x0 = scaled.floor().clamp(0, L - 2).long()  # (B, N_geo_q, 3)
            x1 = x0 + 1
            w1 = (scaled - x0.float()).clamp(0.0, 1.0)  # frac
            w0 = 1.0 - w1

            # split components
            x0_x, x0_y, x0_z = x0[..., 0], x0[..., 1], x0[..., 2]
            x1_x, x1_y, x1_z = x1[..., 0], x1[..., 1], x1[..., 2]
            w0_x, w0_y, w0_z = w0[..., 0], w0[..., 1], w0[..., 2]
            w1_x, w1_y, w1_z = w1[..., 0], w1[..., 1], w1[..., 2]

            # batch index, broadcast.
            b_idx = torch.arange(B, device=grid.device).view(B, 1).expand(B, N_geo_q)

            def gather(ix: Tensor, iy: Tensor, iz: Tensor) -> Tensor:
                # grid stored as (B, L, L, L, d_g); we treat the three
                # spatial axes as (x, y, z) in that order.  Choice is
                # arbitrary as long as we are self-consistent.
                return grid[b_idx, ix, iy, iz]  # (B, N_geo_q, d_g)

            c000 = gather(x0_x, x0_y, x0_z)
            c100 = gather(x1_x, x0_y, x0_z)
            c010 = gather(x0_x, x1_y, x0_z)
            c110 = gather(x1_x, x1_y, x0_z)
            c001 = gather(x0_x, x0_y, x1_z)
            c101 = gather(x1_x, x0_y, x1_z)
            c011 = gather(x0_x, x1_y, x1_z)
            c111 = gather(x1_x, x1_y, x1_z)

            # broadcast scalar weights over d_g
            w0_x_e = w0_x.unsqueeze(-1)
            w1_x_e = w1_x.unsqueeze(-1)
            w0_y_e = w0_y.unsqueeze(-1)
            w1_y_e = w1_y.unsqueeze(-1)
            w0_z_e = w0_z.unsqueeze(-1)
            w1_z_e = w1_z.unsqueeze(-1)

            sampled = (
                c000 * w0_x_e * w0_y_e * w0_z_e
                + c100 * w1_x_e * w0_y_e * w0_z_e
                + c010 * w0_x_e * w1_y_e * w0_z_e
                + c110 * w1_x_e * w1_y_e * w0_z_e
                + c001 * w0_x_e * w0_y_e * w1_z_e
                + c101 * w1_x_e * w0_y_e * w1_z_e
                + c011 * w0_x_e * w1_y_e * w1_z_e
                + c111 * w1_x_e * w1_y_e * w1_z_e
            )  # (B, N_geo_q, d_g)

            alpha = (
                m_geo.alpha_l[i]
                if i < len(m_geo.alpha_l)
                else 1.0
            )
            per_level.append(sampled * alpha)

        # mean over levels (single-level MVP collapses trivially)
        stacked = torch.stack(per_level, dim=0).mean(dim=0)  # (B, N_geo_q, d_g)

        # project d_g → d_h
        # geo_feat_proj is fp32 params; under autocast it produces bf16 out.
        out = self.geo_feat_proj(stacked)  # (B, N_geo_q, d_h)
        return out

    # -------------------------------------------------------------- #
    # property                                                       #
    # -------------------------------------------------------------- #
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
