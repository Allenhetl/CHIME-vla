"""[C10] Hindsight Causal Saliency Head — full implementation (training-only).

Architecture v2.1 §C [C10] (line 1268-1349) defines a *three-step composition*
for the offline trajectory saliency labeller; the simplified version that
ships in :mod:`chime_vla.eval.e1_judgment` only realises Step 1 (Jacobian
magnitude).  This module supplies the missing two steps:

    Step 1 — Jacobian saliency
        ``J_t^(Δ) = ‖∂a*_{t+Δ} / ∂o_t‖_F`` per Δ ∈ {4, 16}, summed over Δ.
        Each per-frame Jacobian is ``(3, H, W)`` (RGB channel × spatial).

    Step 2 — RUDDER reward redistribution
        A small LSTM ``g_θ`` regresses cumulative episode success ``R̂_t``
        from per-frame features ``h_t``.  The first-difference
        ``c_t = |R̂_t - R̂_{t-1}|`` captures "how much did this frame change
        the model's belief about success?" — a complementary, RL-grounded
        saliency signal that does *not* require differentiating an action
        head (good orthogonal noise structure to Step 1).

    Step 3 — grad-cam style decomposition + sigmoid fuse
        Decompose ``J`` into a *spatial* peak (γ_geo) and a *channel-norm*
        magnitude (γ_sem) via :class:`GradCamDecomposer`, z-score per
        trajectory, add ``α_R · z(c_t)``, then sigmoid:

            γ̂_*_t = σ(α_J · z(J_*_t) + α_R · z(c_t))

The output ``{γ̂_geo, γ̂_sem} ∈ [0, 1]^T`` is what gets persisted to
``Hindsight/output/saliency/gamma_hat/.../ep_NNNNNN.pt`` per the schema in
``docs/hindsight_contract.md`` §3 (this module only produces the tensors;
serialisation happens in the F7-Phase-2 offline script).

Key engineering notes (CODE_STANDARDS §1.7 / §1.8):

  * The Jacobian path runs in fp32 (saliency contract).  The base policy
    forward stays in whatever dtype it was loaded in — saliency is read
    off the gradient w.r.t. the *input* RGB tensor, which we keep in
    fp32 with ``requires_grad=True``.  bf16 autocast around the policy
    forward is fine because the gradient w.r.t. the leaf is upcast.
  * Memory / sem write paths are wrapped in ``no_grad`` upstream
    (mirroring :mod:`chime_vla.eval.e1_judgment`) so the gradient
    flowing back from ``a*_{t+Δ}`` only travels through perception →
    ESPC → readout → action.  The Δ-window forward is segment-scoped
    so the autograd graph never holds more than ``Δ_max + 1`` frames
    of activations (≈ 30 GB peak budget for Δ_max=64, T=280 — see
    architecture line 1297-1308).
  * The RUDDER LSTM is *small* (~70 k params at d_hidden=256, d_feat=h_t
    pooled).  It is trained per-call by :meth:`HCSHead.fit_rudder` on a
    handful of demonstration trajectories — F7-Phase-2 will hook it to
    the LIBERO cache; F7-Phase-1 (this module) only smoke-tests on
    synthetic data.

This module deliberately depends only on ``torch`` + ``torch.nn``; it does
*not* import any specific component from :mod:`chime_vla.heads` so it can
be reused with any frozen base policy that exposes the same forward
contract.  See :class:`HCSHead` for the contract.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Step 3 helper — grad-cam decomposition (no learnable params)
# ---------------------------------------------------------------------------


class GradCamDecomposer:
    """Split a per-frame Jacobian magnitude tensor into geo / sem signals.

    The input is the absolute Jacobian magnitude
    ``J ∈ R_{≥0}^{T × C × H × W}`` (C is RGB channel for the saliency-w.r.t.-
    input case; for general gradient inputs C is whatever channel axis is
    spatially adjacent to (H, W)).  The two output streams are:

      - ``J_geo[t]`` — *peak spatial response* of frame t.  We sum over
        channels first (``J.abs().sum(dim=1)``) to get a single per-frame
        heat-map, then take the maximum over (H · W).  This captures
        "the brightest pixel of the saliency map for frame t" — high if
        the t-th observation contains a *localised* feature that the
        future action depends on (a graspable object, a door handle…).
      - ``J_sem[t]`` — *channel-norm* magnitude of frame t.  We mean
        over spatial dims first (``J.abs().mean(dim=(2,3))``) to get a
        per-channel summary, then take the L2 norm across channels.
        This emphasises *what features* (channel-level texture / colour
        / motion) drive the future action, irrespective of where they
        appear in the frame — a complementary "global semantic" signal.

    Both outputs are positive scalars on an arbitrary scale; downstream
    callers (HCSHead) z-score per trajectory before applying the sigmoid.

    No learnable parameters — this is a deterministic pooling.
    """

    @staticmethod
    def decompose(J: Tensor) -> tuple[Tensor, Tensor]:
        """Decompose a Jacobian magnitude into (J_geo, J_sem).

        Args:
            J: ``(T, C, H, W)`` non-negative gradient magnitude.  Any
               sign in the input is folded out via ``.abs()`` so callers
               can pass either signed gradients or already-magnitude
               tensors.

        Returns:
            ``(J_geo, J_sem)`` both ``(T,)`` fp32 tensors in arbitrary
            positive scale.  Caller is expected to z-score per trajectory
            before fusing with other saliency signals.
        """
        if J.dim() != 4:
            raise ValueError(
                f"GradCamDecomposer expects (T, C, H, W); got {tuple(J.shape)}"
            )
        Ja = J.detach().float().abs()
        # geo: peak spatial magnitude, summed across channels first.
        # Captures "where on the frame does this matter" — the peak of
        # the channel-summed heat-map.
        spatial = Ja.sum(dim=1)                            # (T, H, W)
        J_geo = spatial.flatten(1).max(dim=1).values       # (T,)
        # sem: per-channel mean magnitude → norm across channels.
        # Captures "what features in the frame matter" without binding
        # the answer to a specific pixel.
        per_chan = Ja.mean(dim=(2, 3))                     # (T, C)
        J_sem = per_chan.norm(dim=1)                       # (T,)
        return J_geo, J_sem


# ---------------------------------------------------------------------------
# Step 2 — RUDDER reward redistribution LSTM (small, learnable, per-task)
# ---------------------------------------------------------------------------


class RudderLSTM(nn.Module):
    """Tiny LSTM head for RUDDER-style return redistribution.

    The contract is the one from Arjona-Medina et al. 2019 (NeurIPS):

      * For every time step ``t``, predict the *cumulative future return*
        ``R̂_t`` from the prefix ``τ_{≤t}``.  In LIBERO the per-episode
        reward is binary sparse (1.0 only on success at the last frame);
        the LSTM therefore learns "how confident am I that this episode
        succeeds, given everything I've seen up to t?".
      * The per-frame *contribution* is the first-difference of the
        prediction:

            c_t = | σ(R̂_t) - σ(R̂_{t-1}) |          (with c_0 := σ(R̂_0))

        Large ``c_t`` ⇒ the t-th step *resolved* a large chunk of
        prediction uncertainty, which is the RUDDER signal that those
        frames are causally salient for the episode's outcome.

    Architectural choices (intentionally minimal for F7-Phase-1):

      - 1-layer LSTM, d_hidden=256 by default.  Memory budget < 1 MB.
      - Linear head from ``d_hidden → 1`` (real-valued logit).  We pair
        this with ``binary_cross_entropy_with_logits`` against the
        cumulative-success target so the head outputs a logit, not a
        bounded probability — better gradient at the saturated tails.
      - ``forward`` returns the *logit* sequence; ``per_frame_contribution``
        applies the sigmoid before differencing because the contribution
        is meaningful in *probability* space, not logit space.

    F7-Phase-2 will train this on real LIBERO trajectories; F7-Phase-1
    smoke-tests on synthetic per-episode features.
    """

    def __init__(self, d_feat: int, d_hidden: int = 256):
        super().__init__()
        if d_feat <= 0:
            raise ValueError(f"d_feat must be > 0; got {d_feat}")
        self.d_feat = int(d_feat)
        self.d_hidden = int(d_hidden)
        self.lstm = nn.LSTM(
            input_size=self.d_feat,
            hidden_size=self.d_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(self.d_hidden, 1)

    def forward(self, feat_seq: Tensor) -> Tensor:
        """Predict the cumulative-return *logit* sequence.

        Args:
            feat_seq: ``(B, T, d_feat)`` per-frame features.  Any dtype is
                accepted; computation runs in the LSTM's parameter dtype.

        Returns:
            ``(B, T)`` real-valued logits.  Apply ``torch.sigmoid`` for
            the predicted cumulative success probability.
        """
        if feat_seq.dim() != 3:
            raise ValueError(
                f"RudderLSTM.forward expects (B, T, d_feat); "
                f"got {tuple(feat_seq.shape)}"
            )
        x = feat_seq.to(self.head.weight.dtype)
        out, _ = self.lstm(x)            # (B, T, d_hidden)
        logits = self.head(out).squeeze(-1)  # (B, T)
        return logits

    @torch.no_grad()
    def per_frame_contribution(
        self,
        feat_seq: Tensor,
        reward_seq: Tensor | None = None,
    ) -> Tensor:
        """First-difference of the predicted cumulative-return probabilities.

        Args:
            feat_seq:   ``(B, T, d_feat)`` per-frame features.
            reward_seq: ``(B, T)`` cumulative success target — currently
                unused at inference (kept in the signature for API
                stability with the training step which *does* consume
                it as the BCE target).

        Returns:
            ``(B, T)`` non-negative belief-change magnitude in
            probability space.  ``c[:, 0]`` is the prior — i.e. the
            initial probability ``σ(R̂_0)`` — which under a fresh LSTM
            init is ≈ 0.5.  Subsequent entries are
            ``|σ(R̂_t) - σ(R̂_{t-1})|`` and are typically ≪ 0.5.
        """
        del reward_seq  # API compat — only used by the training-time loss.
        logits = self.forward(feat_seq)            # (B, T)
        probs = torch.sigmoid(logits)              # (B, T) in [0, 1]
        if probs.shape[1] == 0:
            return probs
        prev = torch.cat(
            [torch.zeros_like(probs[:, :1]), probs[:, :-1]], dim=1
        )
        return (probs - prev).abs()


# ---------------------------------------------------------------------------
# Orchestrator — full [C10] HCS head
# ---------------------------------------------------------------------------


def _z_score_safe(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Z-score a 1-D tensor, returning zeros if variance is below eps.

    Identical semantics to ``chime_vla.eval.e1_judgment.z_score`` but
    duplicated here to avoid a circular import (e1_judgment depends
    transitively on the lightning module which depends on the training
    package).
    """
    x = x.float()
    if x.numel() == 0:
        return x
    mu = x.mean()
    sigma = x.std(unbiased=False)
    if not torch.isfinite(sigma) or sigma.item() < eps:
        return torch.zeros_like(x)
    return (x - mu) / (sigma + eps)


class HCSHead:
    """[C10] HCS-H — offline, training-only trajectory saliency labeller.

    Composition (architecture §C lines 1227-1262):

        Step 1.  For each Δ in ``self.deltas`` and each base frame t:
                 compute ``g = ∂‖a*_{t+Δ}‖² / ∂rgb_t`` via a segment-scoped
                 forward pass through the frozen base policy.
        Step 2.  Decompose g into ``J_geo, J_sem`` with
                 :class:`GradCamDecomposer`.  Accumulate over Δ (sum).
        Step 3.  If a fitted :class:`RudderLSTM` is attached, evaluate
                 ``c_t`` from per-frame features extracted from the same
                 forward pass.  Per-trajectory z-score both signals,
                 fuse (``α_J · z_J + α_R · z_R``), sigmoid → γ̂.

    The class is *not* an :class:`nn.Module` because it composes a
    frozen base policy (whose params we never touch) with a small
    learnable RUDDER head (which has its own training entry point).
    Treat instances as orchestrators / DSL helpers.

    Args:
        base_policy:  a CHIME-style policy module exposing the
            attributes used by :func:`_forward_segment`
            (``c1, c2, c3, c4, c5, c6, c7, c8, c9, cfg``).  Must already
            be frozen by the caller (typically the M4 ``last.ckpt``).
            For unit smoke-testing a stub with the same surface works.
        deltas:       per-frame horizon offsets to sum over.
            Default ``(4, 16)``; architecture line 1297 allows ``(4, 16, 64)``
            when ≥ 30 GB free.
        rudder:       optional pre-fit :class:`RudderLSTM`.  If ``None``
            the RUDDER signal is treated as zero (``c_t ≡ 0``) and only
            the Jacobian path contributes — useful for ablation.
        alpha_J:      coefficient on the z-scored Jacobian signal.
        alpha_R:      coefficient on the z-scored RUDDER signal.  Set
            to 0 to ablate RUDDER entirely (architecture allows this in
            the E1-fail fallback path).
        device:       ``"cuda"`` / ``"cpu"`` / explicit ``torch.device``.
    """

    def __init__(
        self,
        base_policy: nn.Module,
        deltas: Sequence[int] = (4, 16),
        rudder: RudderLSTM | None = None,
        alpha_J: float = 1.0,
        alpha_R: float = 0.5,
        device: str | torch.device = "cuda",
    ):
        self.base_policy = base_policy
        self.deltas: tuple[int, ...] = tuple(
            sorted({int(d) for d in deltas if int(d) > 0})
        )
        if not self.deltas:
            raise ValueError("HCSHead requires at least one positive Δ")
        self.rudder = rudder
        self.alpha_J = float(alpha_J)
        self.alpha_R = float(alpha_R)
        self.device = torch.device(device)
        self.base_policy_name = type(base_policy).__name__

    # -- RUDDER training -----------------------------------------------------

    def fit_rudder(
        self,
        episode_features: list[Tensor],
        episode_rewards: list[Tensor],
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Train the attached :class:`RudderLSTM` on a small set of trajectories.

        Args:
            episode_features: list of per-episode ``(T_i, d_feat)`` tensors.
                ``d_feat`` must match ``self.rudder.d_feat``.  Variable
                ``T_i`` is supported via right-padding to the max ``T``
                of the batch.
            episode_rewards:  list of per-episode ``(T_i,)`` cumulative
                success targets ∈ {0, 1}.  For LIBERO sparse reward this
                is ``cumsum(reward)`` clipped to 1.
            epochs: number of full-batch passes.  50-100 is the
                architecture-suggested ballpark for ~50-100 trajectories.
            lr:     AdamW learning rate.
            weight_decay: AdamW weight decay.
            verbose: if True, print per-epoch training loss.

        Returns:
            dict with ``"loss_first"`` and ``"loss_last"`` for telemetry.
        """
        if self.rudder is None:
            raise RuntimeError(
                "HCSHead.fit_rudder called but self.rudder is None — "
                "construct HCSHead with a RudderLSTM instance first."
            )
        if len(episode_features) != len(episode_rewards):
            raise ValueError(
                f"episode_features ({len(episode_features)}) and "
                f"episode_rewards ({len(episode_rewards)}) must align"
            )
        if not episode_features:
            raise ValueError("episode_features must not be empty")

        device = self.device
        # Pad to max T for batched LSTM; build a valid mask so padded
        # frames don't contribute to the loss.
        max_T = max(int(f.shape[0]) for f in episode_features)
        d_feat = int(episode_features[0].shape[1])
        if d_feat != self.rudder.d_feat:
            raise ValueError(
                f"feature dim {d_feat} ≠ rudder.d_feat {self.rudder.d_feat}"
            )
        B = len(episode_features)
        feat_buf = torch.zeros(B, max_T, d_feat, device=device, dtype=torch.float32)
        target_buf = torch.zeros(B, max_T, device=device, dtype=torch.float32)
        mask_buf = torch.zeros(B, max_T, device=device, dtype=torch.bool)
        for i, (f, r) in enumerate(zip(episode_features, episode_rewards)):
            T_i = int(f.shape[0])
            feat_buf[i, :T_i] = f.to(device=device, dtype=torch.float32)
            target_buf[i, :T_i] = r.to(device=device, dtype=torch.float32)
            mask_buf[i, :T_i] = True

        self.rudder.to(device).train()
        opt = torch.optim.AdamW(
            self.rudder.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_first: float | None = None
        loss_last = 0.0
        for ep in range(int(epochs)):
            opt.zero_grad()
            logits = self.rudder(feat_buf)                # (B, max_T)
            # Per-frame BCE with mask reduction (CODE_STANDARDS §1.4).
            elem = F.binary_cross_entropy_with_logits(
                logits, target_buf, reduction="none"
            )
            n = mask_buf.float().sum().clamp(min=1.0)
            loss = (elem * mask_buf.float()).sum() / n
            loss.backward()
            opt.step()
            v = float(loss.item())
            if loss_first is None:
                loss_first = v
            loss_last = v
            if verbose:
                print(f"[rudder] ep={ep:03d} loss={v:.4f}")
        self.rudder.eval()
        return {"loss_first": float(loss_first or 0.0), "loss_last": loss_last}

    # -- saliency computation ------------------------------------------------

    @torch.enable_grad()
    def compute(
        self,
        rgb_seq: Tensor,
        proprio_seq: Tensor,
        action_seq: Tensor,
        reward_seq: Tensor | None = None,
    ) -> dict[str, Any]:
        """Run the three-step composition for a single trajectory.

        Args:
            rgb_seq:     ``(T, 3, 224, 224)`` fp32 in [0, 1].
            proprio_seq: ``(T, 8)`` fp32.
            action_seq:  ``(T, 8)`` fp32 — *target* expert action.  We
                back-prop against ``a*_{t+Δ}`` so this is the GT action
                (passed from the demonstration), not the policy output.
                Currently the implementation differentiates
                ``‖a_pred_{t+Δ}‖²`` (matching the simplified path in
                :mod:`chime_vla.eval.e1_judgment` line 244-245); a future
                refinement may switch to ``‖a_pred - a*‖²`` once the
                base policy is well-fit (see TODO at end of file).
            reward_seq:  ``(T,)`` cumulative success target — fed to the
                RUDDER LSTM as the per-frame target.  Optional: if
                ``self.rudder is None`` the value is ignored.

        Returns:
            dict with:
                ``gamma_geo``:     ``(T,)`` fp32 ∈ [0, 1] — final saliency.
                ``gamma_sem``:     ``(T,)`` fp32 ∈ [0, 1].
                ``J_geo_raw``:     ``(T,)`` fp32 — pre-z Jacobian magnitude.
                ``J_sem_raw``:     ``(T,)`` fp32.
                ``rudder_delta``:  ``(T,)`` fp32 — RUDDER per-frame contribution
                                   (zeros if no RUDDER attached).
                ``meta``: dict with strategy / base_policy / delta_set / fields
                          for the gamma_hat .pt schema (docs/hindsight_contract.md §3).
        """
        # ---- 0. Validation --------------------------------------------------
        if rgb_seq.dim() != 4:
            raise ValueError(
                f"rgb_seq must be (T, 3, H, W); got {tuple(rgb_seq.shape)}"
            )
        T = int(rgb_seq.shape[0])
        if proprio_seq.dim() != 2 or proprio_seq.shape[0] != T:
            raise ValueError(
                f"proprio_seq must be (T, P) matching rgb T={T}; "
                f"got {tuple(proprio_seq.shape)}"
            )
        if action_seq.dim() != 2 or action_seq.shape[0] != T:
            raise ValueError(
                f"action_seq must be (T, A) matching rgb T={T}; "
                f"got {tuple(action_seq.shape)}"
            )
        delta_max = max(self.deltas)
        if T <= delta_max:
            raise ValueError(
                f"HCSHead.compute: T={T} must exceed max delta={delta_max}"
            )
        device = self.device
        rgb_dev = rgb_seq.detach().to(device=device, dtype=torch.float32)
        proprio_dev = proprio_seq.detach().to(device=device, dtype=torch.float32)

        # ---- 1. Step 1+2 — Jacobian / grad-cam decomposition ---------------
        # We store the per-frame raw (T, C, H, W) Jacobian magnitudes only as
        # the (T,) summary returned by GradCamDecomposer; keeping the full
        # (T, 3, H, W) tensor would be ~40 MB per Δ, * 280 frames * 2 Δ ≈ OK
        # but unnecessary for downstream BCE supervision.  Sum across Δ.
        raw_geo = torch.zeros(T, dtype=torch.float32)
        raw_sem = torch.zeros(T, dtype=torch.float32)

        # The base policy is expected to be in eval mode and frozen.  We don't
        # call .eval() here in case the caller intentionally enables dropout
        # (they shouldn't, but it's their call).
        n_pairs = 0
        for delta in self.deltas:
            for t in range(T - delta):
                seg_start = t
                seg_end = t + delta + 1
                rgb_seg = rgb_dev[seg_start:seg_end].clone().detach()
                rgb_seg.requires_grad_(True)
                proprio_seg = proprio_dev[seg_start:seg_end]

                a_pred = self._forward_segment(rgb_seg, proprio_seg)
                target = a_pred[delta].float()
                scalar = (target * target).sum()
                grad_rgb = torch.autograd.grad(
                    scalar, rgb_seg, retain_graph=False, create_graph=False
                )[0]
                # Slice the gradient at the t-th relative frame (== rel idx 0
                # by our segment construction) and treat it as the per-frame
                # Jacobian magnitude tensor.  Shape (3, H, W) → reshape to
                # (1, 3, H, W) for the decomposer, then squeeze-out frame dim.
                g_t = grad_rgb[0].detach().float()                  # (3, H, W)
                J_one = g_t.abs().unsqueeze(0)                      # (1, 3, H, W)
                J_geo_one, J_sem_one = GradCamDecomposer.decompose(J_one)
                raw_geo[t] = raw_geo[t] + J_geo_one[0].cpu()
                raw_sem[t] = raw_sem[t] + J_sem_one[0].cpu()
                n_pairs += 1
                # Free segment-scoped graph deterministically.
                del rgb_seg, proprio_seg, a_pred, target, scalar, grad_rgb, g_t

        # ---- 2. Step 2 — RUDDER contribution -------------------------------
        if self.rudder is not None:
            with torch.no_grad():
                feat_seq = self._extract_features(rgb_dev, proprio_dev)  # (T, d)
                if feat_seq.shape[1] != self.rudder.d_feat:
                    raise RuntimeError(
                        f"RUDDER d_feat mismatch: extractor produced {feat_seq.shape[1]}, "
                        f"rudder expects {self.rudder.d_feat}"
                    )
                self.rudder.to(device).eval()
                delta_R = self.rudder.per_frame_contribution(
                    feat_seq.unsqueeze(0).to(device),
                    reward_seq.unsqueeze(0).to(device) if reward_seq is not None else None,
                )[0].detach().float().cpu()
        else:
            delta_R = torch.zeros(T, dtype=torch.float32)

        # ---- 3. Step 3 — z-score, fuse, sigmoid ----------------------------
        z_J_geo = _z_score_safe(raw_geo)
        z_J_sem = _z_score_safe(raw_sem)
        z_R = _z_score_safe(delta_R)
        gamma_geo = torch.sigmoid(self.alpha_J * z_J_geo + self.alpha_R * z_R)
        gamma_sem = torch.sigmoid(self.alpha_J * z_J_sem + self.alpha_R * z_R)

        meta = {
            "strategy": "hcs_v2_full",
            "base_policy": self.base_policy_name,
            "delta_set": list(self.deltas),
            "saliency_method": "exact_jacobian_plus_rudder",
            "alpha_J": self.alpha_J,
            "alpha_R": self.alpha_R,
            "rudder_attached": self.rudder is not None,
            "n_pairs": int(n_pairs),
        }
        return {
            "gamma_geo": gamma_geo,
            "gamma_sem": gamma_sem,
            "J_geo_raw": raw_geo,
            "J_sem_raw": raw_sem,
            "rudder_delta": delta_R,
            "meta": meta,
        }

    # -- internals -----------------------------------------------------------

    def _forward_segment(
        self,
        rgb_segment: Tensor,
        proprio_segment: Tensor,
    ) -> Tensor:
        """Run the per-step CHIME forward across an S-frame segment, B = 1.

        This mirrors :func:`chime_vla.eval.e1_judgment._forward_segment`.
        We rebuild a fresh ``M_work / M_geo / M_sem`` per call so the
        autograd graph stays bounded by the segment length (≤ Δ_max + 1)
        — see CODE_STANDARDS §1.7 / §1.8 for the rationale and the
        architecture line 1297-1308 for the 30 GB budget.

        Returns:
            ``(S, action_dim)`` autograd-tracked tensor.
        """
        # Lazy import: heavy chime_vla deps that the unit tests don't need.
        from chime_vla.memory.geo_grid import GeoGrid
        from chime_vla.memory.sem_bank import SemBank
        from chime_vla.perception.fifo_buffer import WorkBuffer

        model = self.base_policy
        cfg = model.cfg
        device = rgb_segment.device
        S = int(rgb_segment.shape[0])

        B = 1
        c2 = WorkBuffer(cfg.c2, batch_size=B, device=device)
        m_geo = GeoGrid(cfg.c6, batch_size=B, d_g=cfg.c6.d_g, device=device)
        m_sem = SemBank(cfg.c7, batch_size=B, device=device)

        a_pred_steps: list[Tensor] = []
        for s in range(S):
            rgb_s = rgb_segment[s : s + 1]
            proprio_s = proprio_segment[s : s + 1]

            h_t = model.c1(rgb_s, proprio_s)
            m_work_prev = c2.snapshot()
            gamma_geo, gamma_sem = model.c5(h_t, m_work_prev)

            # Manual FIFO append (mirrors training step) — avoid relying on
            # WorkBuffer.append in case it has detach side-effects.
            if c2.K_w > 1:
                shifted = torch.cat(
                    [c2.buffer[:, 1:], h_t.to(c2.buffer.dtype).unsqueeze(1)],
                    dim=1,
                )
            else:
                shifted = h_t.to(c2.buffer.dtype).unsqueeze(1)
            c2.buffer = shifted
            c2._n_appended = torch.clamp(c2._n_appended + 1, max=c2.K_w)
            m_work_post = shifted

            with torch.no_grad():
                model.c3(h_t.detach(), gamma_geo.detach(), m_geo, step=s)
                model.c4(h_t.detach(), gamma_sem.detach(), m_sem, step=s)

            c_t = model.c8(m_work_post, m_geo, m_sem, h_t)
            h_t_cls = h_t.mean(dim=1)
            a_pred_s = model.c9(c_t, h_t_cls)
            a_pred_steps.append(a_pred_s.squeeze(0))

        return torch.stack(a_pred_steps, dim=0)

    @torch.no_grad()
    def _extract_features(
        self,
        rgb_dev: Tensor,
        proprio_dev: Tensor,
    ) -> Tensor:
        """Extract per-frame ``(T, d_feat)`` features for the RUDDER LSTM.

        Uses the base policy's ``c1`` backbone in no-grad mode and
        mean-pools across the token dimension — the resulting vector
        is what we call ``h_t (CLS)`` in architecture §B.2.  This is
        the same pooled hidden state the action expert consumes, so
        RUDDER's cumulative-return prediction operates on the same
        latent feature stream that the base policy uses for control.

        For unit-smoke contexts the caller can monkey-patch this method
        to return synthetic features instead of running the SigLIP
        backbone — see ``tests/test_hcs_head.py``.
        """
        model = self.base_policy
        T = int(rgb_dev.shape[0])
        feats: list[Tensor] = []
        for t in range(T):
            h_t = model.c1(rgb_dev[t : t + 1], proprio_dev[t : t + 1])  # (1, N, d_h)
            feats.append(h_t.mean(dim=1).squeeze(0).float())            # (d_h,)
        return torch.stack(feats, dim=0)


# ---------------------------------------------------------------------------
# F7-Phase-2 TODOs (intentionally not implemented in this commit):
#
#   * ``HCSHead._build_target_action(...)``: currently we differentiate
#     ``‖a_pred_{t+Δ}‖²``; per architecture line 1299 the *exact* signal
#     is ``∂a*_{t+Δ}/∂o_t`` — i.e. the GT expert action is the target.
#     Once the base policy is well-fit this matters less (a_pred ≈ a*),
#     but Phase-2 should switch to ``‖a_pred - a_seq‖²`` for parity with
#     the math.
#   * ``HCSHead.from_pretrained(ckpt_path, ...)``: convenience loader for
#     the M4 ``last.ckpt`` to wire up the frozen base policy.  Phase-2.
#   * Offline batch driver: ``scripts/compute_gamma_hat_offline.py`` that
#     iterates LIBERO episodes, runs ``HCSHead.compute``, and writes
#     ``ep_NNNNNN.pt`` per ``docs/hindsight_contract.md`` §3 — Phase-2.
#   * Streaming RUDDER training: hook into the LIBERO datamodule so the
#     LSTM can be re-fit at the start of each Phase-2 saliency run
#     without loading every episode into memory.
# ---------------------------------------------------------------------------
