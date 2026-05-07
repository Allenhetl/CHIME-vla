"""[C5] Episodic Salience Predictor with Contrast (ESPC).

Component map: C5 (heads, deploy + train).  Reads M_work^{t-1} (the FIFO
**before** the current frame's append, per CODE_STANDARDS §1.3) and emits
two scalar gates per batch row:

    γ_geo, γ_sem ∈ [0, 1]

These gate the write strength of [C3] / [C4] respectively, with the
caller wrapping them in ``sg(.)`` before passing them to write heads
(SG-1, see ``docs/grad_flow_contract.md``).

Architecture (CODE_STRUCTURE §3.3, architecture v2.1 §C / [C5] cards):
    1. ψ encoder over m_work:
         MVP: ``GRU`` over time (cfg.use_gru=True, cfg.psi_layers=1)
         Full: ``1-layer Transformer`` (NOT YET IMPLEMENTED — see TODO)
       Sequence dimension: K_w (each step is a *frame* of M_work),
       per-frame token grid is mean-pooled over N to form the GRU input.
       The chosen sequence axis follows the K_w hint in CODE_STRUCTURE
       §3.3 (m_work has shape ``(B, K_w, N, d_h)`` and the natural causal
       order for next-frame prediction is along K_w).
    2. ``geo_proj`` and ``sem_proj`` linears to ``cfg.d_proj=64``.
       L_HCS-only path (SG-6): trainable ONLY by L_HCS.  The caller wraps
       γ in ``sg(.)`` before passing to [C3] / [C4] (SG-1), so L_main
       cannot reach these projections via the write heads.
    3. EMA-normalised contrast: ``z = (raw_err - μ_ema) / (σ_ema + eps)``
       (μ_ema, σ_ema updated post-step via :meth:`update_ema`; warmup
       ``cfg.ema_warmup_steps`` returns identity normalisation z = raw_err).
    4. ``γ = sigmoid(z / cfg.sigmoid_temp)``.

dtype path (CODE_STANDARDS §1.7):
    inputs bf16 → cast to fp32 internally → γ returned as fp32.
    EMA running stats kept in fp32 (avoids bf16 underflow over 200+ steps).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from chime_vla.config import C5Config


class ESPC(nn.Module):
    """ψ encoder + EMA contrast + γ_geo / γ_sem heads.

    Caller-side SG contract:
        Outputs of :meth:`forward` are the *raw* γ values; the train_step
        wraps them in ``sg(.)`` before handing them to [C3] / [C4]
        (SG-1).  The L_HCS BCE target also uses ``sg(.)`` on the
        γ̂ Hindsight target, never on the predicted γ — the gradient
        flowing into ψ is the only learning signal for [C5].
    """

    def __init__(self, cfg: C5Config, d_h: int):
        super().__init__()
        self.cfg = cfg
        self.d_h: int = d_h
        self.K_w: int = cfg.K_w
        self.psi_layers: int = cfg.psi_layers
        self.use_gru: bool = cfg.use_gru
        self.d_proj: int = cfg.d_proj
        self.ema_coeff: float = cfg.ema_coeff
        self.ema_warmup_steps: int = cfg.ema_warmup_steps
        self.sigmoid_temp: float = cfg.sigmoid_temp

        # ----- ψ predictor (fp32 per CODE_STANDARDS §1.7) -----
        if self.use_gru:
            # MVP: 1-layer GRU over the K_w time axis
            self.psi = nn.GRU(
                input_size=d_h,
                hidden_size=d_h,
                num_layers=self.psi_layers,
                batch_first=True,
            )
        else:
            # TODO: full 1-layer Transformer encoder layer (not yet
            # exercised by tests; gated behind cfg.use_gru=False).
            raise NotImplementedError(
                "[C5] ψ Transformer mode not yet implemented; use cfg.use_gru=True"
            )
        self.psi_norm = nn.LayerNorm(d_h)

        # ----- dual projections (SG-6: trainable only by L_HCS) -----
        self.geo_proj = nn.Linear(d_h, self.d_proj)
        self.sem_proj = nn.Linear(d_h, self.d_proj)

        # Force fp32 weights on every parameter (per §1.7).  Buffers below
        # are also fp32.  Forward casts inputs to fp32 internally.
        self.to(torch.float32)

        # ----- EMA running stats (fp32; not learnable) -----
        # Means and variances are scalars per channel (geo, sem) — the raw
        # signal we standardise is a per-batch-row L2/cosine distance, so
        # one scalar μ / σ² each is sufficient.
        self.register_buffer(
            "running_mean_geo", torch.zeros((), dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "running_var_geo", torch.ones((), dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "running_mean_sem", torch.zeros((), dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "running_var_sem", torch.ones((), dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_step", torch.zeros((), dtype=torch.long), persistent=True
        )

        # Cached pre-EMA batch stats — populated in forward, consumed by
        # update_ema().  Kept as buffers so they roundtrip with state_dict
        # but they are non-persistent (a fresh forward overwrites them).
        self.register_buffer(
            "_last_e_geo", torch.zeros((), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_last_e_sem", torch.zeros((), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_last_e_geo_var", torch.zeros((), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_last_e_sem_var", torch.zeros((), dtype=torch.float32), persistent=False
        )
        # Last forward's ψ prediction (B, d_h) fp32 — exposed via property for
        # the M2 self-supervised L_predict loss (architecture §0.7.4: "[C5 仅
        # prediction-error self-supervised, GRU 实现]").  Set to None until the
        # first forward; not part of state_dict.
        self._last_h_hat_pred: Tensor | None = None
        self._has_pending_stats: bool = False

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, h_t: Tensor, m_work: Tensor) -> tuple[Tensor, Tensor]:
        """Compute ``(γ_geo, γ_sem)`` for the current frame.

        Args:
            h_t:    ``(B, N, d_h)`` bf16/fp32 — current frame tokens.
            m_work: ``(B, K_w, N, d_h)`` bf16/fp32 — FIFO **before**
                    appending h_t (per CODE_STANDARDS §1.3 invariant).
                    Note: SG-5 requires the *caller* to detach m_work
                    before passing it in; ESPC itself does not call
                    ``.detach()`` on it (so this method stays pure).

        Returns:
            ``(γ_geo, γ_sem)`` each ``(B,)`` fp32 in ``[0, 1]``.
            Caller wraps in ``sg(.)`` before passing to [C3] / [C4] (SG-1).
        """
        if h_t.dim() != 3:
            raise ValueError(f"h_t must be (B, N, d_h); got {tuple(h_t.shape)}")
        if m_work.dim() != 4:
            raise ValueError(
                f"m_work must be (B, K_w, N, d_h); got {tuple(m_work.shape)}"
            )

        B = h_t.shape[0]
        # Cast to fp32 for ψ / EMA path (§1.7).
        h_t_f = h_t.to(torch.float32)
        m_work_f = m_work.to(torch.float32)

        # 1. Pool m_work over N → (B, K_w, d_h) GRU input
        m_pooled = m_work_f.mean(dim=2)

        # 2. ψ predicts next-frame hidden: take final GRU hidden state
        if self.use_gru:
            # output: (B, K_w, d_h); h_n: (psi_layers, B, d_h)
            _, h_n = self.psi(m_pooled)
            h_hat_pred = h_n[-1]  # (B, d_h)
        else:
            raise NotImplementedError("[C5] ψ Transformer mode not implemented")
        h_hat_pred = self.psi_norm(h_hat_pred)  # (B, d_h)
        # Cache for L_predict_self_supervised consumers (caller stacks this
        # over T to form (B, T, d_h)).  Held WITH grad — caller may detach
        # the *target* but never the prediction.
        self._last_h_hat_pred = h_hat_pred

        # 3. Pool h_t and project both into d_proj
        h_t_pooled = h_t_f.mean(dim=1)  # (B, d_h)

        # geo channel: L2 distance in geo_proj space
        geo_real = self.geo_proj(h_t_pooled)  # (B, d_proj)
        geo_pred = self.geo_proj(h_hat_pred)  # (B, d_proj)
        e_geo_raw = (geo_real - geo_pred).norm(dim=-1)  # (B,)

        # sem channel: cosine-distance flavour in sem_proj space
        # Use 1 - cos sim to keep range similar; matches arch v2.1 §C "1 - cos".
        sem_real = self.sem_proj(h_t_pooled)
        sem_pred = self.sem_proj(h_hat_pred)
        cos_sim = torch.nn.functional.cosine_similarity(sem_real, sem_pred, dim=-1)
        e_sem_raw = 1.0 - cos_sim  # (B,) in [0, 2]

        # 4. EMA standardisation.  During warmup, identity (mean=0, var=1).
        in_warmup = int(self._step.item()) < self.ema_warmup_steps
        eps = 1e-5
        if in_warmup:
            mu_geo = torch.zeros((), dtype=torch.float32, device=e_geo_raw.device)
            sigma_geo = torch.ones((), dtype=torch.float32, device=e_geo_raw.device)
            mu_sem = torch.zeros((), dtype=torch.float32, device=e_sem_raw.device)
            sigma_sem = torch.ones((), dtype=torch.float32, device=e_sem_raw.device)
        else:
            mu_geo = self.running_mean_geo
            sigma_geo = self.running_var_geo.clamp(min=eps).sqrt()
            mu_sem = self.running_mean_sem
            sigma_sem = self.running_var_sem.clamp(min=eps).sqrt()

        z_geo = (e_geo_raw - mu_geo) / (sigma_geo + eps)
        z_sem = (e_sem_raw - mu_sem) / (sigma_sem + eps)

        gamma_geo = torch.sigmoid(z_geo / self.sigmoid_temp)
        gamma_sem = torch.sigmoid(z_sem / self.sigmoid_temp)

        # 5. Stash this batch's raw stats for the next update_ema() call.
        # Detached so EMA buffers never carry autograd state.
        if self.training:
            with torch.no_grad():
                self._last_e_geo.copy_(e_geo_raw.detach().mean().float())
                self._last_e_sem.copy_(e_sem_raw.detach().mean().float())
                # unbiased=False → use 1/B variance even for B=1 (safe)
                if B > 1:
                    self._last_e_geo_var.copy_(
                        e_geo_raw.detach().var(unbiased=False).float()
                    )
                    self._last_e_sem_var.copy_(
                        e_sem_raw.detach().var(unbiased=False).float()
                    )
                else:
                    self._last_e_geo_var.copy_(torch.tensor(1.0, dtype=torch.float32))
                    self._last_e_sem_var.copy_(torch.tensor(1.0, dtype=torch.float32))
            self._has_pending_stats = True

        # Final dtype guarantee: γ ∈ fp32, shape (B,)
        return gamma_geo.to(torch.float32), gamma_sem.to(torch.float32)

    # ------------------------------------------------------------------
    # last_h_hat_pred — accessor for the M2 self-supervised L_predict loss
    # ------------------------------------------------------------------
    @property
    def last_h_hat_pred(self) -> Tensor | None:
        """Most recent forward's ψ prediction ``(B, d_h)`` fp32.

        ``None`` if no forward has been called.  Used by the M2 fallback
        path's ``L_predict_self_supervised`` (architecture §0.7.4); under
        normal full training the predictor is also trained by L_HCS and
        this signal is a redundancy check rather than the primary learner.

        The returned tensor is the *live* prediction (with gradient
        attached); the caller is responsible for detaching the *target*
        (``h_t``) before computing MSE so no grad flows back into [C1].
        """
        return self._last_h_hat_pred

    # ------------------------------------------------------------------
    # update_ema
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_ema(self) -> None:
        """Push the most recent batch's pre-sigmoid statistics into the EMA.

        Called *after* ``loss.backward + optimizer.step`` to keep the
        contrast baseline up to date.  Uses the per-batch mean/variance
        of ``e_geo`` and ``e_sem`` cached during the last :meth:`forward`.

        During warmup (``self._step < ema_warmup_steps``) we still
        accumulate into the running stats (so they are warm by the time
        warmup ends) but the forward continues to return identity-z γ.
        """
        if not self._has_pending_stats:
            # No-op if forward was not invoked since the last update — keeps
            # state_dict / unit-test ergonomics clean.
            self._step += 1
            return

        c = self.ema_coeff

        # μ_new = c * μ_old + (1 - c) * batch_mean
        self.running_mean_geo.mul_(c).add_(self._last_e_geo, alpha=1.0 - c)
        self.running_mean_sem.mul_(c).add_(self._last_e_sem, alpha=1.0 - c)

        # var_new = c * var_old + (1 - c) * batch_var (running EMA)
        self.running_var_geo.mul_(c).add_(self._last_e_geo_var, alpha=1.0 - c)
        self.running_var_sem.mul_(c).add_(self._last_e_sem_var, alpha=1.0 - c)

        self._step += 1
        self._has_pending_stats = False
