"""5-loss assembly for CHIME-VLA training (CODE_STRUCTURE §7).

All component losses follow the codebase rule (CODE_STANDARDS §1.4):
    reduction = mean over (B, valid-T) with the dataloader's ``valid_mask``.
Implementations should reuse :func:`chime_vla.utils.losses.masked_mse` for
MSE-style targets.

The five (six) losses:
    * L_main : flow-matching action loss (per-step regression on a*)
               MVP: BC MSE — same gradient direction, simpler.  Full flow
               matching loss left for M3+.
    * L_HCS  : BCE between [C5] γ and Hindsight γ̂ (sg-target, SG-5)
    * L_PRH  : per-horizon (ô_{t+k}, â_{t+k}) prediction loss
               M1: PRH module not yet wired into train_step; returns 0.
    * L_CSM  : counterfactual slot importance loss
               M1: CSM not yet implemented; returns 0.
    * L_aux  : −λ_ent · attention entropy over M_work
    * L_GC   : (M5+ optional, MVP off) gradient consistency loss
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from chime_vla.config import C11Config, C12Config  # noqa: F401  (re-export)
from chime_vla.utils.losses import masked_mse


def loss_main(a_pred: Tensor, a_gt: Tensor, valid_mask: Tensor) -> Tensor:
    """L_main — MSE BC loss over valid frames.

    MVP: simple MSE over valid frames; same gradient direction as full
    flow-matching (see CODE_STANDARDS §1.4).  Full flow matching loss
    left for M3+.

    Args:
        a_pred:     ``(B, T, action_dim)`` fp32 — predicted actions.
        a_gt:       ``(B, T, action_dim)`` fp32 — ground-truth actions.
        valid_mask: ``(B, T)`` bool.

    Returns:
        scalar fp32 loss (mean over B × valid-T × action_dim).
    """
    # Cast pred to fp32 for safe loss compute (CODE_STANDARDS §1.7).
    return masked_mse(a_pred.to(torch.float32), a_gt.to(torch.float32), valid_mask)


# Backwards-compatible alias matching the original stub signature.
def flow_match_loss(
    a_pred_seq: list[Tensor] | Tensor,
    a_true: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Alias for :func:`loss_main` accepting a list of per-step (B, action_dim).

    Args:
        a_pred_seq: list-of-T ``(B, action_dim)`` tensors *or* pre-stacked
                    ``(B, T, action_dim)``.
        a_true:     ``(B, T, action_dim)`` fp32.
        valid_mask: ``(B, T)`` bool.

    Returns:
        scalar fp32 loss.
    """
    if isinstance(a_pred_seq, list):
        a_pred = torch.stack(a_pred_seq, dim=1)  # (B, T, action_dim)
    else:
        a_pred = a_pred_seq
    return loss_main(a_pred, a_true, valid_mask)


def loss_aux(attn_entropy: Tensor, lambda_ent: float) -> Tensor:
    """L_aux — ``-λ_ent · attn_entropy.mean()`` (encourage spread over M_work).

    Args:
        attn_entropy: ``(B,)`` fp32 — typically from
                      ``ReadInterface.attn_entropy_to_M_work``.
        lambda_ent:   from :class:`LossConfig`.

    Returns:
        scalar (negative or zero).
    """
    if attn_entropy.numel() == 0:
        return attn_entropy.new_zeros(())
    return -float(lambda_ent) * attn_entropy.float().mean()


# Backwards-compatible alias.
def aux_entropy_loss(attn_entropy: Tensor, lambda_ent: float) -> Tensor:
    return loss_aux(attn_entropy, lambda_ent)


def loss_hcs(
    gamma_pred_geo: Tensor,
    gamma_pred_sem: Tensor,
    gamma_hat_geo: Tensor | None,
    gamma_hat_sem: Tensor | None,
    valid_mask: Tensor,
) -> Tensor:
    """L_HCS — BCE(gamma_pred, sg(gamma_hat)) summed over geo + sem channels.

    Args:
        gamma_pred_geo: ``(B, T)`` fp32 in [0, 1] from [C5].
        gamma_pred_sem: ``(B, T)`` fp32 in [0, 1] from [C5].
        gamma_hat_geo:  ``(B, T)`` fp32 in [0, 1] target (caller must already
                        have applied ``sg(.)``), or None / all-(-1) sentinel
                        if Hindsight not loaded.
        gamma_hat_sem:  same as above, sem channel.
        valid_mask:     ``(B, T)`` bool.

    Returns:
        scalar BCE summed over (geo, sem), mean over B × valid-T.  Returns
        0 if either target is missing or fully sentinel.
    """
    if gamma_hat_geo is None or gamma_hat_sem is None:
        return gamma_pred_geo.new_zeros(())

    # Sentinel handling: -1 marks "no label available"; if every entry is -1
    # we have no signal and skip the loss.
    geo_valid = (gamma_hat_geo >= 0) & valid_mask
    sem_valid = (gamma_hat_sem >= 0) & valid_mask

    if not bool(geo_valid.any()) and not bool(sem_valid.any()):
        return gamma_pred_geo.new_zeros(())

    eps = 1e-6
    pred_g = gamma_pred_geo.clamp(eps, 1 - eps).float()
    pred_s = gamma_pred_sem.clamp(eps, 1 - eps).float()
    tgt_g = gamma_hat_geo.float().clamp(0.0, 1.0)
    tgt_s = gamma_hat_sem.float().clamp(0.0, 1.0)

    bce_g_full = F.binary_cross_entropy(pred_g, tgt_g, reduction="none")
    bce_s_full = F.binary_cross_entropy(pred_s, tgt_s, reduction="none")

    geo_mask_f = geo_valid.float()
    sem_mask_f = sem_valid.float()

    geo_loss = (bce_g_full * geo_mask_f).sum() / geo_mask_f.sum().clamp(min=1.0)
    sem_loss = (bce_s_full * sem_mask_f).sum() / sem_mask_f.sum().clamp(min=1.0)

    return geo_loss + sem_loss


# Backwards-compatible alias accepting pre-summed stack.
def hcs_bce_loss(
    gamma_pred: Tensor,
    gamma_hat: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Single-channel BCE.  Uses :func:`loss_hcs` topology with sem=None."""
    return loss_hcs(gamma_pred, gamma_pred, gamma_hat, gamma_hat, valid_mask) * 0.5


def loss_prh(
    prh_out: dict[int, dict[str, Tensor]] | None,
    future_obs: Tensor | None,
    future_actions: Tensor | None,
    valid_mask: Tensor,
    alpha_a: float,
    horizons: list[int],
    return_per_k: bool = False,
) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
    """L_PRH — Σ_k mean_{B, valid t∈[0,T-k)} ‖ô_{t+k} - o_{t+k}‖² + α_a·‖â_{t+k} - a*_{t+k}‖².

    Each horizon k is averaged over (B × valid-(T-k) × feature_dim) and
    the per-horizon scalars are summed (no further averaging across
    horizons — matches §3.8 spec / CODE_STANDARDS §1.4 reduction rule).

    Horizons k ≥ T are silently skipped (no valid prediction window).

    Args:
        prh_out: ``{k: {'o_hat_seq': (B, T-k, d_h), 'a_hat_seq': (B, T-k,
            action_dim)}}`` produced upstream.  ``None`` short-circuits to 0.
        future_obs:     ``(B, T, d_h)`` fp32 — full per-frame ``h_t`` pool.
            ``loss_prh`` slices ``[:, k:T]`` for each horizon.
        future_actions: ``(B, T, action_dim)`` fp32 — ``a*_t`` (typically
            ``batch['action']``).  Sliced ``[:, k:T]`` per horizon.
        valid_mask:     ``(B, T)`` bool.
        alpha_a:        weight on the action-loss term within L_PRH.
        horizons:       list of k offsets.
        return_per_k:   when True, also return ``{k: per-horizon-loss}``
            (each entry is the per-horizon ``l_obs + α_a·l_act`` scalar
            BEFORE summation).  Skipped horizons / no-window cases are
            absent from the dict.  Default False keeps the single-tensor
            signature for older callers.

    Returns:
        scalar fp32 — 0 if ``prh_out`` is ``None`` / no horizon contributes.
        When ``return_per_k=True``: ``(total, per_k_dict)``.
    """
    zero = valid_mask.new_zeros((), dtype=torch.float32)
    per_k: dict[int, Tensor] = {}
    if prh_out is None or future_obs is None or future_actions is None:
        return (zero, per_k) if return_per_k else zero

    B, T = valid_mask.shape
    total = zero
    any_term = False

    for k in horizons:
        k = int(k)
        if k <= 0 or T - k <= 0:
            continue
        out_k = prh_out.get(k)
        if out_k is None:
            continue
        o_hat = out_k.get("o_hat_seq")
        a_hat = out_k.get("a_hat_seq")
        if o_hat is None or a_hat is None:
            continue

        # Targets: shift forward by k.
        o_target = future_obs[:, k:T].to(torch.float32)        # (B, T-k, d_h)
        a_target = future_actions[:, k:T].to(torch.float32)    # (B, T-k, action_dim)
        sub_mask = valid_mask[:, k:T]                          # (B, T-k)

        if o_hat.shape != o_target.shape or a_hat.shape != a_target.shape:
            raise ValueError(
                f"loss_prh shape mismatch at k={k}: "
                f"o_hat={tuple(o_hat.shape)} vs target={tuple(o_target.shape)}, "
                f"a_hat={tuple(a_hat.shape)} vs target={tuple(a_target.shape)}"
            )

        l_obs = masked_mse(o_hat.to(torch.float32), o_target, sub_mask)
        l_act = masked_mse(a_hat.to(torch.float32), a_target, sub_mask)
        loss_k = l_obs + float(alpha_a) * l_act
        per_k[k] = loss_k
        total = total + loss_k
        any_term = True

    if not any_term:
        return (zero, per_k) if return_per_k else zero
    return (total, per_k) if return_per_k else total


def compute_prh_loss(
    prh_module,
    m_seq: Tensor,
    h_seq: Tensor,
    a_seq: Tensor,
    valid_mask: Tensor,
    horizons: list[int],
    alpha_a: float,
    return_per_k: bool = False,
) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
    """End-to-end PRH loss helper used by :func:`chime_train_step`.

    Forwards ``sg(m_seq[:, :T-k])`` through the PRH module per horizon and
    feeds the result into :func:`loss_prh`.  Skips horizons where
    ``T - k <= 0``.

    Args:
        prh_module: instance of :class:`chime_vla.heads.prh.PRH`.
        m_seq:      ``(B, T, d_h)`` readout vectors (gradient-tracked
            upstream — this helper applies ``.detach()`` per SG-2).
        h_seq:      ``(B, T, d_h)`` per-frame patch-token mean (target).
        a_seq:      ``(B, T, action_dim)`` ground-truth actions.
        valid_mask: ``(B, T)`` bool.
        horizons:   list of k offsets.
        alpha_a:    weight on the action-loss term within L_PRH.
        return_per_k: when True, also return per-horizon ``{k: loss}``
            scalars (each is detach-able for logging).

    Returns:
        scalar fp32 loss; 0 if no horizon yields a valid prediction window.
        When ``return_per_k=True``: ``(total, per_k_dict)``.
    """
    B, T, d_h = m_seq.shape

    # Single forward over all T frames (SG-2: detach m before PRH).
    # PRH returns predictions for every horizon k; per-horizon valid
    # window [0, T-k) is sliced afterwards.
    m_flat = m_seq.detach().reshape(B * T, d_h)
    out = prh_module(m_flat)                          # {k: (o_hat, a_hat)} flat

    prh_out: dict[int, dict[str, Tensor]] = {}
    for k in horizons:
        k = int(k)
        if k <= 0 or T - k <= 0:
            continue
        o_hat_flat, a_hat_flat = out[k]
        # (B*T, d_h) → (B, T, d_h); use only the first T-k frames.
        o_hat = o_hat_flat.reshape(B, T, -1)[:, : T - k]
        a_hat = a_hat_flat.reshape(B, T, -1)[:, : T - k]
        prh_out[k] = {"o_hat_seq": o_hat, "a_hat_seq": a_hat}

    return loss_prh(
        prh_out=prh_out,
        future_obs=h_seq,
        future_actions=a_seq,
        valid_mask=valid_mask,
        alpha_a=alpha_a,
        horizons=horizons,
        return_per_k=return_per_k,
    )


def loss_predict_self_supervised(
    h_hat_pred_seq: Tensor,
    h_target_seq: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """L_predict — self-supervised next-frame prediction MSE for [C5] ψ.

    ψ predicts ``h_t`` from ``M_work^{t-1}`` (a 1-layer GRU over the FIFO).
    In the M2 MVP fallback path (architecture §0.7.4 + §I.4 #1) λ_1 = 0
    permanently — without this self-supervised term, ψ would receive *no*
    gradient signal at all (L_main / L_PRH are SG-blocked from ψ via SG-1
    and SG-2 respectively, and L_aux only trains [C8]'s attention).
    Architecture §0.7.4 explicitly mandates: "[C5 仅 prediction-error
    self-supervised, GRU 实现]".

    Reduction follows CODE_STANDARDS §1.4: mean over (B, valid-T, d_h).
    Caller MUST pass ``h_target_seq`` with ``.detach()`` applied — otherwise
    the gradient would back-propagate into [C1] through the per-frame ``h_t``
    targets, which is forbidden by SG-1 (write heads see ``sg(γ)`` precisely
    so this self-prediction does not back-couple perception).

    Args:
        h_hat_pred_seq: ``(B, T, d_h)`` fp32 — stacked ``c5.last_h_hat_pred``.
            Carries grad — this is the only path that trains ψ on the M2
            fallback YAML.
        h_target_seq:   ``(B, T, d_h)`` fp32 — caller-detached target.  Must
            already be detached; this function does NOT call ``.detach()``
            (so a missing detach in train_step is loud, not silent).
        valid_mask:     ``(B, T)`` bool.

    Returns:
        scalar fp32 loss; 0 if no valid frame.
    """
    if h_hat_pred_seq.shape != h_target_seq.shape:
        raise ValueError(
            f"loss_predict_self_supervised shape mismatch: "
            f"pred={tuple(h_hat_pred_seq.shape)} vs "
            f"target={tuple(h_target_seq.shape)}"
        )
    return masked_mse(
        h_hat_pred_seq.to(torch.float32),
        h_target_seq.to(torch.float32),
        valid_mask,
    )


def loss_csm(csm_w: Tensor | None, beta: float) -> Tensor:
    """L_CSM — -Var_i(w_i) - β · log(Mean_i(w_i) + ε).

    Two terms:
        * ``-Var_i(w_i)`` — maximises slot-importance heterogeneity (we
          *want* CSM to discriminate between slots).
        * ``-β · log Mean_i(w_i)`` — maximises overall slot utilisation
          (prevents the degenerate solution where all slots look
          unimportant).

    Args:
        csm_w: ``(B, n_slots_per_step)`` fp32 importance weights, or None
               (caller passes None when λ_3 == 0 / CSM disabled).
        beta:  log-mean weight from :class:`C12Config`.

    Returns:
        scalar fp32.  Returns 0 when ``csm_w`` is None or empty.
    """
    if csm_w is None or csm_w.numel() == 0:
        return torch.zeros((), dtype=torch.float32)
    eps = 1e-6
    w = csm_w.float().clamp(min=0.0)
    # -Var per row, mean over batch.  unbiased=False so n=1 row is finite.
    var_term = -w.var(dim=-1, unbiased=False).mean()
    log_mean_term = -float(beta) * (w.mean(dim=-1) + eps).log().mean()
    return var_term + log_mean_term


def compute_csm_loss(*args, **kwargs) -> Tensor:
    """Backwards-compatible name.  Returns 0 in M1."""
    return torch.zeros((), dtype=torch.float32)
