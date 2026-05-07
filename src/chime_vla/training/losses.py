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

from chime_vla.config import C11Config, C12Config
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
    prh_out: list | None,
    future_obs: Tensor | None,
    future_actions: Tensor | None,
    valid_mask: Tensor,
    alpha_a: float,
    horizons: list[int],
) -> Tensor:
    """L_PRH — sum_k MSE(o_hat[k] - o[t+k]) + α_a · MSE(a_hat[k] - a*[t+k]).

    M1: PRH not yet wired into train_step; returns 0.  Future M2+ work
    will populate ``prh_out`` and the future tensors.

    Args:
        prh_out:        list-of-T per-step PRH outputs, or None.
        future_obs:     ``(B, T, K, d_h)`` fp32 future observations, or None.
        future_actions: ``(B, T, K, action_dim)`` fp32, or None.
        valid_mask:     ``(B, T)`` bool.
        alpha_a:        weight on the action-loss term within L_PRH.
        horizons:       list of k offsets.

    Returns:
        scalar fp32 — 0 in M1.
    """
    return valid_mask.new_zeros((), dtype=torch.float32)


def compute_prh_loss(*args, **kwargs) -> Tensor:
    """Backwards-compatible name.  Returns 0 in M1."""
    return torch.zeros((), dtype=torch.float32)


def loss_csm(csm_w: Tensor | None, beta: float) -> Tensor:
    """L_CSM — -Var(w_i) - β · log(mean(w_i)).

    M1: CSM not yet implemented; returns 0.

    Args:
        csm_w: ``(B, n_slots_per_step)`` fp32 importance weights, or None.
        beta:  log-mean weight from :class:`C12Config`.

    Returns:
        scalar fp32 — 0 in M1.
    """
    if csm_w is None:
        return torch.zeros((), dtype=torch.float32)
    eps = 1e-6
    w = csm_w.float().clamp(min=eps)
    var_term = -w.var(dim=-1, unbiased=False).mean()
    log_mean_term = -float(beta) * w.mean(dim=-1).clamp(min=eps).log().mean()
    return var_term + log_mean_term


def compute_csm_loss(*args, **kwargs) -> Tensor:
    """Backwards-compatible name.  Returns 0 in M1."""
    return torch.zeros((), dtype=torch.float32)
