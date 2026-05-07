"""5-loss assembly for CHIME-VLA training (CODE_STRUCTURE §7).

All component losses follow the codebase rule (CODE_STANDARDS §1.4):
    reduction = mean over (B, valid-T) with the dataloader's ``valid_mask``.
Implementations should reuse :func:`chime_vla.utils.losses.masked_mse` for
MSE-style targets.

The five (six) losses:
    * L_main : flow-matching action loss (per-step regression on a*)
    * L_HCS  : BCE between [C5] γ and Hindsight γ̂ (sg-target, SG-5)
    * L_PRH  : per-horizon (ô_{t+k}, â_{t+k}) prediction loss
    * L_CSM  : counterfactual slot importance loss
    * L_aux  : −λ_ent · attention entropy over M_work
    * L_GC   : (M5+ optional, MVP off) gradient consistency loss
"""

from __future__ import annotations

import torch
from torch import Tensor

from chime_vla.config import C11Config, C12Config


def flow_match_loss(
    a_pred_seq: list[Tensor] | Tensor,
    a_true: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """L_main — flow-matching regression on actions.

    Args:
        a_pred_seq: list of length T of ``(B, action_dim)`` tensors, or a
                    pre-stacked ``(B, T, action_dim)`` tensor.
        a_true:     ``(B, T, action_dim)`` ground-truth actions.
        valid_mask: ``(B, T)`` bool — True for real frames.

    Returns:
        scalar loss (mean over B × valid-T).
    """
    raise NotImplementedError("training.losses.flow_match_loss — M0 stub")


def hcs_bce_loss(
    gamma_pred: Tensor,
    gamma_hat: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """L_HCS — BCE between [C5] γ prediction and Hindsight γ̂ target.

    Args:
        gamma_pred: ``(B, T)`` fp32 in [0, 1] — predicted by [C5].
        gamma_hat:  ``(B, T)`` fp32 in [0, 1] — Hindsight target.
                    **Caller MUST pass ``sg(gamma_hat)`` per SG-5.**
        valid_mask: ``(B, T)`` bool.

    Returns:
        scalar BCE loss, mean over B × valid-T.
    """
    raise NotImplementedError("training.losses.hcs_bce_loss — M0 stub")


def compute_prh_loss(
    out_seq: list[dict],
    batch: dict[str, Tensor],
    horizons: list[int],
    alpha_a: float,
) -> Tensor:
    """L_PRH — per-horizon prediction loss assembled across the T axis.

    Args:
        out_seq:  list of length T of per-step forward dicts;
                  ``out['prh_out']`` is the dict returned by [C11] PRH.
        batch:    full batch dict (needs h_t cache + actions + valid_mask).
        horizons: e.g. ``[4, 16, 64]`` from :class:`C11Config`.
        alpha_a:  weight on the action-loss term within L_PRH.

    Returns:
        scalar L_PRH summed across horizons (mean over B × valid-T per horizon).
    """
    raise NotImplementedError("training.losses.compute_prh_loss — M0 stub")


def compute_csm_loss(
    out_seq: list[dict],
    m_sem,  # SemBank — circular import, kept untyped here
    frozen_action_expert,  # ActionExpert — same reason
    cfg: C12Config,
) -> Tensor:
    """L_CSM — slot importance log-mean loss (architecture v2.1 §F).

    Args:
        out_seq: per-step forward outputs; ``out['c_t']`` and ``out['m_t']``.
        m_sem:   slot bank (read-only here).
        frozen_action_expert: snapshot of [C9] used inside [C12].
        cfg:     :class:`C12Config`.

    Returns:
        scalar L_CSM.
    """
    raise NotImplementedError("training.losses.compute_csm_loss — M0 stub")


def aux_entropy_loss(attn_entropy: Tensor, lambda_ent: float) -> Tensor:
    """L_aux — ``-λ_ent · attn_entropy.mean()`` (encourage spread over M_work).

    Args:
        attn_entropy: ``(B,)`` fp32 — from
                      ``ReadInterface.attn_entropy_to_M_work``.
        lambda_ent:   from :class:`LossConfig`.

    Returns:
        scalar (negative or zero).
    """
    raise NotImplementedError("training.losses.aux_entropy_loss — M0 stub")
