"""One-step forward + 5-loss assembly (CODE_STRUCTURE §3.10, §7).

Top-level entry point that the LightningModule wraps in
``training_step``.  Implements the per-frame forward order
(CODE_STANDARDS §1.3):

    C1 → C5 → C2.append → {C3, C4} → C8 → C9 → loss

with the SG topology:
    SG-1: write heads receive ``sg(γ_geo) / sg(γ_sem)``
    SG-2: PRH receives ``sg(m_t)`` (M2+; not exercised in M1)
    SG-5: BCE target γ̂ wrapped in ``sg(.)`` before L_HCS
"""

from __future__ import annotations

from typing import Optional, Protocol

import torch
from torch import Tensor

from chime_vla.config import ChimeConfig
from chime_vla.memory.geo_grid import GeoGrid
from chime_vla.memory.sem_bank import SemBank
from chime_vla.perception.fifo_buffer import WorkBuffer
from chime_vla.training.losses import (
    compute_prh_loss,
    loss_aux,
    loss_csm,
    loss_hcs,
    loss_main,
    loss_predict_self_supervised,
)
from chime_vla.training.schedules import lambda_1_schedule


class ChimeVlaModule(Protocol):
    """Structural type — anything exposing the component handles.

    Concrete impl: :class:`chime_vla.training.lightning_module.ChimeVlaLightning`.
    """

    c1: object
    c3: object
    c4: object
    c5: object
    c8: object
    c9: object


def _detach_memory(m_geo: GeoGrid, m_sem: SemBank, c2: WorkBuffer) -> None:
    """In-place BPTT truncation: detach memory state from autograd graph.

    Called every ``cfg.train.bptt_truncate`` steps inside the per-episode
    loop.  Without this, a 200-step LIBERO episode would build up a graph
    too large for backward (OOM on a 24 GB GPU).
    """
    # Geo grids fp32 — detach in-place via copy.
    for L, g in m_geo.grids.items():
        m_geo.grids[L] = g.detach()
    # Sem bank values fp32.
    m_sem.v = m_sem.v.detach()
    # WorkBuffer ring (bf16) — detach.
    c2.buffer = c2.buffer.detach()


def chime_train_step(
    batch: dict[str, Tensor],
    model: ChimeVlaModule,
    cfg: ChimeConfig,
    step: int,
) -> dict[str, Tensor]:
    """Run one full sequence forward + assemble the 5-loss total.

    Args:
        batch: ``{rgb / proprio / action / sub_task_id / episode_id /
                  valid_mask}`` plus optional ``gamma_hat_geo``,
                  ``gamma_hat_sem`` (each ``(B, T)`` fp32).
        model: the LightningModule (or any structurally-typed forward).
        cfg:   :class:`ChimeConfig`.
        step:  current global training step (drives λ_1 schedule).

    Returns:
        dict with keys ``L_main``, ``L_HCS``, ``L_PRH``, ``L_CSM``,
        ``L_aux``, ``total``, ``lambda_1`` plus diagnostics.
    """
    rgb = batch["rgb"]                # (B, T, 3, 224, 224) float32 or uint8
    proprio = batch["proprio"]        # (B, T, 8) fp32
    action_gt = batch["action"]       # (B, T, 8) fp32
    valid_mask = batch["valid_mask"]  # (B, T) bool

    if rgb.dim() != 5:
        raise ValueError(
            f"chime_train_step expected rgb (B, T, 3, 224, 224); "
            f"got {tuple(rgb.shape)}"
        )

    B, T = rgb.shape[:2]
    device = rgb.device

    # ---- (re-)instantiate non-Module memory containers ----
    c2 = WorkBuffer(cfg.c2, batch_size=B, device=device)
    m_geo = GeoGrid(cfg.c6, batch_size=B, d_g=cfg.c6.d_g, device=device)
    m_sem = SemBank(cfg.c7, batch_size=B, device=device)

    a_pred_steps: list[Tensor] = []
    gamma_geo_steps: list[Tensor] = []
    gamma_sem_steps: list[Tensor] = []
    attn_entropy_steps: list[Tensor] = []
    m_t_steps: list[Tensor] = []      # (B, d_h) per t — for PRH input
    h_pool_steps: list[Tensor] = []   # (B, d_h) per t — for PRH obs target
    h_hat_pred_steps: list[Tensor] = []  # (B, d_h) per t — for L_predict (M2 fallback)

    bptt_n = max(1, int(cfg.train.bptt_truncate))

    for t in range(T):
        rgb_t = rgb[:, t]            # (B, 3, 224, 224)
        proprio_t = proprio[:, t]    # (B, 8)

        # 1) [C1] perception — h_t (B, N, d_h) bf16
        h_t = model.c1(rgb_t, proprio_t)

        # 2) [C5] ESPC — uses M_work BEFORE the current frame's append (§1.3).
        #    Pre-append snapshot — first frame is all-zero ring (matches contract).
        m_work_prev = c2.snapshot()
        gamma_geo, gamma_sem = model.c5(h_t, m_work_prev)
        # Capture ψ's prediction for the M2 self-supervised L_predict path
        # (architecture §0.7.4).  Tensor still carries grad through ψ.
        h_hat_pred_t = model.c5.last_h_hat_pred
        if h_hat_pred_t is not None:
            h_hat_pred_steps.append(h_hat_pred_t)

        # 3) [C2] append.  WorkBuffer.append mutates c2.buffer in-place; for
        #    autograd safety in a sequential per-step loop we build the
        #    post-append ring as a fresh tensor and keep c2.buffer in sync.
        if c2.K_w > 1:
            shifted = torch.cat([c2.buffer[:, 1:], h_t.to(c2.buffer.dtype).unsqueeze(1)], dim=1)
        else:
            shifted = h_t.to(c2.buffer.dtype).unsqueeze(1)
        c2.buffer = shifted
        c2._n_appended = torch.clamp(c2._n_appended + 1, max=c2.K_w)
        m_work_post = shifted

        # 4) [C3]/[C4] writes (SG-1: detach γ before passing to write heads).
        # M3+: the write paths use OUT-OF-PLACE scatter / add and reassign
        # ``m_geo.grids[L]`` / ``m_sem.v``, so they participate in the
        # autograd graph cleanly.  L_main now flows
        #     L_main → [C9] → [C8] → m_geo.grids / m_sem.v → [C3]/[C4]
        # which is the M3 deliverable (write-head grad-norm > 1e-5).
        # SG-1 still applies to ``γ_geo / γ_sem`` (detached below); ``h_t``
        # itself stays gradient-tracked so [C1] receives signal through the
        # write heads' projections.
        gamma_geo_sg = gamma_geo.detach()
        gamma_sem_sg = gamma_sem.detach()
        model.c3(h_t, gamma_geo_sg, m_geo, step=t)
        model.c4(h_t, gamma_sem_sg, m_sem, step=t)

        # 5) [C8] readout — c_t (B, N_q + K_w, d_h)
        c_t = model.c8(m_work_post, m_geo, m_sem, h_t)

        # 6) [C9] action expert — a_pred (B, action_dim)
        h_t_cls = h_t.mean(dim=1)  # (B, d_h)
        a_pred_t = model.c9(c_t, h_t_cls)
        a_pred_steps.append(a_pred_t)

        gamma_geo_steps.append(gamma_geo)
        gamma_sem_steps.append(gamma_sem)

        # PRH bookkeeping (M2+).  m_t = mean over readout tokens (post-C8);
        # detached/sg applied later inside compute_prh_loss per SG-2.
        # h_pool = mean over patch tokens (the predicted future obs target).
        m_t_steps.append(c_t.mean(dim=1))
        h_pool_steps.append(h_t.mean(dim=1))

        # entropy diagnostic / L_aux signal
        try:
            ent_t = model.c8.attn_entropy_to_M_work
            if ent_t is not None:
                attn_entropy_steps.append(ent_t)
        except RuntimeError:
            pass

        # ---- BPTT truncation (CODE_STANDARDS §1.8) ----
        if (t + 1) % bptt_n == 0 and (t + 1) < T:
            _detach_memory(m_geo, m_sem, c2)

    # ---- stack per-step outputs ----
    a_pred = torch.stack(a_pred_steps, dim=1)              # (B, T, action_dim)
    gamma_pred_geo = torch.stack(gamma_geo_steps, dim=1)   # (B, T)
    gamma_pred_sem = torch.stack(gamma_sem_steps, dim=1)   # (B, T)
    if len(attn_entropy_steps) > 0:
        attn_entropy = torch.stack(attn_entropy_steps, dim=0).mean(dim=0)  # (B,)
    else:
        attn_entropy = a_pred.new_zeros((B,))

    # ---- losses ----
    L_main = loss_main(a_pred, action_gt, valid_mask)

    gamma_hat_geo = batch.get("gamma_hat_geo")
    gamma_hat_sem = batch.get("gamma_hat_sem")
    L_HCS = loss_hcs(
        gamma_pred_geo,
        gamma_pred_sem,
        gamma_hat_geo,
        gamma_hat_sem,
        valid_mask,
    )
    # ---- L_PRH (M2+) ----
    # Skip the C11 forward entirely when λ_2 == 0 to avoid spending FLOPs
    # / autograd memory on a term that won't contribute.  M0/M1 smoke YAMLs
    # set lambda_2=0; M2 m2_prh_only.yaml flips it to 1.0.
    lam2 = float(cfg.loss.lambda_2)
    L_PRH_per_k: dict[int, Tensor] = {}
    if lam2 != 0.0 and len(m_t_steps) > 0 and getattr(model, "c11", None) is not None:
        m_seq = torch.stack(m_t_steps, dim=1)        # (B, T, d_h)
        h_seq = torch.stack(h_pool_steps, dim=1)     # (B, T, d_h)
        L_PRH, L_PRH_per_k = compute_prh_loss(
            model.c11,
            m_seq,
            h_seq,
            action_gt,
            valid_mask,
            cfg.c11.horizons,
            cfg.c11.alpha_a,
            return_per_k=True,
        )
    else:
        L_PRH = a_pred.new_zeros(())

    # ---- L_CSM (M3+) ----
    # Skip the C12 leave-one-out probe when λ_3 == 0 (M1 default) — the
    # probe is expensive (n_slots_per_step extra read+action forwards
    # per step) and contributes nothing to the gradient when its weight
    # is zero.  When λ_3 != 0 we run it on the *last* timestep only
    # (per architecture v2.1: "每 mini-batch 抽 4 slot") against the
    # frozen [C9] snapshot if available, else against the live [C9]
    # (which carries its own freeze_base contract).
    lam3 = float(cfg.loss.lambda_3)
    csm_w: Tensor | None = None
    if (
        lam3 != 0.0
        and getattr(model, "c12", None) is not None
        and getattr(model, "c8", None) is not None
        and getattr(model, "c9", None) is not None
        and len(a_pred_steps) > 0
    ):
        last_t = T - 1
        # Re-derive the last timestep's c_t / h_pool from the cached
        # readout state.  m_t_steps holds c_t.mean(dim=1) which lost the
        # token axis; we instead recompute c_t at the end-of-loop state
        # of memory containers (m_work_post / m_geo / m_sem are already
        # at their final values after the for-loop terminated).
        h_t_last = model.c1(rgb[:, last_t], proprio[:, last_t])
        h_t_cls_last = h_t_last.mean(dim=1)
        c_t_last = model.c8(c2.buffer, m_geo, m_sem, h_t_last)
        frozen_c9 = getattr(model, "c9_frozen", None) or model.c9
        csm_w = model.c12(
            c_t_last,
            h_t_cls_last,
            c2.buffer,
            m_geo,
            m_sem,
            model.c8,
            frozen_c9,
            h_t_last,
        )

    L_CSM = loss_csm(csm_w, cfg.c12.beta)
    L_aux = loss_aux(attn_entropy, cfg.loss.lambda_ent)

    # ---- L_predict (M2 MVP fallback, architecture §0.7.4) ----
    # Self-supervised next-frame prediction trains ψ when L_HCS is permanently
    # off.  Skip when λ_predict == 0 to avoid the autograd memory cost on
    # legacy YAMLs that don't set the field.
    lam_predict = float(getattr(cfg.loss, "lambda_predict", 0.0))
    if lam_predict != 0.0 and len(h_hat_pred_steps) == T:
        h_hat_pred_seq = torch.stack(h_hat_pred_steps, dim=1)        # (B, T, d_h)
        # Target = each step's ACTUAL h_t pooled over N.  We've already
        # accumulated this in h_pool_steps.  Detach to enforce SG (ψ's grad
        # never flows back into [C1] via the target).
        h_target_seq = torch.stack(h_pool_steps, dim=1).detach()     # (B, T, d_h)
        L_predict = loss_predict_self_supervised(
            h_hat_pred_seq, h_target_seq, valid_mask
        )
    else:
        L_predict = a_pred.new_zeros(())

    lam1 = lambda_1_schedule(step, cfg.loss)

    total = (
        L_main
        + lam1 * L_HCS
        + lam2 * L_PRH
        + float(cfg.loss.lambda_3) * L_CSM
        + lam_predict * L_predict
        + L_aux
    )

    # ---- diagnostics ----
    with torch.no_grad():
        gamma_geo_mean = gamma_pred_geo.float().mean()
        gamma_sem_mean = gamma_pred_sem.float().mean()
        # M_geo occupancy: fraction of voxels with non-zero norm at level 0
        try:
            occ = m_geo.occupancy_pct()
            geo_occ = float(next(iter(occ.values()))) if occ else 0.0
        except Exception:
            geo_occ = 0.0
        # M_sem occupancy: fraction of slots not free
        sem_occ = (~m_sem.slot_free).float().mean().item()

    out = {
        "L_main": L_main,
        "L_HCS": L_HCS if isinstance(L_HCS, Tensor) else torch.as_tensor(L_HCS),
        "L_PRH": L_PRH if isinstance(L_PRH, Tensor) else torch.as_tensor(L_PRH),
        "L_CSM": L_CSM if isinstance(L_CSM, Tensor) else torch.as_tensor(L_CSM),
        "L_aux": L_aux if isinstance(L_aux, Tensor) else torch.as_tensor(L_aux),
        "L_predict": L_predict if isinstance(L_predict, Tensor) else torch.as_tensor(L_predict),
        "total": total,
        "lambda_1": torch.as_tensor(lam1, dtype=torch.float32),
        "gamma_geo_mean": gamma_geo_mean,
        "gamma_sem_mean": gamma_sem_mean,
        "M_geo_occupancy": torch.as_tensor(geo_occ, dtype=torch.float32),
        "M_sem_occupancy": torch.as_tensor(sem_occ, dtype=torch.float32),
        "attn_entropy_M_work": attn_entropy.mean().detach() if attn_entropy.numel() else torch.zeros(()),
        # Per-horizon L_PRH breakdown (M3+ logging).  Only the scalar
        # values are kept — backward graph for the loss term is owned by
        # the summed ``L_PRH`` above.
        "L_PRH_per_k": {int(k): v.detach() for k, v in L_PRH_per_k.items()},
    }
    return out
