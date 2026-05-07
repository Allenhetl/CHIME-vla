"""[C12] Counterfactual Slot Mask (training-only).

Component map: C12 (heads, training-only).  Not an ``nn.Module`` — it is a
*callable* that, for each step, randomly samples ``cfg.n_slots_per_step``
*occupied* entries from :class:`SemBank`, runs the **frozen** [C9] action
expert N+1 times (one baseline + one per masked slot), and returns
per-slot importance weights ``w_i`` proportional to the predicted-action
divergence vs the unmasked baseline.

Outputs feed:
* L_CSM (training-only loss over slot importance distribution)
* CSMLRUEvictor (M3+) for `slot_free`-aware eviction.

SG contract (CODE_STANDARDS SG-4): the only gradient out of CSM flows
through the sampled importance weights into L_CSM; the entire
counterfactual probe runs in ``torch.no_grad()`` so [C9]/[C8] do not
receive any gradient from the L_CSM path (they are the frozen
behavioural reference).

Interface note:
    The original M0 stub signature took ``(m_t, m_sem, frozen_action_expert)``;
    that is insufficient because the leave-one-out probe must *re-read* the
    perturbed M_sem through [C8] before invoking [C9].  This implementation
    therefore expects the wider context required to drive a full
    read+action forward pass.  See the call-site in
    :func:`chime_vla.training.train_step.chime_train_step`.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from chime_vla.action.action_expert import ActionExpert
from chime_vla.config import C12Config
from chime_vla.memory.geo_grid import GeoGrid
from chime_vla.memory.sem_bank import SemBank
from chime_vla.readout.read_interface import ReadInterface


class CSM:
    """Counterfactual slot-importance estimator.

    Not an ``nn.Module`` — callable utility.  ``__init__`` reads only the
    config; the frozen [C8]/[C9] handles are passed at call-time so
    callers can swap snapshots between epochs.
    """

    def __init__(self, cfg: C12Config):
        self.cfg = cfg
        self.n_slots_per_step: int = int(cfg.n_slots_per_step)
        self.beta: float = float(cfg.beta)
        # Side-channel: indices of the slots actually probed in the last
        # __call__ — (B, n_slots_per_step) int64 — useful for downstream
        # CSMLRUEvictor (M3+) and for diagnostics / debugging.
        self.last_sampled_idx: Optional[Tensor] = None

    # ------------------------------------------------------------------ #
    # call                                                               #
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        c_t_base: Tensor,
        h_t_cls: Tensor,
        m_work: Tensor,
        m_geo: GeoGrid,
        m_sem: SemBank,
        read_module: ReadInterface,
        frozen_action_expert: ActionExpert,
        h_t: Optional[Tensor] = None,
    ) -> Tensor:
        """Return per-slot importance ``w_i`` for the sampled subset.

        The probe samples ``cfg.n_slots_per_step`` *occupied* slots per
        batch row, zeros each one in turn (preserving the others), re-runs
        the read interface + frozen action expert, and reports the
        relative L2 distance between the perturbed and baseline actions
        as the slot-importance proxy.

        For deterministic 1-step distill [C9] outputs there is no
        per-sample distribution to KL-divergence; the relative L2 ratio
        ``‖a_base - a_ablate‖_2 / max(‖a_base‖_2, 1e-6)`` is a
        magnitude-normalised stand-in (matches §3.8 spec — KL on
        deterministic regress is undefined).

        Args:
            c_t_base:    ``(B, N_q + K_w, d_h)`` — unperturbed [C8] readout.
            h_t_cls:     ``(B, d_h)`` — pooled current frame for [C9].
            m_work:      ``(B, K_w, N, d_h)`` — current FIFO ring (post-append).
            m_geo:       :class:`GeoGrid` — passed through to read_module.
            m_sem:       :class:`SemBank` — read+temporarily mutated; restored
                         on exit.  Caller need not detach.
            read_module: :class:`ReadInterface` — called under no_grad.
            frozen_action_expert: a *frozen snapshot* of [C9]; CSM calls
                         it under ``torch.no_grad()``.
            h_t:         ``(B, N, d_h)`` — current frame tokens; only used
                         to satisfy the read_module signature.  If ``None``
                         a zeros-tensor with the appropriate shape is
                         constructed (read_module ignores h_t under MVP).

        Returns:
            ``w_i`` : ``(B, n_slots_per_step)`` fp32 — importance weights
            for the *sampled* slots (not for all K_s).  The actual slot
            indices are stored in ``self.last_sampled_idx``.
        """
        if c_t_base.dim() != 3:
            raise ValueError(
                f"[C12] c_t_base expected (B, N_q+K_w, d_h); got {tuple(c_t_base.shape)}"
            )
        B = c_t_base.shape[0]
        device = c_t_base.device
        K_s = m_sem.K_s
        n = self.n_slots_per_step

        # Stand-in h_t for the read_module signature (read_module ignores
        # h_t in MVP — see ReadInterface.forward docstring).
        if h_t is None:
            d_h = c_t_base.shape[-1]
            # We use a zero tensor of the canonical (B, 1, d_h) shape; the
            # forward path doesn't index N so the leading shape is loose.
            h_t = torch.zeros(B, 1, d_h, dtype=m_work.dtype, device=device)

        # Sample n occupied slots per batch row.  When fewer than n slots
        # are occupied we fall back to repeating any occupied slot (so all
        # n probe slots are always valid) — if *no* slot is occupied we
        # return zeros (degenerate case at episode start).
        sampled_idx = self._sample_occupied_slots(m_sem.slot_free, n)
        # sampled_idx: (B, n) int64 on m_sem.slot_free.device

        # Move to compute device & cache for downstream consumers.
        sampled_idx = sampled_idx.to(device=device)
        self.last_sampled_idx = sampled_idx

        # Degenerate: if any row had zero occupied slots, the sampler
        # returned -1 sentinels; clamp to 0 for indexing & emit zero
        # importance for those rows.
        no_occ_mask = (sampled_idx < 0).any(dim=-1)  # (B,)
        sampled_idx_safe = sampled_idx.clamp(min=0)

        # Run the entire probe under no_grad: SG-4 contract.  L_CSM uses
        # the *output* w_i as a constant (it's a scalar derived from
        # detached forward passes), so this also ensures we don't pollute
        # the autograd graph with frozen-expert calls.
        with torch.no_grad():
            # Baseline action.
            a_base = frozen_action_expert(c_t_base, h_t_cls)  # (B, action_dim) fp32
            a_base_norm = a_base.float().norm(dim=-1).clamp(min=1e-6)  # (B,)

            w_list: list[Tensor] = []
            for j in range(n):
                slot_j = sampled_idx_safe[:, j]  # (B,) int64

                # Save & zero the j-th sampled slot for each batch row.
                # m_sem.v: (B, K_s, d_s) fp32.  Indexing m_sem.v[arange(B), slot_j]
                # gives (B, d_s); we save and overwrite in-place.
                b_idx = torch.arange(B, device=device)
                v_save = m_sem.v[b_idx, slot_j].clone()  # (B, d_s) fp32
                m_sem.v[b_idx, slot_j] = 0.0

                # Re-read with perturbed M_sem.  read_module returns
                # (B, N_q + K_w, d_h) bf16.
                c_t_ablate = read_module(m_work, m_geo, m_sem, h_t)
                a_ablate = frozen_action_expert(c_t_ablate, h_t_cls)  # fp32

                # Restore.
                m_sem.v[b_idx, slot_j] = v_save

                # Relative L2 distance as importance proxy.
                diff_norm = (a_base.float() - a_ablate.float()).norm(dim=-1)  # (B,)
                w_j = diff_norm / a_base_norm  # (B,)
                w_list.append(w_j)

            w_i = torch.stack(w_list, dim=-1)  # (B, n) fp32

            # Zero out rows that had no occupied slot (degenerate case).
            if bool(no_occ_mask.any()):
                w_i = w_i.masked_fill(no_occ_mask.unsqueeze(-1), 0.0)

        # Return as fp32 tensor on the compute device.  The caller fuses
        # this into L_CSM via :func:`chime_vla.training.losses.loss_csm`.
        return w_i.float()

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sample_occupied_slots(slot_free: Tensor, n: int) -> Tensor:
        """Sample ``n`` occupied slot indices per batch row, with replacement
        if fewer than ``n`` slots are occupied.  Rows with zero occupied
        slots get ``-1`` in every column (caller masks the output).

        Args:
            slot_free: ``(B, K_s)`` bool — True ⇒ slot is free.
            n:         positive int.

        Returns:
            ``(B, n)`` int64 indices into the K_s axis (or -1 sentinel).
        """
        if n <= 0:
            raise ValueError(f"[C12] n_slots_per_step must be > 0; got {n}")
        B, K_s = slot_free.shape
        device = slot_free.device
        occupied = (~slot_free).to(torch.float32)  # (B, K_s) — 1.0 where occupied.
        n_occ = occupied.sum(dim=-1)               # (B,) fp32

        out = torch.full((B, n), -1, dtype=torch.int64, device=device)

        for b in range(B):
            cnt = int(n_occ[b].item())
            if cnt == 0:
                continue  # leave -1 sentinels for this row
            occ_idx = (~slot_free[b]).nonzero(as_tuple=False).flatten()  # (cnt,)
            if cnt >= n:
                # Sample without replacement.
                perm = torch.randperm(cnt, device=device)[:n]
                out[b] = occ_idx[perm]
            else:
                # Sample without replacement first; pad the rest with
                # uniform-with-replacement draws from the same pool.
                out[b, :cnt] = occ_idx[torch.randperm(cnt, device=device)]
                pad = torch.randint(
                    low=0, high=cnt, size=(n - cnt,), device=device
                )
                out[b, cnt:] = occ_idx[pad]
        return out
