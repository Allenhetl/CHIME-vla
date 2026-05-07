"""[C12] Counterfactual Slot Mask (training-only).

Component map: C12 (heads, training-only).  Not an ``nn.Module`` — it is a
*callable* that, for each step, randomly masks out ``cfg.n_slots_per_step``
entries from :class:`SemBank`, runs the **frozen** [C9] action expert N
times (one per masked slot), and returns per-slot importance weights
``w_i ∈ (0, 1)`` proportional to the predicted-action divergence vs the
unmasked baseline.

Outputs feed:
* L_CSM (training-only loss over slot importance distribution)
* CSMLRUEvictor (M3+) for `slot_free`-aware eviction.

SG contract: this routine wraps its frozen-expert calls in
``torch.no_grad()``; the only gradient out of CSM flows through the
sampled importance weights into L_CSM (which is by construction over a
frozen expert, so does not back-propagate into [C9]).
"""

from __future__ import annotations

import torch
from torch import Tensor

from chime_vla.action.action_expert import ActionExpert
from chime_vla.config import C12Config
from chime_vla.memory.sem_bank import SemBank


class CSM:
    """Counterfactual slot-importance estimator.

    Not an ``nn.Module`` — callable utility.  ``__init__`` reads only the
    config; the frozen action expert is passed at call-time so callers
    can swap snapshots between epochs.

    M0: stub.
    """

    def __init__(self, cfg: C12Config):
        self.cfg = cfg
        self.n_slots_per_step: int = cfg.n_slots_per_step
        self.beta: float = cfg.beta

    def __call__(
        self,
        m_t: Tensor,
        m_sem: SemBank,
        frozen_action_expert: ActionExpert,
    ) -> Tensor:
        """Return per-slot importance ``w_i`` for the sampled subset.

        Args:
            m_t: ``(B, d_h)`` bf16 — readout vector.
            m_sem: :class:`SemBank` — read-only here; CSM only **probes**.
            frozen_action_expert: a *frozen snapshot* of [C9]; CSM calls
                it under ``torch.no_grad()``.

        Returns:
            ``w_i`` : ``(B, n_slots_per_step)`` fp32 — importance weights
            for the *sampled* slots (not for all K_s).  The actual slot
            indices are determined internally and returned via the side
            channel ``self.last_sampled_idx`` (not in M0 stub).
        """
        raise NotImplementedError("[C12] CSM.__call__ — M0 stub")
