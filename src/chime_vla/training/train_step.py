"""One-step forward + 5-loss assembly (CODE_STRUCTURE §3.10, §7).

Top-level entry point that the LightningModule wraps in
``training_step``.  Implements the per-frame forward order
(CODE_STANDARDS §1.3):

    C1 → C5 → C2.append → {C3, C4} → C8 → C9 → loss

with the SG topology:
    SG-1: write heads receive ``sg(γ_geo) / sg(γ_sem)``
    SG-2: PRH receives ``sg(m_t)``
    SG-5: BCE target γ̂ wrapped in ``sg(.)`` before L_HCS

This file also defines the ``ChimeVlaModule`` typing protocol so
:func:`chime_train_step` can take a structurally-typed forward model
without circular imports.
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

from chime_vla.config import ChimeConfig


class ChimeVlaModule(Protocol):
    """Structural type — anything exposing the 13 component handles.

    Concrete impl: :class:`chime_vla.training.lightning_module.ChimeVlaLightning`.
    """

    c1: object
    c2: object
    c3: object
    c4: object
    c5: object
    c8: object
    c9: object
    c11: object
    c9_frozen: object  # frozen [C9] snapshot for CSM


def chime_train_step(
    batch: dict[str, Tensor],
    model: ChimeVlaModule,
    cfg: ChimeConfig,
    step: int,
) -> dict[str, Tensor]:
    """Run one full sequence forward + assemble the 5-loss total.

    Args:
        batch: ``{rgb / proprio / action / sub_task_id / episode_id /
                  valid_mask}`` plus, when ``cfg.hindsight.enabled``,
                  ``gamma_hat_geo`` and ``gamma_hat_sem`` (each ``(B, T)``
                  fp32).  Shapes per ``CODE_STRUCTURE.md §5``.
        model: the LightningModule (or any structurally-typed forward).
        cfg:   :class:`ChimeConfig`.
        step:  current global training step (drives λ_1 schedule).

    Returns:
        dict with keys ``L_main``, ``L_HCS``, ``L_PRH``, ``L_CSM``,
        ``L_aux``, ``total``, ``lambda_1`` plus diagnostics
        (``gamma_geo_mean``, ``M_geo_occupancy_pct``, ...).
    """
    raise NotImplementedError(
        "[train_step] chime_train_step — M0 stub; "
        "see CODE_STRUCTURE.md §7 for canonical pseudocode."
    )
