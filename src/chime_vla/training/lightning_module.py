"""LightningModule wrapper for CHIME-VLA training (CODE_STANDARDS §0 / §5).

Holds all 13 components as submodules, owns the optimiser / scheduler,
and dispatches ``training_step`` / ``validation_step`` to
:func:`chime_vla.training.train_step.chime_train_step`.

Per-step state (M_work / M_geo / M_sem) is *not* a Lightning buffer —
it is rebuilt at each ``on_train_batch_start`` via
:func:`chime_vla.utils.memory_reset.reset_memory`, so DDP doesn't have
to gather episode-scoped tensors.

The frozen [C9] snapshot used by [C12] CSM lives at ``self.c9_frozen``
and is refreshed every N steps (config TBD post-M3).
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch.nn as nn
from torch import Tensor

from chime_vla.action.action_expert import ActionExpert
from chime_vla.config import ChimeConfig
from chime_vla.heads.csm import CSM
from chime_vla.heads.espc import ESPC
from chime_vla.heads.geo_write import GeoWriteHead
from chime_vla.heads.prh import PRH
from chime_vla.heads.sem_write import SemWriteHead
from chime_vla.perception.fifo_buffer import WorkBuffer  # noqa: F401  (typing)
from chime_vla.perception.vlm_backbone import VLMBackbone
from chime_vla.readout.read_interface import ReadInterface


class ChimeVlaLightning(pl.LightningModule):
    """Top-level LightningModule.

    M0: stub — ``training_step`` / ``configure_optimizers`` / ``forward``
    raise ``NotImplementedError``.  ``__init__`` does construct all 13
    submodules so ``isinstance(self, ChimeVlaLightning)`` checks and
    parameter-counting tests work.
    """

    def __init__(self, cfg: ChimeConfig):
        super().__init__()
        self.save_hyperparameters(ignore=[])
        self.cfg = cfg

        # 13 components (10 are nn.Module subclasses; C2/C6/C7 are
        # bare classes, instantiated per-batch by reset_memory).
        self.c1: VLMBackbone = VLMBackbone(cfg.c1)
        # c2 = WorkBuffer — instantiated per-batch (not an nn.Module)
        self.c3: GeoWriteHead = GeoWriteHead(
            cfg.c3, d_h=self.c1.d_h, d_g=cfg.c6.d_g, alpha_l=cfg.c6.alpha_l
        )
        self.c4: SemWriteHead = SemWriteHead(
            cfg.c4, d_h=self.c1.d_h, d_s=cfg.c7.d_s, K_s=cfg.c7.K_s
        )
        self.c5: ESPC = ESPC(cfg.c5, d_h=self.c1.d_h)
        # c6 = GeoGrid — instantiated per-batch
        # c7 = SemBank — instantiated per-batch
        self.c8: ReadInterface = ReadInterface(
            cfg.c8, d_h=self.c1.d_h, d_s=cfg.c7.d_s,
            K_w=cfg.c2.K_w, K_s=cfg.c7.K_s,
        )
        self.c9: ActionExpert = ActionExpert(
            cfg.c9, d_h=self.c1.d_h, action_dim=cfg.data.action_dim
        )
        self.c11: PRH = PRH(
            cfg.c11, d_h=self.c1.d_h, action_dim=cfg.data.action_dim
        )
        self.c12: CSM = CSM(cfg.c12)  # callable, not nn.Module

        # Frozen [C9] snapshot for CSM.  Initialised to a clone of c9 at
        # build time; refreshed periodically by training callback.
        self.c9_frozen: ActionExpert | None = None

    # ----- Lightning hooks (M0 stubs) -----

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Inference-mode forward (deploy path).  M0: stub."""
        raise NotImplementedError("ChimeVlaLightning.forward — M0 stub")

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Wraps :func:`chime_train_step` and returns ``total`` for backward.

        M0: stub.
        """
        raise NotImplementedError("ChimeVlaLightning.training_step — M0 stub")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict:
        """Per-task SR / loss validation.  M0: stub."""
        raise NotImplementedError("ChimeVlaLightning.validation_step — M0 stub")

    def configure_optimizers(self) -> Any:
        """AdamW + cosine warmup per :class:`TrainConfig`.  M0: stub."""
        raise NotImplementedError("ChimeVlaLightning.configure_optimizers — M0 stub")
