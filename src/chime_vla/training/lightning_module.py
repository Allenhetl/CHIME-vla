"""LightningModule wrapper for CHIME-VLA training (CODE_STANDARDS §0 / §5).

Holds the trainable components as submodules, owns the optimiser /
scheduler, and dispatches ``training_step`` / ``validation_step`` to
:func:`chime_vla.training.train_step.chime_train_step`.

Per-step state (M_work / M_geo / M_sem) is *not* a Lightning buffer —
it is rebuilt at each ``chime_train_step`` (per-batch episode boundary)
so DDP doesn't have to gather episode-scoped tensors.

The frozen [C9] snapshot used by [C12] CSM lives at ``self.c9_frozen``
and is refreshed periodically (M3+ work).
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
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
from chime_vla.training.train_step import chime_train_step


class ChimeVlaLightning(pl.LightningModule):
    """Top-level LightningModule for CHIME-VLA training.

    M0/M1: instantiates all 13 submodules in ``__init__`` so isinstance and
    parameter-counting checks work.  M1 implements ``training_step`` and
    ``configure_optimizers`` end-to-end (B=2, T=64 smoke).
    """

    def __init__(self, cfg: ChimeConfig):
        super().__init__()
        # save_hyperparameters: pass plain dict so OmegaConf nodes don't trip.
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(cfg, DictConfig):
                hp = OmegaConf.to_container(cfg, resolve=True)
            else:
                # dataclass — stringify by attribute access; Lightning is lax
                hp = {"milestone": getattr(cfg, "milestone", "M0")}
        except Exception:
            hp = {}
        try:
            self.save_hyperparameters(hp)
        except Exception:
            # Lightning quirks under different config types — skip silently.
            pass

        self.cfg = cfg

        # Components.  C2/C6/C7 are bare classes, instantiated per-batch
        # inside chime_train_step (not nn.Modules).
        self.c1: VLMBackbone = VLMBackbone(cfg.c1)
        self.c3: GeoWriteHead = GeoWriteHead(
            cfg.c3, d_h=self.c1.d_h, d_g=cfg.c6.d_g, alpha_l=cfg.c6.alpha_l
        )
        self.c4: SemWriteHead = SemWriteHead(
            cfg.c4, d_h=self.c1.d_h, d_s=cfg.c7.d_s, K_s=cfg.c7.K_s
        )
        self.c5: ESPC = ESPC(cfg.c5, d_h=self.c1.d_h)
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

        # Frozen [C9] snapshot for CSM (M3+ refresh).  None at init.
        self.c9_frozen: ActionExpert | None = None

    # ----- Lightning hooks -----

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Inference-mode forward (deploy path).  M1: not used in training."""
        raise NotImplementedError(
            "ChimeVlaLightning.forward — deploy/inference path is M3+; "
            "training_step uses chime_train_step directly."
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Single training-step wrapper around :func:`chime_train_step`."""
        step = int(self.global_step)
        out = chime_train_step(batch, self, self.cfg, step=step)

        # Logging.  prog_bar=True for the few we want eyeballs on.
        bs = batch["rgb"].shape[0] if isinstance(batch.get("rgb"), Tensor) else None
        log_kw = dict(on_step=True, on_epoch=False, batch_size=bs)
        self.log("train/loss", out["total"], prog_bar=True, **log_kw)
        self.log("train/L_main", out["L_main"], prog_bar=False, **log_kw)
        self.log("train/L_HCS", out["L_HCS"], prog_bar=False, **log_kw)
        self.log("train/L_PRH", out["L_PRH"], prog_bar=False, **log_kw)
        # M3+ per-horizon L_PRH breakdown — lets us see whether each k
        # learns independently (deliverable: per-k loss decreases).  Skipped
        # silently when L_PRH is the zeroed short-circuit (λ_2==0 path).
        per_k = out.get("L_PRH_per_k") or {}
        for k, loss_k in per_k.items():
            self.log(f"train/L_PRH_k{int(k)}", loss_k, prog_bar=False, **log_kw)
        self.log("train/L_CSM", out["L_CSM"], prog_bar=False, **log_kw)
        self.log("train/L_aux", out["L_aux"], prog_bar=False, **log_kw)
        if "L_predict" in out:
            self.log("train/L_predict", out["L_predict"], prog_bar=False, **log_kw)
        self.log("train/lambda_1", out["lambda_1"], prog_bar=False, **log_kw)
        self.log("train/gamma_geo", out["gamma_geo_mean"], prog_bar=False, **log_kw)
        self.log("train/gamma_sem", out["gamma_sem_mean"], prog_bar=False, **log_kw)
        self.log("train/M_geo_occ", out["M_geo_occupancy"], prog_bar=False, **log_kw)
        self.log("train/M_sem_occ", out["M_sem_occupancy"], prog_bar=False, **log_kw)
        self.log("train/attn_ent", out["attn_entropy_M_work"], prog_bar=False, **log_kw)

        # Update ESPC EMA (post-forward, pre-optim — fine for MVP smoke).
        try:
            self.c5.update_ema()
        except Exception:
            pass

        return out["total"]

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict:
        """Per-task SR / loss validation.  M1: just compute losses."""
        out = chime_train_step(batch, self, self.cfg, step=int(self.global_step))
        bs = batch["rgb"].shape[0] if isinstance(batch.get("rgb"), Tensor) else None
        self.log("val/loss", out["total"], on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("val/L_main", out["L_main"], on_epoch=True, batch_size=bs)
        return out

    def configure_optimizers(self) -> Any:
        """AdamW with cosine schedule + linear warmup (per :class:`TrainConfig`)."""
        train = self.cfg.train
        params = [p for p in self.parameters() if p.requires_grad]
        betas = tuple(train.betas) if not isinstance(train.betas, tuple) else train.betas
        optim = torch.optim.AdamW(
            params,
            lr=float(train.lr),
            betas=betas,
            weight_decay=float(train.wd),
        )

        # Cosine + linear warmup via LambdaLR for portability.
        warmup = max(0, int(train.warmup_steps))

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(max(1, warmup))
            return 1.0

        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
