#!/usr/bin/env python
"""Hydra training entry — CHIME-VLA (CODE_STRUCTURE.md §1).

Usage::

    python scripts/10_train.py
    python scripts/10_train.py +experiment=mvp_libero_long
    python scripts/10_train.py milestone=M1 +train=m1_smoke

M1 deliverable — runs Lightning fit() on the LIBERO-Long cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# Add src to path when run uninstalled (defensive; pip install -e . is preferred).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chime_vla.config import register_config  # noqa: E402

# Register structured config schema with Hydra ConfigStore.
register_config()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Hydra entry point — instantiate datamodule + lightning module + Trainer."""
    print("=" * 60)
    print(f"CHIME-VLA training | milestone={cfg.milestone} "
          f"| experiment={cfg.experiment_name}")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(int(cfg.seed), workers=True)

    # Lazy imports (avoid heavy chime_vla load during --help).
    from chime_vla.config import ChimeConfig  # noqa: E402
    from chime_vla.training.datamodule import LiberoLongDataModule
    from chime_vla.training.lightning_module import ChimeVlaLightning

    # Materialize a real `ChimeConfig` dataclass so all fields with defaults
    # are present (Hydra YAMLs only carry overrides; the dataclass schema is
    # the source of truth for everything else).  We merge by dumping cfg →
    # plain dict and then OmegaConf.merge with the dataclass schema.
    cfg_plain = OmegaConf.to_container(cfg, resolve=True) or {}
    schema = OmegaConf.structured(ChimeConfig())
    OmegaConf.set_struct(schema, False)
    merged = OmegaConf.merge(schema, cfg_plain)
    OmegaConf.set_struct(merged, False)

    dm = LiberoLongDataModule(merged, batch_size=int(merged.train.bs))
    module = ChimeVlaLightning(merged)

    # Trainer kwargs — assemble from cfg.trainer (if present) and sane defaults.
    trainer_kwargs: dict = {}
    if "trainer" in cfg:
        trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True) or {}

    trainer_kwargs.setdefault("max_epochs", int(merged.train.max_epochs))
    trainer_kwargs.setdefault("accelerator", "auto")
    trainer_kwargs.setdefault("devices", 1)
    # bf16-mixed precision per CODE_STANDARDS §1.7.  Override via cfg.trainer.
    trainer_kwargs.setdefault("precision", str(merged.train.precision))
    trainer_kwargs.setdefault("gradient_clip_val", float(merged.train.grad_clip))
    trainer_kwargs.setdefault("accumulate_grad_batches", int(merged.train.accumulate))
    trainer_kwargs.setdefault("log_every_n_steps", 1)
    trainer_kwargs.setdefault("enable_checkpointing", False)

    # Default to a CSVLogger alongside the TB logger so M2 analysis scripts
    # can read step-by-step metrics from a stable schema (lightning_logs/
    # version_*/metrics.csv).  Skip if the user has already configured logger.
    if "logger" not in trainer_kwargs:
        from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
        trainer_kwargs["logger"] = [
            TensorBoardLogger(save_dir="lightning_logs", name=""),
            CSVLogger(save_dir="lightning_logs", name=""),
        ]

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, datamodule=dm)

    # Always save final checkpoint (M1 E1 needs trained model for re-judgment).
    out_dir = Path(merged.output_root) / "runs" / merged.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "last.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    print(f"Training done. Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
