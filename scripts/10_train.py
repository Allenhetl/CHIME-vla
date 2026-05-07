#!/usr/bin/env python
"""Hydra training entry — CHIME-VLA (CODE_STRUCTURE.md §1).

Usage::

    python scripts/10_train.py
    python scripts/10_train.py +experiment=mvp_libero_long
    python scripts/10_train.py milestone=M2 +train=m2_phi_only

M0: stub.  Resolves config and prints it; the trainer-fit call raises
NotImplementedError until M1 deliverables (LightningModule + datamodule
real impl) land.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
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
    """Hydra entry point — print resolved config + raise NotImplementedError."""
    print("=" * 60)
    print("CHIME-VLA training entry (M0 stub)")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    raise NotImplementedError(
        "scripts/10_train.py — M0 stub.  "
        "Real fit() lands at M1 once datamodule + lightning_module are impl."
    )


if __name__ == "__main__":
    main()
