"""Deterministic seeding helper.  Lightning's seed_everything is preferred at
the trainer entry point; this is a thin wrapper for non-Lightning scripts."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python / NumPy / Torch (CPU + CUDA).  Does not toggle determinism flags."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
