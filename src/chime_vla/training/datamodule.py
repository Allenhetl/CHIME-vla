"""LiberoLongDataModule — Lightning DataModule for LIBERO-Long
(CODE_STRUCTURE §5, CODE_STANDARDS §5).

Reads per-episode ``.pt`` cache files produced by
``scripts/00_build_libero_cache.py`` and (when ``cfg.hindsight.enabled``)
joins each episode with its Hindsight γ̂ via
:class:`chime_vla.hindsight.HindsightConsumer`.

Splits: 8/1/1 LIBERO-Long, fixed seed.  Sampler:
``TaskBalancedLengthBucketSampler`` (rank-strided for DDP, inherited from
Hindsight CODE_STANDARDS §5.6).
"""

from __future__ import annotations

from typing import Any, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from chime_vla.config import ChimeConfig
from chime_vla.hindsight.consumer import HindsightConsumer


class _LiberoEpisodeDataset(Dataset):
    """One row = one episode, returns the canonical batch dict.

    M0: stub — :meth:`__getitem__` raises.
    """

    def __init__(self, cfg: ChimeConfig, split: str):
        self.cfg = cfg
        self.split = split
        self.consumer: Optional[HindsightConsumer] = None
        if cfg.hindsight.enabled:
            self.consumer = HindsightConsumer(
                root=cfg.hindsight.gamma_hat_root,
                strategy=cfg.hindsight.strategy,
                task=cfg.hindsight.task,
            )

    def __len__(self) -> int:
        raise NotImplementedError("_LiberoEpisodeDataset.__len__ — M0 stub")

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError("_LiberoEpisodeDataset.__getitem__ — M0 stub")


class LiberoLongDataModule(pl.LightningDataModule):
    """LIBERO-Long DataModule.

    Outputs batch dict (CODE_STRUCTURE §5):
        rgb            : (B, T, 3, 224, 224) float32  OR
        rgb_feature    : (B, T, N=256, d_h=1152) fp16
        proprio        : (B, T, 8) fp32
        action         : (B, T, 8) fp32
        sub_task_id    : (B, T) int32
        episode_id     : (B,) int
        valid_mask     : (B, T) bool
        gamma_hat_geo  : (B, T) fp32   if hindsight.enabled
        gamma_hat_sem  : (B, T) fp32   if hindsight.enabled
    """

    def __init__(self, cfg: ChimeConfig):
        super().__init__()
        self.cfg = cfg
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Cache build / hindsight presence sanity (no GPU work)."""
        # M0 stub — actual cache build is in scripts/00_build_libero_cache.py.
        return None

    def setup(self, stage: str | None = None) -> None:
        """Construct per-split datasets.  M0: stub."""
        raise NotImplementedError("LiberoLongDataModule.setup — M0 stub")

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError("LiberoLongDataModule.train_dataloader — M0 stub")

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError("LiberoLongDataModule.val_dataloader — M0 stub")

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError("LiberoLongDataModule.test_dataloader — M0 stub")

    @staticmethod
    def collate_fn(samples: list[dict]) -> dict:
        """Length-bucket collate; pads to ``cfg.data.T_max`` with valid_mask=False."""
        raise NotImplementedError("LiberoLongDataModule.collate_fn — M0 stub")
