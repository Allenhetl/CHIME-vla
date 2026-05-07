"""LiberoLongDataModule — Lightning DataModule for LIBERO-Long
(CODE_STRUCTURE §5, CODE_STANDARDS §5).

Reads per-episode ``.pt`` cache files produced by
``scripts/00_build_libero_cache.py`` and (when ``cfg.hindsight.enabled``)
joins each episode with its Hindsight γ̂ via
:class:`chime_vla.hindsight.HindsightConsumer`.

Splits: 8/1/1 LIBERO-Long, fixed seed.  M1 implementation uses a simple
random-shuffle split (no length-bucket sampler yet — per-episode T is
truncated/padded to ``cfg.data.T_max``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from chime_vla.config import ChimeConfig
from chime_vla.hindsight.consumer import HindsightConsumer


def _list_episode_ids(cache_dir: Path) -> list[int]:
    """Enumerate episode ids from ``ep_NNNNNN.pt`` filenames in ``cache_dir``."""
    ids: list[int] = []
    pat = re.compile(r"^ep_(\d+)\.pt$")
    for p in cache_dir.iterdir():
        m = pat.match(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def _resolve_cache_dir(cfg: ChimeConfig) -> Path:
    """Find the per-episode cache directory.

    ``cfg.data.cache_root`` may point to ``output/cache/libero_long`` while
    the episodes live under ``output/cache/libero_long/libero_long/``.
    Try both.
    """
    root = Path(cfg.data.cache_root)
    if not root.is_absolute():
        # Anchor relative paths to the repo root (this file's package parent).
        repo_root = Path(__file__).resolve().parents[3]
        root = repo_root / root

    candidates = [root, root / "libero_long", root / cfg.hindsight.task]
    for c in candidates:
        if c.exists() and any(c.glob("ep_*.pt")):
            return c
    raise FileNotFoundError(
        f"No LIBERO cache found.  Tried: {[str(c) for c in candidates]}"
    )


class LiberoLongDataset(Dataset):
    """One row = one episode, returns the canonical batch dict.

    Per-episode .pt schema (see scripts/00_build_libero_cache.py):
        rgb_raw    : uint8 (T, 224, 224, 3)
        proprio    : fp32  (T, 8)
        action     : fp32  (T, 8)
        sub_task_id: int32 (T,)
        rewards    : fp32  (T,)
        episode_id : int
        task_name  : str
        T          : int
        source     : str
    """

    def __init__(
        self,
        cache_dir: Path,
        episode_ids: list[int],
        T_max: int = 256,
        proprio_dim: int = 8,
        action_dim: int = 8,
        consumer: Optional[HindsightConsumer] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.episodes = list(episode_ids)
        self.T_max = int(T_max)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.consumer = consumer

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep_id = self.episodes[idx]
        path = self.cache_dir / f"ep_{ep_id:06d}.pt"
        blob = torch.load(path, map_location="cpu", weights_only=False)

        T_raw = int(blob["T"])
        T = min(T_raw, self.T_max)

        # rgb_raw: uint8 (T, H, W, 3) → float32 [0,1] (T, 3, H, W)
        rgb_u8 = blob["rgb_raw"][:T]  # (T, H, W, 3) uint8
        rgb = rgb_u8.to(torch.float32).div_(255.0).permute(0, 3, 1, 2).contiguous()

        proprio = blob["proprio"][:T].to(torch.float32)
        action = blob["action"][:T].to(torch.float32)
        sub_task_id = blob["sub_task_id"][:T].to(torch.int32)

        # Pad to T_max if needed.
        valid_mask = torch.zeros(self.T_max, dtype=torch.bool)
        valid_mask[:T] = True

        if T < self.T_max:
            pad = self.T_max - T
            rgb = torch.cat(
                [rgb, torch.zeros(pad, 3, rgb.shape[-2], rgb.shape[-1], dtype=rgb.dtype)],
                dim=0,
            )
            proprio = torch.cat(
                [proprio, torch.zeros(pad, self.proprio_dim, dtype=proprio.dtype)],
                dim=0,
            )
            action = torch.cat(
                [action, torch.zeros(pad, self.action_dim, dtype=action.dtype)],
                dim=0,
            )
            sub_task_id = torch.cat(
                [sub_task_id, torch.full((pad,), -1, dtype=sub_task_id.dtype)],
                dim=0,
            )

        out = {
            "rgb": rgb,                  # (T_max, 3, 224, 224) fp32
            "proprio": proprio,          # (T_max, 8) fp32
            "action": action,            # (T_max, 8) fp32
            "sub_task_id": sub_task_id,  # (T_max,) int32
            "valid_mask": valid_mask,    # (T_max,) bool
            "episode_id": int(ep_id),
        }

        # Optional Hindsight γ̂ join.  If unavailable we leave the keys out;
        # the loss layer treats missing keys as "no signal" → L_HCS = 0.
        if self.consumer is not None and self.consumer.has(ep_id):
            sample = self.consumer.load(ep_id)
            gh_geo = torch.full((self.T_max,), -1.0, dtype=torch.float32)
            gh_sem = torch.full((self.T_max,), -1.0, dtype=torch.float32)
            n = min(self.T_max, sample.gamma_geo.shape[0])
            gh_geo[:n] = sample.gamma_geo[:n].to(torch.float32)
            gh_sem[:n] = sample.gamma_sem[:n].to(torch.float32)
            out["gamma_hat_geo"] = gh_geo
            out["gamma_hat_sem"] = gh_sem

        return out


# Backwards-compatible private name (kept so tests/old imports still work).
_LiberoEpisodeDataset = LiberoLongDataset


def _default_collate(samples: list[dict]) -> dict:
    """Collate a list of per-episode dicts into a batch dict.

    All tensor fields stack along dim=0; ``episode_id`` becomes a 1-D
    long tensor of shape ``(B,)``.
    """
    out: dict = {}
    keys = samples[0].keys()
    for k in keys:
        vs = [s[k] for s in samples if k in s]
        if len(vs) != len(samples):
            # Some samples missed this key (e.g. hindsight not present for ep
            # but present for others).  In that case skip this key entirely
            # — losses handle missing γ̂ as "no signal".
            continue
        if isinstance(vs[0], torch.Tensor):
            out[k] = torch.stack(vs, dim=0)
        elif isinstance(vs[0], int):
            out[k] = torch.tensor(vs, dtype=torch.long)
        else:
            out[k] = vs
    return out


class LiberoLongDataModule(pl.LightningDataModule):
    """LIBERO-Long DataModule.

    Outputs batch dict (CODE_STRUCTURE §5):
        rgb            : (B, T, 3, 224, 224) float32
        proprio        : (B, T, 8) fp32
        action         : (B, T, 8) fp32
        sub_task_id    : (B, T) int32
        episode_id     : (B,) long
        valid_mask     : (B, T) bool
        gamma_hat_geo  : (B, T) fp32   if hindsight.enabled and γ̂ on disk
        gamma_hat_sem  : (B, T) fp32   if hindsight.enabled and γ̂ on disk
    """

    def __init__(
        self,
        cfg: ChimeConfig,
        batch_size: int | None = None,
        num_workers: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = int(batch_size if batch_size is not None else cfg.train.bs)
        self.num_workers = int(num_workers)
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def prepare_data(self) -> None:
        return None

    def _build_consumer(self) -> Optional[HindsightConsumer]:
        if not self.cfg.hindsight.enabled:
            return None
        try:
            return HindsightConsumer(
                root=self.cfg.hindsight.gamma_hat_root,
                strategy=self.cfg.hindsight.strategy,
                task=self.cfg.hindsight.task,
            )
        except FileNotFoundError:
            # Hindsight enabled in config but γ̂ not yet on disk.  Fail soft —
            # losses will treat this as "no signal" and L_HCS stays 0.
            return None

    def setup(self, stage: str | None = None) -> None:
        cache_dir = _resolve_cache_dir(self.cfg)
        all_ids = _list_episode_ids(cache_dir)
        if len(all_ids) == 0:
            raise RuntimeError(f"No ep_*.pt files in {cache_dir}")

        # Try to read splits JSON; if absent, fall back to deterministic
        # 8/1/1 random shuffle.
        splits: dict[str, list[int]] | None = None
        splits_path = Path(self.cfg.data.splits_path)
        if not splits_path.is_absolute():
            splits_path = Path(__file__).resolve().parents[3] / splits_path
        if splits_path.exists():
            with splits_path.open() as f:
                splits = json.load(f)

        if splits is None:
            g = torch.Generator().manual_seed(int(self.cfg.seed))
            perm = torch.randperm(len(all_ids), generator=g).tolist()
            shuffled = [all_ids[i] for i in perm]
            n = len(shuffled)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            train_ids = shuffled[:n_train]
            val_ids = shuffled[n_train : n_train + n_val]
            test_ids = shuffled[n_train + n_val :]
        else:
            train_ids = list(splits.get("train", []))
            val_ids = list(splits.get("val", []))
            test_ids = list(splits.get("test", []))

        consumer = self._build_consumer()

        kw = dict(
            cache_dir=cache_dir,
            T_max=self.cfg.data.T_max,
            proprio_dim=self.cfg.data.proprio_dim,
            action_dim=self.cfg.data.action_dim,
            consumer=consumer,
        )
        self.train_ds = LiberoLongDataset(episode_ids=train_ids, **kw)
        self.val_ds = LiberoLongDataset(episode_ids=val_ids, **kw)
        self.test_ds = LiberoLongDataset(episode_ids=test_ids, **kw)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_default_collate,
            drop_last=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_default_collate,
            drop_last=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_default_collate,
            drop_last=False,
            pin_memory=False,
        )

    @staticmethod
    def collate_fn(samples: list[dict]) -> dict:
        return _default_collate(samples)
