"""Hindsight γ̂ consumer — file protocol per ``docs/hindsight_contract.md`` §4.

CHIME-VLA reads Hindsight outputs strictly through the filesystem; no
``from Hindsight.src...`` import is allowed (CODE_STANDARDS §1.6).

Schema of each ``ep_NNNNNN.pt`` (docs/hindsight_contract.md §3):

    {
        "episode_id": int,
        "task_name": str,
        "T": int,
        "gamma_geo": Tensor (T,) float32,
        "gamma_sem": Tensor (T,) float32,
        "valid_mask": Tensor (T,) bool,
        "meta": {strategy, base_policy, delta_set, saliency_method,
                 computed_at, hindsight_commit},
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class HindsightSample:
    """One episode worth of γ̂ labels (per docs/hindsight_contract.md §4)."""

    gamma_geo: torch.Tensor  # (T,) float32
    gamma_sem: torch.Tensor  # (T,) float32
    valid_mask: torch.Tensor  # (T,) bool
    episode_id: int
    meta: dict


class HindsightConsumer:
    """Read-only loader for Hindsight γ̂ artefacts.

    Construction validates the directory exists; per-episode reads are
    lazy and do not cache (a single ep .pt is < 10KB so memory pressure
    is negligible).

    Args:
        root: typically ``cfg.hindsight.gamma_hat_root``.
        strategy: e.g. ``per_task_q75``.
        task: e.g. ``libero_long``.

    Raises:
        FileNotFoundError: if ``root / strategy / task`` does not exist.
    """

    def __init__(
        self,
        root: Path | str,
        strategy: str = "per_task_q75",
        task: str = "libero_long",
    ):
        self.root: Path = Path(root)
        self.strategy: str = strategy
        self.task: str = task
        self.dir: Path = self.root / strategy / task
        if not self.dir.exists():
            raise FileNotFoundError(
                f"Hindsight gamma_hat not found at {self.dir}.  "
                "Run Hindsight scripts/05-07 first."
            )

    # ---- public API per docs/hindsight_contract.md §4 ----

    def load(self, episode_id: int) -> HindsightSample:
        """Load one episode's γ̂.

        Args:
            episode_id: 0-padded to 6 digits in the filename.

        Returns:
            :class:`HindsightSample`.

        Raises:
            FileNotFoundError: if ``ep_{episode_id:06d}.pt`` is missing.
            KeyError: if the ``.pt`` is missing required schema fields.
        """
        path = self.dir / f"ep_{episode_id:06d}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Hindsight episode not found: {path}"
            )
        blob = torch.load(path, map_location="cpu", weights_only=False)
        required = ("gamma_geo", "gamma_sem", "valid_mask", "episode_id")
        missing = [k for k in required if k not in blob]
        if missing:
            raise KeyError(
                f"{path} missing required keys {missing} "
                f"(per docs/hindsight_contract.md §3)"
            )
        return HindsightSample(
            gamma_geo=blob["gamma_geo"],
            gamma_sem=blob["gamma_sem"],
            valid_mask=blob["valid_mask"],
            episode_id=int(blob["episode_id"]),
            meta=dict(blob.get("meta", {})),
        )

    def has(self, episode_id: int) -> bool:
        """Return True iff ``ep_{episode_id:06d}.pt`` exists in the directory.

        M0: returns False unconditionally (the on-disk artefacts arrive
        from Hindsight pipeline; until then nothing is available).
        """
        path = self.dir / f"ep_{episode_id:06d}.pt"
        return path.exists()

    def list_available(self) -> list[int]:
        """Enumerate episode ids present on disk (sorted ascending).

        M0: returns the actual on-disk listing (works even with zero
        files).  Pure filesystem read, no torch.load.
        """
        ids: list[int] = []
        if not self.dir.exists():
            return ids
        for p in self.dir.glob("ep_*.pt"):
            stem = p.stem  # "ep_NNNNNN"
            if not stem.startswith("ep_"):
                continue
            try:
                ids.append(int(stem[3:]))
            except ValueError:
                continue
        return sorted(ids)
