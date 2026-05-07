"""HindsightConsumer file-protocol tests (per `docs/hindsight_contract.md` §8).

Contract: strict-pass from M0 onward — HindsightConsumer is a thin file
reader, so even at M0 it must work.  Synthesizes a fake `ep_000000.pt` in a
tmp dir, then exercises load / has / list_available / FileNotFoundError.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fake_episode(
    root: Path,
    strategy: str,
    task: str,
    episode_id: int,
    T: int = 100,
) -> Path:
    """Write a synthetic ep_NNNNNN.pt that matches the schema in §3."""
    target_dir = root / strategy / task
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_id": episode_id,
        "task_name": task,
        "T": T,
        "gamma_geo": torch.rand(T, dtype=torch.float32),
        "gamma_sem": torch.rand(T, dtype=torch.float32),
        "valid_mask": torch.ones(T, dtype=torch.bool),
        "meta": {
            "strategy": strategy,
            "base_policy": "pi05",
            "delta_set": [4, 16],
            "saliency_method": "EAGN",
            "computed_at": "2025-01-01T00:00:00Z",
            "hindsight_commit": "deadbeef",
        },
    }
    fp = target_dir / f"ep_{episode_id:06d}.pt"
    torch.save(payload, fp)
    return fp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_returns_correct_shapes_dtypes(tmp_path):
    pytest.importorskip("chime_vla")
    from chime_vla.hindsight.consumer import HindsightConsumer

    _write_fake_episode(tmp_path, "per_task_q75", "libero_long", 0, T=100)
    consumer = HindsightConsumer(
        root=tmp_path, strategy="per_task_q75", task="libero_long"
    )
    sample = consumer.load(episode_id=0)

    assert sample.gamma_geo.shape == (100,)
    assert sample.gamma_sem.shape == (100,)
    assert sample.valid_mask.shape == (100,)
    assert sample.gamma_geo.dtype == torch.float32
    assert sample.gamma_sem.dtype == torch.float32
    assert sample.valid_mask.dtype == torch.bool
    assert sample.episode_id == 0
    assert sample.meta["strategy"] == "per_task_q75"


def test_has_and_list_available(tmp_path):
    pytest.importorskip("chime_vla")
    from chime_vla.hindsight.consumer import HindsightConsumer

    for ep in (0, 1, 7):
        _write_fake_episode(tmp_path, "per_task_q75", "libero_long", ep, T=20)
    consumer = HindsightConsumer(
        root=tmp_path, strategy="per_task_q75", task="libero_long"
    )
    assert consumer.has(0)
    assert consumer.has(7)
    assert not consumer.has(99)
    assert sorted(consumer.list_available()) == [0, 1, 7]


def test_load_missing_episode_raises_file_not_found(tmp_path):
    pytest.importorskip("chime_vla")
    from chime_vla.hindsight.consumer import HindsightConsumer

    _write_fake_episode(tmp_path, "per_task_q75", "libero_long", 0, T=10)
    consumer = HindsightConsumer(
        root=tmp_path, strategy="per_task_q75", task="libero_long"
    )
    with pytest.raises(FileNotFoundError):
        consumer.load(episode_id=99999999)


def test_consumer_init_raises_on_missing_root(tmp_path):
    """If the strategy/task subdirectory does not exist, init must raise."""
    pytest.importorskip("chime_vla")
    from chime_vla.hindsight.consumer import HindsightConsumer

    with pytest.raises(FileNotFoundError):
        HindsightConsumer(
            root=tmp_path, strategy="does_not_exist", task="libero_long"
        )
