#!/usr/bin/env python
"""Build per-episode .pt cache from raw LIBERO h5 (CODE_STRUCTURE §5).

Reads:
    /home/sqmluser/data/memory_vla/libero_long/traj_NNNN.h5

Writes:
    output/cache/libero_long/{task}/ep_NNNNNN.pt

Each .pt has the schema documented in ``docs/data_schema.md``::

    {
        "rgb_raw":     uint8 (T, 224, 224, 3),     # always stored at M0
        "rgb_feature": fp16 (T, N=256, d_h=1152),  # only if --extract-features
        "proprio":     fp32 (T, 8),
        "action":      fp32 (T, 8),
        "sub_task_id": int32 (T,),
        "rewards":     fp32 (T,),                  # bonus, useful for diagnostics
        "episode_id":  int,
        "task_name":   str,
        "T":           int,
        "source":      str,                        # absolute path of source h5
    }

M0 default (no --extract-features): stores raw RGB + proprio + action +
sub_task_id + rewards.  --extract-features adds SigLIP pre-cached features
(M1 work; requires GPU).

Run:
    python scripts/00_build_libero_cache.py --n 5 --dry-run
    python scripts/00_build_libero_cache.py --n 5
    python scripts/00_build_libero_cache.py                   # full 379 eps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build LIBERO h5 → per-episode .pt cache for CHIME-VLA."
    )
    p.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/sqmluser/data/memory_vla/libero_long/"),
        help="Directory containing traj_NNNN.h5.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/cache/libero_long"),
        help="Output directory; subdirs per task created automatically.",
    )
    p.add_argument(
        "--task-name",
        type=str,
        default="libero_long",
        help="Task name (used in output subdir and meta).",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Process only the first N episodes (for smoke testing).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended actions but don't write any .pt files. Still"
        " reads h5 to validate schema.",
    )
    p.add_argument(
        "--extract-features",
        action="store_true",
        help="Pre-extract SigLIP features (M1 work; requires GPU).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="torch device for SigLIP feature extraction (only with --extract-features).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batched frames per SigLIP forward (only with --extract-features).",
    )
    p.add_argument(
        "--skip-rgb-raw",
        action="store_true",
        help="Don't store uint8 RGB (saves ~30 MB per episode); only safe if --extract-features.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files; default is to skip already-cached episodes.",
    )
    return p.parse_args(argv)


def find_episodes(input_root: Path) -> list[Path]:
    """Return sorted list of traj_*.h5 files in input_root."""
    files = sorted(input_root.glob("traj_*.h5"))
    if not files:
        raise FileNotFoundError(
            f"No traj_*.h5 found under {input_root}. "
            "Expected raw LIBERO trajectories there."
        )
    return files


def parse_episode_id(path: Path) -> int:
    """traj_0042.h5 -> 42; raises if pattern doesn't match."""
    stem = path.stem  # "traj_0042"
    if not stem.startswith("traj_"):
        raise ValueError(f"Unexpected filename pattern: {path}")
    try:
        return int(stem[5:])
    except ValueError as e:
        raise ValueError(f"Could not parse episode id from {path}") from e


def read_episode(h5_path: Path, episode_id: int, task_name: str) -> dict:
    """Read one h5 trajectory into a dict matching the cache schema.

    Returns:
        dict with all schema fields except rgb_feature (caller adds if requested).
    """
    with h5py.File(h5_path, "r") as f:
        # required datasets per LIBERO schema
        rgb = np.asarray(f["obs/agentview_rgb"])         # (T, 224, 224, 3) uint8
        proprio = np.asarray(f["obs/proprio"])           # (T, 8) float32
        action = np.asarray(f["actions"])                # (T, 8) float32
        sub_task_id = np.asarray(f["sub_task_id"])       # (T,) int32
        rewards = np.asarray(f["rewards"])               # (T,) float32

    # Sanity: T must agree across all fields
    T = int(rgb.shape[0])
    for name, arr in [
        ("proprio", proprio),
        ("action", action),
        ("sub_task_id", sub_task_id),
        ("rewards", rewards),
    ]:
        if arr.shape[0] != T:
            raise ValueError(
                f"{h5_path}: T mismatch — rgb has {T} but {name} has {arr.shape[0]}"
            )

    # dtype contract
    if rgb.dtype != np.uint8:
        raise TypeError(f"{h5_path}: rgb dtype {rgb.dtype} != uint8")
    if rgb.shape[1:] != (224, 224, 3):
        raise ValueError(f"{h5_path}: rgb shape {rgb.shape[1:]} != (224,224,3)")
    if proprio.shape[1] != 8:
        raise ValueError(f"{h5_path}: proprio dim {proprio.shape[1]} != 8")
    if action.shape[1] != 8:
        raise ValueError(f"{h5_path}: action dim {action.shape[1]} != 8")

    return {
        "rgb_raw": torch.from_numpy(rgb),                                       # (T,224,224,3) uint8
        "proprio": torch.from_numpy(proprio.astype(np.float32)),                # (T,8) fp32
        "action": torch.from_numpy(action.astype(np.float32)),                  # (T,8) fp32
        "sub_task_id": torch.from_numpy(sub_task_id.astype(np.int32)),          # (T,) int32
        "rewards": torch.from_numpy(rewards.astype(np.float32)),                # (T,) fp32
        "episode_id": episode_id,
        "task_name": task_name,
        "T": T,
        "source": str(h5_path.resolve()),
    }


def extract_siglip_features(rgb_raw: torch.Tensor, device: str, batch_size: int) -> torch.Tensor:
    """Pre-extract SigLIP features.  M1 work — placeholder.

    Args:
        rgb_raw: (T, 224, 224, 3) uint8.

    Returns:
        rgb_feature: (T, N=256, d_h=1152) fp16.
    """
    raise NotImplementedError(
        "SigLIP feature extraction is M1 work.  Run without --extract-features for M0."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[00_build_libero_cache] args = {vars(args)}")

    if args.skip_rgb_raw and not args.extract_features:
        print(
            "[00_build_libero_cache] ERROR: --skip-rgb-raw requires --extract-features "
            "(otherwise the cache would have no observations)."
        )
        return 2

    h5_files = find_episodes(args.input_root)
    if args.n is not None:
        h5_files = h5_files[: args.n]
    print(f"[00_build_libero_cache] processing {len(h5_files)} episodes")

    out_dir = args.output_root / args.task_name
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[00_build_libero_cache] writing to {out_dir.resolve()}")
    else:
        print(f"[00_build_libero_cache] DRY-RUN: would write to {out_dir.resolve()}")

    n_skipped = 0
    n_written = 0
    n_failed = 0
    total_T = 0
    total_bytes = 0
    t0 = time.time()

    summary_rows: list[dict] = []

    for h5_path in h5_files:
        ep_id = parse_episode_id(h5_path)
        out_path = out_dir / f"ep_{ep_id:06d}.pt"

        if out_path.exists() and not args.overwrite and not args.dry_run:
            print(f"  ep {ep_id:06d}: SKIP (exists)")
            n_skipped += 1
            continue

        try:
            blob = read_episode(h5_path, ep_id, args.task_name)
        except Exception as e:                                          # noqa: BLE001
            print(f"  ep {ep_id:06d}: FAIL — {type(e).__name__}: {e}")
            n_failed += 1
            continue

        if args.extract_features:
            blob["rgb_feature"] = extract_siglip_features(
                blob["rgb_raw"], args.device, args.batch_size
            )

        if args.skip_rgb_raw:
            del blob["rgb_raw"]

        T = blob["T"]
        total_T += T
        summary_rows.append(
            {"episode_id": ep_id, "T": T, "n_subtasks": int(blob["sub_task_id"].max()) + 1}
        )

        if args.dry_run:
            print(
                f"  ep {ep_id:06d}: T={T:3d} subtasks={summary_rows[-1]['n_subtasks']} "
                f"reward_max={blob['rewards'].max().item():.2f} (DRY-RUN, not written)"
            )
            n_written += 1
            continue

        torch.save(blob, out_path)
        size = out_path.stat().st_size
        total_bytes += size
        n_written += 1
        print(f"  ep {ep_id:06d}: T={T:3d}  size={size / 1e6:.1f} MB")

    elapsed = time.time() - t0
    print(
        f"\n[00_build_libero_cache] done in {elapsed:.1f}s — "
        f"written={n_written} skipped={n_skipped} failed={n_failed}"
    )
    print(
        f"  total frames = {total_T}  "
        f"avg T = {total_T / max(n_written, 1):.1f}  "
        f"total size = {total_bytes / 1e9:.2f} GB"
    )

    if not args.dry_run and n_written > 0:
        meta_path = out_dir / "_cache_meta.json"
        meta = {
            "task_name": args.task_name,
            "n_episodes": n_written,
            "total_frames": total_T,
            "extract_features": args.extract_features,
            "skip_rgb_raw": args.skip_rgb_raw,
            "input_root": str(args.input_root.resolve()),
            "summary": summary_rows[:50],   # keep first 50 rows; rest derivable from .pt
            "schema_version": "m0",
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  meta written to {meta_path}")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
