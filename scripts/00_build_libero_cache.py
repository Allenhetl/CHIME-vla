#!/usr/bin/env python
"""Build per-episode .pt cache from raw LIBERO h5 (CODE_STRUCTURE §5).

Reads:
    /home/sqmluser/data/memory_vla/libero_long/traj_NNNN.h5

Writes:
    output/cache/libero_long/{task}/ep_NNNNNN.pt

Each .pt has the schema documented in ``docs/data_schema.md``::

    {
        "rgb_feature": fp16 (T, N=256, d_h=1152),  # SigLIP pre-extract
        "rgb_raw":     uint8 (T, 224, 224, 3),     # optional, for debug
        "proprio":     fp32 (T, 8),
        "action":      fp32 (T, 8),
        "sub_task_id": int32 (T,),
        "episode_id":  int,
        "task_name":   str,
        "T":           int,
    }

M0: stub.  ``--help`` works and ``--dry-run`` exits cleanly without
touching disk; the actual feature extraction body raises NotImplementedError.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
        help="Print intended actions but don't write anything.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="torch device for SigLIP feature extraction.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batched frames per SigLIP forward.",
    )
    p.add_argument(
        "--keep-rgb-raw",
        action="store_true",
        help="Also store the uint8 RGB frames (for viz/debug; ~10x larger).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[00_build_libero_cache] args = {vars(args)}")
    if args.dry_run:
        print("[00_build_libero_cache] --dry-run set; exiting without touching disk.")
        return 0
    raise NotImplementedError(
        "scripts/00_build_libero_cache.py — M0 stub.  "
        "Full implementation lands in M1 (see IMPLEMENTATION_PLAN.md §3)."
    )


if __name__ == "__main__":
    sys.exit(main())
