#!/usr/bin/env python
"""LIBERO-Long held-out SR evaluation.

Usage::

    python scripts/20_eval_sr.py --checkpoint <path.ckpt> --split test

M0: stub.  Argparse + checkpoint discovery work; rollout loop raises
NotImplementedError.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LIBERO-Long success-rate evaluation."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a Lightning .ckpt produced by 10_train.py.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to roll out on.",
    )
    p.add_argument(
        "--n-rollouts",
        type=int,
        default=50,
        help="Number of rollouts per task (default 50).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("output/eval/sr_report.json"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[20_eval_sr] args = {vars(args)}")
    if not args.checkpoint.exists():
        print(f"[20_eval_sr] checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    raise NotImplementedError(
        "scripts/20_eval_sr.py — M0 stub.  "
        "Rollout loop lands at M4 (see IMPLEMENTATION_PLAN.md §6)."
    )


if __name__ == "__main__":
    sys.exit(main())
