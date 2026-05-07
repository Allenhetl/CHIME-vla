#!/usr/bin/env python
"""Sanity check: read one Hindsight γ̂ episode and report shapes / meta.

Usage:
    python scripts/01_run_hindsight_consumer.py --episode-id 0
    python scripts/01_run_hindsight_consumer.py --list

M0: --list works (it just enumerates files); --episode-id raises
NotImplementedError because the underlying ``HindsightConsumer.load`` is
a stub.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chime_vla.hindsight.consumer import HindsightConsumer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hindsight consumer sanity check (CODE_STRUCTURE §3.9)."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/home/sqmluser/workspace/theaj/CHIME-VLA/Hindsight/output/saliency/gamma_hat"
        ),
        help="Hindsight gamma_hat root directory.",
    )
    p.add_argument("--strategy", type=str, default="per_task_q75")
    p.add_argument("--task", type=str, default="libero_long")
    p.add_argument("--episode-id", type=int, default=None)
    p.add_argument("--list", action="store_true", help="List available episode ids.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        consumer = HindsightConsumer(
            root=args.root, strategy=args.strategy, task=args.task
        )
    except FileNotFoundError as e:
        print(f"[01_run_hindsight_consumer] {e}", file=sys.stderr)
        return 2

    if args.list:
        ids = consumer.list_available()
        print(f"[01_run_hindsight_consumer] {len(ids)} episodes available")
        for i in ids[:50]:
            print(f"  ep_{i:06d}.pt")
        if len(ids) > 50:
            print(f"  ... ({len(ids) - 50} more)")
        return 0

    if args.episode_id is None:
        print("[01_run_hindsight_consumer] pass --episode-id N or --list", file=sys.stderr)
        return 2

    sample = consumer.load(args.episode_id)
    print(
        f"[01_run_hindsight_consumer] ep={sample.episode_id}  "
        f"T={sample.gamma_geo.shape[0]}  "
        f"meta={sample.meta}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
