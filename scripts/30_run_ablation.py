#!/usr/bin/env python
"""M6 ablation runner — IMPLEMENTATION_PLAN.md §8 (10 ablations).

Usage::

    python scripts/30_run_ablation.py --ablation-id 1     # γ_const = 1 (no [C5])
    python scripts/30_run_ablation.py --ablation-id 4     # remove L_HCS
    python scripts/30_run_ablation.py --ablation-id 7     # M_geo only

Each ablation maps to a config override applied on top of the M4 full
config.  The mapping table lives in IMPLEMENTATION_PLAN.md §8.

M0: stub.  Argparse + ablation registry work; the actual launch raises
NotImplementedError.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Map ablation-id → human-readable description (per IMPLEMENTATION_PLAN.md §8).
ABLATIONS: dict[int, str] = {
    1: "gamma_const_1 (no [C5] gating)",
    2: "geo_only_gamma (no sem channel γ)",
    3: "sem_only_gamma (no geo channel γ)",
    4: "no_L_HCS (lambda_1=0)",
    5: "no_L_PRH (lambda_2=0)",
    6: "no_L_CSM (lambda_3=0)",
    7: "M_geo_only (K_s=0, single channel)",
    8: "K_s_sweep (32 / 64 / 128)",
    9: "delta_set_sweep ({4} / {4,16} / {4,16,64})",
    10: "slot_free_mask vs zero_fill",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CHIME-VLA M6 ablation launcher (10 entries)."
    )
    p.add_argument(
        "--ablation-id",
        type=int,
        required=True,
        choices=sorted(ABLATIONS.keys()),
        help="Ablation index 1..10 (see IMPLEMENTATION_PLAN.md §8).",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Seeds per ablation (default 3).",
    )
    p.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Optional warm-start checkpoint (M4 main).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/runs/ablation"),
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        f"[30_run_ablation] id={args.ablation_id}  "
        f"name={ABLATIONS[args.ablation_id]}  seeds={args.n_seeds}"
    )
    if args.dry_run:
        return 0
    raise NotImplementedError(
        "scripts/30_run_ablation.py — M0 stub.  "
        "Launch wiring lands at M6 (see IMPLEMENTATION_PLAN.md §8)."
    )


if __name__ == "__main__":
    sys.exit(main())
