#!/usr/bin/env python
"""Analyze M2 training metrics — L_main / L_PRH / L_predict / L_aux.

Reads `lightning_logs/version_*/metrics.csv` and emits:
- ``output/reports/m2_smoke_curves.png`` (4-panel plot)
- ``output/reports/m2_smoke_summary.json``  (first/last 10-step means + ratios)

M2 deliverable per IMPLEMENTATION_PLAN §4 (MVP fallback adapted):
- L_PRH @ k=4,16 monotonic decrease
- L_predict (ψ self-supervised) decrease — ψ actually learning
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def find_latest_metrics_csv(log_dir: Path) -> Path:
    versions = sorted(log_dir.glob("version_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not versions:
        raise FileNotFoundError(f"No version_* under {log_dir}")
    csv = versions[-1] / "metrics.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No metrics.csv in {versions[-1]}")
    return csv


def head_tail_summary(df: pd.DataFrame, key: str, n: int = 10) -> dict:
    """Return mean over first n / last n non-NaN rows for the given column."""
    if key not in df.columns:
        return {"present": False}
    series = df.dropna(subset=[key])
    if len(series) == 0:
        return {"present": False}
    head_mean = float(series.head(n)[key].mean())
    tail_mean = float(series.tail(n)[key].mean())
    return {
        "present": True,
        "n_total": int(len(series)),
        f"first_{n}_mean": head_mean,
        f"last_{n}_mean": tail_mean,
        "ratio_last_to_first": float(tail_mean / head_mean) if head_mean != 0 else float("nan"),
        "delta_last_minus_first": float(tail_mean - head_mean),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", type=Path, default=Path("lightning_logs"))
    p.add_argument("--output-png", type=Path, default=Path("output/reports/m2_smoke_curves.png"))
    p.add_argument("--output-json", type=Path, default=Path("output/reports/m2_smoke_summary.json"))
    args = p.parse_args(argv)

    csv_path = find_latest_metrics_csv(args.log_dir)
    print(f"[41] reading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[41] columns: {sorted(df.columns)}")
    print(f"[41] rows: {len(df)}")

    # Discover loss columns. Lightning typically uses train/{key} or train_{key}.
    candidate_keys = [
        "train/loss",
        "train_loss",
        "train/L_main",
        "train_L_main",
        "L_main",
        "train/L_PRH",
        "L_PRH",
        "train/L_predict",
        "L_predict",
        "train/L_aux",
        "L_aux",
        "train/L_HCS",
        "L_HCS",
        "train/L_CSM",
        "L_CSM",
    ]
    present = {k: head_tail_summary(df, k) for k in candidate_keys}

    summary = {
        "csv_path": str(csv_path),
        "n_rows": int(len(df)),
        "metrics": {k: v for k, v in present.items() if v.get("present")},
        "verdict": {},
    }

    # M2 deliverable check
    prh_summary = present.get("L_PRH", {"present": False})
    if not prh_summary.get("present"):
        prh_summary = present.get("train/L_PRH", {"present": False})
    predict_summary = present.get("L_predict", {"present": False})
    if not predict_summary.get("present"):
        predict_summary = present.get("train/L_predict", {"present": False})

    summary["verdict"]["L_PRH_decreasing"] = (
        prh_summary.get("present", False)
        and prh_summary.get("ratio_last_to_first", 1.0) < 1.0
    )
    summary["verdict"]["L_predict_decreasing"] = (
        predict_summary.get("present", False)
        and predict_summary.get("ratio_last_to_first", 1.0) < 1.0
    )

    # Plot
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_keys = ["train/loss", "L_main", "L_PRH", "L_predict"]
    for ax, key in zip(axes.flat, plot_keys):
        # Try unprefixed too
        col = key if key in df.columns else key.replace("train/", "train_")
        if col not in df.columns:
            ax.text(0.5, 0.5, f"({key} not in csv)", transform=ax.transAxes, ha="center")
            ax.set_title(key)
            continue
        m = df.dropna(subset=[col])
        ax.plot(m["step"], m[col])
        ax.set_title(f"{key}  ({len(m)} pts)")
        ax.set_xlabel("step")
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=80)
    print(f"[41] wrote {args.output_png}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(f"[41] wrote {args.output_json}")

    print()
    print("═══════ M2 verdict ═══════")
    for k, v in summary["metrics"].items():
        first = v.get(f"first_10_mean", "n/a")
        last = v.get(f"last_10_mean", "n/a")
        ratio = v.get("ratio_last_to_first", "n/a")
        if isinstance(first, float):
            first = f"{first:.4f}"
        if isinstance(last, float):
            last = f"{last:.4f}"
        if isinstance(ratio, float):
            ratio = f"{ratio:.3f}"
        print(f"  {k:20s} first10={first}  last10={last}  ratio={ratio}")
    print()
    print(f"  L_PRH decreasing:    {summary['verdict']['L_PRH_decreasing']}")
    print(f"  L_predict decreasing: {summary['verdict']['L_predict_decreasing']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
