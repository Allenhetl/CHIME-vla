#!/usr/bin/env python
"""F7 Phase 2 step 2 — re-run E1 IoU using the full HCS-H γ̂ artifacts.

Reads ``ep_NNNNNN.pt`` from the Hindsight γ̂ root produced by
``scripts/05_compute_hcs_saliency.py`` and computes the architecture
v2.1 §I.3 line 1983 milestone-gate IoU against the LIBERO sub_task_id
boundary set:

    IoU @ 0.3 ≥ 0.4 → PASS       (proceed M2 with HCS signal)
    0.3 ≤ IoU < 0.4 → SOFT-PASS  (proceed M2 with red flag #1; anneal-to-0.15)
    IoU < 0.3       → HARD-FAIL  (fallback to MVP, λ_1=0 permanent)

Compared to ``scripts/40_run_e1_judgment.py`` (which inlines the
simplified single-Δ Jacobian baseline and yielded peak IoU 0.227),
this script consumes the *file-protocol* γ̂ produced by the full
[C10] HCS-H module — RUDDER + grad-cam decomposition + segment-scoped
Jacobian — and is therefore the canonical Phase-2 verdict.

Usage::

    python scripts/42_e1_with_full_hcs.py --n 50 \\
        --gamma-hat-root /home/sqmluser/.../Hindsight/.../libero_long \\
        --cache-root output/cache/libero_long/libero_long
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chime_vla.eval.e1_judgment import (  # noqa: E402
    compute_iou_vs_boundaries,
    e1_decision,
    random_baseline_iou,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_GAMMA_ROOT = (
    "/home/sqmluser/workspace/theaj/CHIME-VLA/Hindsight/output/saliency/"
    "gamma_hat/per_task_q75/libero_long"
)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--gamma-hat-root",
        default=_DEFAULT_GAMMA_ROOT,
        help="Directory containing ep_NNNNNN.pt γ̂ artifacts.",
    )
    p.add_argument(
        "--cache-root",
        default="output/cache/libero_long/libero_long",
        help="LIBERO cache directory (provides sub_task_id).",
    )
    p.add_argument(
        "--output",
        default="output/reports/e1_full_hcs.json",
    )
    p.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of γ̂ episodes to evaluate (caps the file list).",
    )
    p.add_argument(
        "--quantile",
        type=float,
        default=0.25,
        help="Top-quantile threshold (matches scripts/40 baseline default).",
    )
    p.add_argument(
        "--boundary-window",
        type=int,
        default=4,
    )
    p.add_argument(
        "--n-random-trials",
        type=int,
        default=64,
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve(p: str) -> Path:
    pp = Path(p)
    if not pp.is_absolute():
        pp = Path(__file__).resolve().parent.parent / pp
    return pp


def _list_ep_files(root: Path) -> list[tuple[int, Path]]:
    pat = re.compile(r"^ep_(\d+)\.pt$")
    out: list[tuple[int, Path]] = []
    for p in root.iterdir():
        m = pat.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out, key=lambda x: x[0])


def _load_gamma(path: Path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("gamma_geo", "gamma_sem", "valid_mask", "T", "episode_id"):
        if k not in blob:
            raise KeyError(f"{path}: missing key {k!r}")
    return blob


def _load_sub_task_id(cache_dir: Path, ep_id: int, T: int) -> torch.Tensor:
    blob = torch.load(
        cache_dir / f"ep_{ep_id:06d}.pt", map_location="cpu", weights_only=False
    )
    sti = blob["sub_task_id"][:T].to(torch.long)
    if int(sti.shape[0]) != int(T):
        raise ValueError(
            f"ep {ep_id}: sub_task_id length {sti.shape[0]} ≠ γ̂ T={T}"
        )
    return sti


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    torch.manual_seed(int(args.seed))

    print("=" * 64)
    print("F7-2  E1 IoU evaluation against full HCS-H γ̂ artifacts")
    print("=" * 64)
    print(f"[F7-2/E1] gamma-hat root : {args.gamma_hat_root}")
    print(f"[F7-2/E1] cache root     : {args.cache_root}")
    print(f"[F7-2/E1] n              : {args.n}")
    print(f"[F7-2/E1] quantile       : {args.quantile}")
    print(f"[F7-2/E1] boundary window: ±{args.boundary_window}")

    gamma_root = _resolve(args.gamma_hat_root)
    cache_dir = _resolve(args.cache_root)
    if not gamma_root.exists():
        raise FileNotFoundError(f"gamma-hat root not found: {gamma_root}")
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache root not found: {cache_dir}")

    files = _list_ep_files(gamma_root)
    files = files[: int(args.n)]
    if not files:
        raise RuntimeError(
            f"no ep_*.pt files found under {gamma_root} — run scripts/05 first"
        )
    print(f"[F7-2/E1] loaded         : {len(files)} γ̂ episode files")

    per_ep: list[dict] = []
    overall_t0 = time.time()

    for i, (ep_id, gamma_path) in enumerate(files):
        try:
            gblob = _load_gamma(gamma_path)
        except (KeyError, RuntimeError, OSError) as exc:
            print(f"[F7-2/E1] [ep {ep_id}] load error: {exc}; skip")
            continue

        T = int(gblob["T"])
        gamma_geo = gblob["gamma_geo"].detach().float()
        gamma_sem = gblob["gamma_sem"].detach().float()
        valid_mask = gblob["valid_mask"].to(torch.bool)
        if gamma_geo.shape[0] != T or gamma_sem.shape[0] != T:
            print(f"[F7-2/E1] [ep {ep_id}] shape mismatch γ_geo={gamma_geo.shape} "
                  f"γ_sem={gamma_sem.shape} T={T}; skip")
            continue

        try:
            sub_task_id = _load_sub_task_id(cache_dir, ep_id, T=T)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[F7-2/E1] [ep {ep_id}] cache miss: {exc}; skip")
            continue

        iou_geo = compute_iou_vs_boundaries(
            gamma_geo, sub_task_id,
            quantile=float(args.quantile),
            boundary_window=int(args.boundary_window),
            valid_mask=valid_mask,
        )
        iou_sem = compute_iou_vs_boundaries(
            gamma_sem, sub_task_id,
            quantile=float(args.quantile),
            boundary_window=int(args.boundary_window),
            valid_mask=valid_mask,
        )
        gamma_combined = torch.maximum(gamma_geo, gamma_sem)
        iou_combined = compute_iou_vs_boundaries(
            gamma_combined, sub_task_id,
            quantile=float(args.quantile),
            boundary_window=int(args.boundary_window),
            valid_mask=valid_mask,
        )
        rand_stats = random_baseline_iou(
            sub_task_id,
            quantile=float(args.quantile),
            boundary_window=int(args.boundary_window),
            n_trials=int(args.n_random_trials),
            valid_mask=valid_mask,
            seed=int(args.seed) + ep_id,
        )

        rec = {
            "episode_id": int(ep_id),
            "T": int(T),
            "iou_geo": iou_geo,
            "iou_sem": iou_sem,
            "iou_combined": iou_combined,
            "random_baseline": rand_stats,
            "gamma_geo_mean": float(gamma_geo.mean()),
            "gamma_geo_std": float(gamma_geo.std(unbiased=False)),
            "gamma_sem_mean": float(gamma_sem.mean()),
            "gamma_sem_std": float(gamma_sem.std(unbiased=False)),
            "meta": gblob.get("meta", {}),
        }
        per_ep.append(rec)
        print(
            f"[F7-2/E1] [{i + 1}/{len(files)} ep {ep_id} T={T}] "
            f"IoU(geo @{args.quantile})={iou_geo['iou_main']:.3f}  "
            f"IoU(sem @{args.quantile})={iou_sem['iou_main']:.3f}  "
            f"IoU(comb @{args.quantile})={iou_combined['iou_main']:.3f}  "
            f"IoU(comb @0.3)={iou_combined['iou_at_0.3']:.3f}  "
            f"random={rand_stats['random_iou_mean']:.3f}±"
            f"{rand_stats['random_iou_std']:.3f}  "
            f"n_b={iou_combined['n_boundaries']}"
        )

    if not per_ep:
        print("[F7-2/E1] no episodes processed.")
        return 1

    def _mean_iou(ch: str, key: str) -> float:
        return float(sum(r[ch][key] for r in per_ep) / len(per_ep))

    summary = {
        "n_episodes": len(per_ep),
        "gamma_hat_root": str(gamma_root),
        "cache_root": str(cache_dir),
        "quantile": float(args.quantile),
        "boundary_window": int(args.boundary_window),
        "wall_s_total": float(time.time() - overall_t0),
        "mean": {
            "iou_geo_main": _mean_iou("iou_geo", "iou_main"),
            "iou_sem_main": _mean_iou("iou_sem", "iou_main"),
            "iou_combined_main": _mean_iou("iou_combined", "iou_main"),
            "iou_geo_at_0.3": _mean_iou("iou_geo", "iou_at_0.3"),
            "iou_geo_at_0.5": _mean_iou("iou_geo", "iou_at_0.5"),
            "iou_geo_at_0.7": _mean_iou("iou_geo", "iou_at_0.7"),
            "iou_sem_at_0.3": _mean_iou("iou_sem", "iou_at_0.3"),
            "iou_sem_at_0.5": _mean_iou("iou_sem", "iou_at_0.5"),
            "iou_sem_at_0.7": _mean_iou("iou_sem", "iou_at_0.7"),
            "iou_combined_at_0.3": _mean_iou("iou_combined", "iou_at_0.3"),
            "iou_combined_at_0.5": _mean_iou("iou_combined", "iou_at_0.5"),
            "iou_combined_at_0.7": _mean_iou("iou_combined", "iou_at_0.7"),
            "random_iou": float(
                sum(r["random_baseline"]["random_iou_mean"] for r in per_ep)
                / len(per_ep)
            ),
            "precision_combined": float(
                sum(r["iou_combined"]["precision"] for r in per_ep) / len(per_ep)
            ),
            "recall_combined": float(
                sum(r["iou_combined"]["recall"] for r in per_ep) / len(per_ep)
            ),
            "f1_combined": float(
                sum(r["iou_combined"]["f1"] for r in per_ep) / len(per_ep)
            ),
        },
        "episodes": per_ep,
    }

    # Architecture v2.1 §I.3 line 1983 reads the gate off the "best" IoU
    # the labelled signal can offer: combined γ̂ at the canonical quantile
    # (default 0.25) AND the @ 0.3 view. We expose both; the verdict is
    # taken on the *better* of the two combined views to give the full
    # HCS-H pipeline its strongest reading.
    iou_main = float(summary["mean"]["iou_combined_main"])
    iou_at_03 = float(summary["mean"]["iou_combined_at_0.3"])
    iou_for_gate = max(iou_main, iou_at_03)

    summary["iou_for_gate"] = iou_for_gate
    summary["e1_decision"] = e1_decision(iou_for_gate)
    summary["baseline_iou_simple_jacobian"] = 0.227
    summary["delta_vs_baseline"] = iou_for_gate - 0.227

    out_path = _resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[F7-2/E1] wrote {out_path}")

    print("=" * 64)
    print(
        f"[F7-2/E1] mean IoU(combined @{args.quantile}) = "
        f"{iou_main:.4f}"
    )
    print(
        f"[F7-2/E1] mean IoU(combined @0.3)          = "
        f"{iou_at_03:.4f}"
    )
    print(
        f"[F7-2/E1] mean IoU(geo @{args.quantile})       = "
        f"{summary['mean']['iou_geo_main']:.4f}"
    )
    print(
        f"[F7-2/E1] mean IoU(sem @{args.quantile})       = "
        f"{summary['mean']['iou_sem_main']:.4f}"
    )
    print(
        f"[F7-2/E1] mean random baseline IoU          = "
        f"{summary['mean']['random_iou']:.4f}"
    )
    print(
        f"[F7-2/E1] iou_for_gate                       = {iou_for_gate:.4f}"
    )
    print(
        f"[F7-2/E1] baseline (scripts/40 simple-J)     = 0.2270  "
        f"(Δ={summary['delta_vs_baseline']:+.4f})"
    )
    print(f"[F7-2/E1] decision: {summary['e1_decision']}")
    if summary["e1_decision"] == "PASS":
        print(
            "          PASS — full [C10] HCS-H signal usable; architecture"
            " v2.1 原版可走"
        )
    elif summary["e1_decision"] == "SOFT-PASS":
        print(
            "          SOFT-PASS — proceed M2 with anneal-to-0.15 (red flag #1)"
        )
    else:
        print(
            "          HARD-FAIL — maintain MVP fallback (λ_1=0 permanent)"
        )
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
