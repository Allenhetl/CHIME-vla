#!/usr/bin/env python
"""E1 milestone gate: IoU(γ̂, sub_task_id boundary) on LIBERO-Long.

This is the M1 → M2 milestone gate (architecture v2.1 §I.3 line 1983):

    IoU @ 0.3 ≥ 0.4 → PASS       — proceed M2
    0.3 ≤ IoU < 0.4 → SOFT-PASS  — proceed M2 with red-flag #1
    IoU < 0.3 → HARD-FAIL        — fallback to MVP, λ_1 := 0 permanent,
                                   drop [C10][C12][C13]

Usage::

    python scripts/40_run_e1_judgment.py --n 5 \
        --output output/reports/e1_baseline_untrained.json
    python scripts/40_run_e1_judgment.py --n 50 \
        --checkpoint output/runs/m1/last.ckpt
    python scripts/40_run_e1_judgment.py --n 5 --deltas 4    # less RAM

The default uses an *untrained* :class:`ChimeVlaLightning` (random
initialisation) — useful as a sanity baseline.  Pass ``--checkpoint`` to
load a Lightning checkpoint and re-run the gate after K training steps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

# Add ./src to PYTHONPATH so this works pre-install.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chime_vla.config import ChimeConfig  # noqa: E402
from chime_vla.eval.e1_judgment import (  # noqa: E402
    compute_iou_vs_boundaries,
    compute_jacobian_saliency,
    e1_decision,
    random_baseline_iou,
)
from chime_vla.training.lightning_module import ChimeVlaLightning  # noqa: E402


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the M1 E1 IoU milestone-gate evaluation."
    )
    p.add_argument("--n", type=int, default=5, help="episodes to evaluate")
    p.add_argument(
        "--cache-root",
        default="output/cache/libero_long/libero_long",
        help="directory containing ep_NNNNNN.pt files",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="optional Lightning checkpoint to load (untrained CHIME if None)",
    )
    p.add_argument("--device", default="cuda", help="cuda / cpu / cuda:0 ...")
    p.add_argument(
        "--output",
        default="output/reports/e1_judgment.json",
        help="path to JSON summary",
    )
    p.add_argument(
        "--deltas",
        type=int,
        nargs="+",
        default=[4, 16],
        help="HCS-H delta horizons (frames)",
    )
    p.add_argument(
        "--quantile",
        type=float,
        default=0.25,
        help="top-quantile threshold for predicted peaks (default 0.25)",
    )
    p.add_argument(
        "--boundary-window",
        type=int,
        default=4,
        help="±N-frame window around each sub_task_id boundary",
    )
    p.add_argument(
        "--T-max",
        type=int,
        default=320,
        help="cap episode length to bound RAM (LIBERO-Long medians ≈ 280)",
    )
    p.add_argument(
        "--n-random-trials",
        type=int,
        default=64,
        help="MC trials for the random-peak baseline IoU",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--episode-ids",
        type=int,
        nargs="*",
        default=None,
        help="explicit episode ids; default is the first N episodes by id",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _select_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print(f"[E1] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def _build_model(ckpt_path: Optional[str], device: torch.device) -> ChimeVlaLightning:
    cfg = ChimeConfig()  # defaults match configs/base/chime.yaml
    if ckpt_path is not None:
        # Load Lightning checkpoint, instantiating with default cfg first
        # (the cfg-from-checkpoint round-trip is finicky under OmegaConf).
        module = ChimeVlaLightning.load_from_checkpoint(
            ckpt_path, cfg=OmegaConf.structured(cfg), strict=False
        )
        print(f"[E1] loaded checkpoint: {ckpt_path}")
    else:
        module = ChimeVlaLightning(OmegaConf.structured(cfg))
        print("[E1] running on UNTRAINED CHIME (random init).")
    module.to(device)
    module.eval()
    return module


def _load_episode(cache_dir: Path, ep_id: int, T_max: int) -> dict:
    blob = torch.load(
        cache_dir / f"ep_{ep_id:06d}.pt", map_location="cpu", weights_only=False
    )
    T_raw = int(blob["T"])
    T = min(T_raw, int(T_max))
    rgb_u8 = blob["rgb_raw"][:T]
    rgb = rgb_u8.to(torch.float32).div_(255.0).permute(0, 3, 1, 2).contiguous()
    proprio = blob["proprio"][:T].to(torch.float32)
    sub_task_id = blob["sub_task_id"][:T].to(torch.long)
    return {
        "rgb": rgb,
        "proprio": proprio,
        "sub_task_id": sub_task_id,
        "T": T,
        "T_raw": T_raw,
        "episode_id": int(blob.get("episode_id", ep_id)),
        "n_subtasks": int(sub_task_id.max().item()) + 1 if T > 0 else 0,
    }


def _list_episode_ids(cache_dir: Path) -> list[int]:
    import re

    pat = re.compile(r"^ep_(\d+)\.pt$")
    out = []
    for p in cache_dir.iterdir():
        m = pat.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    device = _select_device(args.device)
    torch.manual_seed(int(args.seed))

    print("=" * 60)
    print("CHIME-VLA M1 E1 milestone gate")
    print("=" * 60)
    print(f"[E1] device      : {device}")
    print(f"[E1] checkpoint  : {args.checkpoint or '<untrained>'}")
    print(f"[E1] deltas      : {args.deltas}")
    print(f"[E1] quantile    : {args.quantile}")
    print(f"[E1] window      : ±{args.boundary_window} frames")
    print(f"[E1] T_max       : {args.T_max}")
    print(f"[E1] n_episodes  : {args.n}")

    cache_dir = Path(args.cache_root)
    if not cache_dir.is_absolute():
        cache_dir = Path(__file__).resolve().parent.parent / cache_dir
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache dir not found: {cache_dir}")

    if args.episode_ids:
        ep_ids = list(args.episode_ids)[: int(args.n)]
    else:
        all_ids = _list_episode_ids(cache_dir)
        ep_ids = all_ids[: int(args.n)]
    print(f"[E1] episode ids : {ep_ids}")

    model = _build_model(args.checkpoint, device)

    is_mock = bool(getattr(getattr(model, "c1", None), "is_mock", False))
    if is_mock:
        print("[E1] WARNING: VLMBackbone in mock mode (no SigLIP weights).")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    per_ep: list[dict] = []
    overall_t0 = time.time()
    peak_alloc_b = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for i, ep_id in enumerate(ep_ids):
        ep = _load_episode(cache_dir, ep_id, T_max=int(args.T_max))
        T = ep["T"]
        if T <= max(args.deltas):
            print(
                f"[E1] [ep {ep_id}] skip — T={T} <= max delta={max(args.deltas)}"
            )
            continue

        t0 = time.time()
        try:
            sal = compute_jacobian_saliency(
                model,
                ep["rgb"],
                ep["proprio"],
                deltas=args.deltas,
                device=device,
            )
        except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
            print(f"[E1] OOM on ep {ep_id} (T={T}); retry with deltas=[min].")
            torch.cuda.empty_cache()
            sal = compute_jacobian_saliency(
                model,
                ep["rgb"],
                ep["proprio"],
                deltas=[min(args.deltas)],
                device=device,
                chunk_size=1,
            )

        wall = time.time() - t0

        gamma_geo = sal["gamma_geo"]
        gamma_sem = sal["gamma_sem"]

        iou_geo = compute_iou_vs_boundaries(
            gamma_geo,
            ep["sub_task_id"],
            quantile=float(args.quantile),
            boundary_window=int(args.boundary_window),
        )
        iou_sem = compute_iou_vs_boundaries(
            gamma_sem,
            ep["sub_task_id"],
            quantile=float(args.quantile),
            boundary_window=int(args.boundary_window),
        )

        # Combined γ — element-wise max of the two channels (treat either
        # geometric or semantic salience as "interesting"); this is the
        # default used by the milestone-gate reading.
        gamma_combined = torch.maximum(gamma_geo, gamma_sem)
        iou_combined = compute_iou_vs_boundaries(
            gamma_combined,
            ep["sub_task_id"],
            quantile=float(args.quantile),
            boundary_window=int(args.boundary_window),
        )

        rand_stats = random_baseline_iou(
            ep["sub_task_id"],
            quantile=float(args.quantile),
            boundary_window=int(args.boundary_window),
            n_trials=int(args.n_random_trials),
            seed=int(args.seed) + ep_id,
        )

        # GPU memory snapshot.
        gpu_peak_mb: float | None = None
        if device.type == "cuda":
            mem = torch.cuda.max_memory_allocated(device)
            peak_alloc_b = max(peak_alloc_b, int(mem))
            gpu_peak_mb = float(mem) / (1024 ** 2)

        rec = {
            "episode_id": ep_id,
            "T": T,
            "T_raw": ep["T_raw"],
            "n_pairs": int(sal["n_pairs"]),
            "wall_s": float(wall),
            "gpu_peak_mb": gpu_peak_mb,
            "iou_geo": iou_geo,
            "iou_sem": iou_sem,
            "iou_combined": iou_combined,
            "random_baseline": rand_stats,
            "gamma_geo_mean": float(gamma_geo.mean()),
            "gamma_geo_std": float(gamma_geo.std(unbiased=False)),
            "gamma_sem_mean": float(gamma_sem.mean()),
            "gamma_sem_std": float(gamma_sem.std(unbiased=False)),
            "raw_geo_mean": float(sal["raw_geo"].mean()),
            "raw_sem_mean": float(sal["raw_sem"].mean()),
        }
        per_ep.append(rec)
        print(
            f"[E1] [ep {ep_id} T={T} pairs={rec['n_pairs']} {wall:.1f}s "
            f"peak={gpu_peak_mb:.0f}MB] "
            f"IoU(geo @0.25)={iou_geo['iou_main']:.3f}  "
            f"IoU(sem @0.25)={iou_sem['iou_main']:.3f}  "
            f"IoU(comb @0.25)={iou_combined['iou_main']:.3f}  "
            f"random={rand_stats['random_iou_mean']:.3f}±{rand_stats['random_iou_std']:.3f}  "
            f"n_b={iou_combined['n_boundaries']}"
        )

        # Eagerly free per-episode autograd graph.
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------
    if not per_ep:
        print("[E1] no episodes processed.")
        return 1

    def _mean(field: str) -> float:
        return float(sum(r[field] for r in per_ep) / len(per_ep))

    def _mean_iou(channel: str, key: str) -> float:
        return float(sum(r[channel][key] for r in per_ep) / len(per_ep))

    summary = {
        "n_episodes": len(per_ep),
        "checkpoint": args.checkpoint,
        "untrained": args.checkpoint is None,
        "vlm_is_mock": is_mock,
        "deltas": list(args.deltas),
        "quantile": float(args.quantile),
        "boundary_window": int(args.boundary_window),
        "wall_total_s": float(time.time() - overall_t0),
        "gpu_peak_mb_overall": float(peak_alloc_b) / (1024 ** 2)
        if device.type == "cuda"
        else None,
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
            "wall_s": _mean("wall_s"),
        },
        "episodes": per_ep,
    }

    # The milestone gate is read off the COMBINED γ̂ at top-25 % vs the
    # ±4-frame extended boundary set, which is what we compute as
    # ``iou_combined.iou_main``.  ``iou_at_0.3`` is exposed too for
    # documentation and stricter / more lenient cuts.
    iou_gate = summary["mean"]["iou_combined_main"]
    summary["e1_decision"] = e1_decision(iou_gate)
    summary["iou_for_gate"] = float(iou_gate)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[E1] wrote {out_path}")

    # Banner.
    print("=" * 60)
    print(f"[E1] mean IoU(combined @{args.quantile}) = {iou_gate:.4f}")
    print(
        f"[E1] mean IoU(geo @{args.quantile})      = "
        f"{summary['mean']['iou_geo_main']:.4f}"
    )
    print(
        f"[E1] mean IoU(sem @{args.quantile})      = "
        f"{summary['mean']['iou_sem_main']:.4f}"
    )
    print(
        f"[E1] mean random baseline IoU            = "
        f"{summary['mean']['random_iou']:.4f}"
    )
    print(f"[E1] decision: {summary['e1_decision']}")
    if summary["e1_decision"] == "PASS":
        print("     PASS — proceed M2")
    elif summary["e1_decision"] == "SOFT-PASS":
        print("     SOFT-PASS — proceed M2 with red flag #1")
    else:
        print(
            "     HARD-FAIL — fallback to MVP (λ_1=0 永久, drop [C10][C12][C13])"
        )
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
