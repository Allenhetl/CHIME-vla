#!/usr/bin/env python
"""F7 Phase 2 step 1 — offline Hindsight Causal Saliency labelling.

This script produces the ``γ̂_geo / γ̂_sem`` per-frame saliency labels that
fulfil the file-protocol contract in ``docs/hindsight_contract.md`` §3.
It composes the full [C10] HCS-H pipeline (architecture v2.1 §C lines
1268-1349) over a frozen base-policy checkpoint:

    1. Load M4 ``last.ckpt`` and freeze every parameter.
    2. Extract per-frame mean-pooled ``h_t`` features for the first
       ``--rudder-train-episodes`` episodes (no-grad, fp32).
    3. Train a small :class:`RudderLSTM` to predict the cumulative
       success indicator from these features (BCE-with-logits).  Save
       the trained head to ``output/runs/rudder_for_hcs.pt``.
    4. Iterate the first ``--n-episodes`` episodes; per episode call
       :meth:`HCSHead.compute` to produce ``(γ̂_geo, γ̂_sem)`` and write
       a single ``ep_NNNNNN.pt`` per the documented schema.
    5. Emit ``_meta.json`` with aggregate statistics.

Per architecture v2.1 line 1297-1308 the segment-scoped autograd in
:meth:`HCSHead._forward_segment` keeps GPU memory bounded by
``(Δ_max + 1)`` frames, well within the 30 GB budget on a 4090 even
for ``T ~ 280`` and ``Δ ∈ {4, 16}``.

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/05_compute_hcs_saliency.py \\
        --base-checkpoint output/runs/m4_long_600step/last.ckpt \\
        --n-episodes 50 --rudder-train-episodes 30 --deltas 4 16
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

# Add ./src so this works pre-install.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chime_vla.config import ChimeConfig  # noqa: E402
from chime_vla.training.hcs_head import HCSHead, RudderLSTM  # noqa: E402
from chime_vla.training.lightning_module import ChimeVlaLightning  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_ROOT = (
    "/home/sqmluser/workspace/theaj/CHIME-VLA/Hindsight/output/saliency/"
    "gamma_hat/per_task_q75/libero_long"
)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--base-checkpoint",
        default="output/runs/m4_long_600step/last.ckpt",
        help="Lightning checkpoint of the frozen base policy.",
    )
    p.add_argument(
        "--cache-root",
        default="output/cache/libero_long/libero_long",
        help="LIBERO cache directory (ep_NNNNNN.pt).",
    )
    p.add_argument(
        "--output-root",
        default=_DEFAULT_OUTPUT_ROOT,
        help="Where γ̂.pt files land — defaults to the file-protocol path.",
    )
    p.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Total episodes to label.",
    )
    p.add_argument(
        "--rudder-train-episodes",
        type=int,
        default=30,
        help="Number of episodes used to fit the RUDDER LSTM.",
    )
    p.add_argument(
        "--rudder-epochs",
        type=int,
        default=100,
        help="RUDDER fitting epochs (full-batch AdamW).",
    )
    p.add_argument(
        "--rudder-lr",
        type=float,
        default=1e-3,
    )
    p.add_argument(
        "--rudder-checkpoint",
        default="output/runs/rudder_for_hcs.pt",
        help="Where to save the trained RUDDER head.",
    )
    p.add_argument(
        "--deltas",
        type=int,
        nargs="+",
        default=[4, 16],
        help="HCS-H delta horizons.",
    )
    p.add_argument(
        "--alpha-J",
        type=float,
        default=1.0,
        help="Coefficient on z(J) when fusing.",
    )
    p.add_argument(
        "--alpha-R",
        type=float,
        default=0.5,
        help="Coefficient on z(c_R) when fusing.",
    )
    p.add_argument(
        "--T-max",
        type=int,
        default=320,
        help="Cap episode length (LIBERO-Long medians ≈ 280).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--episode-ids",
        type=int,
        nargs="*",
        default=None,
        help="Explicit episode ids; default uses the first N by id.",
    )
    p.add_argument(
        "--strategy",
        default="per_task_q75",
        help="strategy tag written into meta.",
    )
    p.add_argument(
        "--task-name",
        default="libero_long",
    )
    p.add_argument(
        "--base-policy-tag",
        default="chime_m4_600step",
        help="Free-form base-policy identifier written to meta.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-compute episodes even if their ep_*.pt already exists.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _select_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[F7-2] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def _resolve(p: str) -> Path:
    pp = Path(p)
    if not pp.is_absolute():
        pp = Path(__file__).resolve().parent.parent / pp
    return pp


def _list_episode_ids(cache_dir: Path) -> list[int]:
    pat = re.compile(r"^ep_(\d+)\.pt$")
    out: list[int] = []
    for entry in cache_dir.iterdir():
        m = pat.match(entry.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _load_episode(cache_dir: Path, ep_id: int, T_max: int) -> dict:
    blob = torch.load(
        cache_dir / f"ep_{ep_id:06d}.pt", map_location="cpu", weights_only=False
    )
    T_raw = int(blob["T"])
    T = min(T_raw, int(T_max))
    rgb_u8 = blob["rgb_raw"][:T]                                  # (T, H, W, 3) uint8
    rgb = rgb_u8.to(torch.float32).div_(255.0).permute(0, 3, 1, 2).contiguous()
    proprio = blob["proprio"][:T].to(torch.float32)
    action = blob["action"][:T].to(torch.float32)
    rewards = blob["rewards"][:T].to(torch.float32)
    sub_task_id = blob["sub_task_id"][:T].to(torch.long)
    return {
        "rgb": rgb,
        "proprio": proprio,
        "action": action,
        "rewards": rewards,
        "sub_task_id": sub_task_id,
        "T": T,
        "T_raw": T_raw,
        "episode_id": int(blob.get("episode_id", ep_id)),
    }


def _build_model(ckpt_path: Path, device: torch.device) -> ChimeVlaLightning:
    cfg = ChimeConfig()
    if ckpt_path.exists():
        module = ChimeVlaLightning.load_from_checkpoint(
            str(ckpt_path), cfg=OmegaConf.structured(cfg), strict=False
        )
        print(f"[F7-2] loaded base checkpoint: {ckpt_path}")
    else:
        # Dev / smoke fallback — Phase 2 should always have an M4 ckpt; we
        # keep this so a missing ckpt yields a clear error rather than a
        # mysterious load_from_checkpoint trace.
        raise FileNotFoundError(f"base checkpoint not found: {ckpt_path}")
    module.to(device)
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


@torch.no_grad()
def _extract_features_for_rudder(
    model: ChimeVlaLightning,
    rgb: torch.Tensor,
    proprio: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return ``(T, d_h)`` mean-pooled ``h_t`` features in fp32."""
    T = int(rgb.shape[0])
    feats: list[torch.Tensor] = []
    for t in range(T):
        rgb_dev = rgb[t : t + 1].to(device=device, dtype=torch.float32)
        prop_dev = proprio[t : t + 1].to(device=device, dtype=torch.float32)
        h_t = model.c1(rgb_dev, prop_dev)            # (1, N, d_h)
        feats.append(h_t.mean(dim=1).squeeze(0).float().cpu())
    return torch.stack(feats, dim=0)                  # (T, d_h)


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    device = _select_device(args.device)

    print("=" * 64)
    print("F7-2  Offline HCS-H γ̂ saliency labelling")
    print("=" * 64)
    print(f"[F7-2] device           : {device}")
    print(f"[F7-2] base checkpoint  : {args.base_checkpoint}")
    print(f"[F7-2] cache root       : {args.cache_root}")
    print(f"[F7-2] output root      : {args.output_root}")
    print(f"[F7-2] n-episodes       : {args.n_episodes}")
    print(f"[F7-2] rudder-train ep  : {args.rudder_train_episodes}")
    print(f"[F7-2] deltas           : {args.deltas}")
    print(f"[F7-2] α_J / α_R        : {args.alpha_J} / {args.alpha_R}")

    cache_dir = _resolve(args.cache_root)
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache dir not found: {cache_dir}")
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rudder_ckpt = _resolve(args.rudder_checkpoint)
    rudder_ckpt.parent.mkdir(parents=True, exist_ok=True)

    if args.episode_ids:
        all_ep_ids = list(args.episode_ids)
    else:
        all_ep_ids = _list_episode_ids(cache_dir)
    n_total = min(int(args.n_episodes), len(all_ep_ids))
    ep_ids = all_ep_ids[:n_total]
    n_train = min(int(args.rudder_train_episodes), n_total)
    rudder_ep_ids = ep_ids[:n_train]

    print(f"[F7-2] episode ids      : first {len(ep_ids)} ids "
          f"(rudder train: first {n_train})")

    # ---------------------------------------------------------------
    # 1. Load + freeze base policy
    # ---------------------------------------------------------------
    ckpt_path = _resolve(args.base_checkpoint)
    model = _build_model(ckpt_path, device)
    is_mock = bool(getattr(getattr(model, "c1", None), "is_mock", False))
    if is_mock:
        print("[F7-2] WARNING: VLMBackbone in mock mode (no SigLIP weights).")
    d_h = int(model.c1.d_h)
    print(f"[F7-2] base policy d_h  : {d_h}")

    # ---------------------------------------------------------------
    # 2. Extract features + 3. Fit RUDDER LSTM on first n_train episodes
    # ---------------------------------------------------------------
    print("-" * 64)
    print(f"[F7-2] step 2 — extracting features for RUDDER from {n_train} eps ...")
    rudder_feats: list[torch.Tensor] = []
    rudder_targets: list[torch.Tensor] = []
    t_extract = time.time()
    for i, ep_id in enumerate(rudder_ep_ids):
        ep = _load_episode(cache_dir, ep_id, T_max=int(args.T_max))
        feats = _extract_features_for_rudder(
            model, ep["rgb"], ep["proprio"], device
        )                                                   # (T, d_h) fp32 on cpu
        # Cumulative-success target: 1 from the first success frame onwards;
        # for LIBERO sparse reward this is cumsum(rewards).clamp(max=1).
        # If the trajectory has no success at all, target is all-zeros (the
        # episode never resolves — RUDDER learns "low-confidence prefix").
        cum = ep["rewards"].cumsum(0).clamp(max=1.0)
        rudder_feats.append(feats)
        rudder_targets.append(cum)
        if (i + 1) % 5 == 0 or (i + 1) == n_train:
            print(f"[F7-2]   extracted ep {ep_id} ({i + 1}/{n_train}) "
                  f"T={feats.shape[0]} success={float(cum[-1]):.0f}")
    extract_wall = time.time() - t_extract
    print(f"[F7-2] feature extraction wall: {extract_wall:.1f} s")

    # ---------------------------------------------------------------
    # 3. Train RUDDER
    # ---------------------------------------------------------------
    print("-" * 64)
    print(f"[F7-2] step 3 — fitting RUDDER LSTM for {args.rudder_epochs} epochs ...")
    rudder = RudderLSTM(d_feat=d_h, d_hidden=256).to(device)
    head_for_fit = HCSHead(
        base_policy=model,
        deltas=tuple(int(d) for d in args.deltas),
        rudder=rudder,
        alpha_J=float(args.alpha_J),
        alpha_R=float(args.alpha_R),
        device=device,
    )
    t_rudder = time.time()
    fit_stats = head_for_fit.fit_rudder(
        rudder_feats, rudder_targets,
        epochs=int(args.rudder_epochs),
        lr=float(args.rudder_lr),
        verbose=False,
    )
    rudder_wall = time.time() - t_rudder
    print(f"[F7-2] RUDDER fit wall  : {rudder_wall:.1f} s")
    print(f"[F7-2] RUDDER loss      : {fit_stats['loss_first']:.4f} → "
          f"{fit_stats['loss_last']:.4f}")
    torch.save(
        {
            "state_dict": rudder.state_dict(),
            "d_feat": d_h,
            "d_hidden": 256,
            "epochs": int(args.rudder_epochs),
            "lr": float(args.rudder_lr),
            "n_train_episodes": n_train,
            "fit_stats": fit_stats,
        },
        rudder_ckpt,
    )
    print(f"[F7-2] saved RUDDER to  : {rudder_ckpt}")

    # ---------------------------------------------------------------
    # 4. Per-episode HCS γ̂ computation
    # ---------------------------------------------------------------
    print("-" * 64)
    print(f"[F7-2] step 4 — computing γ̂ for {len(ep_ids)} episodes ...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    git_sha = _git_sha()
    overall_t0 = time.time()
    per_ep_records: list[dict] = []
    n_skipped = 0
    n_written = 0

    for i, ep_id in enumerate(ep_ids):
        out_path = output_root / f"ep_{ep_id:06d}.pt"
        if out_path.exists() and not args.overwrite:
            print(f"[F7-2] [{i + 1}/{len(ep_ids)} ep {ep_id}] skip — exists")
            n_skipped += 1
            continue
        ep = _load_episode(cache_dir, ep_id, T_max=int(args.T_max))
        T = ep["T"]
        delta_max = max(int(d) for d in args.deltas)
        if T <= delta_max:
            print(f"[F7-2] [{i + 1}/{len(ep_ids)} ep {ep_id}] skip — "
                  f"T={T} ≤ Δ_max={delta_max}")
            n_skipped += 1
            continue

        cum_target = ep["rewards"].cumsum(0).clamp(max=1.0)
        t0 = time.time()
        try:
            result = head_for_fit.compute(
                rgb_seq=ep["rgb"],
                proprio_seq=ep["proprio"],
                action_seq=ep["action"],
                reward_seq=cum_target,
            )
        except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
            print(f"[F7-2] OOM on ep {ep_id} (T={T}); retrying with deltas=[min].")
            torch.cuda.empty_cache()
            head_fallback = HCSHead(
                base_policy=model,
                deltas=(min(args.deltas),),
                rudder=rudder,
                alpha_J=float(args.alpha_J),
                alpha_R=float(args.alpha_R),
                device=device,
            )
            result = head_fallback.compute(
                rgb_seq=ep["rgb"],
                proprio_seq=ep["proprio"],
                action_seq=ep["action"],
                reward_seq=cum_target,
            )
        wall = time.time() - t0

        gamma_geo = result["gamma_geo"].detach().to(torch.float32).cpu()
        gamma_sem = result["gamma_sem"].detach().to(torch.float32).cpu()
        valid_mask = torch.ones(T, dtype=torch.bool)

        meta = {
            "strategy": str(args.strategy),
            "base_policy": str(args.base_policy_tag),
            "delta_set": [int(d) for d in args.deltas],
            "saliency_method": "EAGN+RUDDER",
            "computed_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "hindsight_commit": git_sha,
            "alpha_J": float(args.alpha_J),
            "alpha_R": float(args.alpha_R),
            "rudder_attached": True,
            "n_pairs": int(result["meta"].get("n_pairs", 0)),
            "T_max": int(args.T_max),
            "T_raw": int(ep["T_raw"]),
        }
        blob_out = {
            "episode_id": int(ep["episode_id"]),
            "task_name": str(args.task_name),
            "T": int(T),
            "gamma_geo": gamma_geo,
            "gamma_sem": gamma_sem,
            "valid_mask": valid_mask,
            "meta": meta,
        }
        torch.save(blob_out, out_path)
        n_written += 1

        gpu_peak_mb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2)
            if device.type == "cuda"
            else None
        )
        rec = {
            "episode_id": int(ep["episode_id"]),
            "T": int(T),
            "wall_s": float(wall),
            "gpu_peak_mb": gpu_peak_mb,
            "gamma_geo_mean": float(gamma_geo.mean()),
            "gamma_geo_std": float(gamma_geo.std(unbiased=False)),
            "gamma_sem_mean": float(gamma_sem.mean()),
            "gamma_sem_std": float(gamma_sem.std(unbiased=False)),
            "n_pairs": int(meta["n_pairs"]),
        }
        per_ep_records.append(rec)
        print(
            f"[F7-2] [{i + 1}/{len(ep_ids)} ep {ep_id} T={T} pairs={rec['n_pairs']} "
            f"{wall:.1f}s peak={gpu_peak_mb if gpu_peak_mb is None else f'{gpu_peak_mb:.0f}MB'}] "
            f"γ̂_geo μ={rec['gamma_geo_mean']:.3f} σ={rec['gamma_geo_std']:.3f}  "
            f"γ̂_sem μ={rec['gamma_sem_mean']:.3f} σ={rec['gamma_sem_std']:.3f}"
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_wall = time.time() - overall_t0

    # ---------------------------------------------------------------
    # 5. _meta.json
    # ---------------------------------------------------------------
    meta_path = output_root / "_meta.json"
    summary = {
        "task_name": str(args.task_name),
        "strategy": str(args.strategy),
        "base_policy": str(args.base_policy_tag),
        "base_checkpoint": str(ckpt_path),
        "deltas": [int(d) for d in args.deltas],
        "alpha_J": float(args.alpha_J),
        "alpha_R": float(args.alpha_R),
        "n_episodes_requested": int(args.n_episodes),
        "n_episodes_written": int(n_written),
        "n_episodes_skipped": int(n_skipped),
        "n_episodes_total_in_dir": len(per_ep_records) + n_skipped,
        "rudder": {
            "checkpoint": str(rudder_ckpt),
            "n_train_episodes": int(n_train),
            "epochs": int(args.rudder_epochs),
            "lr": float(args.rudder_lr),
            "loss_first": float(fit_stats["loss_first"]),
            "loss_last": float(fit_stats["loss_last"]),
            "wall_s": float(rudder_wall),
            "feature_extraction_wall_s": float(extract_wall),
        },
        "wall_s_total_compute": float(total_wall),
        "wall_s_per_episode_mean": (
            float(sum(r["wall_s"] for r in per_ep_records) / max(1, len(per_ep_records)))
        ),
        "computed_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "hindsight_commit": git_sha,
        "vlm_is_mock": bool(is_mock),
        "episodes": per_ep_records,
    }
    with meta_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print("=" * 64)
    print(f"[F7-2] wrote {n_written} γ̂ episodes to {output_root}")
    print(f"[F7-2] _meta.json        : {meta_path}")
    print(f"[F7-2] total wall        : {total_wall:.1f} s "
          f"({total_wall / 60.0:.1f} min)")
    if per_ep_records:
        print(f"[F7-2] per-episode mean  : "
              f"{summary['wall_s_per_episode_mean']:.1f} s")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
