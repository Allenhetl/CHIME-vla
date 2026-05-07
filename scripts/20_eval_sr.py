#!/usr/bin/env python
"""LIBERO-Long held-out SR proxy evaluation (M4 deliverable).

Implements an **action MSE-based SR proxy** as a stand-in for the full
LIBERO simulator-driven success-rate measurement (which is deferred until
the LIBERO sim integration lands).

For each episode in the held-out test split, we:
    1. Load a trained Lightning checkpoint (or random init for smoke).
    2. Run the full per-step forward through the episode in eval+no_grad.
    3. Record per-step predicted action a_pred, compute MSE vs ground
       truth a*, and an in-tolerance hit rate (|a_pred - a*| < 0.5σ).
    4. Aggregate to mean MSE, per-step error curve, completion rate,
       and per-episode MSE list.

Usage::

    python scripts/20_eval_sr.py \\
        --checkpoint output/runs/m4_long_600step/last.ckpt \\
        --split test \\
        --n-rollouts 20 \\
        --output output/reports/m4_sr_proxy.json

A "rollout" here is one full forward pass over an episode; we are not
doing closed-loop env interaction (no LIBERO sim).  Hence the name SR
*proxy*: action-MSE-based progress signal, suitable as a relative
metric across checkpoints / ablations.
"""

from __future__ import annotations

import argparse
import json
import math
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
from chime_vla.training.datamodule import LiberoLongDataModule  # noqa: E402
from chime_vla.training.lightning_module import ChimeVlaLightning  # noqa: E402


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LIBERO-Long held-out action-MSE SR proxy."
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
        default=20,
        help="Number of episodes to evaluate (capped at split size).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    p.add_argument(
        "--T-max",
        type=int,
        default=256,
        help="Cap episode length for forward pass (LIBERO-Long ~268 frames).",
    )
    p.add_argument(
        "--tolerance-frac",
        type=float,
        default=0.5,
        help="Multiple of σ_action used as hit tolerance for completion proxy.",
    )
    p.add_argument(
        "--allow-untrained",
        action="store_true",
        help="If set and --checkpoint missing, run on random init (smoke).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("output/reports/m4_sr_proxy.json"),
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _select_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[20_eval_sr] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def _build_cfg(T_max: int) -> ChimeConfig:
    """Build the eval-time ChimeConfig.

    Forces the M4-eval contract:
      * hindsight.enabled = False (eval doesn't need γ̂; the cache-loader
        path tolerates missing γ̂ gracefully but we skip it for speed).
      * data.T_max set to args.T_max so the full episode is processed.
      * loss.lambda_2 / lambda_3 = 0 so chime_train_step skips PRH/CSM
        forwards (no need at inference for SR proxy).
    """
    cfg = ChimeConfig()
    cfg.data.T_max = int(T_max)
    cfg.hindsight.enabled = False
    # Skip PRH/CSM/HCS forwards — none affect a_pred values at eval time.
    cfg.loss.lambda_1_target = 0.0
    cfg.loss.lambda_1_schedule = "off"
    cfg.loss.lambda_2 = 0.0
    cfg.loss.lambda_3 = 0.0
    cfg.loss.lambda_predict = 0.0
    return cfg


def _build_model(
    ckpt_path: Optional[Path],
    cfg: ChimeConfig,
    device: torch.device,
    allow_untrained: bool,
) -> ChimeVlaLightning:
    structured = OmegaConf.structured(cfg)
    if ckpt_path is not None and ckpt_path.exists():
        module = ChimeVlaLightning.load_from_checkpoint(
            str(ckpt_path), cfg=structured, strict=False
        )
        print(f"[20_eval_sr] loaded checkpoint: {ckpt_path}")
    else:
        if ckpt_path is not None and not allow_untrained:
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}.  "
                f"Pass --allow-untrained to run on random init."
            )
        module = ChimeVlaLightning(structured)
        print("[20_eval_sr] running on UNTRAINED CHIME (random init).")
    module.to(device)
    module.eval()
    return module


def _action_std_from_dataset(dataset) -> torch.Tensor:
    """Pull the per-dim action std vector loaded by the datamodule.

    Falls back to ones if normalization stats weren't found (in which case
    actions are in raw scale).  Shape: (action_dim,).
    """
    s = getattr(dataset, "action_std", None)
    if s is None:
        # Fall back to action_dim from one sample.
        return torch.ones(int(dataset.action_dim), dtype=torch.float32)
    return s.detach().clone().to(torch.float32)


def _per_step_metrics(
    a_pred: torch.Tensor,         # (B, T, A)
    a_gt: torch.Tensor,           # (B, T, A)
    valid_mask: torch.Tensor,     # (B, T) bool
    sigma: torch.Tensor,          # (A,) — z-score units → 1.0 by construction
    tolerance_frac: float,
) -> dict:
    """Compute per-(batch, t) MSE and per-(batch, t, A) hit mask.

    Returns:
        per_step_mse:         (B, T) — MSE averaged over action dims
        per_step_err_l2:      (B, T) — L2 error magnitude
        per_step_hit:         (B, T) — fraction of action dims within tolerance
        valid_mask:           (B, T) bool — passed through
    """
    diff = a_pred - a_gt                            # (B, T, A)
    sq = diff.pow(2)                                # (B, T, A)
    per_step_mse = sq.mean(dim=-1)                  # (B, T)
    per_step_err_l2 = sq.sum(dim=-1).clamp_min(0).sqrt()  # (B, T)

    # Hit per dim: |Δ| < tolerance_frac * sigma.  When data is z-score
    # normalised (default), sigma == 1 in normalised space, so the bound
    # reduces to tolerance_frac.  We still pass sigma in for correctness
    # if normalisation is disabled.
    tol = (sigma * float(tolerance_frac)).abs()     # (A,)
    hit_per_dim = diff.abs() < tol.view(1, 1, -1)   # (B, T, A)
    per_step_hit = hit_per_dim.float().mean(dim=-1) # (B, T)

    return {
        "per_step_mse": per_step_mse,
        "per_step_err_l2": per_step_err_l2,
        "per_step_hit": per_step_hit,
        "valid_mask": valid_mask,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _select_device(args.device)
    torch.manual_seed(int(args.seed))

    print("=" * 60)
    print("CHIME-VLA M4 — held-out SR proxy (action MSE)")
    print("=" * 60)
    print(f"[20_eval_sr] device     : {device}")
    print(f"[20_eval_sr] checkpoint : {args.checkpoint}")
    print(f"[20_eval_sr] split      : {args.split}")
    print(f"[20_eval_sr] n_rollouts : {args.n_rollouts}")
    print(f"[20_eval_sr] T_max      : {args.T_max}")
    print(f"[20_eval_sr] tolerance  : {args.tolerance_frac:.2f} σ")

    cfg = _build_cfg(T_max=int(args.T_max))

    # ------- datamodule -------
    dm = LiberoLongDataModule(OmegaConf.structured(cfg), batch_size=1)
    dm.setup(stage="test")
    if args.split == "train":
        ds = dm.train_ds
    elif args.split == "val":
        ds = dm.val_ds
    else:
        ds = dm.test_ds
    if ds is None or len(ds) == 0:
        print(f"[20_eval_sr] empty split: {args.split}", file=sys.stderr)
        return 2
    n_total = len(ds)
    n = min(int(args.n_rollouts), n_total)
    print(f"[20_eval_sr] split size : {n_total}  →  evaluating {n}")

    # ------- model -------
    model = _build_model(
        args.checkpoint if args.checkpoint else None,
        cfg,
        device,
        allow_untrained=bool(args.allow_untrained),
    )

    # ------- per-action std (in normalised space; ones unless normalize=False) -------
    sigma = _action_std_from_dataset(ds).to(device)
    # If z-score normalised at the dataset layer, sigma in *processed* units
    # is effectively 1.0 per dim — the data already has std≈1.  Encode that
    # explicitly when the dataset has normalisation stats.
    if getattr(ds, "action_std", None) is not None:
        sigma_eff = torch.ones_like(sigma)
    else:
        sigma_eff = sigma
    print(
        f"[20_eval_sr] tolerance σ (per-dim, normalised) "
        f"= {sigma_eff.detach().cpu().tolist()}"
    )

    # ------- main eval loop -------
    per_episode: list[dict] = []
    # Per-step error curve accumulator: list-of-lists indexed by t, each
    # holding the per-episode MSE at that t.  We then compute mean/median
    # at the end.  Length T_max.
    per_step_mse_accum: list[list[float]] = [[] for _ in range(int(args.T_max))]
    n_hits_total = 0
    n_preds_total = 0

    overall_t0 = time.time()
    for i in range(n):
        sample = ds[i]
        # Build a B=1 batch dict on device.
        batch: dict[str, torch.Tensor] = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(device)
            else:
                batch[k] = v
        # episode_id collated as a long tensor in the datamodule; here we
        # just keep the int (chime_train_step ignores it).
        ep_id = int(sample["episode_id"])
        T_valid = int(batch["valid_mask"].sum().item())

        # Mirror chime_train_step's per-step loop in eval+no_grad mode and
        # collect a_pred (chime_train_step returns only scalar losses, so
        # we run an equivalent forward here to surface the action sequence).
        # Memory containers are freshly allocated inside this helper, so
        # per-episode state reset is guaranteed.
        t0 = time.time()
        a_pred, a_gt, vmask = _forward_collect_actions(model, batch, cfg)
        wall = time.time() - t0
        # a_pred / a_gt: (1, T_max, A); vmask: (1, T_max) bool.

        metrics = _per_step_metrics(
            a_pred=a_pred.float(),
            a_gt=a_gt.float(),
            valid_mask=vmask,
            sigma=sigma_eff,
            tolerance_frac=float(args.tolerance_frac),
        )

        # Per-episode MSE: mean of per_step_mse over valid steps.
        valid = metrics["valid_mask"][0]                          # (T,)
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            print(f"[20_eval_sr] skip ep {ep_id} — no valid steps")
            continue
        psm = metrics["per_step_mse"][0]                          # (T,)
        psh = metrics["per_step_hit"][0]                          # (T,)
        ep_mse = float(psm[valid].mean().item())
        ep_l2 = float(metrics["per_step_err_l2"][0][valid].mean().item())
        # Hit rate: across (T_valid * action_dim) — psh is a per-step
        # FRACTION of dims within tol; mean over valid steps gives the
        # episode-level hit rate.
        ep_hit = float(psh[valid].mean().item())
        n_preds_ep = n_valid * int(a_pred.shape[-1])
        n_hits_ep = int(round(ep_hit * n_preds_ep))

        n_hits_total += n_hits_ep
        n_preds_total += n_preds_ep

        # Stash per-step MSE into the global curve (only for valid steps).
        psm_cpu = psm.detach().cpu().tolist()
        valid_cpu = valid.detach().cpu().tolist()
        for t, (m, v) in enumerate(zip(psm_cpu, valid_cpu)):
            if v and t < len(per_step_mse_accum):
                per_step_mse_accum[t].append(float(m))

        rec = {
            "episode_id": ep_id,
            "T_valid": int(n_valid),
            "wall_s": float(wall),
            "ep_action_mse": ep_mse,
            "ep_err_l2_mean": ep_l2,
            "ep_hit_rate": ep_hit,
            "first_quartile_mse": float(
                psm[valid][: max(1, n_valid // 4)].mean().item()
            ),
            "last_quartile_mse": float(
                psm[valid][-max(1, n_valid // 4) :].mean().item()
            ),
        }
        per_episode.append(rec)
        print(
            f"[20_eval_sr] [ep {ep_id} T={n_valid} {wall:.1f}s] "
            f"action MSE={ep_mse:.4f}  L2={ep_l2:.4f}  "
            f"hit@σ/2={ep_hit*100:.1f}%  "
            f"q1={rec['first_quartile_mse']:.3f} "
            f"q4={rec['last_quartile_mse']:.3f}"
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not per_episode:
        print("[20_eval_sr] no episodes processed.", file=sys.stderr)
        return 1

    # ------- aggregates -------
    mse_list = [r["ep_action_mse"] for r in per_episode]
    mean_mse = float(sum(mse_list) / len(mse_list))
    sorted_mse = sorted(mse_list)
    median_mse = float(sorted_mse[len(sorted_mse) // 2])
    completion_rate = (
        float(n_hits_total) / float(n_preds_total) if n_preds_total > 0 else 0.0
    )

    # Per-step curve: mean over episodes that had a valid step at each t.
    curve_mean: list[float] = []
    curve_n: list[int] = []
    for t, vals in enumerate(per_step_mse_accum):
        if vals:
            curve_mean.append(float(sum(vals) / len(vals)))
            curve_n.append(int(len(vals)))
        else:
            curve_mean.append(float("nan"))
            curve_n.append(0)

    # Quartile error growth on the curve (over t-indices that have data).
    valid_t_vals = [(t, m) for t, m in enumerate(curve_mean) if math.isfinite(m)]
    if valid_t_vals:
        n_t = len(valid_t_vals)
        q1_n = max(1, n_t // 4)
        q1_mean = float(sum(m for _, m in valid_t_vals[:q1_n]) / q1_n)
        q4_mean = float(sum(m for _, m in valid_t_vals[-q1_n:]) / q1_n)
        ratio = float(q4_mean / q1_mean) if q1_mean > 0 else float("inf")
    else:
        q1_mean = q4_mean = float("nan")
        ratio = float("nan")

    # ------- write JSON -------
    summary = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "split": args.split,
        "n_episodes": len(per_episode),
        "n_total_in_split": int(n_total),
        "T_max": int(args.T_max),
        "tolerance_frac": float(args.tolerance_frac),
        "wall_total_s": float(time.time() - overall_t0),
        "summary": {
            "action_mse_mean": mean_mse,
            "action_mse_median": median_mse,
            "completion_rate_proxy": completion_rate,
            "per_step_q1_mean_mse": q1_mean,
            "per_step_q4_mean_mse": q4_mean,
            "per_step_q4_over_q1_ratio": ratio,
        },
        "per_episode_mse": mse_list,
        "per_episode": per_episode,
        "action_mse_per_step_curve": {
            "mean": curve_mean,
            "n_episodes_per_t": curve_n,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[20_eval_sr] wrote {out_path}")

    # ------- verdict banner -------
    print("=" * 60)
    print("        M4 SR proxy")
    print("=" * 60)
    print(f"  n episodes           : {len(per_episode)}")
    print(f"  mean action MSE      : {mean_mse:.4f}")
    print(f"  median per-ep MSE    : {median_mse:.4f}")
    print(
        f"  completion rate proxy: {completion_rate*100:.1f}% "
        f"(tolerance {args.tolerance_frac:.2f}σ)"
    )
    print(
        f"  per-step error growth: q1={q1_mean:.3f}  "
        f"q4={q4_mean:.3f}  ratio={ratio:.2f}"
    )
    print("=" * 60)
    return 0


# ---------------------------------------------------------------------------
# Forward helper — run the per-step CHIME loop and collect a_pred / a_gt.
# ---------------------------------------------------------------------------


def _forward_collect_actions(
    model: ChimeVlaLightning,
    batch: dict[str, torch.Tensor],
    cfg: ChimeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a single eval forward and return (a_pred, a_gt, valid_mask).

    Mirrors the per-step loop in :func:`chime_vla.training.train_step.chime_train_step`
    minus the loss-side bookkeeping; we only need the action predictions.
    """
    from chime_vla.memory.geo_grid import GeoGrid
    from chime_vla.memory.sem_bank import SemBank
    from chime_vla.perception.fifo_buffer import WorkBuffer

    rgb = batch["rgb"]                # (B, T, 3, 224, 224)
    proprio = batch["proprio"]        # (B, T, 8)
    action_gt = batch["action"]       # (B, T, A)
    valid_mask = batch["valid_mask"]  # (B, T) bool
    B, T = rgb.shape[:2]
    device = rgb.device

    c2 = WorkBuffer(cfg.c2, batch_size=B, device=device)
    m_geo = GeoGrid(cfg.c6, batch_size=B, d_g=cfg.c6.d_g, device=device)
    m_sem = SemBank(cfg.c7, batch_size=B, device=device)

    a_pred_steps: list[torch.Tensor] = []
    with torch.no_grad():
        for t in range(T):
            rgb_t = rgb[:, t]
            proprio_t = proprio[:, t]

            h_t = model.c1(rgb_t, proprio_t)
            m_work_prev = c2.snapshot()
            gamma_geo, gamma_sem = model.c5(h_t, m_work_prev)

            if c2.K_w > 1:
                shifted = torch.cat(
                    [c2.buffer[:, 1:], h_t.to(c2.buffer.dtype).unsqueeze(1)],
                    dim=1,
                )
            else:
                shifted = h_t.to(c2.buffer.dtype).unsqueeze(1)
            c2.buffer = shifted
            c2._n_appended = torch.clamp(c2._n_appended + 1, max=c2.K_w)
            m_work_post = shifted

            model.c3(h_t, gamma_geo.detach(), m_geo, step=t)
            model.c4(h_t, gamma_sem.detach(), m_sem, step=t)

            c_t = model.c8(m_work_post, m_geo, m_sem, h_t)
            h_t_cls = h_t.mean(dim=1)
            a_pred_t = model.c9(c_t, h_t_cls)
            a_pred_steps.append(a_pred_t)

    a_pred = torch.stack(a_pred_steps, dim=1)  # (B, T, A)
    return a_pred, action_gt, valid_mask


if __name__ == "__main__":
    sys.exit(main())
