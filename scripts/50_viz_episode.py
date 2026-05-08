#!/usr/bin/env python
"""Render an mp4 with RGB + γ_geo/γ_sem + predicted vs gt action overlay.

Usage:
    python scripts/50_viz_episode.py \
        --checkpoint output/runs/m4_long_600step/last.ckpt \
        --episode-id 45 \
        --output output/viz/m4_ep_045.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import cv2  # noqa: E402

from chime_vla.action.action_expert import ActionExpert  # noqa: E402
from chime_vla.config import ChimeConfig  # noqa: E402
from chime_vla.heads.espc import ESPC  # noqa: E402
from chime_vla.heads.geo_write import GeoWriteHead  # noqa: E402
from chime_vla.heads.sem_write import SemWriteHead  # noqa: E402
from chime_vla.memory.geo_grid import GeoGrid  # noqa: E402
from chime_vla.memory.sem_bank import SemBank  # noqa: E402
from chime_vla.perception.fifo_buffer import WorkBuffer  # noqa: E402
from chime_vla.perception.vlm_backbone import VLMBackbone  # noqa: E402
from chime_vla.readout.read_interface import ReadInterface  # noqa: E402
from chime_vla.training.lightning_module import ChimeVlaLightning  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--episode-id", type=int, default=45)
    p.add_argument(
        "--cache-root",
        type=Path,
        default=Path("output/cache/libero_long/libero_long"),
    )
    p.add_argument("--output", type=Path, default=Path("output/viz/episode.mp4"))
    p.add_argument("--T-max", type=int, default=200)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _load_model(ckpt_path: Path, device: str) -> ChimeVlaLightning:
    cfg = ChimeConfig()
    cfg.loss.lambda_2 = 0.5
    cfg.loss.lambda_predict = 1.0
    cfg.loss.lambda_ent = 0.01
    cfg.c10.enabled = False
    cfg.hindsight.enabled = False
    model = ChimeVlaLightning.load_from_checkpoint(
        ckpt_path, cfg=cfg, strict=False, map_location=device,
    )
    model.eval()
    return model


def _load_episode(cache_dir: Path, episode_id: int, T_max: int):
    path = cache_dir / f"ep_{episode_id:06d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    T = min(int(blob["T"]), T_max)
    return {
        "rgb_raw": blob["rgb_raw"][:T],          # (T, 224, 224, 3) uint8
        "proprio": blob["proprio"][:T].float(),  # (T, 8)
        "action": blob["action"][:T].float(),    # (T, 8)
        "sub_task_id": blob["sub_task_id"][:T].numpy(),  # (T,)
        "T": T,
        "episode_id": episode_id,
    }


@torch.no_grad()
def _forward_episode(
    model: ChimeVlaLightning,
    rgb_raw: torch.Tensor,         # (T, 224, 224, 3) uint8
    proprio: torch.Tensor,         # (T, 8) fp32
    device: str,
):
    """Run model through the full episode; collect per-frame γ + a_pred."""
    cfg = model.cfg
    T = rgb_raw.shape[0]
    # rgb to fp32 [0,1], (T, 3, 224, 224)
    rgb_fp = rgb_raw.float().div_(255.0).permute(0, 3, 1, 2).contiguous()

    # Per-episode memory
    c2 = WorkBuffer(cfg.c2, batch_size=1, device=device)
    m_geo = GeoGrid(cfg.c6, batch_size=1, d_g=cfg.c6.d_g, device=device)
    m_sem = SemBank(cfg.c7, batch_size=1, device=device)

    gammas_geo, gammas_sem, a_preds = [], [], []
    m_geo_occ, m_sem_occ = [], []

    # Run frame-by-frame
    for t in range(T):
        rgb_t = rgb_fp[t : t + 1].to(device)
        proprio_t = proprio[t : t + 1].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            h_t = model.c1(rgb_t, proprio_t)
            m_work_prev = c2.snapshot()
            gamma_geo, gamma_sem = model.c5(h_t, m_work_prev)
            c2.append(h_t)
            model.c3(h_t, gamma_geo.detach(), m_geo, step=t)
            model.c4(h_t, gamma_sem.detach(), m_sem, step=t)
            c_t = model.c8(c2.snapshot(), m_geo, m_sem, h_t)
            a_t = model.c9(c_t, h_t.mean(dim=1))

        gammas_geo.append(float(gamma_geo[0].item()))
        gammas_sem.append(float(gamma_sem[0].item()))
        a_preds.append(a_t[0].float().cpu().numpy())
        m_geo_occ.append(m_geo.occupancy_pct().get(list(m_geo.grids.keys())[0], 0.0))
        m_sem_occ.append((1.0 - float(m_sem.slot_free.float().mean().item())))

    return {
        "gamma_geo": np.asarray(gammas_geo),                     # (T,)
        "gamma_sem": np.asarray(gammas_sem),                     # (T,)
        "a_pred": np.stack(a_preds),                             # (T, 8)
        "m_geo_occ": np.asarray(m_geo_occ),                      # (T,)
        "m_sem_occ": np.asarray(m_sem_occ),                      # (T,)
    }


def _draw_frame(
    rgb: np.ndarray,        # (224, 224, 3) uint8
    t: int,
    T: int,
    gamma_geo: float,
    gamma_sem: float,
    a_pred: np.ndarray,     # (8,)
    a_gt: np.ndarray,       # (8,)
    sub_task_id: int,
    m_geo_occ: float,
    m_sem_occ: float,
) -> np.ndarray:
    # Upscale to 448x448 for readability + add 280-px right panel
    canvas_h = 448
    canvas_w = 448 + 280
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    # Left: upscaled RGB
    rgb_big = cv2.resize(rgb, (448, 448), interpolation=cv2.INTER_LINEAR)
    canvas[:448, :448] = rgb_big

    # Top-left text (over RGB): frame index + sub_task_id
    cv2.putText(
        canvas, f"t={t:3d}/{T} sub={sub_task_id}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )

    # Right panel: stats
    panel_x = 460
    y = 30
    line_h = 22

    def put(text, color=(220, 220, 220)):
        nonlocal y
        cv2.putText(
            canvas, text, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
        y += line_h

    put("=== gamma (write gates) ===", (180, 220, 255))
    geo_color = (0, int(255 * gamma_geo), 0)
    sem_color = (int(255 * gamma_sem), 0, 0)
    put(f"gamma_geo: {gamma_geo:.3f}", geo_color)
    # bar
    bar_y = y - 18
    cv2.rectangle(canvas, (panel_x, bar_y - 2), (panel_x + 200, bar_y + 6), (60, 60, 60), -1)
    cv2.rectangle(
        canvas, (panel_x, bar_y - 2),
        (panel_x + int(200 * gamma_geo), bar_y + 6), geo_color, -1,
    )

    put(f"gamma_sem: {gamma_sem:.3f}", sem_color)
    bar_y = y - 18
    cv2.rectangle(canvas, (panel_x, bar_y - 2), (panel_x + 200, bar_y + 6), (60, 60, 60), -1)
    cv2.rectangle(
        canvas, (panel_x, bar_y - 2),
        (panel_x + int(200 * gamma_sem), bar_y + 6), sem_color, -1,
    )

    y += 8
    put("=== memory ===", (180, 220, 255))
    put(f"M_geo occ: {m_geo_occ * 100:5.2f}%")
    put(f"M_sem occ: {m_sem_occ * 100:5.2f}%")

    y += 8
    put("=== action (norm) ===", (180, 220, 255))
    put("dim    gt   pred   |Δ|")
    for d in range(8):
        delta = abs(float(a_pred[d]) - float(a_gt[d]))
        # color: green if |Δ|<0.5, yellow <1.0, red >=1.0
        col = (0, 255, 0) if delta < 0.5 else (0, 255, 255) if delta < 1.0 else (0, 0, 255)
        put(f" {d}  {a_gt[d]:+5.2f} {a_pred[d]:+5.2f}  {delta:.2f}", col)

    return canvas


def main():
    args = parse_args()

    print(f"[viz] loading checkpoint: {args.checkpoint}")
    device = args.device if torch.cuda.is_available() else "cpu"
    model = _load_model(args.checkpoint, device)

    print(f"[viz] loading episode: {args.episode_id}")
    ep = _load_episode(args.cache_root, args.episode_id, args.T_max)
    T = ep["T"]
    print(f"[viz] episode T={T}, sub_tasks={set(ep['sub_task_id'].tolist())}")

    # Forward
    print(f"[viz] running forward through {T} frames...")
    out = _forward_episode(model, ep["rgb_raw"], ep["proprio"], device)

    # Apply action normalization to ground truth (for comparison with normalized a_pred)
    # The a_pred from model is in z-score space; a_gt from cache is RAW.
    # Get action stats from /data/.../meta/stats.json
    import json as _json
    stats_p = Path("/home/sqmluser/data/memory_vla/libero_long/meta/stats.json")
    if stats_p.exists():
        s = _json.loads(stats_p.read_text())
        a_mean = np.asarray(s["action"]["mean"] + [0.0])  # pad to 8
        a_std = np.asarray(s["action"]["std"] + [1.0])
        a_gt_norm = (ep["action"].numpy() - a_mean) / (a_std + 1e-6)
    else:
        a_gt_norm = ep["action"].numpy()

    # Render
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[viz] writing {args.output} @ {args.fps} fps...")
    writer = imageio.get_writer(
        str(args.output),
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    for t in range(T):
        frame = _draw_frame(
            rgb=ep["rgb_raw"][t].numpy(),
            t=t, T=T,
            gamma_geo=float(out["gamma_geo"][t]),
            gamma_sem=float(out["gamma_sem"][t]),
            a_pred=out["a_pred"][t],
            a_gt=a_gt_norm[t],
            sub_task_id=int(ep["sub_task_id"][t]),
            m_geo_occ=float(out["m_geo_occ"][t]),
            m_sem_occ=float(out["m_sem_occ"][t]),
        )
        writer.append_data(frame)
    writer.close()

    # Summary
    mse = ((out["a_pred"] - a_gt_norm) ** 2).mean(axis=-1)
    print(f"[viz] action MSE per frame: mean={mse.mean():.3f}  median={np.median(mse):.3f}")
    print(f"[viz] gamma_geo  range [{out['gamma_geo'].min():.2f}, {out['gamma_geo'].max():.2f}]  mean={out['gamma_geo'].mean():.2f}")
    print(f"[viz] gamma_sem  range [{out['gamma_sem'].min():.2f}, {out['gamma_sem'].max():.2f}]  mean={out['gamma_sem'].mean():.2f}")
    print(f"[viz] M_geo occ  end={out['m_geo_occ'][-1] * 100:.2f}%")
    print(f"[viz] M_sem occ  end={out['m_sem_occ'][-1] * 100:.2f}%")
    print(f"[viz] DONE → {args.output} ({T} frames, {T / args.fps:.1f}s playback)")


if __name__ == "__main__":
    main()
