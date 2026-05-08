#!/usr/bin/env python
"""Closed-loop rollout: model controls LIBERO robot in MuJoCo simulator.

For each step:
  1. env returns obs (rgb, proprio, ...)
  2. model.forward(rgb, proprio) → predicted normalized action (8 dim)
  3. de-normalize → take first 7 dim → step env
  4. render

Saves mp4 of the rolled-out simulation.

Usage:
    python scripts/51_rollout_closed_loop.py \
        --checkpoint output/runs/m4_long_600step/last.ckpt \
        --task-id 0 \
        --max-steps 200 \
        --output output/viz/m4_closedloop_task0.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Set EGL BEFORE any robosuite/mujoco imports
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import cv2  # noqa: E402

from chime_vla.config import ChimeConfig  # noqa: E402
from chime_vla.memory.geo_grid import GeoGrid  # noqa: E402
from chime_vla.memory.sem_bank import SemBank  # noqa: E402
from chime_vla.perception.fifo_buffer import WorkBuffer  # noqa: E402
from chime_vla.training.lightning_module import ChimeVlaLightning  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--benchmark", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("output/viz/closedloop.mp4"))
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--gripper-binary",
        action="store_true",
        help="Threshold gripper (last action dim) at 0 (LIBERO convention)",
    )
    return p.parse_args()


def _load_model(ckpt_path: Path, device: str) -> ChimeVlaLightning:
    cfg = ChimeConfig()
    cfg.loss.lambda_2 = 0.5
    cfg.loss.lambda_predict = 1.0
    cfg.loss.lambda_ent = 0.01
    cfg.c10.enabled = False
    cfg.hindsight.enabled = False
    model = ChimeVlaLightning.load_from_checkpoint(
        ckpt_path, cfg=cfg, strict=False, map_location=device
    )
    model.eval()
    return model


def _load_action_norm():
    p = Path("/home/sqmluser/data/memory_vla/libero_long/meta/stats.json")
    if not p.exists():
        return None, None
    s = json.loads(p.read_text())
    a_mean = np.asarray(s["action"]["mean"] + [0.0], dtype=np.float32)  # pad to 8
    a_std = np.asarray(s["action"]["std"] + [1.0], dtype=np.float32)
    return a_mean, a_std


def _proprio_from_obs(obs: dict) -> np.ndarray:
    """LIBERO obs has robot0_eef_pos / quat + gripper_qpos. Build 8-dim proprio
    matching cache schema (proprio.shape = (8,) — eef pos 3 + eef quat 4 + gripper 1)."""
    eef_pos = obs.get("robot0_eef_pos", np.zeros(3, dtype=np.float32))
    eef_quat = obs.get("robot0_eef_quat", np.zeros(4, dtype=np.float32))
    gripper = obs.get("robot0_gripper_qpos", np.zeros(1, dtype=np.float32))
    if gripper.size > 1:
        gripper = gripper[:1]
    p = np.concatenate(
        [
            np.asarray(eef_pos, dtype=np.float32).reshape(-1)[:3],
            np.asarray(eef_quat, dtype=np.float32).reshape(-1)[:4],
            np.asarray(gripper, dtype=np.float32).reshape(-1)[:1],
        ]
    )
    if p.size < 8:
        p = np.concatenate([p, np.zeros(8 - p.size, dtype=np.float32)])
    return p[:8]


def _draw_overlay(rgb: np.ndarray, t: int, max_t: int, action_norm: np.ndarray,
                  action_raw: np.ndarray, gamma_geo: float, gamma_sem: float,
                  reward: float, done: bool, success: bool) -> np.ndarray:
    rgb_big = cv2.resize(rgb, (448, 448), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((448, 448 + 280, 3), 30, dtype=np.uint8)
    canvas[:, :448] = rgb_big

    # status overlay
    color = (0, 255, 0) if success else (0, 200, 255) if done else (255, 255, 255)
    status = "SUCCESS" if success else ("DONE" if done else "running")
    cv2.putText(
        canvas, f"t={t:3d}/{max_t} {status}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
    )

    panel_x = 460
    y = 30
    line_h = 22

    def put(text, c=(220, 220, 220)):
        nonlocal y
        cv2.putText(canvas, text, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        y += line_h

    put("=== gamma (CHIME write gates) ===", (180, 220, 255))
    put(f"gamma_geo: {gamma_geo:.3f}", (0, int(255 * gamma_geo), 0))
    put(f"gamma_sem: {gamma_sem:.3f}", (int(255 * gamma_sem), 100, 0))

    y += 8
    put("=== action (raw, sent to env) ===", (180, 220, 255))
    labels = ["dx", "dy", "dz", "dRx", "dRy", "dRz", "grip"]
    for d, lbl in enumerate(labels):
        put(f" {lbl:4s} {action_raw[d]:+.3f}")

    y += 8
    put("=== env feedback ===", (180, 220, 255))
    put(f"reward: {reward:.3f}")
    put(f"done:   {done}")

    return canvas


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[rollout] loading checkpoint: {args.checkpoint}")
    model = _load_model(args.checkpoint, device)

    print(f"[rollout] loading benchmark: {args.benchmark} task {args.task_id}")
    B = benchmark.get_benchmark_dict()[args.benchmark]()
    task = B.get_task(args.task_id)
    print(f"[rollout] task: {task.name}")
    print(f"[rollout] language: {getattr(task, 'language', '?')}")

    env_args = dict(
        bddl_file_name=B.get_task_bddl_file_path(args.task_id),
        camera_heights=224, camera_widths=224,
        camera_names=["agentview"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        render_camera="agentview",
    )
    env = OffScreenRenderEnv(**env_args)

    # Load action normalization (model is in z-score space)
    a_mean, a_std = _load_action_norm()
    print(f"[rollout] action norm loaded: mean[:3]={a_mean[:3] if a_mean is not None else None}")

    # Reset
    env.seed(args.seed)
    obs = env.reset()
    print(f"[rollout] obs keys: {sorted(obs.keys())[:10]}")

    # Per-episode CHIME memory
    cfg = model.cfg
    c2 = WorkBuffer(cfg.c2, batch_size=1, device=device)
    m_geo = GeoGrid(cfg.c6, batch_size=1, d_g=cfg.c6.d_g, device=device)
    m_sem = SemBank(cfg.c7, batch_size=1, device=device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(args.output), fps=args.fps, codec="libx264",
        quality=8, macro_block_size=1,
    )

    success = False
    last_reward = 0.0
    print(f"[rollout] running {args.max_steps} steps...")

    for t in range(args.max_steps):
        rgb = obs.get("agentview_image", None)
        if rgb is None:
            print("[rollout] no rgb, abort"); break
        # robosuite returns flipped image — agentview_image is upside down
        rgb_view = rgb[::-1, :, :].copy()  # flip Y, dtype uint8 (224, 224, 3)

        proprio = _proprio_from_obs(obs)
        rgb_t = torch.from_numpy(rgb_view).float().div_(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(device)

        # CHIME forward
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                h_t = model.c1(rgb_t, proprio_t)
                m_work_prev = c2.snapshot()
                gamma_geo, gamma_sem = model.c5(h_t, m_work_prev)
                c2.append(h_t)
                model.c3(h_t, gamma_geo.detach(), m_geo, step=t)
                model.c4(h_t, gamma_sem.detach(), m_sem, step=t)
                c_t = model.c8(c2.snapshot(), m_geo, m_sem, h_t)
                a_pred_norm = model.c9(c_t, h_t.mean(dim=1))[0].float().cpu().numpy()  # (8,)

        # de-normalize
        if a_mean is not None and a_std is not None:
            a_raw = a_pred_norm * a_std + a_mean
        else:
            a_raw = a_pred_norm

        # LIBERO env expects 7-dim action (xyz delta, rpy delta, gripper)
        a_env = a_raw[:7].astype(np.float32)
        if args.gripper_binary:
            a_env[6] = 1.0 if a_env[6] > 0 else -1.0

        # render frame BEFORE step (to show what model saw)
        frame = _draw_overlay(
            rgb_view, t, args.max_steps,
            action_norm=a_pred_norm, action_raw=a_raw,
            gamma_geo=float(gamma_geo[0].item()),
            gamma_sem=float(gamma_sem[0].item()),
            reward=last_reward, done=False, success=success,
        )
        writer.append_data(frame)

        # step env
        obs, reward, done, info = env.step(a_env)
        last_reward = float(reward)
        if reward > 0.5:  # LIBERO success signal
            success = True
        if done:
            print(f"[rollout] env done at t={t} (reward={reward}, success={success})")
            # write final frame
            rgb = obs.get("agentview_image")
            if rgb is not None:
                frame = _draw_overlay(
                    rgb[::-1, :, :].copy(), t + 1, args.max_steps,
                    action_norm=a_pred_norm, action_raw=a_raw,
                    gamma_geo=float(gamma_geo[0].item()),
                    gamma_sem=float(gamma_sem[0].item()),
                    reward=last_reward, done=done, success=success,
                )
                writer.append_data(frame)
            break

        if t % 25 == 0:
            print(
                f"  t={t:3d} γ_geo={float(gamma_geo[0]):.2f} "
                f"γ_sem={float(gamma_sem[0]):.2f} reward={last_reward:.3f}"
            )

    writer.close()
    env.close()

    print()
    print(f"[rollout] DONE → {args.output}")
    print(f"  steps: {t + 1}")
    print(f"  success: {success}")
    print(f"  final reward: {last_reward:.3f}")


if __name__ == "__main__":
    main()
