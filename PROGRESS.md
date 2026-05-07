# CHIME-VLA Progress Log

> 自动更新文件。最近 5 操作 + 当前 milestone + blockers + next action。

## Current state

- **Milestone**: M1 — **E1 milestone gate 已命中,等待用户决策**
- **Active branch**: `m1/forward-impl`(已推 GitHub)
- **Updated**: 2026-05-08
- **Owner**: 待用户决策(milestone gate)
- **Mode**: M1 forward + smoke 完成 → E1 pipeline 完成 → baseline (untrained) 与 random 等同

## Recent operations (last 8)

| ts | agent | action | result | commit |
|---|---|---|---|---|
| 2026-05-07 | implementer × 3 | M1 [C1][C2][C5] forward 实现 | green; full chain smoke OK | c983d78 |
| 2026-05-07 | data-pipeline (bg) | LIBERO 全 379 ep cache build | green: 15.28 GB / 101k 帧 / 69s | (in main) |
| 2026-05-07 | implementer × 2 | M1 [C3][C4] 写头 + [C6][C7] memory + lru | green; M_geo 稀疏 1.5%; M_sem.v 累加 | e1b6c62 |
| 2026-05-07 | implementer × 2 | M1 [C8] read + [C9] action | green; full 9-comp chain ✓ | 07ccf14 |
| 2026-05-08 | implementer | M1 train_step + losses + datamodule + lightning + smoke | green; 5 batch × 7s; train_loss=8744 finite | 5fdd3f4 |
| 2026-05-08 | experiment-runner | E1 pipeline 实现 + 5 ep untrained baseline | **MILESTONE GATE**: IoU=0.173 ≈ random(0.169) | d27ad01 |

## Blockers

- **🚨 M1 E1 milestone gate 命中**(architecture 存亡判决,自主停下)
  - untrained CHIME 在 5 LIBERO ep 上 IoU @ 0.3 = 0.173,与 random baseline (0.169) 等同——expected behavior for untrained model
  - 真 E1 判决需要先训 CHIME 至少 100-500 step (L_main only,λ_1=0)
  - **不能自主推进**——这是架构最大赌注的判决点(架构 §C 第 2 件赌注,工程直觉成功率 30-50%)

## Next action

**等用户决策**——4 个选项:

| 选项 | 描述 | 时间预算 | 风险 |
|---|---|---|---|
| **(A) 训 CHIME 100-500 step + rerun E1** | 用 m0_smoke / m1_smoke config,L_main only,~5-15 min on 2×4090 | 短 | 训不充分时 IoU 仍可能模糊 |
| **(B) 适配 Hindsight 到 LIBERO + Phase B 训练** | 跑 Hindsight 的完整 RMBench-pipeline 在 LIBERO 上,Phase B = 训 6-layer proxy ~30M params | ~1-2 hr | 工作量大但有成熟代码 |
| **(C) 走 §0.7.4 MVP fallback** | 接受 E1 baseline,永久锁 λ_1=0,砍 [C10][C12][C13] | 跳过 | publishable claim 退到 MVP 级别 |
| **(D) 其他** | 用户自定义路径 | - | - |

**推荐 (A)** —— 时间预算最低,且 baseline 几近 random 强烈暗示 E1 判决需要训练做对照;A 失败再考虑 B 或 C。

## Milestone gate status

### Setup 阶段
- [x] Stage 0/1/2/3/4 全部 done

### Implementation 阶段
- [x] **M0** — Repo skeleton + Hindsight 契约
  - [x] cache 全 379 ep / 15.28 GB
  - [x] 9 个 forward 组件全 import OK + smoke OK
- [x] **M1 forward implementation**
  - [x] [C1] VLMBackbone (SigLIP2 + LoRA, 2.4M trainable)
  - [x] [C2] WorkBuffer (ring, 0 params)
  - [x] [C5] ESPC (GRU + EMA + dual proj, 8.1M trainable)
  - [x] [C3] GeoWriteHead (token_to_voxel + delta-rule scatter)
  - [x] [C4] SemWriteHead (slot_free + softmax penalty + delta-rule)
  - [x] [C6] GeoGrid (multi-res voxel + occupancy_pct + reset)
  - [x] [C7] SemBank (frozen keys + slot_free + LRU)
  - [x] [C8] ReadInterface (cross-attn + 三线性 + slot_free mask + entropy)
  - [x] [C9] ActionExpert (1-step distill + LoRA, 49K trainable)
  - [x] train_step + losses (L_main + L_aux + L_HCS sentinel)
  - [x] datamodule + LightningModule + scripts/10_train.py
  - [x] smoke training: 5 batch × 7s × 22 GB GPU OK
- [x] **M1 E1 pipeline**
  - [x] compute_jacobian_saliency + compute_iou_vs_boundaries
  - [x] random baseline 控制对照
  - [x] scripts/40_run_e1_judgment.py CLI
  - [x] 5 ep untrained baseline run: IoU=0.173 ≈ random
- [ ] **M1 E1 真判决**(等用户决策选项 A/B/C/D)
- [ ] M2 — [C5] + [C11] 独立收敛
- [ ] M3 — 联立 [C3][C6]+[C4][C7](硬件切换 6×A800)
- [ ] M4 — 5-loss + LIBERO SR baseline
- [ ] M6 — Ablation 套件

## Restart protocol

新会话第一条指令:

```
读以下文件接续 CHIME-VLA 项目:
1. /home/sqmluser/workspace/theaj/CHIME-VLA/PROGRESS.md
2. /home/sqmluser/workspace/theaj/CHIME-VLA/PLAN.md
3. /home/sqmluser/workspace/theaj/CHIME-VLA/IMPLEMENTATION_PLAN.md
4. /home/sqmluser/workspace/theaj/CHIME-VLA/output/reports/e1_baseline_untrained.json
5. git log --oneline -20
6. cd /home/sqmluser/workspace/theaj/CHIME-VLA && pytest -x --tb=line 2>&1 | tail -3

milestone gate 状态:M1 E1 pipeline 已建,等用户决策 (A/B/C/D)。
```

## GitHub

- 仓库: https://github.com/Allenhetl/CHIME-vla.git
- 已推分支: main, stage1/consistency-patches, stage2/implementation-plan, stage3/code-skeleton, m1/forward-impl
- 最新 commit: `d27ad01` (m1/forward-impl) — E1 pipeline + baseline
