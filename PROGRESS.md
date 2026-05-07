# CHIME-VLA Progress Log

> 自动更新文件。最近 5 操作 + 当前 milestone + blockers + next action。
> orchestrator agent 在每 /loop tick 末尾由 progress-reporter 子 agent 维护。

## Current state

- **Milestone**: **M0 完成** → M1 待启动(E1 判决 + smoke)
- **Active branch**: `main`(已含全部 stage1-4 + M0 实施;已推 GitHub https://github.com/Allenhetl/CHIME-vla.git)
- **Updated**: 2026-05-07 22:48
- **Owner**: orchestrator(待用户启动 /loop 进入 M1)
- **Mode**: M0 done → M1 准备

## Recent operations (last 5)

| ts | agent | action | result | commit |
|---|---|---|---|---|
| 2026-05-07 | main | git init + chat/ docs baseline | green | 8f5d08c |
| 2026-05-07 | main | Stage 1 一致性补丁 (D1-D5 + §I.6 + grad_flow_contract) | green | 7793198 |
| 2026-05-07 | main | Stage 2 IMPLEMENTATION_PLAN + PLAN + PROGRESS + CODE_STANDARDS + hindsight_contract | green | (stage2 branch) |
| 2026-05-07 | implementer × 2 (parallel) | Stage 3 code skeleton (13 components + configs + tests) | green: pip install OK, 24 tests 12 pass + 10 xfail + 2 xpass | 06c054e |
| 2026-05-07 | main | Stage 4 autonomous_execution.md + PROGRESS schema + restart protocol | green | d4a9a47 |
| 2026-05-07 | main | git: ff-merge stage3 → main + push to https://github.com/Allenhetl/CHIME-vla.git | green: 4 branches pushed | (push) |
| 2026-05-07 | main | M0: scripts/00_build_libero_cache.py 真实实现 + 5 ep dry-run + 5 ep 实写验证 schema | green: T 范围 214-345, sub_task_id 边界 = E1 ground truth | (pending) |

## Blockers

- (none)

## Next action

- **agent**: orchestrator(由 user `/loop 30m` 启动)
- **task**: M1 启动:
  1. data-pipeline:scripts/00_build_libero_cache.py 跑全 379 ep(预计 ~15 GB,1-2 hr)
  2. data-pipeline:在 `Hindsight/` 仓跑 [C10] 离线 saliency on 前 200 LIBERO ep(用 Hindsight scripts/05_compute_saliency.py + 06_compute_thresholds + 07_label_gamma_hat),产出 `Hindsight/output/saliency/gamma_hat/per_task_q75/libero_long/ep_*.pt`
  3. implementer:实现 [C1][C2][C5] forward(stub → 真实),启用 test_forward_shapes 真跑 + grad-flow 占位转 strict
  4. experiment-runner:smoke training (B=2, T=64, 1 epoch),验证 L_main + L_aux finite + 梯度通畅
  5. experiment-runner:E1 判决 — 用 sub_task_id boundary 算 IoU(γ̂_geo top-25%, boundary ± 4)
- **ETA**: M1 完工 ~3 周(per IMPLEMENTATION_PLAN §3),硬件 2×4090
- **首个 milestone gate**: E1 IoU 报告产出(自动停下,user 决定 PASS/SOFT/FAIL)

## Milestone gate status

### Setup 阶段
- [x] Stage 0 — git init(commit 8f5d08c)
- [x] Stage 1 — architecture v2.1 一致性补丁(commit 7793198)
- [x] Stage 2 — IMPLEMENTATION_PLAN + PLAN + PROGRESS + CODE_STANDARDS + hindsight_contract
- [x] Stage 3 — Code skeleton(13 组件 stub + configs + tests + 22 modules import OK)
- [x] Stage 4 — Autonomous execution protocol(orchestrator + 5 子 agent 规约 + git/loop/restart)

### Implementation 阶段
- [x] **M0** Repo skeleton + Hindsight 契约 — IMPLEMENTATION_PLAN §2
  - [x] pyproject.toml + 13 组件 stub
  - [x] Hydra config 树 + 测试占位
  - [x] CODE_STANDARDS + CODE_STRUCTURE + grad_flow_contract + hindsight_contract
  - [x] `scripts/00_build_libero_cache.py` 真实实现 + 5 ep dry-run/实写验证(schema 对齐 sub_task_id 边界 = E1 ground truth)
  - [x] git push to https://github.com/Allenhetl/CHIME-vla.git (main + 3 stage branches)
- [ ] M1 — E1 判决 + smoke(IMPLEMENTATION_PLAN §3)
- [ ] M2 — [C5] + [C11] 独立收敛
- [ ] M3 — 联立 [C3][C6]+[C4][C7](硬件切换 6×A800)
- [ ] M4 — 5-loss + LIBERO SR baseline
- [ ] M6 — Ablation 套件(M5 已砍)

## How to start /loop (用户操作)

```
# 当前所有 stage 已经在 stage3/code-skeleton 分支(含 stage1+2+3+4 改动)
# 1. 看 git log 确认改动
git log --oneline -10

# 2. (可选) merge 到 main
git checkout main
git merge stage3/code-skeleton

# 3. 启动 /loop
/loop 30m
```

## Restart protocol(中途停止后接续)

新会话第一条指令(详见 `docs/autonomous_execution.md` §8):

```
读以下文件接续 CHIME-VLA 项目:
1. /home/sqmluser/workspace/theaj/CHIME-VLA/PROGRESS.md      (本文件,state)
2. /home/sqmluser/workspace/theaj/CHIME-VLA/PLAN.md          (scope 仪表盘)
3. /home/sqmluser/workspace/theaj/CHIME-VLA/IMPLEMENTATION_PLAN.md  (按需)
4. /home/sqmluser/workspace/theaj/CHIME-VLA/chat/architecture_v2.1.md  (按需)
5. git log --oneline -20
6. cd /home/sqmluser/workspace/theaj/CHIME-VLA && pytest -x --tb=line 2>&1 | tail -3

然后:
- "Next action" 单一明确 → dispatch 子 agent 执行
- "Blockers" 非空 → 先解 blocker
- milestone gate 命中且未 ack → 询问 user
- 否则 → /loop 30m 回到自主模式
```
