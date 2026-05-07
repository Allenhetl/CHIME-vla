# CHIME-VLA — 项目计划(精简执行视图)

> 这是一份精简、便于扫读的"项目仪表盘"。详细 milestone 计划见 `IMPLEMENTATION_PLAN.md`,详细架构见 `chat/architecture_v2.1.md`,详细代码结构见 `CODE_STRUCTURE.md`,实时状态见 `PROGRESS.md`。

## 一句话定位

CHIME-VLA = 长程记忆 VLA 架构,通过**事件触发的双通道不可变 memory**(几何 grid + 语义 slot bank)+ **delta-rule 写入** + **三路读出**(working / geo / sem)+ **后视因果显著性监督**,在 LIBERO-Long 上把 SR 推过 OpenVLA + 8-frame history baseline ≥ 10%。

## 当前里程碑

- **现在**: Stage 2(写完整实施方案)→ Stage 3(代码骨架)
- **下一个 hard gate**: M1 E1 判决(IoU @ 0.3 ≥ 0.4 ?)— 架构存亡门
- **最终验收**: M6 ablation 表通过 + 至少 3 项支持核心赌注

## Stage 路径(setup 阶段,~2 周)

| Stage | 产出 | 状态 |
|---|---|---|
| Stage 0 | `git init` + chat/ 文档基线 | ✅ done |
| Stage 1 | `chat/architecture_v2.1.md`(D1-D5 一致性补丁 + §I.6 + grad-flow contract) | ✅ done |
| **Stage 2** | `IMPLEMENTATION_PLAN.md` + `PLAN.md` + `PROGRESS.md` + `CODE_STANDARDS.md` + `docs/hindsight_contract.md` | 🟡 in progress |
| Stage 3 | `pyproject.toml` + `src/chime_vla/` 全骨架 + `configs/` Hydra 树 + `tests/` 占位 + `CODE_STRUCTURE.md` | ⏳ pending |
| Stage 4 | orchestrator agent prompt + `/loop` 启动协议 | ⏳ pending |

## Milestone 路径(实施阶段,~22 周)

| ID | 周次 | 硬件 | 一句话目标 | gate |
|---|---|---|---|---|
| M0 | w1-2 | 2×4090 | repo skeleton + Hindsight 契约就绪 | pytest + import OK |
| **M1** | w3-5 | 2×4090 | **E1 判决**:γ̂ vs sub_task_id IoU @ 0.3 ≥ 0.4 | **架构存亡** |
| M2 | w6-9 | 2×4090 | [C5] + [C11] 独立收敛(IoU>0.5,L_PRH 下降) | grad-flow CI 全绿 |
| M3 | w10-12 | 6×A800 | 联立 L_main + L_PRH,写头梯度通畅,M_geo 稀疏占用 1-10% | M_geo invariant |
| M4 | w13-17 | 6×A800 | 5-loss 训完,LIBERO SR Δ ≥ 10% baseline | SR target |
| M5 | — | — | ~~跨数据集~~ DROPPED | — |
| M6 | w18-22 | 6×A800 | 10 项 ablation 完整,≥3 项支持核心赌注 | paper claim |

## 三件事赌注(架构 §C)

1. **§3.2 delta-rule + LRU 丢弃绕过门控更新**(强度 [中]) — 错则 fallback 可学 g
2. **HCS Jacobian 信噪比够**(强度 [弱],**E1 核心赌注**) — 错则 disable [C10][C12][C13],MVP fallback
3. **按精度分通道(geo + sem)优于按位置分**(强度 [中]) — 错则合并单 bank,损 ~10-15% 长程 SR

## 运行模式

### Setup 阶段(Stage 0-3,本周内)
- 主 agent(本会话)直接顺序执行
- 用户可中途打断,新会话从 PROGRESS.md "Next action" 接续

### Implementation 阶段(M0-M6,~22 周)
- 启动方式:用户输入 `/loop 30m` (或 `/loop 15m` 重码期)
- 主 agent = orchestrator,每 tick:
  1. 读 PROGRESS.md / git log -10 / pytest 状态
  2. 决策:有 blocker 停下;否则派子 agent
  3. 子 agent 类型:implementer / tester / data-pipeline / experiment-runner / progress-reporter
  4. 子 agent 完成 → progress-reporter 写 PROGRESS.md + git commit
  5. ScheduleWakeup(900s impl / 1800s 等实验)
- **自动停下**触发(必须等用户 ack):
  - M1 E1 IoU 报告产出
  - 任何 milestone 退出条件命中
  - test_grad_flow 持续红 > 2 ticks
  - 任何 red-flag(#1-#7)触发
  - 4 ticks 无进度

### 中断 / 重启
- 用户随时可以 Ctrl-C 或退出会话
- 新会话开始第一条指令(用户输入或自动 resume):
  ```
  按以下顺序读:
  1. /home/sqmluser/workspace/theaj/CHIME-VLA/PROGRESS.md
  2. /home/sqmluser/workspace/theaj/CHIME-VLA/PLAN.md
  3. /home/sqmluser/workspace/theaj/CHIME-VLA/IMPLEMENTATION_PLAN.md
  4. /home/sqmluser/workspace/theaj/CHIME-VLA/chat/architecture_v2.1.md
  5. git log --oneline -20
  6. pytest -x tests/test_grad_flow.py
  然后从 PROGRESS.md "Next action" 接续。
  ```

## 关键文件索引

### 架构 / 设计
- `chat/architecture_v2.1.md` — 主架构 spec(2158 行,v2.1 已合并 5 处一致性补丁)
- `chat/chime_vla_proposal.md` — 叙事 proposal(canonical sg matrix §5.4 来源)
- `chat/architecture_v2_FINAL.md` — v2.0 历史版本(归档,只读)

### 工程契约
- `IMPLEMENTATION_PLAN.md` — 详细 M0-M6 milestone(进入条件 / deliverable / kill / agent)
- `CODE_STANDARDS.md` — 代码规范(继承 Hindsight + CHIME 增量)
- `CODE_STRUCTURE.md` — 代码框架文档(待 Stage 3 产出)
- `docs/grad_flow_contract.md` — sg CI gate 规约
- `docs/hindsight_contract.md` — Hindsight ↔ CHIME-VLA 文件协议

### 实时状态
- `PROGRESS.md` — last 5 操作 + blockers + next action
- `git log` — commit 历史

### 数据
- `/home/sqmluser/data/memory_vla/libero_long/traj_*.h5` — LIBERO-Long 原始数据(400+ ep)
- `output/cache/libero_long/{task}/ep_*.pt` — CHIME 预处理 cache(待 M0 产出)
- `Hindsight/output/saliency/gamma_hat/per_task_q75/libero_long/ep_*.pt` — γ̂ 标签(M1 产出)

## 项目元数据

- **Owner**: Allenhetl <orcella_loecker683@mail.com>
- **Started**: 2026-05-07
- **Target deliverable**: workshop paper(M6 完成后)
- **Hardware budget**: 2×RTX 4090 D 48GB(now)+ 6×A800 80GB(M3+,待接入)
- **Data budget**: LIBERO-Long only(M5 跨数据集已砍)
