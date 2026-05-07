# CHIME-VLA Autonomous Execution Protocol

> 本文件是 /loop 自主执行的**完整协议**:orchestrator agent 的 prompt 模板、5 类子 agent 规约、PROGRESS.md schema、git/commit 规约、milestone gate 自动停下条件、新会话重启接续协议。

## 1. 启动方式

### 用户启动 /loop
当所有 Stage 0-3 deliverable 已 merge 到 main(或在 stage-N 分支上)且 PROGRESS.md 标 M0 为 next milestone 时,用户输入:

```
/loop 30m
```

(频率参考表见 §6;一般首次启动用 30m,M0 重码期切 15m,M4 长训练期切 60m。)

### Orchestrator 接管
主 agent(无 /loop 时是用户对话的当前 agent)即扮演 orchestrator。每次 /loop tick 触发一轮 §3 的 checklist。

---

## 2. Orchestrator Prompt 模板

下文是**每次 /loop tick 触发时**主 agent 应当 self-execute 的 checklist。可以直接 copy 粘贴到对话中,也可以由 /loop 自动注入。

```text
=== Orchestrator /loop tick ===

You are the CHIME-VLA orchestrator. Execute this checklist for one /loop tick:

**Phase A: READ STATE**
1. Read /home/sqmluser/workspace/theaj/CHIME-VLA/PROGRESS.md
2. Read git log --oneline -10
3. Run: cd /home/sqmluser/workspace/theaj/CHIME-VLA && pytest -x --tb=line 2>&1 | tail -3
4. (If running training): run nvidia-smi to check GPU usage
5. Note: current milestone, current branch, last 5 ops, blockers, next action

**Phase B: DECIDE**
Pick the SINGLE highest-priority action:
1. **BLOCKER** — if any blocker in PROGRESS.md is non-null OR pytest is red:
   → If grad-flow test red: dispatch tester subagent to diagnose, BLOCK all other dispatches
   → If other blocker: write summary to PROGRESS.md, STOP /loop, notify user
2. **MILESTONE GATE** — if milestone exit conditions met (per IMPLEMENTATION_PLAN §M? exit criteria):
   → Mark gate hit in PROGRESS.md, STOP /loop, notify user for merge approval
3. **RED FLAG** — if any of #1-#7 (per architecture v2.1 §I.4) tripped:
   → STOP /loop, notify user (architectural decision needed)
4. **STALL** — if last 4 ticks made no progress (no new commit on stage/milestone branch):
   → STOP /loop, notify user (possible loop stall)
5. **NORMAL** — pick "Next action" from PROGRESS.md, dispatch corresponding subagent

**Phase C: DISPATCH (only if NORMAL)**
1. Use TaskCreate or Agent to dispatch subagent (see §4 for sub-agent specs)
2. Pass concise task description (1-3 sentences) + reference docs (architecture / IMPLEMENTATION_PLAN / CODE_STRUCTURE)
3. Wait for completion (TaskGet / agent result)
4. On agent green completion:
   - dispatch progress-reporter to update PROGRESS.md (last 5 ops + next action + commits)
   - git commit on the milestone branch with proper prefix (per CODE_STANDARDS §3)

**Phase D: SCHEDULE NEXT TICK**
1. If actively coding: ScheduleWakeup(900s, "next sub-task in M{N}")
2. If polling long-running training: ScheduleWakeup(1800s, "polling M{N} train run")
3. If milestone gate hit / blocker / red flag: do NOT schedule, just notify user
4. Cap: never below 60s, never above 3600s.

**Output to user (each tick):**
- 1-3 sentence summary of what happened this tick (what got dispatched, results, next ETA)
- Cap at 5 sentences total

=== End orchestrator tick ===
```

---

## 3. 子 Agent 类型规约

orchestrator 根据 PROGRESS.md "Next action" 派以下 5 类子 agent 之一(或多个并发)。每类有固定的工具集 + prompt 风格。

### 3.1 implementer

**用途**:实现组件 forward / 接 loss / 添加新模块 / 修 bug。

**工具**:`Read, Edit, Write, Bash(test, type-check), Agent`(若需要再嵌套)

**Prompt 模板**:
```
你是 implementer 子 agent。任务:{具体子任务,1 句}。

参考(必读):
- chat/architecture_v2.1.md §{相关 section}
- CODE_STRUCTURE.md §{相关接口签名}
- CODE_STANDARDS.md §{相关规约}

具体步骤:
1. Read 当前 stub: src/chime_vla/{path}.py
2. 实现 forward(/...)按 CODE_STRUCTURE.md §3 签名
3. 跑 pytest -x tests/{相关测试}.py 验证
4. 跑 tests/test_grad_flow.py 确保 SG-N 没破坏
5. 报告:实现的函数列表 + 测试结果 + 任何与 spec 偏离

约束:
- dtype 严格按 CODE_STANDARDS §1.7 矩阵
- shape 严格按 CODE_STRUCTURE.md §3 签名
- forward 内部不要新增 sg/不-sg 决策(已锁定见 SG matrix)
- ≤500 词报告
```

### 3.2 tester

**用途**:写 / 修 / 诊断测试,grad-flow CI gate 守门。

**工具**:`Read, Bash(pytest), Edit (仅 tests/ 目录)`

**Prompt 模板**:
```
你是 tester 子 agent。任务:{诊断 / 写测试 / 修 CI gate}。

参考:
- docs/grad_flow_contract.md (SG-1..SG-7 期望路径)
- tests/conftest.py (现有 fixture)
- CODE_STANDARDS.md §4 (测试矩阵 strict/xfail 规则)

约束:
- 仅修改 tests/ 目录下文件 + 必要时 conftest.py
- 不能修 src/ 来"让测试通过"——那是 implementer 的事
- 找到 root cause 而非掩盖(例:不要把失败的 strict test 改成 xfail)
- 报告:测试结果 + 诊断结论 + 是 src bug 还是 test bug
```

### 3.3 data-pipeline

**用途**:LIBERO cache 构建 / Hindsight 产物刷新 / 数据 schema 验证。

**工具**:`Read, Bash, Write (仅 output/ + scripts/)`

**Prompt 模板**:
```
你是 data-pipeline 子 agent。任务:{cache 构建 / 刷新 Hindsight / 验证 schema}。

参考:
- docs/hindsight_contract.md (γ̂ 文件协议)
- docs/data_schema.md (LIBERO h5 → cache .pt schema)
- scripts/00_build_libero_cache.py / 01_run_hindsight_consumer.py

约束:
- 仅写到 output/ 目录;不污染 src/ 或 chat/
- cache 产物大小先 dry-run 估算,超过 50GB 报告 user
- 报告:产物路径 + 大小 + episode 数 + schema 自检结果
```

### 3.4 experiment-runner

**用途**:跑训练 / 评估 / ablation 批量 job。

**工具**:`Read, Bash(scripts/*.py), Write (仅 output/runs/, logs/)`

**Prompt 模板**:
```
你是 experiment-runner 子 agent。任务:{跑训练 M{N} / 跑 baseline / 跑 ablation #{i}}。

参考:
- IMPLEMENTATION_PLAN.md §{对应 milestone}
- configs/train/{相关 yaml}
- scripts/10_train.py 或 30_run_ablation.py

具体步骤:
1. 用 Hydra override 启动 (e.g. python scripts/10_train.py +train=m4_full_5loss seed=42)
2. 监控 nvidia-smi 与 wandb / tensorboard
3. 训练完成或中断 → 报告 metrics
4. 把 metrics.json + checkpoints/ 路径写入 PROGRESS.md "Recent operations"

约束:
- 长训练用 run_in_background=true
- 监控指标按 CODE_STANDARDS §5 列出的 9 个必 log 指标
- 训练发散 / NaN → STOP, 报告 user
```

### 3.5 progress-reporter

**用途**:每个 tick / 每个子任务完成后更新 PROGRESS.md。

**工具**:`Read, Edit (仅 PROGRESS.md)`

**Prompt 模板**:
```
你是 progress-reporter 子 agent。任务:更新 PROGRESS.md 反映最新状态。

输入:
- 最近一次操作描述(由 orchestrator 提供)
- 任何新的 blocker
- 下一步建议

具体:
1. Read 当前 PROGRESS.md
2. Edit "Recent operations" 表格:加新行,保留最近 5 行
3. Edit "Blockers" 段:加新 blocker 或清理已解
4. Edit "Next action" 段:基于当前进度更新
5. Edit "Milestone gate status":勾选已完成项

约束:
- 只编辑 PROGRESS.md, 不动其他文件
- 保留原有结构, 不重写
- 输出 ≤100 词:总结改动
```

---

## 4. PROGRESS.md Schema 详解

详见 `PROGRESS.md` 当前文件,关键字段:

```markdown
## Current state
- **Milestone**: M{N}
- **Active branch**: stage{N}/{name} or m{N}/{name}
- **Updated**: ISO 8601 ts
- **Owner**: orchestrator / specific agent name

## Recent operations (last 5)
| ts | agent | action | result | commit |

## Blockers
- "(none)" 或具体描述

## Next action
- **agent**: <type>
- **task**: <1-2 sentences>
- **ETA**: <X tick / X hour / "blocked">

## Milestone gate status
- [x] / [ ] checklist
```

orchestrator 与 progress-reporter 共同维护;**禁止**把"具体计划"写到 PROGRESS.md(那是 IMPLEMENTATION_PLAN.md 的事)。

---

## 5. Git 规约

### branch 策略
- **stage 分支**(setup):`stage1/consistency-patches`, `stage2/implementation-plan`, `stage3/code-skeleton`, `stage4/orchestrator-protocol`
- **milestone 分支**(impl):`m0/skeleton`(可与 stage3 复用), `m1/e1-judgment`, `m2/espc-prh`, `m3/joint-train`, `m4/full-5loss`, `m6/ablation-{1..10}`
- **main**:仅在 milestone gate PASS + user approve 后 merge

### commit message 前缀
- `feat(C{N}):` 新组件实现
- `fix(SG-{N}):` grad-flow 修复
- `fix(slot):` slot lifecycle 修复
- `data:` 数据 pipeline / cache
- `exp(M{N}):` 实验运行 / 调参
- `progress:` PROGRESS.md update only
- `docs:` 文档(包括架构修订)
- `chore:` 配置 / CI / 工具

### 禁止
- `--force` / `--no-verify` / `--no-gpg-sign`(除 user 明确允许)
- 直接 push 到 main
- 在未 ack 的 milestone gate 后 merge

---

## 6. /loop 频率与自适应

| 阶段 | 推荐频率 | 理由 |
|---|---|---|
| Setup (Stage 0-4) | 不需 /loop,主 agent 顺序跑 | 文档 + skeleton, 短任务 |
| M0 / M3 (重码期) | `/loop 15m` | 频繁 commit, 短迭代 |
| M1 / M2 (smoke + 独立验证) | `/loop 30m` | 中等节奏 |
| M4 (full train) | `/loop 60m` | 训练 hr-级别, 频繁 poll 浪费 |
| M6 (ablation 批量) | `/loop 60m` 或 `/loop dynamic` | 单 ablation hr-级别 |
| 等长 experiment | `ScheduleWakeup(1800s)` | 自适应,不打扰 |

**自适应规则**(orchestrator 决策):
- 实验已 launched + 等待中 → `ScheduleWakeup(1800s)`
- 实验完成 + 准备分析 → `ScheduleWakeup(900s)`
- BLOCKER / 红 / gate → 不 schedule,等 user

---

## 7. 自动停下(milestone gate)条件

orchestrator **必须停下 + 通知 user** 的情况(per IMPLEMENTATION_PLAN.md §10):

| 触发 | 原因 |
|---|---|
| M1 E1 IoU 报告产出 | 架构存亡判决,不能自主 |
| 任何 milestone 退出条件命中 | merge 需 user approve |
| `test_grad_flow` 持续红 > 2 tick | 架构违规,需深度审 |
| Red flag #1-#7 触发 | 架构决策需人 |
| 4 tick 无新 commit | 可能 loop stall,需诊断 |
| 训练发散 / NaN | 可能数据 / 超参问题 |
| 实验预算超 budget | 续跑需 user 确认 |

停下时:
1. PROGRESS.md "Blockers" 段写明触发条件 + 最近操作上下文
2. 不 ScheduleWakeup
3. 输出明确通知给 user(`@user STOP — {trigger}: {brief}`)

---

## 8. 重启接续协议

**新会话第一条指令**(用户输入或 /resume 自动注入):

```text
读以下文件接续 CHIME-VLA 项目:
1. /home/sqmluser/workspace/theaj/CHIME-VLA/PROGRESS.md
2. /home/sqmluser/workspace/theaj/CHIME-VLA/PLAN.md
3. /home/sqmluser/workspace/theaj/CHIME-VLA/IMPLEMENTATION_PLAN.md (skim 相关 milestone)
4. /home/sqmluser/workspace/theaj/CHIME-VLA/chat/architecture_v2.1.md (按需)
5. git log --oneline -20
6. cd /home/sqmluser/workspace/theaj/CHIME-VLA && pytest -x --tb=line 2>&1 | tail -3

然后:
- 如 "Next action" 是单个明确 task: 直接 dispatch 子 agent 执行
- 如 "Blockers" 非空: 先解 blocker
- 如 milestone gate 命中且未 ack: 询问 user 是否合并
- 否则: 启动 /loop 30m 回到自主模式
```

把这条粘到任何新会话开始,即可无缝接续。

---

## 9. 监控指标(orchestrator 每 tick 必看)

| 指标 | 来源 | 阈值 / 判定 |
|---|---|---|
| pytest 状态 | `pytest -x tests/ --tb=line` | 任何 fail/error → blocker |
| grad-flow 状态 | `pytest tests/test_grad_flow.py` | M2+ 任何红 → 全停 |
| GPU 使用 | `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv` | 训练时 > 30 GB OK |
| 当前 commit | `git log -1 --format=%h` | 与 PROGRESS 对齐 |
| disk 使用 | `du -sh output/` | > 100 GB 报告 user |
| 距上次 commit | `git log -1 --format=%ar` | > 4 tick 无 commit → stall |

---

## 10. 例子:第一次 /loop tick

假设当前是 Stage 4 已 merge,M0 待开始:

**PROGRESS.md "Next action"**:
> agent: implementer
> task: 实现 [C5] ESPC ψ + EMA 标准化 forward,使 forward smoke 在 batch=2 T=64 跑通

**orchestrator 行为**:
1. Read PROGRESS.md → 看到 next action
2. git log → 看到最后一次 commit 是 `feat: Stage 3 ...`
3. pytest → 12 passed + 10 xfailed + 2 xpassed,无 fail
4. nvidia-smi → 2×4090 idle
5. **DECIDE: NORMAL** → dispatch implementer
6. **DISPATCH**: implementer prompt 含具体子任务 + 引用 §C5 卡片 + 引用 SG-1/SG-5/SG-6
7. 等 implementer 完成(15-30 min)
8. implementer 报告:`feat(C5): 实现 ψ GRU forward + EMA running stats`
9. dispatch progress-reporter 更新 PROGRESS.md
10. git commit
11. **SCHEDULE**: ScheduleWakeup(900s, "next: [C5] grad-flow test verification")
