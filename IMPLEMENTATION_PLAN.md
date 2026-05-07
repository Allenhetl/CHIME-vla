# CHIME-VLA 完整实施方案

> 本文件是 M0-M6 milestone 的"工程版"展开,与架构文档 `chat/architecture_v2.1.md` 互补。架构定义"是什么";本文件定义"什么时候做、做到什么算完、什么情况停下来"。

## 0. 项目状态快照 (last update: 2026-05-07)

- **Repo**: `/home/sqmluser/workspace/theaj/CHIME-VLA/`,git initialized,main 分支 + stage1/consistency-patches 分支
- **Architecture**: v2.1 已合并 5 处一致性补丁 + §I.6 工程决策锁定
- **硬件路径**:
  - **当前(M0-M2)**: 2× RTX 4090 D 48GB(`/home/sqmluser/workspace/theaj/CHIME-VLA/`)——代码骨架、smoke test、E1 离线、ESPC/PRH 独立验证
  - **后续(M3-M6)**: 6× A800 80GB(待接入)——联立训练、5-loss 全开、ablation
- **数据**: LIBERO-Long 已下载 `/home/sqmluser/data/memory_vla/libero_long/traj_*.h5`(400+ ep);BridgeV2/RoboCasa/CALVIN 全部跳过;M5 跨数据集验证砍掉,M6 ablation 仅 LIBERO-Long
- **Hindsight 仓**: `./Hindsight/` 是 [C10] HCS-H 离线标签生成器(已实现 SigLIP wrapper / DDP / Hydra+dataclass / dtype 矩阵 / 9 阶段 pipeline);CHIME-VLA 通过文件协议消费其产物,详见 `docs/hindsight_contract.md`

## 1. Milestone 总览

| ID | 周次 | 硬件 | 状态 |
|---|---|---|---|
| **M0** Repo skeleton + Hindsight 契约 | w1-2 | 2×4090 | pending |
| **M1** E1 判决 + smoke | w3-5 | 2×4090 | pending |
| **M2** [C5] + [C11] 独立收敛 | w6-9 | 2×4090 | pending |
| **M3** 联立 [C3][C6]+[C4][C7] | w10-12 | 6×A800 | pending |
| **M4** 5-loss + LIBERO SR baseline | w13-17 | 6×A800 | pending |
| **M5** ~~跨数据集~~ | — | — | dropped (LIBERO-only) |
| **M6** Ablation 套件(10 项) | w18-22 | 6×A800 | pending |

总计:**~22 周**(原方案 24 周,M5 砍掉省 2 周)。

---

## 2. M0 — Repo skeleton + Hindsight 契约

### 进入条件
- Stage 1 一致性补丁 merged 到 main
- `chat/architecture_v2.1.md` 通过 self-check(见 Stage 1 验证)

### Deliverable

**代码骨架**(详见 CODE_STRUCTURE.md):
- [ ] `pyproject.toml`(继承 Hindsight 依赖 + CHIME 增量)
- [ ] `src/chime_vla/{perception,memory,heads,readout,action,training,utils,hindsight}/` 全 stub(13 组件,每个 docstring + interface signature + `raise NotImplementedError`)
- [ ] `src/chime_vla/config.py` Hydra structured config(60-80 字段)
- [ ] `configs/{default.yaml, model/, train/, data/, experiment/}` Hydra 树
- [ ] `tests/test_grad_flow.py` 7 个 SG xfail 占位(per `docs/grad_flow_contract.md`)
- [ ] `tests/{test_forward_shapes.py, test_slot_lifecycle.py, test_memory_reset.py, test_loss_finite.py}` 占位

**数据管道**:
- [ ] `scripts/00_build_libero_cache.py`(LIBERO h5 → per-episode `.pt` cache)
- [ ] cache schema 文档化在 `docs/data_schema.md`

**Hindsight 契约**:
- [ ] `docs/hindsight_contract.md` 已写(详见该文件)
- [ ] `src/chime_vla/hindsight/consumer.py` stub(读 Hindsight 产物 `.pt`)

**CI / 工程**:
- [ ] `.github/workflows/ci.yml`(或 pre-commit hook)运行 `pytest -x tests/test_grad_flow.py` 必须绿(M0 阶段允许 xfail)
- [ ] `ruff` line-length=100,继承 Hindsight 配置
- [ ] `mypy` 至少在 `src/chime_vla/heads/` 与 `training/` 严格

**文档**:
- [ ] `PLAN.md`(精简执行视图,本文件的 TL;DR)
- [ ] `PROGRESS.md`(初始 M0 状态)
- [ ] `CODE_STANDARDS.md`(继承 Hindsight + CHIME 增量)
- [ ] `CODE_STRUCTURE.md`(目录树 + 接口签名 + 配置 schema)
- [ ] `README.md`(quick start + how to /loop)

### Kill / red flag
- LIBERO h5 schema mismatch(`obs/agentview_rgb` 不存在或 dtype 不对)→ 数据格式调研,可能要写 LeRobot adapter
- `pip install -e .` 失败 → 依赖冲突,必须修

### 退出条件(M0 完成判据)
- [ ] `pytest -x tests/` 全部 pass(允许 xfail,禁止 error)
- [ ] `python -c "from chime_vla.training.train_step import train_step"` 不报错
- [ ] `python scripts/00_build_libero_cache.py --n 5 --dry-run` 跑通
- [ ] M0 milestone gate review:用户 ack 即合并 stage1+stage2+stage3 到 main

### 所需 agent
- **implementer**(主)— 写所有 stub 模块、配置、测试占位
- **data-pipeline** — 写 `00_build_libero_cache.py` + h5 schema 验证
- **tester** — 7 个 SG xfail 测试 + 形状契约测试
- **progress-reporter** — 完成后更新 PROGRESS.md

并发度:3 agent 同时跑(implementer / data-pipeline / tester)。

---

## 3. M1 — E1 判决点 + smoke

### 进入条件
- M0 全部 deliverable green
- LIBERO-Long cache 构建完毕(`output/cache/libero_long/{task}/ep_NNNN.pt`)
- Hindsight 仓在 `/home/sqmluser/workspace/theaj/CHIME-VLA/Hindsight/` 可访问

### Deliverable

**Forward smoke**(2×4090,batch=2,T=64):
- [ ] [C1] SigLIP backbone forward(继承 Hindsight wrapper)→ h_t shape 验证
- [ ] [C2] FIFO append + ring buffer 行为单测
- [ ] [C5] ESPC ψ + EMA 标准化 forward(MVP 用 GRU,1-layer)
- [ ] [C3] geo write head + delta-rule 加入 M_geo
- [ ] [C4] sem write head + slot_free mask + delta-rule
- [ ] [C6][C7] memory tensor 容器 + episode reset
- [ ] [C8] read interface(N_q + N_geo_q + 三线性采样)
- [ ] [C9] action expert(π0 flow matching head,1-step distill MVP)
- [ ] L_main + L_aux 有限非 NaN(无 L_HCS / L_PRH / L_CSM)
- [ ] 一个 toy episode 完整 forward + backward(只 L_main)gradient 不爆

**E1 判决**(架构存亡门):
- [ ] 用 Hindsight 仓的 `scripts/05_compute_saliency.py` 在前 200 LIBERO-Long episode 上跑 [C10] HCS-H,产 `gamma_hat.pt`
- [ ] 用 LIBERO sub_task_id 边界(`obs.sub_task_id` 变化点)作为人工 boundary 标注
- [ ] 计算 `IoU(γ̂_geo z-score top-25%, sub_task_id boundary ± 4 frames)`
- [ ] 计算 `IoU(γ̂_sem z-score top-25%, sub_task_id boundary ± 4 frames)`
- [ ] 报告 IoU @ 0.3, 0.5, 0.7 三个阈值

**判决**:
| IoU @ 0.3 | 结果 | 后续 |
|---|---|---|
| ≥ 0.4 | **PASS** | 进入 M2,λ_1 anneal 启动(0 → 0.3,5k step) |
| 0.3 ≤ IoU < 0.4 | **SOFT-PASS** | 进入 M2 但 λ_1 目标值降到 0.15;记 red flag #1 |
| < 0.3 | **HARD FAIL** | 砍 [C10][C12][C13],λ_1 永久锁 0,走 §0.7.4 MVP fallback;不进入 M2 联立训练阶段 |

### Red flags(M1 阶段)
- **#1 E1 IoU < 0.3** → MVP fallback(详见 §0.7.4)
- **#2 γ_sem variance < 0.05 after 1 epoch warmup** → EMA warmup 系数曲线调整 / sigmoid → hard threshold + soft margin
- **#5 OOM on 4090 at batch 2** → 启用 grad checkpoint;若仍 OOM,downgrade ViT-B → ViT-S

### 所需 agent
- **implementer** — 接 [C5][C3][C4][C8] 真实 forward(从 stub → 实现)
- **experiment-runner** — 跑 Hindsight 在 200 ep 上;算 IoU
- **tester** — 启用 `test_forward_shapes.py` 形状契约 CI gate

### Milestone gate(自动停下)
**E1 IoU 报告产出后,orchestrator 自动停下并通知用户**——这是架构存亡决策点,不能自主推进。

---

## 4. M2 — [C5] + [C11] 独立收敛

### 进入条件
- M1 PASS 或 SOFT-PASS(用户已批准继续)
- λ_1 schedule 起点 step 已写入 config

### Deliverable
- [ ] [C5]+L_HCS-only 训练:在 LIBERO-Long 全集上跑,冻结 [C1] LoRA
  - 验证:`IoU(γ_sem_predicted, sub_task_id boundary ± 4) > 0.5`
  - 训练时间预算:2×4090 上 ~24 hr
- [ ] [C11] PRH 独立训练:用 sg(m_t) 输入(m_t 来自固定 model snapshot 或随机 hidden state),预测 (o_{t+k}, a*_{t+k}),k ∈ {4, 16, 64}
  - 验证:L_PRH 在 k=4 与 k=16 单调下降 ≥ 1k step;k=64 可观测下降趋势(LIBERO T~214 限制了 k=64 的有效样本数)
- [ ] **SG-1..SG-7 CI 全绿**(从 xfail 转为强制 pass)

### Red flags
- **#2 γ_sem variance 塌缩** → EMA warmup
- **#3 L_PRH @ k=64 不下降 ≥ 5k step** → 暂存 red flag,待 M3 联立看是否回升
- **#7 SG-1..SG-7 任意 1 条持续红 > 2 ticks** → BLOCK 所有 impl,tester agent 优先诊断

### 退出条件
- [ ] [C5] IoU > 0.5 报告
- [ ] [C11] L_PRH @ k=4,16 收敛曲线
- [ ] grad-flow CI 全绿
- [ ] M2 milestone PR review:用户 ack 即 merge stage2 → main

### 所需 agent
- **implementer** — [C5] EMA 标准化 + [C11] 6 个 prediction MLP
- **experiment-runner** — 两个独立训练 job
- **tester** — 启用 7 个 SG 测试 + slot lifecycle test
- **progress-reporter** — 每天更新 PROGRESS.md

### Milestone gate
M2 退出条件命中 → 自动停下,等用户合并 → 进入 M3 + 切换硬件到 6×A800。

---

## 5. M3 — 联立 [C3][C6]+[C4][C7] (硬件切换到 6×A800)

### 进入条件
- M2 PASS,所有 CI 绿
- 6×A800 access provisioned(用户确认)
- 数据已迁移到 A800 节点 / 网络挂载可用

### Deliverable
- [ ] L_main + L_PRH 联立训练全架构(λ_1=0,L_CSM 暂关)
  - 写头梯度范数 > 1e-5 持续 ≥ 5k step
  - M_geo 占用率 1-10%(稀疏写不变量,architecture line 540)
  - L_PRH 在 k=4,16,64 全部下降
- [ ] slot lifecycle test 全绿(`test_slot_lifecycle.py`)
- [ ] memory reset test 全绿(`test_memory_reset.py`)
- [ ] grad checkpoint + DDP 在 A800 上稳定(无 NCCL timeout)

### Red flags
- **#3 L_PRH @ k=64 不下降** → §H Trade-off 4 触发,K_s 升 ≥128
- **#4 写头 dead grad** → 检查 sg(γ) 是否被错误传播
- **#5 OOM at A800 80GB** → batch 24 → 16,启用更激进 grad ckpt

### 退出条件
- [ ] 联立训练 5 epoch 稳定运行(无 NaN, 无 dead grad)
- [ ] M_geo 稀疏占用监控通过
- [ ] L_PRH 三 horizon 全部下降

### 所需 agent
- **implementer** — 接 train_step.py 的完整 forward + 双 loss 联立
- **experiment-runner** — A800 集群 launch + 监控
- **tester** — slot lifecycle + memory reset CI gate
- **data-pipeline** — A800 节点上数据迁移(可能涉及网络拷贝或挂载)

---

## 6. M4 — 5-loss 完整训练 + LIBERO SR baseline

### 进入条件
- M3 PASS,联立训练稳定
- λ_1 anneal 已启动(从 M1 出口起算的 5k step 内)

### Deliverable
- [ ] 完整 5-loss 训练(L_main + L_HCS + L_PRH + L_CSM + L_aux,L_GC 关闭)
- [ ] LIBERO-Long 全 10 task 训练 ~5 epoch
- [ ] held-out 评估:在 LIBERO-Long 测试 split 上 SR
- [ ] baseline:OpenVLA + 8-frame history(从 OpenVLA repo fine-tune 到 LIBERO,或用现有 checkpoint)
- [ ] **目标:SR Δ ≥ 10%**(CHIME-VLA - OpenVLA baseline)

### Red flags
- **#6 5-loss balancing 第 8 周仍在 sweep** → 切 GradNorm / PCGrad 自动平衡
- L_HCS 抢主导致 L_main 反弹 → λ_1 schedule 调整目标值或 anneal 速度
- LIBERO SR 不达标 → 排查:γ̂ 标签质量、PRH horizon 选择、读出端 attention entropy

### 退出条件
- [ ] LIBERO-Long held-out SR Δ ≥ 10% over baseline
- [ ] 训练曲线干净(无 spike,grad 稳定)

### 所需 agent
- **implementer** — loss balancing 调参 + GradNorm/PCGrad 实现(可能需要)
- **experiment-runner**(2 个并发)— full 5-loss 训练 + baseline OpenVLA fine-tune
- **progress-reporter** — 周报

---

## 7. M5 — DROPPED(per user 决策)

原方案的"LIBERO-Long + CALVIN ABCD→D 全架构跑通"被砍掉,因 CALVIN/RoboCasa/BridgeV2 数据未下载,且用户决定 LIBERO-only 走全套。M5 时间窗(原 4 周)并入 M4 延长 + M6 提前。

---

## 8. M6 — Ablation 套件(10 项)

### 进入条件
- M4 PASS,LIBERO SR Δ ≥ 10% baseline
- main checkpoint 锁定(用于所有 ablation 的对照)

### Deliverable
按架构 §F.6 的 10 项,但**仅在 LIBERO-Long 上跑相对 SR**:

| # | Ablation | 期望 SR Δ vs full | 验证什么赌注 |
|---|---|---|---|
| 1 | γ_const = 1(不用 [C5])vs γ from [C5] | **必跑**:Δ ≥ 5% 才算 [C5] ROI 正 | §F-3 / §F.6 line 1765 |
| 2 | γ_const = γ_geo only(无 sem 通道) | -3~5% | 双通道必要性 |
| 3 | γ_const = γ_sem only(无 geo 通道) | -3~5% | 双通道必要性 |
| 4 | 删 L_HCS(λ_1=0) | -5~8% if E1 PASS | L_HCS 价值 |
| 5 | 删 L_PRH(λ_2=0) | -5~10% on 长程 task | L_PRH 长程 ROI |
| 6 | 删 L_CSM(λ_3=0) | -2~5% | slot 异质化必要性 |
| 7 | 单通道(M_geo only,K_s=0) | -10~15% | 双通道核心赌注 |
| 8 | K_s ∈ {32, 64, 128} 扫 | 64 vs 128 应平 / 32 应降 | §H Trade-off 4 |
| 9 | Δ ∈ {4} only / {4,16} / {4,16,64} | {4,16} vs {4,16,64} 应平 | Δ_max 选择 |
| 10 | slot_free mask vs naive zero-fill | mask > zero ≥ 1% | D5 修订正确性 |

每项跑 3 seed,LIBERO-Long held-out SR,均值 + 95% CI。

### Red flags(M6)
- **#7 L_main alone 持平 5-loss 全开** → paper claim 失效,重写论文方向
- ablation 出现 unstable seeds → 增加 seed 到 5 或 ablate 数据 split

### 退出条件
- [ ] 10 项 ablation 表完整,带 95% CI
- [ ] 至少 3 项 ablation 支持架构核心赌注(双通道、L_HCS、长 horizon PRH)
- [ ] M6 milestone gate:用户 review 论文 claim 是否成立,决定是否进入写作

### 所需 agent
- **experiment-runner**(3 个并发)— ablation 跑批
- **progress-reporter** — 每完成一项 ablation 更新 PROGRESS.md
- **implementer**(按需)— ablation 配置文件准备

---

## 9. 跨 milestone 的工程纪律

### Stop-grad CI gate(强制 blocking)
- `tests/test_grad_flow.py` 7 个 SG-N 测试,任何 1 条红 = 所有 impl PR 暂停
- M0 阶段允许 xfail(stub 没实际逻辑),M2 起所有必须强制 pass

### 训练监控(每 step / epoch)
| 指标 | 监控频率 | 阈值 |
|---|---|---|
| `L_main` | per step | 不应回升 |
| `L_HCS / L_PRH / L_CSM` | per step | 各自单调下降(M2+) |
| `γ_geo / γ_sem` mean & variance | per epoch | variance > 0.05 |
| `M_geo` voxel occupancy | per epoch | 1-10% |
| `M_sem` slot 利用率(`1 - mean(slot_free)`)| per epoch | > 0.3 |
| `[C8] attn_to_M_work` entropy | per epoch | > entropy_floor (per SG-7) |
| L_PRH per-k loss(k=4,16,64) | per epoch | 三 horizon 独立下降 |
| visual / proprio grad_ratio | per epoch | 不能塌到 100:1 以上 |
| 写头 grad_norm | per step | > 1e-5 |

### 数据/超参锁定
- LIBERO-Long 8/1/1 split,固定 seed,split 文件 commit 进 repo
- 所有超参写入 `configs/`,任何"今天调一下试试"必须 commit 一个 experiment yaml

### 实验工件
- 每次 train run:`output/runs/<exp_name>/<timestamp>/{config.yaml, ckpts/, logs/, metrics.json}`
- 复用 Hindsight 的 provenance 工具(`Hindsight/src/utils/git_info.py`):每次 run 写 `config_resolved.yaml + git_commit.txt + requirements_freeze.txt`

---

## 10. 自主执行接口(orchestrator 调用约定)

详见 `CODE_STRUCTURE.md` §自主执行 / `PLAN.md` §运行模式。本文件仅声明 milestone gate 的"自动停下"判据(orchestrator 必须遵守):

| 触发 | 动作 |
|---|---|
| M1 E1 IoU 报告产出 | STOP,通知用户决定 PASS / SOFT / FAIL |
| 任何 milestone 退出条件命中 | STOP,等待 user merge approve |
| `test_grad_flow` 持续红 > 2 ticks | STOP,tester 优先诊断 |
| Red flag 触发(任意 #1-#7)| STOP,等架构决策 |
| 实验预算超 2×4090 / 6×A800 时间 | STOP,确认续跑 |
| 4 ticks 无进度 | STOP,可能 loop stall |
