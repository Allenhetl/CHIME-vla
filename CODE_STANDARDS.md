# CHIME-VLA Code Standards

> 本文档**继承** `Hindsight/CODE_STANDARDS.md` 与 `Hindsight/CODE_STANDARDS_V2.md` 的全部条文,以下仅列 CHIME-VLA 的**增量与差异**。Hindsight 没说的,默认按 Hindsight 标准执行;Hindsight 与本文件冲突的,以本文件为准。

## §0 继承

直接继承 Hindsight CODE_STANDARDS:
- §0.1 技术栈(Python 3.10+, PyTorch ≥2.1 <2.5, Lightning, Hydra+structured config, BF16 训练 / FP32 saliency, PyAV / imageio)
- §0.2 复现性(provenance 三件套:`config_resolved.yaml + git_commit.txt + requirements_freeze.txt`)
- §2 Config 系统(structured dataclass + Hydra YAML 树 + 命名约定)
- §3 数据存储契约(per-episode `.pt` cache + 全局 `meta.json` + splits JSON + thresholds JSON)
- §4 命名规范(tensor 维度大写字母 / 文件 snake_case / 类 PascalCase)
- §5 Lightning 模块标准(LightningDataModule 接口、DDP `find_unused_parameters_false`、`TaskBalancedLengthBucketSampler` rank-strided)

## §1 CHIME-VLA 特有约束

### 1.1 Stop-grad 矩阵(强制 CI gate)

**所有 sg 边界必须有对应的 unit test**,详见 `docs/grad_flow_contract.md`。

`tests/test_grad_flow.py` 7 个 SG-N 测试:**任意 1 条红 = 所有 impl PR 暂停**。

CI 规则:
- `pytest -x tests/test_grad_flow.py` 必须绿(M2 起强制,M0/M1 阶段允许 xfail 占位)
- 添加新 loss / 新 head → 必须同步评估 sg 影响,补对应 SG-N test
- 重构 [C5] / [C8] / [C9] / [C10] / [C11] / [C12] → 触发 grad-flow review

### 1.2 Memory tensor 的 batch 维契约

**所有 memory 数据结构必须显式带 batch 维**:

```python
# 正确
M_geo: Tensor of shape (B, L, D, H, W, C_g)            # L=级数, MVP=1
M_sem.v: Tensor of shape (B, K_s, d_s)
M_sem.k: Tensor of shape (B, K_s, d_s)                 # frozen random keys
slot_free: Tensor of shape (B, K_s), dtype bool        # episode-scoped mask
M_work: Tensor of shape (B, K_w, N, d_h)               # FIFO ring

# 错误(没有 batch 维, 训练时多 episode 状态会串)
M_geo: Tensor of shape (L, D, H, W, C_g)
```

Episode 边界由 `chime_vla.utils.memory_reset.reset_memory(state, batch_indices)` 显式调用清零,不允许隐式假设"每个 step 自动 reset"。

### 1.3 Forward 顺序锁定

每帧 t 内组件必须按 `chat/architecture_v2.1.md` §B.2 的伪代码顺序调用:

```
C1 → C5 → C2.append → {C3, C4} → C8 → C9 → loss
```

**关键不变量**(违反即报错):
- C5 调用时 `M_work` 尚未 append h_t(C5 看到 `M_work^{t-1}`)
- C3 / C4 调用时 γ 已经过 `sg(...)`(SG-1)
- L_PRH 路径上 `m_t = sg(m_t)` 显式标记(SG-2)

### 1.4 Loss reduction 一律 mean over (B, valid-T)

所有 component loss(L_main / L_HCS / L_PRH / L_CSM / L_aux / L_GC)统一:
- 求和方式:`mean over (B, valid-T)` with episode mask
- 复用 `Hindsight/src/utils/losses.py::masked_mse` 模式
- 禁止使用 `sum` reduction(除非显式注释为何不能 mean)

### 1.5 λ_1 schedule(强制 step-aware)

`L_total = L_main + lambda_1(step) * L_HCS + lambda_2 * L_PRH + ...`

`lambda_1` 必须是 `step` 的函数,不是常量。Schedule:
- step < `cfg.loss.step_e1_pass`: 0
- step ∈ [`cfg.loss.step_e1_pass`, `cfg.loss.step_e1_pass + 5000`): 线性 0 → `cfg.loss.lambda_1_target`(0.3)
- step ≥ `cfg.loss.step_e1_pass + 5000`: `cfg.loss.lambda_1_target`

实现见 `src/chime_vla/training/schedules.py::lambda_1_schedule`。

### 1.6 Hindsight 文件协议(不直接 import)

CHIME-VLA **不允许** `from Hindsight.src...` 直接 import Hindsight 代码(decouple)。所有数据通过文件协议:

```
Hindsight/output/saliency/gamma_hat/per_task_q75/libero_long/ep_NNNNNN.pt
                         ↑ schema lock per docs/hindsight_contract.md
```

CHIME-VLA 端只通过 `chime_vla.hindsight.consumer.HindsightConsumer` 读取。

### 1.7 dtype 矩阵(覆写 Hindsight)

| 计算路径 | dtype | 备注 |
|---|---|---|
| `[C1]` SigLIP forward | bf16 | 继承 Hindsight |
| `[C2]` FIFO + `[C8]` cross-attn | bf16 | autocast |
| `[C5]` ESPC ψ + EMA | **fp32** | EMA running stats 不能 bf16,数值下溢 |
| `[C3] [C4]` 写头 + `[C6] [C7]` memory | fp32 | delta-rule 累加跨 T~200 step,bf16 会损失精度 |
| `[C9]` flow matching ODE | bf16 | autocast |
| `[C10]` Jacobian(offline,Hindsight 端)| fp32 | 已锁,见 Hindsight CODE_STANDARDS §12.7 |
| Loss 反向 | fp32 | 与 Hindsight Phase C 一致 |

实现:用 `torch.cuda.amp.autocast` + 手动 `.to(torch.float32)` 在内存写入 / EMA 路径强制 fp32。

### 1.8 Episode 内 BPTT 截断(L_PRH / L_HCS)

L_PRH / L_HCS 在 episode 内反向跨度:
- `M_work` 跨 K_w=8 step 可微
- `M_geo / M_sem` delta-rule 跨整个 episode(T~200)累加,但**不 BPTT 整段**
- 默认设置:每 32 step `detach()` 一次 memory 状态(可微窗口 32 step,跨 32 step 边界用 stop-grad)

config:
```yaml
train:
  bptt_truncate: 32      # detach memory state every N steps
```

### 1.9 Slot lifecycle 不变量

`slot_free` mask 必须满足:
- episode 开始:全 1
- evict slot i:`v_i ← 0`,`slot_free[i] ← 1`,`k_i 不变`
- 写入 slot i:`v_i ← v_i + Δv`,`slot_free[i] ← 0`
- softmax 路由前:logit_i ← logit_i − 1e9 · slot_free[i]
- 读出 attention 前:同样 logit penalty

`tests/test_slot_lifecycle.py` 是 CI gate。

## §2 配置系统(CHIME 增量)

### 2.1 Structured Config 必须包含的子配置

```python
@dataclass
class ChimeConfig:
    # 13 组件配置
    c1: C1Config      # backbone
    c2: C2Config      # FIFO
    c3: C3Config      # geo write
    c4: C4Config      # sem write
    c5: C5Config      # ESPC
    c6: C6Config      # geo grid
    c7: C7Config      # sem bank
    c8: C8Config      # read interface
    c9: C9Config      # action expert
    c10: C10Config    # HCS-H (training-only)
    c11: C11Config    # PRH (training-only)
    c12: C12Config    # CSM (training-only)
    # 训练 / 数据
    loss: LossConfig
    train: TrainConfig
    data: DataConfig
    hindsight: HindsightConfig    # 文件协议根路径
    # 元数据
    seed: int = 42
    experiment_name: str = "default"
    milestone: str = "M0"          # M0..M6
```

### 2.2 必须支持的 override 路径

继承 Hindsight §2.4 的全部 override,加上 CHIME 特有:
```bash
# 切换到 MVP 配置
+experiment=mvp_libero_long

# 关闭 [C10] (E1 fail 后 fallback)
hindsight.enabled=false loss.lambda_1_target=0

# 调试某个组件
+model=chime_full c5.psi_layers=2

# milestone 切换
milestone=M2 +train=m2_phi_only
```

### 2.3 Config 跨 milestone 锁定

每个 milestone 有专属 train config(`configs/train/m{0,1,2,3,4,6}_*.yaml`),不允许跨 milestone 共用——避免"今天调一下试试"造成无 commit 的悄悄改动。

## §3 PR / 分支规约

继承 Hindsight CODE_STANDARDS_V2 §1.3,加上:

- **Branch per milestone**:`stage1/consistency-patches`,`stage2/implementation-plan`,`stage3/code-skeleton`,然后 `m0/skeleton`,`m1/e1-judgment`,`m2/espc-prh`,...
- **Commit 前缀**(扩展 Hindsight):
  - `feat(C5):` / `feat(C10):` ... — 新组件实现
  - `fix(SG-N):` — grad-flow 修复
  - `fix(slot):` — slot lifecycle 修复
  - `data:` — 数据 pipeline / cache
  - `exp(M2):` — 实验运行 / 调参 commit
  - `progress:` — 仅 PROGRESS.md 更新
  - `docs:` — 文档(包括 architecture_vN.M.md 修订)
  - `chore:` — 配置 / CI / 工具
- **Merge 时机**:仅在 milestone gate PASS 后由用户 approve 后合并到 main
- **绝不**:`--force` / `--no-verify` / 覆盖未审 commit

## §4 测试矩阵

| 测试 | 类型 | 触发 | M0 | M1 | M2+ |
|---|---|---|---|---|---|
| `test_grad_flow.py` (7 SG) | CI gate | 每 PR | xfail | xfail | strict |
| `test_slot_lifecycle.py` | CI gate | 每 PR | xfail | xfail | strict |
| `test_memory_reset.py` | CI gate | 每 PR | xfail | strict | strict |
| `test_forward_shapes.py` | CI gate | 每 PR | strict | strict | strict |
| `test_loss_finite.py` | CI gate | 每 PR | xfail | strict | strict |
| `test_hindsight_consumer.py` | unit | 每 PR | strict | strict | strict |
| sanity callback (per-task val mse 等) | runtime | 训练时 | — | — | enabled |

## §5 监控与日志(继承 Hindsight + CHIME 增量)

继承 Hindsight TensorBoard + CSVLogger 双路写入。CHIME 增量:

- **必须 log 的指标**(per epoch,见 IMPLEMENTATION_PLAN §9):
  - `gamma_geo/mean`, `gamma_geo/var`, `gamma_sem/mean`, `gamma_sem/var`
  - `M_geo/occupancy_pct`(总 voxel 中非零比例)
  - `M_sem/utilization`(`1 - mean(slot_free)`)
  - `attn_to_M_work/entropy_min`(SG-7 指标)
  - `L_PRH/per_k/k=4`, `L_PRH/per_k/k=16`, `L_PRH/per_k/k=64`(分 horizon)
  - `lambda_1`(当前 step 值)
  - `grad_norm/write_heads`, `grad_norm/c5_psi`
- **Wandb 集成**(可选,M3+ 启用):同时 push 到 wandb,tag = milestone + experiment_name
