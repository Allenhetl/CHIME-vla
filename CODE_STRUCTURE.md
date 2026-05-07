# CHIME-VLA Code Structure

> 本文件是代码实施的**单一信息源(single source of truth)**。架构 spec 见 `chat/architecture_v2.1.md`,工程规范见 `CODE_STANDARDS.md`,实施计划见 `IMPLEMENTATION_PLAN.md`。本文件聚焦"代码长什么样"。

## 1. 目录树

```
CHIME-VLA/
├── pyproject.toml
├── README.md                          # quick start
├── PLAN.md                            # 项目仪表盘
├── PROGRESS.md                        # 实时状态
├── CODE_STANDARDS.md                  # 代码规范
├── CODE_STRUCTURE.md                  # 本文件
├── IMPLEMENTATION_PLAN.md             # M0-M6 milestone
├── .gitignore
├── chat/                              # 设计文档(只读)
│   ├── architecture_v2_FINAL.md       # v2.0 历史归档
│   ├── architecture_v2.1.md           # v2.1 主架构 spec(canonical)
│   └── chime_vla_proposal.md
├── docs/
│   ├── grad_flow_contract.md          # SG-1..SG-7 CI gate 规约
│   ├── hindsight_contract.md          # Hindsight 文件协议
│   └── data_schema.md                 # LIBERO h5 → cache .pt schema
├── src/chime_vla/
│   ├── __init__.py
│   ├── config.py                      # Hydra structured config 全集
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── vlm_backbone.py            # [C1] SigLIP-ViT + LoRA
│   │   └── fifo_buffer.py             # [C2] FIFO ring buffer
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── geo_grid.py                # [C6] M_geo multi-res voxel
│   │   ├── sem_bank.py                # [C7] M_sem slot bank + slot_free
│   │   └── lru.py                     # CSM-importance + timestamp eviction
│   ├── heads/
│   │   ├── __init__.py
│   │   ├── espc.py                    # [C5] ESPC ψ + EMA + projections
│   │   ├── geo_write.py               # [C3] geo write head
│   │   ├── sem_write.py               # [C4] sem write head
│   │   ├── prh.py                     # [C11] Predictive Read Head (training-only)
│   │   └── csm.py                     # [C12] Counterfactual Slot Mask (training-only)
│   ├── readout/
│   │   ├── __init__.py
│   │   └── read_interface.py          # [C8] cross-attn + 三线性采样
│   ├── action/
│   │   ├── __init__.py
│   │   └── action_expert.py           # [C9] π0 flow matching head + LoRA
│   ├── hindsight/
│   │   ├── __init__.py
│   │   └── consumer.py                # 读 Hindsight γ̂ .pt
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_step.py              # 一帧 forward + 5-loss assembly
│   │   ├── losses.py                  # L_main / L_HCS / L_PRH / L_CSM / L_aux / L_GC
│   │   ├── schedules.py               # λ_1 step-aware schedule
│   │   ├── lightning_module.py        # ChimeVlaLightning
│   │   └── datamodule.py              # LiberoLongDataModule
│   └── utils/
│       ├── __init__.py
│       ├── distributed.py             # all_gather_concat (复用 Hindsight)
│       ├── seeding.py                 # (复用 Hindsight)
│       ├── letterbox.py               # 224x224 输入(复用 Hindsight)
│       ├── losses.py                  # masked_mse (复用 Hindsight)
│       ├── grad_flow_check.py         # SG-1..SG-7 runtime verifier
│       └── memory_reset.py            # episode boundary reset hook
├── configs/
│   ├── default.yaml                   # 顶层组合
│   ├── base/
│   │   └── chime.yaml                 # 默认 ChimeConfig 实例
│   ├── model/
│   │   ├── chime_full.yaml            # 13 组件全开 (M3+)
│   │   └── chime_mvp.yaml             # §0.7.4 MVP (ViT-B + 1-step + GRU psi)
│   ├── train/
│   │   ├── m0_smoke.yaml              # forward smoke(B=2,T=64)
│   │   ├── m1_smoke.yaml              # 加 [C10] consumer
│   │   ├── m2_phi_only.yaml           # [C5]+L_HCS only
│   │   ├── m2_prh_only.yaml           # [C11] independent
│   │   ├── m3_main_prh.yaml           # L_main + L_PRH
│   │   ├── m4_full_5loss.yaml         # 5-loss
│   │   └── m6_ablation_template.yaml  # ablation 用
│   ├── data/
│   │   └── libero_long.yaml
│   └── experiment/
│       └── debug.yaml                 # @package _global_
├── scripts/
│   ├── 00_build_libero_cache.py       # h5 → per-episode .pt
│   ├── 01_run_hindsight_consumer.py   # 读 Hindsight 产物 sanity check
│   ├── 10_train.py                    # Hydra 训练入口
│   ├── 20_eval_sr.py                  # LIBERO held-out SR 评估
│   └── 30_run_ablation.py             # M6 ablation 批量跑
└── tests/
    ├── conftest.py                    # fixtures(synthetic batch / mock memory)
    ├── test_grad_flow.py              # SG-1..SG-7 (CI gate, blocking)
    ├── test_slot_lifecycle.py         # slot evict + slot_free invariant
    ├── test_memory_reset.py           # episode 边界清零
    ├── test_forward_shapes.py         # 每个接口形状契约
    ├── test_loss_finite.py            # 4-step toy 5-loss 有限
    └── test_hindsight_consumer.py     # γ̂ .pt schema
```

## 2. Hydra Structured Config(完整 schema)

`src/chime_vla/config.py`:

```python
from dataclasses import dataclass, field
from typing import Optional

# ===== Component configs =====

@dataclass
class C1Config:
    backbone: str = "siglip_vit_b"          # "siglip_vit_l" / "siglip_vit_b" / "siglip_vit_s"
    lora_r: int = 16
    freeze_backbone: bool = True            # MVP 默认 freeze, full 可解开

@dataclass
class C2Config:
    K_w: int = 8                            # FIFO 长度
    d_h: int = 1152                         # token 维度
    N: int = 256                            # token 数

@dataclass
class C3Config:
    voxel_proj_hidden: int = 256
    write_levels: list[int] = field(default_factory=lambda: [16])  # MVP 单层 16^3, full [8, 16, 32]

@dataclass
class C4Config:
    qv_proj_hidden: int = 256
    softmax_temp: float = 0.5

@dataclass
class C5Config:
    psi_layers: int = 1
    use_gru: bool = True                    # MVP=True, full=False(用 1-layer transformer)
    d_proj: int = 64                        # geo_proj / sem_proj 输出维度
    ema_coeff: float = 0.99
    ema_warmup_steps: int = 2000
    sigmoid_temp: float = 1.0

@dataclass
class C6Config:
    levels: list[int] = field(default_factory=lambda: [16])     # MVP 单分辨率
    d_g: int = 64                           # 每 voxel 维度
    alpha_l: list[float] = field(default_factory=lambda: [1.0]) # MVP 单层 alpha=1.0
    workspace_bounds: list[float] = field(default_factory=lambda: [-1.0, 1.0, -1.0, 1.0, 0.0, 1.0])  # x_min x_max y_min y_max z_min z_max

@dataclass
class C7Config:
    K_s: int = 64
    d_s: int = 256
    evict_strategy: str = "timestamp"       # MVP="timestamp", M3+="csm_lru"

@dataclass
class C8Config:
    N_q: int = 16
    N_geo_q: int = 16
    use_kv_cache: bool = True               # 推理优化

@dataclass
class C9Config:
    head: str = "flow_matching_pi0"
    lora_r: int = 16
    one_step_distill: bool = True           # MVP=True, full=False(4-8 ODE step)
    freeze_base: bool = True

@dataclass
class C10Config:
    enabled: bool = True                    # E1 fail 后置 False
    base_policy: str = "pi05"               # "pi05" / "chime_early_ckpt"
    deltas: list[int] = field(default_factory=lambda: [4, 16])  # full 加 64
    rudder_dim: int = 256
    saliency_method: str = "EAGN"           # "EAGN" / "exact_jacobian"

@dataclass
class C11Config:
    horizons: list[int] = field(default_factory=lambda: [4, 16, 64])
    alpha_a: float = 1.0                    # action loss 权重
    pred_mlp_hidden: int = 512

@dataclass
class C12Config:
    n_slots_per_step: int = 4
    beta: float = 0.1                       # log-mean 项系数

# ===== Loss / training / data =====

@dataclass
class LossConfig:
    lambda_1_target: float = 0.3            # L_HCS 权重最终值
    lambda_1_schedule: str = "anneal_post_e1"  # "anneal_post_e1" / "constant" / "off"
    step_e1_pass: int = 0                   # E1 PASS 时刻; 0 = 启动即 anneal
    anneal_steps: int = 5000                # 0 → λ_1_target 线性 anneal step 数
    lambda_2: float = 0.5                   # L_PRH
    lambda_3: float = 0.1                   # L_CSM
    lambda_4: float = 0.0                   # L_GC, MVP off
    lambda_ent: float = 0.01                # L_aux
    entropy_floor: float = 1.0              # SG-7 监控阈值

@dataclass
class TrainConfig:
    lr: float = 1e-4
    bs: int = 24                            # per-rank
    precision: str = "bf16-mixed"
    max_epochs: int = 5
    grad_clip: float = 1.0
    warmup_steps: int = 500
    optimizer: str = "adamw"
    betas: tuple[float, float] = (0.9, 0.95)
    wd: float = 0.01
    bptt_truncate: int = 32                 # detach memory state every N step
    grad_ckpt: bool = True
    accumulate: int = 1

@dataclass
class DataConfig:
    root: str = "/home/sqmluser/data/memory_vla/libero_long/"
    cache_root: str = "output/cache/libero_long"
    splits_path: str = "output/splits/libero_long_8_1_1.json"
    T_max: int = 256
    img_size: int = 224
    proprio_dim: int = 8
    action_dim: int = 8
    normalize: bool = True

@dataclass
class HindsightConfig:
    enabled: bool = True                    # E1 fail → False
    gamma_hat_root: str = "/home/sqmluser/workspace/theaj/CHIME-VLA/Hindsight/output/saliency/gamma_hat"
    strategy: str = "per_task_q75"
    task: str = "libero_long"

@dataclass
class ChimeConfig:
    c1: C1Config = field(default_factory=C1Config)
    c2: C2Config = field(default_factory=C2Config)
    c3: C3Config = field(default_factory=C3Config)
    c4: C4Config = field(default_factory=C4Config)
    c5: C5Config = field(default_factory=C5Config)
    c6: C6Config = field(default_factory=C6Config)
    c7: C7Config = field(default_factory=C7Config)
    c8: C8Config = field(default_factory=C8Config)
    c9: C9Config = field(default_factory=C9Config)
    c10: C10Config = field(default_factory=C10Config)
    c11: C11Config = field(default_factory=C11Config)
    c12: C12Config = field(default_factory=C12Config)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hindsight: HindsightConfig = field(default_factory=HindsightConfig)
    seed: int = 42
    experiment_name: str = "default"
    milestone: str = "M0"
    output_root: str = "output"
```

## 3. 关键接口签名

### 3.1 [C1] VLM Backbone

```python
# src/chime_vla/perception/vlm_backbone.py
class VLMBackbone(nn.Module):
    def __init__(self, cfg: C1Config): ...
    def forward(self, rgb: Tensor, proprio: Tensor) -> Tensor:
        """rgb: (B, 3, 224, 224) float32  proprio: (B, P=8) float32
        returns: h_t (B, N=256, d_h=1152) bf16"""
```

### 3.2 [C2] FIFO Buffer

```python
# src/chime_vla/perception/fifo_buffer.py
class WorkBuffer:
    """无可学参数, 维护 (B, K_w, N, d_h) ring."""
    def __init__(self, cfg: C2Config, batch_size: int, device): ...
    def reset(self, batch_indices: Optional[Tensor] = None) -> None: ...
    def append(self, h_t: Tensor) -> Tensor:
        """h_t: (B, N, d_h)  returns: M_work (B, K_w, N, d_h)"""
    def snapshot(self) -> Tensor: ...
```

### 3.3 [C5] ESPC

```python
# src/chime_vla/heads/espc.py
class ESPC(nn.Module):
    def __init__(self, cfg: C5Config, d_h: int): ...
    def forward(self, h_t: Tensor, m_work: Tensor) -> tuple[Tensor, Tensor]:
        """h_t: (B, N, d_h)  m_work: (B, K_w, N, d_h)  (M_work^{t-1} BEFORE append)
        returns: (gamma_geo, gamma_sem) each (B,) in [0, 1]"""
    def update_ema(self) -> None: ...
```

### 3.4 [C3] / [C4] 写头

```python
# src/chime_vla/heads/geo_write.py
class GeoWriteHead(nn.Module):
    def __init__(self, cfg: C3Config, d_h: int, d_g: int, alpha_l: list[float]): ...
    def forward(self, h_t: Tensor, gamma_geo: Tensor, m_geo: GeoGrid) -> None:
        """h_t: (B, N, d_h)  gamma_geo: (B,) [SG-1: caller passes sg(.)]
        Mutates m_geo in-place via delta-rule scatter."""

# src/chime_vla/heads/sem_write.py
class SemWriteHead(nn.Module):
    def __init__(self, cfg: C4Config, d_h: int, d_s: int, K_s: int): ...
    def forward(self, h_t: Tensor, gamma_sem: Tensor, m_sem: SemBank) -> None:
        """gamma_sem: (B,) [SG-1: caller passes sg(.)]
        Uses slot_free mask + softmax logit penalty per [C7]."""
```

### 3.5 [C6] / [C7] Memory containers

```python
# src/chime_vla/memory/geo_grid.py
class GeoGrid:
    """multi-res voxel grid, no learnable params."""
    levels: list[int]                     # e.g. [16] or [8, 16, 32]
    grids: dict[int, Tensor]              # level -> (B, D, H, W, d_g)
    timestamp: dict[int, Tensor]          # level -> (B, D, H, W) int (last write step)
    alpha_l: list[float]
    def __init__(self, cfg: C6Config, batch_size: int, d_g: int, device): ...
    def reset(self, batch_indices: Optional[Tensor] = None) -> None: ...
    def occupancy_pct(self) -> dict[int, float]: ...

# src/chime_vla/memory/sem_bank.py
class SemBank:
    """slot bank with frozen random keys + episode-scoped slot_free mask."""
    k: Tensor                             # (B, K_s, d_s)  frozen per-episode
    v: Tensor                             # (B, K_s, d_s)  delta-rule accumulated
    slot_free: Tensor                     # (B, K_s) bool
    timestamp: Tensor                     # (B, K_s) int (last write step)
    def __init__(self, cfg: C7Config, batch_size: int, device): ...
    def reset(self, batch_indices: Optional[Tensor] = None, regen_keys: bool = True) -> None: ...
    def evict(self, batch_idx: int, slot_idx: int) -> None:
        """v_i ← 0, slot_free[i] ← 1, k_i unchanged."""
```

### 3.6 [C8] Read Interface

```python
# src/chime_vla/readout/read_interface.py
class ReadInterface(nn.Module):
    def __init__(self, cfg: C8Config, d_h: int, d_s: int, K_w: int, K_s: int): ...
    def forward(
        self, m_work: Tensor, m_geo: GeoGrid, m_sem: SemBank, h_t: Tensor,
        prh_path: bool = False,
    ) -> Tensor:
        """returns c_t (B, N_q + K_w, d_h).
        prh_path=True → caller signals SG-2: query proj will be sg-isolated."""
    @property
    def attn_entropy_to_M_work(self) -> Tensor:
        """Last forward's attention entropy over M_work (for L_aux + SG-7 monitor)."""
```

### 3.7 [C9] Action Expert

```python
# src/chime_vla/action/action_expert.py
class ActionExpert(nn.Module):
    def __init__(self, cfg: C9Config, d_h: int, action_dim: int): ...
    def forward(self, c_t: Tensor, h_t_cls: Tensor) -> Tensor:
        """c_t: (B, N_q + K_w, d_h)  h_t_cls: (B, d_h)
        returns predicted action (B, action_dim=8)"""
    def freeze(self) -> None: ...
```

### 3.8 [C11] / [C12] Training-only heads

```python
# src/chime_vla/heads/prh.py
class PRH(nn.Module):
    def __init__(self, cfg: C11Config, d_h: int, action_dim: int): ...
    def forward(self, m_t_sg: Tensor) -> dict[int, tuple[Tensor, Tensor]]:
        """m_t_sg: (B, d_h)  caller MUST pass sg(.) per SG-2.
        returns {k: (o_hat, a_hat)} for k in horizons."""

# src/chime_vla/heads/csm.py
class CSM:
    """Not a Module; a callable that calls frozen [C9] N_slots_per_step times."""
    def __init__(self, cfg: C12Config): ...
    def __call__(
        self, m_t: Tensor, m_sem: SemBank, frozen_action_expert: ActionExpert,
    ) -> Tensor:
        """returns w_i (B, n_slots_per_step) — slot importance weights."""
```

### 3.9 Hindsight Consumer

```python
# src/chime_vla/hindsight/consumer.py  (per docs/hindsight_contract.md §4)
class HindsightConsumer:
    def __init__(self, root: Path, strategy: str, task: str): ...
    def load(self, episode_id: int) -> HindsightSample: ...
    def has(self, episode_id: int) -> bool: ...
    def list_available(self) -> list[int]: ...
```

### 3.10 训练 step 入口

```python
# src/chime_vla/training/train_step.py
def chime_train_step(
    batch: dict[str, Tensor],       # rgb / proprio / action / sub_task_id / episode_id / valid_mask
                                    # + (gamma_hat_geo / gamma_hat_sem) if hindsight.enabled
    model: ChimeVlaModule,
    cfg: ChimeConfig,
    step: int,
) -> dict[str, Tensor]:
    """returns {'L_main', 'L_HCS', 'L_PRH', 'L_CSM', 'L_aux', 'total', plus diagnostics}."""
```

## 4. 测试矩阵(per CODE_STANDARDS §4)

```python
# tests/test_grad_flow.py — 7 SG tests, CI gate

@pytest.mark.xfail(reason="M0: stubs not impl", strict=False)
def test_sg_1_gamma_to_psi(model_full): ...

@pytest.mark.xfail(reason="M0: stubs not impl", strict=False)
def test_sg_2_prh_query_to_perception(model_full): ...

# ... SG-3..SG-7
```

xfail 在 M0/M1 阶段允许;M2 起 strict=True,任何 xfail 通过(测试反常 pass)也红。

## 5. 数据 pipeline

```
LIBERO h5 (raw)
   /home/sqmluser/data/memory_vla/libero_long/traj_NNNN.h5
       ├── obs/agentview_rgb        (T, 224, 224, 3) uint8
       ├── obs/proprio              (T, 8) float32
       ├── obs/sub_task_id          (T,) int32
       ├── actions                  (T, 8) float32
       └── episode_id               int

scripts/00_build_libero_cache.py 读取 + SigLIP 特征预提取(可选)
       ↓
output/cache/libero_long/{task}/ep_NNNNNN.pt
   {
     'rgb_feature': fp16 (T, N=256, d_h=1152),    # 预提取节省训练时间
     'rgb_raw': uint8 (T, 224, 224, 3),           # debug/viz 用, 可选保留
     'proprio': fp32 (T, 8),
     'action': fp32 (T, 8),
     'sub_task_id': int32 (T,),
     'episode_id': int,
     'task_name': str,
     'T': int,
   }

LiberoLongDataModule (length-bucket sampler)
       ↓
batch: {
   'rgb': (B, T, 3, 224, 224) or 'rgb_feature': (B, T, N, d_h),
   'proprio': (B, T, 8),
   'action': (B, T, 8),
   'sub_task_id': (B, T),
   'episode_id': (B,),
   'valid_mask': (B, T) bool,                     # True for valid frames
   # if hindsight.enabled:
   'gamma_hat_geo': (B, T) float32,
   'gamma_hat_sem': (B, T) float32,
}
```

## 6. Hindsight 复用清单(直接 import / copy)

> **重要**:CHIME-VLA 不直接 `from Hindsight.src.X import Y`(decouple 政策 per CODE_STANDARDS §1.6)。
> 复用方式有二:
> 1. **Copy-into-source**:把 Hindsight 的 utility 文件拷贝到 `src/chime_vla/utils/`,在 commit 中标注源 commit sha
> 2. **Sibling-import via path**:仅在脚本/工具(`scripts/`)中可用 `sys.path` 注入,**禁止在生产代码中**

| Hindsight 来源 | CHIME-VLA 目的地 | 复用方式 |
|---|---|---|
| `Hindsight/src/encoder/siglip_wrapper.py` | `src/chime_vla/perception/_siglip_wrapper.py` | copy + 改名 |
| `Hindsight/src/utils/distributed.py` | `src/chime_vla/utils/distributed.py` | copy verbatim |
| `Hindsight/src/utils/losses.py` (`masked_mse`) | `src/chime_vla/utils/losses.py` | copy verbatim |
| `Hindsight/src/utils/seeding.py` | `src/chime_vla/utils/seeding.py` | copy verbatim |
| `Hindsight/src/utils/letterbox.py` | `src/chime_vla/utils/letterbox.py` | copy verbatim |
| `Hindsight/src/utils/git_info.py` | `src/chime_vla/utils/git_info.py` | copy verbatim |
| `Hindsight/output/saliency/gamma_hat/...` | (不复制,文件协议消费) | runtime read via `HindsightConsumer` |

## 7. Forward 顺序伪代码(canonical, per architecture v2.1 §B.2)

```python
# 一帧 t 内 (training mode, 完整 5-loss)
def chime_forward_one_step(state, batch, t, cfg):
    # 1. Perception
    h_t = model.c1(batch['rgb'][:, t], batch['proprio'][:, t])         # (B, N, d_h)

    # 2. ESPC reads M_work^{t-1} BEFORE append
    gamma_geo, gamma_sem = model.c5(h_t, state.M_work)                 # (B,), (B,)

    # 3. FIFO append
    state.M_work = state.c2.append(h_t)                                # (B, K_w, N, d_h)

    # 4. Write heads (in-place mutation of memory)
    model.c3(h_t, sg(gamma_geo), state.M_geo)                          # SG-1
    model.c4(h_t, sg(gamma_sem), state.M_sem)                          # SG-1, slot_free aware

    # 5. Read interface
    c_t = model.c8(state.M_work, state.M_geo, state.M_sem, h_t,
                   prh_path=False)                                      # main path

    # 6. Action
    a_t = model.c9(c_t, h_t.mean(dim=1))                                # CLS-style pool

    # 7. Aux: PRH on m_t (sg) — only for training
    m_t = c_t.mean(dim=1)                                               # (B, d_h)
    prh_out = model.c11(sg(m_t))                                        # SG-2

    return {
        'h_t': h_t, 'gamma_geo': gamma_geo, 'gamma_sem': gamma_sem,
        'a_pred': a_t, 'c_t': c_t, 'prh_out': prh_out,
    }

# 完整 step (5-loss assembly)
def chime_train_step(batch, model, cfg, step):
    state = ChimeState.init(model, batch_size=batch['rgb'].shape[0])
    out_seq = []
    for t in range(batch['rgb'].shape[1]):
        out_t = chime_forward_one_step(state, batch, t, cfg)
        out_seq.append(out_t)

    # 5 losses
    L_main = flow_match_loss([o['a_pred'] for o in out_seq], batch['action'], batch['valid_mask'])
    L_aux = -cfg.loss.lambda_ent * model.c8.attn_entropy_to_M_work.mean()

    L_HCS = torch.zeros(())
    if cfg.hindsight.enabled:
        L_HCS = (
            bce_loss(stack([o['gamma_geo'] for o in out_seq]),
                     sg(batch['gamma_hat_geo']), batch['valid_mask'])
            + bce_loss(stack([o['gamma_sem'] for o in out_seq]),
                       sg(batch['gamma_hat_sem']), batch['valid_mask'])
        )

    L_PRH = compute_prh_loss(out_seq, batch, cfg.c11.horizons, cfg.c11.alpha_a)
    L_CSM = compute_csm_loss(out_seq, state.M_sem, model.c9_frozen, cfg.c12)

    lambda_1 = lambda_1_schedule(step, cfg.loss)
    total = (L_main + lambda_1 * L_HCS + cfg.loss.lambda_2 * L_PRH
             + cfg.loss.lambda_3 * L_CSM + L_aux)

    return {'L_main': L_main, 'L_HCS': L_HCS, 'L_PRH': L_PRH, 'L_CSM': L_CSM,
            'L_aux': L_aux, 'total': total, 'lambda_1': lambda_1}
```

## 8. M0 deliverable checklist(用于 implementer agent 验收)

- [ ] `pyproject.toml` 安装可用(`pip install -e .`)
- [ ] `from chime_vla.config import ChimeConfig` import OK
- [ ] 13 个组件文件每个都有 class stub + docstring + interface signature(可 `raise NotImplementedError`)
- [ ] `tests/test_forward_shapes.py` 至少 1 个 test 真跑 pass(stub 也能 pass 形状契约)
- [ ] `tests/test_grad_flow.py` 7 个 SG xfail 占位(pytest 不报 error)
- [ ] `python scripts/00_build_libero_cache.py --help` 输出 OK
- [ ] `python -c "import hydra; from chime_vla.config import ChimeConfig; print('ok')"` 输出 ok
- [ ] `configs/default.yaml` Hydra 加载无 error
- [ ] `pytest -x tests/` 全部 pass(允许 xfail)
