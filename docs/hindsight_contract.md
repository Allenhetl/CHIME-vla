# Hindsight ↔ CHIME-VLA File Protocol

> CHIME-VLA 不直接 import Hindsight code(decouple)。Hindsight 通过文件系统产出 `gamma_hat.pt`,CHIME-VLA 通过 `chime_vla.hindsight.consumer` 读取。两端只通过本契约通讯。

## 1. 谁是谁

- **Hindsight(`./Hindsight/`)**:offline saliency labeler。产出每帧的 γ̂_geo, γ̂_sem ∈ [0,1]——"这一帧值不值得记入 long-term memory"的离线监督信号。对应架构 [C10] HCS-H。
- **CHIME-VLA(`./`)**:online VLA 系统。training time 消费 γ̂ 作为 L_HCS BCE target;deploy time 不需要 γ̂(由 [C5] ESPC 自跑产生 γ)。

## 2. 文件路径契约

```
/home/sqmluser/workspace/theaj/CHIME-VLA/Hindsight/output/saliency/gamma_hat/
    └── per_task_q75/                   # strategy: per-task quantile-75% threshold
        └── libero_long/
            ├── ep_000000.pt
            ├── ep_000001.pt
            ├── ...
            └── ep_NNNNNN.pt
```

- **strategy 名**:`per_task_q75` 是默认。Hindsight 还会产其他策略(如 `global_q70`),CHIME-VLA 通过 config 选一个。
- **task 名**:`libero_long`。未来若加新数据集,会在同级新建子目录(`bridge_v2`, `robocasa`, ...)。
- **episode 编号**:`ep_NNNNNN.pt`,与 LIBERO `traj_NNNN.h5` 的 episode index 一一对应(`NNNN` 是 4-digit zero-padded episode id 在 Hindsight 端,转 6-digit 为 CHIME 端)。

## 3. `.pt` schema(每个 episode)

```python
{
    "episode_id": int,              # 全局唯一 id, 与 LIBERO traj_NNNN 对齐
    "task_name": str,               # "libero_long"
    "T": int,                       # episode 长度(帧数)
    "gamma_geo": Tensor,            # shape (T,), dtype float32, range [0, 1]
    "gamma_sem": Tensor,            # shape (T,), dtype float32, range [0, 1]
    "valid_mask": Tensor,           # shape (T,), dtype bool, False = 不可信(数据缺失帧)
    "meta": {
        "strategy": str,            # "per_task_q75"
        "base_policy": str,         # "pi05" / "chime_early_ckpt" / etc
        "delta_set": list[int],     # [4, 16] 或 [4, 16, 64]
        "saliency_method": str,     # "EAGN" / "exact_jacobian"
        "computed_at": str,         # ISO timestamp
        "hindsight_commit": str,    # git sha of Hindsight repo
    }
}
```

**保证**:
- `gamma_geo[t] + gamma_sem[t]` 没有约束(两条独立)
- `valid_mask` 在 Hindsight 数据缺失或 NaN 时为 False;CHIME-VLA 在算 L_HCS 时必须用此 mask 过滤
- `T` 与 LIBERO `traj_NNNN.h5` 的 `obs/agentview_rgb` 第一维必须相等;不等 → 数据对齐错误,raise

## 4. CHIME-VLA 端读取接口

文件:`src/chime_vla/hindsight/consumer.py`

```python
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class HindsightSample:
    gamma_geo: torch.Tensor      # (T,) float32
    gamma_sem: torch.Tensor      # (T,) float32
    valid_mask: torch.Tensor     # (T,) bool
    episode_id: int
    meta: dict

class HindsightConsumer:
    def __init__(self, root: Path, strategy: str = "per_task_q75",
                 task: str = "libero_long"):
        self.dir = root / strategy / task
        if not self.dir.exists():
            raise FileNotFoundError(f"Hindsight gamma_hat not found at {self.dir}. "
                                    f"Run Hindsight scripts/05-07 first.")

    def load(self, episode_id: int) -> HindsightSample: ...

    def has(self, episode_id: int) -> bool: ...

    def list_available(self) -> list[int]: ...
```

**调用约定**:
- DataModule 端在 `__getitem__` 时同步加载该 episode 的 `gamma_hat`
- 加载失败(文件不存在 / schema mismatch)→ 显式 raise,不静默 skip
- consumer 不缓存(一次性读取,LIBERO 一个 episode `.pt` < 10KB)

## 5. 使能 / 失能

- **M1 E1 PASS / SOFT-PASS** → consumer 启用,L_HCS 接通(λ_1 schedule 启动)
- **M1 E1 HARD FAIL** → consumer 不加载,L_HCS 永久 0(λ_1 = 0 锁定)。config flag:`hindsight.enabled = False`

## 6. Hindsight 端如何产出

参考 Hindsight 仓的:
- `Hindsight/PLAN.md`(整体 9 阶段 pipeline)
- `Hindsight/scripts/05_compute_saliency.py`(EAGN / exact saliency 计算)
- `Hindsight/scripts/06_compute_thresholds.py`(per-task quantile-75 阈值)
- `Hindsight/scripts/07_label_gamma_hat.py`(按阈值二值化 + 写出最终 `gamma_hat`)

CHIME-VLA 不重新实现这条 pipeline,只指向 Hindsight 的产物目录。

## 7. 版本兼容性

- Hindsight `meta.strategy` 字符串发生变化 → CHIME-VLA config 跟着改
- `meta.hindsight_commit` 不一致(同一 episode 多次 recompute)→ CHIME-VLA 取最新文件,记录到 PROGRESS.md
- schema 字段新增不影响读取(consumer 用 `dict.get`);schema 字段删除 → BLOCK,人工审查

## 8. 单测

`tests/test_hindsight_consumer.py`:
- 假数据:写一个合成 `ep_000000.pt`,验证读取 + shape 匹配 + dtype 正确
- 缺失数据:`load(99999999)` 应 raise FileNotFoundError
- schema 异常:写一个缺 `gamma_geo` 字段的 `.pt`,验证 raise
