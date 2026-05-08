"""Hydra structured-config schema for CHIME-VLA (CODE_STRUCTURE.md §2).

Single source of truth for component / loss / training / data / hindsight
configuration.  Every Python module that takes a config takes one of the
``CnConfig`` dataclasses below; the top-level ``ChimeConfig`` is what Hydra
loads / overrides via YAML.

The ``register_config()`` helper registers ``ChimeConfig`` (and its leaf
dataclasses) with Hydra's ``ConfigStore`` so that ``configs/default.yaml``
can use ``defaults: - base/chime`` etc. without separate Python imports
in every script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ===== Component configs (C1..C12) =====


@dataclass
class C1Config:
    """[C1] VLM backbone (SigLIP-ViT + LoRA)."""

    backbone: str = "siglip_vit_b"  # "siglip_vit_l" / "siglip_vit_b" / "siglip_vit_s"
    lora_r: int = 16
    freeze_backbone: bool = True  # MVP default freeze, full unfreeze


@dataclass
class C2Config:
    """[C2] FIFO ring buffer (M_work)."""

    K_w: int = 8  # FIFO length
    d_h: int = 1152  # token dim
    N: int = 256  # tokens per frame


@dataclass
class C3Config:
    """[C3] Geometric write head (delta-rule scatter into M_geo)."""

    voxel_proj_hidden: int = 256
    write_levels: list[int] = field(default_factory=lambda: [16])  # MVP=[16], full=[8,16,32]


@dataclass
class C4Config:
    """[C4] Semantic write head (slot-routed delta-rule into M_sem)."""

    qv_proj_hidden: int = 256
    softmax_temp: float = 0.5
    # M6 ablation 5: free-slot logit penalty. 1e9 = D5 修订规约;
    # 0.0 = naive zero-fill (无 slot_free mask 隔离).
    slot_free_penalty: float = 1.0e9


@dataclass
class C5Config:
    """[C5] ESPC ψ + EMA + projections (geo / sem 双通道 γ)."""

    psi_layers: int = 1
    use_gru: bool = True  # MVP=True (GRU), full=False (1-layer transformer)
    d_proj: int = 64  # geo_proj / sem_proj output dim
    ema_coeff: float = 0.99
    ema_warmup_steps: int = 2000
    sigmoid_temp: float = 1.0
    K_w: int = 8  # mirrors C2.K_w; ESPC needs to know FIFO depth


@dataclass
class C6Config:
    """[C6] Geometric voxel grid M_geo."""

    levels: list[int] = field(default_factory=lambda: [16])  # MVP single resolution
    d_g: int = 64  # per-voxel dim
    alpha_l: list[float] = field(default_factory=lambda: [1.0])  # MVP single level alpha=1.0
    workspace_bounds: list[float] = field(
        default_factory=lambda: [-1.0, 1.0, -1.0, 1.0, 0.0, 1.0]
    )  # x_min x_max y_min y_max z_min z_max


@dataclass
class C7Config:
    """[C7] Semantic slot bank M_sem (slot_free aware)."""

    K_s: int = 64
    d_s: int = 256
    evict_strategy: str = "timestamp"  # MVP="timestamp", M3+="csm_lru"


@dataclass
class C8Config:
    """[C8] Read interface (cross-attn + trilinear sampling)."""

    N_q: int = 16
    N_geo_q: int = 16
    use_kv_cache: bool = True  # inference optimisation


@dataclass
class C9Config:
    """[C9] Action expert (π0 flow-matching head)."""

    head: str = "flow_matching_pi0"
    lora_r: int = 16
    one_step_distill: bool = True  # MVP=True, full=False (4-8 ODE steps)
    freeze_base: bool = True


@dataclass
class C10Config:
    """[C10] HCS-H (offline, lives in Hindsight repo).  Mirrored for config provenance."""

    enabled: bool = True  # E1 fail → False
    base_policy: str = "pi05"  # "pi05" / "chime_early_ckpt"
    deltas: list[int] = field(default_factory=lambda: [4, 16])  # full adds 64
    rudder_dim: int = 256
    saliency_method: str = "EAGN"  # "EAGN" / "exact_jacobian"


@dataclass
class C11Config:
    """[C11] Predictive Read Head (training-only)."""

    horizons: list[int] = field(default_factory=lambda: [4, 16, 64])
    alpha_a: float = 1.0  # action loss weight
    pred_mlp_hidden: int = 512


@dataclass
class C12Config:
    """[C12] Counterfactual Slot Mask (training-only)."""

    n_slots_per_step: int = 4
    beta: float = 0.1  # log-mean term coefficient


# ===== Loss / training / data / hindsight =====


@dataclass
class LossConfig:
    """Loss weights and λ_1 schedule (CODE_STANDARDS §1.5)."""

    lambda_1_target: float = 0.3  # final L_HCS weight
    lambda_1_schedule: str = "anneal_post_e1"  # "anneal_post_e1" / "constant" / "off"
    step_e1_pass: int = 0  # E1 PASS step; 0 = anneal from start
    anneal_steps: int = 5000  # 0 → λ_1_target linear anneal length
    lambda_2: float = 0.5  # L_PRH
    lambda_3: float = 0.1  # L_CSM
    lambda_4: float = 0.0  # L_GC, MVP off
    lambda_ent: float = 0.01  # L_aux
    # M2 MVP fallback (§0.7.4): self-supervised L_predict for [C5] ψ.
    # Default 0.0 keeps pre-M2 configs identical (legacy YAMLs leave it off);
    # m2_mvp_fallback.yaml overrides to 1.0 to give ψ its only grad signal
    # once L_HCS is permanently zero.
    lambda_predict: float = 0.0
    entropy_floor: float = 1.0  # SG-7 monitor threshold


@dataclass
class TrainConfig:
    """Optimiser / DDP / BPTT settings."""

    lr: float = 1e-4
    bs: int = 24  # per-rank
    precision: str = "bf16-mixed"
    max_epochs: int = 5
    grad_clip: float = 1.0
    warmup_steps: int = 500
    optimizer: str = "adamw"
    betas: tuple[float, float] = (0.9, 0.95)
    wd: float = 0.01
    bptt_truncate: int = 32  # detach memory state every N steps
    grad_ckpt: bool = True
    accumulate: int = 1


@dataclass
class DataConfig:
    """LIBERO-Long dataset paths and shapes."""

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
    """File-protocol pointer to Hindsight γ̂ output (decoupled, see §1.6)."""

    enabled: bool = True  # E1 fail → False
    gamma_hat_root: str = (
        "/home/sqmluser/workspace/theaj/CHIME-VLA/Hindsight/output/saliency/gamma_hat"
    )
    strategy: str = "per_task_q75"
    task: str = "libero_long"


@dataclass
class ChimeConfig:
    """Top-level CHIME-VLA structured config (Hydra root)."""

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


# ===== Hydra ConfigStore registration =====


def register_config() -> None:
    """Register ``ChimeConfig`` and leaf dataclasses with Hydra's ConfigStore.

    Idempotent — safe to call multiple times.  Call at the top of each
    script entry-point before ``@hydra.main``.
    """
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="chime_config_schema", node=ChimeConfig)
    # Leaf nodes — useful for ``+c5=...`` overrides that point at a YAML.
    cs.store(group="c1", name="schema", node=C1Config)
    cs.store(group="c2", name="schema", node=C2Config)
    cs.store(group="c3", name="schema", node=C3Config)
    cs.store(group="c4", name="schema", node=C4Config)
    cs.store(group="c5", name="schema", node=C5Config)
    cs.store(group="c6", name="schema", node=C6Config)
    cs.store(group="c7", name="schema", node=C7Config)
    cs.store(group="c8", name="schema", node=C8Config)
    cs.store(group="c9", name="schema", node=C9Config)
    cs.store(group="c10", name="schema", node=C10Config)
    cs.store(group="c11", name="schema", node=C11Config)
    cs.store(group="c12", name="schema", node=C12Config)
    cs.store(group="loss", name="schema", node=LossConfig)
    cs.store(group="train", name="schema", node=TrainConfig)
    cs.store(group="data", name="schema", node=DataConfig)
    cs.store(group="hindsight", name="schema", node=HindsightConfig)


__all__ = [
    "C1Config",
    "C2Config",
    "C3Config",
    "C4Config",
    "C5Config",
    "C6Config",
    "C7Config",
    "C8Config",
    "C9Config",
    "C10Config",
    "C11Config",
    "C12Config",
    "LossConfig",
    "TrainConfig",
    "DataConfig",
    "HindsightConfig",
    "ChimeConfig",
    "register_config",
]
