"""[C9] Action Expert -- pi0 flow-matching head + LoRA.

Component map: C9 (action, deploy + train).  Consumes the readout
``c_t`` plus a CLS-pooled current-frame embedding, and emits an 8-DoF
action via flow matching.

Modes:
    * MVP / cfg.one_step_distill=True -- single-step (consistency-distilled)
      action head: a direct MLP regressor over ``[mean(c_t) || h_t_cls]``.
      This corresponds to the section 0.7.4 / IMPLEMENTATION_PLAN MVP path
      ("pi0 1-step inference").  No iterative ODE; the head learns the
      implicit map from context to integrated action a*.
    * Full / cfg.one_step_distill=False -- 4-8 ODE steps with
      EulerMaruyama sampler (M3+ work; raises NotImplementedError here).

LoRA / freeze contract:
    * cfg.freeze_base=True (MVP default) -- the core projection is frozen
      from init except for a LoRA rank-r adapter on the hidden Linear; the
      adapter plus the action-head bias are trainable.
    * :meth:`freeze` -- freezes *every* base parameter.  LoRA adapter
      parameters remain trainable so the frozen [C9] snapshot used by
      [C12] CSM can still be fine-tuned independently from the live policy.
      (Used by L_PRH-only training paths and by CSM leave-one-out probes.)

dtype path (CODE_STANDARDS section 1.7):
    inputs c_t / h_t_cls bf16 -> internal MLP runs under bf16 autocast on
    CUDA, plain float32 on CPU (autocast is a no-op there) -> final action
    is cast to fp32 because actions are physical quantities and the
    flow-matching literature standardises on fp32 outputs (numerical
    stability; downstream loss runs in fp32 per CODE_STANDARDS section 1.7).

Smoke test (per task spec):

    >>> import torch
    >>> from chime_vla.config import ChimeConfig
    >>> from chime_vla.action.action_expert import ActionExpert
    >>> cfg = ChimeConfig()
    >>> ae = ActionExpert(cfg.c9, cfg.c2.d_h, cfg.data.action_dim)
    >>> B = 2
    >>> c_t = torch.randn(B, cfg.c8.N_q + cfg.c2.K_w, cfg.c2.d_h,
    ...                   dtype=torch.bfloat16)
    >>> h_cls = torch.randn(B, cfg.c2.d_h, dtype=torch.bfloat16)
    >>> a = ae(c_t, h_cls)
    >>> a.shape, a.dtype
    (torch.Size([2, 8]), torch.float32)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from chime_vla.config import C9Config


class _LoRALinear(nn.Module):
    """Self-contained rank-r LoRA adapter wrapping an ``nn.Linear``.

    Implementation mirrors the helper in ``perception/vlm_backbone.py`` but
    is intentionally re-implemented here to keep the action package
    decoupled (no cross-package imports of private helpers, per the
    decoupling guidance in the task spec).

    y = base(x) + (alpha / r) * (x @ A^T) @ B^T
    where A is kaiming-uniform-init and B is zero-init so the adapter
    contribution is exactly zero at init (preserves base behaviour).
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.in_f = int(base.in_features)
        self.out_f = int(base.out_features)
        self.r = int(r)
        self.alpha = float(alpha if alpha is not None else r)
        self.scaling = self.alpha / max(1, self.r)
        # A: (r, in_f), B: (out_f, r).  Init A kaiming, B zero.
        self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_f))
        self.lora_B = nn.Parameter(torch.zeros(self.out_f, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B remains zero -> initial adapter delta == 0.

    def forward(self, x: Tensor) -> Tensor:
        # On CPU bf16 autocast is a no-op for nn.Linear, so the activation
        # dtype may not match the base weight dtype.  Cast the input down to
        # the base dtype for the base path; LoRA path stays in activation
        # dtype to keep the adapter math in (typically) bf16 on CUDA.
        base_w_dtype = self.base.weight.dtype
        x_base = x.to(base_w_dtype) if x.dtype != base_w_dtype else x
        out = self.base(x_base).to(x.dtype)
        if self.r <= 0:
            return out
        # Cast LoRA params to activation dtype so bf16 autocast works on
        # CUDA and pure-fp32 paths on CPU also stay consistent.
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        delta = (x @ a.t()) @ b.t()
        return out + self.scaling * delta


class ActionExpert(nn.Module):
    """pi0 flow-matching action expert with LoRA adapters.

    MVP path (cfg.one_step_distill=True):
        a = head(GELU(lora_hidden([mean(c_t) ; h_t_cls])))

    Full path (cfg.one_step_distill=False):
        TODO(M3+): 4-8 step ODE flow-matching sampler.  Currently raises
        NotImplementedError.

    Args:
        cfg: :class:`C9Config` (head name, lora rank, distill flag,
            freeze_base flag).
        d_h: token dimension shared across [C2]/[C8] (typically 1152).
        action_dim: action vector dimension (LIBERO = 8).
    """

    def __init__(self, cfg: C9Config, d_h: int, action_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_h: int = int(d_h)
        self.action_dim: int = int(action_dim)
        self.head_kind: str = cfg.head
        self.lora_r: int = int(cfg.lora_r)
        self.one_step_distill: bool = bool(cfg.one_step_distill)
        self.freeze_base: bool = bool(cfg.freeze_base)

        # Hidden width is fixed at 512 per task spec / pi0-MVP convention.
        # (Could be promoted to C9Config in M3+ if hidden width tuning matters.)
        self.hidden_dim: int = 512

        # Trunk: [mean(c_t) ; h_t_cls] -> hidden -> action.
        # The "base" Linear(2*d_h, hidden) is the candidate for freezing
        # under cfg.freeze_base; LoRA adapter on top supplies trainable
        # capacity.  The output projection (hidden -> action_dim) is small
        # and always trainable -- per task spec ("action-head bias", and
        # because freezing it would render the frozen-base path unable to
        # produce well-calibrated actions during downstream re-training).
        base_hidden = nn.Linear(2 * self.d_h, self.hidden_dim)
        nn.init.xavier_uniform_(base_hidden.weight)
        nn.init.zeros_(base_hidden.bias)
        self.hidden = _LoRALinear(base_hidden, r=self.lora_r)

        self.act = nn.GELU()

        self.head = nn.Linear(self.hidden_dim, self.action_dim)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Apply freeze_base immediately if requested (MVP default = True):
        # base trunk weights frozen, LoRA + head trainable.
        if self.freeze_base:
            for p in self.hidden.base.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, c_t: Tensor, h_t_cls: Tensor) -> Tensor:
        """Predict the 8-DoF action.

        Args:
            c_t:    ``(B, N_q + K_w, d_h)`` bf16 -- readout context tokens.
            h_t_cls: ``(B, d_h)`` bf16 -- pooled current frame
                (typically ``h_t.mean(dim=1)``; see ``chime_train_step``
                pseudocode in CODE_STRUCTURE section 7).

        Returns:
            action: ``(B, action_dim=8)`` fp32 -- predicted action vector.
            (Direct regression target under MVP one-step distill; a
            velocity / final integrated action under full ODE mode.)
        """
        if c_t.dim() != 3:
            raise ValueError(
                "ActionExpert.forward expected c_t of shape (B, N_q+K_w, d_h); "
                f"got {tuple(c_t.shape)}"
            )
        if c_t.shape[-1] != self.d_h:
            raise ValueError(
                f"ActionExpert.forward: c_t last dim {c_t.shape[-1]} != d_h={self.d_h}"
            )
        if h_t_cls.dim() != 2 or h_t_cls.shape[-1] != self.d_h:
            raise ValueError(
                "ActionExpert.forward expected h_t_cls of shape (B, d_h); "
                f"got {tuple(h_t_cls.shape)}"
            )
        if c_t.shape[0] != h_t_cls.shape[0]:
            raise ValueError(
                "ActionExpert.forward: batch size mismatch between c_t "
                f"({c_t.shape[0]}) and h_t_cls ({h_t_cls.shape[0]})"
            )

        if not self.one_step_distill:
            # M3+ work -- 4-8 step ODE flow-matching sampler.
            raise NotImplementedError(
                "[C9] ActionExpert: 4-step ODE flow-matching is M3+ work; "
                "set cfg.c9.one_step_distill=True for the MVP path."
            )

        # Mean-pool c_t over (N_q + K_w) tokens to a single (B, d_h)
        # context vector, then concatenate with the CLS-pooled current
        # frame.  This is the simplest 1-step distill consistent with
        # pi0-style flow-matching MVP (architecture v2.1 [C9] card).
        device_type = "cuda" if c_t.is_cuda else "cpu"
        with torch.amp.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=(device_type == "cuda"),
        ):
            ctx = c_t.mean(dim=1)  # (B, d_h)
            x = torch.cat([ctx, h_t_cls.to(ctx.dtype)], dim=-1)  # (B, 2*d_h)
            h = self.act(self.hidden(x))                         # (B, hidden)
            # Final head also needs CPU dtype-coercion for the same reason
            # as ``_LoRALinear`` above (autocast is a no-op on CPU).
            head_w_dtype = self.head.weight.dtype
            h_cast = h.to(head_w_dtype) if h.dtype != head_w_dtype else h
            a = self.head(h_cast).to(h.dtype)                    # (B, action_dim)

        # Action returned in fp32 (CODE_STANDARDS section 1.7: actions are
        # physical quantities; flow-matching loss runs in fp32).
        return a.to(torch.float32)

    # ------------------------------------------------------------------
    # Freeze
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze every *base* parameter; keep LoRA adapter trainable.

        Used by:
            * [C12] CSM, which calls a frozen [C9] N times per step to
              probe slot importance (SG-4: natural sg through frozen [C9]).
            * L_PRH-only training paths that need a stable action head
              while [C11] PRH learns from m_t.

        Concretely:
            * ``hidden.base.{weight,bias}`` -- frozen
            * ``head.{weight,bias}``         -- frozen
            * ``hidden.lora_A``, ``hidden.lora_B`` -- still trainable
              (this lets the frozen-base [C9] snapshot retain a small
              tunable surface for downstream adapters; the CSM probe
              treats the *whole* module as a single forward callable so
              the LoRA-trainable surface is opaque to it).
        """
        for name, p in self.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
