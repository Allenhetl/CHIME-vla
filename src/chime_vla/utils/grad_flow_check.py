"""SG-1..SG-7 runtime verifier helpers (CODE_STANDARDS §1.1).

These helpers back ``tests/test_grad_flow.py`` and can also be wired
into the LightningModule as periodic assertions during training.

The contract list (per ``docs/grad_flow_contract.md``):

* SG-1: γ_geo / γ_sem ↛ ψ via L_main path (write heads receive sg(γ))
* SG-2: PRH path query projections ↛ perception (caller passes sg(m_t))
* SG-3: M_geo / M_sem delta-rule writes ↛ M_work history beyond bptt_truncate
* SG-4: CSM probes use frozen [C9] (no_grad)
* SG-5: L_HCS BCE target γ̂ is sg'd
* SG-6: M_sem.k frozen-random keys ↛ projections (no grad through k)
* SG-7: attention entropy over M_work above ``loss.entropy_floor``
        — soft monitor, not strict gate.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor


def assert_no_grad(
    tensor: Tensor,
    *,
    name: str,
    sg_id: str,
) -> None:
    """Assert ``tensor.requires_grad`` is False (post-sg() boundary check).

    Args:
        tensor: tensor to check.
        name:   human-readable identifier (e.g. "gamma_geo").
        sg_id:  SG-N tag for the failing assertion message.

    Raises:
        AssertionError if ``tensor.requires_grad`` is True.
    """
    if tensor.requires_grad:
        raise AssertionError(
            f"[{sg_id}] tensor '{name}' has requires_grad=True; "
            f"expected sg(.) boundary."
        )


def assert_grad_blocked_through(
    output: Tensor,
    inputs: Iterable[Tensor],
    *,
    sg_id: str,
) -> None:
    """Assert that ``output.backward()`` produces NO grad on any of ``inputs``.

    Runs a one-shot ``torch.autograd.grad`` with ``allow_unused=True`` and
    asserts every result is None or zero.

    Use only in tests / one-shot CI checks — performs a real backward.
    """
    raise NotImplementedError("grad_flow_check.assert_grad_blocked_through — M0 stub")


def assert_attn_entropy_floor(entropy: Tensor, floor: float, *, sg_id: str = "SG-7") -> None:
    """Soft monitor: log a warning if mean entropy < floor.

    M0 stub: numeric check intentionally not performed yet (forward not
    implemented).
    """
    raise NotImplementedError("grad_flow_check.assert_attn_entropy_floor — M0 stub")


def list_trainable_modules(model: nn.Module) -> list[tuple[str, int]]:
    """Return ``[(qualified_name, num_trainable_params)]`` for every submodule.

    Cheap reflection helper used by tests to confirm freeze contracts
    (e.g. SigLIP body should have 0 trainable when ``cfg.c1.freeze_backbone``
    is True).
    """
    out: list[tuple[str, int]] = []
    for name, module in model.named_modules():
        n = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        if n > 0:
            out.append((name, n))
    return out
