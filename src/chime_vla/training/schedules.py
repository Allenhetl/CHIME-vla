"""λ_1 (L_HCS weight) schedule (CODE_STANDARDS §1.5).

Step-aware, three-segment piecewise function:

    step < step_e1_pass                          → 0
    step_e1_pass ≤ step < step_e1_pass + N       → linear 0 → λ_target
    step ≥ step_e1_pass + N                      → λ_target

where N = ``cfg.anneal_steps`` (default 5000) and λ_target =
``cfg.lambda_1_target`` (default 0.3).

Three schedule modes (selected by ``cfg.lambda_1_schedule``):
    * ``"anneal_post_e1"`` — the piecewise schedule above (default)
    * ``"constant"``       — return ``λ_target`` regardless of step
    * ``"off"``            — return 0.0 regardless of step (E1 HARD FAIL fallback)

This is the **only** non-stub function in the M0 skeleton (it is a few
lines of arithmetic and is needed by tests already at M0).
"""

from __future__ import annotations

from chime_vla.config import LossConfig


def lambda_1_schedule(step: int, cfg: LossConfig) -> float:
    """Return the L_HCS weight at the given training step.

    Args:
        step: current global training step (>= 0).
        cfg:  :class:`LossConfig` with the schedule parameters.

    Returns:
        λ_1 ∈ [0, lambda_1_target] (float).

    Raises:
        ValueError: if ``cfg.lambda_1_schedule`` is unknown.
    """
    mode = cfg.lambda_1_schedule
    target = float(cfg.lambda_1_target)

    if mode == "off":
        return 0.0
    if mode == "constant":
        return target
    if mode != "anneal_post_e1":
        raise ValueError(
            f"unknown lambda_1_schedule mode '{mode}'; "
            "expected one of 'anneal_post_e1' / 'constant' / 'off'."
        )

    start = int(cfg.step_e1_pass)
    n = max(1, int(cfg.anneal_steps))
    if step < start:
        return 0.0
    if step >= start + n:
        return target
    # linear ramp 0 → target over n steps
    frac = (step - start) / n
    return target * frac
