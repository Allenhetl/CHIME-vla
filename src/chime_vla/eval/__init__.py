"""Evaluation utilities for CHIME-VLA milestone gates.

Currently exposes the M1 E1 milestone-gate metric:
``IoU(γ̂, sub_task_id boundary)`` (see architecture v2.1 §I.3 line 1983,
plus :mod:`chime_vla.eval.e1_judgment`).
"""

from chime_vla.eval.e1_judgment import (
    compute_iou_vs_boundaries,
    compute_jacobian_saliency,
    e1_decision,
    random_baseline_iou,
    z_score,
)

__all__ = [
    "compute_jacobian_saliency",
    "compute_iou_vs_boundaries",
    "random_baseline_iou",
    "e1_decision",
    "z_score",
]
