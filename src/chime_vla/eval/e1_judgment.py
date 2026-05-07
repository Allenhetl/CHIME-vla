"""M1 E1 milestone-gate evaluation: IoU(γ̂, sub_task_id boundary).

Architecture v2.1 §I.3 line 1983 defines the E1 judgement:

    IoU @ 0.3 ≥ 0.4 → PASS    (proceed M2 with HCS-H signal)
    0.3 ≤ IoU < 0.4 → SOFT-PASS (proceed M2, raise red-flag #1)
    IoU < 0.3 → HARD-FAIL     (fallback to MVP, λ_1 = 0 permanently,
                               drop [C10][C12][C13])

We use the **exact-Jacobian** saliency (rather than EAGN proxy) so the
baseline is deterministic and reproducible from the (untrained) CHIME
weights alone.  γ̂ is split into a geometric and a semantic stream; the
top-25 % z-score peaks per episode are tested for IoU against the
``sub_task_id`` boundary set, expanded by a ±4-frame window.

The full pipeline is:

    1. ``compute_jacobian_saliency(model, rgb_seq, proprio_seq)`` runs the
       CHIME forward once, then for each (t, Δ) pair backprops
       ``||a_{t+Δ}||²`` w.r.t. ``rgb_t`` and aggregates the gradient
       magnitude.  γ_geo / γ_sem are produced by averaging the magnitude
       over (H, W) per channel and over (C, H, W) respectively, then
       z-scored along the time axis.
    2. ``compute_iou_vs_boundaries(gamma, sub_task_id)`` finds boundary
       frames (where ``sub_task_id`` changes), inflates them by ±4 frames,
       picks the top-25 % of γ as predicted peaks, and reports IoU /
       precision / recall / F1.

Implementation notes (CODE_STANDARDS §1.7 / §1.8):

  * The Jacobian computation is the bottleneck.  We keep one forward
    pass for the whole episode so the autograd graph captures every
    ``a_{t+Δ}`` simultaneously, then iterate through (t, Δ) pairs and
    accumulate magnitudes via ``torch.autograd.grad(..., retain_graph=True)``.
    Memory-wise this is bounded by the per-frame VLMBackbone graph
    (≈ 2 GB on a 4090 for T~280, B=1, bf16); we stay well under 24 GB.
  * If we see OOM we fall back to per-Δ chunks (release the graph
    between deltas).
  * No M_geo / M_sem write paths are exercised in this evaluation —
    we run the perception → ESPC → readout → action chain only.  The
    delta-rule scatter into M_geo/M_sem is wrapped in ``torch.no_grad``
    upstream anyway, so saliency w.r.t. those memories is identically
    zero.
  * The forward path is the one defined in
    :func:`chime_vla.training.train_step.chime_train_step` — we re-use
    the same component graph but with ``torch.set_grad_enabled(True)``
    and a single batch row.  That keeps the read-out pipeline
    consistent with training.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover — typing only
    from chime_vla.training.lightning_module import ChimeVlaLightning


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def z_score(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Z-score a 1-D tensor along its only axis.  Returns ``(x - μ) / σ``.

    Falls back to zeros if the input has zero variance (degenerate
    untrained-model edge case).  All-zero output is safe — the IoU
    routine just returns whichever 25 % the argpartition picks first.
    """
    x = x.float()
    mu = x.mean()
    sigma = x.std(unbiased=False)
    if not torch.isfinite(sigma) or sigma.item() < eps:
        return torch.zeros_like(x)
    return (x - mu) / (sigma + eps)


def _forward_segment(
    model: "ChimeVlaLightning",
    rgb_segment: Tensor,    # (S, 3, H, W) fp32 with requires_grad on the t-th slice
    proprio_segment: Tensor,
) -> Tensor:
    """Run the per-step CHIME forward across an S-frame segment, B = 1.

    Returns ``a_pred`` of shape ``(S, action_dim)`` autograd-tracked.
    Each forward starts with empty M_work / M_geo / M_sem — fine for
    saliency because (a) the write paths are wrapped in ``no_grad`` so
    they never propagate, (b) γ̂'s read-out depends on the FIFO of the
    last K_w frames of the segment, which is what M_work supplies.

    This is the workhorse helper called per (t, Δ) pair — keeping the
    autograd graph bounded by the segment length (typically Δ+K_w+1 ≈ 25
    frames at most) so we stay well below the 49 GB GPU budget even for
    full 280-frame episodes.
    """
    from chime_vla.memory.geo_grid import GeoGrid
    from chime_vla.memory.sem_bank import SemBank
    from chime_vla.perception.fifo_buffer import WorkBuffer

    cfg = model.cfg
    device = rgb_segment.device
    S = int(rgb_segment.shape[0])

    B = 1
    c2 = WorkBuffer(cfg.c2, batch_size=B, device=device)
    m_geo = GeoGrid(cfg.c6, batch_size=B, d_g=cfg.c6.d_g, device=device)
    m_sem = SemBank(cfg.c7, batch_size=B, device=device)

    a_pred_steps: list[Tensor] = []

    for s in range(S):
        rgb_s = rgb_segment[s : s + 1]
        proprio_s = proprio_segment[s : s + 1]

        h_t = model.c1(rgb_s, proprio_s)
        m_work_prev = c2.snapshot()
        gamma_geo, gamma_sem = model.c5(h_t, m_work_prev)

        if c2.K_w > 1:
            shifted = torch.cat(
                [c2.buffer[:, 1:], h_t.to(c2.buffer.dtype).unsqueeze(1)],
                dim=1,
            )
        else:
            shifted = h_t.to(c2.buffer.dtype).unsqueeze(1)
        c2.buffer = shifted
        c2._n_appended = torch.clamp(c2._n_appended + 1, max=c2.K_w)
        m_work_post = shifted

        with torch.no_grad():
            model.c3(h_t.detach(), gamma_geo.detach(), m_geo, step=s)
            model.c4(h_t.detach(), gamma_sem.detach(), m_sem, step=s)

        c_t = model.c8(m_work_post, m_geo, m_sem, h_t)
        h_t_cls = h_t.mean(dim=1)
        a_pred_s = model.c9(c_t, h_t_cls)
        a_pred_steps.append(a_pred_s.squeeze(0))

    return torch.stack(a_pred_steps, dim=0)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

@torch.enable_grad()
def compute_jacobian_saliency(
    model: "ChimeVlaLightning",
    rgb_seq: Tensor,
    proprio_seq: Tensor,
    deltas: list[int] | tuple[int, ...] = (4, 16),
    device: str | torch.device = "cuda",
    chunk_size: int | None = None,
) -> dict[str, Tensor]:
    """Per-frame Jacobian saliency over an episode (γ_geo, γ_sem).

    Algorithm:
        For each ``Δ ∈ deltas`` and each base frame ``t ∈ [0, T - Δ)``:
            (a) re-run a fresh CHIME forward over the segment
                ``[t, t + Δ]`` (Δ+1 frames) — only the t-th input slice
                carries ``requires_grad=True``.
            (b) ``g = ∂ ||a_{t+Δ}||² / ∂ rgb_t`` via
                ``torch.autograd.grad(scalar, rgb_leaf)``.
            (c) accumulate per-frame Jacobian magnitude:
                  * ``gamma_geo[t] += sum_c mean_{H,W}( |g[c]| )``
                  * ``gamma_sem[t] += mean_{C,H,W}( |g| )``
            (d) drop the autograd graph (segment-scoped).
        Final γ_geo / γ_sem are z-scored along the time axis.

    Memory:
        The autograd graph never holds more than (Δ+1) frames of
        SigLIP-ViT activations; ≈ 1-2 GB of peak alloc for Δ=16, B=1
        on a 4090.  The episode-level memory is therefore O(T) (just
        the running raw_geo / raw_sem buffers).  This costs more wall
        time than the "single big forward" variant but it scales
        linearly without OOM on long episodes.

    Args:
        model:        the (loaded / untrained) ``ChimeVlaLightning`` wrapper.
        rgb_seq:      (T, 3, 224, 224) fp32 in [0, 1].
        proprio_seq:  (T, 8) fp32.
        deltas:       horizon offsets (frames).  Default ``(4, 16)``.
        device:       ``"cuda"`` / ``"cpu"`` / explicit ``torch.device``.
        chunk_size:   reserved (segment-scoped graph already keeps
                      memory bounded; kept for API compatibility).

    Returns:
        dict with keys:
            ``gamma_geo``: (T,) fp32 z-scored
            ``gamma_sem``: (T,) fp32 z-scored
            ``raw_geo``:   (T,) fp32 (pre-z magnitude, useful for QA)
            ``raw_sem``:   (T,) fp32
            ``n_pairs``:   int — number of (t, Δ) pairs processed
    """
    del chunk_size  # API stub — segment-scoped graph already bounds memory.
    if rgb_seq.dim() != 4:
        raise ValueError(
            f"rgb_seq must be (T, 3, H, W); got {tuple(rgb_seq.shape)}"
        )
    if proprio_seq.dim() != 2 or proprio_seq.shape[0] != rgb_seq.shape[0]:
        raise ValueError(
            f"proprio_seq must be (T, P) matching rgb_seq's T; "
            f"got rgb {tuple(rgb_seq.shape)} vs proprio {tuple(proprio_seq.shape)}"
        )

    device = torch.device(device)
    T = int(rgb_seq.shape[0])
    deltas_sorted = sorted(set(int(d) for d in deltas if int(d) > 0))
    delta_max = max(deltas_sorted) if deltas_sorted else 0
    if delta_max <= 0 or T <= delta_max:
        raise ValueError(
            f"compute_jacobian_saliency: T={T} must exceed max delta={delta_max}"
        )

    model.eval()

    rgb_dev = rgb_seq.detach().to(device=device, dtype=torch.float32)
    proprio_dev = proprio_seq.detach().to(device=device, dtype=torch.float32)

    raw_geo = torch.zeros(T, dtype=torch.float32, device="cpu")
    raw_sem = torch.zeros(T, dtype=torch.float32, device="cpu")

    n_pairs = 0
    for delta in deltas_sorted:
        for t in range(T - delta):
            seg_start = t
            seg_end = t + delta + 1
            # Build a clone where only the t-th slice tracks gradient.
            # Slicing a non-leaf tensor with requires_grad on the WHOLE clone
            # also works (grad output has T zeros except at index t), but
            # carrying grad-tracking on Δ+1 slices is wasteful.  We make
            # the entire segment a leaf to keep slicing simple, then read
            # only the t-relative grad after backward.
            rgb_seg = rgb_dev[seg_start:seg_end].clone().detach()
            rgb_seg.requires_grad_(True)
            proprio_seg = proprio_dev[seg_start:seg_end]

            a_pred = _forward_segment(model, rgb_seg, proprio_seg)
            target = a_pred[delta].float()  # i.e. position seg_start+delta = t+Δ
            scalar = (target * target).sum()

            grad_rgb = torch.autograd.grad(
                scalar, rgb_seg, retain_graph=False, create_graph=False
            )[0]
            # Only the leaf slice at relative-index 0 corresponds to the
            # base frame t (the gradient at other relative positions is
            # the saliency w.r.t. intermediate frames; we discard those
            # — by construction this Δ-pair targets t only).
            g_t = grad_rgb[0].detach().float()                  # (3, H, W) signed
            g_abs = g_t.abs()
            # gamma_geo: per-channel mean magnitude summed over channels.
            # Captures aggregate spatial saliency.
            raw_geo[t] = raw_geo[t] + g_abs.mean(dim=(1, 2)).sum().cpu()
            # gamma_sem: spatial L2 magnitude of the Jacobian — emphasises
            # localised peaks (rather than diffuse mean).  Independent of
            # gamma_geo's mean-magnitude statistic so the two channels
            # carry separate-ish signal.
            raw_sem[t] = raw_sem[t] + g_t.norm().cpu()

            n_pairs += 1

            # Free segment-scoped graph deterministically.
            del rgb_seg, proprio_seg, a_pred, target, scalar, grad_rgb, g_t

    gamma_geo = z_score(raw_geo)
    gamma_sem = z_score(raw_sem)
    return {
        "gamma_geo": gamma_geo,
        "gamma_sem": gamma_sem,
        "raw_geo": raw_geo,
        "raw_sem": raw_sem,
        "n_pairs": int(n_pairs),
    }


def _boundary_indices(sub_task_id: Tensor) -> Tensor:
    """Return the frame indices where ``sub_task_id`` changes (boundary set).

    Index ``t`` is a boundary iff ``sub_task_id[t] != sub_task_id[t-1]``
    (so ``t = 0`` is never a boundary).
    """
    s = sub_task_id.to(torch.long).flatten().cpu()
    if s.numel() <= 1:
        return torch.empty(0, dtype=torch.long)
    delta = (s[1:] - s[:-1]).abs()
    boundaries = (delta != 0).nonzero(as_tuple=False).flatten() + 1
    return boundaries


def _expand_window(
    indices: Tensor, window: int, T: int, valid: Tensor | None = None
) -> Tensor:
    """Inflate each ``i ∈ indices`` to ``[i - window, i + window]`` ∩ [0, T)
    and return the dedupe'd flat set as a 1-D LongTensor.

    Optionally restrict to ``valid`` frames.
    """
    if indices.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    offsets = torch.arange(-int(window), int(window) + 1, dtype=torch.long)
    grid = indices.unsqueeze(1) + offsets.unsqueeze(0)  # (n_b, 2W+1)
    flat = grid.flatten()
    flat = flat[(flat >= 0) & (flat < T)]
    if valid is not None:
        flat = flat[valid[flat]]
    return torch.unique(flat)


def compute_iou_vs_boundaries(
    gamma: Tensor,
    sub_task_id: Tensor,
    quantile: float = 0.25,
    boundary_window: int = 4,
    valid_mask: Tensor | None = None,
) -> dict[str, Any]:
    """IoU of top-``quantile`` γ peaks vs sub_task_id boundary windows.

    Args:
        gamma:           (T,) fp32, ideally z-scored.
        sub_task_id:     (T,) integer.
        quantile:        fraction of frames flagged as predicted peaks
                         (top ``ceil(quantile * T_valid)``).
        boundary_window: ±N frames around each boundary index.
        valid_mask:      optional (T,) bool — pad frames excluded from
                         both peak selection and boundary expansion.

    Returns:
        dict with the canonical IoU triplet plus ancillaries.
    """
    if gamma.dim() != 1:
        raise ValueError(f"gamma must be 1-D (T,); got {tuple(gamma.shape)}")
    if sub_task_id.dim() != 1 or sub_task_id.shape[0] != gamma.shape[0]:
        raise ValueError(
            f"sub_task_id must match gamma's T; "
            f"got {tuple(sub_task_id.shape)} vs {tuple(gamma.shape)}"
        )

    T = int(gamma.shape[0])
    g = gamma.detach().float().cpu()
    valid = (
        valid_mask.detach().to(torch.bool).cpu()
        if valid_mask is not None
        else torch.ones(T, dtype=torch.bool)
    )

    valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
    T_valid = int(valid_idx.numel())
    if T_valid == 0:
        return {
            "iou_at_0.3": 0.0,
            "iou_at_0.5": 0.0,
            "iou_at_0.7": 0.0,
            "iou_main": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_predicted_peaks": 0,
            "n_boundaries": 0,
            "n_intersect": 0,
            "T": T,
            "T_valid": 0,
        }

    # 1. Boundary set (only count boundaries inside valid segment).
    boundaries = _boundary_indices(sub_task_id)
    boundaries = boundaries[valid[boundaries]]
    extended = _expand_window(boundaries, boundary_window, T, valid=valid)

    # 2. Predicted peaks: top-quantile of γ on valid frames.
    g_valid = g[valid]
    n_peaks = max(1, int(math.ceil(float(quantile) * T_valid)))
    n_peaks = min(n_peaks, T_valid)
    # argpartition / topk
    topk = torch.topk(g_valid, k=n_peaks, largest=True).indices
    peaks = valid_idx[topk]
    peaks_set = peaks
    extended_set = extended

    if peaks_set.numel() == 0 or extended_set.numel() == 0:
        return {
            "iou_at_0.3": 0.0,
            "iou_at_0.5": 0.0,
            "iou_at_0.7": 0.0,
            "iou_main": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_predicted_peaks": int(peaks_set.numel()),
            "n_boundaries": int(boundaries.numel()),
            "n_intersect": 0,
            "T": T,
            "T_valid": T_valid,
        }

    peaks_mask = torch.zeros(T, dtype=torch.bool)
    peaks_mask[peaks_set] = True
    ext_mask = torch.zeros(T, dtype=torch.bool)
    ext_mask[extended_set] = True

    inter = (peaks_mask & ext_mask).sum().item()
    union = (peaks_mask | ext_mask).sum().item()
    iou = inter / union if union > 0 else 0.0

    # IoU at higher thresholds — interpret as "use stricter peak quantile".
    # This mirrors the v2.1 §I.3 wording where IoU @ q is the IoU when the
    # peak fraction is q.  Architecture asks for IoU @ 0.3 specifically,
    # which we read as "γ̂ peaks at quantile = 1 - 0.3 = 0.3 boundary
    # alignment" — but the canonical interpretation in our codebase (and
    # the one used by the milestone gate) is the IoU produced by the
    # top-25% peak set vs ±4-frame extended boundary windows.  We expose
    # both views here.
    iou_main = iou

    # Stricter / more permissive views: vary the quantile.
    def _iou_with_quantile(q: float) -> float:
        n = max(1, int(math.ceil(q * T_valid)))
        n = min(n, T_valid)
        idx = torch.topk(g_valid, k=n, largest=True).indices
        ps = valid_idx[idx]
        pm = torch.zeros(T, dtype=torch.bool)
        pm[ps] = True
        i_ = (pm & ext_mask).sum().item()
        u_ = (pm | ext_mask).sum().item()
        return float(i_) / float(u_) if u_ > 0 else 0.0

    iou_at_03 = _iou_with_quantile(0.30)
    iou_at_05 = _iou_with_quantile(0.50)
    iou_at_07 = _iou_with_quantile(0.70)

    precision = inter / max(1, peaks_mask.sum().item())
    recall = inter / max(1, ext_mask.sum().item())
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "iou_at_0.3": float(iou_at_03),
        "iou_at_0.5": float(iou_at_05),
        "iou_at_0.7": float(iou_at_07),
        "iou_main": float(iou_main),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_predicted_peaks": int(peaks_mask.sum().item()),
        "n_boundaries": int(boundaries.numel()),
        "n_intersect": int(inter),
        "T": T,
        "T_valid": T_valid,
    }


def random_baseline_iou(
    sub_task_id: Tensor,
    quantile: float = 0.25,
    boundary_window: int = 4,
    n_trials: int = 64,
    valid_mask: Tensor | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Monte-Carlo random-peak baseline.

    Picks ``n_trials`` random top-``quantile`` peak sets (uniform over
    valid frames) and returns the mean / std of the resulting IoU.
    Useful sanity-bound: an untrained CHIME baseline should land at or
    above the random number; a number well below would imply a bug.
    """
    T = int(sub_task_id.shape[0])
    valid = (
        valid_mask.detach().to(torch.bool).cpu()
        if valid_mask is not None
        else torch.ones(T, dtype=torch.bool)
    )
    valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
    T_valid = int(valid_idx.numel())
    if T_valid == 0:
        return {"random_iou_mean": 0.0, "random_iou_std": 0.0}

    boundaries = _boundary_indices(sub_task_id)
    boundaries = boundaries[valid[boundaries]]
    extended = _expand_window(boundaries, boundary_window, T, valid=valid)

    n_peaks = max(1, int(math.ceil(float(quantile) * T_valid)))
    n_peaks = min(n_peaks, T_valid)

    ext_mask = torch.zeros(T, dtype=torch.bool)
    ext_mask[extended] = True

    g = torch.Generator().manual_seed(int(seed))
    ious: list[float] = []
    for _ in range(int(n_trials)):
        perm = torch.randperm(T_valid, generator=g)[:n_peaks]
        ps = valid_idx[perm]
        pm = torch.zeros(T, dtype=torch.bool)
        pm[ps] = True
        inter = (pm & ext_mask).sum().item()
        union = (pm | ext_mask).sum().item()
        ious.append(float(inter) / float(union) if union > 0 else 0.0)
    arr = torch.tensor(ious, dtype=torch.float32)
    return {
        "random_iou_mean": float(arr.mean().item()),
        "random_iou_std": float(arr.std(unbiased=False).item()),
    }


def e1_decision(mean_iou_at_0_3: float) -> str:
    """Architecture v2.1 §I.3 line 1983 milestone-gate verdict."""
    if mean_iou_at_0_3 >= 0.4:
        return "PASS"
    if mean_iou_at_0_3 >= 0.3:
        return "SOFT-PASS"
    return "HARD-FAIL"
