"""[C2] FIFO ring buffer — M_work (CODE_STRUCTURE §3.2).

Component map: C2 (perception, deploy + train).  Maintains a fixed-length
ring of the last ``K_w`` per-frame token tensors ``h_t``.  No learnable
parameters — implemented as a bare class holding a ``torch.Tensor`` so it
is *not* an ``nn.Module`` (the contract: append is in-place, no grad
through state on overflow boundaries; explicit ``detach`` is the caller's
job).

Batch contract (CODE_STANDARDS §1.2):
    M_work : (B, K_w, N, d_h) bf16

Episode boundary semantics (CODE_STANDARDS §1.2):
    Caller must invoke :meth:`reset` at episode start; otherwise stale
    state from the previous episode bleeds across.  See
    ``chime_vla.utils.memory_reset.reset_memory``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from chime_vla.config import C2Config


class WorkBuffer:
    """FIFO ring of the last K_w frame tokens.

    Not an ``nn.Module`` — buffers are tensors held as instance attributes.
    Compatible with autograd: write-time append participates in the
    computation graph until the caller explicitly detaches (BPTT truncate).

    Layout convention (CODE_STANDARDS §1.2):
        ``buffer[b, K_w-1]`` is the most-recent frame; ``buffer[b, 0]`` is the
        oldest.  Empty slots prior to ``K_w`` valid appends are zero (they
        sit in the leading positions, e.g. after one append only
        ``buffer[b, K_w-1]`` is non-zero).
    """

    def __init__(self, cfg: C2Config, batch_size: int, device: torch.device | str):
        self.cfg = cfg
        self.K_w: int = cfg.K_w
        self.N: int = cfg.N
        self.d_h: int = cfg.d_h
        self.B: int = batch_size
        self.device: torch.device = torch.device(device)

        # Ring storage; bf16 per CODE_STANDARDS §1.7 (C2 = bf16).
        self.buffer: Tensor = torch.zeros(
            (self.B, self.K_w, self.N, self.d_h),
            dtype=torch.bfloat16,
            device=self.device,
        )
        # Number of valid frames already pushed in [0, K_w].  Once equal to
        # K_w the ring is full and append rotates.  Tracked per-batch-slot so
        # that ``reset(batch_indices)`` can rewind a subset of episodes.
        self._n_appended: Tensor = torch.zeros(
            self.B, dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, batch_indices: Optional[Tensor] = None) -> None:
        """Zero-fill the ring (or selected episodes) and rewind ``n_appended``.

        Args:
            batch_indices: ``(B',)`` long tensor of batch slots to reset.  If
                None, every slot is reset (full episode boundary).
        """
        if batch_indices is None:
            self.buffer.zero_()
            self._n_appended.zero_()
            return

        # Subset reset: index_fill / scatter style.  Move the indices to our
        # device defensively in case the caller built them on a different one.
        idx = batch_indices.to(device=self.device, dtype=torch.long)
        self.buffer[idx] = 0
        self._n_appended[idx] = 0

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------
    def append(self, h_t: Tensor) -> Tensor:
        """Append the current frame to the ring; returns the *new* M_work view.

        Args:
            h_t: ``(B, N, d_h)`` token tensor for this frame.  Cast to bf16 to
                match buffer dtype if the caller hands us a different float
                type (autocast contexts may give us fp32).

        Returns:
            M_work: ``(B, K_w, N, d_h)`` bf16 — index ``K_w-1`` is the just-
            appended frame; index 0 is the oldest of the last ``K_w``.
        """
        if h_t.shape != (self.B, self.N, self.d_h):
            raise ValueError(
                f"WorkBuffer.append expected h_t of shape "
                f"({self.B}, {self.N}, {self.d_h}); got {tuple(h_t.shape)}"
            )

        h_bf16 = h_t.to(dtype=self.buffer.dtype, device=self.buffer.device)

        # Shift-left: drop the oldest slot, slide the rest forward by one.
        # ``clone()`` avoids the aliasing trap of slice-assignment on
        # overlapping ranges (PyTorch will warn / produce wrong values).
        if self.K_w > 1:
            self.buffer[:, : self.K_w - 1] = self.buffer[:, 1 : self.K_w].clone()

        # Most-recent frame goes to the last slot.
        self.buffer[:, self.K_w - 1] = h_bf16

        # Per-sample counter, capped at K_w.
        self._n_appended = torch.clamp(self._n_appended + 1, max=self.K_w)

        return self.buffer

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def snapshot(self) -> Tensor:
        """Return a copy of the current ring contents.

        Returns ``(B, K_w, N, d_h)`` bf16.  ``clone()`` so external mutations
        cannot bleed back into the internal state.
        """
        return self.buffer.clone()
