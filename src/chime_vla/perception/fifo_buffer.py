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

    M0: stub — see ``raise NotImplementedError`` in :meth:`append`.
    """

    def __init__(self, cfg: C2Config, batch_size: int, device: torch.device | str):
        self.cfg = cfg
        self.K_w: int = cfg.K_w
        self.N: int = cfg.N
        self.d_h: int = cfg.d_h
        self.B: int = batch_size
        self.device: torch.device = torch.device(device)

        # Allocated lazily by :meth:`reset`; M0 stub keeps placeholder shape.
        self.buffer: Tensor = torch.zeros(
            (self.B, self.K_w, self.N, self.d_h), dtype=torch.bfloat16, device=self.device
        )
        # Number of valid frames already pushed in [0, K_w].  Once equal to
        # K_w the ring is full and append rotates.
        self.write_ptr: Tensor = torch.zeros(self.B, dtype=torch.long, device=self.device)

    def reset(self, batch_indices: Optional[Tensor] = None) -> None:
        """Zero-fill the ring (or selected episodes) and rewind ``write_ptr``.

        Args:
            batch_indices: ``(B',)`` long tensor of batch slots to reset.  If
                None, every slot is reset (full episode boundary).
        """
        raise NotImplementedError("[C2] WorkBuffer.reset — M0 stub")

    def append(self, h_t: Tensor) -> Tensor:
        """Append the current frame to the ring; returns the *new* M_work view.

        Args:
            h_t: ``(B, N, d_h)`` bf16 token tensor for this frame.

        Returns:
            M_work: ``(B, K_w, N, d_h)`` bf16 — most-recent-first ordering.
        """
        raise NotImplementedError("[C2] WorkBuffer.append — M0 stub")

    def snapshot(self) -> Tensor:
        """Return a *detached* copy of the current ring contents.

        Useful for logging / shape contracts — does not participate in
        autograd.  Returns ``(B, K_w, N, d_h)``.
        """
        raise NotImplementedError("[C2] WorkBuffer.snapshot — M0 stub")
