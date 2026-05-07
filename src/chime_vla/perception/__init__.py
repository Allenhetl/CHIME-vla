"""Perception subpackage — [C1] backbone + [C2] FIFO ring buffer."""

from chime_vla.perception.fifo_buffer import WorkBuffer
from chime_vla.perception.vlm_backbone import VLMBackbone

__all__ = ["VLMBackbone", "WorkBuffer"]
