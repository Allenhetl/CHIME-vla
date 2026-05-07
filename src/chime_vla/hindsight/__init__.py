"""Hindsight subpackage — file-protocol consumer of γ̂ labels.

CHIME-VLA does NOT directly import Hindsight code (CODE_STANDARDS §1.6).
This subpackage reads the .pt artefacts emitted by Hindsight scripts/05-07.
"""

from chime_vla.hindsight.consumer import HindsightConsumer, HindsightSample

__all__ = ["HindsightConsumer", "HindsightSample"]
