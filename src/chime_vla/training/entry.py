"""Console-script entry point: ``chime-train``.

Thin shim that registers the structured config and delegates to
``scripts/10_train.py``'s ``@hydra.main`` function.  Allows
``chime-train trainer.fast_dev_run=True`` from any cwd post
``pip install -e .``.

M0: stub.  Full Hydra wiring lives in ``scripts/10_train.py``.
"""

from __future__ import annotations


def main() -> None:
    """Entry point.  M0: stub."""
    raise NotImplementedError(
        "chime_vla.training.entry:main — M0 stub.  "
        "Use ``python scripts/10_train.py`` directly until M1."
    )
