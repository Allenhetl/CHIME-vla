"""Provenance helpers (CODE_STANDARDS §6.3).

Every script entry-point dumps config_resolved.yaml / git_commit.txt /
requirements_freeze.txt / seed.txt next to its outputs so a checkpoint /
report can be traced back to a deterministic environment."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def get_git_commit() -> str:
    """Return current HEAD sha (short).  Returns 'unknown' if not in a git tree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return "unknown"


def get_pip_freeze() -> str:
    """Return ``pip freeze`` output.  Empty string on failure."""
    try:
        out = subprocess.run(
            ["pip", "freeze"], capture_output=True, text=True, check=False, timeout=30
        )
        if out.returncode == 0:
            return out.stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return ""


def save_run_provenance(output_dir: Path, cfg: DictConfig) -> None:
    """Drop config / git / pip / seed alongside script outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config_resolved.yaml")
    (output_dir / "git_commit.txt").write_text(get_git_commit() + "\n")
    (output_dir / "requirements_freeze.txt").write_text(get_pip_freeze())
    seed = OmegaConf.select(cfg, "seed", default=0)
    (output_dir / "seed.txt").write_text(f"{seed}\n")


def copy_file(src: Path, dst: Path) -> None:
    """Tiny shim so callers don't need to import shutil."""
    shutil.copy2(src, dst)
