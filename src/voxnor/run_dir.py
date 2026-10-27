"""Run output directories that record everything needed to reproduce a run."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from voxnor.config import RunConfig, save_config

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "mistral-common",
    "datasets",
    "jiwer",
)


def create_run_dir(cfg: RunConfig, now: datetime | None = None) -> Path:
    """Create `runs/<timestamp>_<name>/` with the resolved config and environment."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    path = Path(cfg.output_root) / f"{stamp}_{cfg.name}"
    path.mkdir(parents=True, exist_ok=False)
    save_config(cfg, path / "config.yaml")
    (path / "environment.json").write_text(
        json.dumps(environment_info(), indent=2) + "\n", encoding="utf-8"
    )
    return path


def environment_info() -> dict[str, Any]:
    import torch

    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **git_state(),
    }


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }


def setup_logging(path: Path) -> None:
    """Log to stdout and to `path` (appending, so a resumed run keeps its history)."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    # One line per HTTP request to the Hub otherwise.
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
