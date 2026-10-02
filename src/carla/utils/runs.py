"""Run directories: resolved config + code version saved next to every output."""

import subprocess
from datetime import datetime
from pathlib import Path

from carla.config import config_hash, save_config
from carla.config.schema import ExperimentConfig


def create_run_dir(config: ExperimentConfig) -> Path:
    """`<output_dir>/<name>-<timestamp>-<config hash>/` with config.yaml and git.txt."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(config.runtime.output_dir) / f"{config.name}-{stamp}-{config_hash(config)}"
    path.mkdir(parents=True, exist_ok=False)
    save_config(config, path / "config.yaml")
    (path / "git.txt").write_text(_git_version(), encoding="utf-8")
    return path


def _git_version() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown\n"
    return f"{commit}{' (uncommitted changes)' if dirty else ''}\n"
