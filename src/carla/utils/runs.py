"""Run directories: resolved config + code version saved next to every output."""

import platform
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from carla.config import config_hash, save_config
from carla.config.schema import ExperimentConfig
from carla.utils.logging import write_json


def record_session(path: Path, device: str, *, resume: Path | None = None) -> dict:
    """Append machine/software provenance, including when a run moves to another laptop."""
    import json

    metadata_path = path / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {"schema_version": 1, "sessions": []}
    )
    packages = {}
    for package in ("torch", "numpy", "gymnasium", "matplotlib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "unavailable"
    session = {
        "started": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": device,
        "packages": packages,
        "git": _git_version().strip(),
        "resume": str(resume) if resume else None,
    }
    if device.startswith("cuda"):
        import torch

        session["gpu"] = torch.cuda.get_device_name(torch.device(device))
    metadata["sessions"].append(session)
    metadata["status"] = "running"
    write_json(metadata_path, metadata)
    return metadata


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
