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


def record_session(
    path: Path,
    device: str,
    *,
    resume: Path | None = None,
    init_from: Path | None = None,
    lineage_start_env_steps: int | None = 0,
) -> dict:
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
        "init_from": str(init_from) if init_from else None,
    }
    if device.startswith("cuda"):
        import torch

        session["gpu"] = torch.cuda.get_device_name(torch.device(device))
    metadata["sessions"].append(session)
    if "lineage_start_env_steps" not in metadata:
        metadata["lineage_start_env_steps"] = lineage_start_env_steps
    metadata["status"] = "running"
    write_json(metadata_path, metadata)
    return metadata


def checkpoint_global_step(
    path: Path, data: dict | None = None, seen: set[Path] | None = None
) -> int | None:
    """Decision count of a checkpoint across its training ancestors.

    New checkpoints carry their starting offset. Older ones can be traced through the
    recorded initial checkpoint while those files are still available.
    """
    import json

    import torch

    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"cycle in checkpoint lineage: {path}")
    seen.add(path)
    if data is None:
        if not path.exists():
            return None
        data = torch.load(path, map_location="cpu", weights_only=False)
    step = int(data.get("counters", {}).get("env_steps", 0))
    training_state = data.get("training_state") or {}
    if training_state.get("lineage_start_env_steps") is not None:
        return int(training_state["lineage_start_env_steps"]) + step
    run = path.parent.parent if path.parent.name == "checkpoints" else path.parent
    metadata_path = run / "metadata.json"
    if not metadata_path.exists():
        return step
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("lineage_start_env_steps") is not None:
        return int(metadata["lineage_start_env_steps"]) + step
    sessions = metadata.get("sessions", [])
    source = sessions[0].get("init_from") if sessions else None
    if not source:
        return step
    parent = checkpoint_global_step(Path(source), seen=seen)
    return None if parent is None else parent + step


def run_lineage_start(path: Path) -> int | None:
    """Global decision count at the start of this run, or None if its source is missing."""
    import json

    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return 0
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "lineage_start_env_steps" in metadata:
        value = metadata["lineage_start_env_steps"]
        return None if value is None else int(value)
    sessions = metadata.get("sessions", [])
    source = sessions[0].get("init_from") if sessions else None
    if source and Path(source).name == "bc.pt" and not Path(source).exists():
        return 0
    return checkpoint_global_step(Path(source)) if source else 0


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
