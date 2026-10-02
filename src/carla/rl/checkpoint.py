"""Checkpoints: model + optimiser + config + counters + RNG states."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from carla.config import to_dict


def save_checkpoint(path: str | Path, policy, optimizer=None, config=None, **counters: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "config": to_dict(config) if config is not None else None,
            "counters": counters,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),  # noqa: NPY002
                "torch": torch.get_rng_state(),
            },
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, policy, optimizer=None, restore_rng: bool = False) -> dict:
    """Load weights (and optimiser/RNG state); returns the stored counters and config."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(data["policy"])
    if optimizer is not None and data["optimizer"] is not None:
        optimizer.load_state_dict(data["optimizer"])
    if restore_rng:
        random.setstate(data["rng"]["python"])
        np.random.set_state(data["rng"]["numpy"])  # noqa: NPY002
        torch.set_rng_state(data["rng"]["torch"])
    return {"counters": data["counters"], "config": data["config"]}
