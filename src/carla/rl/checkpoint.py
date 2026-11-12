"""Checkpoints: model + optimiser + config + counters + RNG states."""

import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from carla.config import to_dict


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextmanager
def isolated_rng(seed: int | None = None):
    """Evaluation must not perturb subsequent training actions or minibatch shuffles."""
    state = rng_state()
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % 2**32)  # noqa: NPY002
            torch.manual_seed(seed)
        yield
    finally:
        restore_rng_state(state)


def save_checkpoint(
    path: str | Path, policy, optimizer=None, config=None, training_state=None, **counters: Any
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "config": to_dict(config) if config is not None else None,
            "counters": counters,
            "training_state": training_state,
            "rng": rng_state(),
        },
        temporary,
    )
    temporary.replace(path)
    return path


def load_checkpoint(path: str | Path, policy, optimizer=None, restore_rng: bool = False) -> dict:
    """Load weights (and optimiser/RNG state); returns the stored counters and config."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(data["policy"])
    if optimizer is not None and data["optimizer"] is not None:
        optimizer.load_state_dict(data["optimizer"])
    if restore_rng:
        restore_rng_state(data["rng"])
    return {
        "counters": data["counters"],
        "config": data["config"],
        "training_state": data.get("training_state"),
    }
