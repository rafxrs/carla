"""Seeding for reproducible runs.

Every source of randomness gets its own seed derived from the experiment seed with
`derive_seed(seed, *keys)`: games (`"game", i`), workers (`"worker", w`), agents
(`"seat", s`), torch. Never share one RNG between processes.
"""

import os
import random

import numpy as np

from carla.engine.rng import derive_seed

__all__ = ["derive_seed", "seed_everything"]


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy and (if installed) torch; optionally force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 (legacy global RNG used by libraries)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
