"""Episode-level strategy assignment: which z each seat plays under (see `carla.envs.strategy`).

The learner's variant follows `strategy.selection` (prescribed archetype, the selector head, or
none). Policy-played opponent seats get the variant their `ExternalAgent.variant` names (the
league's matchmaking chooses it) or, without one, a random variant among the archetypes, none
and (for networks with a selector) select.
"""

from pathlib import Path

import numpy as np
import torch

from carla.config.schema import ExperimentConfig, StrategyConfig
from carla.envs.catan_env import CatanEnv, ExternalAgent
from carla.envs.strategy import (
    NONE,
    SELECT,
    StrategyController,
    StrategyLibrary,
    learner_variant,
    make_controller,
)

LIBRARY_KEY = "strategy_library"
"""Checkpoint entry holding the z library arrays (written by strategy behaviour cloning)."""


def load_library(
    config: ExperimentConfig, checkpoint: str | Path | None = None
) -> StrategyLibrary | None:
    """The z library of `strategy.library`, else the one stored in `checkpoint`."""
    if not config.strategy.enabled:
        return None
    if config.strategy.library:
        return StrategyLibrary.load(config.strategy.library)
    for path in (checkpoint, config.league.init_checkpoint, config.ppo.reference_checkpoint):
        if path and Path(path).exists():
            data = torch.load(path, map_location="cpu", weights_only=False)
            if data.get(LIBRARY_KEY):
                return StrategyLibrary(data[LIBRARY_KEY])
    raise ValueError(
        "strategy.enabled needs strategy.library or a checkpoint that stores a strategy library"
    )


class StrategyAssigner:
    """Creates each episode's controllers with its own RNG stream."""

    def __init__(self, config: StrategyConfig, library: StrategyLibrary | None, seed: int) -> None:
        if config.enabled and library is not None:
            missing = [a for a in config.archetypes if a not in library.archetypes]
            if missing:
                raise ValueError(f"strategy library lacks archetypes {missing}")
        self.config = config
        self.library = library
        self.rng = np.random.default_rng(seed)

    def _child(self) -> np.random.Generator:
        return np.random.default_rng(int(self.rng.integers(2**63)))

    def learner(self, variant: str | None = None) -> StrategyController | None:
        if not self.config.enabled:
            return None
        rng = self._child()
        if variant is None:
            variant = learner_variant(self.config, rng)
        return make_controller(self.config, variant, self.library, rng)

    def opponent(self, variant: str | None, selector: bool) -> StrategyController | None:
        if not self.config.enabled:
            return None
        rng = self._child()
        if variant is None:
            options = [*self.config.archetypes, NONE, *([SELECT] if selector else [])]
            variant = options[int(rng.integers(len(options)))]
        if variant == SELECT and not selector:
            variant = NONE
        return make_controller(self.config, variant, self.library, rng)

    def prepare(
        self, env: CatanEnv, selectors: dict[str, bool], learner_variant: str | None = None
    ) -> None:
        """Assign the learner's and the policy opponents' controllers for a new episode.

        `selectors` maps an `ExternalAgent.key` to whether its network has a selector head.
        """
        env.strategy = self.learner(learner_variant)
        for agent in env.opponents:
            if isinstance(agent, ExternalAgent):
                agent.strategy = self.opponent(agent.variant, selectors.get(agent.key, False))

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def restore(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]
