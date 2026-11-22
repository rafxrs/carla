"""A trained checkpoint playing one seat, e.g. `make_agent("ppo:runs/x/latest.pt")`."""

import functools
import random
from collections.abc import Sequence
from pathlib import Path

import torch

from carla.agents.base import Agent
from carla.config.loader import from_dict
from carla.config.schema import ExperimentConfig
from carla.engine.actions import Action
from carla.engine.events import Event
from carla.engine.game import Game
from carla.envs.action_space import decode_action
from carla.envs.observation import ObservationEncoder
from carla.models.base import PolicyNetwork, batch_observations
from carla.models.factory import build_policy

PREFIX = "ppo:"


@functools.lru_cache(maxsize=8)
def load_checkpoint(path: str) -> tuple[ExperimentConfig, PolicyNetwork]:
    """Config and frozen CPU policy of a training checkpoint (cached: games share one copy)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("config") is None:
        raise ValueError(f"checkpoint has no experiment config: {path}")
    config = from_dict(ExperimentConfig, data["config"])
    policy = build_policy(config.model)
    policy.load_state_dict(data["policy"])
    policy.eval()
    return config, policy


class PolicyAgent(Agent):
    """Plays like evaluation: greedy (or sampled) actions, belief state carried across decisions.

    The engine only hands agents the game, so the agent replays new moves on a private copy to
    recover the events its seat would have observed since its previous decision, exactly as
    `CatanAECEnv` delivers them during training.
    """

    def __init__(self, checkpoint: str | Path, *, deterministic: bool = True) -> None:
        self.checkpoint = str(checkpoint)
        self.config, self.policy = load_checkpoint(self.checkpoint)
        self.encoder = ObservationEncoder(self.config.env.max_events)
        self.deterministic = deterministic
        self.name = PREFIX + self.checkpoint
        self._start: Game | None = None
        self._shadow: Game | None = None
        self._pending: list[Event] = []
        self._state: torch.Tensor | None = None
        self._rng = random.Random(0)
        self.fallbacks = 0  # Decoded actions that were not legal (should stay 0).
        self.last_observation: dict | None = None

    def reset(self, game: Game, seat: int, seed: int) -> None:
        if game.config != self.config.engine:
            raise ValueError(
                f"{self.checkpoint} was trained with different engine rules; "
                "pass the experiment config it was trained with (--config)"
            )
        self._start = Game(game.config, game.seed, game.board)  # The position before move 1.
        self._restart(game, seat)
        self._rng = random.Random(seed)

    def _restart(self, game: Game, seat: int) -> None:
        """Replay the game so far on a fresh copy; the belief starts over (used after a seek)."""
        assert self._start is not None
        self._shadow = self._start.clone()
        self._pending = []
        self._state = self.policy.initial_state(1)
        self._catch_up(game, seat)

    def _catch_up(self, game: Game, seat: int) -> None:
        assert self._shadow is not None
        shadow = self._shadow
        for step in game.history[shadow.num_steps :]:
            events = shadow.step(step.action, step.outcome, validate=False)
            self._pending.extend(e.visible_to(seat) for e in events)

    @torch.no_grad()
    def act(self, game: Game, seat: int, legal: Sequence[Action]) -> Action:
        if self._shadow is None:
            self.reset(game, seat, 0)
        assert self._shadow is not None
        known = self._shadow.history
        if game.num_steps < len(known) or game.history[: len(known)] != known:
            self._restart(game, seat)
        else:
            self._catch_up(game, seat)
        obs = self.last_observation = self.encoder.encode(game, seat, self._pending)
        self._pending = []
        batch = batch_observations([obs])
        out = self.policy(batch, self._state)
        self._state = out.state
        heads = out.distribution(batch["masks"]).sample(deterministic=self.deterministic)
        action = decode_action(heads[0].cpu().numpy(), seat)
        # `legal` lists 1-for-1 offers only; the engine also accepts the multi-card offers the
        # trade decoder writes. The masks should make every decoded action legal.
        if action not in legal and not game.is_legal(action):
            self.fallbacks += 1
            action = self._rng.choice(list(legal))
        return action
