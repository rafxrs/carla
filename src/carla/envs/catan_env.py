"""Environments over the engine.

`CatanAECEnv`: turn-based multi-agent env (agent-environment-cycle style, like PettingZoo's AEC
API without the dependency). The caller acts for whichever seat must decide; rollout workers use
it for self-play with one policy (copy) per seat.

`CatanEnv`: Gymnasium single-agent view. One learner seat; the other seats are `Agent`s embedded
in the env and played automatically between the learner's decisions.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from carla.agents.base import Agent
from carla.agents.random_agent import RandomAgent
from carla.config.schema import EnvConfig
from carla.engine.actions import Action
from carla.engine.config import EngineConfig
from carla.engine.constants import NUM_PLAYERS
from carla.engine.events import Event
from carla.engine.game import Game
from carla.engine.rng import derive_seed
from carla.engine.text import describe_state
from carla.envs.action_space import HEAD_NAMES, HEAD_SIZES, decode_action, encode_action
from carla.envs.observation import ObservationEncoder
from carla.envs.rewards import public_victory_points, step_rewards


@dataclass
class AECStep:
    actor: int
    """Seat that must act next (-1 once the game is over)."""
    rewards: np.ndarray
    """Reward of every seat for this step."""
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


class CatanAECEnv:
    """Self-play environment: `reset`, then alternate `observe()` and `step(action)` for
    `env.actor` until a step reports termination/truncation.

    A seat's observation includes the public events since its previous observation (redacted
    for that seat). Rewards are per step and seat; sum a seat's rewards between its decisions.
    """

    def __init__(
        self,
        engine_config: EngineConfig | None = None,
        env_config: EnvConfig | None = None,
        encoder: ObservationEncoder | None = None,
    ) -> None:
        self.engine_config = engine_config or EngineConfig()
        self.config = env_config or EnvConfig()
        self.encoder = encoder or ObservationEncoder(self.config.max_events)
        self._game: Game | None = None
        self._pending: list[list[Event]] = [[] for _ in range(NUM_PLAYERS)]

    @property
    def game(self) -> Game:
        if self._game is None:
            raise RuntimeError("call reset() first")
        return self._game

    @property
    def actor(self) -> int:
        return self.game.decision_player

    def reset(self, seed: int) -> int:
        """Start a new game; returns the first actor."""
        self._game = Game(self.engine_config, seed)
        self._pending = [[] for _ in range(NUM_PLAYERS)]
        return self.actor

    def observe(self, seat: int | None = None) -> dict:
        """Observation of `seat` (default: the actor). Consumes that seat's pending events."""
        seat = self.actor if seat is None else seat
        events, self._pending[seat] = self._pending[seat], []
        return self.encoder.encode(self.game, seat, events)

    def step(self, action: Action | Sequence[int] | np.ndarray) -> AECStep:
        """Apply `action` (an engine Action or head indices) for the actor."""
        game = self.game
        if not isinstance(action, Action):
            action = decode_action(action, game.decision_player)
        vp_before = public_victory_points(game) if self.config.vp_reward else None
        events = game.step(action)
        for seat in range(NUM_PLAYERS):
            self._pending[seat].extend(e.visible_to(seat) for e in events)
        rewards = step_rewards(vp_before, game, self.config)
        info: dict[str, Any] = {"events": events}
        if game.is_over:
            info["winner"] = game.winner
            return AECStep(-1, rewards, not game.is_truncated, game.is_truncated, info)
        return AECStep(self.actor, rewards, False, False, info)

    def hidden_truth(self, observer: int) -> dict[str, np.ndarray]:
        """Opponents' hidden cards in relative seat order: targets of the belief module's
        auxiliary loss (training only, never an input)."""
        s = self.game.state
        order = [(observer + k) % NUM_PLAYERS for k in range(1, NUM_PLAYERS)]
        return {
            "resources": np.asarray([s.hands[p] for p in order], dtype=np.float32),
            "dev_cards": np.asarray([s.dev_hands[p] for p in order], dtype=np.float32),
        }


class _LegalActionSpace(spaces.MultiDiscrete):
    """Sample a legal head vector for the environment's current learner decision."""

    def __init__(self, env: "CatanEnv") -> None:
        super().__init__([HEAD_SIZES[h] for h in HEAD_NAMES])
        self.env = env

    def sample(self, mask=None, probability=None):
        if mask is not None or probability is not None or self.env.aec._game is None:
            return super().sample(mask=mask, probability=probability)
        game = self.env.game
        if game.is_over or game.decision_player != self.env.learner:
            return super().sample()
        legal = game.legal_actions()
        action = legal[int(self.np_random.integers(len(legal)))]
        heads = encode_action(action, self.env.learner)
        return np.maximum(heads, 0)


class CatanEnv(gym.Env):
    """One learner seat against three embedded opponents (`RandomAgent`s by default).

    Actions are head indices (`MultiDiscrete` in `HEAD_NAMES` order, unused heads ignored) or
    engine `Action`s; illegal actions raise `IllegalActionError`. `info["legal_actions"]` lists
    the learner's legal actions and `info["hidden_truth"]` the opponents' cards.
    """

    metadata = {"render_modes": ["ansi"]}  # noqa: RUF012 (gymnasium API)

    def __init__(
        self,
        engine_config: EngineConfig | None = None,
        env_config: EnvConfig | None = None,
        opponents: Sequence[Agent] | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.aec = CatanAECEnv(engine_config, env_config)
        self.config = self.aec.config
        self.opponents = (
            list(opponents) if opponents is not None else [RandomAgent() for _ in range(3)]
        )
        if len(self.opponents) != NUM_PLAYERS - 1:
            raise ValueError("CatanEnv needs exactly 3 opponents")
        self.render_mode = render_mode
        self.observation_space = self.aec.encoder.observation_space()
        self.action_space = _LegalActionSpace(self)
        self.learner = 0
        self._seat_agents: dict[int, Agent] = {}

    @property
    def game(self) -> Game:
        return self.aec.game

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(2**62))
        seat = self.config.learner_seat
        self.learner = int(self.np_random.integers(NUM_PLAYERS)) if seat < 0 else seat
        self.aec.reset(game_seed)
        others = [p for p in range(NUM_PLAYERS) if p != self.learner]
        self._seat_agents = dict(zip(others, self.opponents, strict=True))
        for p, agent in self._seat_agents.items():
            agent.reset(self.game, p, derive_seed(game_seed, "seat", p))
        _, obs = self._play_opponents(0.0)
        return obs, self._info()

    def step(self, action):
        if self.game.is_over or self.game.decision_player != self.learner:
            raise RuntimeError("the episode is over; call reset()")
        result = self.aec.step(action)
        reward, obs = self._play_opponents(float(result.rewards[self.learner]))
        over, truncated = self.game.is_over, self.game.is_truncated
        return obs, reward, over and not truncated, truncated, self._info()

    def _play_opponents(self, reward: float) -> tuple[float, dict]:
        """Let opponents act until the learner must decide or the game ends."""
        game = self.game
        while not game.is_over and game.decision_player != self.learner:
            seat = game.decision_player
            action = self._seat_agents[seat].act(game, seat, game.legal_actions())
            reward += float(self.aec.step(action).rewards[self.learner])
        return reward, self.aec.observe(self.learner)

    def _info(self) -> dict[str, Any]:
        over = self.game.is_over
        info: dict[str, Any] = {
            "learner": self.learner,
            "legal_actions": () if over else self.game.legal_actions(),
            "hidden_truth": self.aec.hidden_truth(self.learner),
        }
        if over:
            info["winner"] = self.game.winner
        return info

    def render(self) -> str | None:
        return describe_state(self.game) if self.render_mode == "ansi" else None
