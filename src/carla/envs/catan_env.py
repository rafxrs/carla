"""Environments over the engine.

`CatanAECEnv`: turn-based multi-agent env (agent-environment-cycle style, like PettingZoo's AEC
API without the dependency). The caller acts for whichever seat must decide; rollout workers use
it for self-play with one policy (copy) per seat.

`CatanEnv`: Gymnasium single-agent view. One learner seat; the other seats are `Agent`s embedded
in the env and played automatically between the learner's decisions.
"""

from collections.abc import Generator, Sequence
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
from carla.envs.rewards import potential, public_victory_points, step_rewards
from carla.envs.strategy import StrategyController, StrategyStats, track_step


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
        self.encoder = encoder or ObservationEncoder(
            self.config.max_events, self.config.strategic_features
        )
        self._game: Game | None = None
        self._pending: list[list[Event]] = [[] for _ in range(NUM_PLAYERS)]
        self.stats: list[StrategyStats] = [StrategyStats() for _ in range(NUM_PLAYERS)]
        """Public build statistics of every seat (`carla.envs.strategy`)."""
        self.last_action_types: list[int] = [-1] * NUM_PLAYERS

    def __setstate__(self, state: dict) -> None:
        # Envs pickled in checkpoints from before strategy statistics existed: start them
        # empty (they only feed planning features and diagnostics, unused by older models).
        self.__dict__.update(state)
        self.__dict__.setdefault("stats", [StrategyStats() for _ in range(NUM_PLAYERS)])
        self.__dict__.setdefault("last_action_types", [-1] * NUM_PLAYERS)

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
        self.stats = [StrategyStats() for _ in range(NUM_PLAYERS)]
        self.last_action_types = [-1] * NUM_PLAYERS
        return self.actor

    def observe(self, seat: int | None = None) -> dict:
        """Observation of `seat` (default: the actor). Consumes that seat's pending events."""
        seat = self.actor if seat is None else seat
        events, self._pending[seat] = self._pending[seat], []
        if not self.config.strategic_features:
            return self.encoder.encode(self.game, seat, events)
        return self.encoder.encode(
            self.game,
            seat,
            events,
            stats=self.stats,
            last_action_type=self.last_action_types[seat],
        )

    def step(self, action: Action | Sequence[int] | np.ndarray) -> AECStep:
        """Apply an engine Action or policy indices, including offer tokens, for the actor."""
        game = self.game
        if not isinstance(action, Action):
            action = decode_action(action, game.decision_player)
        vp_before = public_victory_points(game) if self.config.vp_reward else None
        phase, actor = game.phase, game.decision_player
        events = game.step(action)
        track_step(self.stats, events, phase)
        self.last_action_types[actor] = int(action.type)
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


class ExternalAgent(Agent):
    """An opponent seat whose decisions the caller supplies, e.g. a policy batched across envs.

    `CatanEnv.reset_steps` and `step_steps` yield the seat each time it must decide; the caller
    sends back its action (an engine `Action` or policy head indices). `key` names the network
    that plays it and `state` holds that network's recurrent state for this seat.
    """

    variant: str | None = None  # Class defaults for agents pickled before these existed.
    strategy: "StrategyController | None" = None

    def __init__(self, name: str, key: str, variant: str | None = None) -> None:
        self.name = name
        self.key = key
        self.state: np.ndarray | None = None
        self.variant = variant
        """Strategy variant of a strategy-conditioned network ("select", "none", an archetype),
        or None to let the caller choose."""
        self.strategy: StrategyController | None = None
        """This seat's strategy for the episode (set by the caller)."""

    def reset(self, game: Game, seat: int, seed: int) -> None:
        self.state = None

    def act(self, game: Game, seat: int, legal: Sequence[Action]) -> Action:
        raise RuntimeError("external opponents act through CatanEnv.reset_steps/step_steps")


EnvSteps = Generator[int, Any, Any]
"""Yields each external seat that must decide; receives its action; returns the env result."""


def run_steps(steps: EnvSteps) -> Any:
    """Drive `steps` to its result; it must not need an external decision."""
    try:
        seat = next(steps)
    except StopIteration as stop:
        return stop.value
    steps.close()
    raise RuntimeError(f"seat {seat} is an ExternalAgent; drive reset_steps/step_steps instead")


class _LegalActionSpace(spaces.MultiDiscrete):
    """Sample a legal head vector for the environment's current learner decision."""

    def __init__(self, env: "CatanEnv") -> None:
        # The policy writes -1 to heads its chosen action does not use.
        start = [0] + [-1] * (len(HEAD_NAMES) - 1)
        nvec = [HEAD_SIZES["type"]] + [HEAD_SIZES[h] + 1 for h in HEAD_NAMES[1:]]
        super().__init__(nvec, start=start)
        self.env = env

    def sample(self, mask=None, probability=None):
        if mask is not None or probability is not None or self.env.aec._game is None:
            return super().sample(mask=mask, probability=probability)
        game = self.env.game
        if game.is_over or game.decision_player != self.env.learner:
            return super().sample()
        legal = game.legal_actions()
        action = legal[int(self.np_random.integers(len(legal)))]
        return encode_action(action, self.env.learner)


class CatanEnv(gym.Env):
    """One learner seat against three embedded opponents (`RandomAgent`s by default).

    Actions are head indices (`MultiDiscrete` in `HEAD_NAMES` order, unused heads are `-1`) or
    engine `Action`s; illegal actions raise `IllegalActionError`. `info["legal_actions"]` lists
    the learner's legal actions and `info["hidden_truth"]` the opponents' cards. Use
    `TradeActionWrapper` to declare decoder actions to Gymnasium clients. `ExternalAgent`
    opponents are played by the caller through `reset_steps` and `step_steps`.
    """

    metadata = {"render_modes": ["ansi"]}  # noqa: RUF012 (gymnasium API)
    strategy: StrategyController | None = None  # Class default for envs pickled before it.

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
        self._potential = 0.0
        self.strategy: StrategyController | None = None
        """The learner's strategy for the episode (set by the caller before or at reset);
        with one, `info["strategy_reward"]` holds the strategy pseudo-reward of each step."""

    @property
    def game(self) -> Game:
        return self.aec.game

    def seat_agent(self, seat: int) -> Agent:
        return self._seat_agents[seat]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return run_steps(self.reset_steps(seed=seed, options=options))

    def step(self, action):
        return run_steps(self.step_steps(action))

    def reset_steps(self, *, seed: int | None = None, options: dict | None = None) -> EnvSteps:
        """`reset` that yields to the caller whenever an `ExternalAgent` seat must decide."""
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(2**62))
        seat = self.config.learner_seat
        self.learner = int(self.np_random.integers(NUM_PLAYERS)) if seat < 0 else seat
        self.aec.reset(game_seed)
        others = [p for p in range(NUM_PLAYERS) if p != self.learner]
        self._seat_agents = dict(zip(others, self.opponents, strict=True))
        for p, agent in self._seat_agents.items():
            agent.reset(self.game, p, derive_seed(game_seed, "seat", p))
        _, obs = yield from self._play_opponents(0.0)
        self._potential = potential(self.game, self.learner, self.config)
        return obs, self._info()

    def step_steps(self, action) -> EnvSteps:
        """`step` that yields to the caller whenever an `ExternalAgent` seat must decide."""
        if self.game.is_over or self.game.decision_player != self.learner:
            raise RuntimeError("the episode is over; call reset()")
        result = self.aec.step(action)
        reward, obs = yield from self._play_opponents(float(result.rewards[self.learner]))
        if self.config.shaping != "none":
            # F = gamma * Phi(s') - Phi(s) between consecutive learner decisions; Phi = 0 at the end.
            next_potential = potential(self.game, self.learner, self.config)
            reward += self.config.shaping_gamma * next_potential - self._potential
            self._potential = next_potential
        over, truncated = self.game.is_over, self.game.is_truncated
        info = self._info()
        if self.strategy is not None:
            info["strategy_reward"] = self.strategy.reward(self.game, self.aec.stats[self.learner])
        return obs, reward, over and not truncated, truncated, info

    def _play_opponents(self, reward: float) -> Generator[int, Any, tuple[float, dict]]:
        """Let opponents act until the learner must decide or the game ends."""
        game = self.game
        while not game.is_over and game.decision_player != self.learner:
            seat = game.decision_player
            agent = self._seat_agents[seat]
            if isinstance(agent, ExternalAgent):
                action = yield seat
            else:
                action = agent.act(game, seat, game.legal_actions())
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
