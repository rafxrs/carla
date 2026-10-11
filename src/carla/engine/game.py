"""`Game`: the engine's public entry point (state + RNG + history + legality checks)."""

import hashlib
import json
from collections.abc import Sequence
from typing import Any, NamedTuple

from carla.engine import rules
from carla.engine.actions import Action, ActionType
from carla.engine.board import Board
from carla.engine.config import EngineConfig
from carla.engine.events import Event
from carla.engine.rng import Rng, derive_seed
from carla.engine.state import GameState, Phase


class IllegalActionError(ValueError):
    pass


class Step(NamedTuple):
    """One applied decision. `outcome` is the chance result it triggered (dice pair, development
    card drawn, resource stolen) or None."""

    player: int
    action: Action
    outcome: Any


class Snapshot(NamedTuple):
    state: GameState
    rng_state: tuple
    history: tuple[Step, ...]


_CHANCE_TYPES = (ActionType.ROLL_DICE, ActionType.BUY_DEV_CARD, ActionType.MOVE_ROBBER)


class Game:
    """A seeded game. Given (config, seed, board) the same actions always give the same game.

    The board comes from `config.board_seed` if set, else from `seed`; it can also be passed
    explicitly (e.g. a hand-made test position). Board generation and play use independent RNG
    streams, so an explicit board does not change the dice.
    """

    __slots__ = ("_legal", "_legal_set", "history", "rng", "seed", "state")

    def __init__(
        self, config: EngineConfig | None = None, seed: int = 0, board: Board | None = None
    ) -> None:
        config = config or EngineConfig()
        if board is None:
            board_seed = (
                config.board_seed if config.board_seed is not None else derive_seed(seed, "board")
            )
            board = Board.random(Rng(board_seed), config.forbid_adjacent_red_numbers)
        self.seed = seed
        self.state = GameState.initial(board, config)
        self.rng = Rng(derive_seed(seed, "play"))
        self.history: list[Step] = []
        self._legal: tuple[Action, ...] | None = None
        self._legal_set: frozenset[Action] | None = None

    # ------------------------------------------------------------------ queries

    @property
    def config(self) -> EngineConfig:
        return self.state.config

    @property
    def board(self) -> Board:
        return self.state.board

    @property
    def phase(self) -> Phase:
        return self.state.phase

    @property
    def current_player(self) -> int:
        return self.state.current_player

    @property
    def decision_player(self) -> int:
        """Seat that must act now, or -1 once the game is over."""
        return self.state.decision_player

    @property
    def turn(self) -> int:
        return self.state.turn

    @property
    def num_steps(self) -> int:
        return len(self.history)

    @property
    def is_over(self) -> bool:
        return self.state.phase == Phase.GAME_OVER

    @property
    def winner(self) -> int | None:
        return self.state.winner if self.state.winner >= 0 else None

    @property
    def is_truncated(self) -> bool:
        """Ended by `max_turns` without a winner."""
        return self.is_over and self.state.winner < 0

    def victory_points(self, player: int, include_hidden: bool = True) -> int:
        return rules.victory_points(self.state, player, include_hidden)

    def legal_actions(self) -> tuple[Action, ...]:
        """Legal decisions for `decision_player` (cached until the next step)."""
        if self._legal is None:
            self._legal = tuple(rules.legal_actions(self.state))
        return self._legal

    def is_legal(self, action: Action) -> bool:
        if self._legal_set is None:
            self._legal_set = frozenset(self.legal_actions())
        if action.type == ActionType.PROPOSE_TRADE:
            return (
                type(action.type) is ActionType
                and action == Action(ActionType.PROPOSE_TRADE, trade=action.trade)
                and (rules.can_propose_trade(self.state) or rules.can_counter_trade(self.state))
                and rules.valid_trade_offer(self.state, action.trade)
            )
        return action in self._legal_set

    # ------------------------------------------------------------------ transitions

    def step(self, action: Action, outcome: Any = None, *, validate: bool = True) -> list[Event]:
        """Apply `action` for `decision_player` and return the resulting events.

        `outcome` forces the chance result of this action (see `Step`); the RNG advances exactly
        as if it had been sampled, which keeps replays of recorded games bit-identical.
        """
        if validate and not self.is_legal(action):
            raise IllegalActionError(
                f"{action} is not legal for seat {self.decision_player} in {self.phase.name}"
            )
        if outcome is not None and action.type not in _CHANCE_TYPES:
            raise ValueError(f"{action.type.name} has no chance outcome")
        player = self.state.decision_player
        events, used = rules.apply_action(self.state, action, self.rng, outcome)
        self.history.append(Step(player, action, used))
        self._legal = self._legal_set = None
        return events

    def play(self, actions: Sequence[Action]) -> list[Event]:
        """Apply several actions; returns all their events."""
        events: list[Event] = []
        for action in actions:
            events.extend(self.step(action))
        return events

    # ------------------------------------------------------------------ snapshots & serialisation

    def snapshot(self) -> Snapshot:
        """Cheap full copy of the mutable game (state, RNG, history)."""
        return Snapshot(self.state.copy(), self.rng.getstate(), tuple(self.history))

    def restore(self, snapshot: Snapshot) -> None:
        """Return to `snapshot`; the snapshot itself stays reusable."""
        self.state = snapshot.state.copy()
        self.rng.setstate(snapshot.rng_state)
        self.history = list(snapshot.history)
        self._legal = self._legal_set = None

    def set_state(self, state: GameState) -> None:
        """Replace the state (e.g. after editing a position by hand). History is kept as is."""
        self.state = state
        self._legal = self._legal_set = None

    def clone(self) -> "Game":
        new = object.__new__(Game)
        new.seed = self.seed
        new.state = self.state.copy()
        new.rng = Rng(0)
        new.rng.setstate(self.rng.getstate())
        new.history = list(self.history)
        new._legal = self._legal
        new._legal_set = self._legal_set
        return new

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible position (state + RNG), enough to continue the game exactly.
        The move history is not included; use `GameRecord` for full logs."""
        version, internal, gauss = self.rng.getstate()
        return {
            "seed": self.seed,
            "num_steps": self.num_steps,
            "state": self.state.to_dict(),
            "rng": [version, list(internal), gauss],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Game":
        new = object.__new__(Game)
        new.seed = data["seed"]
        new.state = GameState.from_dict(data["state"])
        version, internal, gauss = data["rng"]
        new.rng = Rng(0)
        new.rng.setstate((version, tuple(internal), gauss))
        new.history = []
        new._legal = new._legal_set = None
        return new

    def state_hash(self) -> str:
        """Short digest of the game state (RNG excluded), e.g. to verify replays."""
        payload = json.dumps(self.state.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()

    def __repr__(self) -> str:
        return (
            f"Game(seed={self.seed}, step={self.num_steps}, turn={self.turn}, "
            f"phase={self.phase.name}, decision_player={self.decision_player})"
        )
