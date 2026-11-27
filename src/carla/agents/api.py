"""Versioned, serializable boundary for untrusted and foreign agents.

Only this exporter reads the referee. Views contain no environment seed or RNG.
JSON storage makes the frozen view deeply immutable; ``state`` returns a new value.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from carla.engine.actions import Action
from carla.engine.events import Event
from carla.engine.game import Game

PROTOCOL = "carla-agent"
VERSION = 1


@dataclass(frozen=True)
class AgentSpec:
    implementation: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentCapabilities:
    players: tuple[int, ...] = (4,)
    victory_points: tuple[int, ...] | None = None
    domestic_trading: str = "supported"

    def validate(self, game: Game) -> None:
        if 4 not in self.players:
            raise ValueError("agent does not support four players")
        if self.victory_points is not None and game.config.vp_to_win not in self.victory_points:
            raise ValueError("agent does not support this victory point target")
        if game.config.allow_domestic_trade and self.domestic_trading == "unsupported":
            raise ValueError("agent does not support domestic-trading games")


@dataclass(frozen=True)
class AgentView:
    seat: int
    step: int
    state_json: str
    legal: tuple[Action, ...]

    @property
    def state(self) -> dict:
        return json.loads(self.state_json)

    def to_json(self) -> dict:
        return {
            "seat": self.seat,
            "step": self.step,
            "state": self.state,
            "legal": [a.to_json() for a in self.legal],
        }

    @classmethod
    def from_game(cls, game: Game, seat: int, *, legal: bool = True) -> "AgentView":
        state = game.state.to_dict()
        state["hand_sizes"] = [sum(h) for h in game.state.hands]
        state["development_sizes"] = [sum(h) for h in game.state.dev_hands]
        state["hands"] = [h if p == seat else None for p, h in enumerate(state["hands"])]
        state["dev_hands"] = [h if p == seat else None for p, h in enumerate(state["dev_hands"])]
        state["development_remaining"] = sum(state.pop("dev_deck"))
        if game.state.current_player != seat:
            state["dev_bought_this_turn"] = None
        actions = game.legal_actions() if legal and game.decision_player == seat else ()
        return cls(seat, game.num_steps, json.dumps(state, sort_keys=True), tuple(actions))


def event_json(event: Event) -> dict:
    return {name: getattr(event, name) for name in event._fields}
