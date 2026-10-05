"""Game logs: a machine-readable JSON record that can be replayed to any step, and a text log.

The record stores the seed, rules, board and every decision together with the chance outcome it
triggered, so a replay reproduces the game exactly (including the RNG state) and does not depend
on how the dice were originally rolled. Per-step state hashes detect divergence when replaying a
log with a changed engine.
"""

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carla import __version__
from carla.engine.actions import Action
from carla.engine.board import Board
from carla.engine.config import EngineConfig
from carla.engine.constants import NUM_PLAYERS
from carla.engine.events import Event
from carla.engine.game import Game, Step
from carla.engine.state import Phase
from carla.engine.text import describe_board, describe_event, seat

FORMAT = "carla-game-record"
FORMAT_VERSION = 1
_SETUP_PHASES = (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)


class ReplayError(RuntimeError):
    """The record does not replay to the same game (corrupt log or changed engine rules)."""


@dataclass
class GameRecord:
    seed: int
    config: EngineConfig
    board: Board
    steps: list[Step]
    hashes: list[str] | None = None
    """`Game.state_hash()` after each step."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form (agent names, experiment id, ...); `result` is filled from the game."""

    @classmethod
    def from_game(
        cls, game: Game, metadata: dict[str, Any] | None = None, with_hashes: bool = True
    ) -> "GameRecord":
        """Record of `game` from its first step (games restored from a position cannot be logged)."""
        record = cls(game.seed, game.config, game.board, list(game.history), None, {})
        record.metadata = {"carla_version": __version__, **(metadata or {})}
        record.metadata["result"] = game_result(game)
        if with_hashes:
            record.hashes = [g.state_hash() for g, _ in record.iter_steps(verify=False)]
        return record

    def __len__(self) -> int:
        return len(self.steps)

    # ------------------------------------------------------------------ replay

    def new_game(self) -> Game:
        return Game(self.config, self.seed, self.board)

    def iter_steps(self, verify: bool = True) -> Iterator[tuple[Game, list[Event]]]:
        """Replay step by step, yielding the (same, mutated) game after each step and its events."""
        game = self.new_game()
        for i in range(len(self.steps)):
            yield game, self._apply(game, i, verify)

    def replay(self, upto: int | None = None, verify: bool = True) -> Game:
        """The game after its first `upto` steps (all if None): step `#k` of the text log is
        position `replay(k)`."""
        n = len(self.steps) if upto is None else upto
        if not 0 <= n <= len(self.steps):
            raise IndexError(f"position {n} outside 0..{len(self.steps)}")
        game = self.new_game()
        for i in range(n):
            self._apply(game, i, verify)
        return game

    def _apply(self, game: Game, i: int, verify: bool) -> list[Event]:
        step = self.steps[i]
        if step.player != game.decision_player:
            raise ReplayError(f"step {i + 1}: expected seat {game.decision_player} to act")
        try:
            events = game.step(step.action, step.outcome)
        except ValueError as exc:
            raise ReplayError(f"step {i + 1}: {exc}") from exc
        if verify and self.hashes is not None and game.state_hash() != self.hashes[i]:
            raise ReplayError(f"step {i + 1}: state differs from the recorded game")
        return events

    # ------------------------------------------------------------------ JSON

    def to_json(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "seed": self.seed,
            "config": self.config.to_dict(),
            "board": self.board.to_dict(),
            "metadata": self.metadata,
            "steps": [
                [
                    s.player,
                    s.action.to_json(),
                    list(s.outcome) if isinstance(s.outcome, tuple) else s.outcome,
                ]
                for s in self.steps
            ],
            "hashes": self.hashes,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "GameRecord":
        if data.get("format") != FORMAT or data.get("version") != FORMAT_VERSION:
            raise ValueError("not a CARLA game record (or an unsupported version)")
        steps = [
            Step(player, Action.from_json(action), tuple(o) if isinstance(o, list) else o)
            for player, action, o in data["steps"]
        ]
        return cls(
            seed=data["seed"],
            config=EngineConfig.from_dict(data["config"]),
            board=Board.from_dict(data["board"]),
            steps=steps,
            hashes=data.get("hashes"),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> Path:
        """Write JSON (gzip-compressed if the name ends in .gz)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_json(), separators=(",", ":"))
        if path.suffix == ".gz":
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(payload)
        else:
            path.write_text(payload, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GameRecord":
        path = Path(path)
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return cls.from_json(json.load(f))
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ text

    def to_text(self) -> str:
        """Readable log. `#k` labels step k; `replay(k)` loads the position right after it."""
        result = self.metadata.get("result", {})
        names = self.metadata.get("agents")
        lines = [
            f"CARLA game | seed {self.seed} | {len(self.steps)} steps | {_result_text(result)}"
        ]
        if names:
            lines.append("seats: " + " | ".join(f"{seat(p)} {n}" for p, n in enumerate(names)))
        lines += [describe_board(self.board), "", "== Setup =="]
        game = self.new_game()
        header_turn = -1
        for i in range(len(self.steps)):
            if game.phase not in _SETUP_PHASES and game.turn != header_turn:
                header_turn = game.turn
                lines.append(f"== Turn {game.turn + 1} | {seat(game.current_player)} ==")
            events = self._apply(game, i, verify=False)
            label = f"#{i + 1}"
            for j, event in enumerate(events):
                lines.append(f"{label if j == 0 else '':<7}{describe_event(event, self.board)}")
        return "\n".join(lines) + "\n"

    def save_text(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_text(), encoding="utf-8")
        return path


def game_result(game: Game) -> dict[str, Any]:
    return {
        "winner": game.winner,
        "truncated": game.is_truncated,
        "turns": game.turn,
        "steps": game.num_steps,
        "victory_points": [game.victory_points(p) for p in range(NUM_PLAYERS)],
    }


def _result_text(result: dict[str, Any]) -> str:
    if not result:
        return "unfinished"
    vps = " ".join(f"{seat(p)}:{vp}" for p, vp in enumerate(result["victory_points"]))
    if result["winner"] is None:
        outcome = "truncated" if result["truncated"] else "unfinished"
    else:
        outcome = f"{seat(result['winner'])} wins"
    return f"{outcome} after {result['turns']} turns | VP {vps}"
