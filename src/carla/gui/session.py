"""What the window shows: a game record, a view position in it and the event log, plus for live
games the agents that extend the record (pure: no Arcade).

Position k is the game after its first k steps, exactly as `GameRecord.replay(k)` builds it;
checkpoints every few steps make seeking cheap in both directions.
"""

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from carla.agents.base import Agent
from carla.engine.actions import Action, ActionType
from carla.engine.config import EngineConfig
from carla.engine.constants import NUM_PLAYERS
from carla.engine.events import Event
from carla.engine.game import Game, IllegalActionError, Snapshot
from carla.engine.record import GameRecord
from carla.engine.rng import derive_seed
from carla.engine.state import Phase
from carla.engine.text import describe_event, seat

_SETUP = (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)


class HumanAgent(Agent):
    """Seat played through the GUI. The window hands it the player's choice (`submit`), then the
    session asks it to `act` like any other agent, so it never blocks the event loop."""

    name = "human"

    def __init__(self) -> None:
        self._choice: Action | None = None

    def submit(self, action: Action) -> None:
        self._choice = action

    @property
    def ready(self) -> bool:
        return self._choice is not None

    def act(self, game: Game, seat: int, legal: Sequence[Action]) -> Action:
        action, self._choice = self._choice, None
        if action is None:
            raise RuntimeError("HumanAgent.act called before the player chose an action")
        if action not in legal:
            raise IllegalActionError(f"{action} is not legal for seat {seat} in {game.phase.name}")
        return action


@dataclass(frozen=True)
class LogEntry:
    """A log line: an event of step `step` (1-based), or a header when `event` is None."""

    step: int
    player: int
    event: Event | None = None
    header: str = ""


class GameSession:
    """A record, the position shown (`game` equals `record.replay(position)`) and the log; live
    games also have one agent per seat, which add steps at the end of the record."""

    CHECKPOINT_EVERY = 32

    def __init__(
        self,
        record: GameRecord,
        agents: Sequence[Agent] | None = None,
        viewer: int | None = None,
    ) -> None:
        self.record = record
        self.agents = list(agents) if agents is not None else None
        self.viewer = viewer
        """Seat whose hidden information may be shown; None shows everything (spectators)."""
        self.game = record.replay(0)
        self.log: list[LogEntry] = []
        self._log_steps: list[int] = []
        self._checkpoints: dict[int, Snapshot] = {0: self.game.snapshot()}
        self._turn_logged = -1
        self.version = 0
        """Changes whenever the position shown changes."""
        self.finished = False
        """The game has reached its end (whichever position is shown)."""
        self._log(LogEntry(0, -1, header="Setup"))
        if record.steps:
            self._index_record()

    @classmethod
    def new_game(
        cls,
        config: EngineConfig,
        seed: int,
        agents: Sequence[Agent],
        viewer: int | None = None,
    ) -> "GameSession":
        """A live game from its first step; agents are seeded like `carla.eval.play_game`."""
        board = Game(config, seed).board
        metadata = {"agents": [a.name for a in agents]}
        session = cls(GameRecord(seed, config, board, [], None, metadata), agents, viewer)
        for s, agent in enumerate(agents):
            agent.reset(session.game, s, derive_seed(seed, "seat", s))
        return session

    def _index_record(self) -> None:
        """Replay the record once: verifies it and fills the log and the checkpoints."""
        before = (0, Phase.SETUP_SETTLEMENT, 0)
        for i, (game, events) in enumerate(self.record.iter_steps(), start=1):
            self._log_step(i, self.record.steps[i - 1].player, events, *before)
            before = (game.turn, game.phase, game.current_player)
            if i % self.CHECKPOINT_EVERY == 0:
                self._checkpoints[i] = game.snapshot()
            self.finished = game.is_over

    # ------------------------------------------------------------------ state

    @property
    def position(self) -> int:
        return self.game.num_steps

    @property
    def length(self) -> int:
        return len(self.record.steps)

    @property
    def at_end(self) -> bool:
        return self.position == self.length

    @property
    def live(self) -> bool:
        return self.agents is not None

    @property
    def can_extend(self) -> bool:
        """New steps can be played: a live game shown at its last position, not finished."""
        return self.live and self.at_end and not self.game.is_over

    @property
    def human_seat(self) -> int | None:
        return next((s for s, a in enumerate(self.agents or ()) if isinstance(a, HumanAgent)), None)

    @property
    def waiting_for_human(self) -> bool:
        return (
            self.agents is not None
            and self.can_extend
            and isinstance(self.agents[self.game.decision_player], HumanAgent)
        )

    def seat_names(self) -> list[str]:
        """Agent name per seat ("You" for the human; empty when a record does not say)."""
        if self.agents is not None:
            return ["You" if isinstance(a, HumanAgent) else a.name for a in self.agents]
        names = self.record.metadata.get("agents")
        return [str(n) for n in names] if names else [""] * NUM_PLAYERS

    def last_dice(self) -> tuple[int, int] | None:
        """The most recent roll up to the position shown."""
        for step in reversed(self.game.history):
            if step.action.type == ActionType.ROLL_DICE:
                return step.outcome
        return None

    # ------------------------------------------------------------------ moving

    def seek(self, k: int) -> None:
        """Show position `k` (clamped to the record)."""
        k = max(0, min(k, self.length))
        here = self.position
        if k == here:
            return
        if k < here or k - here > self.CHECKPOINT_EVERY:
            base = max(c for c in self._checkpoints if c <= k)
            if k < here or base > here:
                self.game.restore(self._checkpoints[base])
        steps = self.record.steps
        while self.game.num_steps < k:
            step = steps[self.game.num_steps]
            self.game.step(step.action, step.outcome)
        self.version += 1

    def step_back(self) -> bool:
        if self.position == 0:
            return False
        self.seek(self.position - 1)
        return True

    def step_forward(self) -> bool:
        """Show the next recorded step, or at the end of a live game let the next bot move."""
        if not self.at_end:
            self.seek(self.position + 1)
            return True
        return self.advance()

    def advance(self) -> bool:
        """Let the bot whose decision it is move (live games at their last position only)."""
        if self.agents is None or not self.can_extend:
            return False
        seat_ = self.game.decision_player
        agent = self.agents[seat_]
        if isinstance(agent, HumanAgent):
            return False
        self._play(agent.act(self.game, seat_, self.game.legal_actions()))
        return True

    def submit(self, action: Action) -> list[Event]:
        """Play the human seat's `action` (IllegalActionError if it is not legal)."""
        if self.agents is None or not self.waiting_for_human:
            raise RuntimeError("the human seat is not to act")
        seat_ = self.game.decision_player
        human = self.agents[seat_]
        assert isinstance(human, HumanAgent)
        human.submit(action)
        return self._play(human.act(self.game, seat_, self.game.legal_actions()))

    def _play(self, action: Action) -> list[Event]:
        game = self.game
        before = (game.turn, game.phase, game.current_player)
        player = game.decision_player
        events = game.step(action)
        self.record.steps.append(game.history[-1])
        self._log_step(game.num_steps, player, events, *before)
        if game.num_steps % self.CHECKPOINT_EVERY == 0:
            self._checkpoints[game.num_steps] = game.snapshot()
        self.finished = game.is_over
        self.version += 1
        return events

    # ------------------------------------------------------------------ log and records

    def _log(self, entry: LogEntry) -> None:
        self.log.append(entry)
        self._log_steps.append(entry.step)

    def _log_step(
        self, step: int, player: int, events: list[Event], turn: int, phase: Phase, current: int
    ) -> None:
        if phase not in _SETUP and turn != self._turn_logged:
            self._turn_logged = turn
            self._log(LogEntry(step, current, header=f"Turn {turn + 1} · {seat(current)}"))
        for event in events:
            self._log(LogEntry(step, event.player if event.player >= 0 else player, event))

    def visible_count(self) -> int:
        """Number of log entries up to the position shown."""
        return bisect_right(self._log_steps, self.position)

    def visible_log(self) -> list[LogEntry]:
        """Log entries up to the position shown."""
        return self.log[: self.visible_count()]

    def entry_text(self, entry: LogEntry) -> str:
        """The entry as the viewer may read it (hidden cards stay hidden)."""
        if entry.event is None:
            return entry.header
        event = entry.event if self.viewer is None else entry.event.visible_to(self.viewer)
        return describe_event(event, self.game.board)

    def final_record(self, metadata: dict[str, Any] | None = None) -> GameRecord:
        """Record of the whole game (with state hashes), e.g. to save it."""
        game = self.game if self.at_end else self.record.replay()
        keep = {
            k: v for k, v in self.record.metadata.items() if k not in ("result", "carla_version")
        }
        return GameRecord.from_game(game, {**keep, **(metadata or {})})


class Pacer:
    """Paces automatic moves (bots, playback): one move every `delay / speed` seconds."""

    SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

    def __init__(self, delay: float = 0.6, playing: bool = True) -> None:
        self.delay = max(0.0, delay)
        self.playing = playing
        self.speed_index = self.SPEEDS.index(1.0)
        self._clock = 0.0

    @property
    def speed(self) -> float:
        return self.SPEEDS[self.speed_index]

    def faster(self) -> None:
        self.speed_index = min(self.speed_index + 1, len(self.SPEEDS) - 1)

    def slower(self) -> None:
        self.speed_index = max(self.speed_index - 1, 0)

    def restart(self) -> None:
        """Wait a full interval before the next move."""
        self._clock = 0.0

    def due(self, dt: float, factor: float = 1.0, limit: int = 50) -> int:
        """Moves due after `dt` more seconds (`factor` scales the delay; 0 while paused)."""
        if not self.playing:
            return 0
        interval = self.delay * factor / self.speed
        if interval <= 0:
            return 1
        self._clock += dt
        moves = min(int(self._clock / interval), limit)
        self._clock = 0.0 if moves == limit else self._clock - moves * interval
        return moves
