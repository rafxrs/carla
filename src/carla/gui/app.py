"""Arcade front-end: play a seat against bots, watch bots play, or replay a recorded game.

Play: the human seat is a `HumanAgent` fed by clicks on highlighted legal targets (corners,
edges, tiles from `game.legal_actions()`) and by buttons and pickers (`controller.Controller`);
the other seats are any registered agents, moving on the update tick after a configurable delay.
Watch / replay: step through a `GameRecord` (position k is `record.replay(k)`) with play/pause,
step, seek and speed controls, beside a log of `describe_event` lines.

`GuiApp` holds all behaviour and builds every frame as plain `render.Layer`s; Arcade is imported
only when a window is opened, so `import carla.gui` works without the `gui` extra.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from carla.agents import make_agent
from carla.engine.actions import Action, ActionType
from carla.engine.config import EngineConfig
from carla.engine.constants import NUM_PLAYERS
from carla.engine.game import IllegalActionError
from carla.engine.record import GameRecord
from carla.engine.rng import derive_seed
from carla.engine.state import Phase
from carla.gui import theme
from carla.gui.board_view import BoardMarks, BoardView
from carla.gui.controller import Controller
from carla.gui.geometry import Rect, compute_layout
from carla.gui.panels import LogView, draw_side_panel, draw_status, panel_view
from carla.gui.render import Layer, Painter
from carla.gui.session import GameSession, HumanAgent, Pacer
from carla.gui.widgets import (
    Button,
    PickerLayout,
    Slider,
    button_at,
    draw_button,
    draw_picker,
    draw_slider,
    layout_picker,
)

if TYPE_CHECKING:
    import arcade

__all__ = ["GuiApp", "GuiOptions", "HumanAgent", "main"]


@dataclass(frozen=True)
class GuiOptions:
    """What the window starts with; `carla gui` fills it from an experiment config."""

    mode: str = "play"
    """"play" (a seat against bots), "watch" (bots only) or "replay" (a saved record)."""
    record: Path | None = None
    seat: int = 0
    """The human's seat in Play mode."""
    opponents: tuple[str, ...] = ("heuristic",)
    """Agent names for the bot seats (all four in Watch mode), cycled."""
    delay: float = 0.6
    """Seconds between automatic moves at speed 1×."""
    seed: int = 0
    """Experiment seed: the i-th game of the window uses `derive_seed(seed, "game", i)`."""
    engine: EngineConfig = field(default_factory=EngineConfig)
    output_dir: Path = Path("runs")
    """Saved game records go to `output_dir/gui`."""


# Arcade (pyglet) key codes are X11 keysyms; letters are their lowercase ASCII codes.
KEY_COMMANDS = {
    32: "play_pause",  # space
    65361: "step_back",  # left
    65363: "step_forward",  # right
    65360: "start",  # home
    65367: "end",  # end
    43: "faster",  # +
    61: "faster",  # =
    65451: "faster",  # keypad +
    45: "slower",  # -
    65453: "slower",  # keypad -
    65307: "escape",
    104: "help",  # h
    65470: "help",  # F1
    114: "roll",  # r
    101: "end_turn",  # e
    98: "buy_dev",  # b
    100: "play_dev",  # d
    116: "bank_trade",  # t
    112: "trade",  # p
    115: "save",  # s
    110: "new_game",  # n
}
DIGIT_KEYS = {**{48 + i: i for i in range(1, 10)}, **{65456 + i: i for i in range(1, 10)}}
MOUSE_LEFT, MOUSE_RIGHT = 1, 4
PLAY_COMMANDS = ("roll", "end_turn", "buy_dev", "play_dev", "bank_trade", "trade")
FLASH_SECONDS = 1.6
MESSAGE_SECONDS = 4.0
_TRANSPORT = (
    ("start", "start"),
    ("step_back", "back"),
    ("play_pause", ""),
    ("step_forward", "forward"),
    ("end", "end"),
)
"""Playback buttons (command, icon) at the left of the playback bar; play/pause in the middle."""
_STEP_W, _STEP_GAP = 34, 4
_SPEED_LABEL_W = 52
_SPEED_W = 30 + _SPEED_LABEL_W + 30
"""Width of the slower/faster buttons and the speed label at the right of the playback bar."""
_COUNTER_W = 100
_RESULTS_W = 84

HELP_LINES = (
    ("Space", "play / pause (bots, playback)"),
    ("← / →", "step back / forward"),
    ("Home / End", "first / last position (End returns to the live game)"),
    ("+ / −", "faster / slower"),
    ("R  E  B  D  T", "roll, end turn, buy and play development cards, bank trade"),
    ("P", "offer a trade (when domestic trade is enabled)"),
    ("1 – 9", "choose an option of the open dialog"),
    ("Esc", "cancel the dialog, close this help"),
    ("S  /  N", "save the game record / start a new game"),
    ("H", "show or hide this help"),
    ("Mouse", "click highlighted spots; the wheel scrolls the log; drag the top bar to seek"),
)


class GuiApp:
    """The window's behaviour: modes, input, timing and the layers of each frame."""

    def __init__(self, options: GuiOptions) -> None:
        if options.mode not in ("play", "watch", "replay"):
            raise ValueError(f"unknown GUI mode {options.mode!r}")
        if options.mode == "replay" and options.record is None:
            raise ValueError("replay mode needs a record")
        self.options = options
        self.mode = options.mode
        self.games = 0
        self.pacer = Pacer(options.delay)
        self.board = BoardView()
        self.log_view = LogView()
        self.layout = compute_layout(*theme.WINDOW_SIZE)
        self.board.set_viewport(self.layout.viewport)
        self.mouse = (-1.0, -1.0)
        self.time = 0.0
        self.help = False
        self.results = False
        """The game-over dialog is open."""
        self.message = ""
        self._message_time = -MESSAGE_SECONDS
        self._dragging = False
        self._results_seen = False
        self._flash: tuple[str, int] | None = None
        self._flash_time = -FLASH_SECONDS
        self._synced = -1
        self.window: arcade.Window | None = None
        self.painter: Painter | None = None
        seat = options.seat if self.mode == "play" else -1
        self.controller = Controller(seat, options.engine.allow_domestic_trade)
        self.session = self._new_session()
        self._sync()

    # ------------------------------------------------------------------ sessions

    def _new_session(self) -> GameSession:
        o = self.options
        if self.mode == "replay":
            assert o.record is not None
            return GameSession(GameRecord.load(o.record))
        seed = derive_seed(o.seed, "game", self.games)
        self.games += 1
        names = list(o.opponents or ("heuristic",))
        bots = (names[i % len(names)] for i in range(NUM_PLAYERS))
        agents = [
            HumanAgent() if self.mode == "play" and seat == o.seat else make_agent(next(bots))
            for seat in range(NUM_PLAYERS)
        ]
        viewer = o.seat if self.mode == "play" else None
        return GameSession.new_game(o.engine, seed, agents, viewer)

    def new_game(self) -> None:
        if self.mode == "replay":
            return
        self.session = self._new_session()
        self.log_view = LogView()
        self.results = self._results_seen = False
        self._synced = -1
        self.pacer.restart()
        self._sync()
        self.notify(f"New game (seed {self.session.record.seed})")

    def _sync(self) -> None:
        """Refresh what depends on the position after it changed."""
        s = self.session
        if s.version == self._synced:
            return
        self._synced = s.version
        if self.mode == "play" and s.waiting_for_human:
            self.controller.reset(s.game.state, s.game.legal_actions())
        else:
            self.controller.reset(None)
        history = s.game.history
        self._flash = _flash_target(history[-1].action) if history else None
        self._flash_time = self.time

    @property
    def live_play(self) -> bool:
        """Play mode with the game still running."""
        return self.mode == "play" and not self.session.finished

    def title(self) -> str:
        if self.mode == "replay" and self.options.record is not None:
            return f"CARLA · Replay · {self.options.record.name}"
        return f"CARLA · {'Play' if self.mode == 'play' else 'Watch'}"

    # ------------------------------------------------------------------ actions

    def notify(self, message: str) -> None:
        self.message, self._message_time = message, self.time

    def act(self, action: Action | None) -> None:
        """Play the human's completed `action` (None: input still incomplete)."""
        if action is None:
            return
        try:
            self.session.submit(action)
        except (IllegalActionError, NotImplementedError, RuntimeError) as exc:
            self.notify(f"Not possible: {exc}")
        self.pacer.restart()
        self._sync()

    def command(self, name: str) -> None:
        """Run a button or keyboard command."""
        s = self.session
        if name in PLAY_COMMANDS:
            if self.live_play:
                self.act(self.controller.press(name))
        elif name == "play_pause":
            self.pacer.playing = not self.pacer.playing
            if self.pacer.playing and s.at_end and not s.can_extend and s.length:
                s.seek(0)  # replaying a finished game from its end starts over
            self.notify("Playing" if self.pacer.playing else "Paused: Space resumes")
        elif name == "step_back":
            if self.mode != "play":
                self.pacer.playing = False  # stepping by hand pauses the playback
            s.step_back()
        elif name == "step_forward":
            if self.mode != "play":
                self.pacer.playing = False
            s.step_forward()
        elif name == "start":
            s.seek(0)
        elif name == "end":
            s.seek(s.length)
        elif name == "faster":
            self.pacer.faster()
            self.notify(f"Speed {_speed_text(self.pacer.speed)}")
        elif name == "slower":
            self.pacer.slower()
            self.notify(f"Speed {_speed_text(self.pacer.speed)}")
        elif name == "help":
            self.help = not self.help
        elif name == "escape":
            self.escape()
        elif name == "save":
            self.save()
        elif name == "new_game":
            if s.finished or self.mode == "watch":
                self.new_game()
        elif name == "review":
            self.results = False
            s.viewer = None
            self.notify("Reviewing the game: every card is shown")
        elif name == "results":
            self.results = True
        elif name == "close":
            self.results = False
        self._sync()

    def escape(self) -> None:
        if self.help:
            self.help = False
        elif self.results:
            self.results = False
        elif not self.controller.cancel() and self.log_view.scroll:
            self.log_view.scroll = 0

    def save(self) -> Path | None:
        """Write the game record as JSON under `runtime.output_dir`/gui."""
        if self.mode == "replay":
            self.notify(f"Already saved: {self.options.record}")
            return None
        s = self.session
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = Path(self.options.output_dir) / "gui"
        path = directory / f"game-{stamp}-seed{s.record.seed}.json"
        s.final_record().save(path)
        self.notify(f"Saved {path}")
        return path

    # ------------------------------------------------------------------ event handlers

    def on_update(self, delta_time: float) -> None:
        self.time += delta_time
        if not self.help:
            self._autoplay(delta_time)
        self._sync()
        s = self.session
        if s.live and s.finished and not self._results_seen:
            self._results_seen = self.results = True

    def _autoplay(self, dt: float) -> None:
        s = self.session
        if self.live_play:
            if not s.at_end:
                return  # browsing the history: bots wait
            if s.waiting_for_human:
                self.pacer.restart()
                return
            factor = 0.35 if s.game.phase == Phase.DISCARD else 1.0
            for _ in range(self.pacer.due(dt, factor)):
                if not s.advance():
                    break
            return
        if s.at_end and not s.can_extend:
            return
        for _ in range(self.pacer.due(dt)):
            if not s.step_forward():
                break

    def on_resize(self, width: float, height: float) -> None:
        self.layout = compute_layout(width, height)
        self.board.set_viewport(self.layout.viewport)

    def on_mouse_motion(self, x: float, y: float, dx: float, dy: float) -> None:
        self.mouse = (x, y)

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int) -> None:
        self.mouse = (x, y)
        if button == MOUSE_RIGHT:
            self.escape()
            return
        if button != MOUSE_LEFT:
            return
        if self.help:
            self.help = False
            return
        if self.results:
            hit = button_at(self.results_buttons(), x, y)
            if hit is not None:
                self.command(hit.id)
            return
        if self.controller.picker is not None:
            self._click_picker(x, y)
            return
        hit = button_at(self.buttons(), x, y)
        if hit is not None:
            if hit.enabled:
                self.command(hit.id)
            return
        slider = self.slider()
        if slider is not None and slider.rect.inset(-6, -10).contains(x, y):
            self._dragging = True
            self.session.seek(slider.value_at(x))
            self._sync()
            return
        if self.layout.viewport.contains(x, y):
            self._click_board(x, y)

    def on_mouse_drag(
        self, x: float, y: float, dx: float, dy: float, buttons: int, modifiers: int
    ) -> None:
        self.mouse = (x, y)
        slider = self.slider()
        if self._dragging and slider is not None:
            self.session.seek(slider.value_at(x))
            self._sync()

    def on_mouse_release(self, x: float, y: float, button: int, modifiers: int) -> None:
        self._dragging = False

    def on_mouse_scroll(self, x: float, y: float, scroll_x: float, scroll_y: float) -> None:
        if self.layout.log.contains(x, y):
            self.log_view.scroll = max(0, self.log_view.scroll + round(scroll_y * 3))

    def on_key_press(self, symbol: int, modifiers: int) -> bool:
        if symbol in DIGIT_KEYS and self.controller.picker is not None:
            self.act(self.controller.choose_index(DIGIT_KEYS[symbol] - 1))
        elif symbol in KEY_COMMANDS:
            self.command(KEY_COMMANDS[symbol])
        else:
            return False
        return True  # handled: stops Esc from closing the window

    def _click_picker(self, x: float, y: float) -> None:
        layout = self.picker_layout()
        if layout is None:
            return
        option = layout.option_at(x, y)
        if option is not None:
            self.act(self.controller.choose(option.key))
            return
        on_cancel = layout.cancel is not None and layout.cancel.contains(x, y)
        if on_cancel or not layout.panel.contains(x, y):
            self.controller.cancel()

    def _click_board(self, x: float, y: float) -> None:
        c, tr = self.controller, self.board.transform
        if not c.active:
            return
        corner = tr.corner_at(x, y, c.corner_targets())
        if corner is not None:
            self.act(c.click_corner(corner))
            return
        edge = tr.edge_at(x, y, c.edge_targets())
        if edge is not None:
            self.act(c.click_edge(edge))
            return
        tile = tr.tile_at(x, y, c.tile_targets())
        if tile is not None:
            self.act(c.click_tile(tile))

    def hover_target(self) -> tuple[str, int] | None:
        """Legal target under the mouse, if any."""
        x, y = self.mouse
        c, tr = self.controller, self.board.transform
        if not c.active or not self.layout.viewport.contains(x, y) or self.results or self.help:
            return None
        corner = tr.corner_at(x, y, c.corner_targets())
        if corner is not None:
            return "corner", corner
        edge = tr.edge_at(x, y, c.edge_targets())
        if edge is not None:
            return "edge", edge
        tile = tr.tile_at(x, y, c.tile_targets())
        return None if tile is None else ("tile", tile)

    # ------------------------------------------------------------------ widgets

    def buttons(self) -> list[Button]:
        """Every clickable button outside dialogs: playback, then the play actions."""
        return [*self.playback_buttons(), *self.action_buttons()]

    def playback_buttons(self) -> list[Button]:
        """The playback bar: steps around play/pause on the left, then (after the seek slider)
        the Results button of a finished game and the speed on the right."""
        bar, s = self.layout.playback, self.session
        y, h = bar.cy - 15, 30
        enabled = {
            "start": s.position > 0,
            "step_back": s.position > 0,
            "play_pause": True,
            "step_forward": not s.at_end or (s.can_extend and not s.waiting_for_human),
            "end": not s.at_end,
        }
        out = []
        for i, (command, icon) in enumerate(_TRANSPORT):
            icon = icon or ("pause" if self.pacer.playing else "play")
            rect = Rect(bar.x + 10 + i * (_STEP_W + _STEP_GAP), y, _STEP_W, h)
            out.append(Button(command, command, rect, enabled[command], icon=icon))
        right = bar.right - 10
        out.append(Button("faster", "+", Rect(right - 30, y, 30, h), icon="plus"))
        out.append(Button("slower", "-", Rect(right - _SPEED_W, y, 30, h), icon="minus"))
        if s.live and s.finished:
            rect = Rect(right - _SPEED_W - 10 - _RESULTS_W, y, _RESULTS_W, h)
            out.append(Button("results", "Results", rect))
        return out

    def action_buttons(self) -> list[Button]:
        """The human's play buttons along the bottom (live Play mode only)."""
        if not self.live_play:
            return []
        bar = self.layout.actions
        y, h = bar.cy - 19, 38
        left, right = bar.x + 12, bar.right - 12
        specs = self.controller.buttons()
        widths = [len(spec.label) * 7.6 + 38 for spec in specs]
        scale = min(1.0, (right - left) / (sum(widths) + 8 * (len(widths) - 1)))
        out, x = [], left
        for spec, w in zip(specs, widths, strict=True):
            w *= scale
            accent = spec.id == "roll" and spec.enabled
            rect = Rect(x, y, w, h)
            out.append(Button(spec.id, spec.label, rect, spec.enabled, spec.hotkey, accent=accent))
            x += w + 8 * scale
        return out

    def slider(self) -> Slider | None:
        """The seek bar between the steps and the speed, if the window leaves room for it."""
        bar, s = self.layout.playback, self.session
        left = bar.x + 10 + len(_TRANSPORT) * (_STEP_W + _STEP_GAP) + 10
        right = bar.right - 10 - _SPEED_W - _COUNTER_W
        if s.live and s.finished:
            right -= _RESULTS_W + 10
        if right - left < 60:
            return None
        return Slider(Rect(left, bar.cy - 10, right - left, 20), s.position, s.length)

    def picker_layout(self) -> PickerLayout | None:
        picker = self.controller.picker
        return None if picker is None else layout_picker(picker, self.layout.viewport)

    def results_panel(self) -> Rect:
        v = self.layout.viewport
        return Rect(v.cx - 230, v.cy - 150, 460, 300)

    def results_buttons(self) -> list[Button]:
        p = self.results_panel()
        items = (
            ("save", "Save record", "S"),
            ("review", "Review", ""),
            ("new_game", "New game", "N"),
            ("close", "Close", "Esc"),
        )
        cells = Rect(p.x + 16, p.y + 16, p.w - 32, 36).columns(len(items), 8)
        return [Button(i, text, r, hotkey=k) for (i, text, k), r in zip(items, cells, strict=True)]

    # ------------------------------------------------------------------ frame

    def marks(self) -> BoardMarks:
        c, state = self.controller, self.session.game.state
        corners = {i: a.type == ActionType.BUILD_CITY for i, a in c.corner_targets().items()}
        strength = max(0.0, 1 - (self.time - self._flash_time) / FLASH_SECONDS)
        rolled = state.last_roll if state.has_rolled and state.last_roll != 7 else 0
        return BoardMarks(
            corners=corners,
            edges=frozenset(c.edge_targets()),
            tiles=frozenset(c.tile_targets()),
            hover=self.hover_target(),
            player=c.seat,
            flash=self._flash,
            flash_strength=strength,
            rolled=rolled,
        )

    def status(self) -> tuple[str, str, theme.Color]:
        """(badge, text, colour) of the status line."""
        s = self.session
        game, names = s.game, s.seat_names()
        badges = {"play": f"PLAY · P{self.options.seat}", "watch": "WATCH", "replay": "REPLAY"}
        badge = badges[self.mode]
        if self.time - self._message_time < MESSAGE_SECONDS and self.message:
            return badge, self.message, theme.ACCENT
        if game.is_over:
            winner = game.winner
            if winner is None:
                text = "Game over: turn limit reached, no winner"
            elif winner == s.human_seat:
                text = "You win!"
            else:
                text = f"Game over: {_seat(winner, names)} wins"
            return badge, f"{text} · step {s.position}/{s.length}", theme.TEXT
        if not s.at_end:
            back = "End returns to the game" if s.live else "Space plays"
            return badge, f"Step {s.position} of {s.length} · {back}", theme.TEXT_MUTED
        if self.controller.active:
            return badge, "Your turn · " + self.controller.prompt(), theme.TEXT
        if not self.pacer.playing:
            return badge, "Paused · Space resumes", theme.TEXT_MUTED
        seat = game.decision_player
        doing = "is discarding" if game.phase == Phase.DISCARD else "is thinking…"
        return badge, f"{_seat(seat, names)} {doing}", theme.TEXT_MUTED

    def frame(self) -> list[Layer]:
        """Everything to draw, back to front."""
        s, layout = self.session, self.layout
        layers = [
            self.board.static_layer(s.game.board, layout.board),
            self.board.pieces_layer(s.game.state, self.marks(), self.time),
        ]
        ui = Layer("ui")
        draw_status(ui, layout.status, *self.status(), "H: help")
        self._draw_playback(ui)
        self._draw_actions(ui)
        draw_side_panel(ui, layout.side, panel_view(s))
        self.log_view.draw(ui, layout.log, s)
        layers.append(ui)
        if self.controller.picker is not None:
            layers.append(self._picker_layer())
        if self.results:
            layers.append(self._results_layer())
        if self.help:
            layers.append(self._help_layer())
        return layers

    def _draw_playback(self, layer: Layer) -> None:
        bar = self.layout.playback
        layer.rect(bar.x, bar.y, bar.w, bar.h, theme.with_alpha(theme.BACKGROUND, 150))
        layer.rect(bar.x, bar.top - 1, bar.w, 1, theme.with_alpha(theme.PANEL_EDGE, 160))
        x, y = self.mouse
        buttons = self.playback_buttons()
        for b in buttons:
            draw_button(layer, b, b.rect.contains(x, y) and b.enabled)
        slower = next(b for b in buttons if b.id == "slower")
        speed = _speed_text(self.pacer.speed)
        x_speed = slower.rect.right + _SPEED_LABEL_W / 2
        layer.text(speed, x_speed, bar.cy, theme.FONT, theme.TEXT, True, "center", "center")
        slider = self.slider()
        if slider is not None:
            draw_slider(layer, slider, slider.rect.inset(-6, -10).contains(x, y) or self._dragging)
            count, x_count = f"{slider.value} / {slider.maximum}", slider.rect.right + 14
            muted = theme.TEXT_MUTED
            layer.text(count, x_count, bar.cy, theme.FONT, muted, False, "left", "center")

    def _draw_actions(self, layer: Layer) -> None:
        bar = self.layout.actions
        layer.rect(bar.x, bar.y, bar.w, bar.h, theme.with_alpha(theme.BACKGROUND, 150))
        x, y = self.mouse
        for b in self.action_buttons():
            draw_button(layer, b, b.rect.contains(x, y) and b.enabled)

    def _picker_layer(self) -> Layer:
        layer = Layer("picker")
        board = self.layout.board
        layer.rect(board.x, board.y, board.w, board.h, theme.SHADE)
        picker, layout = self.controller.picker, self.picker_layout()
        assert picker is not None and layout is not None
        x, y = self.mouse
        option = layout.option_at(x, y)
        hovered = option.key if option is not None else None
        if layout.cancel is not None and layout.cancel.contains(x, y):
            hovered = "cancel"
        draw_picker(layer, picker, layout, hovered, self.controller.offer)
        return layer

    def _results_layer(self) -> Layer:
        layer = Layer("results")
        board, p = self.layout.board, self.results_panel()
        layer.rect(board.x, board.y, board.w, board.h, theme.SHADE)
        layer.rounded_rect(p.x, p.y, p.w, p.h, 12, theme.PANEL)
        layer.rounded_frame(p.x, p.y, p.w, p.h, 12, 1.5, theme.PANEL_EDGE)
        s = self.session
        game, names = s.game, s.seat_names()
        winner = game.winner
        if winner is None:
            title, color = "No winner: turn limit reached", theme.TEXT
        else:
            you = winner == s.human_seat
            title = "You win!" if you else f"{_seat(winner, names)} wins"
            color = theme.PLAYER[winner]
        layer.text(title, p.cx, p.top - 34, theme.FONT_TITLE + 4, color, True, "center", "center")
        detail = f"{game.turn} turns · {s.length} steps · seed {s.record.seed}"
        muted = theme.TEXT_MUTED
        layer.text(detail, p.cx, p.top - 64, theme.FONT, muted, False, "center", "center")
        order = sorted(range(NUM_PLAYERS), key=lambda q: -game.victory_points(q))
        for row, q in enumerate(order):
            y = p.top - 100 - row * 28
            layer.rounded_rect(p.x + 60, y - 9, 18, 18, 4, theme.PLAYER[q])
            name = f"P{q}  {names[q]}"
            layer.text(name, p.x + 90, y, theme.FONT, theme.TEXT, q == winner, "left", "center")
            vp = f"{game.victory_points(q)} VP"
            layer.text(vp, p.right - 60, y, theme.FONT, theme.TEXT, True, "right", "center")
        x, y = self.mouse
        for b in self.results_buttons():
            draw_button(layer, b, b.rect.contains(x, y))
        return layer

    def _help_layer(self) -> Layer:
        layer = Layer("help")
        w, h = self.layout.width, self.layout.height
        layer.rect(0, 0, w, h, theme.SHADE)
        pw, ph = 640, 70 + 26 * len(HELP_LINES)
        p = Rect(w / 2 - pw / 2, h / 2 - ph / 2, pw, ph)
        layer.rounded_rect(p.x, p.y, p.w, p.h, 12, theme.PANEL)
        layer.rounded_frame(p.x, p.y, p.w, p.h, 12, 1.5, theme.PANEL_EDGE)
        y = p.top - 22
        layer.text(
            "Keyboard and mouse", p.x + 24, y, theme.FONT_LARGE, theme.TEXT, True, "left", "center"
        )
        close = "Esc or click to close"
        layer.text(
            close, p.right - 24, y, theme.FONT_SMALL, theme.TEXT_DIM, False, "right", "center"
        )
        for i, (keys, what) in enumerate(HELP_LINES):
            y = p.top - 60 - i * 26
            layer.text(keys, p.x + 24, y, theme.FONT, theme.ACCENT, True, "left", "center")
            layer.text(what, p.x + 160, y, theme.FONT, theme.TEXT, False, "left", "center")
        return layer

    # ------------------------------------------------------------------ Arcade

    def open(self, visible: bool = True) -> "arcade.Window":
        """Create the Arcade window and route its events here."""
        import arcade

        width, height = theme.WINDOW_SIZE
        window = arcade.Window(width, height, self.title(), resizable=True, visible=visible)
        window.set_minimum_size(*theme.MIN_WINDOW_SIZE)
        window.background_color = theme.BACKGROUND
        self.window, self.painter = window, Painter()
        window.push_handlers(self)
        self.on_resize(*window.get_size())
        return window

    def on_draw(self) -> None:
        if self.window is None or self.painter is None:
            return
        self.window.clear()
        self.window.default_camera.use()
        self.painter.draw(self.frame())

    def run(self) -> None:
        import arcade

        self.open()
        arcade.run()


def _flash_target(action: Action) -> tuple[str, int] | None:
    t = action.type
    if t in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
        return "corner", action.corner
    if t == ActionType.BUILD_ROAD:
        return "edge", action.edge
    if t == ActionType.MOVE_ROBBER:
        return "tile", action.tile
    return None


def _seat(seat: int, names: list[str]) -> str:
    """ "P2 (heuristic)", or just "P2" when the agent is unknown."""
    return f"P{seat} ({names[seat]})" if names[seat] else f"P{seat}"


def _speed_text(speed: float) -> str:
    return f"{speed:g}×"


def main(options: GuiOptions | str | Path | None = None) -> None:
    """Open the GUI: Play mode by default, or the replay of the record at `options`."""
    if options is None:
        options = GuiOptions()
    elif isinstance(options, str | Path):
        options = GuiOptions(mode="replay", record=Path(options))
    GuiApp(options).run()
