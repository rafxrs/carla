"""Smoke test of the real Arcade window, hidden (skipped without Arcade or a display)."""

import importlib
import os
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import pytest

from carla.engine import actions as A
from carla.engine.actions import Action
from carla.engine.actions import ActionType as T
from carla.engine.config import EngineConfig
from carla.engine.constants import DevCard
from carla.engine.record import GameRecord
from carla.engine.rng import Rng
from carla.engine.simulate import play_random_game
from carla.gui.app import DIGIT_KEYS, KEY_COMMANDS, MOUSE_LEFT, MOUSE_RIGHT, GuiApp, GuiOptions
from carla.gui.geometry import Point
from tests.engine.scenario import give, post_setup_game

if TYPE_CHECKING:
    from arcade import Window

REQUIRE_GUI = os.environ.get("CARLA_REQUIRE_GUI") == "1"

try:
    arcade = importlib.import_module("arcade") if REQUIRE_GUI else pytest.importorskip("arcade")
except Exception as exc:  # pyglet raises when it cannot reach a display
    if REQUIRE_GUI:
        raise
    pytest.skip(f"no display: {exc}", allow_module_level=True)


@pytest.fixture
def open_app():
    apps: list[GuiApp] = []

    def make(options: GuiOptions) -> tuple[GuiApp, "Window"]:
        app = GuiApp(options)
        try:
            window = app.open(visible=False)
        except Exception as exc:  # no OpenGL 3.3 context available
            if REQUIRE_GUI:
                pytest.fail(f"Arcade window could not open: {exc}")
            pytest.skip(f"cannot open a window: {exc}")
        apps.append(app)
        return app, window

    yield make
    for app in apps:
        if app.window is not None and not app.window.closed:
            app.window.close()


def drawn_something(app: GuiApp) -> bool:
    """Draw a frame now and check it is not blank (frames drawn by the event loop are already
    flipped away when the test looks)."""
    app.on_draw()
    colours = arcade.get_image(components=3).getcolors(maxcolors=4096)
    return colours is None or len(colours) > 100  # None: more colours than that


def send(window: "Window", *event: object) -> None:
    """Dispatch an event; pyglet may queue it until the window next dispatches its events."""
    window.dispatch_event(*event)
    window.dispatch_events()


def test_play_window_draws_and_routes_input(open_app):
    app, window = open_app(GuiOptions(mode="play", delay=0.0))
    window.test(frames=3)  # the event loop draws through the app's handlers
    assert app.painter is not None and app.painter.frame >= 1 and drawn_something(app)
    corner = min(app.controller.corner_targets())
    x, y = app.board.transform.corners[corner]
    send(window, "on_mouse_motion", x, y, 0, 0)
    send(window, "on_mouse_press", x, y, arcade.MOUSE_BUTTON_LEFT, 0)
    assert app.session.position == 1 and app.mouse == (x, y)
    send(window, "on_key_press", arcade.key.H, 0)
    assert app.help and drawn_something(app)
    send(window, "on_key_press", arcade.key.ESCAPE, 0)
    assert not app.help and not window.closed  # Esc closes dialogs, never the window
    send(window, "on_resize", 1000, 660)
    assert app.layout.width == 1000 and drawn_something(app)


def test_domestic_trade_picker_draws_in_arcade_window(open_app):
    config = EngineConfig(allow_domestic_trade=True)
    app, _window = open_app(GuiOptions(mode="play", engine=config, delay=0.0))
    game = post_setup_game(config=config)
    give(game, 0, (2, 0, 0, 0, 0))
    app.controller.reset(game.state, game.legal_actions())
    assert app.controller.press("trade") is None
    app.controller.choose(("give", 0, 1))
    app.controller.choose(("give", 0, 1))
    app.controller.choose(("get", 1, 1))
    assert app.controller.picker is not None and app.controller.picker.kind == "offer"
    assert drawn_something(app)


def test_counteroffer_and_confirmation_draw_in_arcade_window(open_app):
    config = EngineConfig(allow_domestic_trade=True)
    app, _window = open_app(GuiOptions(mode="play", engine=config, delay=0.0))
    game = post_setup_game(config=config)
    give(game, 0, (2, 0, 1, 0, 0))
    give(game, 1, (0, 1, 0, 0, 1))
    game.step(A.ONE_FOR_ONE[0][4])
    app.on_resize(1000, 660)
    app.controller.seat = 1
    app.controller.reset(game.state, game.legal_actions())
    assert app.controller.picker is not None and app.controller.picker.kind == "response"
    assert drawn_something(app)
    app.controller.choose("counter")
    assert app.controller.picker is not None
    assert app.controller.picker.cancel_label == "Back (Esc)"
    assert drawn_something(app)
    game.step(A.ONE_FOR_ONE[4][2])
    game.step(A.REJECT_TRADE)
    game.step(A.REJECT_TRADE)
    app.controller.seat = 0
    app.controller.reset(game.state, game.legal_actions())
    assert app.controller.picker is not None and app.controller.picker.kind == "confirm"
    assert app.controller.picker.options[0].detail == "Give 1 wool\nGet 1 ore"
    assert drawn_something(app)


def test_watch_and_replay_windows(open_app, tmp_path):
    app, window = open_app(GuiOptions(mode="watch", delay=0.0))
    window.test(frames=5)
    assert app.session.position >= 3 and drawn_something(app)
    window.close()
    path = GameRecord.from_game(play_random_game(1)).save(tmp_path / "game.json")
    app, window = open_app(GuiOptions(mode="replay", record=path, delay=0.0))
    send(window, "on_key_press", arcade.key.END, 0)
    send(window, "on_key_press", arcade.key.H, 0)
    assert app.session.at_end and app.help and drawn_something(app)


def test_key_and_mouse_codes_match_arcade():
    k = arcade.key
    expected = {
        k.SPACE: "play_pause",
        k.LEFT: "step_back",
        k.RIGHT: "step_forward",
        k.HOME: "start",
        k.END: "end",
        k.PLUS: "faster",
        k.EQUAL: "faster",
        k.NUM_ADD: "faster",
        k.MINUS: "slower",
        k.NUM_SUBTRACT: "slower",
        k.ESCAPE: "escape",
        k.H: "help",
        k.F1: "help",
        k.R: "roll",
        k.E: "end_turn",
        k.B: "buy_dev",
        k.D: "play_dev",
        k.T: "bank_trade",
        k.P: "trade",
        k.S: "save",
        k.N: "new_game",
    }
    assert expected == KEY_COMMANDS
    assert DIGIT_KEYS[k.KEY_1] == DIGIT_KEYS[k.NUM_1] == 1
    assert DIGIT_KEYS[k.KEY_9] == DIGIT_KEYS[k.NUM_9] == 9
    assert (MOUSE_LEFT, MOUSE_RIGHT) == (arcade.MOUSE_BUTTON_LEFT, arcade.MOUSE_BUTTON_RIGHT)


def click(window: "Window", xy: tuple[float, float]) -> None:
    for event in ("on_mouse_motion", "on_mouse_press", "on_mouse_release"):
        args = (0, 0) if event == "on_mouse_motion" else (arcade.MOUSE_BUTTON_LEFT, 0)
        send(window, event, xy[0], xy[1], *args)


def screen_clicks(app: GuiApp, action: Action, legal: Sequence[Action]) -> Iterator[Point]:
    """Where a player clicks, one point at a time, to make `action` (pickers open in between)."""
    tr, t = app.board.transform, action.type

    def button(name: str) -> Point:
        hit = next(b for b in app.buttons() if b.id == name)
        assert hit.enabled
        return hit.rect.cx, hit.rect.cy

    def option(key: object) -> Point:
        layout = app.picker_layout()
        assert layout is not None
        cell = next(r for o, r in layout.options if o.key == key)
        return cell.cx, cell.cy

    simple = {T.ROLL_DICE: "roll", T.END_TURN: "end_turn", T.BUY_DEV_CARD: "buy_dev"}
    if t in simple:
        yield button(simple[t])
    elif t in (T.BUILD_SETTLEMENT, T.BUILD_CITY):
        yield tr.corners[action.corner]
    elif t == T.BUILD_ROAD:
        yield tr.edges[action.edge]
    elif t == T.MOVE_ROBBER:
        yield tr.tiles[action.tile]
        if sum(a.type == t and a.tile == action.tile for a in legal) > 1:
            yield option(action.player)
    elif t == T.DISCARD:
        yield option(action.resource)
    elif t == T.MARITIME_TRADE:
        yield button("bank_trade")
        yield option(action.resource)
        yield option(action.resource2)
    else:
        assert t == T.PLAY_DEV_CARD
        card = next(
            c
            for c in app.hand_bar().cards
            if c.kind == "dev" and c.index == action.dev and c.playable
        )
        yield card.rect.x + 3, card.rect.cy  # the left edge stays visible when cards overlap
        if action.dev in (DevCard.YEAR_OF_PLENTY, DevCard.MONOPOLY):
            yield option(action.resource)
        if action.dev == DevCard.YEAR_OF_PLENTY:
            yield option(action.resource2)


def test_a_human_finishes_a_game_against_heuristic_bots_by_clicking(open_app):
    app, window = open_app(GuiOptions(mode="play", seat=1, delay=0.0))
    s, rng, moves = app.session, Rng(1), 0
    assert s.seat_names() == ["heuristic", "You", "heuristic", "heuristic"]
    while not s.game.is_over:
        if not s.waiting_for_human:
            app.on_update(0.05)
            continue
        legal = s.game.legal_actions()
        kinds = sorted({a.type for a in legal})
        kind = kinds[rng.randbelow(len(kinds))]
        target = rng.choice([a for a in legal if a.type == kind])
        for xy in screen_clicks(app, target, legal):
            click(window, xy)
        assert s.game.history[-1].action == target
        moves += 1
        if moves % 10 == 0:
            assert drawn_something(app)
    app.on_update(0.05)
    assert app.results and moves > 20 and drawn_something(app)
