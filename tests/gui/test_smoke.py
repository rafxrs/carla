"""Smoke test of the real Arcade window, hidden (skipped without Arcade or a display)."""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from arcade import Window

try:
    arcade = pytest.importorskip("arcade")
except Exception as exc:  # pyglet raises when it cannot reach a display
    pytest.skip(f"no display: {exc}", allow_module_level=True)

from carla.engine.record import GameRecord
from carla.engine.simulate import play_random_game
from carla.gui.app import DIGIT_KEYS, KEY_COMMANDS, MOUSE_LEFT, MOUSE_RIGHT, GuiApp, GuiOptions


@pytest.fixture
def open_app():
    apps: list[GuiApp] = []

    def make(options: GuiOptions) -> tuple[GuiApp, "Window"]:
        app = GuiApp(options)
        try:
            window = app.open(visible=False)
        except Exception as exc:  # no OpenGL 3.3 context available
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
