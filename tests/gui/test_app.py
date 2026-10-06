"""GuiApp behaviour without a window: input routing, shortcuts, modes and dialogs."""

import os
import subprocess
import sys

import pytest

from carla.engine.actions import ActionType
from carla.engine.record import GameRecord
from carla.engine.rng import Rng
from carla.engine.simulate import play_random_game
from carla.engine.state import Phase
from carla.gui.app import HELP_LINES, KEY_COMMANDS, MOUSE_RIGHT, GuiApp, GuiOptions
from tests.engine.scenario import give

SETUP = (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)


def key(command: str) -> int:
    return next(code for code, name in KEY_COMMANDS.items() if name == command)


def click(app: GuiApp, xy: tuple[float, float], button: int = 1) -> None:
    app.on_mouse_press(xy[0], xy[1], button, 0)


def press(app: GuiApp, command: str) -> None:
    assert app.on_key_press(key(command), 0)


def run_bots(app: GuiApp) -> None:
    """Update until the human must act (delay 0: one bot move per update)."""
    for _ in range(5000):
        if app.session.waiting_for_human or app.session.game.is_over:
            return
        app.on_update(0.05)
    raise AssertionError("the bots never handed over")


def click_first_target(app: GuiApp) -> None:
    c, tr = app.controller, app.board.transform
    if c.corner_targets():
        click(app, tr.corners[min(c.corner_targets())])
    elif c.edge_targets():
        click(app, tr.edges[min(c.edge_targets())])
    else:
        click(app, tr.tiles[min(c.tile_targets())])
        if c.picker is not None:  # several victims
            press_digit(app, 1)


def press_digit(app: GuiApp, digit: int) -> None:
    assert app.on_key_press(48 + digit, 0)


def play_setup(app: GuiApp) -> None:
    while app.session.game.phase in SETUP:
        if app.session.waiting_for_human:
            before = app.session.position
            click_first_target(app)
            assert app.session.position == before + 1
        else:
            app.on_update(0.05)
    run_bots(app)


def play_app(**options) -> GuiApp:
    return GuiApp(GuiOptions(mode="play", delay=0.0, **options))


def test_play_mode_builds_by_clicking_highlighted_spots():
    app = play_app(seat=0)
    assert app.controller.active and app.session.seat_names()[0] == "You"
    assert [layer.name for layer in app.frame()] == ["board", "pieces", "ui"]
    corner = min(app.controller.corner_targets())
    click(app, (0, 0))  # outside the board: nothing happens
    assert app.session.position == 0
    app.mouse = app.board.transform.corners[corner]
    assert app.hover_target() == ("corner", corner) and app.marks().hover == ("corner", corner)
    click(app, app.mouse)
    assert app.session.position == 1 and app.session.game.state.corner_owner[corner] == 0
    assert app.status()[1].startswith("Your turn · Place a road")
    click_first_target(app)
    assert app.session.position == 2 and not app.controller.active
    assert "is thinking" in app.status()[1]
    run_bots(app)
    assert app.session.game.phase == Phase.SETUP_SETTLEMENT and app.session.position == 14


def test_other_seats_and_opponents_are_respected():
    app = play_app(seat=2, opponents=("random", "heuristic"))
    assert app.session.seat_names() == ["random", "heuristic", "You", "random"]
    assert not app.controller.active  # seat 0 starts
    run_bots(app)
    assert app.session.game.decision_player == 2 and app.controller.active


def test_keyboard_plays_a_turn():
    app = play_app()
    play_setup(app)
    assert app.session.game.phase == Phase.PRE_ROLL
    press(app, "end_turn")  # not legal before rolling: ignored
    assert app.session.game.history[-1].action.type != ActionType.END_TURN
    press(app, "roll")
    assert app.session.game.history[-1].action.type == ActionType.ROLL_DICE
    while app.session.game.phase != Phase.MAIN:  # a 7: discard and move the robber by clicks
        if app.controller.picker is not None:
            press_digit(
                app, next(i for i, o in enumerate(app.controller.picker.options, 1) if o.enabled)
            )
        else:
            click_first_target(app)
        run_bots(app)
    press(app, "end_turn")
    assert app.session.game.history[-1].action.type == ActionType.END_TURN


def test_pickers_by_mouse_digits_escape_and_right_click():
    app = play_app()
    play_setup(app)
    press(app, "roll")
    run_bots(app)
    while app.session.game.phase != Phase.MAIN:
        click_first_target(app)
        run_bots(app)
    give(app.session.game, 0, (4, 4, 4, 4, 4))
    app.session.version += 1  # the position changed behind the session's back
    app.on_update(0.0)
    press(app, "bank_trade")
    assert app.controller.picker is not None and app.controller.picker.kind == "give"
    press(app, "escape")
    assert app.controller.picker is None
    press(app, "bank_trade")
    click(app, (2, 2))  # outside the dialog: cancels it
    assert app.controller.picker is None
    press(app, "bank_trade")
    click(app, (100, 100), MOUSE_RIGHT)
    assert app.controller.picker is None
    press(app, "bank_trade")
    layout = app.picker_layout()
    assert layout is not None and [layer.name for layer in app.frame()][-1] == "picker"
    lumber = next(r for o, r in layout.options if o.key == 0)
    click(app, (lumber.cx, lumber.cy))
    assert app.controller.picker is not None and app.controller.picker.kind == "get"
    hand = list(app.session.game.state.hands[0])
    press_digit(app, 4)  # the fourth option of the "get" picker: ore
    last = app.session.game.history[-1].action
    assert (last.type, last.resource, last.resource2) == (ActionType.MARITIME_TRADE, 0, 4)
    assert app.session.game.state.hands[0][4] == hand[4] + 1 and app.controller.picker is None


def test_watch_mode_plays_pauses_steps_and_seeks():
    app = GuiApp(GuiOptions(mode="watch", delay=0.0, opponents=("heuristic",)))
    assert app.session.seat_names() == ["heuristic"] * 4 and app.slider() is not None
    for _ in range(40):
        app.on_update(0.05)
    assert app.session.position == 40
    press(app, "play_pause")
    app.on_update(1.0)
    assert app.session.position == 40 and app.status()[1].startswith("Paused")
    press(app, "step_back")
    app.time += 10  # past the "Paused" notification
    assert app.session.position == 39 and app.status()[1].startswith("Step 39 of 40")
    press(app, "step_forward")
    press(app, "step_forward")  # at the end of a live game: the next bot moves
    assert app.session.position == 41
    press(app, "start")
    assert app.session.position == 0
    press(app, "end")
    assert app.session.position == 41
    slider = app.slider()
    assert slider is not None
    click(app, (slider.rect.x + slider.rect.w / 2, slider.rect.cy))
    assert app.session.position in (20, 21)
    app.on_mouse_drag(slider.rect.x - 50, slider.rect.cy, -1, 0, 1, 0)
    assert app.session.position == 0
    app.on_mouse_release(0, 0, 1, 0)
    app.on_mouse_drag(slider.rect.right, slider.rect.cy, 1, 0, 1, 0)
    assert app.session.position == 0  # no longer dragging
    speed = app.pacer.speed
    press(app, "faster")
    assert app.pacer.speed == 2 * speed and app.status()[1] == "Speed 2×"
    press(app, "slower")
    press(app, "slower")
    assert app.pacer.speed == speed / 2


def test_transport_buttons_follow_the_position():
    app = GuiApp(GuiOptions(mode="watch", delay=0.0))
    buttons = {b.id: b for b in app.buttons()}
    assert not buttons["start"].enabled and buttons["step_forward"].enabled
    click(app, (buttons["step_forward"].rect.cx, buttons["step_forward"].rect.cy))
    assert app.session.position == 1 and not app.pacer.playing
    click(app, (buttons["play_pause"].rect.cx, buttons["play_pause"].rect.cy))
    assert app.pacer.playing


def test_replay_mode(tmp_path):
    record = GameRecord.from_game(play_random_game(4))
    path = record.save(tmp_path / "game.json.gz")
    app = GuiApp(GuiOptions(mode="replay", record=path, delay=0.0))
    assert app.title().endswith("game.json.gz") and app.session.position == 0
    for _ in range(10):
        app.on_update(0.05)
    assert app.session.position == 10
    press(app, "end")
    app.on_update(0.05)
    assert app.session.at_end and not app.results  # recordings have no results dialog
    winner = app.session.game.winner  # agents unknown: seats are just "P1"
    assert app.session.seat_names() == [""] * 4
    assert app.status()[1].startswith(f"Game over: P{winner} wins")
    press(app, "play_pause")
    press(app, "play_pause")  # playing from the end starts over
    assert app.session.position == 0 and app.pacer.playing
    assert app.save() is None and "Already saved" in app.message
    press(app, "new_game")
    assert app.session.record is not None and app.session.length == len(record)


def test_game_over_dialog_saves_reviews_and_restarts(tmp_path):
    app = play_app(output_dir=tmp_path)
    rng = Rng(5)
    s = app.session
    while not s.game.is_over:
        if s.waiting_for_human:
            app.act(rng.choice(s.game.legal_actions()))
        else:
            app.on_update(0.05)
    app.on_update(0.05)
    assert app.results and [layer.name for layer in app.frame()][-1] == "results"
    assert app.slider() is not None  # the finished game can be browsed
    save = next(b for b in app.results_buttons() if b.id == "save")
    click(app, (save.rect.cx, save.rect.cy))
    (saved,) = (tmp_path / "gui").glob("game-*.json")
    loaded = GameRecord.load(saved)
    assert loaded.replay().state_hash() == s.game.state_hash()
    assert loaded.metadata["agents"] == ["human", "heuristic", "heuristic", "heuristic"]
    press(app, "escape")
    assert not app.results
    press(app, "start")  # browsing a finished game keeps the playback controls
    assert not app.live_play and app.slider() is not None
    assert {"start", "results"} <= {b.id for b in app.buttons()}
    press(app, "end")
    reopen = next(b for b in app.buttons() if b.id == "results")
    click(app, (reopen.rect.cx, reopen.rect.cy))
    assert app.results and s.viewer == 0
    review = next(b for b in app.results_buttons() if b.id == "review")
    click(app, (review.rect.cx, review.rect.cy))
    assert not app.results and s.viewer is None  # every card is revealed
    press(app, "new_game")
    assert app.session is not s and app.session.position == 0 and app.games == 2
    assert app.session.record.seed != s.record.seed and app.controller.active


def test_help_overlay_pauses_the_bots():
    app = GuiApp(GuiOptions(mode="watch", delay=0.0))
    press(app, "help")
    assert app.help and [layer.name for layer in app.frame()][-1] == "help"
    app.on_update(1.0)
    assert app.session.position == 0
    click(app, (5, 5))
    assert not app.help
    press(app, "help")
    press(app, "escape")
    assert not app.help
    app.on_update(0.05)
    assert app.session.position == 1


def test_resize_and_log_scrolling():
    app = GuiApp(GuiOptions(mode="watch", delay=0.0))
    for _ in range(60):
        app.on_update(0.05)
    app.on_resize(1000, 660)
    assert app.layout.width == 1000 and app.board.transform.viewport == app.layout.viewport
    log = app.layout.log
    app.on_mouse_scroll(log.cx, log.cy, 0, 2)
    assert app.log_view.scroll == 6
    app.on_mouse_scroll(10, 10, 0, 5)  # not over the log
    assert app.log_view.scroll == 6
    app.frame()
    press(app, "escape")  # Esc jumps back to the newest lines
    assert app.log_view.scroll == 0
    assert not app.on_key_press(ord("z"), 0)  # unknown keys are left to the window


def test_invalid_options():
    with pytest.raises(ValueError, match="mode"):
        GuiApp(GuiOptions(mode="spectate"))
    with pytest.raises(ValueError, match="record"):
        GuiApp(GuiOptions(mode="replay"))


def test_gui_imports_and_builds_frames_without_arcade():
    code = (
        "import sys\n"
        "sys.modules['arcade'] = None\n"
        "sys.modules['pyglet'] = None\n"
        "import carla.gui\n"
        "from carla.gui.app import GuiApp, GuiOptions, HumanAgent, main\n"
        "app = GuiApp(GuiOptions(mode='watch', delay=0.0))\n"
        "app.on_update(0.1)\n"
        "assert app.frame()\n"
        "try:\n"
        "    app.run()\n"
        "except ImportError:\n"
        "    print('no arcade')\n"
    )
    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "no arcade"


def inside(inner, outer) -> bool:
    return outer.contains(inner.x, inner.y) and outer.contains(inner.right, inner.top)


def overlaps(a, b) -> bool:
    return a.x < b.right and b.x < a.right and a.y < b.top and b.y < a.top


@pytest.mark.parametrize("size", [(1280, 800), (1000, 660), (1600, 1000)])
@pytest.mark.parametrize("mode", ["play", "watch", "replay"])
def test_playback_controls_sit_in_the_bar_under_the_status(tmp_path, mode, size):
    record = GameRecord.from_game(play_random_game(4)).save(tmp_path / "game.json")
    app = GuiApp(GuiOptions(mode=mode, record=record, delay=0.0))
    app.on_resize(*size)
    bar, viewport = app.layout.playback, app.layout.viewport
    ids = [b.id for b in app.playback_buttons()]
    assert ids[:5] == ["start", "step_back", "play_pause", "step_forward", "end"]
    assert {"slower", "faster"} <= set(ids)
    slider = app.slider()
    assert slider is not None and slider.rect.w >= 140
    for rect in (*(b.rect for b in app.playback_buttons()), slider.rect.inset(-6, -10)):
        assert inside(rect, bar) and not overlaps(rect, viewport)
    for b in app.action_buttons():
        assert inside(b.rect, app.layout.actions) and not overlaps(b.rect, viewport)
    assert any("drag the top bar to seek" in what for _, what in HELP_LINES)


def test_seeking_during_live_play_pauses_the_game_until_the_end():
    app = play_app()
    play_setup(app)
    here = app.session.position
    slider = app.slider()
    assert slider is not None
    click(app, (slider.rect.x, slider.rect.cy))  # back to the first position
    assert app.session.position == 0 and not app.controller.active
    app.on_update(1.0)
    assert app.session.position == 0  # nobody moves while the history is shown
    end = next(b for b in app.playback_buttons() if b.id == "end")
    click(app, (end.rect.cx, end.rect.cy))
    assert app.session.position == here and app.controller.active
