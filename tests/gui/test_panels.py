from carla.agents import make_agent
from carla.engine.config import EngineConfig
from carla.engine.constants import DevCard
from carla.engine.record import GameRecord
from carla.engine.simulate import play_random_game
from carla.engine.state import Phase
from carla.gui.geometry import compute_layout
from carla.gui.panels import (
    LogView,
    draw_side_panel,
    draw_status,
    ellipsize,
    panel_view,
    plural,
    wrap,
)
from carla.gui.render import Layer
from carla.gui.session import GameSession, HumanAgent
from tests.engine.scenario import finish_setup, give, give_dev

LAYOUT = compute_layout(1280, 800)


def session(viewer: int | None) -> GameSession:
    agents = [
        HumanAgent() if viewer == 0 else make_agent("random"),
        *(make_agent("random") for _ in range(3)),
    ]
    s = GameSession.new_game(EngineConfig(), 5, agents, viewer)
    finish_setup(s.game)
    give(s.game, 0, (1, 2, 0, 0, 3))
    give(s.game, 1, (4, 0, 0, 1, 0))
    give_dev(s.game, 0, DevCard.VICTORY_POINT)
    give_dev(s.game, 1, DevCard.VICTORY_POINT, 2)
    give_dev(s.game, 1, DevCard.KNIGHT)
    return s


def test_the_player_sees_only_its_own_cards():
    s = session(viewer=0)
    view = panel_view(s)
    me, other = view.players[0], view.players[1]
    assert view.hand.seat == 0 and view.hand.title == "Your hand"
    assert view.hand.resources == tuple(s.game.state.hands[0])
    assert view.hand.dev == ((DevCard.VICTORY_POINT, 1, 0),)
    assert me.vp_cards == 1 and me.vp == 3  # two setup settlements plus the hidden card
    assert other.vp_cards == 0 and other.vp == 2  # its two VP cards stay hidden
    assert (other.cards, other.dev_cards) == (s.game.state.hand_size(1), 3)  # sizes are public


def test_spectators_see_everything():
    s = session(viewer=None)
    view = panel_view(s)
    assert view.players[1].vp == 4 and view.players[1].vp_cards == 2
    assert view.hand.seat == s.game.decision_player == 0 and view.hand.title == "P0's hand"
    assert view.title == "Turn 1" and view.phase == "P0 · Roll"
    assert view.dice is None and not view.rolled
    assert view.bank == tuple(s.game.state.bank) and view.dev_deck == 25 - 4


def test_titles_follow_the_game():
    game = play_random_game(3)
    s = GameSession(GameRecord.from_game(game))
    s.seek(s.length)
    assert panel_view(s).title == "Game over" and panel_view(s).phase.startswith(f"P{game.winner}")
    s.seek(0)
    assert panel_view(s).title == "Setup"
    discards = [i for i, step in enumerate(game.history) if step.action.type.name == "DISCARD"]
    for i in discards:  # a seat other than the roller discarding
        s.seek(i)
        if s.game.decision_player != s.game.current_player:
            break
    view = panel_view(s)
    assert s.game.phase == Phase.DISCARD and "to act" in view.phase
    assert any(p.deciding and not p.current for p in view.players)


def test_side_panel_and_status_draw_their_text():
    s = session(viewer=0)
    layer = Layer("ui")
    draw_side_panel(layer, LAYOUT.side, panel_view(s))
    texts = {t.text for t in layer.texts}
    assert {"Your hand", "Bank", "P0  You", "1 VP card"} <= texts
    assert all(LAYOUT.side.contains(t.x, t.y) for t in layer.texts)
    draw_status(layer, LAYOUT.status, "PLAY", "Your turn", (255, 255, 255, 255), "H: help")
    assert {"PLAY", "Your turn", "H: help"} <= {t.text for t in layer.texts}


def test_text_helpers():
    lines = wrap(
        "P0 builds a settlement at corner 12 [lumber 8, brick 5, wool 10; 3:1 port]", 200, 9
    )
    assert len(lines) > 1 and all(len(line) <= 32 for line in lines)
    assert wrap("", 100, 9) == [""]
    assert ellipsize("short", 300, 9) == "short"
    assert ellipsize("a" * 100, 100, 9).endswith("…") and len(ellipsize("a" * 100, 100, 9)) < 100
    assert (plural(1, "card"), plural(0, "card"), plural(3, "knight")) == (
        "1 card",
        "0 cards",
        "3 knights",
    )


def test_log_window_scrolls_and_wraps():
    game = play_random_game(2)
    s = GameSession(GameRecord.from_game(game))
    s.seek(s.length)
    log = LogView()
    lines, total = log.window(s, 150, 10)
    assert len(lines) == 10 and total > len(s.log)  # narrow: some entries take two lines
    assert lines[-1][1] == wrap(s.entry_text(s.log[-1]), 150, LogView.SIZE)[-1]
    log.scroll = 10**6  # clamped to the top
    lines, _ = log.window(s, 150, 10)
    assert log.scroll == total - 10 and lines[0][0] is s.log[0]
    s.seek(5)
    lines, total = log.window(s, 150, 100)
    assert log.scroll == 0 and all(entry.step <= 5 for entry, _ in lines)
    layer = Layer("ui")
    log.draw(layer, LAYOUT.log, s)
    assert layer.texts[0].text == "Log" and len(layer.texts) > 2
