import pytest

from carla.agents import make_agent
from carla.engine.config import EngineConfig
from carla.engine.constants import DevCard
from carla.engine.record import GameRecord
from carla.engine.simulate import play_random_game
from carla.engine.state import Phase
from carla.gui.controller import ButtonSpec
from carla.gui.geometry import Rect, compute_layout
from carla.gui.panels import (
    HandView,
    LogView,
    card_tooltip,
    draw_hand_bar,
    draw_side_panel,
    draw_status,
    ellipsize,
    hand_view,
    layout_hand_bar,
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
    view, hand = panel_view(s), hand_view(s)
    me, other = view.players[0], view.players[1]
    assert hand.seat == 0 and hand.title == "Your hand"
    assert hand.resources == tuple(s.game.state.hands[0])
    assert hand.dev == ((DevCard.VICTORY_POINT, 1, 0),)
    assert me.vp_cards == 1 and me.vp == 3  # two setup settlements plus the hidden card
    assert other.vp_cards == 0 and other.vp == 2  # its two VP cards stay hidden
    assert (other.cards, other.dev_cards) == (s.game.state.hand_size(1), 3)  # sizes are public


def test_spectators_see_everything():
    s = session(viewer=None)
    view, hand = panel_view(s), hand_view(s)
    assert view.players[1].vp == 4 and view.players[1].vp_cards == 2
    assert hand.seat == s.game.decision_player == 0 and hand.title == "P0's hand"
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
    assert {
        "Bank",
        "P0  You",
        "1 VP card",
    } <= texts and "Your hand" not in texts  # see the hand bar
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


HAND = HandView(
    0, "Your hand", (2, 0, 1, 5, 3), ((DevCard.KNIGHT, 2, 1), (DevCard.VICTORY_POINT, 1, 0))
)
SPECS = [
    ButtonSpec("roll", "Roll", "R", False),
    ButtonSpec("end_turn", "End turn", "E", True),
    ButtonSpec("buy_dev", "Buy dev card", "B", True),
    ButtonSpec("play_dev", "Play dev card", "D", True),
    ButtonSpec("bank_trade", "Bank trade", "T", False),
]


def inside(inner: Rect, outer: Rect) -> bool:
    return outer.contains(inner.x, inner.y) and outer.contains(inner.right, inner.top)


def apart(a: Rect, b: Rect) -> bool:
    return a.right <= b.x or b.right <= a.x or a.top <= b.y or b.top <= a.y


def test_hand_bar_layout():
    bar = layout_hand_bar(LAYOUT.hand, HAND, SPECS, frozenset({DevCard.KNIGHT}))
    resources = [c for c in bar.cards if c.kind == "resource"]
    dev = [c for c in bar.cards if c.kind == "dev"]
    assert [(c.index, c.count) for c in resources] == list(enumerate(HAND.resources))
    # two knights, one bought this turn, then the victory point
    assert [(c.index, c.count, c.dimmed, c.playable) for c in dev] == [
        (DevCard.KNIGHT, 1, False, True),
        (DevCard.KNIGHT, 1, True, False),
        (DevCard.VICTORY_POINT, 1, False, False),
    ]
    assert [b.id for b in bar.buttons] == ["bank_trade", "buy_dev", "roll", "end_turn"]
    assert [(b.enabled, b.hotkey) for b in bar.buttons] == [
        (False, "T"),
        (True, "B"),
        (False, "R"),
        (True, "E"),
    ]
    rects = [c.rect for c in bar.cards] + [b.rect for b in bar.buttons]
    assert all(inside(r, LAYOUT.hand) for r in rects)
    assert all(apart(a, b) for i, a in enumerate(rects) for b in rects[i + 1 :])
    assert resources[-1].rect.right < bar.separator < dev[0].rect.x
    assert max(c.rect.right for c in bar.cards) < min(b.rect.x for b in bar.buttons)
    assert bar.buttons[-1].rect.right == pytest.approx(LAYOUT.hand.right - 10)  # by the log
    assert bar.card_at(dev[0].rect.cx, dev[0].rect.cy) == dev[0]
    assert bar.card_at(LAYOUT.hand.x + 2, LAYOUT.hand.y + 2) is None


def test_hand_bar_shrinks_then_overlaps_cards_in_a_narrow_window():
    lots = HandView(0, "Your hand", (19,) * 5, tuple((d, 3, 1) for d in DevCard))
    narrow = Rect(0, 0, 560, 84)
    bar = layout_hand_bar(narrow, lots, SPECS, frozenset(DevCard))
    dev = [c for c in bar.cards if c.kind == "dev"]
    assert len(dev) == 10 and all(inside(c.rect, narrow) for c in bar.cards)
    assert all(inside(b.rect, narrow) for b in bar.buttons)
    assert max(c.rect.right for c in bar.cards) < min(b.rect.x for b in bar.buttons)
    assert dev[1].rect.x < dev[0].rect.right  # fanned: they overlap ...
    assert bar.card_at(dev[0].rect.x + 2, dev[0].rect.cy) == dev[0]  # ... but stay reachable
    resources = [c.rect for c in bar.cards[:5]]
    assert all(apart(a, b) for i, a in enumerate(resources) for b in resources[i + 1 :])


def test_read_only_hand_bar_uses_the_whole_width():
    bar = layout_hand_bar(LAYOUT.hand, HAND)
    assert bar.buttons == () and not any(c.playable for c in bar.cards)
    assert all(inside(c.rect, LAYOUT.hand) for c in bar.cards)


def test_hand_bar_drawing_and_tooltips():
    bar = layout_hand_bar(LAYOUT.hand, HAND, SPECS, frozenset({DevCard.KNIGHT}))
    layer = Layer("ui")
    knight = next(c for c in bar.cards if c.kind == "dev")
    draw_hand_bar(layer, bar, (knight.rect.cx, knight.rect.cy))
    texts = [t.text for t in layer.texts]
    assert "Your hand · 11 cards" in texts and {"Trade", "Buy dev", "Roll", "End turn"} <= set(
        texts
    )
    assert texts.count("1") >= 4 and "5" in texts  # count badges (no badge on the empty slot)
    assert card_tooltip(knight) == "Knight: click to play"
    assert card_tooltip(bar.cards[3]) == "Grain ×5"
    new = next(c for c in bar.cards if c.dimmed)
    assert card_tooltip(new) == "Knight: bought this turn"
    vp = next(c for c in bar.cards if c.index == DevCard.VICTORY_POINT and c.kind == "dev")
    assert card_tooltip(vp) == "Victory Point: counts automatically"
    empty = Layer("ui")
    draw_hand_bar(empty, layout_hand_bar(LAYOUT.hand, HandView(2, "P2's hand", (0,) * 5, ())))
    assert "No development cards" in {t.text for t in empty.texts}


def test_action_tiles_show_their_labels_all_or_none():
    assert layout_hand_bar(LAYOUT.hand, HAND, SPECS).labelled
    lots = HandView(0, "Your hand", (19,) * 5, tuple((d, 3, 1) for d in DevCard))
    narrow = layout_hand_bar(Rect(0, 0, 560, 84), lots, SPECS, frozenset(DevCard))
    assert not narrow.labelled
    layer = Layer("ui")
    draw_hand_bar(layer, narrow)
    assert not {"Trade", "Buy dev", "Roll", "End turn"} & {t.text for t in layer.texts}
    assert {"T", "B", "R", "E"} <= {t.text for t in layer.texts}  # hotkeys stay
