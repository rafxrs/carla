"""The trade panel: layout, hit-testing and drawing (pure)."""

import pytest

from carla.gui import theme
from carla.gui.controller import TradeButton, TradePanel, TradeRow
from carla.gui.geometry import compute_layout
from carla.gui.render import Layer
from carla.gui.trade import draw_trade, layout_trade, trade_tooltip

COMPOSER = TradePanel(
    "offer",
    "Trade",
    (
        TradeButton("invite", "Ask for offers"),
        TradeButton("bank", "Trade with bank", False, True),
        TradeButton("propose", "Offer to players", True, True),
    ),
    "Bank: 1 card for 4 lumber",
    give=(2, 0, 1, 0, 0),
    get=(0, 0, 0, 0, 1),
    wants=(False, True, False, True, True),
)
ANSWERS = TradePanel(
    "responses",
    "Your trade offer",
    (TradeButton("cancel_trade", "Cancel trade"),),
    "Click Trade beside the deal you want",
    rows=(
        TradeRow(0, "You", "offers", (2, 0, 0, 0, 0), (0, 0, 0, 1, 0)),
        TradeRow(
            1,
            "P1",
            "accepts",
            (2, 0, 0, 0, 0),
            (0, 0, 0, 1, 0),
            TradeButton(("confirm", 1), "Trade"),
        ),
        TradeRow(
            2,
            "P2",
            "counters",
            (1, 1, 1, 1, 0),
            (0, 0, 0, 0, 4),
            TradeButton(("confirm", 2), "Trade", False),
        ),
        TradeRow(3, "P3", "declines"),
    ),
)


def inside(inner, outer) -> bool:
    return outer.contains(inner.x, inner.y) and outer.contains(inner.right, inner.top)


@pytest.mark.parametrize("size", [(1280, 800), (1000, 660)])
@pytest.mark.parametrize("panel", [COMPOSER, ANSWERS], ids=["composer", "answers"])
def test_the_panel_sits_above_the_hand_and_holds_its_parts(panel, size):
    area = compute_layout(*size).viewport
    layout = layout_trade(panel, area)
    p = layout.panel
    assert inside(p, area) and p.y == pytest.approx(area.y + 8)  # docked above the hand bar
    parts = [c.rect for c, _ in layout.cards] + [r for _, r in layout.buttons] + list(layout.rows)
    assert all(inside(r, p) for r in parts)
    buttons = [r for _, r in layout.buttons]
    for i, a in enumerate(buttons):
        for b in buttons[i + 1 :]:
            assert a.right <= b.x or b.right <= a.x or a.top <= b.y or b.top <= a.y


def test_composer_cards_and_hits():
    layout = layout_trade(COMPOSER, compute_layout(1280, 800).viewport)
    keys = [k for _, k in layout.cards]
    assert keys == [("give", 0), ("give", 2), ("get", 4), *(("want", r) for r in range(5))]
    give_row, get_row = layout.rows
    lumber, wool = (c for c, k in layout.cards if isinstance(k, tuple) and k[0] == "give")
    assert (lumber.count, wool.count) == (2, 1) and lumber.rect.right < wool.rect.x
    assert get_row.top < give_row.y  # "You give" above "You get"
    picks = [c for c, k in layout.cards if isinstance(k, tuple) and k[0] == "want"]
    assert [c.playable for c in picks] == list(COMPOSER.wants)
    assert [c.dimmed for c in picks] == [not w for w in COMPOSER.wants]
    assert layout.key_at(lumber.rect.cx, lumber.rect.cy) == ("give", 0)
    assert layout.key_at(picks[0].rect.cx, picks[0].rect.cy) is None  # lumber is being given
    assert layout.key_at(picks[4].rect.cx, picks[4].rect.cy) == ("want", 4)
    by_key = {b.key: r for b, r in layout.buttons}
    assert layout.key_at(by_key["bank"].cx, by_key["bank"].cy) is None  # disabled
    assert layout.key_at(by_key["propose"].cx, by_key["propose"].cy) == "propose"
    assert layout.close is not None and layout.key_at(layout.close.cx, layout.close.cy) == "close"
    assert layout.key_at(layout.panel.x + 3, layout.panel.y + 3) is None


def test_negotiation_rows_show_terms_as_cards():
    layout = layout_trade(ANSWERS, compute_layout(1280, 800).viewport)
    assert len(layout.rows) == 4 and layout.close is None and layout.picks is None
    assert all(k is None and not c.playable for c, k in layout.cards)  # shown, not clicked
    give_x, get_x = layout.columns
    p2 = layout.rows[2]
    cards = [c for c, _ in layout.cards if p2.contains(c.rect.cx, c.rect.cy)]
    assert [(c.index, c.count) for c in cards] == [(0, 1), (1, 1), (2, 1), (3, 1), (4, 4)]
    assert all(c.rect.x >= give_x for c in cards[:4]) and cards[4].rect.x == pytest.approx(get_x)
    assert cards[3].rect.right < get_x  # four cards fit their column
    trades = [(b.key, r) for b, r in layout.buttons if b.key != "cancel_trade"]
    assert [k for k, _ in trades] == [("confirm", 1), ("confirm", 2)]
    (_, p1), (_, p2_button) = trades
    assert layout.rows[1].contains(p1.cx, p1.cy)
    assert layout.key_at(p1.cx, p1.cy) == ("confirm", 1)
    assert layout.key_at(p2_button.cx, p2_button.cy) is None  # P0 cannot pay that


def test_drawing_shows_titles_cards_and_numbered_buttons():
    layout = layout_trade(COMPOSER, compute_layout(1280, 800).viewport)
    want = next(c for c, k in layout.cards if k == ("want", 4))
    layer = Layer("trade")
    draw_trade(layer, COMPOSER, layout, (want.rect.cx, want.rect.cy))
    texts = [t.text for t in layer.texts]
    assert {"Trade", "You give", "You get", "Ask for", COMPOSER.hint, "Esc"} <= set(texts)
    assert {"Ask for offers", "Trade with bank", "Offer to players", "1", "2", "3"} <= set(texts)
    assert texts.count("+") == 5 and "2" in texts  # a plus on each resource to ask for
    assert theme.ACCENT in {tuple(layer.data[i + 2 : i + 6]) for i in range(0, len(layer.data), 6)}
    answers = Layer("trade")
    draw_trade(answers, ANSWERS, layout_trade(ANSWERS, compute_layout(1000, 660).viewport))
    texts = [t.text for t in answers.texts]
    assert {"Your trade offer", "You give", "You get", "accepts", "counters", "declines"} <= set(
        texts
    )
    assert texts.count("Trade") == 2 and "Cancel trade" in texts


def test_tooltips_say_what_a_click_does():
    layout = layout_trade(COMPOSER, compute_layout(1280, 800).viewport)
    tips = {k: trade_tooltip(c, k) for c, k in layout.cards}
    assert tips[("give", 0)] == "Lumber ×2: click to take one back"
    assert tips[("get", 4)] == "Ore ×1: click to take one off"
    assert tips[("want", 1)] == "Ask for one brick"
    assert tips[("want", 0)] == ""  # not available: no tooltip
    shown = layout_trade(ANSWERS, compute_layout(1280, 800).viewport).cards[0]
    assert trade_tooltip(*shown) == "Lumber ×2"
