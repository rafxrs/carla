import pytest

from carla.engine import actions as A
from carla.engine.actions import TradeOffer
from carla.engine.config import EngineConfig
from carla.engine.constants import DevCard
from carla.engine.state import Phase
from carla.gui import theme
from carla.gui.controller import Controller, Option, Picker
from carla.gui.geometry import Rect
from carla.gui.render import Layer
from carla.gui.widgets import (
    Button,
    Card,
    Slider,
    button_at,
    card_at,
    draw_button,
    draw_card,
    draw_dev_glyph,
    draw_icon,
    draw_picker,
    draw_resource_glyph,
    draw_slider,
    draw_tile_button,
    draw_tooltip,
    layout_picker,
)
from tests.engine.scenario import give, give_dev, post_setup_game

AREA = Rect(10, 62, 880, 698)


def test_buttons_hit_and_draw():
    ok = Button("roll", "Roll", Rect(0, 0, 80, 30), hotkey="R")
    off = Button("end_turn", "End turn", Rect(90, 0, 100, 30), enabled=False)
    assert button_at([ok, off], 40, 15) is ok and button_at([ok, off], 100, 10) is off
    assert button_at([ok, off], 85, 15) is None
    layer = Layer("ui")
    draw_button(layer, ok, hovered=True)
    draw_button(layer, off)
    texts = {t.text: t for t in layer.texts}
    assert set(texts) == {"Roll", "R", "End turn"}
    assert texts["End turn"].color == theme.BUTTON_TEXT_DISABLED


ICONS = ["play", "pause", "back", "forward", "start", "end", "minus", "plus"]


@pytest.mark.parametrize("icon", [*ICONS, "trade", "buy_card", "dice", "hourglass"])
def test_icons_are_drawn_from_shapes(icon):
    layer = Layer("ui")
    draw_icon(layer, icon, 50, 50, 20, theme.TEXT)
    assert layer.vertex_count >= 3 and not layer.texts
    xs = [layer.data[i] for i in range(0, len(layer.data), 6)]
    assert min(xs) >= 50 - 20 and max(xs) <= 50 + 20


def test_slider():
    slider = Slider(Rect(100, 0, 200, 20), value=25, maximum=100)
    assert slider.knob_x() == pytest.approx(150)
    assert slider.value_at(100) == 0 and slider.value_at(300) == 100
    assert slider.value_at(201) == 50 and slider.value_at(-50) == 0 and slider.value_at(999) == 100
    assert Slider(Rect(0, 0, 100, 20), 0, 0).value_at(70) == 0
    layer = Layer("ui")
    draw_slider(layer, slider, hovered=True)
    assert layer.vertex_count > 0


def pickers() -> list[Picker]:
    """One picker of every kind, from real positions."""
    out = []
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    for card in DevCard:
        give_dev(game, 0, card)
    give(game, 0, (4, 4, 4, 4, 4))
    controller = Controller(0, domestic_trade=True)
    for button, choices in (("play_dev", []), ("play_dev", [2]), ("play_dev", [3])):
        controller.reset(game.state, game.legal_actions())
        controller.press(button)
        for key in choices:
            controller.choose(key)
        assert controller.picker is not None
        out.append(controller.picker)
    for button in ("bank_trade", "trade"):
        controller.reset(game.state, game.legal_actions())
        controller.press(button)
        assert controller.picker is not None
        out.append(controller.picker)
    controller.choose(("give", 0, 1))
    assert controller.picker is not None
    out.append(controller.picker)
    s = game.state
    s.phase, s.pending_trade = Phase.TRADE_RESPONSE, TradeOffer((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))
    controller.reset(s, [A.ACCEPT_TRADE, A.REJECT_TRADE])
    assert controller.picker is not None
    out.append(controller.picker)
    controller.reset(s, [A.CONFIRM_TRADE_WITH[1], A.CANCEL_TRADE])
    assert controller.picker is not None
    out.append(controller.picker)
    out.append(Picker("victim", "Steal a card from", (Option(1, "P1", player=1), Option(2, "P2"))))
    return out


@pytest.mark.parametrize("picker", pickers(), ids=lambda p: p.kind)
def test_picker_layout_and_drawing(picker):
    layout = layout_picker(picker, AREA)
    p = layout.panel
    assert AREA.contains(p.x, p.y) and AREA.contains(p.right, p.top)
    rects = [r for _, r in layout.options]
    for i, r in enumerate(rects):
        assert p.contains(r.x, r.y) and p.contains(r.right, r.top)
        for other in rects[i + 1 :]:  # no two option cells overlap
            assert r.right <= other.x or other.right <= r.x or r.top <= other.y or other.top <= r.y
    for option, r in layout.options:
        assert layout.option_at(r.cx, r.cy) == option
    assert layout.option_at(p.x - 5, p.y - 5) is None
    assert (layout.cancel is not None) == (picker.cancellable or picker.kind == "offer")
    layer = Layer("picker")
    draw_picker(
        layer, picker, layout, hovered=picker.options[0].key, offer=((1, 0, 0, 0, 0), (0,) * 5)
    )
    assert picker.title in {t.text for t in layer.texts}


def test_offer_composer_has_a_stepper_per_resource_and_side():
    picker = next(p for p in pickers() if p.kind == "offer")
    layout = layout_picker(picker, AREA)
    keys = {o.key for o, _ in layout.options}
    assert keys == {(side, r, d) for side in ("give", "get") for r in range(5) for d in (1, -1)} | {
        "propose",
        "invite",
    }


def test_three_trade_partners_fit_the_smallest_window():
    picker = Picker(
        "confirm",
        "Complete your trade",
        (
            *(
                Option(
                    seat,
                    f"Trade with P{seat}",
                    detail="Give 2 lumber, 1 wool, 1 grain\nGet 2 brick, 1 grain, 1 ore",
                    player=seat,
                )
                for seat in (1, 2, 3)
            ),
            Option("cancel", "Cancel trade"),
        ),
        cancellable=False,
    )
    area = Rect(0, 0, 640, 450)
    layout = layout_picker(picker, area)
    assert area.contains(layout.panel.x, layout.panel.y)
    assert area.contains(layout.panel.right, layout.panel.top)
    assert all(
        layout.panel.contains(rect.x, rect.y) and layout.panel.contains(rect.right, rect.top)
        for _, rect in layout.options
    )
    layer = Layer("picker")
    draw_picker(layer, picker, layout)
    assert len([item for item in layer.texts if item.text.startswith("Give 2 lumber")]) == 3


def test_resource_pickers_use_five_columns():
    picker = Picker("monopoly", "Monopoly", tuple(Option(r, f"r{r}", resource=r) for r in range(5)))
    layout = layout_picker(picker, AREA)
    assert len({r.y for _, r in layout.options}) == 1  # a single row


def vertices(layer: Layer) -> list[tuple[float, float]]:
    return [(layer.data[i], layer.data[i + 1]) for i in range(0, len(layer.data), 6)]


def colours(layer: Layer) -> set[tuple[float, ...]]:
    return {tuple(layer.data[i + 2 : i + 6]) for i in range(0, len(layer.data), 6)}


@pytest.mark.parametrize("resource", range(5))
def test_resource_glyphs_are_one_colour_inside_their_box(resource):
    layer = Layer("ui")
    draw_resource_glyph(layer, resource, 100, 50, 40, theme.INK)
    assert layer.vertex_count > 0 and colours(layer) == {theme.INK}
    assert all(abs(x - 100) <= 22 and abs(y - 50) <= 22 for x, y in vertices(layer))


@pytest.mark.parametrize("card", list(DevCard))
def test_development_glyphs_use_the_ink_and_cut_details_in_the_face(card):
    layer = Layer("ui")
    draw_dev_glyph(layer, card, 100, 50, 40, theme.TEXT, theme.CARD_BACK)
    assert layer.vertex_count > 0 and colours(layer) <= {theme.TEXT, theme.CARD_BACK}
    assert all(abs(x - 100) <= 22 and abs(y - 50) <= 22 for x, y in vertices(layer))


def test_cards_show_their_symbol_and_count():
    rect = Rect(100, 20, 46, 64)
    lumber = Layer("ui")
    draw_card(lumber, Card("resource", 0, 3, rect))
    assert [t.text for t in lumber.texts] == ["3"] and theme.TERRAIN[0] in colours(lumber)
    empty = Layer("ui")
    draw_card(empty, Card("resource", 1, 0, rect))  # none held: an outlined, faint slot
    assert not empty.texts and theme.TERRAIN[1] not in colours(empty)
    assert 0 < empty.vertex_count < lumber.vertex_count
    knight = Layer("ui")
    draw_card(knight, Card("dev", DevCard.KNIGHT, 2, rect))
    new = Layer("ui")
    draw_card(new, Card("dev", DevCard.KNIGHT, 1, rect, dimmed=True))
    assert theme.CARD_BACK in colours(knight) and theme.CARD_BACK not in colours(new)
    assert Card("dev", DevCard.MONOPOLY, 1, rect).label == "Monopoly"
    assert Card("resource", 4, 1, rect).label == "Ore"


def test_hovered_cards_lift_and_playable_ones_light_up():
    rect = Rect(100, 20, 46, 64)
    card = Card("dev", DevCard.KNIGHT, 1, rect, playable=True)
    still, hovered = Layer("ui"), Layer("ui")
    draw_card(still, card)
    draw_card(hovered, card, hovered=True)
    assert min(y for _, y in vertices(hovered)) == pytest.approx(
        min(y for _, y in vertices(still)) + 4
    )
    assert theme.ACCENT in colours(hovered) and theme.ACCENT not in colours(still)
    held = Layer("ui")
    draw_card(held, Card("dev", DevCard.KNIGHT, 1, rect), hovered=True)  # not playable now
    assert theme.ACCENT not in colours(held)


def test_card_at_prefers_the_card_drawn_last():
    first = Card("dev", 0, 1, Rect(0, 0, 40, 60))
    second = Card("dev", 1, 1, Rect(20, 0, 40, 60))  # overlaps the right half of the first
    assert card_at([first, second], 10, 30) is first
    assert card_at([first, second], 30, 30) is second
    assert card_at([first, second], 70, 30) is None


def test_tile_buttons_and_tooltips():
    roomy = Button("roll", "Roll", Rect(0, 0, 76, 80), hotkey="R", icon="dice", accent=True)
    layer = Layer("ui")
    draw_tile_button(layer, roomy, hovered=True)
    assert [t.text for t in layer.texts] == ["Roll", "R"] and theme.ACCENT in colours(layer)
    small = Layer("ui")
    draw_tile_button(
        small, Button("buy_dev", "Buy dev", Rect(0, 0, 40, 40), False, "B", "buy_card")
    )
    assert not small.texts and theme.BUTTON_DISABLED in colours(small)  # icon only
    tip = Layer("ui")
    draw_tooltip(tip, "Year of Plenty", 5, 100, Rect(0, 0, 300, 200))
    assert tip.texts[0].text == "Year of Plenty"
    assert all(x >= 0 and y > 100 for x, y in vertices(tip))  # kept inside, above the point
