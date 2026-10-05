import pytest

from carla.engine import actions as A
from carla.engine.actions import TradeOffer
from carla.engine.constants import DevCard
from carla.engine.state import Phase
from carla.gui import theme
from carla.gui.controller import Controller, Option, Picker
from carla.gui.geometry import Rect
from carla.gui.render import Layer
from carla.gui.widgets import (
    Button,
    Slider,
    button_at,
    draw_button,
    draw_icon,
    draw_picker,
    draw_slider,
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


@pytest.mark.parametrize(
    "icon", ["play", "pause", "back", "forward", "start", "end", "minus", "plus"]
)
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
    game = post_setup_game()
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
        "propose"
    }


def test_resource_pickers_use_five_columns():
    picker = Picker("monopoly", "Monopoly", tuple(Option(r, f"r{r}", resource=r) for r in range(5)))
    layout = layout_picker(picker, AREA)
    assert len({r.y for _, r in layout.options}) == 1  # a single row
