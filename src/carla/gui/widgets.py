"""Buttons, the seek slider, picker dialogs and cards: layout and hit-testing (pure) plus
drawing into a `Layer`. Icons and card art are drawn from triangles, circles and bars, never
from image files."""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from math import cos, radians, sin

from carla.engine.constants import DevCard, Resource
from carla.gui import theme
from carla.gui.controller import Option, Picker
from carla.gui.geometry import Point, Rect
from carla.gui.render import Layer, scaled
from carla.gui.theme import Color

RADIUS = 6


@dataclass(frozen=True)
class Button:
    id: str
    label: str
    rect: Rect
    enabled: bool = True
    hotkey: str = ""
    icon: str = ""
    """Glyph (see `draw_icon`): drawn instead of the label, or above it on a tile button."""
    accent: bool = False


def button_at(buttons: Sequence[Button], x: float, y: float) -> Button | None:
    return next((b for b in buttons if b.rect.contains(x, y)), None)


def draw_button(layer: Layer, button: Button, hovered: bool = False) -> None:
    r = button.rect
    if not button.enabled:
        fill, ink = theme.BUTTON_DISABLED, theme.BUTTON_TEXT_DISABLED
    else:
        fill, ink = (theme.BUTTON_HOVER if hovered else theme.BUTTON), theme.TEXT
    layer.rounded_rect(r.x, r.y, r.w, r.h, RADIUS, fill)
    if button.accent and button.enabled:
        layer.rounded_frame(r.x, r.y, r.w, r.h, RADIUS, 2, theme.ACCENT)
    if button.icon:
        draw_icon(layer, button.icon, r.cx, r.cy, min(r.w, r.h) * 0.42, ink)
        return
    hint = button.hotkey and r.w > 70
    x = r.cx - (7 if hint else 0)
    layer.text(button.label, x, r.cy, theme.FONT, ink, button.enabled, "center", "center")
    if hint:
        layer.text(
            button.hotkey,
            r.right - 9,
            r.y + 6,
            theme.FONT_SMALL - 1,
            theme.TEXT_DIM,
            False,
            "right",
            "bottom",
        )


def draw_icon(layer: Layer, icon: str, cx: float, cy: float, size: float, color: Color) -> None:
    """Playback, stepper and action glyphs, `size` pixels tall: play, pause, back, forward,
    start, end, minus, plus, trade, buy_card, dice, hourglass."""
    h = size / 2

    def play(x: float, direction: int = 1) -> None:
        layer.triangle(
            (x - h * 0.8 * direction, cy - h),
            (x - h * 0.8 * direction, cy + h),
            (x + h * 0.8 * direction, cy),
            color,
        )

    def bar(x: float) -> None:
        layer.rect(x - h * 0.17, cy - h, h * 0.34, size, color)

    if icon == "play":
        play(cx + h * 0.1)
    elif icon == "pause":
        bar(cx - h * 0.35)
        bar(cx + h * 0.35)
    elif icon in ("back", "forward"):
        d = 1 if icon == "forward" else -1
        play(cx - h * 0.25 * d, d)
        bar(cx + h * 0.75 * d)
    elif icon in ("start", "end"):
        d = 1 if icon == "end" else -1
        play(cx - h * 0.75 * d, d)
        play(cx + h * 0.35 * d, d)
        bar(cx + h * 1.2 * d)
    elif icon in ("minus", "plus"):
        layer.rect(cx - h, cy - h * 0.16, size, h * 0.32, color)
        if icon == "plus":
            layer.rect(cx - h * 0.16, cy - h, h * 0.32, size, color)
    elif icon == "trade":  # one arrow each way
        for d, y in ((1, cy + h * 0.4), (-1, cy - h * 0.4)):
            neck = cx + h * 0.3 * d
            layer.line(cx - h * 0.95 * d, y, neck, y, h * 0.26, color)
            layer.triangle(
                (neck, y + h * 0.38), (neck, y - h * 0.38), (cx + h * 0.95 * d, y), color
            )
    elif icon == "buy_card":  # a card with a plus on it
        layer.rounded_frame(
            cx - h * 0.62, cy - h * 0.9, h * 1.24, h * 1.8, h * 0.2, h * 0.16, color
        )
        layer.rect(cx - h * 0.4, cy - h * 0.08, h * 0.8, h * 0.16, color)
        layer.rect(cx - h * 0.08, cy - h * 0.4, h * 0.16, h * 0.8, color)
    elif icon == "dice":
        layer.rounded_frame(
            cx - h * 0.85, cy - h * 0.85, h * 1.7, h * 1.7, h * 0.3, h * 0.15, color
        )
        for dx, dy in ((-1, 1), (1, 1), (0, 0), (-1, -1), (1, -1)):
            layer.circle(cx + dx * h * 0.4, cy + dy * h * 0.4, h * 0.13, color)
    elif icon == "hourglass":
        for d in (1, -1):
            layer.rect(cx - h * 0.62, cy + d * h * 0.82 - h * 0.08, h * 1.24, h * 0.16, color)
            neck = (cx - h * 0.48, cy + d * h * 0.74), (cx + h * 0.48, cy + d * h * 0.74)
            layer.triangle(*neck, (cx, cy), color)


@dataclass(frozen=True)
class Slider:
    """Seek bar over positions 0..maximum."""

    rect: Rect
    value: int
    maximum: int

    def value_at(self, x: float) -> int:
        if self.maximum <= 0 or self.rect.w <= 0:
            return 0
        fraction = (x - self.rect.x) / self.rect.w
        return max(0, min(self.maximum, round(fraction * self.maximum)))

    def knob_x(self) -> float:
        fraction = self.value / self.maximum if self.maximum > 0 else 0.0
        return self.rect.x + fraction * self.rect.w


def draw_slider(layer: Layer, slider: Slider, hovered: bool = False) -> None:
    r = slider.rect
    track_h = 6
    y = r.cy - track_h / 2
    layer.rounded_rect(r.x, y, r.w, track_h, 3, theme.BUTTON)
    knob = slider.knob_x()
    if knob > r.x:
        layer.rounded_rect(r.x, y, knob - r.x, track_h, 3, theme.ACCENT)
    radius = 9 if hovered else 7
    layer.circle(knob, r.cy, radius + 2, theme.BACKGROUND)
    layer.circle(knob, r.cy, radius, theme.ACCENT if hovered else theme.TEXT)


# ------------------------------------------------------------------ pickers


@dataclass(frozen=True)
class PickerLayout:
    panel: Rect
    options: tuple[tuple[Option, Rect], ...]
    cancel: Rect | None

    def option_at(self, x: float, y: float) -> Option | None:
        return next((o for o, r in self.options if r.contains(x, y)), None)


_TITLE_H, _DETAIL_H, _PAD, _GAP = 34, 22, 16, 10


def layout_picker(picker: Picker, area: Rect) -> PickerLayout:
    """Centre the dialog in `area`: a grid of option cells and an optional Cancel button."""
    if picker.kind == "offer":
        return _layout_offer(picker, area)
    resources = any(o.resource >= 0 for o in picker.options)
    columns = min(len(picker.options), 5 if resources else 3) or 1
    cell_w, cell_h = (104, 74) if resources else (176, 52)
    rows = (len(picker.options) + columns - 1) // columns
    width = max(columns * cell_w + (columns - 1) * _GAP, 320) + 2 * _PAD
    height = _TITLE_H + (_DETAIL_H if picker.detail else 0) + rows * (cell_h + _GAP) + 2 * _PAD
    height += 44 if picker.cancellable else 0
    width, height = min(width, area.w - 20), min(height, area.h - 20)
    panel = Rect(area.cx - width / 2, area.cy - height / 2, width, height)
    top = panel.top - _PAD - _TITLE_H - (_DETAIL_H if picker.detail else 0)
    grid_w = columns * cell_w + (columns - 1) * _GAP
    left = panel.cx - grid_w / 2
    cells = []
    for i, option in enumerate(picker.options):
        row, col = divmod(i, columns)
        in_row = min(columns, len(picker.options) - row * columns)  # centre a short last row
        x0 = left + (columns - in_row) * (cell_w + _GAP) / 2
        y = top - (row + 1) * cell_h - row * _GAP
        cells.append((option, Rect(x0 + col * (cell_w + _GAP), y, cell_w, cell_h)))
    cancel = Rect(panel.cx - 70, panel.y + _PAD, 140, 32) if picker.cancellable else None
    return PickerLayout(panel, tuple(cells), cancel)


def _layout_offer(picker: Picker, area: Rect) -> PickerLayout:
    """Trade offer composer: per resource a give and a get counter with +/- buttons."""
    width, height = 600, 300
    panel = Rect(area.cx - width / 2, area.cy - height / 2, width, height)
    by_key = {o.key: o for o in picker.options}
    cells = []
    col_w = (width - 2 * _PAD - 110) / 5
    for side, row_y in (("give", panel.top - 120), ("get", panel.top - 190)):
        for r in range(5):
            x = panel.x + _PAD + 110 + r * col_w
            for delta, dx in ((-1, 6), (1, col_w - 34)):
                option = by_key.get((side, r, delta))
                if option is not None:
                    cells.append((option, Rect(x + dx, row_y, 28, 28)))
    propose = by_key.get("propose")
    if propose is not None:
        cells.append((propose, Rect(panel.right - _PAD - 150, panel.y + _PAD, 150, 34)))
    cancel = Rect(panel.x + _PAD, panel.y + _PAD, 120, 34)
    return PickerLayout(panel, tuple(cells), cancel)


def draw_picker(
    layer: Layer,
    picker: Picker,
    layout: PickerLayout,
    hovered: Hashable = None,
    offer: tuple[Sequence[int], Sequence[int]] | None = None,
) -> None:
    """The dialog, its options and Cancel; `hovered` is the key of the option under the mouse
    ("cancel" for the Cancel button); `offer` gives the counters of the trade composer."""
    p = layout.panel
    layer.rounded_rect(p.x + 3, p.y - 4, p.w, p.h, 10, theme.with_alpha(theme.BLACK, 90))
    layer.rounded_rect(p.x, p.y, p.w, p.h, 10, theme.PANEL)
    layer.rounded_frame(p.x, p.y, p.w, p.h, 10, 1.5, theme.PANEL_EDGE)
    layer.text(
        picker.title,
        p.cx,
        p.top - _PAD - 14,
        theme.FONT_LARGE,
        theme.TEXT,
        True,
        "center",
        "center",
    )
    if picker.detail:
        layer.text(
            picker.detail,
            p.cx,
            p.top - _PAD - _TITLE_H - 6,
            theme.FONT,
            theme.TEXT_MUTED,
            False,
            "center",
            "center",
        )
    if picker.kind == "offer":
        _draw_offer(layer, layout, offer or ((0,) * 5, (0,) * 5))
    numbered = picker.kind != "offer" and len(layout.options) <= 9
    for i, (option, rect) in enumerate(layout.options):
        _draw_option(layer, option, rect, option.key == hovered, i + 1 if numbered else 0)
    if layout.cancel is not None:
        c = layout.cancel
        label = "Cancel (Esc)" if picker.kind != "offer" else "Cancel"
        draw_button(layer, Button("cancel", label, c), hovered == "cancel")


def _draw_option(layer: Layer, option: Option, r: Rect, hovered: bool, number: int) -> None:
    enabled = option.enabled
    stepper = option.label in ("+", "-")
    if option.resource >= 0 and not stepper:
        _draw_resource_option(layer, option, r, hovered)
    else:
        if stepper:
            fill = theme.BUTTON_HOVER if hovered and enabled else theme.BUTTON
            fill = fill if enabled else theme.BUTTON_DISABLED
            layer.rounded_rect(r.x, r.y, r.w, r.h, 5, fill)
            ink = theme.TEXT if enabled else theme.BUTTON_TEXT_DISABLED
            draw_icon(
                layer, "plus" if option.label == "+" else "minus", r.cx, r.cy, r.h * 0.45, ink
            )
            return
        draw_button(layer, Button(str(option.key), option.label, r, enabled), hovered)
        if option.player >= 0:
            layer.circle(r.x + 16, r.cy, 7, theme.PLAYER[option.player])
        if option.detail:
            ink = theme.TEXT_MUTED if enabled else theme.BUTTON_TEXT_DISABLED
            layer.text(
                option.detail, r.cx, r.y + 7, theme.FONT_SMALL, ink, False, "center", "bottom"
            )
    if number:
        layer.text(
            str(number),
            r.x + 6,
            r.top - 5,
            theme.FONT_SMALL - 1,
            theme.TEXT_DIM,
            False,
            "left",
            "top",
        )


def _draw_resource_option(layer: Layer, option: Option, r: Rect, hovered: bool) -> None:
    enabled = option.enabled
    fill = theme.BUTTON_HOVER if hovered and enabled else theme.BUTTON
    layer.rounded_rect(r.x, r.y, r.w, r.h, RADIUS, fill if enabled else theme.BUTTON_DISABLED)
    if hovered and enabled:
        layer.rounded_frame(r.x, r.y, r.w, r.h, RADIUS, 2, theme.ACCENT)
    chip = theme.TERRAIN[option.resource]
    alpha = 255 if enabled else 70
    resource_chip(layer, option.resource, r.cx, r.top - 22, 26, theme.with_alpha(chip, alpha))
    ink = theme.TEXT if enabled else theme.BUTTON_TEXT_DISABLED
    layer.text(option.label, r.cx, r.y + 22, theme.FONT_SMALL + 1, ink, enabled, "center", "center")
    if option.detail:
        muted = theme.TEXT_MUTED if enabled else theme.BUTTON_TEXT_DISABLED
        layer.text(
            option.detail, r.cx, r.y + 8, theme.FONT_SMALL - 1, muted, False, "center", "center"
        )


def _draw_offer(
    layer: Layer, layout: PickerLayout, offer: tuple[Sequence[int], Sequence[int]]
) -> None:
    p = layout.panel
    col_w = (p.w - 2 * _PAD - 110) / 5
    for r in range(5):
        x = p.x + _PAD + 110 + r * col_w + col_w / 2
        resource_chip(layer, r, x, p.top - 82, 24, theme.TERRAIN[r])
    for (label, counts), y in zip(
        (("You give", offer[0]), ("You get", offer[1])), (p.top - 106, p.top - 176), strict=True
    ):
        layer.text(label, p.x + _PAD, y, theme.FONT, theme.TEXT_MUTED, False, "left", "center")
        for r in range(5):
            x = p.x + _PAD + 110 + r * col_w + col_w / 2
            layer.text(str(counts[r]), x, y, theme.FONT_LARGE, theme.TEXT, True, "center", "center")


def resource_chip(
    layer: Layer, resource: int, cx: float, cy: float, size: float, fill: Color
) -> None:
    """Rounded square in the terrain colour with the resource's initial."""
    layer.rounded_rect(cx - size / 2, cy - size / 2, size, size, size * 0.22, fill)
    ink = theme.with_alpha(theme.ink_on(resource), fill[3])
    layer.text(
        theme.RESOURCE_INITIALS[resource], cx, cy, size * 0.42, ink, True, "center", "center"
    )


# ------------------------------------------------------------------ cards


CARD_RATIO = 0.72
"""Width over height of a card."""


@dataclass(frozen=True)
class Card:
    """A card of the hand bar: a resource or a development card and how many are held."""

    kind: str
    """"resource" or "dev"."""
    index: int
    """The `Resource` or `DevCard`."""
    count: int
    rect: Rect
    dimmed: bool = False
    """Development cards bought this turn, which cannot be played yet."""
    playable: bool = False
    """A click plays it."""

    @property
    def label(self) -> str:
        labels = theme.RESOURCE_LABELS if self.kind == "resource" else theme.DEV_LABELS
        return labels[self.index]


def card_at(cards: Sequence[Card], x: float, y: float) -> Card | None:
    """The card under the point; the last drawn wins where cards overlap."""
    return next((c for c in reversed(cards) if c.rect.contains(x, y)), None)


def draw_card(layer: Layer, card: Card, hovered: bool = False) -> None:
    """An upright card with its symbol and a count badge; an empty slot when none is held."""
    r = card.rect
    radius = max(3.0, r.w * 0.14)
    resource = card.kind == "resource"
    if card.count <= 0:  # keeps the row in place
        layer.rounded_frame(r.x, r.y, r.w, r.h, radius, 1.5, theme.PANEL_EDGE)
        if resource:
            faint = theme.with_alpha(theme.TEXT_DIM, 150)
            draw_resource_glyph(layer, card.index, r.cx, r.cy, r.w * 0.6, faint)
        return
    lift = 4.0 if hovered else 0.0
    x, y, w, h = r.x, r.y + lift, r.w, r.h
    if resource:
        face, ink = theme.TERRAIN[card.index], theme.with_alpha(theme.ink_on(card.index), 235)
    else:
        face = theme.CARD_BACK
        ink = theme.ACCENT if card.index == DevCard.VICTORY_POINT else theme.TEXT
    if card.dimmed:
        face, ink = theme.mix(face, theme.PANEL, 0.6), theme.mix(ink, theme.PANEL, 0.55)
    layer.rounded_rect(x + 2, y - 3, w, h, radius, theme.with_alpha(theme.BLACK, 80))
    layer.rounded_rect(x, y, w, h, radius, face)
    edge = theme.mix(face, theme.WHITE, 0.3)
    layer.rounded_frame(x + 3, y + 3, w - 6, h - 6, max(1.0, radius - 3), 1.2, edge)
    size = min(w, h) * 0.62
    if resource:
        draw_resource_glyph(layer, card.index, x + w / 2, y + h * 0.47, size, ink)
    else:
        draw_dev_glyph(layer, card.index, x + w / 2, y + h * 0.47, size, ink, face)
    if hovered and card.playable:
        layer.rounded_frame(x, y, w, h, radius, 2, theme.ACCENT)
    badge = max(7.0, w * 0.19)
    bx, by = x + badge * 0.55, y + h - badge * 0.55
    layer.circle(bx, by, badge + 1.5, theme.PANEL_EDGE)
    layer.circle(bx, by, badge, theme.PANEL)
    layer.text(str(card.count), bx, by, theme.FONT_SMALL, theme.TEXT, True, "center", "center")


def draw_resource_glyph(
    layer: Layer, resource: int, cx: float, cy: float, size: float, color: Color
) -> None:
    """Symbol of a resource in one colour, about `size` pixels tall: a fir tree, bricks, a
    sheep, an ear of grain, mountains (the board's terrain motifs, simplified)."""
    s = size

    def p(dx: float, dy: float) -> Point:
        return cx + dx * s, cy + dy * s

    if resource == Resource.LUMBER:
        layer.rect(cx - 0.06 * s, cy - 0.5 * s, 0.12 * s, 0.22 * s, color)
        for base, top, half in ((-0.32, 0.1, 0.36), (-0.1, 0.32, 0.3), (0.12, 0.5, 0.22)):
            layer.triangle(p(-half, base), p(half, base), p(0, top), color)
    elif resource == Resource.BRICK:
        rows = (
            ((-0.44, -0.02), (0.02, 0.44)),
            ((-0.44, -0.24), (-0.2, 0.2), (0.24, 0.44)),
            ((-0.44, -0.02), (0.02, 0.44)),
        )
        for i, row in enumerate(rows):
            for a, b in row:
                layer.rect(cx + a * s, cy + (-0.42 + 0.3 * i) * s, (b - a) * s, 0.24 * s, color)
    elif resource == Resource.WOOL:
        fleece = ((-0.2, 0.02, 0.15), (0.0, 0.1, 0.16), (0.16, 0.03, 0.15))
        for dx, dy, r in (*fleece, (-0.08, -0.08, 0.14), (0.08, -0.08, 0.14)):
            layer.circle(*p(dx, dy), r * s, color)
        layer.circle(*p(0.33, 0.12), 0.11 * s, color)
        for dx in (-0.16, 0.08):
            layer.rect(cx + dx * s, cy - 0.42 * s, 0.07 * s, 0.24 * s, color)
    elif resource == Resource.GRAIN:
        layer.line(*p(0, -0.5), *p(0, 0.3), 0.06 * s, color)
        for dy in (-0.22, -0.02, 0.18):
            for d in (-1, 1):
                kernel = (p(0, dy), p(0.12 * d, dy + 0.02), p(0.2 * d, dy + 0.16))
                layer.polygon((*kernel, p(0.06 * d, dy + 0.12)), color)
        layer.polygon((p(0, 0.28), p(0.07, 0.38), p(0, 0.52), p(-0.07, 0.38)), color)
    elif resource == Resource.ORE:
        layer.triangle(p(-0.48, -0.38), p(0.22, -0.38), p(-0.13, 0.4), color)
        layer.triangle(p(-0.02, -0.38), p(0.48, -0.38), p(0.23, 0.12), color)


def draw_dev_glyph(
    layer: Layer, card: int, cx: float, cy: float, size: float, ink: Color, face: Color
) -> None:
    """Symbol of a development card, about `size` pixels tall: a shield (Knight), two roads,
    two cards and a plus (Year of Plenty), a crown (Monopoly), a star (Victory Point). `face`
    is the card colour, used to cut details out of the symbol."""
    s = size

    def p(dx: float, dy: float) -> Point:
        return cx + dx * s, cy + dy * s

    if card == DevCard.KNIGHT:
        shield = [p(-0.34, 0.38), p(0.34, 0.38), p(0.34, 0.0), p(0.0, -0.46), p(-0.34, 0.0)]
        layer.polygon(shield, ink)
        layer.polygon(scaled(shield, p(0, 0.03), 0.7), face)
        layer.rect(cx - 0.04 * s, cy - 0.26 * s, 0.08 * s, 0.5 * s, ink)
        layer.rect(cx - 0.17 * s, cy + 0.06 * s, 0.34 * s, 0.08 * s, ink)
    elif card == DevCard.ROAD_BUILDING:
        layer.line(*p(-0.34, -0.36), *p(-0.06, 0.36), 0.15 * s, ink, True)
        layer.line(*p(0.06, -0.36), *p(0.34, 0.36), 0.15 * s, ink, True)
    elif card == DevCard.YEAR_OF_PLENTY:
        layer.rounded_rect(cx - 0.4 * s, cy - 0.32 * s, 0.42 * s, 0.56 * s, 0.07 * s, ink)
        fx, fy, fw, fh, gap = cx - 0.04 * s, cy - 0.22 * s, 0.42 * s, 0.56 * s, 0.06 * s
        layer.rounded_rect(fx - gap, fy - gap, fw + 2 * gap, fh + 2 * gap, 0.1 * s, face)
        layer.rounded_rect(fx, fy, fw, fh, 0.07 * s, ink)
        mx, my = fx + fw / 2, fy + fh / 2
        layer.rect(mx - 0.03 * s, my - 0.13 * s, 0.06 * s, 0.26 * s, face)
        layer.rect(mx - 0.13 * s, my - 0.03 * s, 0.26 * s, 0.06 * s, face)
    elif card == DevCard.MONOPOLY:
        layer.rect(cx - 0.38 * s, cy - 0.34 * s, 0.76 * s, 0.16 * s, ink)
        spikes = ((-0.38, -0.12, -0.3, 0.26), (-0.14, 0.14, 0.0, 0.36), (0.12, 0.38, 0.3, 0.26))
        for left, right, tip, top in spikes:
            layer.triangle(p(left, -0.2), p(right, -0.2), p(tip, top), ink)
            layer.circle(*p(tip, top + 0.04), 0.06 * s, ink)
    elif card == DevCard.VICTORY_POINT:
        star = [
            p(r * cos(radians(90 + 36 * k)), r * sin(radians(90 + 36 * k)))
            for k, r in enumerate((0.48, 0.2) * 5)
        ]
        for a, b in zip(star, (*star[1:], star[0]), strict=True):
            layer.triangle((cx, cy), a, b, ink)


def draw_tile_button(layer: Layer, button: Button, hovered: bool = False) -> None:
    """A large action button: its icon above its label, the hotkey in the corner."""
    r = button.rect
    if not button.enabled:
        fill, ink = theme.BUTTON_DISABLED, theme.BUTTON_TEXT_DISABLED
    else:
        fill, ink = (theme.BUTTON_HOVER if hovered else theme.BUTTON), theme.TEXT
    layer.rounded_rect(r.x, r.y, r.w, r.h, 8, fill)
    if button.accent and button.enabled:
        layer.rounded_frame(r.x, r.y, r.w, r.h, 8, 2, theme.ACCENT)
    fits = len(button.label) * theme.FONT_SMALL * 1.333 * 0.62 <= r.w - 8
    labelled = fits and r.h >= 52
    icon_y = r.cy + (r.h * 0.13 if labelled else 0)
    draw_icon(layer, button.icon, r.cx, icon_y, min(r.w, r.h) * 0.36, ink)
    if labelled:
        layer.text(button.label, r.cx, r.y + 8, theme.FONT_SMALL, ink, True, "center", "bottom")
    if button.hotkey and r.w >= 48:
        dim = theme.TEXT_DIM if button.enabled else theme.BUTTON_TEXT_DISABLED
        x, y = r.right - 6, r.top - 5
        layer.text(button.hotkey, x, y, theme.FONT_SMALL - 1, dim, False, "right", "top")


def draw_tooltip(layer: Layer, text: str, x: float, y: float, bounds: Rect) -> None:
    """A small label centred above the point (x, y), kept inside `bounds`."""
    w, h = len(text) * theme.FONT_SMALL * 1.333 * theme.CHAR_WIDTH + 20, 22
    left = max(bounds.x + 4, min(x - w / 2, bounds.right - w - 4))
    bottom = y + 6
    layer.rounded_rect(left + 1, bottom - 2, w, h, 6, theme.with_alpha(theme.BLACK, 90))
    layer.rounded_rect(left, bottom, w, h, 6, theme.PANEL_RAISED)
    layer.rounded_frame(left, bottom, w, h, 6, 1, theme.PANEL_EDGE)
    mid = left + w / 2, bottom + h / 2
    layer.text(text, *mid, theme.FONT_SMALL, theme.TEXT, False, "center", "center")
