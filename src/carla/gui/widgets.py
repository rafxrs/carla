"""Buttons, the seek slider and picker dialogs: layout and hit-testing (pure) plus drawing into a
`Layer`. Icons are drawn from triangles and bars, never from image files."""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from carla.gui import theme
from carla.gui.controller import Option, Picker
from carla.gui.geometry import Rect
from carla.gui.render import Layer
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
    """Glyph drawn instead of the label: play, pause, back, forward, start, end, minus, plus."""
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
    """Playback and stepper glyphs, `size` pixels tall."""
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
