"""The trade panel docked above the hand bar (see `controller.TradePanel`): layout and
hit-testing (pure), then drawing with the same cards as the hand.

A composer has a "You give" row (cards clicked in the hand; a click takes one back), a "You get"
row (a click takes one off) and the five resources to ask for. A negotiation has one row per
seat, its answer and its terms as cards under "give" and "get" columns.
"""

from collections.abc import Hashable
from dataclasses import dataclass

from carla.engine.constants import NUM_RESOURCES
from carla.gui import theme
from carla.gui.controller import TradeButton, TradePanel
from carla.gui.geometry import Point, Rect
from carla.gui.render import Layer
from carla.gui.widgets import CARD_RATIO, Button, Card, draw_button, draw_card, draw_icon

_WIDTH, _PAD, _HEAD, _GAP = 560.0, 14.0, 30.0, 8.0
_LABEL_W = 78.0
"""Room for "You give" / "You get" left of a composer's cards."""
_CARD_W, _PICK_W, _MINI_W, _CARD_GAP = 36.0, 30.0, 26.0, 5.0
_ROW_H, _MINI_ROW_H, _COLUMNS_H = 58.0, 44.0, 18.0
_HINT_H, _BUTTON_H, _BUTTON_W = 22.0, 34.0, 150.0
_NAME_W, _ROW_BUTTON_W = 120.0, 78.0
_STATUS_INK = {
    "offers": theme.TEXT_MUTED,
    "accepts": theme.TEXT,
    "counters": theme.TEXT,
    "deciding…": theme.ACCENT,
}


@dataclass(frozen=True)
class TradeLayout:
    panel: Rect
    cards: tuple[tuple[Card, Hashable], ...]
    """Each card drawn and the key a click on it sends (None: only shown)."""
    buttons: tuple[tuple[TradeButton, Rect], ...]
    """Row buttons, then the footer buttons."""
    rows: tuple[Rect, ...]
    """The give and get rows of a composer, or one per negotiation row."""
    columns: tuple[float, float]
    """x of the give and get columns of a negotiation."""
    picks: Rect | None
    """The five resources a composer may ask for."""
    close: Rect | None

    def card_at(self, x: float, y: float) -> tuple[Card, Hashable] | None:
        return next(((c, k) for c, k in reversed(self.cards) if c.rect.contains(x, y)), None)

    def key_at(self, x: float, y: float) -> Hashable | None:
        """What a click at (x, y) sends: an enabled button, a clickable card or the close box."""
        for button, rect in self.buttons:
            if rect.contains(x, y):
                return button.key if button.enabled else None
        if self.close is not None and self.close.contains(x, y):
            return "close"
        hit = self.card_at(x, y)
        return hit[1] if hit is not None and hit[0].playable else None


def layout_trade(panel: TradePanel, area: Rect) -> TradeLayout:
    """Dock the panel at the bottom of `area` (the board viewport), just above the hand."""
    width = min(_WIDTH, area.w - 16)
    if panel.composer:
        body = 2 * _ROW_H + _GAP
    else:
        body = _COLUMNS_H + len(panel.rows) * (_MINI_ROW_H + 4) if panel.rows else 0.0
    footer = _BUTTON_H + _GAP if panel.buttons else 0.0
    height = 2 * _PAD + _HEAD + body + (_HINT_H if panel.hint else 4.0) + footer
    rect = Rect(area.cx - width / 2, area.y + 8, width, min(height, area.h - 16))
    inner = rect.inset(_PAD)
    head, rest = inner.take_top(_HEAD)
    n = len(panel.buttons)
    w = min(_BUTTON_W, (inner.w - (n - 1) * _GAP) / max(1, n))
    x = inner.right - n * w - (n - 1) * _GAP
    footer_buttons = [
        (b, Rect(x + i * (w + _GAP), inner.y, w, _BUTTON_H)) for i, b in enumerate(panel.buttons)
    ]
    if panel.composer:
        cards, rows, picks = _composer(panel, rest)
        close = Rect(head.right - 24, head.cy - 12, 24, 24)
        return TradeLayout(rect, cards, tuple(footer_buttons), rows, (0.0, 0.0), picks, close)
    cards, rows, row_buttons, columns = _negotiation(panel, rest)
    buttons = (*row_buttons, *footer_buttons)
    return TradeLayout(rect, cards, buttons, rows, columns, None, None)


def _card(x: float, cy: float, w: float) -> Rect:
    h = w / CARD_RATIO
    return Rect(x, cy - h / 2, w, h)


def _composer(
    panel: TradePanel, rest: Rect
) -> tuple[tuple[tuple[Card, Hashable], ...], tuple[Rect, ...], Rect]:
    give_row, rest = rest.take_top(_ROW_H)
    get_row = Rect(rest.x, rest.top - _GAP - _ROW_H, rest.w, _ROW_H)
    cards: list[tuple[Card, Hashable]] = []
    for row, counts, side in ((give_row, panel.give, "give"), (get_row, panel.get, "get")):
        x = row.x + _LABEL_W
        for r in range(NUM_RESOURCES):
            if counts[r]:
                rect = _card(x, row.cy, _CARD_W)
                cards.append((Card("resource", r, counts[r], rect, playable=True), (side, r)))
                x += _CARD_W + _CARD_GAP
    span = NUM_RESOURCES * (_PICK_W + _CARD_GAP) - _CARD_GAP
    picks = Rect(get_row.right - span, get_row.y, span, get_row.h)
    for r in range(NUM_RESOURCES):
        ok = panel.wants[r]
        rect = _card(picks.x + r * (_PICK_W + _CARD_GAP), get_row.cy, _PICK_W)
        cards.append((Card("resource", r, 1, rect, dimmed=not ok, playable=ok), ("want", r)))
    return tuple(cards), (give_row, get_row), picks


def _negotiation(panel: TradePanel, rest: Rect) -> tuple[
    tuple[tuple[Card, Hashable], ...],
    tuple[Rect, ...],
    tuple[tuple[TradeButton, Rect], ...],
    tuple[float, float],
]:
    side_w = (rest.w - _NAME_W - _ROW_BUTTON_W - 3 * _GAP) / 2
    columns = (rest.x + _NAME_W + _GAP, rest.x + _NAME_W + side_w + 2 * _GAP)
    cards: list[tuple[Card, Hashable]] = []
    rows, buttons = [], []
    y = rest.top - _COLUMNS_H
    for row in panel.rows:
        r = Rect(rest.x, y - _MINI_ROW_H, rest.w, _MINI_ROW_H)
        rows.append(r)
        for x, counts in zip(columns, (row.give, row.get), strict=True):
            for res in range(NUM_RESOURCES):
                if counts[res]:
                    cards.append(
                        (Card("resource", res, counts[res], _card(x, r.cy, _MINI_W)), None)
                    )
                    x += _MINI_W + _CARD_GAP
        if row.button is not None:
            rect = Rect(r.right - _ROW_BUTTON_W - 6, r.cy - 14, _ROW_BUTTON_W, 28)
            buttons.append((row.button, rect))
        y -= _MINI_ROW_H + 4
    return tuple(cards), tuple(rows), tuple(buttons), columns


def trade_tooltip(card: Card, key: Hashable) -> str:
    """What a hovered card of the panel says about itself."""
    name = f"{card.label} ×{card.count}"
    if not isinstance(key, tuple):
        return name
    if not card.playable:
        return ""
    side = key[0]
    if side == "want":
        return f"Ask for one {card.label.lower()}"
    return f"{name}: click to take one {'back' if side == 'give' else 'off'}"


def draw_trade(
    layer: Layer, panel: TradePanel, layout: TradeLayout, mouse: Point = (-1.0, -1.0)
) -> None:
    p = layout.panel
    layer.rounded_rect(p.x + 3, p.y - 4, p.w, p.h, 10, theme.with_alpha(theme.BLACK, 90))
    layer.rounded_rect(p.x, p.y, p.w, p.h, 10, theme.PANEL)
    layer.rounded_frame(p.x, p.y, p.w, p.h, 10, 1.5, theme.PANEL_EDGE)
    head_y = p.top - _PAD - _HEAD / 2
    layer.text(
        panel.title, p.x + _PAD, head_y, theme.FONT_LARGE, theme.TEXT, True, "left", "center"
    )
    hovered = layout.key_at(*mouse)
    if layout.close is not None:
        c = layout.close
        if hovered == "close":
            layer.rounded_rect(c.x, c.y, c.w, c.h, 6, theme.BUTTON_HOVER)
        ink = theme.TEXT if hovered == "close" else theme.TEXT_MUTED
        draw_icon(layer, "close", c.cx, c.cy, 11, ink)
        dim = theme.TEXT_DIM
        layer.text("Esc", c.x - 4, c.cy, theme.FONT_SMALL - 1, dim, False, "right", "center")
    if panel.composer:
        _draw_composer(layer, panel, layout)
    else:
        _draw_negotiation(layer, panel, layout)
    if panel.hint:
        y = p.y + _PAD + (_BUTTON_H + _GAP if panel.buttons else 0.0) + _HINT_H / 2
        muted = theme.TEXT_MUTED
        layer.text(panel.hint, p.x + _PAD, y, theme.FONT_SMALL, muted, False, "left", "center")
    lifted = layout.card_at(*mouse)
    for card, key in layout.cards:
        if (card, key) != lifted:
            draw_card(layer, card, badge=_badge(key))
    if lifted is not None:  # on top of its neighbours
        card, key = lifted
        draw_card(layer, card, card.playable or key is None, _badge(key))
    for number, (b, rect) in enumerate(layout.buttons, start=1):
        button = Button(str(b.key), b.label, rect, b.enabled, str(number), accent=b.accent)
        draw_button(layer, button, b.enabled and hovered == b.key)


def _badge(key: Hashable) -> str | None:
    """The resources to ask for show a plus instead of a count."""
    return "+" if isinstance(key, tuple) and key[0] == "want" else None


def _draw_composer(layer: Layer, panel: TradePanel, layout: TradeLayout) -> None:
    dim, muted = theme.TEXT_DIM, theme.TEXT_MUTED
    notes = ("Click your cards below", "Pick on the right")
    for row, label, counts, note in zip(
        layout.rows, ("You give", "You get"), (panel.give, panel.get), notes, strict=True
    ):
        layer.text(label, row.x, row.cy, theme.FONT, muted, True, "left", "center")
        if not any(counts):
            x = row.x + _LABEL_W
            layer.text(note, x, row.cy, theme.FONT_SMALL, dim, False, "left", "center")
    picks, get_row = layout.picks, layout.rows[1]
    if picks is None:
        return
    taken = sum(1 for r in range(NUM_RESOURCES) if panel.get[r])
    if get_row.x + _LABEL_W + taken * (_CARD_W + _CARD_GAP) < picks.x - 64:
        layer.text(
            "Ask for", picks.x - 8, get_row.cy, theme.FONT_SMALL, dim, False, "right", "center"
        )


def _draw_negotiation(layer: Layer, panel: TradePanel, layout: TradeLayout) -> None:
    if not layout.rows:
        return
    muted = theme.TEXT_MUTED
    header_y = layout.rows[0].top + _COLUMNS_H / 2
    for x, label in zip(layout.columns, panel.columns, strict=True):
        layer.text(label, x, header_y, theme.FONT_SMALL, muted, True, "left", "center")
    for row, r in zip(panel.rows, layout.rows, strict=True):
        layer.rounded_rect(r.x, r.y, r.w, r.h, 6, theme.mix(theme.PANEL, theme.PANEL_RAISED, 0.5))
        layer.rounded_rect(r.x, r.y, 5, r.h, 2.5, theme.PLAYER[row.player])
        layer.text(row.label, r.x + 14, r.cy + 8, theme.FONT, theme.TEXT, True, "left", "center")
        ink = _STATUS_INK.get(row.status, theme.TEXT_DIM)
        layer.text(row.status, r.x + 14, r.cy - 9, theme.FONT_SMALL, ink, False, "left", "center")
