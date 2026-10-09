"""Side panel (phase and dice, players, bank), hand bar, log panel and status line.

`panel_view` and `hand_view` decide what the viewer may see (in Play mode: the human's own cards
and victory point cards, and only card counts for the other seats); the layout and draw
functions only lay it out.
"""

import textwrap
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from carla.engine.constants import NUM_PLAYERS, NUM_RESOURCES, DevCard
from carla.engine.state import Phase
from carla.gui import theme
from carla.gui.controller import ButtonSpec
from carla.gui.geometry import Point, Rect
from carla.gui.render import Layer
from carla.gui.session import GameSession, LogEntry
from carla.gui.theme import Color
from carla.gui.widgets import (
    CARD_RATIO,
    Button,
    Card,
    card_at,
    draw_card,
    draw_tile_button,
    resource_chip,
    tile_label_fits,
)


@dataclass(frozen=True)
class PlayerRow:
    seat: int
    name: str
    vp: int
    """Points the viewer can count (hidden victory point cards only for seats it may see)."""
    vp_cards: int
    """Victory point cards included in `vp`."""
    cards: int
    dev_cards: int
    knights: int
    road: int
    longest_road: bool
    largest_army: bool
    current: bool
    deciding: bool


@dataclass(frozen=True)
class HandView:
    seat: int
    title: str
    resources: tuple[int, ...]
    dev: tuple[tuple[int, int, int], ...]
    """(card, held, bought this turn) for each development card held."""


@dataclass(frozen=True)
class PanelView:
    title: str
    phase: str
    dice: tuple[int, int] | None
    rolled: bool
    """The dice shown were rolled this turn (older rolls are drawn faded)."""
    players: tuple[PlayerRow, ...]
    bank: tuple[int, ...]
    dev_deck: int


def panel_view(session: GameSession) -> PanelView:
    game, viewer = session.game, session.viewer
    s = game.state
    names = session.seat_names()
    over = game.is_over
    players = []
    for p in range(NUM_PLAYERS):
        sees = viewer is None or viewer == p
        players.append(
            PlayerRow(
                seat=p,
                name=names[p],
                vp=game.victory_points(p, include_hidden=sees),
                vp_cards=s.dev_hands[p][DevCard.VICTORY_POINT] if sees else 0,
                cards=s.hand_size(p),
                dev_cards=sum(s.dev_hands[p]),
                knights=s.dev_played[p][DevCard.KNIGHT],
                road=s.road_lengths[p],
                longest_road=s.longest_road_owner == p,
                largest_army=s.largest_army_owner == p,
                current=s.current_player == p and not over,
                deciding=s.decision_player == p,
            )
        )
    if over:
        title = "Game over"
        phase = "no winner (turn limit)" if game.winner is None else f"P{game.winner} wins"
    else:
        setup = s.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)
        title = "Setup" if setup else f"Turn {s.turn + 1}"
        phase = f"P{s.current_player} · {theme.PHASE_LABELS[s.phase]}"
        if s.decision_player != s.current_player:
            phase += f" · P{s.decision_player} to act"
    return PanelView(
        title=title,
        phase=phase,
        dice=session.last_dice(),
        rolled=s.has_rolled,
        players=tuple(players),
        bank=tuple(s.bank),
        dev_deck=sum(s.dev_deck),
    )


def hand_view(session: GameSession, offered: Sequence[int] = ()) -> HandView:
    """The hand the bottom bar shows: the human's in Play mode (less the cards `offered` in the
    trade panel), else the seat to act's."""
    s, viewer = session.game.state, session.viewer
    if viewer is not None:
        focus, title = viewer, "Your hand"
    else:
        focus = s.decision_player if s.decision_player >= 0 else s.current_player
        title = f"P{focus}'s hand"
    held = s.dev_hands[focus]
    new = s.dev_bought_this_turn if focus == s.current_player else [0] * len(held)
    dev = tuple((d, held[d], new[d]) for d in DevCard if held[d])
    offered = offered or (0,) * NUM_RESOURCES
    resources = tuple(n - k for n, k in zip(s.hands[focus], offered, strict=True))
    return HandView(focus, title, resources, dev)


# ------------------------------------------------------------------ text helpers


def _chars(width: float, size: float) -> int:
    """Characters of the default font at `size` points that fit `width` pixels (estimate)."""
    return int(width / (size * 1.333 * theme.CHAR_WIDTH))


def wrap(text: str, width: float, size: float) -> list[str]:
    """Split `text` into lines that fit `width` pixels."""
    return textwrap.wrap(text, max(8, _chars(width, size))) or [""]


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def ellipsize(text: str, width: float, size: float) -> str:
    chars = max(4, _chars(width, size))
    return text if len(text) <= chars else text[: chars - 1].rstrip() + "…"


# ------------------------------------------------------------------ side panel


def draw_side_panel(layer: Layer, rect: Rect, view: PanelView) -> None:
    layer.rect(rect.x, rect.y, rect.w, rect.h, theme.PANEL)
    layer.rect(rect.x, rect.y, 1.5, rect.h, theme.PANEL_EDGE)
    inner = rect.inset(14, 10)
    header, rest = inner.take_top(52)
    _draw_header(layer, header, view)
    bank_h = 58
    rows = Rect(rest.x, rest.y + bank_h, rest.w, rest.h - bank_h)
    row_h = min(64.0, (rows.h - 6) / NUM_PLAYERS)
    for i, player in enumerate(view.players):
        _draw_player(layer, Rect(rows.x, rows.top - (i + 1) * row_h, rows.w, row_h - 5), player)
    _draw_bank(layer, Rect(rest.x, rest.y, rest.w, bank_h), view)


def _draw_header(layer: Layer, r: Rect, view: PanelView) -> None:
    layer.text(view.title, r.x, r.top - 2, theme.FONT_TITLE, theme.TEXT, True, "left", "top")
    phase = ellipsize(view.phase, r.w - 120, theme.FONT)
    layer.text(phase, r.x, r.y + 4, theme.FONT, theme.TEXT_MUTED, False, "left", "bottom")
    if view.dice is None:
        return
    size = 30
    for i, value in enumerate(view.dice):
        draw_die(
            layer, r.right - (2 - i) * (size + 7), r.cy - size / 2 + 2, size, value, not view.rolled
        )
    ink = theme.TEXT if view.rolled else theme.TEXT_DIM
    x = r.right - 2 * (size + 7) - 6
    layer.text(f"{sum(view.dice)} =", x, r.cy + 2, theme.FONT_LARGE, ink, True, "right", "center")


_PIP_SPOTS = {
    1: ((0, 0),),
    2: ((-1, 1), (1, -1)),
    3: ((-1, 1), (0, 0), (1, -1)),
    4: ((-1, 1), (1, 1), (-1, -1), (1, -1)),
    5: ((-1, 1), (1, 1), (0, 0), (-1, -1), (1, -1)),
    6: ((-1, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (1, -1)),
}


def draw_die(
    layer: Layer, x: float, y: float, size: float, value: int, faded: bool = False
) -> None:
    """A die face; `faded` for a roll from an earlier turn."""
    face = theme.mix(theme.PANEL_RAISED, theme.DICE, 0.3) if faded else theme.DICE
    layer.rounded_rect(x, y, size, size, size * 0.2, face)
    cx, cy, step = x + size / 2, y + size / 2, size * 0.26
    pip = theme.TEXT_MUTED if faded else theme.DICE_PIP
    for dx, dy in _PIP_SPOTS.get(value, ()):
        layer.circle(cx + dx * step, cy + dy * step, size * 0.085, pip, 10)


def _draw_player(layer: Layer, r: Rect, p: PlayerRow) -> None:
    color = theme.PLAYER[p.seat]
    fill = theme.PANEL_RAISED if p.current else theme.mix(theme.PANEL, theme.PANEL_RAISED, 0.35)
    layer.rounded_rect(r.x, r.y, r.w, r.h, 6, fill)
    if p.current:
        layer.rounded_frame(r.x, r.y, r.w, r.h, 6, 1.5, theme.with_alpha(color, 220))
    layer.rounded_rect(r.x, r.y, 6, r.h, 3, color)
    name = f"P{p.seat}  {p.name}"
    layer.text(name, r.x + 16, r.top - 7, theme.FONT, theme.TEXT, True, "left", "top")
    stats = " · ".join(
        (
            plural(p.cards, "card"),
            f"{p.dev_cards} dev",
            plural(p.knights, "knight"),
            f"road {p.road}",
        )
    )
    stats = ellipsize(stats, r.w - 80, theme.FONT_SMALL)
    layer.text(
        stats, r.x + 16, r.y + 6, theme.FONT_SMALL, theme.TEXT_MUTED, False, "left", "bottom"
    )
    layer.text(
        str(p.vp), r.right - 32, r.cy + 1, theme.FONT_TITLE, theme.TEXT, True, "right", "center"
    )
    layer.text(
        "VP", r.right - 9, r.cy, theme.FONT_SMALL, theme.TEXT_MUTED, False, "right", "center"
    )
    badges = []  # (label, highlighted)
    if p.longest_road:
        badges.append(("Longest Road", True))
    if p.largest_army:
        badges.append(("Largest Army", True))
    if p.vp_cards:
        badges.append((f"{p.vp_cards} VP card{'s' if p.vp_cards > 1 else ''}", False))
    if p.deciding and not p.current:
        badges.append(("to act", False))
    x = r.x + 16 + len(name) * theme.FONT * 1.333 * theme.CHAR_WIDTH + 12
    for label, highlighted in badges:
        width = len(label) * (theme.FONT_SMALL - 1) * 1.333 * 0.58 + 12
        if x + width > r.right - 56:
            break
        layer.rounded_rect(
            x, r.top - 23, width, 16, 8, theme.ACCENT if highlighted else theme.BUTTON
        )
        ink = theme.INK if highlighted else theme.TEXT
        layer.text(
            label, x + width / 2, r.top - 15, theme.FONT_SMALL - 1, ink, True, "center", "center"
        )
        x += width + 5


def _draw_bank(layer: Layer, r: Rect, view: PanelView) -> None:
    layer.rect(r.x, r.top - 4, r.w, 1, theme.PANEL_EDGE)
    layer.text("Bank", r.x, r.top - 12, theme.FONT_SMALL, theme.TEXT_MUTED, True, "left", "top")
    deck = f"development deck {view.dev_deck}"
    layer.text(deck, r.right, r.top - 12, theme.FONT_SMALL, theme.TEXT_MUTED, False, "right", "top")
    for res, cell in enumerate(Rect(r.x, r.y + 2, r.w, 24).columns(NUM_RESOURCES, 8)):
        resource_chip(layer, res, cell.x + 11, cell.cy, 22, theme.TERRAIN[res])
        layer.text(
            str(view.bank[res]),
            cell.x + 28,
            cell.cy,
            theme.FONT,
            theme.TEXT,
            True,
            "left",
            "center",
        )


# ------------------------------------------------------------------ hand bar

_ACTIONS = (
    ("buy_dev", "Buy dev", "buy_card"),
    ("roll", "Roll", "dice"),
    ("end_turn", "End turn", "hourglass"),
)
"""Hand bar buttons, left to right: controller button id, label, icon."""
_PAD, _LABEL_H, _GAP, _SEP = 10.0, 18.0, 6.0, 22.0
_CARD_W, _MIN_CARD_W = 46.0, 28.0
_TILE_W, _MIN_TILE_W, _TILE_GAP = 76.0, 52.0, 6.0


@dataclass(frozen=True)
class HandBar:
    """Layout of the hand bar (pure, for drawing and hit-testing): the resource cards, a
    separator, the development cards, then the action buttons on the right (Play mode)."""

    rect: Rect
    hand: HandView
    cards: tuple[Card, ...]
    separator: float
    """x of the line between the resource and the development cards."""
    buttons: tuple[Button, ...]

    def card_at(self, x: float, y: float) -> Card | None:
        return card_at(self.cards, x, y)

    @property
    def labelled(self) -> bool:
        """Every action tile shows its label (all or none, so the row looks even)."""
        return all(tile_label_fits(b) for b in self.buttons)


def _dev_groups(hand: HandView) -> list[tuple[int, int, bool]]:
    """(card, count, bought this turn) of each group of development cards, in card order."""
    out = []
    for card, held, new in hand.dev:
        if held > new:
            out.append((card, held - new, False))
        if new:
            out.append((card, new, True))
    return out


def layout_hand_bar(
    rect: Rect,
    hand: HandView,
    actions: Sequence[ButtonSpec] | None = None,
    playable: frozenset[int] = frozenset(),
    offerable: frozenset[int] = frozenset(),
) -> HandBar:
    """Lay out `hand` in `rect`; `actions` (the controller's buttons) adds the action tiles,
    `playable` marks the development cards a click plays and `offerable` the resources a click
    puts in a trade. Cards shrink, then development cards overlap, when the bar is too
    narrow."""
    inner = rect.inset(_PAD)
    specs = {spec.id: spec for spec in actions or ()}
    shown = [(specs[name], label, icon) for name, label, icon in _ACTIONS if name in specs]
    groups = _dev_groups(hand)
    tiles = len(shown)
    tile_w = _TILE_W
    if tiles:
        natural = (NUM_RESOURCES + len(groups)) * (_CARD_W + _GAP) + _SEP
        spare = inner.w - natural - 2 * _PAD - tiles * _TILE_W - (tiles - 1) * _TILE_GAP
        tile_w = max(_MIN_TILE_W, _TILE_W + min(0.0, spare) / tiles)
    span = tiles * tile_w + max(0, tiles - 1) * _TILE_GAP
    buttons = []
    for i, (spec, label, icon) in enumerate(shown):
        x = inner.right - span + i * (tile_w + _TILE_GAP)
        accent = spec.id == "roll" and spec.enabled
        tile = Rect(x, inner.y, tile_w, inner.h)
        buttons.append(Button(spec.id, label, tile, spec.enabled, spec.hotkey, icon, accent))
    right = inner.right - span - 2 * _PAD if tiles else inner.right
    area = Rect(inner.x, inner.y, max(0.0, right - inner.x), inner.h - _LABEL_H)
    slots = NUM_RESOURCES + len(groups)
    w = min(_CARD_W, area.h * CARD_RATIO)
    if slots * w + (slots - 1) * _GAP + _SEP > area.w:
        w = max(_MIN_CARD_W, (area.w - _SEP - (slots - 1) * _GAP) / slots)
    h = w / CARD_RATIO
    y = area.y + (area.h - h) / 2
    cards = [
        Card(
            "resource",
            r,
            hand.resources[r],
            Rect(area.x + r * (w + _GAP), y, w, h),
            playable=r in offerable and hand.resources[r] > 0,
        )
        for r in range(NUM_RESOURCES)
    ]
    separator = area.x + NUM_RESOURCES * (w + _GAP) - _GAP + _SEP / 2
    first = separator + _SEP / 2
    step = w + _GAP
    if len(groups) > 1 and first + (len(groups) - 1) * step + w > area.right:
        step = max(8.0, (area.right - first - w) / (len(groups) - 1))  # fan them out
    for i, (card, count, new) in enumerate(groups):
        play = card in playable and not new
        cards.append(Card("dev", card, count, Rect(first + i * step, y, w, h), new, play))
    return HandBar(rect, hand, tuple(cards), separator, tuple(buttons))


def draw_hand_bar(layer: Layer, bar: HandBar, mouse: Point = (-1.0, -1.0)) -> None:
    r, hand = bar.rect, bar.hand
    layer.rect(r.x, r.y, r.w, r.h, theme.PANEL)
    layer.rect(r.x, r.top - 1.5, r.w, 1.5, theme.PANEL_EDGE)
    x, y = r.x + _PAD, r.top - _PAD
    layer.circle(x + 4, y - 6, 4, theme.PLAYER[hand.seat])
    title = f"{hand.title} · {plural(sum(hand.resources), 'card')}"
    layer.text(title, x + 14, y, theme.FONT_SMALL, theme.TEXT_MUTED, True, "left", "top")
    hovered = bar.card_at(*mouse)
    for card in bar.cards:
        if card is not hovered:
            draw_card(layer, card)
    if hovered is not None:
        draw_card(layer, hovered, hovered=True)  # on top of its neighbours
    first = bar.cards[0].rect
    layer.rect(bar.separator - 0.75, first.y + 4, 1.5, first.h - 8, theme.PANEL_EDGE)
    if not any(card.kind == "dev" for card in bar.cards):
        room = min((b.rect.x for b in bar.buttons), default=r.right - _PAD) - bar.separator - 24
        if room > 60:
            note = ellipsize("No development cards", room, theme.FONT_SMALL)
            muted = theme.TEXT_DIM
            layer.text(
                note, bar.separator + 14, first.cy, theme.FONT_SMALL, muted, False, "left", "center"
            )
    labelled = bar.labelled
    for b in bar.buttons:
        draw_tile_button(layer, b, b.enabled and b.rect.contains(*mouse), labelled)


def card_tooltip(card: Card) -> str:
    """What a hovered card says about itself."""
    if card.kind == "resource":
        return f"{card.label} ×{card.count}" + (": click to trade" if card.playable else "")
    if card.dimmed:
        return f"{card.label}: bought this turn"
    if card.playable:
        return f"{card.label}: click to play"
    if card.index == DevCard.VICTORY_POINT:
        return f"{card.label}: counts automatically"
    return card.label


# ------------------------------------------------------------------ log


class LogView:
    """Scrollable log of the position shown; lines are wrapped once per entry and only the
    visible window is laid out each frame."""

    LINE_H = 17
    SIZE = theme.FONT_SMALL + 0.5

    def __init__(self) -> None:
        self.scroll = 0
        """Lines scrolled up from the newest one (0 follows new lines)."""
        self._wrapped: list[list[str]] = []
        self._prefix = [0]
        """_prefix[i]: number of lines of the first i entries."""
        self._key: tuple = ()

    def window(
        self, session: GameSession, width: float, rows: int
    ) -> tuple[list[tuple[LogEntry, str]], int]:
        """The `rows` lines to show after scrolling, and the total number of lines."""
        key = (id(session), session.viewer, round(width))
        if key != self._key:
            self._key, self._wrapped, self._prefix = key, [], [0]
        n = session.visible_count()
        while len(self._wrapped) < n:
            entry = session.log[len(self._wrapped)]
            lines = wrap(session.entry_text(entry), width, self.SIZE)
            self._wrapped.append(lines)
            self._prefix.append(self._prefix[-1] + len(lines))
        total = self._prefix[n]
        self.scroll = max(0, min(self.scroll, total - rows))
        end = total - self.scroll
        start = max(0, end - rows)
        out = []
        i = bisect_right(self._prefix, start) - 1
        while i < n and self._prefix[i] < end:
            for k, line in enumerate(self._wrapped[i]):
                if start <= self._prefix[i] + k < end:
                    out.append((session.log[i], line))
            i += 1
        return out, total

    def draw(self, layer: Layer, rect: Rect, session: GameSession) -> None:
        layer.rect(rect.x, rect.y, rect.w, rect.h, theme.mix(theme.PANEL, theme.BACKGROUND, 0.5))
        layer.rect(rect.x, rect.top - 1, rect.w, 1, theme.PANEL_EDGE)
        layer.rect(rect.x, rect.y, 1.5, rect.h, theme.PANEL_EDGE)
        layer.text(
            "Log",
            rect.x + 14,
            rect.top - 9,
            theme.FONT_SMALL,
            theme.TEXT_MUTED,
            True,
            "left",
            "top",
        )
        if self.scroll:
            hint = f"{self.scroll} newer lines below"
            layer.text(
                hint,
                rect.right - 16,
                rect.top - 9,
                theme.FONT_SMALL - 1,
                theme.ACCENT,
                False,
                "right",
                "top",
            )
        body = Rect(rect.x + 14, rect.y + 8, rect.w - 34, rect.h - 34)
        rows = max(1, int(body.h // self.LINE_H))
        lines, total = self.window(session, body.w - 14, rows)
        y, previous = body.top, None
        for entry, line in lines:
            if entry.event is None:
                layer.text(line, body.x, y, self.SIZE, theme.ACCENT, True, "left", "top")
            else:
                if entry is not previous and 0 <= entry.player < NUM_PLAYERS:
                    layer.rounded_rect(body.x, y - 11, 7, 7, 2, theme.PLAYER[entry.player])
                layer.text(line, body.x + 14, y, self.SIZE, theme.TEXT, False, "left", "top")
            previous = entry
            y -= self.LINE_H
        if total > rows:
            track = Rect(rect.right - 11, body.y, 4, body.h)
            layer.rounded_rect(track.x, track.y, track.w, track.h, 2, theme.PANEL_RAISED)
            size = max(16.0, track.h * rows / total)
            offset = (track.h - size) * (self.scroll / max(1, total - rows))
            layer.rounded_rect(track.x, track.y + offset, track.w, size, 2, theme.TEXT_DIM)


# ------------------------------------------------------------------ status line


def draw_status(layer: Layer, rect: Rect, badge: str, text: str, color: Color, hint: str) -> None:
    """Mode badge, the main message (prompt, bot activity or notification) and a key hint."""
    layer.rect(rect.x, rect.y, rect.w, rect.h, theme.with_alpha(theme.BACKGROUND, 150))
    x = rect.x + 12
    if badge:
        width = len(badge) * theme.FONT_SMALL * 1.333 * 0.64 + 18
        layer.rounded_rect(x, rect.cy - 10, width, 20, 10, theme.ACCENT)
        layer.text(
            badge, x + width / 2, rect.cy, theme.FONT_SMALL, theme.INK, True, "center", "center"
        )
        x += width + 12
    hint_w = len(hint) * theme.FONT_SMALL * 1.333 * theme.CHAR_WIDTH + 16
    text = ellipsize(text, rect.right - x - hint_w - 12, theme.FONT)
    layer.text(text, x, rect.cy, theme.FONT, color, False, "left", "center")
    layer.text(
        hint, rect.right - 12, rect.cy, theme.FONT_SMALL, theme.TEXT_MUTED, False, "right", "center"
    )
