"""Turns the human seat's clicks and button presses into engine `Action`s (pure: no Arcade).

The controller never decides what is legal: highlighted targets, enabled buttons and picker
options all come from the legal actions passed to `reset`, and every action it returns is one of
them. The one exception is a domestic trade offer, which the engine validates instead of
enumerating (and does not implement yet).
"""

from collections.abc import Hashable, Sequence
from typing import NamedTuple

from carla.engine import actions as A
from carla.engine.actions import Action, ActionType, TradeOffer
from carla.engine.constants import NUM_PLAYERS, NUM_RESOURCES, DevCard
from carla.engine.rules import trade_ratios
from carla.engine.state import GameState, Phase
from carla.engine.text import resources_text
from carla.gui.theme import DEV_LABELS, RESOURCE_LABELS

_T = ActionType
_RESOURCES = range(NUM_RESOURCES)


class Option(NamedTuple):
    """One choice of a picker; `resource` and `player` tell the view how to decorate it."""

    key: Hashable
    label: str
    enabled: bool = True
    detail: str = ""
    resource: int = -1
    player: int = -1


class Picker(NamedTuple):
    """A choice the human must make (or may cancel) to complete an action."""

    kind: str
    title: str
    options: tuple[Option, ...]
    cancellable: bool = True
    detail: str = ""


class ButtonSpec(NamedTuple):
    id: str
    label: str
    hotkey: str
    enabled: bool


_BUTTONS = (
    ("roll", "Roll", "R", _T.ROLL_DICE),
    ("end_turn", "End turn", "E", _T.END_TURN),
    ("buy_dev", "Buy dev card", "B", _T.BUY_DEV_CARD),
    ("play_dev", "Play dev card", "D", _T.PLAY_DEV_CARD),
    ("bank_trade", "Bank trade", "T", _T.MARITIME_TRADE),
)
_SIMPLE = {"roll": _T.ROLL_DICE, "end_turn": _T.END_TURN, "buy_dev": _T.BUY_DEV_CARD}


def _cards(n: int) -> str:
    return f"{n} card" if n == 1 else f"{n} cards"


class Controller:
    """Input state machine of the human seat, one decision at a time.

    Board clicks (`click_corner/edge/tile`), buttons (`press`) and picker choices (`choose`)
    return the completed `Action`, or None while more input is needed or the input is ignored.
    """

    def __init__(self, seat: int, domestic_trade: bool = False) -> None:
        self.seat = seat
        self.domestic_trade = domestic_trade
        """Show the propose/accept/reject/confirm/cancel trade interface."""
        self.state: GameState | None = None
        self.legal: tuple[Action, ...] = ()
        self.picker: Picker | None = None
        self._legal_set: frozenset[Action] = frozenset()
        self._by_type: dict[ActionType, list[Action]] = {}
        self._corners: dict[int, Action] = {}
        self._edges: dict[int, Action] = {}
        self._tiles: dict[int, list[Action]] = {}
        self._first = -1
        """First resource of a two-step choice (Year of Plenty, bank trade)."""
        self._robber_tile = -1
        self._give = [0] * NUM_RESOURCES
        self._get = [0] * NUM_RESOURCES

    # ------------------------------------------------------------------ position

    def reset(self, state: GameState | None, legal: Sequence[Action] = ()) -> None:
        """Start a decision of the human seat in `state` (no `legal` actions: not its turn)."""
        self.state = state
        self.legal = tuple(legal) if state is not None else ()
        self._legal_set = frozenset(self.legal)
        self._by_type = {}
        for a in self.legal:
            self._by_type.setdefault(a.type, []).append(a)
        builds = (
            *self._by_type.get(_T.BUILD_SETTLEMENT, ()),
            *self._by_type.get(_T.BUILD_CITY, ()),
        )
        self._corners = {a.corner: a for a in builds}
        self._edges = {a.edge: a for a in self._by_type.get(_T.BUILD_ROAD, ())}
        self._tiles = {}
        for a in self._by_type.get(_T.MOVE_ROBBER, ()):
            self._tiles.setdefault(a.tile, []).append(a)
        self._close()
        if _T.DISCARD in self._by_type:
            self.picker = self._discard_picker()
        elif _T.ACCEPT_TRADE in self._by_type or _T.REJECT_TRADE in self._by_type:
            self.picker = self._response_picker()
        elif _T.CONFIRM_TRADE in self._by_type or _T.CANCEL_TRADE in self._by_type:
            self.picker = self._confirm_picker()

    @property
    def active(self) -> bool:
        """The human must act now."""
        return bool(self.legal)

    @property
    def offer(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Domestic trade being composed: (cards given, cards wanted) per resource."""
        return tuple(self._give), tuple(self._get)

    def prompt(self) -> str:
        """What the human should do next, in a few words."""
        if self.picker is not None:
            return self.picker.title
        s = self.state
        if s is None or not self.legal:
            return ""
        phase = s.phase
        if phase == Phase.SETUP_SETTLEMENT:
            second = s.setup_step >= NUM_PLAYERS
            return "Place a settlement" + (" (it collects starting resources)" if second else "")
        if phase == Phase.SETUP_ROAD:
            return "Place a road next to your new settlement"
        if phase == Phase.PRE_ROLL:
            dev = (
                ", or play a development card first (D)"
                if _T.PLAY_DEV_CARD in self._by_type
                else ""
            )
            return "Roll the dice (R)" + dev
        if phase == Phase.MOVE_ROBBER:
            return "Move the robber: click a highlighted tile"
        if phase == Phase.ROAD_BUILDING:
            return f"Road Building: place a free road ({s.free_roads} left)"
        return "Build on a highlighted spot, trade, or end your turn (E)"

    # ------------------------------------------------------------------ board

    def corner_targets(self) -> dict[int, Action]:
        """Clickable corners and the build each stands for (none while a picker is open)."""
        return {} if self.picker else self._corners

    def edge_targets(self) -> dict[int, Action]:
        return {} if self.picker else self._edges

    def tile_targets(self) -> dict[int, list[Action]]:
        """Clickable robber tiles and the moves (one per possible victim) each stands for."""
        return {} if self.picker else self._tiles

    def click_corner(self, corner: int) -> Action | None:
        return None if self.picker else self._corners.get(corner)

    def click_edge(self, edge: int) -> Action | None:
        return None if self.picker else self._edges.get(edge)

    def click_tile(self, tile: int) -> Action | None:
        moves = None if self.picker else self._tiles.get(tile)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        self._robber_tile = tile
        self.picker = self._victim_picker(moves)
        return None

    # ------------------------------------------------------------------ buttons and pickers

    def buttons(self) -> list[ButtonSpec]:
        """Action buttons; disabled when not legal or while a mandatory picker is open."""
        free = self.picker is None or self.picker.cancellable
        specs = [
            ButtonSpec(name, label, key, free and kind in self._by_type)
            for name, label, key, kind in _BUTTONS
        ]
        if self.domestic_trade:
            specs.append(ButtonSpec("trade", "Offer trade", "P", free and self._can_offer()))
        return specs

    def press(self, button: str) -> Action | None:
        spec = next((b for b in self.buttons() if b.id == button), None)
        if spec is None or not spec.enabled:
            return None
        self._close()
        if button == "play_dev":
            self.picker = self._dev_picker()
        elif button == "bank_trade":
            self.picker = self._give_picker()
        elif button == "trade":
            self._give, self._get = [0] * NUM_RESOURCES, [0] * NUM_RESOURCES
            self.picker = self._offer_picker()
        else:
            return self._by_type[_SIMPLE[button]][0]
        return None

    def choose(self, key: Hashable) -> Action | None:
        """Pick the option `key` of the open picker."""
        picker = self.picker
        option = next((o for o in picker.options if o.key == key), None) if picker else None
        if picker is None or option is None or not option.enabled:
            return None
        kind = picker.kind
        if kind == "dev":
            return self._choose_dev(key)
        if kind == "plenty":
            return self._choose_plenty(key)
        if kind == "give":
            self._first = option.resource
            self.picker = self._get_picker()
            return None
        if kind == "offer":
            return self._choose_offer(key)
        action = self._picked_action(kind, key)
        self._close()
        return action

    def choose_index(self, index: int) -> Action | None:
        """Pick the `index`-th option (0-based, as displayed), e.g. from the number keys."""
        if self.picker is None or self.picker.kind == "offer":
            return None
        options = self.picker.options
        return self.choose(options[index].key) if 0 <= index < len(options) else None

    def cancel(self) -> bool:
        """Close a cancellable picker (Esc); mandatory ones (discard, trade answers) stay."""
        if self.picker is None or not self.picker.cancellable:
            return False
        self._close()
        return True

    # ------------------------------------------------------------------ internals

    def _close(self) -> None:
        self.picker = None
        self._first = self._robber_tile = -1

    def _hand(self) -> list[int]:
        assert self.state is not None
        return self.state.hands[self.seat]

    def _plays(self, card: DevCard) -> list[Action]:
        return [a for a in self._by_type.get(_T.PLAY_DEV_CARD, ()) if a.dev == card]

    def _resource_options(
        self, enabled: set[int], details: Sequence[str] = ()
    ) -> tuple[Option, ...]:
        return tuple(
            Option(r, RESOURCE_LABELS[r], r in enabled, details[r] if details else "", resource=r)
            for r in _RESOURCES
        )

    def _can_offer(self) -> bool:
        s = self.state
        return s is not None and s.phase == Phase.MAIN and self.active and any(self._hand())

    def _picked_action(self, kind: str, key: Hashable) -> Action | None:
        if kind == "monopoly":
            return A.MONOPOLY_ON[key] if isinstance(key, int) else None
        if kind == "get":
            return A.MARITIME[self._first][key] if isinstance(key, int) else None
        if kind == "discard":
            return A.DISCARD_OF[key] if isinstance(key, int) else None
        if kind == "victim":
            return next(a for a in self._tiles[self._robber_tile] if a.player == key)
        if kind == "response":
            return A.ACCEPT_TRADE if key == "accept" else A.REJECT_TRADE
        if kind == "confirm":
            return A.CONFIRM_TRADE_WITH[key] if isinstance(key, int) else A.CANCEL_TRADE
        raise ValueError(f"unknown picker {kind!r}")

    def _dev_picker(self) -> Picker:
        assert self.state is not None
        held = self.state.dev_hands[self.seat]
        new = self.state.dev_bought_this_turn
        options = []
        for card in (
            DevCard.KNIGHT,
            DevCard.ROAD_BUILDING,
            DevCard.YEAR_OF_PLENTY,
            DevCard.MONOPOLY,
        ):
            detail = f"{held[card]} held" + (f", {new[card]} new" if new[card] else "")
            options.append(Option(int(card), DEV_LABELS[card], bool(self._plays(card)), detail))
        if held[DevCard.VICTORY_POINT]:
            n = held[DevCard.VICTORY_POINT]
            options.append(Option(int(DevCard.VICTORY_POINT), DEV_LABELS[4], False, f"{n} held"))
        return Picker("dev", "Play a development card", tuple(options))

    def _choose_dev(self, card: Hashable) -> Action | None:
        if card == DevCard.YEAR_OF_PLENTY:
            self.picker = self._plenty_picker()
            return None
        if card == DevCard.MONOPOLY:
            plays = self._plays(DevCard.MONOPOLY)
            options = self._resource_options({a.resource for a in plays})
            self.picker = Picker("monopoly", "Monopoly: claim every card of", options)
            return None
        action = next(a for a in self._by_type[_T.PLAY_DEV_CARD] if a.dev == card)
        self._close()
        return action

    def _plenty_picker(self) -> Picker:
        plays = self._plays(DevCard.YEAR_OF_PLENTY)
        first = self._first
        if first < 0:
            ok = {r for a in plays for r in (a.resource, a.resource2)}
            return Picker("plenty", "Year of Plenty: take a first card", self._resource_options(ok))
        ok = {r for r in _RESOURCES if A.YEAR_OF_PLENTY[first][r] in self._legal_set}
        title = f"Year of Plenty: {RESOURCE_LABELS[first]} and a second card"
        return Picker("plenty", title, self._resource_options(ok))

    def _choose_plenty(self, key: Hashable) -> Action | None:
        if not isinstance(key, int):
            return None
        if self._first < 0:
            self._first = key
            self.picker = self._plenty_picker()
            return None
        action = A.YEAR_OF_PLENTY[self._first][key]
        self._close()
        return action

    def _give_picker(self) -> Picker:
        assert self.state is not None
        ratios, hand = trade_ratios(self.state, self.seat), self._hand()
        trades = self._by_type.get(_T.MARITIME_TRADE, ())
        options = tuple(
            Option(
                r,
                f"{ratios[r]} {RESOURCE_LABELS[r]}",
                any(a.resource == r for a in trades),
                f"you have {hand[r]}",
                resource=r,
            )
            for r in _RESOURCES
        )
        return Picker("give", "Bank trade: give", options)

    def _get_picker(self) -> Picker:
        assert self.state is not None
        give, bank = self._first, self.state.bank
        ratio = trade_ratios(self.state, self.seat)[give]
        options = tuple(
            Option(
                r,
                f"1 {RESOURCE_LABELS[r]}",
                A.MARITIME[give][r] in self._legal_set,
                f"bank has {bank[r]}",
                resource=r,
            )
            for r in _RESOURCES
            if r != give
        )
        return Picker("get", f"Bank trade: {ratio} {RESOURCE_LABELS[give]} for", options)

    def _victim_picker(self, moves: list[Action]) -> Picker:
        assert self.state is not None
        s = self.state
        options = tuple(
            Option(a.player, f"P{a.player}", True, _cards(s.hand_size(a.player)), player=a.player)
            for a in moves
        )
        return Picker("victim", "Steal a card from", options)

    def _discard_picker(self) -> Picker:
        assert self.state is not None
        left = self.state.discards_pending[self.seat]
        hand = self._hand()
        ok = {a.resource for a in self._by_type[_T.DISCARD]}
        options = self._resource_options(ok, [f"you have {n}" for n in hand])
        title = f"Discard {left} more {'card' if left == 1 else 'cards'}"
        detail = "A 7 was rolled: give back half of your cards, one at a time"
        return Picker("discard", title, options, cancellable=False, detail=detail)

    def _response_picker(self) -> Picker:
        assert self.state is not None
        s = self.state
        offer = s.pending_trade or TradeOffer((0,) * NUM_RESOURCES, (0,) * NUM_RESOURCES)
        title = f"P{s.current_player} offers {resources_text(offer.give)}"
        options = (
            Option("accept", "Accept", _T.ACCEPT_TRADE in self._by_type),
            Option("reject", "Reject", _T.REJECT_TRADE in self._by_type),
        )
        detail = f"and wants {resources_text(offer.receive)} in return"
        return Picker("response", title, options, cancellable=False, detail=detail)

    def _confirm_picker(self) -> Picker:
        partners = tuple(
            Option(a.player, f"Trade with P{a.player}", True, player=a.player)
            for a in self._by_type.get(_T.CONFIRM_TRADE, ())
        )
        cancel = Option("cancel", "Cancel trade", _T.CANCEL_TRADE in self._by_type)
        detail = "" if partners else "Nobody accepted"
        return Picker("confirm", "Complete your trade", (*partners, cancel), False, detail)

    def _offer_picker(self) -> Picker:
        hand, give, get = self._hand(), self._give, self._get
        options = []
        for r in _RESOURCES:
            options += [
                Option(("give", r, 1), "+", give[r] < hand[r], resource=r),
                Option(("give", r, -1), "-", give[r] > 0, resource=r),
                Option(("get", r, 1), "+", True, resource=r),
                Option(("get", r, -1), "-", get[r] > 0, resource=r),
            ]
        options.append(Option("propose", "Propose", any(give) and any(get)))
        detail = f"Give {resources_text(give)} for {resources_text(get)}"
        return Picker("offer", "Offer a trade", tuple(options), detail=detail)

    def _choose_offer(self, key: Hashable) -> Action | None:
        if key == "propose":
            offer = TradeOffer(tuple(self._give), tuple(self._get))
            self._close()
            return Action(_T.PROPOSE_TRADE, trade=offer)
        if isinstance(key, tuple):
            side, r, delta = key
            (self._give if side == "give" else self._get)[r] += delta
            self.picker = self._offer_picker()
        return None
