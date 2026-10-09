"""Turns the human seat's clicks and button presses into engine `Action`s (pure: no Arcade).

The controller never decides what is legal: highlighted targets, enabled buttons and picker
options come from the legal actions passed to `reset`. Trades start from the hand: clicking a
resource card puts it in an offer, which goes to the bank (as maritime trades) or to the other
players (checked with the engine's `valid_trade_offer`, since multi-card offers are not
enumerated).
"""

from collections.abc import Collection, Hashable, Sequence
from typing import NamedTuple

from carla.engine import actions as A
from carla.engine.actions import Action, ActionType, TradeOffer
from carla.engine.constants import NUM_PLAYERS, NUM_RESOURCES, RESOURCE_CARDS_PER_TYPE, DevCard
from carla.engine.rules import trade_ratios, valid_trade_offer
from carla.engine.state import GameState, Phase
from carla.gui.theme import DEV_LABELS, RESOURCE_LABELS

_T = ActionType
_RESOURCES = range(NUM_RESOURCES)
_NONE = (0,) * NUM_RESOURCES


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


class TradeButton(NamedTuple):
    key: Hashable
    label: str
    enabled: bool = True
    accent: bool = False


class TradeRow(NamedTuple):
    """One seat's line of a negotiation: its answer and the terms, as the cards the trading
    seat would give and get."""

    player: int
    label: str
    status: str = ""
    give: tuple[int, ...] = _NONE
    get: tuple[int, ...] = _NONE
    button: TradeButton | None = None


class TradePanel(NamedTuple):
    """What the trade panel above the hand shows.

    "offer" and "counter" compose an offer: the cards to `give` (clicked in the hand) and to
    `get` (clicked among the five resources, those `wants` allows). "respond" and "responses"
    show a negotiation as `rows` of terms under the two `columns`."""

    kind: str
    title: str
    buttons: tuple[TradeButton, ...] = ()
    hint: str = ""
    give: tuple[int, ...] = _NONE
    get: tuple[int, ...] = _NONE
    wants: tuple[bool, ...] = (False,) * NUM_RESOURCES
    rows: tuple[TradeRow, ...] = ()
    columns: tuple[str, str] = ("You give", "You get")

    @property
    def composer(self) -> bool:
        return self.kind in ("offer", "counter")

    def all_buttons(self) -> list[TradeButton]:
        """Row buttons, then the footer buttons: the order of the number keys."""
        return [*(r.button for r in self.rows if r.button is not None), *self.buttons]

    def enabled_keys(self) -> set[Hashable]:
        """Every key a click on the panel may send now."""
        keys: set[Hashable] = {b.key for b in self.all_buttons() if b.enabled}
        if self.composer:
            keys.add("close")
            keys |= {("give", r) for r in _RESOURCES if self.give[r]}
            keys |= {("get", r) for r in _RESOURCES if self.get[r]}
            keys |= {("want", r) for r in _RESOURCES if self.wants[r]}
        return keys


def negotiation_panel(
    state: GameState, viewer: int | None = None, legal: Collection[Action] = ()
) -> TradePanel | None:
    """The domestic trade under negotiation, if any, from the trading (current) seat's side: its
    offer, then each other seat's answer. `legal` holds that seat's choices when it must pick a
    partner now (empty: read-only)."""
    s = state
    if s.phase not in (Phase.TRADE_RESPONSE, Phase.TRADE_CONFIRM):
        return None
    p, offer = s.current_player, s.pending_trade
    you = viewer == p
    choosing = bool(legal)
    rows = []
    if offer is not None:
        rows.append(TradeRow(p, "You" if you else f"P{p}", "offers", offer.give, offer.receive))
    for i in range(1, NUM_PLAYERS):
        q = (p + i) % NUM_PLAYERS
        counter = s.trade_counteroffers[q]
        give, get = _NONE, _NONE
        if counter is not None:
            status, give, get = "counters", counter.receive, counter.give
        elif s.trade_acceptances[q] and offer is not None:
            status, give, get = "accepts", offer.give, offer.receive
        elif s.phase == Phase.TRADE_CONFIRM or i <= s.trade_responses:
            status = "declines"
        else:
            status = "deciding…" if q == s.decision_player else "waiting"
        button = None
        if choosing and any(give):
            button = TradeButton(("confirm", q), "Trade", A.CONFIRM_TRADE_WITH[q] in legal, True)
        rows.append(TradeRow(q, "You" if q == viewer else f"P{q}", status, give, get, button))
    if offer is None:
        title = "You asked for offers" if you else f"P{p} asks for offers"
    else:
        title = "Your trade offer" if you else f"P{p} offers a trade"
    hint = ""
    if choosing:
        partners = any(r.button is not None and r.button.enabled for r in rows)
        hint = "Click Trade beside the deal you want" if partners else "No deal to make"
    elif s.phase == Phase.TRADE_CONFIRM:
        hint = "Choosing a deal…" if you else f"P{p} is choosing a deal…"
    buttons = ()
    if choosing:
        buttons = (TradeButton("cancel_trade", "Cancel trade", A.CANCEL_TRADE in legal),)
    columns = ("You give", "You get") if you else (f"P{p} gives", f"P{p} gets")
    return TradePanel("responses", title, buttons, hint, rows=tuple(rows), columns=columns)


_BUTTONS = (
    ("roll", "Roll", "R", _T.ROLL_DICE),
    ("end_turn", "End turn", "E", _T.END_TURN),
    ("buy_dev", "Buy dev card", "B", _T.BUY_DEV_CARD),
    ("play_dev", "Play dev card", "D", _T.PLAY_DEV_CARD),
)
_SIMPLE = {"roll": _T.ROLL_DICE, "end_turn": _T.END_TURN, "buy_dev": _T.BUY_DEV_CARD}
_ANSWERS = {
    "invite": A.INVITE_TRADE,
    "accept": A.ACCEPT_TRADE,
    "reject": A.REJECT_TRADE,
    "cancel_trade": A.CANCEL_TRADE,
}


def _cards(n: int) -> str:
    return f"{n} card" if n == 1 else f"{n} cards"


class Controller:
    """Input state machine of the human seat, one decision at a time.

    Board clicks (`click_corner/edge/tile`), buttons (`press`), picker choices (`choose`) and
    trade panel clicks (`choose_trade`) return the completed `Action`, or None while more input
    is needed or the input is ignored. `offer_card` adds a hand card to the trade offer.
    """

    def __init__(self, seat: int) -> None:
        self.seat = seat
        self.state: GameState | None = None
        self.legal: tuple[Action, ...] = ()
        self.picker: Picker | None = None
        self._legal_set: frozenset[Action] = frozenset()
        self._by_type: dict[ActionType, list[Action]] = {}
        self._corners: dict[int, Action] = {}
        self._edges: dict[int, Action] = {}
        self._tiles: dict[int, list[Action]] = {}
        self._first = -1
        """First resource of a two-step choice (Year of Plenty)."""
        self._robber_tile = -1
        self._trade = ""
        """The offer being composed: "offer" (own turn), "counter" (answering one) or ""."""
        self._give = [0] * NUM_RESOURCES
        self._get = [0] * NUM_RESOURCES
        self._queue: list[Action] = []

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
        self._queue = []
        if _T.DISCARD in self._by_type:
            self.picker = self._discard_picker()

    @property
    def active(self) -> bool:
        """The human must act now."""
        return bool(self.legal)

    @property
    def offer(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Trade being composed: (cards given, cards wanted) per resource."""
        return tuple(self._give), tuple(self._get)

    def prompt(self) -> str:
        """What the human should do next, in a few words."""
        if self.picker is not None:
            return self.picker.title
        panel = self.trade_panel()
        if panel is not None:
            return panel.title
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
        return "Build on a highlighted spot, click your cards to trade, or end your turn (E)"

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
        return [
            ButtonSpec(name, label, key, free and kind in self._by_type)
            for name, label, key, kind in _BUTTONS
        ]

    def press(self, button: str) -> Action | None:
        spec = next((b for b in self.buttons() if b.id == button), None)
        if spec is None or not spec.enabled:
            return None
        self._close()
        if button == "play_dev":
            self.picker = self._dev_picker()
            return None
        return self._by_type[_SIMPLE[button]][0]

    def playable_cards(self) -> frozenset[int]:
        """Development cards a click would play now (none while a mandatory picker is open)."""
        if self.picker is not None and not self.picker.cancellable:
            return frozenset()
        return frozenset(int(a.dev) for a in self._by_type.get(_T.PLAY_DEV_CARD, ()))

    def play_card(self, card: int) -> Action | None:
        """Play development card `card` as if picked in the Play dev card picker: Year of Plenty
        and Monopoly then ask for their resources."""
        if card not in self.playable_cards():
            return None
        self.press("play_dev")
        return self.choose(card)

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
        action = self._picked_action(kind, key)
        self._close()
        return action

    def choose_index(self, index: int) -> Action | None:
        """Pick the `index`-th option (0-based, as displayed) of the open picker, else the
        `index`-th button of the trade panel, e.g. from the number keys."""
        if self.picker is not None:
            options = self.picker.options
            return self.choose(options[index].key) if 0 <= index < len(options) else None
        panel = self.trade_panel()
        buttons = panel.all_buttons() if panel is not None else []
        return self.choose_trade(buttons[index].key) if 0 <= index < len(buttons) else None

    def cancel(self) -> bool:
        """Close a cancellable picker or the offer being composed (Esc): a counteroffer goes
        back to the answer. Mandatory choices (discards, trade answers) stay."""
        if self.picker is not None:
            if not self.picker.cancellable:
                return False
            self._close()
            return True
        if self._trade:
            self._close()
            return True
        return False

    # ------------------------------------------------------------------ trading

    def offerable_cards(self) -> frozenset[int]:
        """Resources whose hand card a click adds to the offer (opening the trade panel)."""
        s = self.state
        if s is None or not self.legal or (self.picker is not None and not self.picker.cancellable):
            return frozenset()
        hand, give, get = self._hand(), self._give, self._get
        if self._trade == "counter":
            limit = s.config.max_trade_cards_per_side
            if limit and sum(give) >= limit:
                return frozenset()
        elif not self._trade and not self._can_trade():
            return frozenset()
        return frozenset(r for r in _RESOURCES if give[r] < hand[r] and not get[r])

    def offer_card(self, resource: int) -> None:
        """Add one card of the hand to the offer, opening the trade panel first (a counteroffer
        while answering someone else's trade)."""
        if resource not in self.offerable_cards():
            return
        if not self._trade:
            self._open_trade()
        self._give[resource] += 1

    def toggle_trade(self) -> None:
        """Open an empty offer (the T key), or close the one being composed."""
        s = self.state
        if self._trade:
            self._close()
        elif s is not None and s.phase == Phase.MAIN and self._can_trade():
            self._open_trade()

    def trade_panel(self) -> TradePanel | None:
        """The trade panel of the human's decision: the offer being composed, an offer to answer
        or the answers to pick a deal from."""
        s = self.state
        if s is None or not self.legal:
            return None
        if self._trade:
            return self._composer_panel()
        if s.phase == Phase.TRADE_RESPONSE:
            return self._respond_panel()
        if s.phase == Phase.TRADE_CONFIRM:
            return negotiation_panel(s, self.seat, self.legal)
        return None

    def choose_trade(self, key: Hashable) -> Action | None:
        """Click `key` of the trade panel: ("give" or "get", r) takes a card of the offer back,
        ("want", r) asks for one more, other keys are its buttons."""
        panel = self.trade_panel()
        if panel is None or key not in panel.enabled_keys():
            return None
        if isinstance(key, tuple):
            what, value = key
            if what == "confirm":
                return A.CONFIRM_TRADE_WITH[value]
            if what == "want":
                self._get[value] += 1
            else:
                (self._give if what == "give" else self._get)[value] -= 1
            return None
        if key in ("close", "back"):
            self.cancel()
            return None
        if key == "counter":
            self._open_trade()
            return None
        if key == "bank":
            first, *rest = self._bank_trades()
            self._close()
            self._queue = rest
            return first
        if key == "propose":
            action = Action(_T.PROPOSE_TRADE, trade=TradeOffer(tuple(self._give), tuple(self._get)))
        else:
            action = _ANSWERS[str(key)]
        self._close()
        return action

    def take_queued(self) -> list[Action]:
        """The other maritime trades of a multi-card bank trade, to play right after the one
        `choose_trade` returned."""
        queued, self._queue = self._queue, []
        return queued

    # ------------------------------------------------------------------ internals

    def _close(self) -> None:
        self.picker = None
        self._first = self._robber_tile = -1
        self._trade = ""
        self._give, self._get = [0] * NUM_RESOURCES, [0] * NUM_RESOURCES

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

    def _picked_action(self, kind: str, key: Hashable) -> Action | None:
        if kind == "monopoly":
            return A.MONOPOLY_ON[key] if isinstance(key, int) else None
        if kind == "discard":
            return A.DISCARD_OF[key] if isinstance(key, int) else None
        if kind == "victim":
            return next(a for a in self._tiles[self._robber_tile] if a.player == key)
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

    def _can_trade(self) -> bool:
        """An offer can start now: on the main phase (bank or players), or a counteroffer."""
        assert self.state is not None
        kinds = self._by_type
        if self.state.phase == Phase.TRADE_RESPONSE:
            return _T.PROPOSE_TRADE in kinds
        return any(k in kinds for k in (_T.MARITIME_TRADE, _T.PROPOSE_TRADE, _T.INVITE_TRADE))

    def _open_trade(self) -> None:
        assert self.state is not None
        self._close()
        self._trade = "counter" if self.state.phase == Phase.TRADE_RESPONSE else "offer"

    def _bank_trades(self) -> list[Action]:
        """The maritime trades that make the offer, if it is exactly that: each resource given in
        a multiple of its rate, and one card asked (and in the bank) per batch."""
        s = self.state
        if s is None or self._trade != "offer" or _T.MARITIME_TRADE not in self._by_type:
            return []
        ratios, give, get = trade_ratios(s, self.seat), self._give, self._get
        if any(give[r] % ratios[r] or get[r] > s.bank[r] for r in _RESOURCES):
            return []
        batches = [r for r in _RESOURCES for _ in range(give[r] // ratios[r])]
        wanted = [r for r in _RESOURCES for _ in range(get[r])]
        if not batches or len(batches) != len(wanted):
            return []
        trades = [
            a for g, w in zip(batches, wanted, strict=True) if (a := A.MARITIME[g][w]) is not None
        ]
        if len(trades) != len(batches):
            return []
        return trades if trades[0] in self._legal_set else []

    def _get_cap(self) -> int:
        """Most cards the offer may ask for: the players' limit per side, or what the bank
        would give for the whole hand."""
        assert self.state is not None
        limit = self.state.config.max_trade_cards_per_side or RESOURCE_CARDS_PER_TYPE
        if self._trade == "counter":
            return limit
        kinds = self._by_type
        players = limit if _T.PROPOSE_TRADE in kinds else 0
        ratios = trade_ratios(self.state, self.seat)
        bank = sum(n // k for n, k in zip(self._hand(), ratios, strict=True))
        return max(players, bank if _T.MARITIME_TRADE in kinds else 0)

    def _composer_panel(self) -> TradePanel:
        s = self.state
        assert s is not None
        kinds = self._by_type
        give, get = tuple(self._give), tuple(self._get)
        terms = TradeOffer(give, get)
        valid = _T.PROPOSE_TRADE in kinds and valid_trade_offer(s, terms)
        cap = self._get_cap()
        wants = tuple(not give[r] and sum(get) < cap for r in _RESOURCES)
        if self._trade == "counter":
            p = s.current_player
            buttons = (
                TradeButton("back", "Back"),
                TradeButton("propose", "Send counteroffer", valid, True),
            )
            if not any(give):
                hint = f"Click your cards below to offer them to P{p}"
            else:
                hint = "" if any(get) else f"Pick the cards you want from P{p}"
            title = f"Counteroffer to P{p}"
            return TradePanel("counter", title, buttons, hint, give, get, wants)
        domestic = s.config.allow_domestic_trade
        footer = [TradeButton("bank", "Trade with bank", bool(self._bank_trades()), True)]
        if domestic:
            footer.insert(0, TradeButton("invite", "Ask for offers", _T.INVITE_TRADE in kinds))
            footer.append(TradeButton("propose", "Offer to players", valid, True))
        return TradePanel("offer", "Trade", tuple(footer), self._offer_hint(), give, get, wants)

    def _offer_hint(self) -> str:
        assert self.state is not None
        give, get = self._give, self._get
        if not any(give):
            return "Click your cards below to offer them"
        ratios = trade_ratios(self.state, self.seat)
        rates = " or ".join(
            f"{ratios[r]} {RESOURCE_LABELS[r].lower()}" for r in _RESOURCES if give[r]
        )
        hint = f"Bank: 1 card for {rates}"
        if not any(get):
            hint = "Pick the cards you want · " + hint
        limit = self.state.config.max_trade_cards_per_side
        if _T.PROPOSE_TRADE in self._by_type and limit and max(sum(give), sum(get)) > limit:
            hint += f" · players: {limit} cards a side at most"
        return hint

    def _respond_panel(self) -> TradePanel:
        s = self.state
        assert s is not None
        p, offer, kinds = s.current_player, s.pending_trade, self._by_type
        counter = _T.PROPOSE_TRADE in kinds
        buttons = [TradeButton("reject", "Decline", _T.REJECT_TRADE in kinds)]
        if offer is None:
            buttons.append(TradeButton("counter", "Make an offer", counter, True))
            hint = f"Click your cards below to make P{p} an offer" if counter else ""
            return TradePanel("respond", f"P{p} asks for offers", tuple(buttons), hint)
        buttons.append(TradeButton("counter", "Counteroffer", counter))
        buttons.append(TradeButton("accept", "Accept", _T.ACCEPT_TRADE in kinds, True))
        hint = "" if _T.ACCEPT_TRADE in kinds else "You do not have the cards to accept"
        row = TradeRow(p, f"P{p}", "offers", offer.receive, offer.give)
        return TradePanel("respond", f"P{p} offers a trade", tuple(buttons), hint, rows=(row,))
