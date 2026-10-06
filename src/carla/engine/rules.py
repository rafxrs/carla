"""Rules of the 4-player base game as functions over `GameState`.

`legal_actions` enumerates every legal decision; `apply_action` performs one (mutating the state)
and returns the broadcast events. Neither validates its input: `Game` does that.

Domestic trade (gated by `EngineConfig.allow_domestic_trade`):
  1. MAIN: the current player may PROPOSE_TRADE(offer) or INVITE_TRADE up to
     `max_trade_proposals_per_turn` times.
     One-for-one offers are enumerated for the baseline policy; other valid offers are validated
     on submission (give <= own hand, give/receive disjoint and non-empty).
  2. TRADE_RESPONSE: every other seat, in turn order, accepts (if able), rejects, or makes one
     counteroffer. A counteroffer is expressed from that responder's perspective.
  3. TRADE_CONFIRM: the active player picks one accepted offer or affordable counteroffer,
     or cancels. Trades always include the active player.
  4. Back to MAIN. All negotiation terms and responses are public events.
"""

from typing import Any

from carla.engine import actions as A
from carla.engine.actions import Action, ActionType
from carla.engine.constants import (
    AWARD_VP,
    CITIES_PER_PLAYER,
    CITY_COST,
    DESERT,
    DEV_CARD_COST,
    GENERIC_PORT,
    LARGEST_ARMY_MIN,
    LONGEST_ROAD_MIN,
    NUM_PLAYERS,
    NUM_RESOURCES,
    ROAD_COST,
    SETTLEMENT_COST,
    SETTLEMENTS_PER_PLAYER,
    Building,
    DevCard,
)
from carla.engine.events import Event, EventKind
from carla.engine.rng import Rng
from carla.engine.state import SETUP_ORDER, GameState, Phase
from carla.engine.topology import (
    CORNER_EDGES,
    CORNER_LINKS,
    CORNER_NEIGHBORS,
    CORNER_TILES,
    EDGE_CORNERS,
    NUM_CORNERS,
    NUM_EDGES,
    NUM_TILES,
    PORT_CORNERS,
    TILE_CORNERS,
)

KNIGHT = DevCard.KNIGHT
ROAD_BUILDING = DevCard.ROAD_BUILDING
YEAR_OF_PLENTY = DevCard.YEAR_OF_PLENTY
MONOPOLY = DevCard.MONOPOLY
VICTORY_POINT = DevCard.VICTORY_POINT
SETTLEMENT = Building.SETTLEMENT
CITY = Building.CITY
_RESOURCES = range(NUM_RESOURCES)
_SEATS = range(NUM_PLAYERS)


class ChanceOutcomeError(ValueError):
    """A forced chance outcome is impossible in the current state."""


# --------------------------------------------------------------------------- queries


def victory_points(s: GameState, player: int, include_hidden: bool = True) -> int:
    """Victory points of `player`; hidden victory point cards only count if `include_hidden`."""
    vp = (SETTLEMENTS_PER_PLAYER - s.settlements_left[player]) + 2 * (
        CITIES_PER_PLAYER - s.cities_left[player]
    )
    if s.longest_road_owner == player:
        vp += AWARD_VP
    if s.largest_army_owner == player:
        vp += AWARD_VP
    if include_hidden:
        vp += s.dev_hands[player][VICTORY_POINT]
    return vp


def can_afford(hand: list[int], cost: tuple[int, ...]) -> bool:
    return all(h >= c for h, c in zip(hand, cost, strict=True))


def can_propose_trade(s: GameState) -> bool:
    """Whether the current player may initiate another offer this turn."""
    p = s.current_player
    config = s.config
    proposal_limit = config.max_trade_proposals_per_turn
    trade_limit = config.max_domestic_trades_per_turn
    return (
        config.allow_domestic_trade
        and s.phase == Phase.MAIN
        and s.decision_player == p
        and (proposal_limit is None or s.trade_proposals_this_turn < proposal_limit)
        and (trade_limit is None or s.domestic_trades_this_turn < trade_limit)
        and any(s.hands[p])
    )


def can_counter_trade(s: GameState) -> bool:
    """The responding player may offer terms using only cards in their own hand."""
    return (
        s.config.allow_domestic_trade
        and s.phase == Phase.TRADE_RESPONSE
        and s.decision_player != s.current_player
        and any(s.hands[s.decision_player])
    )


def valid_trade_offer(s: GameState, offer: A.TradeOffer | None) -> bool:
    """Validate public terms against the acting offer maker's hand, never the other hand."""
    if not isinstance(offer, A.TradeOffer):
        return False
    give, receive = offer
    if type(give) is not tuple or type(receive) is not tuple:
        return False
    if len(give) != NUM_RESOURCES or len(receive) != NUM_RESOURCES:
        return False
    if any(type(n) is not int or n < 0 for n in (*give, *receive)):
        return False
    if not any(give) or not any(receive):
        return False
    if any(give[r] and receive[r] for r in _RESOURCES):
        return False
    limit = s.config.max_trade_cards_per_side
    if limit and (sum(give) > limit or sum(receive) > limit):
        return False
    return can_afford(s.hands[s.decision_player], give)


def trade_ratios(s: GameState, player: int) -> list[int]:
    """Cards of each resource `player` must give the bank for one card of another type."""
    ratios = [4] * NUM_RESOURCES
    owner = s.corner_owner
    generic = False
    for kind, (a, b) in zip(s.board.port_types, PORT_CORNERS, strict=True):
        if owner[a] == player or owner[b] == player:
            if kind == GENERIC_PORT:
                generic = True
            else:
                ratios[kind] = 2
    if generic:
        return [min(r, 3) for r in ratios]
    return ratios


def distance_rule_ok(s: GameState, corner: int) -> bool:
    """Corner is empty and none of its neighbours holds a building."""
    owner = s.corner_owner
    return owner[corner] < 0 and all(owner[n] < 0 for n in CORNER_NEIGHBORS[corner])


def setup_settlement_corners(s: GameState) -> list[int]:
    return [c for c in range(NUM_CORNERS) if distance_rule_ok(s, c)]


def settlement_corners(s: GameState, player: int) -> list[int]:
    """Corners where `player` may build a settlement during normal play (cost aside)."""
    edge_owner = s.edge_owner
    candidates = {c for e in range(NUM_EDGES) if edge_owner[e] == player for c in EDGE_CORNERS[e]}
    return sorted(c for c in candidates if distance_rule_ok(s, c))


def city_corners(s: GameState, player: int) -> list[int]:
    owner, building = s.corner_owner, s.corner_building
    return [c for c in range(NUM_CORNERS) if owner[c] == player and building[c] == SETTLEMENT]


def road_edges(s: GameState, player: int) -> list[int]:
    """Empty edges connected to `player`'s buildings, or to their roads through a corner that
    holds no opponent building (cost aside)."""
    edge_owner, owner = s.edge_owner, s.corner_owner
    frontier = {c for c in range(NUM_CORNERS) if owner[c] == player}
    for e in range(NUM_EDGES):
        if edge_owner[e] == player:
            a, b = EDGE_CORNERS[e]
            if owner[a] < 0:
                frontier.add(a)
            if owner[b] < 0:
                frontier.add(b)
    return sorted({e for c in frontier for e in CORNER_EDGES[c] if edge_owner[e] < 0})


def robber_victims(s: GameState, tile: int, player: int) -> list[int]:
    """Opponents with a building on `tile` and at least one resource card."""
    owner, hands = s.corner_owner, s.hands
    victims = {
        o for c in TILE_CORNERS[tile] if (o := owner[c]) >= 0 and o != player and any(hands[o])
    }
    return sorted(victims)


def playable_dev_cards(s: GameState, player: int) -> list[int]:
    """Count of each development card `player` may play right now (VP cards are never played)."""
    if s.dev_played_this_turn or player != s.current_player:
        return [0] * len(DevCard)
    held, bought = s.dev_hands[player], s.dev_bought_this_turn
    counts = [held[d] - bought[d] for d in DevCard]
    counts[VICTORY_POINT] = 0
    return counts


def longest_road_length(s: GameState, player: int) -> int:
    """Longest trail (no edge reused) in `player`'s roads, cut at opponent buildings."""
    edge_owner, owner = s.edge_owner, s.corner_owner
    own = [e for e in range(NUM_EDGES) if edge_owner[e] == player]
    if not own:
        return 0
    best = 0

    def walk(corner: int, used: int, length: int) -> None:
        nonlocal best
        if length > best:
            best = length
        if length and owner[corner] >= 0 and owner[corner] != player:
            return
        for edge, far in CORNER_LINKS[corner]:
            if edge_owner[edge] == player and not (used >> edge) & 1:
                walk(far, used | (1 << edge), length + 1)

    for corner in sorted({c for e in own for c in EDGE_CORNERS[e]}):
        walk(corner, 0, 0)
        if best == len(own):
            break
    return best


def legal_actions(s: GameState) -> list[Action]:
    """All legal decisions for `s.decision_player`, in a deterministic order."""
    phase, p = s.phase, s.decision_player
    if phase == Phase.MAIN:
        return _main_actions(s, p)
    if phase == Phase.PRE_ROLL:
        return [A.ROLL, *_dev_card_actions(s, p)]
    if phase == Phase.DISCARD:
        hand = s.hands[p]
        return [A.DISCARD_OF[r] for r in _RESOURCES if hand[r]]
    if phase == Phase.MOVE_ROBBER:
        return _robber_actions(s, p)
    if phase == Phase.ROAD_BUILDING:
        return [A.ROAD_AT[e] for e in road_edges(s, p)]
    if phase == Phase.SETUP_SETTLEMENT:
        return [A.SETTLEMENT_AT[c] for c in setup_settlement_corners(s)]
    if phase == Phase.SETUP_ROAD:
        edge_owner = s.edge_owner
        return [A.ROAD_AT[e] for e in CORNER_EDGES[s.setup_corner] if edge_owner[e] < 0]
    if phase == Phase.GAME_OVER:
        return []
    if phase == Phase.TRADE_RESPONSE:
        out = [A.REJECT_TRADE]
        if s.pending_trade is not None and can_afford(s.hands[p], s.pending_trade.receive):
            out.insert(0, A.ACCEPT_TRADE)
        if can_counter_trade(s):
            out.extend(
                offer
                for give in _RESOURCES
                if s.hands[p][give]
                for offer in A.ONE_FOR_ONE[give]
                if offer is not None
            )
        return out
    if phase == Phase.TRADE_CONFIRM:
        return [
            *(
                A.CONFIRM_TRADE_WITH[q]
                for q in _SEATS
                if s.trade_acceptances[q]
                or (
                    s.trade_counteroffers[q] is not None
                    and can_afford(s.hands[s.current_player], s.trade_counteroffers[q].receive)
                )
            ),
            A.CANCEL_TRADE,
        ]
    raise ValueError(f"unknown game phase {phase}")


def _main_actions(s: GameState, p: int) -> list[Action]:
    hand = s.hands[p]
    out = [A.END_TURN]
    if s.roads_left[p] and can_afford(hand, ROAD_COST):
        out.extend(A.ROAD_AT[e] for e in road_edges(s, p))
    if s.settlements_left[p] and can_afford(hand, SETTLEMENT_COST):
        out.extend(A.SETTLEMENT_AT[c] for c in settlement_corners(s, p))
    if s.cities_left[p] and can_afford(hand, CITY_COST):
        out.extend(A.CITY_AT[c] for c in city_corners(s, p))
    if any(s.dev_deck) and can_afford(hand, DEV_CARD_COST):
        out.append(A.BUY_DEV)
    out.extend(_dev_card_actions(s, p))
    ratios, bank = trade_ratios(s, p), s.bank
    for give in _RESOURCES:
        if hand[give] >= ratios[give]:
            row = A.MARITIME[give]
            out.extend(a for r in _RESOURCES if bank[r] and (a := row[r]) is not None)
    if can_propose_trade(s):
        out.append(A.INVITE_TRADE)
        out.extend(
            offer
            for give in _RESOURCES
            if hand[give]
            for offer in A.ONE_FOR_ONE[give]
            if offer is not None
        )
    return out


def _dev_card_actions(s: GameState, p: int) -> list[Action]:
    playable = playable_dev_cards(s, p)
    out = []
    if playable[KNIGHT]:
        out.append(A.PLAY_KNIGHT)
    if playable[ROAD_BUILDING] and s.roads_left[p] and road_edges(s, p):
        out.append(A.PLAY_ROAD_BUILDING)
    if playable[YEAR_OF_PLENTY]:
        bank = s.bank
        for a in _RESOURCES:
            for b in range(a, NUM_RESOURCES):
                if bank[a] and bank[b] >= (2 if a == b else 1):
                    out.append(A.YEAR_OF_PLENTY[a][b])
    if playable[MONOPOLY]:
        out.extend(A.MONOPOLY_ON)
    return out


def _robber_actions(s: GameState, p: int) -> list[Action]:
    out = []
    for t in range(NUM_TILES):
        if t == s.robber_tile:
            continue
        row = A.ROBBER_TO[t]
        victims = robber_victims(s, t, p)
        if victims:
            out.extend(row[v + 1] for v in victims)
        else:
            out.append(row[0])
    return out


# --------------------------------------------------------------------------- transitions


def apply_action(
    s: GameState, action: Action, rng: Rng, outcome: Any = None
) -> tuple[list[Event], Any]:
    """Apply a legal `action` for `s.decision_player`.

    Chance (dice, development card drawn, card stolen) is always sampled from `rng`, so the RNG
    advances identically whether or not `outcome` is given; a non-None `outcome` then overrides
    the sample. Returns (events, outcome used or None).
    """
    events: list[Event] = []
    used = _HANDLERS[action.type](s, action, s.decision_player, rng, outcome, events)
    if s.phase != Phase.GAME_OVER and s.setup_step >= len(SETUP_ORDER):
        cp = s.current_player
        if victory_points(s, cp) >= s.config.vp_to_win:
            _finish(s, cp, events)
        elif (
            action.type == ActionType.END_TURN
            and s.config.max_turns is not None
            and s.turn >= s.config.max_turns
        ):
            _finish(s, -1, events)
    return events, used


def _finish(s: GameState, winner: int, events: list[Event]) -> None:
    s.winner = winner
    s.phase = Phase.GAME_OVER
    s.decision_player = -1
    events.append(Event(EventKind.GAME_OVER, winner))


def _pay(s: GameState, p: int, cost: tuple[int, ...]) -> None:
    hand, bank = s.hands[p], s.bank
    for r in _RESOURCES:
        hand[r] -= cost[r]
        bank[r] += cost[r]


def _after_interrupt(s: GameState) -> Phase:
    return Phase.MAIN if s.has_rolled else Phase.PRE_ROLL


def _roll_dice(s, action, p, rng, outcome, events):
    d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
    if outcome is not None:
        d1, d2 = outcome
        if not (1 <= d1 <= 6 and 1 <= d2 <= 6):
            raise ChanceOutcomeError(f"invalid dice {outcome}")
    total = d1 + d2
    s.has_rolled = True
    s.last_roll = total
    events.append(Event(EventKind.DICE_ROLLED, p, value=total, counts=(d1, d2)))
    if total == 7:
        limit = s.config.discard_limit
        for q in _SEATS:
            n = sum(s.hands[q])
            s.discards_pending[q] = n // 2 if n > limit else 0
        nxt = _next_discarder(s)
        if nxt >= 0:
            s.phase = Phase.DISCARD
            s.decision_player = nxt
        else:
            s.phase = Phase.MOVE_ROBBER
    else:
        _produce(s, total, events)
        s.phase = Phase.MAIN
    return (d1, d2)


def _produce(s: GameState, number: int, events: list[Event]) -> None:
    owner, building, board = s.corner_owner, s.corner_building, s.board
    gains = [[0] * NUM_RESOURCES for _ in _SEATS]
    for t in board.tiles_by_number[number]:
        if t == s.robber_tile:
            continue
        r = board.tile_resource[t]
        for c in TILE_CORNERS[t]:
            if (o := owner[c]) >= 0:
                gains[o][r] += building[c]
    bank = s.bank
    for r in _RESOURCES:
        receivers = [q for q in _SEATS if gains[q][r]]
        if sum(gains[q][r] for q in receivers) > bank[r]:
            # Shortage: nobody is paid, unless a single player is owed this resource.
            if len(receivers) == 1:
                gains[receivers[0]][r] = bank[r]
            else:
                for q in receivers:
                    gains[q][r] = 0
    for q in _SEATS:
        gain = gains[q]
        if any(gain):
            hand = s.hands[q]
            for r in _RESOURCES:
                hand[r] += gain[r]
                bank[r] -= gain[r]
            events.append(Event(EventKind.RESOURCES_PRODUCED, q, counts=tuple(gain)))


def _next_discarder(s: GameState) -> int:
    for i in _SEATS:
        q = (s.current_player + i) % NUM_PLAYERS
        if s.discards_pending[q]:
            return q
    return -1


def _discard(s, action, p, rng, outcome, events):
    r = action.resource
    s.hands[p][r] -= 1
    s.bank[r] += 1
    s.discards_pending[p] -= 1
    events.append(Event(EventKind.DISCARDED, p, value=r))
    if not s.discards_pending[p]:
        nxt = _next_discarder(s)
        if nxt >= 0:
            s.decision_player = nxt
        else:
            s.phase = Phase.MOVE_ROBBER
            s.decision_player = s.current_player


def _move_robber(s, action, p, rng, outcome, events):
    s.robber_tile = action.tile
    events.append(Event(EventKind.ROBBER_MOVED, p, location=action.tile))
    stolen = None
    victim = action.player
    if victim < 0 and outcome is not None:
        raise ChanceOutcomeError("nothing is stolen, so there is no outcome to force")
    if victim >= 0:
        hand = s.hands[victim]
        stolen = rng.weighted_index(hand)
        if outcome is not None:
            if not (0 <= outcome < NUM_RESOURCES and hand[outcome]):
                raise ChanceOutcomeError(f"seat {victim} holds no resource {outcome}")
            stolen = outcome
        hand[stolen] -= 1
        s.hands[p][stolen] += 1
        events.append(Event(EventKind.RESOURCE_STOLEN, p, target=victim, value=stolen))
    s.phase = _after_interrupt(s)
    return stolen


def _build_settlement(s, action, p, rng, outcome, events):
    c = action.corner
    setup = s.phase == Phase.SETUP_SETTLEMENT
    if not setup:
        _pay(s, p, SETTLEMENT_COST)
    s.corner_owner[c] = p
    s.corner_building[c] = SETTLEMENT
    s.settlements_left[p] -= 1
    events.append(Event(EventKind.SETTLEMENT_BUILT, p, location=c))
    cut = {s.edge_owner[e] for e in CORNER_EDGES[c]} - {-1, p}
    if cut:
        for q in cut:
            s.road_lengths[q] = longest_road_length(s, q)
        _update_longest_road(s, events)
    if setup:
        if s.setup_step >= NUM_PLAYERS:
            _grant_starting_resources(s, p, c, events)
        s.setup_corner = c
        s.phase = Phase.SETUP_ROAD


def _grant_starting_resources(s: GameState, p: int, c: int, events: list[Event]) -> None:
    gain = [0] * NUM_RESOURCES
    for t in CORNER_TILES[c]:
        r = s.board.tile_resource[t]
        if r != DESERT and s.bank[r] > gain[r]:
            gain[r] += 1
    if any(gain):
        for r in _RESOURCES:
            s.hands[p][r] += gain[r]
            s.bank[r] -= gain[r]
        events.append(Event(EventKind.STARTING_RESOURCES, p, counts=tuple(gain)))


def _build_city(s, action, p, rng, outcome, events):
    c = action.corner
    _pay(s, p, CITY_COST)
    s.corner_building[c] = CITY
    s.cities_left[p] -= 1
    s.settlements_left[p] += 1
    events.append(Event(EventKind.CITY_BUILT, p, location=c))


def _build_road(s, action, p, rng, outcome, events):
    e = action.edge
    phase = s.phase
    if phase == Phase.MAIN:
        _pay(s, p, ROAD_COST)
    s.edge_owner[e] = p
    s.roads_left[p] -= 1
    events.append(Event(EventKind.ROAD_BUILT, p, location=e))
    s.road_lengths[p] = longest_road_length(s, p)
    _update_longest_road(s, events)
    if phase == Phase.SETUP_ROAD:
        _advance_setup(s)
    elif phase == Phase.ROAD_BUILDING:
        s.free_roads -= 1
        if not s.free_roads or not s.roads_left[p] or not road_edges(s, p):
            s.free_roads = 0
            s.phase = _after_interrupt(s)


def _advance_setup(s: GameState) -> None:
    s.setup_step += 1
    s.setup_corner = -1
    if s.setup_step == len(SETUP_ORDER):
        s.phase = Phase.PRE_ROLL
        s.current_player = s.decision_player = SETUP_ORDER[0]
    else:
        s.phase = Phase.SETUP_SETTLEMENT
        s.current_player = s.decision_player = SETUP_ORDER[s.setup_step]


def _buy_dev_card(s, action, p, rng, outcome, events):
    _pay(s, p, DEV_CARD_COST)
    deck = s.dev_deck
    card = rng.weighted_index(deck)
    if outcome is not None:
        if not (0 <= outcome < len(deck) and deck[outcome]):
            raise ChanceOutcomeError(f"no development card {outcome} left in the deck")
        card = outcome
    deck[card] -= 1
    s.dev_hands[p][card] += 1
    s.dev_bought_this_turn[card] += 1
    events.append(Event(EventKind.DEV_CARD_BOUGHT, p, value=card))
    return card


def _play_dev_card(s, action, p, rng, outcome, events):
    card = action.dev
    s.dev_hands[p][card] -= 1
    s.dev_played[p][card] += 1
    s.dev_played_this_turn = True
    events.append(Event(EventKind.DEV_CARD_PLAYED, p, value=card))
    if card == KNIGHT:
        _update_largest_army(s, p, events)
        s.phase = Phase.MOVE_ROBBER
    elif card == ROAD_BUILDING:
        s.free_roads = 2
        s.phase = Phase.ROAD_BUILDING
    elif card == YEAR_OF_PLENTY:
        gain = [0] * NUM_RESOURCES
        gain[action.resource] += 1
        gain[action.resource2] += 1
        for r in _RESOURCES:
            s.bank[r] -= gain[r]
            s.hands[p][r] += gain[r]
        events.append(Event(EventKind.YEAR_OF_PLENTY_TAKEN, p, counts=tuple(gain)))
    elif card == MONOPOLY:
        r = action.resource
        taken = [0] * NUM_PLAYERS
        for q in _SEATS:
            if q != p:
                taken[q] = s.hands[q][r]
                s.hands[q][r] = 0
        s.hands[p][r] += sum(taken)
        events.append(Event(EventKind.MONOPOLY_TAKEN, p, value=r, counts=tuple(taken)))


def _maritime_trade(s, action, p, rng, outcome, events):
    give, receive = action.resource, action.resource2
    ratio = trade_ratios(s, p)[give]
    hand, bank = s.hands[p], s.bank
    hand[give] -= ratio
    bank[give] += ratio
    bank[receive] -= 1
    hand[receive] += 1
    delta = [0] * NUM_RESOURCES
    delta[give], delta[receive] = -ratio, 1
    events.append(Event(EventKind.MARITIME_TRADED, p, counts=tuple(delta)))


def _end_turn(s, action, p, rng, outcome, events):
    events.append(Event(EventKind.TURN_ENDED, p))
    s.turn += 1
    s.current_player = s.decision_player = (p + 1) % NUM_PLAYERS
    s.phase = Phase.PRE_ROLL
    s.has_rolled = False
    s.last_roll = 0
    s.dev_played_this_turn = False
    s.dev_bought_this_turn = [0] * len(DevCard)
    s.trade_proposals_this_turn = 0
    s.domestic_trades_this_turn = 0


def _domestic_trade(s, action, p, rng, outcome, events):
    kind = action.type
    proposer = s.current_player
    if s.phase == Phase.MAIN and kind in (ActionType.PROPOSE_TRADE, ActionType.INVITE_TRADE):
        if kind == ActionType.PROPOSE_TRADE:
            assert valid_trade_offer(s, action.trade)
        s.pending_trade = action.trade if kind == ActionType.PROPOSE_TRADE else None
        s.trade_proposals_this_turn += 1
        s.trade_responses = 0
        s.trade_acceptances = [False] * NUM_PLAYERS
        s.trade_counteroffers = [None] * NUM_PLAYERS
        s.phase = Phase.TRADE_RESPONSE
        s.decision_player = (proposer + 1) % NUM_PLAYERS
        if kind == ActionType.PROPOSE_TRADE:
            events.append(
                Event(
                    EventKind.TRADE_PROPOSED,
                    proposer,
                    give=action.trade.give,
                    receive=action.trade.receive,
                )
            )
        else:
            events.append(Event(EventKind.TRADE_INVITED, proposer))
        return
    if kind in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE, ActionType.PROPOSE_TRADE):
        accepted = kind == ActionType.ACCEPT_TRADE
        s.trade_acceptances[p] = accepted
        if kind == ActionType.PROPOSE_TRADE:
            assert valid_trade_offer(s, action.trade)
            s.trade_counteroffers[p] = action.trade
        s.trade_responses += 1
        if kind == ActionType.PROPOSE_TRADE:
            events.append(
                Event(
                    EventKind.TRADE_COUNTERED,
                    p,
                    target=proposer,
                    give=action.trade.give,
                    receive=action.trade.receive,
                )
            )
        else:
            events.append(Event(EventKind.TRADE_RESPONDED, p, target=proposer, value=int(accepted)))
        if s.trade_responses == NUM_PLAYERS - 1:
            s.phase = Phase.TRADE_CONFIRM
            s.decision_player = proposer
        else:
            s.decision_player = (proposer + 1 + s.trade_responses) % NUM_PLAYERS
        return
    if kind == ActionType.CONFIRM_TRADE:
        partner = action.player
        counter = s.trade_counteroffers[partner]
        if counter is None:
            assert s.pending_trade is not None
            give, receive = s.pending_trade
        else:
            give, receive = counter.receive, counter.give
        for r in _RESOURCES:
            s.hands[proposer][r] += receive[r] - give[r]
            s.hands[partner][r] += give[r] - receive[r]
        s.domestic_trades_this_turn += 1
        events.append(
            Event(
                EventKind.TRADE_EXECUTED,
                proposer,
                target=partner,
                give=give,
                receive=receive,
            )
        )
    elif kind == ActionType.CANCEL_TRADE:
        events.append(Event(EventKind.TRADE_CANCELLED, proposer))
    else:
        raise ValueError(f"unexpected trade action {kind}")
    s.pending_trade = None
    s.trade_responses = 0
    s.trade_acceptances = [False] * NUM_PLAYERS
    s.trade_counteroffers = [None] * NUM_PLAYERS
    s.phase = Phase.MAIN
    s.decision_player = proposer


def _update_longest_road(s: GameState, events: list[Event]) -> None:
    """Award Longest Road: a tie never takes the card from its holder; when the holder is no
    longer longest, the card goes to a unique leader (>= 5 roads) or is set aside."""
    lengths, holder = s.road_lengths, s.longest_road_owner
    best = max(lengths)
    if holder >= 0 and lengths[holder] == best and best >= LONGEST_ROAD_MIN:
        return
    leaders = [q for q in _SEATS if lengths[q] == best]
    new = leaders[0] if best >= LONGEST_ROAD_MIN and len(leaders) == 1 else -1
    if new != holder:
        s.longest_road_owner = new
        events.append(Event(EventKind.LONGEST_ROAD_CHANGED, new, target=holder))


def _update_largest_army(s: GameState, p: int, events: list[Event]) -> None:
    knights, holder = s.dev_played[p][KNIGHT], s.largest_army_owner
    if holder == p or knights < LARGEST_ARMY_MIN:
        return
    if holder < 0 or knights > s.dev_played[holder][KNIGHT]:
        s.largest_army_owner = p
        events.append(Event(EventKind.LARGEST_ARMY_CHANGED, p, target=holder))


_HANDLERS = {
    ActionType.ROLL_DICE: _roll_dice,
    ActionType.END_TURN: _end_turn,
    ActionType.BUILD_SETTLEMENT: _build_settlement,
    ActionType.BUILD_CITY: _build_city,
    ActionType.BUILD_ROAD: _build_road,
    ActionType.BUY_DEV_CARD: _buy_dev_card,
    ActionType.PLAY_DEV_CARD: _play_dev_card,
    ActionType.MOVE_ROBBER: _move_robber,
    ActionType.DISCARD: _discard,
    ActionType.MARITIME_TRADE: _maritime_trade,
    ActionType.PROPOSE_TRADE: _domestic_trade,
    ActionType.INVITE_TRADE: _domestic_trade,
    ActionType.ACCEPT_TRADE: _domestic_trade,
    ActionType.REJECT_TRADE: _domestic_trade,
    ActionType.CONFIRM_TRADE: _domestic_trade,
    ActionType.CANCEL_TRADE: _domestic_trade,
}
