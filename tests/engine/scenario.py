"""Helpers to build hand-made positions, plus independent reference checks of the rules."""

from carla.engine import rules
from carla.engine.board import Board
from carla.engine.config import EngineConfig
from carla.engine.constants import (
    CITIES_PER_PLAYER,
    DEV_CARD_COUNTS,
    GENERIC_PORT,
    LARGEST_ARMY_MIN,
    LONGEST_ROAD_MIN,
    NUM_PLAYERS,
    NUM_RESOURCES,
    RESOURCE_CARDS_PER_TYPE,
    ROADS_PER_PLAYER,
    SETTLEMENTS_PER_PLAYER,
    Building,
    DevCard,
)
from carla.engine.game import Game
from carla.engine.rng import Rng
from carla.engine.state import SETUP_ORDER, Phase
from carla.engine.topology import (
    CORNER_EDGES,
    CORNER_NEIGHBORS,
    EDGE_CORNERS,
    NUM_CORNERS,
    NUM_EDGES,
    NUM_TILES,
    TILE_CORNERS,
)

BOARD = Board.random(Rng(7))


def edge_between(a: int, b: int) -> int:
    return next(e for e in CORNER_EDGES[a] if b in EDGE_CORNERS[e])


def post_setup_game(
    board: Board = BOARD, seed: int = 0, config: EngineConfig | None = None, rolled: bool = True
) -> Game:
    """Empty board, setup skipped, seat 0 to act in MAIN (or PRE_ROLL if not `rolled`)."""
    game = Game(config, seed, board)
    s = game.state
    s.setup_step = len(SETUP_ORDER)
    s.phase = Phase.MAIN if rolled else Phase.PRE_ROLL
    s.has_rolled = rolled
    game.set_state(s)
    return game


def put_settlement(game: Game, player: int, corner: int, city: bool = False) -> None:
    s = game.state
    s.corner_owner[corner] = player
    s.corner_building[corner] = Building.CITY if city else Building.SETTLEMENT
    if city:
        s.cities_left[player] -= 1
    else:
        s.settlements_left[player] -= 1
    for q in range(NUM_PLAYERS):
        s.road_lengths[q] = rules.longest_road_length(s, q)
    game.set_state(s)


def put_road(game: Game, player: int, edge: int) -> None:
    s = game.state
    s.edge_owner[edge] = player
    s.roads_left[player] -= 1
    s.road_lengths[player] = rules.longest_road_length(s, player)
    game.set_state(s)


def give(game: Game, player: int, counts) -> None:
    """Move resource cards from the bank to `player` (counts per resource)."""
    s = game.state
    for r, n in enumerate(counts):
        s.bank[r] -= n
        s.hands[player][r] += n
    game.set_state(s)


def give_dev(game: Game, player: int, card: DevCard, n: int = 1) -> None:
    s = game.state
    s.dev_deck[card] -= n
    s.dev_hands[player][card] += n
    game.set_state(s)


def finish_setup(game: Game) -> None:
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        game.step(game.legal_actions()[0])


# --------------------------------------------------------------------------- invariants


def check_invariants(game: Game) -> None:
    s = game.state
    for r in range(NUM_RESOURCES):
        assert s.bank[r] >= 0
        assert all(s.hands[p][r] >= 0 for p in range(NUM_PLAYERS))
        assert s.bank[r] + sum(s.hands[p][r] for p in range(NUM_PLAYERS)) == RESOURCE_CARDS_PER_TYPE
    for d in DevCard:
        held = sum(s.dev_hands[p][d] for p in range(NUM_PLAYERS))
        played = sum(s.dev_played[p][d] for p in range(NUM_PLAYERS))
        assert s.dev_deck[d] >= 0
        assert s.dev_deck[d] + held + played == DEV_CARD_COUNTS[d]
    assert all(s.dev_played[p][DevCard.VICTORY_POINT] == 0 for p in range(NUM_PLAYERS))

    for p in range(NUM_PLAYERS):
        mine = [c for c in range(NUM_CORNERS) if s.corner_owner[c] == p]
        settlements = sum(s.corner_building[c] == Building.SETTLEMENT for c in mine)
        cities = sum(s.corner_building[c] == Building.CITY for c in mine)
        roads = sum(o == p for o in s.edge_owner)
        assert settlements + s.settlements_left[p] == SETTLEMENTS_PER_PLAYER
        assert cities + s.cities_left[p] == CITIES_PER_PLAYER
        assert roads + s.roads_left[p] == ROADS_PER_PLAYER
        assert s.road_lengths[p] == rules.longest_road_length(s, p)
    for c in range(NUM_CORNERS):
        assert (s.corner_owner[c] < 0) == (s.corner_building[c] == Building.NONE)
        if s.corner_owner[c] >= 0:
            assert all(s.corner_owner[n] < 0 for n in CORNER_NEIGHBORS[c]), "distance rule"

    lr = s.longest_road_owner
    if lr >= 0:
        assert s.road_lengths[lr] >= LONGEST_ROAD_MIN
        assert s.road_lengths[lr] == max(s.road_lengths)
    la = s.largest_army_owner
    knights = [s.dev_played[p][DevCard.KNIGHT] for p in range(NUM_PLAYERS)]
    if la >= 0:
        assert knights[la] >= LARGEST_ARMY_MIN and knights[la] == max(knights)
    else:
        assert max(knights) < LARGEST_ARMY_MIN

    assert 0 <= s.robber_tile < NUM_TILES
    if game.is_over:
        assert game.legal_actions() == ()
        if game.winner is not None:
            assert game.victory_points(game.winner) >= s.config.vp_to_win
            assert game.winner == s.current_player
    else:
        assert 0 <= s.decision_player < NUM_PLAYERS
        assert game.legal_actions(), "a live game always offers an action"
        if s.phase == Phase.TRADE_RESPONSE:
            assert 0 <= s.trade_responses < NUM_PLAYERS - 1
            assert s.decision_player == (s.current_player + 1 + s.trade_responses) % NUM_PLAYERS
            assert not s.trade_acceptances[s.current_player]
            assert s.trade_counteroffers[s.current_player] is None
        elif s.phase == Phase.TRADE_CONFIRM:
            assert s.decision_player == s.current_player
            assert s.trade_responses == NUM_PLAYERS - 1
        elif s.phase != Phase.DISCARD:
            assert s.decision_player == s.current_player
        if s.phase not in (Phase.TRADE_RESPONSE, Phase.TRADE_CONFIRM):
            assert s.pending_trade is None
            assert s.trade_responses == 0
            assert not any(s.trade_acceptances)
            assert not any(s.trade_counteroffers)


# --------------------------------------------------------------------------- reference rules


def ref_road_ok(game: Game, p: int, e: int) -> bool:
    s = game.state
    if s.edge_owner[e] >= 0:
        return False
    for c in EDGE_CORNERS[e]:
        if s.corner_owner[c] == p:
            return True
        if s.corner_owner[c] < 0 and any(s.edge_owner[f] == p for f in CORNER_EDGES[c] if f != e):
            return True
    return False


def ref_settlement_ok(game: Game, p: int, c: int) -> bool:
    s = game.state
    return (
        s.corner_owner[c] < 0
        and all(s.corner_owner[n] < 0 for n in CORNER_NEIGHBORS[c])
        and any(s.edge_owner[e] == p for e in CORNER_EDGES[c])
    )


def ref_ratio(game: Game, p: int, r: int) -> int:
    s = game.state
    ratio = 4
    for c in range(NUM_CORNERS):
        if s.corner_owner[c] == p:
            port = s.board.corner_port[c]
            if port == GENERIC_PORT:
                ratio = min(ratio, 3)
            elif port == r:
                ratio = 2
    return ratio


def reference_main_build_actions(game: Game) -> set[tuple]:
    """(type name, location) of every build/robber/trade decision the reference rules allow."""
    s, p = game.state, game.decision_player
    hand, out = s.hands[p], set()
    if s.phase == Phase.MAIN:
        if s.roads_left[p] and hand[0] >= 1 and hand[1] >= 1:
            out |= {("BUILD_ROAD", e) for e in range(NUM_EDGES) if ref_road_ok(game, p, e)}
        if s.settlements_left[p] and min(hand[:4]) >= 1:
            out |= {
                ("BUILD_SETTLEMENT", c) for c in range(NUM_CORNERS) if ref_settlement_ok(game, p, c)
            }
        if s.cities_left[p] and hand[3] >= 2 and hand[4] >= 3:
            out |= {
                ("BUILD_CITY", c)
                for c in range(NUM_CORNERS)
                if s.corner_owner[c] == p and s.corner_building[c] == Building.SETTLEMENT
            }
        for g in range(NUM_RESOURCES):
            for r in range(NUM_RESOURCES):
                if g != r and hand[g] >= ref_ratio(game, p, g) and s.bank[r] > 0:
                    out.add(("MARITIME_TRADE", (g, r)))
    if s.phase == Phase.MOVE_ROBBER:
        for t in range(NUM_TILES):
            if t == s.robber_tile:
                continue
            victims = {
                s.corner_owner[c]
                for c in TILE_CORNERS[t]
                if s.corner_owner[c] not in (-1, p) and sum(s.hands[s.corner_owner[c]]) > 0
            }
            out |= {("MOVE_ROBBER", (t, v)) for v in victims} or {("MOVE_ROBBER", (t, -1))}
    return out


def engine_main_build_actions(game: Game) -> set[tuple]:
    """Same projection as `reference_main_build_actions`, from the engine's legal actions."""
    out = set()
    if game.phase not in (Phase.MAIN, Phase.MOVE_ROBBER):
        return out
    for a in game.legal_actions():
        name = a.type.name
        if name == "BUILD_ROAD":
            out.add((name, a.edge))
        elif name in ("BUILD_SETTLEMENT", "BUILD_CITY"):
            out.add((name, a.corner))
        elif name == "MARITIME_TRADE":
            out.add((name, (a.resource, a.resource2)))
        elif name == "MOVE_ROBBER":
            out.add((name, (a.tile, a.player)))
    return out
