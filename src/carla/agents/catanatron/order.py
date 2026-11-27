"""Catanatron's playable-action order for CARLA actions, without importing Catanatron.

ValueFunction and AlphaBeta keep the first best action in the order Catanatron generates its
playable actions (`catanatron.models.actions.generate_playable_actions`), and exact ties are
common: about 15 per ValueFunction game, mostly discards, roads, bank trades, robber moves and
"play a card or end the turn". Under CARLA's referee the wrapper rebuilds a Catanatron game
for every decision (`vendor/rebuild.py`), so its order is the one of that rebuilt game. Breaking
ties in CARLA's order instead made the upstream port measurably weaker than the original.

This module reproduces that order for the upstream-mode ports:

- Exact: dev cards before rolling or ending the turn, settlements by node id, cities in the
  rebuild's settlement order, robber moves by land tile then victim, discards, monopoly.
- Python set iteration, rebuilt from the same values: Year of Plenty, bank trades, robber
  victims (a `Color` hashes as its name). With `PYTHONHASHSEED` fixed this matches the
  wrapper; bank-trade tuples hold `None`, whose hash is an address before Python 3.12, so
  their order also matches only within one process, as upstream's own does.
- Approximate: roads. Catanatron lists `set` of edges grown from its connected components; the
  components are emulated from the public board, so the order differs only when set slots
  collide in a different insertion order.

The tables are Catanatron's fixed base-map numbering in CARLA's frame (`coordinates.py`); a
test regenerates them from Catanatron when it is installed.
"""

from carla.engine.actions import ActionType
from carla.engine.state import Phase
from carla.engine.topology import EDGE_CORNERS

NODES = (
    45, 48, 50, 47, 46, 49, 51, 43, 19, 22, 52, 44, 21, 20, 23, 53, 40, 16, 0, 6, 24, 42, 18, 5,
    1, 7, 25, 41, 17, 4, 2, 8, 26, 39, 15, 3, 9, 27, 38, 14, 12, 10, 28, 37, 13, 11, 29, 36, 34,
    32, 30, 35, 33, 31,
)  # fmt: skip
"""Catanatron node id of each CARLA corner."""
LAND_ORDER = (9, 10, 14, 13, 8, 4, 5, 11, 15, 18, 17, 16, 12, 7, 3, 0, 1, 2, 6)
"""CARLA tiles in the iteration order of Catanatron's `map.land_tiles`."""
TILE_NODES = (
    (45, 46, 19, 21, 43, 47), (48, 49, 22, 20, 19, 46), (50, 51, 52, 23, 22, 49),
    (43, 21, 16, 18, 40, 44), (19, 20, 0, 5, 16, 21), (22, 23, 6, 1, 0, 20),
    (52, 53, 24, 7, 6, 23), (40, 18, 17, 39, 41, 42), (16, 5, 4, 15, 17, 18),
    (0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 2, 1), (24, 25, 26, 27, 8, 7), (17, 15, 14, 37, 38, 39),
    (4, 3, 12, 13, 14, 15), (2, 9, 10, 11, 12, 3), (8, 27, 28, 29, 10, 9),
    (14, 13, 34, 35, 36, 37), (12, 11, 32, 33, 34, 13), (10, 29, 30, 31, 32, 11),
)  # fmt: skip
"""Node ids of each CARLA tile in the order of Catanatron's `tile.nodes`."""
ADJACENCY = (
    (5, 1, 20), (2, 0, 6), (1, 3, 9), (2, 4, 12), (3, 5, 15), (4, 0, 16), (1, 7, 23), (8, 6, 24),
    (7, 9, 27), (8, 2, 10), (9, 11, 29), (10, 12, 32), (11, 3, 13), (12, 14, 34), (13, 15, 37),
    (14, 4, 17), (18, 5, 21), (15, 18, 39), (17, 16, 40), (21, 20, 46), (0, 19, 22),
    (16, 19, 43), (20, 23, 49), (6, 22, 52), (7, 25, 53), (26, 24), (25, 27), (26, 8, 28),
    (27, 29), (28, 10, 30), (29, 31), (30, 32), (31, 11, 33), (32, 34), (33, 13, 35), (34, 36),
    (35, 37), (36, 14, 38), (37, 39), (38, 17, 41), (42, 18, 44), (39, 42), (41, 40),
    (44, 21, 47), (40, 43), (47, 46), (19, 45, 48), (43, 45), (46, 49), (22, 48, 50),
    (49, 51), (52, 50), (51, 23, 53), (24, 52),
)  # fmt: skip
"""Neighbours of each Catanatron node in the order of its buildable graph's adjacency."""
COLORS = ("RED", "BLUE", "ORANGE", "WHITE")
"""Catanatron's colours by seat; an enum member hashes as its name."""
RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
"""Catanatron's resource names in CARLA's resource order."""

CORNER_OF = {node: corner for corner, node in enumerate(NODES)}
EDGE_OF = {tuple(sorted((NODES[a], NODES[b]))): edge for edge, (a, b) in enumerate(EDGE_CORNERS)}
T = ActionType


def _buildable_edges(s, seat):
    """`Board.buildable_edges`: uncoloured edges around the seat's road network, in set order."""
    owner = s.corner_owner
    road = {}
    for edge, (a, b) in enumerate(EDGE_CORNERS):
        if s.edge_owner[edge] >= 0:
            road[tuple(sorted((NODES[a], NODES[b])))] = s.edge_owner[edge]
    nodes = {NODES[c] for c in range(len(owner)) if owner[c] == seat}
    for (a, b), p in road.items():
        if p == seat:
            for n in (a, b):
                if owner[CORNER_OF[n]] in (-1, seat):
                    nodes.add(n)
    expandable_nodes = set().union(nodes)
    expandable, seen = set(), set()
    for n in expandable_nodes:  # networkx Graph.edges(nbunch)
        for m in ADJACENCY[n]:
            if m not in seen and tuple(sorted((n, m))) not in road:
                expandable.add(tuple(sorted((n, m))))
        seen.add(n)
    return [EDGE_OF[e] for e in expandable]


def _year_of_plenty(bank):
    options = set()
    for i in range(5):
        for j in range(i, 5):
            draw = [0] * 5
            draw[i] += 1
            draw[j] += 1
            if all(bank[r] >= draw[r] for r in range(5)):
                options.add((RESOURCES[i], RESOURCES[j]))
            else:
                if bank[i] >= 1:
                    options.add((RESOURCES[i],))
                if bank[j] >= 1:
                    options.add((RESOURCES[j],))
    return [tuple(RESOURCES.index(r) for r in o) for o in options]


def _maritime(hand, bank, ratios):
    offers = set()
    for i, resource in enumerate(RESOURCES):
        if hand[i] >= ratios[i]:
            out = [resource] * ratios[i] + [None] * (4 - ratios[i])
            for j, other in enumerate(RESOURCES):
                if j != i and bank[j] > 0:
                    offers.add((*out, other))
    return [(RESOURCES.index(o[0]), RESOURCES.index(o[4])) for o in offers]


def _robber(s, seat):
    out = []
    for tile in LAND_ORDER:
        if tile == s.robber_tile:
            continue
        victims = set()
        for node in TILE_NODES[tile]:
            p = s.corner_owner[CORNER_OF[node]]
            if p >= 0 and p != seat and sum(s.hands[p]) >= 1:
                victims.add(COLORS[p])
        if victims:
            out.extend((tile, COLORS.index(c)) for c in victims)
        else:
            out.append((tile, -1))
    return out


def _settlement_nodes(s, seat):
    """`buildings_by_color[SETTLEMENT]` after `rebuild`: those with an own road first."""
    roads = set()
    for edge, (a, b) in enumerate(EDGE_CORNERS):
        if s.edge_owner[edge] == seat:
            roads.update((a, b))
    corners = [
        c for c in range(len(s.corner_owner)) if s.corner_owner[c] == seat and s.corner_building[c]
    ]
    ordered = sorted(corners, key=lambda c: c not in roads)
    return [c for c in ordered if s.corner_building[c] == 1]


def _keys(s, seat):
    """Catanatron's playable actions as CARLA action keys, in generation order."""
    phase = s.phase
    if phase == Phase.SETUP_SETTLEMENT:
        return [(T.BUILD_SETTLEMENT, c) for c in sorted(range(54), key=NODES.__getitem__)]
    if phase in (Phase.SETUP_ROAD, Phase.ROAD_BUILDING):
        return [(T.BUILD_ROAD, e) for e in _buildable_edges(s, seat)]
    if phase == Phase.MOVE_ROBBER:
        return [(T.MOVE_ROBBER, t, v) for t, v in _robber(s, seat)]
    if phase == Phase.DISCARD:
        return [(T.DISCARD, r) for r in range(5)]
    from carla.engine.rules import trade_ratios

    keys = [(T.PLAY_DEV_CARD, 2, *pick) for pick in _year_of_plenty(s.bank)]
    keys += [(T.PLAY_DEV_CARD, 3, r) for r in range(5)]
    keys += [(T.PLAY_DEV_CARD, 0), (T.PLAY_DEV_CARD, 1)]
    if phase == Phase.PRE_ROLL:
        return [*keys, (T.ROLL_DICE,)]
    keys.append((T.END_TURN,))
    keys += [(T.BUILD_ROAD, e) for e in _buildable_edges(s, seat)]
    keys += [(T.BUILD_SETTLEMENT, c) for c in sorted(range(54), key=NODES.__getitem__)]
    keys += [(T.BUILD_CITY, c) for c in _settlement_nodes(s, seat)]
    keys.append((T.BUY_DEV_CARD,))
    keys += [
        (T.MARITIME_TRADE, g, r) for g, r in _maritime(s.hands[seat], s.bank, trade_ratios(s, seat))
    ]
    return keys


def key(action):
    t = action.type
    if t in (T.BUILD_SETTLEMENT, T.BUILD_CITY):
        return (t, action.corner)
    if t == T.BUILD_ROAD:
        return (t, action.edge)
    if t == T.MOVE_ROBBER:
        return (t, action.tile, action.player)
    if t == T.DISCARD:
        return (t, action.resource)
    if t == T.MARITIME_TRADE:
        return (t, action.resource, action.resource2)
    if t == T.PLAY_DEV_CARD:
        if action.dev == 2:
            pick = tuple(sorted((action.resource, action.resource2)))
            return (t, 2, *pick)
        if action.dev == 3:
            return (t, 3, action.resource)
        return (t, action.dev)
    return (t,)


def catanatron_order(state, seat, actions):
    """`actions` sorted into Catanatron's generation order; unknown ones keep theirs, last."""
    rank = {}
    for i, k in enumerate(_keys(state, seat)):
        rank.setdefault(k, i)
    last = len(rank)
    return sorted(actions, key=lambda a: rank.get(key(a), last))
