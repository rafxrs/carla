"""Classic-board coordinate frame exported by the pinned OSP topology mapper.

Tables map CARLA ids to JSettlers coordinates. They are independent of terrain,
ports, and tile numbers; reference checks validate the mapping on every run.
"""

from carla.engine.topology import TILE_CORNERS

NODES = (
    56,
    90,
    124,
    39,
    73,
    107,
    141,
    54,
    88,
    122,
    156,
    37,
    71,
    105,
    139,
    173,
    52,
    86,
    120,
    154,
    188,
    35,
    69,
    103,
    137,
    171,
    205,
    50,
    84,
    118,
    152,
    186,
    220,
    67,
    101,
    135,
    169,
    203,
    82,
    116,
    150,
    184,
    218,
    99,
    133,
    167,
    201,
    114,
    148,
    182,
    216,
    131,
    165,
    199,
)
EDGES = (
    39,
    56,
    73,
    90,
    107,
    124,
    38,
    72,
    106,
    140,
    37,
    54,
    71,
    88,
    105,
    122,
    139,
    156,
    36,
    70,
    104,
    138,
    172,
    35,
    52,
    69,
    86,
    103,
    120,
    137,
    154,
    171,
    188,
    34,
    68,
    102,
    136,
    170,
    204,
    50,
    67,
    84,
    101,
    118,
    135,
    152,
    169,
    186,
    203,
    66,
    100,
    134,
    168,
    202,
    82,
    99,
    116,
    133,
    150,
    167,
    184,
    201,
    98,
    132,
    166,
    200,
    114,
    131,
    148,
    165,
    182,
    199,
)
_NODE_IDS = {coord: i for i, coord in enumerate(NODES)}
# SOCBoard.HEXNODES sums to 51; each classic hex's six corner coordinates
# therefore sum to six times its hex coordinate plus 51.
HEXES = tuple((sum(NODES[c] for c in corners) - 51) // 6 for corners in TILE_CORNERS)
LAND_HEX_ORDER = tuple(sorted(range(19), key=HEXES.__getitem__))


def potential_order(candidates):
    # Classic SOCPlayer HashSet retains its initial capacity of 128. Positive
    # integer keys here have no high bits for HashMap's supplemental xor.
    return sorted(candidates, key=lambda c: (NODES[c] & 127, NODES[c]))


def hashtable_order(candidates, capacity=11):
    """Integer Hashtable enumeration; caller supplies insertion order."""
    buckets = [[] for _ in range(capacity)]
    for count, corner in enumerate(candidates):
        if count >= int(capacity * 0.75):
            old = buckets
            capacity = capacity * 2 + 1
            buckets = [[] for _ in range(capacity)]
            for bucket in reversed(old):
                for previous in bucket:
                    buckets[NODES[previous] % capacity].insert(0, previous)
        buckets[NODES[corner] % capacity].insert(0, corner)
    return [corner for bucket in reversed(buckets) for corner in bucket]


def initial_targets(corner):
    return [
        _NODE_IDS[coord]
        for delta in (2, 34, 32, -2, -34, -32)
        if (coord := NODES[corner] + delta) in _NODE_IDS
    ]


_EDGE_IDS = {coord: e for e, coord in enumerate(EDGES)}


def adjacent_edges(edge):
    """SOCBoard.getAdjacentEdgesToEdge order for a classic edge (land edges only)."""
    coord = EDGES[edge]
    if ((coord & 0x0F) + (coord >> 4)) % 2 == 0:
        deltas = (-0x10, 0x01, 0x10, -0x01)
    elif (coord >> 4) % 2 == 0:
        deltas = (-0x11, 0x01, 0x11, -0x01)
    else:
        deltas = (-0x10, 0x11, 0x10, -0x11)
    return tuple(_EDGE_IDS[c] for d in deltas if (c := coord + d) in _EDGE_IDS)
