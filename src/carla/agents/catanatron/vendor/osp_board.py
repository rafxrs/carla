"""
Catanatron's board in the Open Seat Protocol's keys (see ../PROTOCOL.md): hexes in axial coordinates [q, r], an
intersection as its three hexes, a path as its two intersections, a harbor on a path.

Catanatron places tiles at cube coordinates (x, y, z) with east = (1, -1, 0) and south-east = (0, -1, 1)
(catanatron/models/coordinate_system.py); the protocol's q grows to the east and r to the south-east, so q = x and
r = z. Every tile, water and harbors included, carries its own node ids, so an intersection's three hexes are the tiles
that share its node id. Nothing here is copied from another engine: the keys follow from Catanatron's map alone.
"""

from collections import defaultdict

from catanatron.models.enums import BRICK, ORE, SHEEP, WHEAT, WOOD
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS
from catanatron.models.tiles import LandTile, Port

# SGN's tile codes by Catanatron's resource (None is the desert)
TILE_CODES = {WOOD: "FO", BRICK: "H", SHEEP: "P", WHEAT: "F", ORE: "M", None: "D"}
# SGN's harbor codes by the resource of a 2:1 harbor (None is a 3:1 harbor)
HARBOR_CODES = {None: "HR", WOOD: "HRL", BRICK: "HRB", SHEEP: "HRW", WHEAT: "HRG", ORE: "HRO"}


def axial(coordinate):
    """The protocol's hex [q, r] of a Catanatron cube coordinate (x, y, z)"""
    x, _, z = coordinate
    return (x, z)


def node_keys(catan_map):
    """Every land node's key: the sorted three hexes (land, water or harbor) that share its node id"""
    hexes_at = defaultdict(set)
    for coordinate, tile in catan_map.tiles.items():
        for node_id in tile.nodes.values():
            hexes_at[node_id].add(axial(coordinate))
    return {node_id: tuple(sorted(hexes_at[node_id])) for node_id in catan_map.land_nodes}


def path_key(keys, edge):
    """A Catanatron edge (two node ids) as the protocol's path: its two intersections, sorted"""
    a, b = edge
    return tuple(sorted((keys[a], keys[b])))


def path_keys(catan_map, keys):
    """The paths of every edge of a land tile"""
    return {path_key(keys, edge) for tile in catan_map.land_tiles.values() for edge in tile.edges.values()}


def harbors(catan_map, keys):
    """Each harbor's path and SGN code: the two nodes on the side of the harbor tile that its direction names"""
    result = []
    for tile in catan_map.tiles.values():
        if isinstance(tile, Port):
            a, b = PORT_DIRECTION_TO_NODEREFS[tile.direction]
            result.append((path_key(keys, (tile.nodes[a], tile.nodes[b])), HARBOR_CODES[tile.resource]))
    return sorted(result)


def shape(catan_map):
    """The board's shape as the protocol's reference file holds it (../osp/classic-board.json), every list sorted"""
    keys = node_keys(catan_map)
    land = sorted(axial(c) for c, tile in catan_map.tiles.items() if isinstance(tile, LandTile))
    sea = sorted(axial(c) for c, tile in catan_map.tiles.items() if not isinstance(tile, LandTile))
    return {
        "land": land,
        "sea": sea,
        "intersections": sorted(keys.values()),
        "paths": sorted(path_keys(catan_map, keys)),
        "harbors": sorted(path for path, _ in harbors(catan_map, keys)),
    }


def cube(hex_):
    """Catanatron's cube coordinate (x, y, z) of the protocol's hex [q, r]"""
    q, r = hex_
    return (q, -q - r, r)


def catan_map(osp_board):
    """
    A Catanatron map of the protocol's board (a classic board: Catanatron's base topology, whose harbors sit where the
    protocol's do). Catanatron lays its tiles out in the order of the base template, drawing each tile's resource, number
    and harbor from the end of lists, so the lists are given in that order, reversed.
    """
    from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap, initialize_tiles

    tiles = {cube(t["tile"]): t for t in osp_board["tiles"]}
    resources_by_code = {code: resource for resource, code in TILE_CODES.items()}
    harbors_by_code = {code: resource for resource, code in HARBOR_CODES.items()}
    # The harbor paths of the base template's harbor tiles, the same on every base map
    template = CatanMap.from_template(BASE_MAP_TEMPLATE)
    template_keys = node_keys(template)
    harbor_at = {}
    for coordinate, tile in template.tiles.items():
        if isinstance(tile, Port):
            a, b = PORT_DIRECTION_TO_NODEREFS[tile.direction]
            harbor_at[path_key(template_keys, (tile.nodes[a], tile.nodes[b]))] = coordinate
    codes_at = {}
    for harbor in osp_board["harbors"]:
        path = tuple(sorted(tuple(tuple(h) for h in end) for end in harbor["path"]))
        if path not in harbor_at:
            continue  # CARLA uses different coastal port slots; install exact node cache below.
        codes_at[harbor_at[path]] = harbor["hCode"]

    resources, numbers, ports = [], [], []
    for coordinate, kind in BASE_MAP_TEMPLATE.topology.items():
        if isinstance(kind, tuple):
            ports.append(harbors_by_code[codes_at.get(coordinate, "HR")])
        elif kind == LandTile:
            tile = tiles[coordinate]
            resources.append(resources_by_code[tile["tCode"]])
            if tile["rNumber"] is not None:
                numbers.append(tile["rNumber"])
    built = initialize_tiles(BASE_MAP_TEMPLATE, numbers[::-1], ports[::-1], resources[::-1], number_placement="random")
    result = CatanMap.from_tiles(built)
    inverse = {key: node for node, key in node_keys(result).items()}
    result.port_nodes = defaultdict(set)
    for harbor in osp_board["harbors"]:
        resource = harbors_by_code[harbor["hCode"]]
        for end in harbor["path"]:
            result.port_nodes[resource].add(inverse[tuple(sorted(tuple(h) for h in end))])
    return result


def board(catan_map, robber_coordinate):
    """The protocol's board: every hex with its SGN tile code and number, the harbors, the robber"""
    keys = node_keys(catan_map)
    tiles = []
    for coordinate, tile in sorted(catan_map.tiles.items(), key=lambda item: axial(item[0])):
        if isinstance(tile, LandTile):
            tiles.append({"tile": list(axial(coordinate)), "tCode": TILE_CODES[tile.resource], "rNumber": tile.number})
        else:
            tiles.append({"tile": list(axial(coordinate)), "tCode": "S", "rNumber": None})
    return {
        "tiles": tiles,
        "harbors": [{"path": [[list(h) for h in end] for end in path], "hCode": code} for path, code in harbors(catan_map, keys)],
        "robber": list(axial(robber_coordinate)),
    }
