"""Topology-derived bijections, with ports mapped explicitly by the state codec."""

from functools import lru_cache

from carla.engine.topology import _LATTICE, EDGE_CORNERS, TILE_AXIAL


def corner_keys():
    offsets = ((0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1))
    return tuple(
        tuple(
            sorted(
                (q, r)
                for q in range(-4, 5)
                for r in range(-4, 5)
                if (x - (2 * q + r), y - 3 * r) in offsets
            )
        )
        for x, y in _LATTICE
    )


def transform(point, rotation, reflected):
    q, r = point
    if reflected:
        q, r = r, q
    for _ in range(rotation):
        q, r = -r, q + r
    return q, r


@lru_cache(maxsize=1)
def mappings():
    from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap

    from carla.agents.catanatron.vendor.osp_board import node_keys

    template = CatanMap.from_template(BASE_MAP_TEMPLATE)
    native_keys = node_keys(template)
    inverse = {key: node for node, key in native_keys.items()}
    for reflection in (False, True):
        for rotation in range(6):
            keys = [
                tuple(sorted(transform(p, rotation, reflection) for p in key))
                for key in corner_keys()
            ]
            if set(keys) == set(inverse):
                nodes = tuple(inverse[k] for k in keys)
                tiles = tuple(transform(p, rotation, reflection) for p in TILE_AXIAL)
                edges = tuple(tuple(sorted((nodes[a], nodes[b]))) for a, b in EDGE_CORNERS)
                return keys, nodes, tiles, edges
    raise ValueError("CARLA/Catanatron classic board or port topology is incompatible")
