"""Per-game board layout: terrain, number tokens and port types on the static topology."""

from dataclasses import dataclass, field
from typing import Any

from carla.engine.constants import (
    DESERT,
    GENERIC_PORT,
    NUMBER_TOKENS,
    PORT_TYPES,
    RESOURCE_NAMES,
    TERRAIN_COUNTS,
)
from carla.engine.rng import Rng
from carla.engine.topology import (
    CORNER_TILES,
    EDGE_CORNERS,
    NUM_CORNERS,
    NUM_PORTS,
    NUM_TILES,
    PORT_CORNERS,
    TILE_NEIGHBORS,
)

_RED_NUMBERS = (6, 8)


@dataclass(frozen=True)
class Board:
    """Immutable layout. `tile_resource[t]` is a Resource or DESERT; `tile_number[t]` is 0 on the
    desert; `port_types[i]` is the Resource of a 2:1 port or GENERIC_PORT, at slot PORT_EDGES[i].
    """

    tile_resource: tuple[int, ...]
    tile_number: tuple[int, ...]
    port_types: tuple[int, ...]
    corner_port: tuple[int, ...] = field(init=False, repr=False, compare=False)
    """Port type reachable from each corner, or -1."""
    tiles_by_number: tuple[tuple[int, ...], ...] = field(init=False, repr=False, compare=False)
    """Tiles carrying each dice sum (index 0..12)."""
    desert_tile: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.tile_resource) != NUM_TILES or len(self.tile_number) != NUM_TILES:
            raise ValueError("a board has 19 tiles")
        if len(self.port_types) != NUM_PORTS:
            raise ValueError("a board has 9 port slots")
        for r, n in zip(self.tile_resource, self.tile_number, strict=True):
            if (r == DESERT) != (n == 0) or n not in (0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12):
                raise ValueError(f"invalid tile (resource={r}, number={n})")
        if not all(0 <= p <= GENERIC_PORT for p in self.port_types):
            raise ValueError("invalid port type")
        if DESERT not in self.tile_resource:
            raise ValueError("the robber starts on a desert; the board needs one")

        corner_port = [-1] * NUM_CORNERS
        for kind, corners in zip(self.port_types, PORT_CORNERS, strict=True):
            for c in corners:
                corner_port[c] = kind
        by_number: list[list[int]] = [[] for _ in range(13)]
        for t, n in enumerate(self.tile_number):
            by_number[n].append(t)
        object.__setattr__(self, "corner_port", tuple(corner_port))
        object.__setattr__(self, "tiles_by_number", tuple(tuple(ts) for ts in by_number))
        object.__setattr__(self, "desert_tile", self.tile_resource.index(DESERT))

    @classmethod
    def random(cls, rng: Rng, forbid_adjacent_red: bool = True) -> "Board":
        """Shuffle terrain, ports and number tokens (optionally keeping 6s and 8s apart)."""
        terrain = [r for r, count in TERRAIN_COUNTS.items() for _ in range(count)]
        rng.shuffle(terrain)
        ports = list(PORT_TYPES)
        rng.shuffle(ports)
        tokens = list(NUMBER_TOKENS)
        while True:
            rng.shuffle(tokens)
            it = iter(tokens)
            numbers = tuple(0 if r == DESERT else next(it) for r in terrain)
            if not forbid_adjacent_red or not _adjacent_red(numbers):
                return cls(tuple(terrain), numbers, tuple(ports))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_resource": list(self.tile_resource),
            "tile_number": list(self.tile_number),
            "port_types": list(self.port_types),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Board":
        return cls(
            tuple(data["tile_resource"]), tuple(data["tile_number"]), tuple(data["port_types"])
        )

    def describe_tile(self, t: int) -> str:
        r = self.tile_resource[t]
        return "desert" if r == DESERT else f"{RESOURCE_NAMES[r]} {self.tile_number[t]}"

    def describe_corner(self, c: int) -> str:
        tiles = ", ".join(self.describe_tile(t) for t in CORNER_TILES[c])
        port = self.corner_port[c]
        port_text = "" if port < 0 else f"; {describe_port(port)}"
        return f"corner {c} [{tiles}{port_text}]"

    def describe_edge(self, e: int) -> str:
        a, b = EDGE_CORNERS[e]
        return f"edge {e} [{a}-{b}]"


def describe_port(kind: int) -> str:
    return "3:1 port" if kind == GENERIC_PORT else f"2:1 {RESOURCE_NAMES[kind]} port"


def _adjacent_red(numbers: tuple[int, ...]) -> bool:
    return any(
        numbers[t] in _RED_NUMBERS and numbers[u] in _RED_NUMBERS
        for t in range(NUM_TILES)
        for u in TILE_NEIGHBORS[t]
    )
