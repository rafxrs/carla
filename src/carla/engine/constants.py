"""Constants of the 4-player base game."""

from enum import IntEnum

NUM_PLAYERS = 4
NUM_RESOURCES = 5


class Resource(IntEnum):
    LUMBER = 0
    BRICK = 1
    WOOL = 2
    GRAIN = 3
    ORE = 4


DESERT = -1
"""Terrain value of the resource-less tile in `Board.tile_resource`."""


class DevCard(IntEnum):
    KNIGHT = 0
    ROAD_BUILDING = 1
    YEAR_OF_PLENTY = 2
    MONOPOLY = 3
    VICTORY_POINT = 4


NUM_DEV_CARD_TYPES = 5


class Building(IntEnum):
    """Value doubles as the number of resources a building collects per production."""

    NONE = 0
    SETTLEMENT = 1
    CITY = 2


GENERIC_PORT = 5
"""Port type of 3:1 ports. Port types 0..4 are 2:1 ports for that `Resource`."""

# Costs are indexed by Resource.
ROAD_COST = (1, 1, 0, 0, 0)
SETTLEMENT_COST = (1, 1, 1, 1, 0)
CITY_COST = (0, 0, 0, 2, 3)
DEV_CARD_COST = (0, 0, 1, 1, 1)

RESOURCE_CARDS_PER_TYPE = 19
DEV_CARD_COUNTS = (14, 2, 2, 2, 5)  # indexed by DevCard
TERRAIN_COUNTS = {
    Resource.LUMBER: 4,
    Resource.BRICK: 3,
    Resource.WOOL: 4,
    Resource.GRAIN: 4,
    Resource.ORE: 3,
    DESERT: 1,
}
NUMBER_TOKENS_BY_LETTER = (5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11)
"""Number on each token, in the order of the letters A to R on their backs (OEIS A054863)."""
NUMBER_TOKENS = tuple(sorted(NUMBER_TOKENS_BY_LETTER))
PORT_TYPES = (GENERIC_PORT,) * 4 + tuple(Resource)

SETTLEMENTS_PER_PLAYER = 5
CITIES_PER_PLAYER = 4
ROADS_PER_PLAYER = 15
LONGEST_ROAD_MIN = 5
LARGEST_ARMY_MIN = 3
AWARD_VP = 2  # value of Longest Road / Largest Army

PIPS = {n: 6 - abs(7 - n) for n in range(2, 13)}
"""Number of the 36 dice outcomes that roll each sum."""

RESOURCE_NAMES = tuple(r.name.lower() for r in Resource)
DEV_CARD_NAMES = tuple(d.name.lower().replace("_", " ") for d in DevCard)
