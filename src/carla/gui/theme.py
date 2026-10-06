"""The GUI's look in one place: a flat, muted palette, display labels and base sizes.

Player colours come from the Okabe-Ito palette, which stays distinguishable under the common
colour-vision deficiencies; seats are also labelled P0-P3 wherever their colour appears.
"""

from carla.engine.constants import DESERT, Resource
from carla.engine.state import Phase

Color = tuple[int, int, int, int]


def rgb(value: int, alpha: int = 255) -> Color:
    """Colour from 0xRRGGBB."""
    return (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF, alpha


def with_alpha(color: Color, alpha: float) -> Color:
    return color[0], color[1], color[2], max(0, min(255, round(alpha)))


def mix(a: Color, b: Color, t: float) -> Color:
    """Blend from `a` (t=0) to `b` (t=1)."""
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
        round(a[3] + (b[3] - a[3]) * t),
    )


BLACK = rgb(0x000000)
WHITE = rgb(0xFFFFFF)

# ------------------------------------------------------------------ board
SEA = rgb(0x2B5468)
SHALLOWS = rgb(0x376B80)
BEACH = rgb(0xE0CFA0)
TERRAIN: dict[int, Color] = {
    Resource.LUMBER: rgb(0x55794A),
    Resource.BRICK: rgb(0xC0714F),
    Resource.WOOL: rgb(0x9DC07A),
    Resource.GRAIN: rgb(0xE3C062),
    Resource.ORE: rgb(0x8C929E),
    DESERT: rgb(0xDDCB9B),
}
"""Fill of each terrain, keyed by `Board.tile_resource` (a Resource or DESERT)."""
TERRAIN_DETAIL: dict[int, Color] = {
    Resource.LUMBER: rgb(0x35583A),
    Resource.BRICK: rgb(0xA5583C),
    Resource.WOOL: rgb(0xF4F1E6),
    Resource.GRAIN: rgb(0xC49A3C),
    Resource.ORE: rgb(0x656C79),
    DESERT: rgb(0xC9B27C),
}
"""Colour of the small procedural motifs (trees, mounds, sheep, furrows, peaks, dunes)."""
LIGHT_TERRAIN = frozenset({Resource.WOOL, Resource.GRAIN, DESERT})
"""Terrains light enough that dark ink reads better than white on them."""
TRUNK = rgb(0x6B4F35)
SNOW = rgb(0xEEF1F4)
SHEEP_HEAD = rgb(0x4A4A48)

TOKEN = rgb(0xF5EDD6)
TOKEN_EDGE = rgb(0x9C8A68)
INK = rgb(0x2E2A24)
HOT = rgb(0xC0392B)
"""Numbers 6 and 8, the most frequent rolls."""
PORT = rgb(0xF5EDD6)
PORT_EDGE = rgb(0x6E5A40)
PIER = rgb(0xB89B6E)
ROBBER = rgb(0x34343B)
ROBBER_EDGE = rgb(0x141418)
ROBBER_SHINE = rgb(0x5E5E6A)
PIECE_EDGE = rgb(0x1C1C1C)
TARGET = rgb(0xFFFFFF)
"""Pulsing highlight of legal targets."""

PLAYER: tuple[Color, Color, Color, Color] = (
    rgb(0xD55E00),  # vermillion
    rgb(0x0072B2),  # blue
    rgb(0xCC79A7),  # reddish purple
    rgb(0xF0E442),  # yellow
)

# ------------------------------------------------------------------ interface
BACKGROUND = rgb(0x1B2128)
PANEL = rgb(0x232B33)
PANEL_RAISED = rgb(0x2D3741)
PANEL_EDGE = rgb(0x3A4652)
TEXT = rgb(0xE9EBEE)
TEXT_MUTED = rgb(0x9AA5B1)
TEXT_DIM = rgb(0x66717D)
ACCENT = rgb(0xF2C057)
BUTTON = rgb(0x34404C)
BUTTON_HOVER = rgb(0x46566A)
BUTTON_DISABLED = rgb(0x29323B)
BUTTON_TEXT_DISABLED = rgb(0x5E6873)
SHADE = rgb(0x0B0F13, 170)
CARD_BACK = rgb(0x5B6B7C)
DICE = rgb(0xF7F3EA)
DICE_PIP = rgb(0x2B2B2B)


def ink_on(terrain: int) -> Color:
    """Text colour for labels drawn on a terrain-coloured surface."""
    return INK if terrain in LIGHT_TERRAIN else WHITE


# ------------------------------------------------------------------ labels
RESOURCE_LABELS = ("Lumber", "Brick", "Wool", "Grain", "Ore")
RESOURCE_INITIALS = ("L", "B", "W", "G", "O")
DEV_LABELS = ("Knight", "Road Building", "Year of Plenty", "Monopoly", "Victory Point")
PHASE_LABELS = {
    Phase.SETUP_SETTLEMENT: "Setup: settlement",
    Phase.SETUP_ROAD: "Setup: road",
    Phase.PRE_ROLL: "Roll",
    Phase.DISCARD: "Discard",
    Phase.MOVE_ROBBER: "Move the robber",
    Phase.MAIN: "Build & trade",
    Phase.ROAD_BUILDING: "Road Building",
    Phase.TRADE_RESPONSE: "Trade offer",
    Phase.TRADE_CONFIRM: "Trade confirmation",
    Phase.GAME_OVER: "Game over",
}

# ------------------------------------------------------------------ sizes (logical pixels, points)
WINDOW_SIZE = (1280, 800)
MIN_WINDOW_SIZE = (1000, 660)
SIDE_WIDTH = (330, 420)
"""Minimum and maximum width of the right-hand column (side panel and log)."""
SIDE_HEIGHT = 520
"""Height the side panel would like; the log gets the rest of the column."""
LOG_MIN_HEIGHT = 150
STATUS_HEIGHT = 40
PLAYBACK_HEIGHT = 40
ACTION_HEIGHT = 62
PAD = 10
FONT = 11
FONT_SMALL = 9
FONT_LARGE = 14
FONT_TITLE = 18
CHAR_WIDTH = 0.56
"""Average glyph width in ems of the default font, used to wrap text without measuring it."""
