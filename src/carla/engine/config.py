"""Rule configuration of a game."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EngineConfig:
    """Rule knobs. Defaults are the official 4-player base game."""

    vp_to_win: int = 10
    discard_limit: int = 7
    """On a 7, players holding more than this many resource cards discard half (rounded down)."""
    max_turns: int | None = None
    """End the game without a winner after this many player turns (None: play to the end)."""
    allow_domestic_trade: bool = False
    """Player-to-player trading. Not implemented yet: True raises NotImplementedError."""
    max_trade_proposals_per_turn: int = 3
    """Domestic trade (reserved): proposals the current player may make per turn."""
    board_seed: int | None = None
    """Use the same board in every game (generated from this seed) instead of one per game seed."""
    forbid_adjacent_red_numbers: bool = True
    """Re-deal number tokens until no two 6/8 tokens sit on neighbouring tiles."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineConfig":
        return cls(**data)
