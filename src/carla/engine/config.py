"""Rule configuration of a game."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EngineConfig:
    """Rule knobs. Domestic trading is opt-in; its finite defaults bound experiments."""

    vp_to_win: int = 10
    discard_limit: int = 7
    """On a 7, players holding more than this many resource cards discard half (rounded down)."""
    max_turns: int | None = None
    """End the game without a winner after this many player turns (None: play to the end)."""
    allow_domestic_trade: bool = False
    """Enable player-to-player trading during the main phase."""
    max_trade_proposals_per_turn: int | None = 3
    """Maximum negotiations initiated in one turn. None removes the limit; 0 disables them."""
    max_domestic_trades_per_turn: int | None = 3
    """Maximum completed player-to-player trades in one turn. None removes the limit."""
    max_trade_cards_per_side: int = 4
    """Maximum cards given or requested in one offer. Set to 0 for no limit."""
    board_seed: int | None = None
    """Use the same board in every game (generated from this seed) instead of one per game seed."""
    forbid_adjacent_red_numbers: bool = True
    """Lay number tokens as in the official variable set-up (no neighbouring 6/8, equal numbers or
    2/12); False shuffles them freely."""

    def __post_init__(self) -> None:
        for name in (
            "max_trade_proposals_per_turn",
            "max_domestic_trades_per_turn",
            "max_trade_cards_per_side",
        ):
            value = getattr(self, name)
            if value is None and name != "max_trade_cards_per_side":
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer or None where allowed")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineConfig":
        return cls(**data)
