"""Rule configuration of a game."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EngineConfig:
    """Rule knobs. Defaults are the official 4-player base game, without domestic trade (the
    trade limits are experiment controls, not rules from the rulebook)."""

    vp_to_win: int = 10
    discard_limit: int = 7
    """On a 7, players holding more than this many resource cards discard half (rounded down)."""
    max_turns: int | None = None
    """End the game without a winner after this many player turns (None: play to the end)."""
    allow_domestic_trade: bool = False
    """Player-to-player trading in the main phase (protocol in `rules`)."""
    max_trade_proposals_per_turn: int | None = 3
    """Negotiations (offers and invitations) the current player may open per turn (None: no limit)."""
    max_domestic_trades_per_turn: int | None = 3
    """Domestic trades the current player may complete per turn (None: no limit)."""
    max_trade_cards_per_side: int | None = 4
    """Cards an offer or counteroffer may give, and may ask for (None: no limit; at least 1)."""
    board_seed: int | None = None
    """Use the same board in every game (generated from this seed) instead of one per game seed."""
    forbid_adjacent_red_numbers: bool = True
    """Lay number tokens as in the official variable set-up (no neighbouring 6/8, equal numbers or
    2/12); False shuffles them freely."""

    def __post_init__(self) -> None:
        for name in _TRADE_LIMITS:
            value = getattr(self, name)
            low = 1 if name == "max_trade_cards_per_side" else 0  # an offer has a card a side
            if value is not None and (type(value) is not int or value < low):
                raise ValueError(f"{name} must be None or an integer >= {low}, not {value!r}")

    def to_dict(self) -> dict[str, Any]:
        """All fields, except the limits added with domestic trade while it is off: a game
        without it serializes (and hashes) exactly as before, so older records still replay."""
        out = asdict(self)
        if not self.allow_domestic_trade:
            for name in _TRADE_LIMITS[1:]:
                del out[name]
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineConfig":
        return cls(**data)


_TRADE_LIMITS = (
    "max_trade_proposals_per_turn",
    "max_domestic_trades_per_turn",
    "max_trade_cards_per_side",
)
"""The first one predates domestic trade (it was reserved for it), the others came with it."""
