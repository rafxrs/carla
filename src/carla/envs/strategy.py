"""Strategy statistic z, its library, and the per-seat strategy controller.

z is the Catan analogue of AlphaStar's strategy statistic: an opening build order (the first
`BUILD_ORDER_LENGTH` purchases after setup, each a road, settlement, city or development card)
plus cumulative build counts (settlements, cities, roads and development cards bought, knights
played, longest road and largest army held at the end). Every entry is public: builds, purchases
and plays are public events, so z statistics can be computed for any seat without hidden cards.

Teacher games give hindsight z's (what the teacher actually did), grouped by archetype in a
`StrategyLibrary`. A `StrategyController` holds one seat's z for an episode: prescribed from
the library, chosen by the policy's selector head at reconsideration points, or absent.

The strategy pseudo-reward is a separate objective (with its own critic), not potential-based
shaping: per learner decision it pays decay(turn) * -(D(s') - D(s)), the decrease of the
distance D between the seat's statistics so far and z. D adds the normalized edit distance of
the build-order prefix and the L1 distance of the normalized counts. The reward stops after
the last `StrategyConfig.decay_turns` entry, so late-game play optimizes only the outcome.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from carla.config.schema import StrategyConfig
from carla.engine.constants import DevCard
from carla.engine.events import Event, EventKind
from carla.engine.state import Phase

PURCHASE_KINDS = ("road", "settlement", "city", "dev")
NUM_PURCHASE_KINDS = len(PURCHASE_KINDS)
BUILD_ORDER_LENGTH = 10
STAT_NAMES = (
    "settlements",
    "cities",
    "roads",
    "dev_bought",
    "knights",
    "longest_road",
    "largest_army",
)
STAT_SCALE = np.asarray([5.0, 4.0, 15.0, 10.0, 7.0, 1.0, 1.0], dtype=np.float32)
"""Normalizers of the counts: settlements and cities built after setup, roads built after setup
(free roads included), cards bought, knights played; the awards are 0/1."""
NUM_STATS = len(STAT_NAMES)
Z_BUILD = BUILD_ORDER_LENGTH * NUM_PURCHASE_KINDS
Z_PRESENT = Z_BUILD + NUM_STATS
Z_REWARD = Z_PRESENT + 1
Z_DIM = Z_REWARD + 1
"""Layout: build order one-hots (10 x 4), normalized counts (7), present flag, reward flag."""

STRATEGY_KEY = "strategy"
"""Observation entry with the seat's current z (zeros when absent)."""
SELECT_KEY = "strategy_select"
"""Observation entry: 1 when this decision is a selection point of the selector head."""
CHOICE_KEY = "strategy_choice"
"""Observation entry: the selected archetype (num_strategies = none), -1 if no selection."""

_PURCHASE_EVENTS = {
    EventKind.ROAD_BUILT: 0,
    EventKind.SETTLEMENT_BUILT: 1,
    EventKind.CITY_BUILT: 2,
    EventKind.DEV_CARD_BOUGHT: 3,
}


@dataclass
class StrategyStats:
    """Public build statistics of one seat (after setup)."""

    build_order: list[int] = field(default_factory=list)
    counts: np.ndarray = field(default_factory=lambda: np.zeros(NUM_STATS, dtype=np.float32))

    def update(self, events: list[Event], seat: int, setup: bool) -> bool:
        """Fold one engine step's events in; True if the statistics changed."""
        changed = False
        for e in events:
            if e.player != seat and e.kind not in (
                EventKind.LONGEST_ROAD_CHANGED,
                EventKind.LARGEST_ARMY_CHANGED,
            ):
                continue
            kind = _PURCHASE_EVENTS.get(e.kind)
            if kind is not None:
                if setup:
                    continue
                if len(self.build_order) < BUILD_ORDER_LENGTH:
                    self.build_order.append(kind)
                if kind == 0:
                    self.counts[2] += 1
                elif kind == 1:
                    self.counts[0] += 1
                elif kind == 2:
                    self.counts[1] += 1
                else:
                    self.counts[3] += 1
                changed = True
            elif e.kind == EventKind.DEV_CARD_PLAYED and e.value == DevCard.KNIGHT:
                self.counts[4] += 1
                changed = True
            elif e.kind == EventKind.LONGEST_ROAD_CHANGED:
                self.counts[5] = float(e.player == seat)
                changed = True
            elif e.kind == EventKind.LARGEST_ARMY_CHANGED:
                self.counts[6] = float(e.player == seat)
                changed = True
        return changed

    def normalized(self) -> np.ndarray:
        return np.minimum(self.counts / STAT_SCALE, 1.5).astype(np.float32)

    def copy(self) -> "StrategyStats":
        return StrategyStats(list(self.build_order), self.counts.copy())


def track_step(stats: list[StrategyStats], events: list[Event], phase_before: Phase) -> None:
    """Update every seat's statistics with the events of one step (`phase_before`: the phase
    in which the step was taken; setup placements are not purchases)."""
    setup = phase_before in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)
    for seat, seat_stats in enumerate(stats):
        seat_stats.update(events, seat, setup)


def encode_z(stats: StrategyStats, reward_active: bool = True) -> np.ndarray:
    """The z of a finished (hindsight) or ongoing game's statistics."""
    z = np.zeros(Z_DIM, dtype=np.float32)
    for i, kind in enumerate(stats.build_order[:BUILD_ORDER_LENGTH]):
        z[i * NUM_PURCHASE_KINDS + kind] = 1.0
    z[Z_BUILD:Z_PRESENT] = stats.normalized()
    z[Z_PRESENT] = 1.0
    z[Z_REWARD] = float(reward_active)
    return z


def zero_z() -> np.ndarray:
    return np.zeros(Z_DIM, dtype=np.float32)


def z_build_order(z: np.ndarray) -> list[int]:
    rows = z[:Z_BUILD].reshape(BUILD_ORDER_LENGTH, NUM_PURCHASE_KINDS)
    return [int(row.argmax()) for row in rows if row.max() > 0]


def levenshtein(a: list[int], b: list[int]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def strategy_distance(
    stats: StrategyStats, z: np.ndarray, build_order_weight: float, statistics_weight: float
) -> float:
    """D(stats, z): edit distance of the build-order prefixes of equal length (normalized by
    the full length) plus the mean L1 distance of the normalized counts."""
    if z[Z_PRESENT] <= 0:
        return 0.0
    target = z_build_order(z)
    n = min(len(stats.build_order), len(target))
    bo = levenshtein(stats.build_order[:n], target[:n]) / BUILD_ORDER_LENGTH
    # Built more than the target's whole order: the extra purchases count as insertions.
    if len(stats.build_order) > len(target):
        bo += (min(len(stats.build_order), BUILD_ORDER_LENGTH) - len(target)) / BUILD_ORDER_LENGTH
    counts = float(np.abs(stats.normalized() - z[Z_BUILD:Z_PRESENT]).mean())
    return build_order_weight * bo + statistics_weight * counts


def reward_decay(turn: int, decay_turns: tuple[int, ...]) -> float:
    factor = 1.0
    for i, limit in enumerate(decay_turns):
        if turn >= limit:
            factor = 0.0 if i == len(decay_turns) - 1 else factor * 0.5
    return factor


class StrategyLibrary:
    """Hindsight z's of teacher games, grouped by archetype (stored as .npz)."""

    def __init__(self, archetypes: dict[str, np.ndarray]) -> None:
        if not archetypes or any(len(zs) == 0 for zs in archetypes.values()):
            raise ValueError("a strategy library needs at least one z per archetype")
        for zs in archetypes.values():
            if zs.ndim != 2 or zs.shape[1] != Z_DIM:
                raise ValueError(f"library z's must have shape (n, {Z_DIM})")
        self.archetypes = {k: np.asarray(v, dtype=np.float32) for k, v in archetypes.items()}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.archetypes)

    def sample(self, archetype: str, rng: np.random.Generator, reward_active: bool) -> np.ndarray:
        zs = self.archetypes[archetype]
        z = zs[int(rng.integers(len(zs)))].copy()
        z[Z_PRESENT] = 1.0
        z[Z_REWARD] = float(reward_active)
        return z

    def prototype(self, archetype: str) -> np.ndarray:
        return self.archetypes[archetype].mean(0)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return dict(self.archetypes)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.archetypes)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StrategyLibrary":
        with np.load(path) as data:
            return cls({k: data[k] for k in data.files})


SELECT = "select"
NONE = "none"


class StrategyController:
    """One seat's strategy for one episode.

    mode "prescribed": a fixed z (from the library); "none": no z; "select": the policy's
    selector head chooses an archetype (or none) at the seat's first decision and reconsiders
    at its roll every `reconsider_turns` own turns. The selection is a separate decision from
    the action: it is stored in the observation (`SELECT_KEY`, `CHOICE_KEY`) and trained with
    its own log-probability term, so the action heads always see the z they act under.
    """

    def __init__(
        self,
        config: StrategyConfig,
        mode: str,
        *,
        library: StrategyLibrary | None = None,
        archetype: str | None = None,
        rng: np.random.Generator | None = None,
        reward_active: bool = False,
    ) -> None:
        if mode not in ("prescribed", NONE, SELECT):
            raise ValueError(f"unknown strategy mode {mode!r}")
        if mode == "prescribed" and (library is None or archetype is None):
            raise ValueError("a prescribed strategy needs a library and an archetype")
        self.config = config
        self.mode = mode
        self.library = library
        self.rng = rng or np.random.default_rng(0)
        self.reward_active = reward_active and mode != NONE
        self.archetype = archetype if mode == "prescribed" else None
        self.z = (
            library.sample(archetype, self.rng, self.reward_active)
            if mode == "prescribed" and library is not None and archetype is not None
            else zero_z()
        )
        self.choices: list[int] = []
        """Selections made this episode (archetype index; len(archetypes) = none)."""
        self.switches = 0
        self._last_selection_turn: int | None = None
        self._distance: float | None = None
        self._pending_cost = 0.0

    @property
    def label(self) -> str:
        if self.mode == SELECT:
            return SELECT
        return self.archetype or NONE

    def needs_selection(self, game, seat: int) -> bool:
        if self.mode != SELECT or game.is_over or game.decision_player != seat:
            return False
        if self._last_selection_turn is None:
            return True
        own_turns = (game.turn - self._last_selection_turn) // 4
        return (
            game.phase == Phase.PRE_ROLL
            and game.current_player == seat
            and own_turns >= self.config.reconsider_turns
        )

    def select(self, choice: int, game, stats: StrategyStats) -> None:
        """Apply the selector's choice (index into `config.archetypes`, or none)."""
        names = self.config.archetypes
        if not 0 <= choice <= len(names):
            raise ValueError("strategy choice out of range")
        previous = self.choices[-1] if self.choices else None
        if previous is not None and previous != choice:
            self.switches += 1
            self._pending_cost += self.config.switch_cost
        self.choices.append(choice)
        self._last_selection_turn = game.turn
        if choice == len(names) or self.library is None:
            self.archetype = None
            self.z = zero_z()
        else:
            self.archetype = names[choice]
            self.z = self.library.sample(self.archetype, self.rng, self.reward_active)
        # A new z starts a new distance baseline: the switch itself pays nothing but its cost.
        self._distance = self._current_distance(stats)

    def _current_distance(self, stats: StrategyStats) -> float:
        return strategy_distance(
            stats, self.z, self.config.build_order_weight, self.config.statistics_weight
        )

    def reward(self, game, stats: StrategyStats) -> float:
        """Strategy pseudo-reward since the previous call (0 when inactive)."""
        if not self.reward_active:
            self._pending_cost = 0.0
            return 0.0
        distance = self._current_distance(stats)
        previous = distance if self._distance is None else self._distance
        self._distance = distance
        value = reward_decay(game.turn, self.config.decay_turns) * (previous - distance)
        value -= self._pending_cost
        self._pending_cost = 0.0
        return float(value)

    def adherence(self, stats: StrategyStats) -> float:
        """Distance between the seat's statistics and its z (diagnostic; 0 without z)."""
        return self._current_distance(stats)


def make_controller(
    config: StrategyConfig,
    variant: str,
    library: StrategyLibrary | None,
    rng: np.random.Generator,
) -> StrategyController:
    """Controller for a named variant: "select", "none" or an archetype name."""
    reward = bool(rng.random() < config.pseudo_reward_prob)
    if variant == SELECT:
        return StrategyController(config, SELECT, library=library, rng=rng, reward_active=reward)
    if variant == NONE or library is None:
        return StrategyController(config, NONE, rng=rng)
    return StrategyController(
        config, "prescribed", library=library, archetype=variant, rng=rng, reward_active=reward
    )


def learner_variant(config: StrategyConfig, rng: np.random.Generator) -> str:
    """The learner's variant for a new episode under `config.selection`."""
    if config.selection == NONE or rng.random() < config.none_prob:
        return NONE
    prescribed = config.selection == "prescribed" or (
        config.selection == "mixed" and rng.random() < config.prescribed_prob
    )
    if prescribed:
        return config.archetypes[int(rng.integers(len(config.archetypes)))]
    return SELECT


def strategy_observation(obs: dict, controller: StrategyController | None, select: bool) -> dict:
    """`obs` with the seat's z and selection fields (zeros without a controller)."""
    z = zero_z() if controller is None else controller.z
    choice = -1
    if select and controller is not None and controller.choices:
        choice = controller.choices[-1]
    return {
        **obs,
        STRATEGY_KEY: np.asarray(z, dtype=np.float32),
        SELECT_KEY: np.asarray(float(select), dtype=np.float32),
        CHOICE_KEY: np.asarray(choice, dtype=np.int64),
    }
