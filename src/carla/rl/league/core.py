"""League bookkeeping without torch: players, the four-player payoff, PFSP and matchmaking.

Adapted from AlphaStar's league (Vinyals et al. 2019, Methods and supplementary pseudocode):

- Players: learning players (main agents, main exploiters, league exploiters), frozen
  historical copies of them, protected bots (the heuristic teachers) and protected fixed
  checkpoints. A strategy-conditioned policy appears in the payoff once per strategy variant
  ("select", "none" or an archetype), so conditioned variants are matched and tracked apart.
- Payoff: four-player games are reduced to pairwise results between the learner and each
  opponent seat: a win if the learner finished ahead (the winner is ahead of everyone, others
  rank by victory points), a draw on equal points. P[A beats B] is then 0.5 for equals as in a
  two-player game, so AlphaStar's thresholds (0.3, 0.7) keep their meaning. Counts decay
  (0.99 per game) so the payoff follows the learners. Table win rates (the learner won the
  game, 0.25 for equals) are tracked separately for the exploiters' own-success criterion.
- Matchmaking objective (not a loss): PFSP samples each opponent seat with probability
  f(P[learner beats candidate]); f_hard = (1 - x)^2 focuses on the hardest players, f_var =
  x(1 - x) on players of similar strength. Tables are complete four-player compositions: the
  learner's seat is random in the env and the three opponents are drawn per table.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from carla.config.schema import LeagueConfig

MAIN = "main"
MAIN_EXPLOITER = "main_exploiter"
LEAGUE_EXPLOITER = "league_exploiter"
ROLES = (MAIN, MAIN_EXPLOITER, LEAGUE_EXPLOITER)


def pfsp(win_rates: np.ndarray, weighting: str = "squared") -> np.ndarray:
    """PFSP probabilities from the learner's win rates against each candidate."""
    functions = {
        "variance": lambda x: x * (1 - x),
        "linear": lambda x: 1 - x,
        "linear_capped": lambda x: np.minimum(0.5, 1 - x),
        "squared": lambda x: (1 - x) ** 2,
    }
    rates = np.asarray(win_rates, dtype=np.float64)
    if rates.size == 0:
        return rates
    probabilities = functions[weighting](rates)
    total = probabilities.sum()
    if total < 1e-10:
        return np.ones_like(rates) / len(rates)
    return probabilities / total


def remove_monotonic_suffix(win_rates: Sequence[float], players: Sequence) -> tuple[list, list]:
    """AlphaStar's forgetting check: drop the trailing run of historical players whose win
    rates only decrease (the newest, still-improving copies are not "forgotten")."""
    rates = list(win_rates)
    if not rates:
        return [], []
    for i in range(len(rates) - 1, 0, -1):
        if rates[i - 1] < rates[i]:
            return rates[: i + 1], list(players[: i + 1])
    return [], []


def pairwise_result(learner_score: float, opponent_score: float) -> str:
    if learner_score > opponent_score:
        return "win"
    if learner_score < opponent_score:
        return "loss"
    return "draw"


def final_scores(winner: int | None, victory_points: Sequence[int]) -> list[float]:
    """Rank scores of the four seats: the winner first, then victory points."""
    return [vp + (100.0 if seat == winner else 0.0) for seat, vp in enumerate(victory_points)]


@dataclass
class LeaguePlayer:
    id: str
    kind: str
    """"learner", "historical", "bot" or "checkpoint"."""
    role: str | None = None
    """Learner role, or the role of the learner a historical player copies."""
    parent: str | None = None
    """Learner id a historical player was copied from."""
    strategy_aware: bool = False
    selector: bool = False
    protected: bool = False
    created_step: int = 0
    path: str | None = None
    """Weights file (historical and checkpoint players)."""
    variant: str | None = None
    """Fixed strategy variant (exploiter generations); None: any variant."""
    generation: int = 0
    snapshots: int = 0
    """Snapshots taken of a learner so far (ids stay unique after pruning)."""

    def variants(self, allowed: Sequence[str]) -> list[str | None]:
        if not self.strategy_aware:
            return [None]
        if self.variant is not None:
            return [self.variant]
        return [v for v in allowed if v != "select" or self.selector]

    def key(self, variant: str | None) -> str:
        return self.id if variant is None else f"{self.id}@{variant}"


def player_of(key: str) -> str:
    return key.split("@", 1)[0]


class Payoff:
    """Decayed pairwise results between learners (home) and opponent keys (away)."""

    def __init__(self, decay: float = 0.99) -> None:
        self.decay = decay
        self.games: dict = defaultdict(float)
        self.wins: dict = defaultdict(float)
        self.draws: dict = defaultdict(float)
        self.table_games: dict = defaultdict(float)
        self.table_wins: dict = defaultdict(float)

    def _record(self, home: str, away: str, result: str) -> None:
        pair = (home, away)
        for table in (self.games, self.wins, self.draws):
            table[pair] *= self.decay
        self.games[pair] += 1
        if result == "win":
            self.wins[pair] += 1
        elif result == "draw":
            self.draws[pair] += 1

    def update(self, home: str, away_key: str, result: str) -> None:
        """One pairwise result; also counted for the opponent player over all its variants."""
        self._record(home, away_key, result)
        if player_of(away_key) != away_key:
            self._record(home, player_of(away_key), result)

    def update_table(self, home: str, group: str, won: bool) -> None:
        pair = (home, group)
        self.table_games[pair] = self.table_games[pair] * self.decay + 1
        self.table_wins[pair] = self.table_wins[pair] * self.decay + float(won)

    def win_rate(self, home: str, away: str) -> float:
        games = self.games.get((home, away), 0.0)
        if games == 0:
            return 0.5
        return (self.wins.get((home, away), 0.0) + 0.5 * self.draws.get((home, away), 0.0)) / games

    def table_win_rate(self, home: str, group: str) -> float:
        games = self.table_games.get((home, group), 0.0)
        return 0.25 if games == 0 else self.table_wins.get((home, group), 0.0) / games

    def count(self, home: str, away: str) -> float:
        return self.games.get((home, away), 0.0)

    def rates(self, home: str, aways: Sequence[str]) -> np.ndarray:
        return np.asarray([self.win_rate(home, a) for a in aways], dtype=np.float64)

    def state(self) -> dict:
        return {
            "decay": self.decay,
            "games": dict(self.games),
            "wins": dict(self.wins),
            "draws": dict(self.draws),
            "table_games": dict(self.table_games),
            "table_wins": dict(self.table_wins),
        }

    @classmethod
    def from_state(cls, state: dict) -> "Payoff":
        payoff = cls(state["decay"])
        for name in ("games", "wins", "draws", "table_games", "table_wins"):
            getattr(payoff, name).update(state[name])
        return payoff


Slot = tuple[str, str | None]
"""(player id, variant) of one opponent seat."""


@dataclass
class Plan:
    """Matchmaking for one collection: tables the learner's new episodes draw from."""

    learner: str
    tables: list[tuple[Slot, Slot, Slot]]
    branches: list[str]
    """Branch that produced each table (diagnostics)."""
    learner_variant: str | None = None
    counts: dict = field(default_factory=dict)


class Matchmaker:
    """AlphaStar's matchmaking for each role, over four-player tables."""

    def __init__(self, config: LeagueConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    # ------------------------------------------------------------------ candidates

    def _keys(self, players: Sequence[LeaguePlayer]) -> list[Slot]:
        out = []
        for player in players:
            for variant in player.variants(self.config.variants):
                out.append((player.id, variant))
        return out

    def _pfsp_slots(
        self, learner: str, slots: list[Slot], players: dict, payoff: Payoff, weighting: str, n: int
    ) -> list[Slot]:
        keys = [players[pid].key(v) for pid, v in slots]
        probabilities = pfsp(payoff.rates(learner, keys), weighting)
        picks = self.rng.choice(len(slots), size=n, p=probabilities)
        return [slots[int(i)] for i in picks]

    def _random_variant(self, player: LeaguePlayer) -> str | None:
        variants = player.variants(self.config.variants)
        return variants[int(self.rng.integers(len(variants)))]

    def frozen(self, players: dict) -> list[LeaguePlayer]:
        return [p for p in players.values() if p.kind != "learner"]

    # ------------------------------------------------------------------ branches

    def table(self, learner: LeaguePlayer, players: dict, payoff: Payoff) -> tuple[list[Slot], str]:
        role = learner.role
        if role == MAIN:
            return self._main_table(learner, players, payoff)
        if role == MAIN_EXPLOITER:
            return self._main_exploiter_table(learner, players, payoff)
        if role == LEAGUE_EXPLOITER:
            slots = self._keys(self.frozen(players))
            return (
                self._pfsp_slots(
                    learner.id, slots, players, payoff, self.config.pfsp_league_exploiter, 3
                ),
                "pfsp",
            )
        raise ValueError(f"unknown learner role {role!r}")

    def _mains(self, players: dict) -> list[LeaguePlayer]:
        return [p for p in players.values() if p.kind == "learner" and p.role == MAIN]

    def _main_table(self, learner, players, payoff) -> tuple[list[Slot], str]:
        sp, pfsp_share, _ = self.config.main_branches
        coin = self.rng.random()
        frozen = self.frozen(players)
        if coin < pfsp_share and frozen:
            slots = self._keys(frozen)
            return (
                self._pfsp_slots(learner.id, slots, players, payoff, self.config.pfsp_main, 3),
                "pfsp",
            )
        mains = self._mains(players)
        opponent = mains[int(self.rng.integers(len(mains)))]
        if coin < pfsp_share + (1 - pfsp_share - sp) and frozen:
            verification = self._verification(learner, opponent, players, payoff)
            if verification is not None:
                return verification, "verification"
        return self._selfplay(learner, opponent, players, payoff)

    def _selfplay(self, learner, opponent, players, payoff) -> tuple[list[Slot], str]:
        if opponent.id == learner.id or payoff.win_rate(learner.id, opponent.id) > (
            self.config.selfplay_floor
        ):
            return [(opponent.id, self._random_variant(opponent)) for _ in range(3)], "selfplay"
        history = [p for p in self.frozen(players) if p.parent == opponent.id]
        if not history:
            return [(opponent.id, self._random_variant(opponent)) for _ in range(3)], "selfplay"
        slots = self._keys(history)
        return (
            self._pfsp_slots(learner.id, slots, players, payoff, "variance", 3),
            "selfplay-curriculum",
        )

    def _verification(self, learner, opponent, players, payoff) -> list[Slot] | None:
        exploiters = [
            p
            for p in self.frozen(players)
            if p.kind == "historical" and p.role in (MAIN_EXPLOITER, LEAGUE_EXPLOITER)
        ]
        slots = self._keys(exploiters)
        keys = [players[pid].key(v) for pid, v in slots]
        rates = payoff.rates(learner.id, keys)
        if len(rates) and rates.min() < self.config.exploit_threshold:
            return self._pfsp_slots(learner.id, slots, players, payoff, "squared", 3)
        history = sorted(
            (p for p in self.frozen(players) if p.parent == opponent.id),
            key=lambda p: p.created_step,
        )
        rates = list(payoff.rates(learner.id, [p.id for p in history]))
        rates, history = remove_monotonic_suffix(rates, history)
        if rates and min(rates) < self.config.forgotten_threshold:
            slots = self._keys(history)
            return self._pfsp_slots(learner.id, slots, players, payoff, "squared", 3)
        return None

    def _main_exploiter_table(self, learner, players, payoff) -> tuple[list[Slot], str]:
        mains = self._mains(players)
        main = mains[int(self.rng.integers(len(mains)))]
        rate = payoff.win_rate(learner.id, main.id)
        if rate > self.config.exploiter_target or self.rng.random() < 0.5:
            return [(main.id, self._random_variant(main)) for _ in range(3)], "exploit-main"
        history = [p for p in self.frozen(players) if p.parent == main.id]
        if not history:
            return [(main.id, self._random_variant(main)) for _ in range(3)], "exploit-main"
        slots = self._keys(history)
        return (
            self._pfsp_slots(learner.id, slots, players, payoff, self.config.pfsp_exploiter, 3),
            "exploit-history",
        )

    def plan(self, learner: LeaguePlayer, players: dict, payoff: Payoff, n: int) -> Plan:
        tables, branches = [], []
        for _ in range(n):
            slots, branch = self.table(learner, players, payoff)
            order = self.rng.permutation(3)
            tables.append(tuple(slots[int(i)] for i in order))
            branches.append(branch)
        counts: dict = defaultdict(int)
        for branch in branches:
            counts[branch] += 1
        return Plan(learner.id, tables, branches, learner.variant, dict(counts))


# ---------------------------------------------------------------------- snapshots and resets


def ready_to_checkpoint(
    learner: LeaguePlayer, steps_since: int, players: dict, payoff: Payoff, config: LeagueConfig
) -> bool:
    """AlphaStar's snapshot rules with steps in learner decisions (see `LeagueConfig`)."""
    if learner.role == MAIN:
        if steps_since < config.main_snapshot_steps:
            return False
        history = [p for p in players.values() if p.kind != "learner"]
        rates = payoff.rates(learner.id, [p.id for p in history])
        beaten = len(rates) > 0 and rates.min() > config.beat_threshold
        return beaten or steps_since >= config.main_snapshot_max_steps
    if steps_since < config.exploiter_min_steps:
        return False
    if steps_since >= config.exploiter_max_steps:
        return True
    if learner.role == MAIN_EXPLOITER:
        mains = [p for p in players.values() if p.kind == "learner" and p.role == MAIN]
        rates = payoff.rates(learner.id, [p.id for p in mains])
        own = payoff.table_win_rate(learner.id, "main")
        # Beating the mains must come with own wins: making them lose is not enough.
        return bool(
            len(rates)
            and rates.min() > config.beat_threshold
            and own >= config.exploiter_min_win_rate
        )
    history = [p for p in players.values() if p.kind != "learner"]
    rates = payoff.rates(learner.id, [p.id for p in history])
    return bool(len(rates) and rates.min() > config.beat_threshold)


def usefulness(
    player: LeaguePlayer,
    learners: Sequence[LeaguePlayer],
    payoff: Payoff,
    behaviour: dict[str, np.ndarray],
) -> float:
    """How much a historical player still teaches: hardness for the learners plus how
    distinct its behaviour is from the other historical players (min L1 distance)."""
    hardness = max((1 - payoff.win_rate(p.id, player.id) for p in learners), default=0.0)
    own = behaviour.get(player.id)
    if own is None:
        return hardness + 0.5
    others = [v for k, v in behaviour.items() if k != player.id]
    distinct = min((float(np.abs(own - v).mean()) for v in others), default=1.0)
    return hardness + min(1.0, distinct)


def prune_candidates(players: dict, payoff: Payoff, behaviour: dict, keep: int) -> list[str]:
    """Unprotected historical players to drop (least useful first) beyond `keep`; the newest
    copy of every learner is kept so self-play curricula always have one."""
    learners = [p for p in players.values() if p.kind == "learner"]
    historical = [p for p in players.values() if p.kind == "historical" and not p.protected]
    if len(historical) <= keep:
        return []
    newest = {}
    for p in historical:
        if p.parent not in newest or p.created_step > newest[p.parent].created_step:
            newest[p.parent] = p
    candidates = [p for p in historical if newest.get(p.parent) is not p]
    candidates.sort(key=lambda p: usefulness(p, learners, payoff, behaviour))
    return [p.id for p in candidates[: len(historical) - keep]]


def elo_like(payoff: Payoff, home: str, aways: Sequence[str]) -> float:
    """Mean logit of the pairwise rates (a compact strength summary for the journal)."""
    rates = np.clip(payoff.rates(home, aways), 0.01, 0.99)
    return float(np.mean([math.log(r / (1 - r)) for r in rates])) if len(rates) else 0.0
