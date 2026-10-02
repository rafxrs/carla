"""N-game tournaments between named agents, over several seeds, optionally in parallel."""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from carla.agents.registry import make_agent
from carla.engine.config import EngineConfig
from carla.engine.constants import NUM_PLAYERS
from carla.engine.rng import derive_seed
from carla.eval.metrics import AgentSummary, summarize
from carla.eval.runner import GameResult, play_game


@dataclass
class TournamentResult:
    results: list[GameResult]
    summary: dict[str, AgentSummary]
    """Pooled over all seeds."""
    per_seed: dict[int, dict[str, AgentSummary]]


def schedule(
    agents: Sequence[str], num_games: int, seed: int, rotate_seats: bool = True
) -> list[tuple[int, tuple[str, ...]]]:
    """(game seed, agent name per seat) for each game. Rotation puts every agent in every seat
    equally often (use a multiple of 4 games)."""
    if len(agents) != NUM_PLAYERS:
        raise ValueError(f"need {NUM_PLAYERS} agents, got {len(agents)}")
    games = []
    for i in range(num_games):
        k = i % NUM_PLAYERS if rotate_seats else 0
        games.append((derive_seed(seed, "game", i), tuple(agents[k:]) + tuple(agents[:k])))
    return games


def _play(job: tuple[int, tuple[str, ...], EngineConfig, bool]) -> GameResult:
    game_seed, names, config, record = job
    return play_game([make_agent(n) for n in names], game_seed, config, record=record)


def run_tournament(
    agents: Sequence[str],
    num_games: int,
    seeds: Sequence[int] = (0,),
    engine_config: EngineConfig | None = None,
    rotate_seats: bool = True,
    num_workers: int = 1,
    confidence: float = 0.95,
    records_dir: str | Path | None = None,
) -> TournamentResult:
    """Play `num_games` per seed. Results are identical for any `num_workers`."""
    config = engine_config or EngineConfig()
    jobs = [
        (game_seed, names, config, records_dir is not None)
        for seed in seeds
        for game_seed, names in schedule(agents, num_games, seed, rotate_seats)
    ]
    if num_workers > 1:
        with ProcessPoolExecutor(num_workers) as pool:
            results = list(pool.map(_play, jobs, chunksize=max(1, len(jobs) // (4 * num_workers))))
    else:
        results = [_play(job) for job in jobs]
    if records_dir is not None:
        for i, r in enumerate(results):
            r.record.save(Path(records_dir) / f"game_{i:05d}_{r.seed}.json.gz")
    per_seed = {
        seed: summarize(results[i * num_games : (i + 1) * num_games], confidence)
        for i, seed in enumerate(seeds)
    }
    return TournamentResult(results, summarize(results, confidence), per_seed)
