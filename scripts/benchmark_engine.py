"""Throughput of the engine, bots and environment (single process, then N processes).

Usage: python scripts/benchmark_engine.py [--seconds 5] [--processes 4]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from carla.agents import make_agent
from carla.engine.simulate import play_random_game
from carla.envs import CatanEnv, sample_heads
from carla.eval.runner import play_game


def random_games(seconds: float, seed: int = 0) -> tuple[int, int, float]:
    start, games, steps = time.perf_counter(), 0, 0
    while time.perf_counter() - start < seconds:
        steps += play_random_game(seed + games).num_steps
        games += 1
    return games, steps, time.perf_counter() - start


def heuristic_games(seconds: float) -> tuple[int, int, float]:
    start, games, steps = time.perf_counter(), 0, 0
    while time.perf_counter() - start < seconds:
        steps += play_game([make_agent("heuristic") for _ in range(4)], seed=games).steps
        games += 1
    return games, steps, time.perf_counter() - start


def env_steps(seconds: float) -> tuple[int, int, float]:
    env, rng = CatanEnv(), np.random.default_rng(0)
    start, episodes, steps = time.perf_counter(), 0, 0
    obs, _ = env.reset(seed=0)
    while time.perf_counter() - start < seconds:
        obs, _, terminated, truncated, _ = env.step(sample_heads(obs["masks"], rng))
        steps += 1
        if terminated or truncated:
            episodes += 1
            obs, _ = env.reset()
    return episodes, steps, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()

    for label, fn, unit in (
        ("random self-play (engine)", random_games, "decisions"),
        ("heuristic self-play", heuristic_games, "decisions"),
        ("CatanEnv, random learner vs 3 random", env_steps, "learner steps"),
    ):
        games, steps, elapsed = fn(args.seconds)
        print(f"{label:<40} {games / elapsed:8.1f} games/s {steps / elapsed:10.0f} {unit}/s")

    with ProcessPoolExecutor(args.processes) as pool:
        start = time.perf_counter()
        runs = list(pool.map(random_games, [args.seconds] * args.processes, range(0, 10**6, 10**5)))
        elapsed = time.perf_counter() - start
    games, steps = sum(r[0] for r in runs), sum(r[1] for r in runs)
    print(
        f"{f'random self-play x{args.processes} processes':<40} {games / elapsed:8.1f} games/s "
        f"{steps / elapsed:10.0f} decisions/s"
    )


if __name__ == "__main__":
    main()
