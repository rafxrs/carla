"""Evaluate a saved MLP policy on held-out games against fixed bots."""

from dataclasses import dataclass, replace
from pathlib import Path

import torch

from carla.agents import make_agent
from carla.config.loader import from_dict
from carla.config.schema import ExperimentConfig
from carla.engine.rng import derive_seed
from carla.envs.catan_env import CatanEnv
from carla.eval.metrics import wilson_interval
from carla.models.base import batch_observations
from carla.models.factory import build_policy
from carla.rl.checkpoint import load_checkpoint


@dataclass(frozen=True)
class PolicyEvaluation:
    opponent: str
    games: int
    wins: int
    win_rate: float
    ci_low: float
    ci_high: float
    mean_vp: float


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    opponent: str = "heuristic",
    boards: int = 10,
    seed: int = 10_000,
) -> PolicyEvaluation:
    """Play each board in each seat; greedy masked actions use public observations only."""
    if opponent not in ("random", "heuristic"):
        raise ValueError("opponent must be random or heuristic")
    if boards <= 0:
        raise ValueError("boards must be positive")
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if data.get("config") is None:
        raise ValueError("checkpoint has no experiment config")
    config = from_dict(ExperimentConfig, data["config"])
    if config.model.encoder != "mlp" or config.model.belief != "none":
        raise ValueError("this evaluator supports the MLP baseline only")
    policy = build_policy(config.model)
    load_checkpoint(checkpoint, policy)
    policy.eval()
    wins = 0
    total_vp = 0
    for board in range(boards):
        game_seed = derive_seed(seed, "evaluation-board", board)
        for seat in range(4):
            env = CatanEnv(
                config.engine,
                replace(config.env, learner_seat=seat),
                [make_agent(opponent) for _ in range(3)],
            )
            obs, _ = env.reset(seed=game_seed)
            done = False
            with torch.no_grad():
                while not done:
                    batch = batch_observations([obs])
                    action = policy(batch).distribution(batch["masks"]).sample(deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action[0].numpy())
                    done = terminated or truncated
            wins += info["winner"] == seat
            total_vp += env.game.victory_points(seat)
            env.close()
    games = boards * 4
    low, high = wilson_interval(wins, games)
    return PolicyEvaluation(opponent, games, wins, wins / games, low, high, total_vp / games)
