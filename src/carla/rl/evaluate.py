"""Evaluate a saved MLP/GNN policy on held-out games against fixed bots."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

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


@dataclass
class _ActiveGame:
    """State for one concurrently evaluated game."""

    env: CatanEnv
    seat: int
    obs: dict[str, Any]


def _resolve_device(device: str) -> torch.device:
    """Resolve 'auto' to CUDA when available, otherwise CPU."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors in a nested observation/mask structure."""
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    opponent: str = "heuristic",
    boards: int = 10,
    seed: int = 10_000,
    device: str = "auto",
) -> PolicyEvaluation:
    """Evaluate a policy on held-out boards using batched greedy inference.

    Each board is played once with the learner in each of the four seats.

    All unfinished games are evaluated together in a single policy batch.
    Environment stepping remains sequential, but neural-network inference is
    batched and runs on CUDA automatically when available.
    """
    if opponent not in ("random", "heuristic"):
        raise ValueError("opponent must be random or heuristic")
    if boards <= 0:
        raise ValueError("boards must be positive")

    eval_device = _resolve_device(device)

    # Read config on CPU first.
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if data.get("config") is None:
        raise ValueError("checkpoint has no experiment config")

    config = from_dict(ExperimentConfig, data["config"])

    if config.model.encoder not in ("mlp", "gnn") or config.model.belief != "none":
        raise ValueError(
            "this evaluator supports feed-forward MLP/GNN policies without belief"
        )

    # Load checkpoint on CPU, then move the complete policy—including
    # registered GNN topology buffers—to the evaluation device.
    policy = build_policy(config.model)
    load_checkpoint(checkpoint, policy)
    policy.to(eval_device)
    policy.eval()

    active_games: list[_ActiveGame] = []

    # Create every board/seat combination up front.
    for board in range(boards):
        game_seed = derive_seed(seed, "evaluation-board", board)

        for seat in range(4):
            env = CatanEnv(
                config.engine,
                replace(config.env, learner_seat=seat),
                [make_agent(opponent) for _ in range(3)],
            )

            obs, _ = env.reset(seed=game_seed)

            active_games.append(
                _ActiveGame(
                    env=env,
                    seat=seat,
                    obs=obs,
                )
            )

    wins = 0
    total_vp = 0

    try:
        with torch.inference_mode():
            while active_games:
                # ----------------------------------------------------------
                # 1. Batch the current learner observation from every
                #    unfinished game.
                # ----------------------------------------------------------
                observations = [game.obs for game in active_games]
                batch = batch_observations(observations)
                batch = _to_device(batch, eval_device)

                # ----------------------------------------------------------
                # 2. One policy forward pass for ALL active games.
                # ----------------------------------------------------------
                output = policy(batch)
                distribution = output.distribution(batch["masks"])

                actions = distribution.sample(deterministic=True)

                # Environment expects NumPy/CPU actions.
                actions_np = actions.detach().cpu().numpy()

                # ----------------------------------------------------------
                # 3. Apply each returned action to its corresponding game.
                # ----------------------------------------------------------
                still_active: list[_ActiveGame] = []

                for game, action in zip(active_games, actions_np, strict=True):
                    obs, _, terminated, truncated, info = game.env.step(action)

                    done = terminated or truncated

                    if done:
                        wins += int(info["winner"] == game.seat)
                        total_vp += game.env.game.victory_points(game.seat)
                        game.env.close()
                    else:
                        game.obs = obs
                        still_active.append(game)

                active_games = still_active

    finally:
        # Ensure environments are closed if evaluation raises unexpectedly.
        for game in active_games:
            game.env.close()

    games = boards * 4
    low, high = wilson_interval(wins, games)

    return PolicyEvaluation(
        opponent=opponent,
        games=games,
        wins=wins,
        win_rate=wins / games,
        ci_low=low,
        ci_high=high,
        mean_vp=total_vp / games,
    )