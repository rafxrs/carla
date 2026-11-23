"""Frozen-policy evaluation on held-out boards with individual game records."""

import csv
from dataclasses import dataclass, field, replace
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
from carla.rl.checkpoint import isolated_rng
from carla.rl.metrics import EVALUATION_FIELDS
from carla.utils.logging import CsvJournal


@dataclass(frozen=True)
class PolicyEvaluation:
    opponent: str
    games: int
    wins: int
    win_rate: float
    ci_low: float
    ci_high: float
    mean_vp: float
    records: list[dict] = field(default_factory=list)


@dataclass
class _ActiveGame:
    env: CatanEnv
    obs: dict
    board: int
    seat: int
    reward_sum: float = 0.0
    learner_decisions: int = 0


@torch.no_grad()
def evaluate_policy(
    policy,
    config: ExperimentConfig,
    *,
    opponent: str = "heuristic",
    boards: int = 10,
    seed: int = 10_000,
    deterministic: bool = True,
    batch_games: int = 64,
) -> PolicyEvaluation:
    """Batch active games, playing each held-out board in all four learner seats.

    Environments step sequentially; inference uses the policy's device. RNG and module modes
    are restored, and records retain board/seat order regardless of game completion order.
    """
    VALID_EVAL_OPPONENTS = (
        "random",
        "heuristic05",
        "heuristic10",
        "heuristic15",
        "heuristic25",
        "heuristic50",
        "heuristic75",
        "heuristic",
    )
    
    if opponent not in VALID_EVAL_OPPONENTS or boards <= 0:
        raise ValueError(
            "evaluation needs a registered evaluation opponent and positive boards"
        )
    if batch_games <= 0:
        raise ValueError("batch_games must be positive")
    if config.model.encoder not in ("mlp", "gnn"):
        raise ValueError("this evaluator supports MLP/GNN policies only")
    modes = [(module, module.training) for module in policy.modules()]
    device = next(policy.parameters()).device
    records = []
    try:
        policy.eval()
        with isolated_rng(seed):
            jobs = [
                (board, seat, derive_seed(seed, "evaluation-board", board))
                for board in range(boards)
                for seat in range(4)
            ]
            for start in range(0, len(jobs), batch_games):
                envs = []
                try:
                    active = []
                    for board, seat, game_seed in jobs[start : start + batch_games]:
                        env = CatanEnv(
                            config.engine,
                            replace(config.env, learner_seat=seat),
                            [make_agent(opponent) for _ in range(3)],
                        )
                        envs.append(env)  # Also close environments whose reset raises.
                        obs, _ = env.reset(seed=game_seed)
                        active.append(_ActiveGame(env, obs, board, seat))
                    state = policy.initial_state(len(active), device)
                    while active:
                        batch = batch_observations([game.obs for game in active], device=device)
                        out = policy(batch, state)
                        actions = (
                            out.distribution(batch["masks"])
                            .sample(deterministic=deterministic)
                            .cpu()
                            .numpy()
                        )
                        next_active, kept_indices = [], []
                        for index, game in enumerate(active):
                            env, seat = game.env, game.seat
                            game.obs, reward, terminated, truncated, info = env.step(actions[index])
                            game.reward_sum += reward
                            game.learner_decisions += 1
                            if not (terminated or truncated):
                                next_active.append(game)
                                kept_indices.append(index)
                                continue
                            winner = info["winner"]
                            records.append(
                                {
                                    "eval_seed": seed,
                                    "board": game.board,
                                    "game_seed": env.game.seed,
                                    "seat": seat,
                                    "opponent": opponent,
                                    "mode": "greedy" if deterministic else "sampled",
                                    "outcome": (
                                        "truncated"
                                        if truncated
                                        else ("win" if winner == seat else "loss")
                                    ),
                                    "winner": -1 if winner is None else winner,
                                    "return": game.reward_sum,
                                    "learner_decisions": game.learner_decisions,
                                    "turns": env.game.turn,
                                    "victory_points": env.game.victory_points(seat),
                                }
                            )
                        # State rows follow survivors in exactly the same order as observations.
                        state = (
                            out.state.index_select(
                                0,
                                torch.as_tensor(
                                    kept_indices, device=out.state.device, dtype=torch.long
                                ),
                            )
                            if out.state is not None
                            else None
                        )
                        active = next_active
                finally:
                    for env in envs:
                        env.close()
    finally:
        for module, training in modes:
            module.training = training
    records.sort(key=lambda record: (record["board"], record["seat"]))
    wins = sum(record["outcome"] == "win" for record in records)
    low, high = wilson_interval(wins, len(records))
    return PolicyEvaluation(
        opponent,
        len(records),
        wins,
        wins / len(records),
        low,
        high,
        sum(r["victory_points"] for r in records) / len(records),
        records,
    )


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    opponent: str = "heuristic",
    boards: int = 10,
    seed: int = 10_000,
    deterministic: bool = True,
    batch_games: int = 64,
    device: str = "auto",
) -> PolicyEvaluation:
    with isolated_rng():
        resolved_device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if data.get("config") is None:
            raise ValueError("checkpoint has no experiment config")
        config = from_dict(ExperimentConfig, data["config"])
        policy = build_policy(config.model)
        policy.load_state_dict(data["policy"])
        policy.to(resolved_device)
        return evaluate_policy(
            policy,
            config,
            opponent=opponent,
            boards=boards,
            seed=seed,
            deterministic=deterministic,
            batch_games=batch_games,
        )


def export_evaluation(
    checkpoint: Path, result: PolicyEvaluation, path: Path, *, split: str = "test"
) -> None:
    """Export final tests separately so subsequent training resume cannot truncate them."""
    if path.name == "evaluation_games.csv":
        raise ValueError("use a separate export, e.g. evaluation_test_games.csv")
    checkpoint, path = checkpoint.resolve(), path.resolve()
    run = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = from_dict(ExperimentConfig, data["config"])
    if (
        split == "test"
        and cfg.eval.every_env_steps
        and any(r["eval_seed"] == cfg.eval.seed for r in result.records)
    ):
        raise ValueError(
            "test seed must differ from the training validation seed; pass --seed 20000"
        )
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
    checkpoint_name = checkpoint.relative_to(run).as_posix()
    keys = {
        (
            r["checkpoint"],
            r["split"],
            r["opponent"],
            r["mode"],
            int(r["eval_seed"]),
            int(r["board"]),
            int(r["seat"]),
        )
        for r in existing
    }
    if any(
        (checkpoint_name, split, r["opponent"], r["mode"], r["eval_seed"], r["board"], r["seat"])
        in keys
        for r in result.records
    ):
        raise ValueError("evaluation export already contains these checkpoint/games")
    path.parent.mkdir(parents=True, exist_ok=True)
    journal = CsvJournal(path, EVALUATION_FIELDS, len(existing))
    state = data.get("training_state") or {}
    for record in result.records:
        journal.write(
            {
                "update": data["counters"]["updates"],
                "env_steps": data["counters"]["env_steps"],
                "evaluation_target": data["counters"]["env_steps"],
                "training_seconds": state.get("training_seconds", 0.0),
                "wall_seconds": state.get("wall_seconds", 0.0),
                "checkpoint": checkpoint_name,
                "split": split,
                **record,
            }
        )
