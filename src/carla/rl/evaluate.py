"""Frozen-policy evaluation on held-out boards with individual game records."""

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from carla.agents import TEACHERS, make_agent
from carla.config.loader import from_dict
from carla.config.schema import ExperimentConfig
from carla.engine.rng import derive_seed
from carla.envs.catan_env import CatanEnv
from carla.envs.strategy import NONE, SELECT, STAT_NAMES, StrategyLibrary, make_controller
from carla.eval.metrics import wilson_interval
from carla.models.factory import build_policy
from carla.rl.acting import decide
from carla.rl.checkpoint import isolated_rng
from carla.rl.metrics import EVALUATION_FIELDS
from carla.utils.logging import CsvJournal

EVAL_BOTS = ("random", "heuristic", *TEACHERS)


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


def opponent_table(opponent: str) -> list[str]:
    """Three bot names: one name for every seat, or a comma-separated table of three."""
    names = [part.strip() for part in opponent.split(",") if part.strip()]
    if len(names) == 1:
        names = names * 3
    if len(names) != 3 or any(name not in EVAL_BOTS for name in names):
        raise ValueError(f"evaluation opponents must be one or three of {EVAL_BOTS}")
    return names


def default_variant(policy) -> str:
    """Evaluation strategy of a policy: its selector when it has one, else no z."""
    if getattr(policy, "num_strategies", 0):
        return SELECT
    return NONE


@torch.no_grad()
def evaluate_policy(
    policy,
    config: ExperimentConfig,
    *,
    opponent: str = "heuristic",
    boards: int = 10,
    seed: int = 10_000,
    deterministic: bool = True,
    strategy: str | None = None,
    library: StrategyLibrary | None = None,
) -> PolicyEvaluation:
    """Each held-out board is played in all four seats; RNG and module modes are restored.

    `opponent` names a bot or a comma-separated table of three. A strategy-conditioned policy
    plays `strategy` ("select", "none" or an archetype; default: select if it has a selector,
    else none); archetypes need the z `library`.
    """
    table = opponent_table(opponent)
    if boards <= 0:
        raise ValueError("evaluation needs positive boards")
    if config.model.encoder not in ("mlp", "gnn", "strategy"):
        raise ValueError("this evaluator supports MLP/GNN/strategy policies only")
    variant = strategy or default_variant(policy)
    if variant not in (SELECT, NONE) and library is None:
        raise ValueError(f"strategy {variant!r} needs a strategy library")
    modes = [(module, module.training) for module in policy.modules()]
    device = next(policy.parameters()).device
    records = []
    strategy_config = replace(config.strategy, pseudo_reward_prob=0.0)
    try:
        policy.eval()
        with isolated_rng(seed):
            for board in range(boards):
                game_seed = derive_seed(seed, "evaluation-board", board)
                for seat in range(4):
                    env = CatanEnv(
                        config.engine,
                        replace(config.env, learner_seat=seat),
                        [make_agent(name) for name in table],
                    )
                    controller = (
                        make_controller(
                            strategy_config,
                            variant,
                            library,
                            np.random.default_rng(derive_seed(game_seed, "strategy", seat)),
                        )
                        if policy.uses_strategy
                        else None
                    )
                    env.strategy = None
                    try:
                        obs, _ = env.reset(seed=game_seed)
                        done, reward_sum, decisions = False, 0.0, 0
                        state = policy.initial_state(1, device)
                        while not done:
                            step = decide(
                                policy,
                                [obs],
                                [controller],
                                [env.game],
                                [seat],
                                [env.aec.stats[seat]],
                                deterministic=deterministic,
                                state=state,
                            )
                            state = step.output.state
                            obs, reward, terminated, truncated, info = env.step(step.heads[0])
                            reward_sum += reward
                            decisions += 1
                            done = terminated or truncated
                        winner = info["winner"]
                        stats = env.aec.stats[seat]
                        records.append(
                            {
                                "eval_seed": seed,
                                "board": board,
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
                                "return": reward_sum,
                                "learner_decisions": decisions,
                                "turns": env.game.turn,
                                "victory_points": env.game.victory_points(seat),
                                "strategy": variant if policy.uses_strategy else "",
                                "strategy_choices": (
                                    "-".join(map(str, controller.choices)) if controller else ""
                                ),
                                **{
                                    f"stat_{name}": float(value)
                                    for name, value in zip(STAT_NAMES, stats.counts, strict=True)
                                },
                            }
                        )
                    finally:
                        env.close()
    finally:
        for module, training in modes:
            module.training = training
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
    strategy: str | None = None,
) -> PolicyEvaluation:
    with isolated_rng():
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if data.get("config") is None:
            raise ValueError("checkpoint has no experiment config")
        config = from_dict(ExperimentConfig, data["config"])
        policy = build_policy(config.model)
        policy.load_state_dict(data["policy"])
        library = (
            StrategyLibrary(data["strategy_library"]) if data.get("strategy_library") else None
        )
        return evaluate_policy(
            policy,
            config,
            opponent=opponent,
            boards=boards,
            seed=seed,
            deterministic=deterministic,
            strategy=strategy,
            library=library,
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
