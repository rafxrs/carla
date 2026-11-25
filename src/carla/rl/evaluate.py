"""Frozen-policy evaluation on held-out boards with individual game records."""

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from carla.agents import AGENTS, make_agent
from carla.config.loader import from_dict
from carla.config.schema import ExperimentConfig
from carla.engine.rng import derive_seed
from carla.envs.catan_env import CatanEnv
from carla.eval.diagnostics import GameplayDiagnostics, diagnose_game, summarize_diagnostics
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
    diagnostics: GameplayDiagnostics | None = None


MIXED_EVALUATION_FIELDS = ("opponent_composition", "opponent_seats")


def _opponent_spec(opponent: str | None, opponents: Sequence[str] | None):
    if opponent is not None and opponents is not None:
        raise ValueError("choose opponent or opponents, not both")
    kinds = (
        (("heuristic" if opponent is None else opponent),) * 3
        if opponents is None
        else tuple(opponents)
    )
    if isinstance(opponents, str) or len(kinds) != 3:
        raise ValueError("mixed evaluation requires exactly three opponents")
    if any(not isinstance(kind, str) or kind not in AGENTS for kind in kinds):
        raise ValueError(f"evaluation needs registered opponents; known: {sorted(AGENTS)}")
    identity = (
        kinds[0] if len(set(kinds)) == 1 else "mixed:" + json.dumps(kinds, separators=(",", ":"))
    )
    return kinds, identity


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
    opponent: str | None = None,
    opponents: Sequence[str] | None = None,
    boards: int = 10,
    seed: int = 10_000,
    deterministic: bool = True,
    batch_games: int = 64,
    diagnostics: bool = False,
    trajectories: bool = False,
    checkpoint_id: str | None = None,
) -> PolicyEvaluation:
    """Batch active games, playing each held-out board in all four learner seats.

    Environments step sequentially; inference uses the policy's device. RNG and module modes
    are restored, and records retain board/seat order regardless of game completion order.
    """
    kinds, opponent = _opponent_spec(opponent, opponents)
    if boards <= 0:
        raise ValueError("evaluation needs positive boards")
    if trajectories and not diagnostics:
        raise ValueError("trajectories require diagnostics")
    if batch_games <= 0:
        raise ValueError("batch_games must be positive")
    if config.model.encoder not in ("mlp", "gnn"):
        raise ValueError("this evaluator supports MLP/GNN policies only")
    modes = [(module, module.training) for module in policy.modules()]
    device = next(policy.parameters()).device
    records = []
    diagnostic_result = GameplayDiagnostics() if diagnostics else None
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
                            [make_agent(kind) for kind in kinds],
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
                            if opponents is not None or diagnostic_result is not None:
                                seat_agents = dict(
                                    zip((p for p in range(4) if p != seat), kinds, strict=True)
                                )
                                composition = json.dumps(kinds, separators=(",", ":"))
                                seats = json.dumps(
                                    ["learner" if p == seat else seat_agents[p] for p in range(4)],
                                    separators=(",", ":"),
                                )
                            if opponents is not None:
                                records[-1].update(
                                    opponent_composition=composition, opponent_seats=seats
                                )
                            if diagnostic_result is not None:
                                detail = diagnose_game(env.game, trajectories=trajectories)
                                common = {
                                    "checkpoint": checkpoint_id,
                                    "eval_seed": seed,
                                    "board": game.board,
                                    "game_seed": env.game.seed,
                                    "learner_seat": seat,
                                    "opponent": opponent,
                                    "opponent_composition": composition,
                                    "opponent_seats": seats,
                                    "mode": records[-1]["mode"],
                                    "winner": records[-1]["winner"],
                                    "game_turns": env.game.turn,
                                    "game_steps": env.game.num_steps,
                                }
                                for rows, destination in (
                                    (detail.players, diagnostic_result.players),
                                    (detail.trades, diagnostic_result.trades),
                                    (detail.offers, diagnostic_result.offers),
                                    (detail.trajectories, diagnostic_result.trajectories),
                                ):
                                    for row in rows:
                                        player = row["player_seat"]
                                        row.update(common)
                                        row.update(
                                            role="learner" if player == seat else "opponent",
                                            agent_type=(
                                                "learner" if player == seat else seat_agents[player]
                                            ),
                                            outcome=(
                                                "truncated"
                                                if truncated
                                                else ("win" if player == winner else "loss")
                                            ),
                                        )
                                        destination.append(row)
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
    if diagnostic_result is not None:
        for rows in (
            diagnostic_result.players,
            diagnostic_result.trades,
            diagnostic_result.offers,
            diagnostic_result.trajectories,
        ):
            rows.sort(
                key=lambda row: (
                    row["board"],
                    row["learner_seat"],
                    row.get("turn", -1),
                    row["player_seat"],
                )
            )
        diagnostic_result.summary = summarize_diagnostics(diagnostic_result.players)
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
        diagnostic_result,
    )


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    opponent: str | None = None,
    opponents: Sequence[str] | None = None,
    boards: int = 10,
    seed: int = 10_000,
    deterministic: bool = True,
    batch_games: int = 64,
    device: str = "auto",
    diagnostics: bool = False,
    trajectories: bool = False,
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
            opponents=opponents,
            boards=boards,
            seed=seed,
            deterministic=deterministic,
            batch_games=batch_games,
            diagnostics=diagnostics,
            trajectories=trajectories,
            checkpoint_id=str(Path(checkpoint).resolve()),
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
    fields = EVALUATION_FIELDS + (
        MIXED_EVALUATION_FIELDS if any("opponent_composition" in r for r in result.records) else ()
    )
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames == list(EVALUATION_FIELDS + MIXED_EVALUATION_FIELDS):
                fields = EVALUATION_FIELDS + MIXED_EVALUATION_FIELDS
            elif fields != EVALUATION_FIELDS and reader.fieldnames == list(EVALUATION_FIELDS):
                raise ValueError(
                    "mixed evaluation needs a separate CSV; existing homogeneous exports are preserved"
                )
            existing = list(reader)
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
    journal = CsvJournal(path, fields, len(existing))
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
