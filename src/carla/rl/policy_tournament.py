"""Paired four-seat tournaments for trained checkpoints and fixed bots."""

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from carla.agents import make_agent
from carla.config.loader import from_dict
from carla.config.schema import ExperimentConfig
from carla.engine.actions import ActionType
from carla.engine.rng import derive_seed
from carla.envs.action_space import decode_action
from carla.envs.catan_env import CatanAECEnv
from carla.eval.metrics import wilson_interval
from carla.models.base import batch_observations
from carla.models.factory import build_policy
from carla.rl.checkpoint import isolated_rng

TOURNAMENT_FIELDS = (
    "board",
    "game_seed",
    "rotation",
    "slot",
    "player",
    "seat",
    "outcome",
    "winner",
    "return",
    "decisions",
    "turns",
    "victory_points",
    "trade_offers",
    "multi_card_offers",
)


@dataclass(frozen=True)
class TournamentEntry:
    slot: int
    player: str
    games: int
    wins: int
    win_rate: float
    ci_low: float
    ci_high: float
    mean_vp: float


@dataclass(frozen=True)
class PolicyTournament:
    entries: list[TournamentEntry]
    records: list[dict]


@torch.no_grad()
def run_policy_tournament(
    players: list[str | Path],
    *,
    boards: int = 10,
    seed: int = 20_000,
    deterministic: bool = True,
    csv_path: Path | None = None,
) -> PolicyTournament:
    """Rotate four players through every seat on each board.

    Offer counts include initial proposals and counteroffers; multi-card means more
    than one card on either side. Counts describe attempts, not completed exchanges.
    """
    if len(players) != 4 or boards <= 0:
        raise ValueError("tournament needs four players and a positive board count")
    policies = {}
    config = None
    for slot, player in enumerate(players):
        if str(player) in ("random", "heuristic"):
            continue
        path = Path(player)
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("config") is None:
            raise ValueError(f"checkpoint has no experiment config: {path}")
        current = from_dict(ExperimentConfig, data["config"])
        if current.model.encoder not in ("mlp", "gnn") or current.model.belief != "none":
            raise ValueError("policy tournament supports feed-forward MLP/GNN checkpoints only")
        if config is not None and (current.engine != config.engine or current.env != config.env):
            raise ValueError("tournament checkpoints must use the same engine and environment")
        config = current
        policy = build_policy(current.model)
        policy.load_state_dict(data["policy"])
        policy.eval()
        policies[slot] = policy
    if config is None:
        raise ValueError("tournament needs at least one checkpoint")
    records = []
    with isolated_rng(seed):
        for board in range(boards):
            game_seed = derive_seed(seed, "evaluation-board", board)
            for rotation in range(4):
                env = CatanAECEnv(config.engine, config.env)
                env.reset(game_seed)
                seats = [(slot - rotation) % 4 for slot in range(4)]
                slots = {seat: slot for slot, seat in enumerate(seats)}
                bots = {}
                for slot, player in enumerate(players):
                    if slot not in policies:
                        bot = make_agent(str(player))
                        bot.reset(
                            env.game, seats[slot], derive_seed(game_seed, "seat", seats[slot])
                        )
                        bots[slot] = bot
                rewards, decisions = [0.0] * 4, [0] * 4
                trade_offers, multi_card_offers = [0] * 4, [0] * 4
                while not env.game.is_over:
                    seat = env.actor
                    slot = slots[seat]
                    if slot in policies:
                        obs = env.observe(seat)
                        batch = batch_observations([obs])
                        action = (
                            policies[slot](batch)
                            .distribution(batch["masks"])
                            .sample(deterministic=deterministic)[0]
                            .cpu()
                            .numpy()
                        )
                    else:
                        env.observe(seat)  # Clear events for this bot seat.
                        action = bots[slot].act(env.game, seat, env.game.legal_actions())
                    parsed = action if slot not in policies else decode_action(action, seat)
                    if parsed.type == ActionType.PROPOSE_TRADE:
                        assert parsed.trade is not None
                        trade_offers[slot] += 1
                        if max(sum(parsed.trade.give), sum(parsed.trade.receive)) > 1:
                            multi_card_offers[slot] += 1
                    result = env.step(action)
                    for player_slot, player_seat in enumerate(seats):
                        rewards[player_slot] += float(result.rewards[player_seat])
                    decisions[slot] += 1
                winner = env.game.winner
                for slot, player in enumerate(players):
                    seat = seats[slot]
                    records.append(
                        {
                            "board": board,
                            "game_seed": game_seed,
                            "rotation": rotation,
                            "slot": slot,
                            "player": str(player),
                            "seat": seat,
                            "outcome": (
                                "truncated"
                                if env.game.is_truncated
                                else ("win" if winner == seat else "loss")
                            ),
                            "winner": -1 if winner is None else winner,
                            "return": rewards[slot],
                            "decisions": decisions[slot],
                            "turns": env.game.turn,
                            "victory_points": env.game.victory_points(seat),
                            "trade_offers": trade_offers[slot],
                            "multi_card_offers": multi_card_offers[slot],
                        }
                    )
    entries = []
    for slot, player in enumerate(players):
        rows = [record for record in records if record["slot"] == slot]
        wins = sum(record["outcome"] == "win" for record in rows)
        low, high = wilson_interval(wins, len(rows))
        entries.append(
            TournamentEntry(
                slot,
                str(player),
                len(rows),
                wins,
                wins / len(rows),
                low,
                high,
                sum(record["victory_points"] for record in rows) / len(rows),
            )
        )
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, TOURNAMENT_FIELDS)
            writer.writeheader()
            writer.writerows(records)
    return PolicyTournament(entries, records)
