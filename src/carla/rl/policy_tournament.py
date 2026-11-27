"""Paired four-seat tournaments for trained checkpoints and fixed bots."""

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from carla.agents import make_agent
from carla.config.schema import ExperimentConfig
from carla.engine.rng import derive_seed
from carla.eval.metrics import wilson_interval
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
    from carla.agents.policy_agent import PolicyAgent
    from carla.agents.registry import is_agent_name
    from carla.eval.runner import play_game

    config = None
    names = []
    for player in players:
        name = str(player)
        if not is_agent_name(name):
            name = "ppo:" + name
        names.append(name)
        if name.startswith("ppo:"):
            agent = PolicyAgent(name[4:], deterministic=deterministic)
            current = agent.config
            if config is not None and (
                current.engine != config.engine or current.env != config.env
            ):
                raise ValueError("tournament checkpoints must use the same engine and environment")
            config = current
    if config is None:
        from carla.engine.config import EngineConfig

        config = ExperimentConfig(engine=EngineConfig(max_turns=500))
    records = []
    with isolated_rng(seed):
        for board in range(boards):
            game_seed = derive_seed(seed, "evaluation-board", board)
            for rotation in range(4):
                seats = [(slot - rotation) % 4 for slot in range(4)]
                agents = [None] * 4
                for slot, name in enumerate(names):
                    agent = make_agent(name)
                    if isinstance(agent, PolicyAgent):
                        agent.deterministic = deterministic
                    agents[seats[slot]] = agent
                result = play_game(agents, game_seed, config.engine)
                for slot, player in enumerate(players):
                    seat = seats[slot]
                    outcome = (
                        "truncated"
                        if result.winner is None
                        else "win" if result.winner == seat else "loss"
                    )
                    reward = (
                        config.env.truncation_reward
                        if result.winner is None
                        else (
                            config.env.win_reward
                            if result.winner == seat
                            else config.env.loss_reward
                        )
                    )
                    reward += config.env.vp_reward * result.public_victory_points[seat]
                    records.append(
                        {
                            "board": board,
                            "game_seed": game_seed,
                            "rotation": rotation,
                            "slot": slot,
                            "player": str(player),
                            "seat": seat,
                            "outcome": outcome,
                            "winner": -1 if result.winner is None else result.winner,
                            "return": reward,
                            "decisions": result.decisions[seat],
                            "turns": result.turns,
                            "victory_points": result.victory_points[seat],
                            "trade_offers": result.trade_offers[seat],
                            "multi_card_offers": result.multi_card_offers[seat],
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
