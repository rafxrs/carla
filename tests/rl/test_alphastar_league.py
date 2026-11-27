"""League core (payoff, PFSP, matchmaking, snapshots, pruning) and an end-to-end league run."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from carla.config import load_config
from carla.config.schema import LeagueConfig
from carla.rl.league.core import (
    LEAGUE_EXPLOITER,
    MAIN,
    MAIN_EXPLOITER,
    LeaguePlayer,
    Matchmaker,
    Payoff,
    final_scores,
    pairwise_result,
    pfsp,
    prune_candidates,
    ready_to_checkpoint,
    remove_monotonic_suffix,
)
from carla.rl.league.trainer import train_league
from carla.rl.league.worker import TableLeague
from carla.rl.strategy_imitation import strategy_behaviour_clone


def test_pfsp_weightings():
    rates = np.asarray([0.1, 0.5, 0.9, 1.0])
    hard = pfsp(rates, "squared")
    assert hard[0] > hard[1] > hard[2] and hard[3] == 0 and hard.sum() == pytest.approx(1)
    var = pfsp(rates, "variance")
    assert var[1] == var.max() and var[3] == 0
    assert pfsp(np.ones(3), "squared").tolist() == pytest.approx([1 / 3] * 3)
    capped = pfsp(np.asarray([0.0, 0.4]), "linear_capped")
    assert capped[0] == pytest.approx(capped[1])


def test_forgetting_check_drops_the_improving_suffix():
    assert remove_monotonic_suffix([0.9, 0.6, 0.8, 0.7, 0.5], list("abcde")) == (
        [0.9, 0.6, 0.8],
        ["a", "b", "c"],
    )
    assert remove_monotonic_suffix([0.9, 0.8], ["a", "b"]) == ([], [])


def test_four_player_results_reduce_to_pairwise_comparisons():
    scores = final_scores(2, [7, 9, 10, 7])
    assert pairwise_result(scores[0], scores[2]) == "loss"
    assert pairwise_result(scores[1], scores[0]) == "win"
    assert pairwise_result(scores[0], scores[3]) == "draw"


def test_payoff_decays_and_aggregates_variants():
    payoff = Payoff(decay=0.5)
    assert payoff.win_rate("m", "x") == 0.5 and payoff.table_win_rate("m", "all") == 0.25
    payoff.update("m", "h:0@city", "win")
    payoff.update("m", "h:0@none", "loss")
    assert payoff.win_rate("m", "h:0@city") == 1 and payoff.win_rate("m", "h:0@none") == 0
    assert payoff.count("m", "h:0") == pytest.approx(1.5)  # 1 * 0.5 + 1
    assert payoff.win_rate("m", "h:0") == pytest.approx(0.5 / 1.5)
    restored = Payoff.from_state(payoff.state())
    assert restored.win_rate("m", "h:0") == payoff.win_rate("m", "h:0")


def _players():
    players = {
        "main_0": LeaguePlayer("main_0", "learner", role=MAIN, strategy_aware=True, selector=True),
        "main_exploiter_0": LeaguePlayer(
            "main_exploiter_0", "learner", role=MAIN_EXPLOITER, strategy_aware=True, variant="city"
        ),
        "league_exploiter_0": LeaguePlayer(
            "league_exploiter_0", "learner", role=LEAGUE_EXPLOITER, strategy_aware=True
        ),
        "bot:heuristic": LeaguePlayer("bot:heuristic", "bot", protected=True),
    }
    for i in range(3):
        players[f"main_0:{i}"] = LeaguePlayer(
            f"main_0:{i}",
            "historical",
            role=MAIN,
            parent="main_0",
            strategy_aware=True,
            selector=True,
            created_step=i,
        )
    players["main_exploiter_0:0"] = LeaguePlayer(
        "main_exploiter_0:0",
        "historical",
        role=MAIN_EXPLOITER,
        parent="main_exploiter_0",
        strategy_aware=True,
        variant="devcard",
        created_step=5,
    )
    return players


def test_matchmaking_branches_per_role():
    config = LeagueConfig()
    players, payoff = _players(), Payoff()
    maker = Matchmaker(config, np.random.default_rng(0))
    plan = maker.plan(players["main_0"], players, payoff, 300)
    assert {"selfplay", "pfsp"} <= set(plan.counts)
    for table, branch in zip(plan.tables, plan.branches, strict=True):
        assert len(table) == 3
        if branch == "selfplay":
            assert all(pid == "main_0" for pid, _ in table)
        assert all(
            not pid.startswith(("main_exploiter_0", "league")) or ":" in pid for pid, _ in table
        )
    # A historical exploiter the main agent loses to triggers the verification branch.
    for _ in range(20):
        payoff.update("main_0", "main_exploiter_0:0@devcard", "loss")
    plan = maker.plan(players["main_0"], players, payoff, 400)
    assert plan.counts.get("verification", 0) > 0
    exploiter = maker.plan(players["main_exploiter_0"], players, payoff, 50)
    assert set(exploiter.counts) <= {"exploit-main", "exploit-history"}
    assert exploiter.learner_variant == "city"
    league = maker.plan(players["league_exploiter_0"], players, payoff, 50)
    assert all(players[pid].kind != "learner" for t in league.tables for pid, _ in t)
    # Fixed-variant historical players only play their variant.
    assert all(v == "devcard" for t in league.tables for pid, v in t if pid == "main_exploiter_0:0")


def test_snapshot_rules_require_own_wins_for_main_exploiters():
    config = replace(
        LeagueConfig(),
        exploiter_min_steps=10,
        exploiter_max_steps=100,
        main_snapshot_steps=10,
        main_snapshot_max_steps=50,
    )
    players, payoff = _players(), Payoff(decay=1.0)
    exploiter = players["main_exploiter_0"]
    assert not ready_to_checkpoint(exploiter, 5, players, payoff, config)
    for _ in range(10):
        payoff.update("main_exploiter_0", "main_0", "win")
        payoff.update_table("main_exploiter_0", "main", False)
    # Ahead of the main agent in every game, but never winning: kingmaking is not enough.
    assert not ready_to_checkpoint(exploiter, 20, players, payoff, config)
    for _ in range(10):
        payoff.update_table("main_exploiter_0", "main", True)
    assert ready_to_checkpoint(exploiter, 20, players, payoff, config)
    assert ready_to_checkpoint(exploiter, 100, players, Payoff(), config)
    main = players["main_0"]
    assert not ready_to_checkpoint(main, 20, players, Payoff(), config)
    assert ready_to_checkpoint(main, 50, players, Payoff(), config)


def test_pruning_keeps_protected_newest_and_hard_or_distinct_players():
    players, payoff = _players(), Payoff(decay=1.0)
    for _ in range(10):
        payoff.update("main_0", "main_0:0", "win")
        payoff.update("main_0", "main_0:1", "loss")
    behaviour = {"main_0:0": np.zeros(7), "main_0:1": np.zeros(7), "main_0:2": np.zeros(7)}
    dropped = prune_candidates(players, payoff, behaviour, keep=3)
    assert dropped == ["main_0:0"]
    assert "bot:heuristic" not in prune_candidates(players, payoff, behaviour, keep=0)
    assert "main_0:2" not in prune_candidates(players, payoff, behaviour, keep=0)


class _Game:
    def __init__(self, winner, vps):
        self.winner, self._vps, self.is_truncated = winner, vps, False

    def victory_points(self, seat):
        return self._vps[seat]


class _Env:
    def __init__(self, winner, vps):
        from carla.config.schema import EnvConfig

        self.game, self.learner, self.config = _Game(winner, vps), 0, EnvConfig()
        self.agents = {
            s: type("A", (), {"league_key": "main_0@none" if s == 1 else "bot:heuristic"})()
            for s in (1, 2, 3)
        }

    def seat_agent(self, seat):
        return self.agents[seat]


def test_exploiter_objectives_distinguish_own_success_from_main_losing():
    league = TableLeague("main_exploiter_0", {}, np.random.default_rng(0), None)
    league.main_ids = {"main_0"}
    env = _Env(winner=2, vps=[8, 6, 10, 5])  # a bot wins; the exploiter beat the main agent
    assert league.terminal_reward(env, -1.0, True) == -1.0
    league.objective, league.rank_bonus = "win_and_rank", 0.25
    assert league.terminal_reward(env, -1.0, True) == pytest.approx(-0.75)
    league.objective = "main_loses"
    assert league.terminal_reward(env, -1.0, True) == pytest.approx(1.0)
    assert league.terminal_reward(env, 0.3, False) == 0.3


def test_league_runs_snapshots_resets_and_resumes(tmp_path):
    cfg = load_config("configs/alphacat_smoke.yaml")
    cfg = replace(
        cfg,
        engine=replace(cfg.engine, max_turns=4),
        runtime=replace(cfg.runtime, num_workers=1, envs_per_worker=2, output_dir=str(tmp_path)),
        ppo=replace(cfg.ppo, rollout_steps=16, minibatch_size=32),
        eval=replace(cfg.eval, every_env_steps=0),
    )
    bc = strategy_behaviour_clone(cfg, tmp_path / "bc", games=3, epochs=1, eval_boards=0)
    cfg = replace(
        cfg,
        league=replace(
            cfg.league,
            init_checkpoint=str(bc),
            total_env_steps=320,
            updates_per_turn=1,
            main_snapshot_steps=32,
            main_snapshot_max_steps=64,
            exploiter_min_steps=32,
            exploiter_max_steps=32,
            max_historical=2,
            tables_per_plan=8,
        ),
    )
    run = train_league(cfg)
    events = (run / "events.csv").read_text()
    assert "snapshot" in events and "reset,main_exploiter_0" in events and "prune" in events
    for name in ("league_state.pt", "payoff.csv", "roster.json", "learners/main_0.pt"):
        assert (run / name).exists()
    steps = torch.load(run / "league_state.pt", weights_only=False)["total_steps"]
    assert steps >= 320
    train_league(replace(cfg, league=replace(cfg.league, total_env_steps=480)), resume=run)
    assert torch.load(run / "league_state.pt", weights_only=False)["total_steps"] >= 480
