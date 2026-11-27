"""League opponents: policy-played seats batched across envs, snapshots and fixed checkpoints."""

from collections import Counter
from dataclasses import replace

import pytest
import torch

from carla.agents import HeuristicAgent
from carla.config import ModelConfig, OpponentPoolConfig, OpponentSpec
from carla.config.schema import EnvConfig
from carla.engine.config import EngineConfig
from carla.envs import CatanEnv
from carla.envs.catan_env import ExternalAgent
from carla.eval.analysis import read_csv
from carla.models import build_policy
from carla.rl.checkpoint import save_checkpoint
from carla.rl.opponents import League, is_league, load_opponent
from carla.rl.rollout import collect_rollouts, reset_envs, run_env_steps
from carla.rl.train import train
from tests.rl.test_experiments import tiny_config, weights


def _gnn_belief(seed: int):
    torch.manual_seed(seed)
    return build_policy(
        ModelConfig(encoder="gnn", hidden_dim=16, mlp_layers=1, gnn_layers=1, belief="gru")
    )


def _pool(*specs: OpponentSpec, every: int = 2, keep: int = 2) -> OpponentPoolConfig:
    return OpponentPoolConfig(pool=specs, snapshot_every=every, max_snapshots=keep)


def test_plain_reset_refuses_external_opponents_but_reset_steps_yields_their_seats():
    opponents = [ExternalAgent("self", "self"), HeuristicAgent(), HeuristicAgent()]
    # The learner sits last, so the external seat 0 places the first settlement.
    env = CatanEnv(EngineConfig(max_turns=2), EnvConfig(learner_seat=3), opponents=opponents)
    with pytest.raises(RuntimeError, match="ExternalAgent"):
        env.reset(seed=0)
    steps = env.reset_steps(seed=0)
    assert next(steps) == 0 and env.seat_agent(0) is opponents[0]


def test_external_seats_play_whole_games_with_their_own_recurrent_state():
    policy = _gnn_belief(0)
    envs = [
        CatanEnv(
            EngineConfig(max_turns=3),
            opponents=[ExternalAgent("self", "self"), HeuristicAgent(), ExternalAgent("s", "self")],
        )
        for _ in range(3)
    ]
    observations = reset_envs(envs, policy, seeds=[1, 2, 3])
    assert len(observations) == 3
    finished = 0
    for _ in range(200):
        heads = [env.action_space.sample() for env in envs]
        results = run_env_steps(
            envs, [env.step_steps(h) for env, h in zip(envs, heads, strict=True)], policy
        )
        for env, (_, _, terminated, truncated, _) in zip(envs, results, strict=True):
            if terminated or truncated:
                finished += 1
                run_env_steps([env], [env.reset_steps()], policy)
    assert finished > 0
    externals = [a for env in envs for a in env.opponents if isinstance(a, ExternalAgent)]
    assert any(agent.state is not None for agent in externals)


def test_league_samples_by_weight_and_keeps_the_latest_snapshots():
    league = League(
        _pool(OpponentSpec("self", 3), OpponentSpec("snapshot", 1), OpponentSpec("heuristic", 2)),
        EngineConfig(),
        EnvConfig(),
        seed=0,
    )
    assert is_league(league.config)
    league.begin_rollout()
    names = Counter(agent.name for _ in range(600) for agent in league.sample())
    assert set(names) == {"self", "heuristic"}  # No snapshot yet: those seats play "self".
    assert 0.6 < names["self"] / 1800 < 0.73 and 0.27 < names["heuristic"] / 1800 < 0.4
    policy = _gnn_belief(0)
    for version in range(7):
        league.add_snapshot(policy, version)
    assert list(league.snapshots) == [4, 6]  # Every second version, at most two kept.
    league.begin_rollout()
    assert league.active in (4, 6)
    snapshot = next(a for _ in range(50) for a in league.sample() if a.name.startswith("snap"))
    assert isinstance(snapshot, ExternalAgent)
    assert league.network(snapshot.key, policy) is league.snapshots[league.active]
    assert league.network("snapshot:0", policy) is policy  # Dropped snapshots fall back.
    assert league.network("self", policy) is policy
    assert not league.snapshots[4].training
    assert all(not p.requires_grad for p in league.snapshots[4].parameters())


def test_checkpoint_opponents_must_share_the_engine_rules(tmp_path):
    cfg = tiny_config(tmp_path)
    path = save_checkpoint(tmp_path / "opponent.pt", build_policy(cfg.model), config=cfg)
    assert not load_opponent(str(path), cfg.engine, cfg.env).training
    with pytest.raises(ValueError, match="engine rules"):
        load_opponent(str(path), replace(cfg.engine, vp_to_win=8), cfg.env)
    league = League(
        _pool(OpponentSpec("checkpoint", path=str(path), name="old")),
        cfg.engine,
        cfg.env,
        seed=0,
    )
    assert {agent.name for agent in league.sample()} == {"old"}


def test_league_rollouts_record_sampled_opponents():
    policy = _gnn_belief(0)
    league = League(
        _pool(OpponentSpec("self"), OpponentSpec("snapshot"), OpponentSpec("heuristic")),
        EngineConfig(max_turns=2),
        EnvConfig(),
        seed=0,
    )
    league.add_snapshot(policy, 0)
    envs = [CatanEnv(EngineConfig(max_turns=2), opponents=league.sample()) for _ in range(2)]
    from carla.rl.metrics import EpisodeTracker

    records: list[dict] = []
    batches, _ = collect_rollouts(
        envs,
        policy,
        64,
        episode_seeds=[1, 2],
        trackers=[EpisodeTracker(), EpisodeTracker()],
        episode_records=records,
        league=league,
    )
    assert all(len(batch) == 64 for batch in batches)
    seen = {name for record in records for name in record["opponent"].split(",")}
    assert records and seen <= {"self", "snapshot:0", "heuristic"} and len(seen) > 1


@pytest.mark.parametrize("workers", [1, 2])
def test_league_training_resumes_exactly(tmp_path, workers):
    first = tiny_config(tmp_path / "opponent")
    opponent = train(first) / "latest.pt"
    cfg = tiny_config(tmp_path / "league", steps=16)
    cfg = replace(
        cfg,
        opponents=_pool(
            OpponentSpec("self"),
            OpponentSpec("checkpoint", path=str(opponent), name="old"),
            OpponentSpec("heuristic"),
        ),
        runtime=replace(cfg.runtime, num_workers=workers),
    )
    run = train(cfg)
    extended = replace(cfg, ppo=replace(cfg.ppo, total_env_steps=32))
    train(extended, resume=run / "latest.pt")
    reference = train(
        replace(extended, runtime=replace(cfg.runtime, output_dir=str(tmp_path / "reference")))
    )
    assert all(
        torch.equal(value, weights(reference / "latest.pt")[name])
        for name, value in weights(run / "latest.pt").items()
    )
    episodes = read_csv(run / "episodes.csv")
    assert episodes == read_csv(reference / "episodes.csv")
    assert {name for row in episodes for name in row["opponent"].split(",")} <= {
        "self",
        "old",
        "heuristic",
    }


def test_league_training_with_snapshots_runs(tmp_path):
    cfg = tiny_config(tmp_path, steps=32)
    cfg = replace(
        cfg,
        opponents=_pool(OpponentSpec("snapshot"), OpponentSpec("random"), every=1),
        runtime=replace(cfg.runtime, num_workers=2),
    )
    episodes = read_csv(train(cfg) / "episodes.csv")
    names = {name for row in episodes for name in row["opponent"].split(",")}
    assert any(name.startswith("snapshot:") for name in names)
