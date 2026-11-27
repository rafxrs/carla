"""Strategy-conditioned rollouts, sequence PPO, UPGO, strategy BC and training runs."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from carla.agents import make_agent
from carla.agents.policy_agent import PolicyAgent
from carla.config import load_config
from carla.envs import CatanEnv
from carla.envs.strategy import SELECT_KEY, Z_DIM, StrategyLibrary
from carla.eval.behaviour import play_tracked
from carla.models import build_policy
from carla.rl.acting import selection_log_prob
from carla.rl.gae import compute_upgo
from carla.rl.ppo import PPOLearner, _concat, _select, _tensor, head_kl
from carla.rl.rollout import collect_rollouts, reset_envs
from carla.rl.strategies import StrategyAssigner
from carla.rl.strategy_imitation import strategy_behaviour_clone
from carla.rl.train import train


def smoke(tmp_path, **ppo):
    cfg = load_config("configs/alphacat_smoke.yaml")
    return replace(
        cfg,
        engine=replace(cfg.engine, max_turns=6),
        ppo=replace(cfg.ppo, **ppo),
        runtime=replace(cfg.runtime, output_dir=str(tmp_path), checkpoint_every=1),
        strategy=replace(cfg.strategy, reconsider_turns=1, pseudo_reward_prob=1.0),
        league=replace(cfg.league, init_checkpoint=None),
    )


def _library(cfg):
    return StrategyLibrary(
        {
            a: np.random.default_rng(i).random((3, Z_DIM)).astype(np.float32)
            for i, a in enumerate(cfg.strategy.archetypes)
        }
    )


def test_upgo_follows_good_actions_and_bootstraps_after_bad_ones():
    rewards = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    values = np.asarray([0.5, 0.9, 0.2], dtype=np.float32)
    dones = np.asarray([0, 0, 1], dtype=np.float32)
    out = compute_upgo(rewards, values, dones, 0.0, 1.0)
    assert out[2] == 1.0
    # Q(s1, a1) = r1 + V(s2) = 0.2 < V(s1) = 0.9: bootstrap with V(s1) at t = 0.
    assert out[1] == pytest.approx(1.0) and out[0] == pytest.approx(0.9)
    good = compute_upgo(rewards, np.asarray([0.5, 0.1, 0.2], np.float32), dones, 0.0, 1.0)
    assert good[0] == pytest.approx(1.0)


def test_sequence_ppo_recomputes_the_rollout_log_probabilities_exactly(tmp_path):
    cfg = smoke(tmp_path)
    torch.manual_seed(0)
    policy = build_policy(cfg.model)
    envs = [
        CatanEnv(cfg.engine, cfg.env, [make_agent("heuristic_city") for _ in range(3)])
        for _ in range(3)
    ]
    strategies = StrategyAssigner(cfg.strategy, _library(cfg), 0)
    reset_envs(envs, policy, None, [1, 2, 3], strategies)
    batches, _ = collect_rollouts(envs, policy, 40, strategies=strategies)
    assert any(b.dones.any() for b in batches), "episodes must end inside the rollout"
    learner = PPOLearner(policy, cfg.ppo, strategy_coef=0.5)
    assert learner.sequence_mode
    obs = _tensor(_concat([b.observations for b in batches]), torch.device("cpu"))
    assert float(obs[SELECT_KEY].sum()) > 3
    actions = torch.as_tensor(np.concatenate([b.actions for b in batches]))
    old = torch.as_tensor(np.concatenate([b.log_probs for b in batches]))
    policy.eval()
    seen = torch.zeros(len(old), dtype=torch.bool)
    with torch.no_grad():
        for group in learner._minibatches(learner._units(batches)):
            for rows, (starts, resets, steps) in group:
                mini = _select(obs, rows)
                out = policy.unroll(mini, obs["belief_state"][starts].float(), resets, steps)
                dist = out.distribution(mini["masks"])
                logp = dist.log_prob(actions[rows]) + selection_log_prob(out, mini)
                torch.testing.assert_close(logp, old[rows], atol=2e-4, rtol=0)
                seen[rows] = True
    assert seen.all()
    stats = learner.update(batches)
    assert np.isfinite(stats["policy_loss"]) and stats["selections"] > 0
    assert "strategy_value_loss" in stats and "upgo_loss" in stats and "selector_entropy" in stats


def test_head_kl_is_zero_against_itself_and_positive_otherwise(tmp_path):
    cfg = smoke(tmp_path)
    policy = build_policy(cfg.model).eval()
    env = CatanEnv(cfg.engine, cfg.env, [make_agent("random") for _ in range(3)])
    strategies = StrategyAssigner(cfg.strategy, _library(cfg), 0)
    reset_envs([env], policy, None, [0], strategies)
    batches, _ = collect_rollouts([env], policy, 8, strategies=strategies)
    obs = _tensor(batches[0].observations, torch.device("cpu"))
    with torch.no_grad():
        out = policy(obs)
        dist = out.distribution(obs["masks"])
        actions = torch.as_tensor(batches[0].actions)
        same = head_kl({k: v.clone() for k, v in dist.logp.items()}, dist, actions)
        torch.manual_seed(9)
        other = build_policy(cfg.model).eval()(obs).distribution(obs["masks"])
        different = head_kl(other.logp, dist, actions)
    assert torch.allclose(same, torch.zeros_like(same), atol=1e-6)
    assert (different >= -1e-6).all() and different.sum() > 0


def test_strategy_bc_writes_a_library_and_plays_every_variant(tmp_path):
    cfg = smoke(tmp_path)
    path = strategy_behaviour_clone(cfg, tmp_path / "bc", games=3, epochs=1, eval_boards=0)
    data = torch.load(path, weights_only=False)
    assert set(data["strategy_library"]) == set(cfg.strategy.archetypes)
    assert (tmp_path / "bc" / "strategy_library.npz").exists()
    for variant in ("city", "none", "select"):
        agent = PolicyAgent(f"{path}@{variant}")
        assert agent.variant == variant
        opponents = [make_agent("heuristic") for _ in range(3)]
        game, _ = play_tracked([agent, *opponents], seed=3, engine=cfg.engine)
        assert game.is_over and agent.fallbacks == 0
        if variant == "select":
            assert agent.strategy.choices


def test_strategy_training_runs_and_resumes(tmp_path):
    cfg = smoke(tmp_path)
    bc = strategy_behaviour_clone(cfg, tmp_path / "bc", games=3, epochs=1, eval_boards=0)
    cfg = replace(cfg, ppo=replace(cfg.ppo, total_env_steps=128))
    run = train(cfg, init_from=bc)
    data = torch.load(run / "latest.pt", weights_only=False)
    assert data["strategy_library"] and data["counters"]["env_steps"] == 128
    longer = replace(cfg, ppo=replace(cfg.ppo, total_env_steps=192))
    train(longer, resume=run / "latest.pt")
    assert torch.load(run / "latest.pt", weights_only=False)["counters"]["env_steps"] == 192


def test_envs_pickled_before_strategy_statistics_still_resume():
    import pickle

    env = CatanEnv(opponents=[make_agent("random") for _ in range(3)])
    env.reset(seed=0)
    del env.__dict__["strategy"]
    for name in ("stats", "last_action_types"):
        del env.aec.__dict__[name]
    restored = pickle.loads(pickle.dumps(env))
    assert restored.strategy is None and len(restored.aec.stats) == 4
    restored.step(restored.action_space.sample())
