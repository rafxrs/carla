"""Strategy network: shapes, sequence unrolls, information boundaries, selection."""

import numpy as np
import pytest
import torch

from carla.agents import make_agent
from carla.config.schema import EnvConfig, ModelConfig, StrategyConfig
from carla.envs import CatanEnv
from carla.envs.strategy import (
    CHOICE_KEY,
    SELECT,
    SELECT_KEY,
    STRATEGY_KEY,
    Z_DIM,
    StrategyLibrary,
    make_controller,
    strategy_observation,
)
from carla.models import build_policy
from carla.models.base import TARGET_KEY, batch_observations, slice_tree, with_state
from carla.rl.acting import decide, selection_log_prob

CONFIG = ModelConfig(
    encoder="strategy",
    hidden_dim=32,
    gnn_layers=1,
    attention_heads=2,
    seat_attention_layers=1,
    belief="gru",
    belief_dim=8,
    core="lstm",
    core_dim=16,
    core_layers=2,
    strategy_conditioning=True,
    strategy_dim=8,
    strategy_selector=True,
    num_strategies=3,
    privileged_value=True,
    trade_decoder="gru",
)


def _env(seed=0):
    env = CatanEnv(
        env_config=EnvConfig(strategic_features=True),
        opponents=[make_agent("heuristic") for _ in range(3)],
    )
    obs, _ = env.reset(seed=seed)
    return env, obs


def _obs(policy, env, obs):
    obs = {**obs, TARGET_KEY: env.aec.hidden_truth(env.learner)}
    return strategy_observation(with_state(policy, obs), None, False)


def test_state_layout_and_outputs():
    torch.manual_seed(0)
    policy = build_policy(CONFIG)
    assert policy.state_dim == 8 + 2 * 2 * 16 and policy.recurrent and policy.uses_strategy
    env, obs = _env()
    batch = batch_observations([_obs(policy, env, obs)] * 3)
    out = policy(batch)
    assert out.strategy_logits.shape == (3, 4) and out.strategy_value.shape == (3,)
    assert out.state.shape == (3, policy.state_dim)
    actions = out.distribution(batch["masks"]).sample()
    assert torch.isfinite(out.distribution(batch["masks"]).log_prob(actions)).all()


def test_unroll_matches_stepwise_forward_with_resets():
    torch.manual_seed(1)
    policy = build_policy(CONFIG).eval()
    env, obs = _env()
    rows = []
    for _ in range(12):
        rows.append(_obs(policy, env, obs))
        obs = env.step(env.action_space.sample())[0]
    steps, width = 4, 3
    batch = batch_observations(rows)
    resets = torch.zeros(steps, width, dtype=torch.bool)
    resets[2, 1] = True
    state0 = torch.randn(width, policy.state_dim) * 0.1
    with torch.no_grad():
        unrolled = policy.unroll(batch, state0, resets, steps)
        state, values = state0, []
        for t in range(steps):
            state = torch.where(resets[t][:, None], torch.zeros_like(state), state)
            out = policy(slice_tree(batch, slice(t * width, (t + 1) * width)), state)
            state = out.state
            values.append(out.value)
    torch.testing.assert_close(torch.cat(values), unrolled.value)
    torch.testing.assert_close(state, unrolled.state)


def test_privileged_cards_reach_the_critic_but_never_the_policy():
    torch.manual_seed(2)
    policy = build_policy(CONFIG).eval()
    env, obs = _env()
    row = _obs(policy, env, obs)
    other = {**row, TARGET_KEY: {k: v + 3 for k, v in row[TARGET_KEY].items()}}
    with torch.no_grad():
        a = policy(batch_observations([row]))
        b = policy(batch_observations([other]))
    for key in a.logits:
        torch.testing.assert_close(a.logits[key], b.logits[key])
    torch.testing.assert_close(a.strategy_logits, b.strategy_logits)
    assert not torch.allclose(a.value, b.value)


def test_z_conditions_the_heads_but_not_the_selector():
    torch.manual_seed(3)
    policy = build_policy(CONFIG).eval()
    env, obs = _env()
    row = _obs(policy, env, obs)
    other = {**row, STRATEGY_KEY: np.ones(Z_DIM, dtype=np.float32)}
    with torch.no_grad():
        a = policy(batch_observations([row]))
        b = policy(batch_observations([other]))
    torch.testing.assert_close(a.strategy_logits, b.strategy_logits)
    assert not torch.allclose(a.logits["type"], b.logits["type"])
    assert not torch.allclose(a.trade_context, b.trade_context)
    assert not torch.allclose(a.strategy_value, b.strategy_value)


def test_decide_selects_then_acts_under_the_chosen_z_and_scores_both():
    torch.manual_seed(4)
    policy = build_policy(CONFIG).eval()
    config = StrategyConfig()
    library = StrategyLibrary(
        {a: np.full((2, Z_DIM), i + 1, np.float32) for i, a in enumerate(config.archetypes)}
    )
    env, obs = _env()
    controller = make_controller(config, SELECT, library, np.random.default_rng(0))
    row = with_state(policy, {**obs, TARGET_KEY: env.aec.hidden_truth(env.learner)})
    decision = decide(
        policy, [row], [controller], [env.game], [env.learner], [env.aec.stats[env.learner]]
    )
    stored = decision.observations[0]
    assert stored[SELECT_KEY] == 1 and stored[CHOICE_KEY] == controller.choices[-1]
    np.testing.assert_array_equal(stored[STRATEGY_KEY], controller.z)
    with torch.no_grad():
        batch = batch_observations([stored])
        out = policy(batch)
        dist = out.distribution(batch["masks"])
        actions = torch.as_tensor(decision.heads)
        recomputed = dist.log_prob(actions) + selection_log_prob(out, batch)
    assert recomputed.item() == pytest.approx(float(decision.log_probs[0]), abs=1e-5)
    assert selection_log_prob(out, batch).item() < 0
    # The next decision of the same turn is not a selection point.
    assert not controller.needs_selection(env.game, env.learner) or env.game.turn > 0
