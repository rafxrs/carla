import numpy as np
import pytest
import torch

from carla.config.schema import ModelConfig, OpponentPoolConfig, OpponentSpec, PPOConfig
from carla.engine.config import EngineConfig
from carla.envs import CatanEnv
from carla.envs.observation import EVENT_FEATURES
from carla.models import batch_observations, build_policy
from carla.models.base import STATE_KEY, TARGET_KEY, with_state
from carla.models.belief import BeliefModule
from carla.models.gnn import GNNPolicy
from carla.models.set_attention import SetAttentionPool
from carla.rl.opponents import bot_tables
from carla.rl.ppo import PPOLearner
from carla.rl.rollout import collect_rollouts

FULL = ModelConfig(
    encoder="gnn",
    hidden_dim=32,
    gnn_layers=2,
    set_attention=True,
    belief="gru",
    belief_dim=16,
    trade_decoder="gru",
)


def test_set_attention_pool_is_permutation_invariant_and_ignores_padding():
    torch.manual_seed(0)
    pool = SetAttentionPool(8, heads=2).eval()
    elements = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, True, False], [False, False, False, False]])
    pooled = pool(elements, mask)
    assert pooled.shape == (2, 8) and torch.isfinite(pooled).all()
    assert torch.equal(pooled[1], torch.zeros(8))
    permuted = pool(elements[:, [2, 0, 1, 3]], mask[:, [2, 0, 1, 3]])
    torch.testing.assert_close(pooled, permuted)
    changed = elements.clone()
    changed[0, 3] = 100.0
    torch.testing.assert_close(pool(changed, mask)[0], pooled[0])


def test_belief_module_keeps_state_without_events_and_folds_events():
    torch.manual_seed(0)
    belief = BeliefModule(EVENT_FEATURES, 16)
    state = torch.randn(2, 16)
    events = torch.randn(2, 5, EVENT_FEATURES)
    mask = torch.tensor([[False] * 5, [True, True, False, False, False]])
    features, predictions, new_state = belief(events, mask, state)
    assert torch.equal(new_state[0], state[0]) and not torch.equal(new_state[1], state[1])
    assert features.shape == (2, 16) and predictions["resources"].shape == (2, 3, 5)
    # Padding after the last valid event cannot change the result.
    events[1, 2:] = 50.0
    torch.testing.assert_close(belief(events, mask, state)[2], new_state)


def test_full_model_state_round_trips_through_observations():
    torch.manual_seed(0)
    policy = build_policy(FULL).eval()
    initial = policy.initial_state(3)
    assert policy.recurrent and initial is not None and initial.shape == (3, 16)
    obs, _ = CatanEnv(EngineConfig(allow_domestic_trade=True)).reset(seed=0)
    stored = with_state(policy, obs, torch.randn(16))
    batch = batch_observations([stored])
    from_obs = policy(batch)
    explicit = policy(batch, torch.as_tensor(stored[STATE_KEY])[None])
    torch.testing.assert_close(from_obs.value, explicit.value)
    state = from_obs.state
    assert state is not None and state.shape == (1, 16)
    assert with_state(build_policy(ModelConfig()), obs) is obs


def test_full_model_rollout_and_ppo_update_train_the_belief():
    torch.manual_seed(1)
    policy = build_policy(FULL)
    assert isinstance(policy, GNNPolicy) and policy.belief is not None
    belief = policy.belief
    envs = [CatanEnv(EngineConfig(max_turns=20, allow_domestic_trade=True)) for _ in range(2)]
    batches, current = collect_rollouts(envs, policy, 24, episode_seeds=[0, 1])
    assert all(STATE_KEY in obs and TARGET_KEY in obs for obs in current)
    states = batches[0].observations[STATE_KEY]
    assert states.shape == (24, 16) and not np.allclose(states[1:], 0)
    # Every episode start resets the recurrent state.
    starts = np.flatnonzero(batches[0].dones[:-1]) + 1
    assert all(np.allclose(states[t], 0) for t in starts)
    assert batches[0].observations[TARGET_KEY]["resources"].shape == (24, 3, 5)
    learner = PPOLearner(policy, PPOConfig(epochs=1, minibatch_size=16))
    belief_before = [p.detach().clone() for p in belief.parameters()]
    stats = learner.update(batches)
    assert stats["auxiliary_loss"] > 0 and all(np.isfinite(v) for v in stats.values())
    assert any(
        not torch.equal(old, new)
        for old, new in zip(belief_before, belief.parameters(), strict=True)
    )


def test_bot_tables_mix_kinds_in_proportion():
    pool = OpponentPoolConfig(pool=(OpponentSpec("random", 2), OpponentSpec("heuristic", 1)))
    tables = bot_tables(pool, 2, 3)
    assert len(tables) == 2 and all(len(worker) == 3 for worker in tables)
    flat = [kind for worker in tables for table in worker for kind in table]
    assert flat.count("random") == 12 and flat.count("heuristic") == 6
    assert all(sorted(table) == ["heuristic", "random", "random"] for w in tables for table in w)
    single = bot_tables(OpponentPoolConfig(pool=(OpponentSpec("heuristic"),)), 1, 2)
    assert single == [[["heuristic"] * 3] * 2]
    with pytest.raises(ValueError, match="registered bot kinds with positive weight"):
        bot_tables(OpponentPoolConfig(pool=(OpponentSpec("self"),)), 1, 1)
