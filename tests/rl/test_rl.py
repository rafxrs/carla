import numpy as np
import pytest

from carla.rl.gae import compute_gae


def test_gae_matches_hand_computation():
    rewards = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    values = np.array([0.5, 0.4, 0.3, 0.2], dtype=np.float32)
    dones = np.array([0, 0, 1, 0], dtype=np.float32)
    adv, ret = compute_gae(rewards, values, dones, last_value=0.1, gamma=0.9, lam=0.8)
    expected = np.zeros(4)
    expected[3] = 0.0 + 0.9 * 0.1 - 0.2
    expected[2] = 1.0 - 0.3  # episode ends: no bootstrap
    expected[1] = (0.0 + 0.9 * 0.3 - 0.4) + 0.9 * 0.8 * expected[2]
    expected[0] = (0.0 + 0.9 * 0.4 - 0.5) + 0.9 * 0.8 * expected[1]
    np.testing.assert_allclose(adv, expected, rtol=1e-6)
    np.testing.assert_allclose(ret, expected + values, rtol=1e-6)


def test_gae_handles_parallel_envs():
    rewards = np.ones((5, 3), dtype=np.float32)
    values = np.zeros((5, 3), dtype=np.float32)
    adv, _ = compute_gae(rewards, values, np.zeros((5, 3)), np.zeros(3), gamma=1.0, lam=1.0)
    np.testing.assert_allclose(adv[:, 0], [5, 4, 3, 2, 1])


def test_rollout_and_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from carla.config import ExperimentConfig, ModelConfig
    from carla.engine.actions import NUM_ACTION_TYPES
    from carla.engine.config import EngineConfig
    from carla.envs import CatanEnv
    from carla.models import build_policy
    from carla.rl.checkpoint import load_checkpoint, save_checkpoint
    from carla.rl.rollout import collect_rollout

    torch.manual_seed(0)
    env = CatanEnv(EngineConfig(max_turns=30))
    policy = build_policy(ModelConfig(hidden_dim=32, mlp_layers=1))
    batch, _ = collect_rollout(env, policy, num_steps=40, episode_seed=3)
    assert len(batch) == 40 and batch.observations["masks"]["type"].shape == (40, NUM_ACTION_TYPES)
    assert batch.dones.sum() >= 1
    type_masks = batch.observations["masks"]["type"]
    assert all(type_masks[t][batch.actions[t][0]] for t in range(len(batch)))

    path = save_checkpoint(tmp_path / "c.pt", policy, config=ExperimentConfig(), env_steps=40)
    clone = build_policy(ModelConfig(hidden_dim=32, mlp_layers=1))
    meta = load_checkpoint(path, clone)
    assert meta["counters"] == {"env_steps": 40} and meta["config"]["name"] == "default"
    for a, b in zip(policy.parameters(), clone.parameters(), strict=True):
        assert torch.equal(a, b)


def test_ppo_update_changes_finite_weights_and_rejects_stale_batch():
    torch = pytest.importorskip("torch")
    from carla.config import ModelConfig, PPOConfig
    from carla.engine.config import EngineConfig
    from carla.envs import CatanEnv
    from carla.models import build_policy
    from carla.rl.ppo import PPOLearner
    from carla.rl.rollout import collect_rollout

    torch.manual_seed(2)
    policy = build_policy(ModelConfig(hidden_dim=16, mlp_layers=1))
    learner = PPOLearner(policy, PPOConfig(epochs=1, minibatch_size=8))
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    batch, _ = collect_rollout(CatanEnv(EngineConfig(max_turns=12)), policy, 8, episode_seed=2)
    stats = learner.update([batch])
    assert learner.version == 1
    assert all(np.isfinite(value) for value in stats.values())
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())
    assert any(
        not torch.equal(old, new) for old, new in zip(before, policy.parameters(), strict=True)
    )
    with pytest.raises(ValueError, match="current policy version"):
        learner.update([batch])


def test_batched_rollouts_use_one_forward_pass_per_step_and_keep_envs_separate(monkeypatch):
    torch = pytest.importorskip("torch")
    from carla.config import ModelConfig, PPOConfig
    from carla.engine.config import EngineConfig
    from carla.envs import CatanEnv
    from carla.models import build_policy
    from carla.rl.ppo import PPOLearner
    from carla.rl.rollout import collect_rollouts

    torch.manual_seed(3)
    policy = build_policy(ModelConfig(hidden_dim=16, mlp_layers=1))
    learner = PPOLearner(policy, PPOConfig(epochs=1, minibatch_size=16))
    envs = [CatanEnv(EngineConfig(max_turns=2)) for _ in range(3)]
    sizes = []
    forward = policy.forward

    def counted(obs, state=None):
        sizes.append(len(obs["globals"]))
        return forward(obs, state)

    monkeypatch.setattr(policy, "forward", counted)
    batches, observations = collect_rollouts(
        envs, policy, 24, episode_seeds=[3, 4, 5], policy_version=learner.version
    )
    assert sizes == [3] * 25  # 24 action selections and one bootstrap evaluation
    assert len(batches) == len(observations) == 3
    assert all(len(batch) == 24 and batch.policy_version == 0 for batch in batches)
    assert sum(int(batch.dones.sum()) for batch in batches) >= 1
    for batch in batches:
        masks = batch.observations["masks"]["type"]
        assert all(masks[t, batch.actions[t, 0]] for t in range(len(batch)))
    stats = learner.update(batches)
    assert all(np.isfinite(value) for value in stats.values())
    assert learner.version == 1
    more, _ = collect_rollouts(envs, policy, 2, observations, policy_version=learner.version)
    assert len(more) == 3 and all(len(batch) == 2 for batch in more)
    with pytest.raises(ValueError, match="observations must match"):
        collect_rollouts(envs, policy, 1, observations[:1])


def test_masked_entropy_has_finite_gradients():
    torch = pytest.importorskip("torch")
    from carla.config import ModelConfig
    from carla.envs import CatanEnv
    from carla.models import build_policy
    from carla.models.base import batch_observations

    policy = build_policy(ModelConfig(hidden_dim=16, mlp_layers=1))
    obs, _ = CatanEnv().reset(seed=7)
    batch = batch_observations([obs])
    policy(batch).distribution(batch["masks"]).entropy().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )
