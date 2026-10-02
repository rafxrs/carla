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
    from carla.engine.config import EngineConfig
    from carla.envs import CatanEnv
    from carla.models import build_policy
    from carla.rl.checkpoint import load_checkpoint, save_checkpoint
    from carla.rl.rollout import collect_rollout

    torch.manual_seed(0)
    env = CatanEnv(EngineConfig(max_turns=30))
    policy = build_policy(ModelConfig(hidden_dim=32, mlp_layers=1))
    batch, _ = collect_rollout(env, policy, num_steps=40, episode_seed=3)
    assert len(batch) == 40 and batch.observations["masks"]["type"].shape == (40, 15)
    assert batch.dones.sum() >= 1
    type_masks = batch.observations["masks"]["type"]
    assert all(type_masks[t][batch.actions[t][0]] for t in range(len(batch)))

    path = save_checkpoint(tmp_path / "c.pt", policy, config=ExperimentConfig(), env_steps=40)
    clone = build_policy(ModelConfig(hidden_dim=32, mlp_layers=1))
    meta = load_checkpoint(path, clone)
    assert meta["counters"] == {"env_steps": 40} and meta["config"]["name"] == "default"
    for a, b in zip(policy.parameters(), clone.parameters(), strict=True):
        assert torch.equal(a, b)
