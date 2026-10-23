"""Experience collection.

`collect_rollout` is the synchronous reference (one env, one process). The training setup is
semi-asynchronous PPO (TODO(rl), see `RolloutWorker`): `runtime.num_workers` CPU processes each
run `runtime.envs_per_worker` envs with their own copy of the current policy and of the
opponents, and stream `RolloutBatch`es to a single learner process (GPU when available).
"""

import numpy as np
import torch

from carla.envs.catan_env import CatanEnv
from carla.models.base import PolicyNetwork, batch_observations
from carla.rl.buffer import RolloutBatch


@torch.no_grad()
def collect_rollout(
    env: CatanEnv,
    policy: PolicyNetwork,
    num_steps: int,
    obs: dict | None = None,
    episode_seed: int | None = None,
    policy_version: int = 0,
) -> tuple[RolloutBatch, dict]:
    """Play `num_steps` learner decisions (resetting on episode end) and return the batch and
    the observation to continue from. Recurrent state is not handled yet (TODO(rl))."""
    if obs is None:
        obs, _ = env.reset(seed=episode_seed)
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    device = next(policy.parameters()).device
    policy.eval()
    steps: list[dict] = []
    actions, log_probs, values, rewards, dones = [], [], [], [], []
    for _ in range(num_steps):
        batch = batch_observations([obs], device=device)
        out = policy(batch)
        dist = out.distribution(batch["masks"])
        heads = dist.sample()
        steps.append(obs)
        actions.append(heads[0].cpu().numpy())
        log_probs.append(dist.log_prob(heads).item())
        values.append(out.value.item())
        obs, reward, terminated, truncated, _ = env.step(heads[0].cpu().numpy())
        rewards.append(reward)
        dones.append(terminated or truncated)
        if terminated or truncated:
            obs, _ = env.reset()
    last_value = policy(batch_observations([obs], device=device)).value.item()
    stacked = batch_observations(steps)
    batch = RolloutBatch(
        observations=_to_numpy(stacked),
        actions=np.stack(actions),
        log_probs=np.asarray(log_probs, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        last_value=last_value,
        policy_version=policy_version,
    )
    return batch, obs


def _to_numpy(tree: dict) -> dict:
    return {k: _to_numpy(v) if isinstance(v, dict) else v.cpu().numpy() for k, v in tree.items()}


class RolloutWorker:
    """Worker process of semi-asynchronous PPO. TODO(rl).

    Loop: (1) if the learner published newer weights, load them (rollouts may lag by at most
    `ppo.max_policy_lag` versions); (2) for each episode, sample the three opponents from the
    `OpponentPool`; (3) play `ppo.rollout_steps` decisions per env, batching the policy over the
    worker's envs with `torch.set_num_threads(runtime.torch_threads_per_worker)`; (4) send each
    `RolloutBatch` (tagged with its policy version) to the learner queue. Seeds: worker w uses
    `derive_seed(config.seed, "worker", w)`.
    """

    def __init__(self, worker_id: int, config, weights_queue, batch_queue) -> None:
        raise NotImplementedError("rl workstream: semi-asynchronous rollout worker")
