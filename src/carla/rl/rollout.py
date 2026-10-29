"""Experience collection.

`collect_rollouts` synchronously batches one policy forward pass across several environments.
The separate `RolloutWorker` process and opponent pool remain future work.
"""

from collections.abc import Sequence

import numpy as np
import torch

from carla.envs.catan_env import CatanEnv
from carla.models.base import PolicyNetwork, batch_observations
from carla.rl.buffer import RolloutBatch


@torch.no_grad()
def collect_rollouts(
    envs: Sequence[CatanEnv],
    policy: PolicyNetwork,
    num_steps: int,
    observations: Sequence[dict] | None = None,
    episode_seeds: Sequence[int] | None = None,
    policy_version: int = 0,
) -> tuple[list[RolloutBatch], list[dict]]:
    """Collect `num_steps` decisions per env, retaining independent episode boundaries."""
    if not envs or num_steps <= 0:
        raise ValueError("need at least one environment and a positive num_steps")
    if observations is not None and len(observations) != len(envs):
        raise ValueError("observations must match environments")
    if episode_seeds is not None and len(episode_seeds) != len(envs):
        raise ValueError("episode_seeds must match environments")
    current = (
        list(observations)
        if observations is not None
        else [
            env.reset(seed=None if episode_seeds is None else episode_seeds[i])[0]
            for i, env in enumerate(envs)
        ]
    )
    device = next(policy.parameters()).device
    policy.eval()
    n = len(envs)
    steps: list[list[dict]] = [[] for _ in envs]
    actions: list[list[np.ndarray]] = [[] for _ in envs]
    log_probs: list[list[float]] = [[] for _ in envs]
    values: list[list[float]] = [[] for _ in envs]
    rewards: list[list[float]] = [[] for _ in envs]
    dones: list[list[bool]] = [[] for _ in envs]
    for _ in range(num_steps):
        batch = batch_observations(current, device=device)
        out = policy(batch)
        dist = out.distribution(batch["masks"])
        heads = dist.sample()
        head_arrays = heads.cpu().numpy()
        logp_arrays = dist.log_prob(heads).cpu().numpy()
        value_arrays = out.value.cpu().numpy()
        for i, env in enumerate(envs):
            steps[i].append(current[i])
            actions[i].append(head_arrays[i])
            log_probs[i].append(float(logp_arrays[i]))
            values[i].append(float(value_arrays[i]))
            obs, reward, terminated, truncated, _ = env.step(head_arrays[i])
            done = terminated or truncated
            rewards[i].append(reward)
            dones[i].append(done)
            current[i] = env.reset()[0] if done else obs
    last_values = policy(batch_observations(current, device=device)).value.cpu().numpy()
    batches = [
        RolloutBatch(
            observations=_to_numpy(batch_observations(steps[i])),
            actions=np.stack(actions[i]),
            log_probs=np.asarray(log_probs[i], dtype=np.float32),
            values=np.asarray(values[i], dtype=np.float32),
            rewards=np.asarray(rewards[i], dtype=np.float32),
            dones=np.asarray(dones[i], dtype=np.float32),
            last_value=float(last_values[i]),
            policy_version=policy_version,
        )
        for i in range(n)
    ]
    return batches, current


@torch.no_grad()
def collect_rollout(
    env: CatanEnv,
    policy: PolicyNetwork,
    num_steps: int,
    obs: dict | None = None,
    episode_seed: int | None = None,
    policy_version: int = 0,
) -> tuple[RolloutBatch, dict]:
    """Single-environment compatibility wrapper around `collect_rollouts`."""
    batches, observations = collect_rollouts(
        [env],
        policy,
        num_steps,
        observations=None if obs is None else [obs],
        episode_seeds=None if episode_seed is None else [episode_seed],
        policy_version=policy_version,
    )
    return batches[0], observations[0]


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
