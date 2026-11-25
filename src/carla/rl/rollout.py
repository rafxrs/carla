"""Experience collection.

`collect_rollouts` batches one policy forward pass across several environments.
`process_rollouts` runs this collector in synchronous CPU processes. The semi-asynchronous
self-play `RolloutWorker` and opponent pool remain future work.
"""

from collections.abc import Sequence

import numpy as np
import torch

from carla.envs.catan_env import CatanEnv
from carla.models.base import TARGET_KEY, PolicyNetwork, batch_observations, with_state
from carla.rl.buffer import RolloutBatch
from carla.rl.metrics import EpisodeTracker, RolloutDiagnostics


@torch.no_grad()
def collect_rollouts(
    envs: Sequence[CatanEnv],
    policy: PolicyNetwork,
    num_steps: int,
    observations: Sequence[dict] | None = None,
    episode_seeds: Sequence[int] | None = None,
    policy_version: int = 0,
    *,
    trackers: Sequence[EpisodeTracker] | None = None,
    episode_records: list[dict] | None = None,
    diagnostics: RolloutDiagnostics | None = None,
    env_steps: int = 0,
    env_id_offset: int = 0,
    training_records: list[dict] | None = None,
    worker_id: int = 0,
) -> tuple[list[RolloutBatch], list[dict]]:
    """Collect `num_steps` decisions per env, retaining independent episode boundaries.

    Recurrent policies carry their state inside the returned observations (see `with_state`),
    and every stored observation also holds the belief targets (`TARGET_KEY`).
    """
    if not envs or num_steps <= 0:
        raise ValueError("need at least one environment and a positive num_steps")
    if observations is not None and len(observations) != len(envs):
        raise ValueError("observations must match environments")
    if episode_seeds is not None and len(episode_seeds) != len(envs):
        raise ValueError("episode_seeds must match environments")
    if trackers is not None and len(trackers) != len(envs):
        raise ValueError("trackers must match environments")
    recurrent = policy.recurrent
    current = (
        list(observations)
        if observations is not None
        else [
            env.reset(seed=None if episode_seeds is None else episode_seeds[i])[0]
            for i, env in enumerate(envs)
        ]
    )
    if recurrent:
        current = [
            with_state(policy, _with_target(obs, env))
            for obs, env in zip(current, envs, strict=True)
        ]
    device = next(policy.parameters()).device
    policy.eval()
    n = len(envs)
    steps: list[list[dict]] = [[] for _ in envs]
    actions: list[list[np.ndarray]] = [[] for _ in envs]
    log_probs: list[list[float]] = [[] for _ in envs]
    values: list[list[float]] = [[] for _ in envs]
    rewards: list[list[float]] = [[] for _ in envs]
    dones: list[list[bool]] = [[] for _ in envs]
    for step in range(num_steps):
        batch = batch_observations(current, device=device)
        out = policy(batch)
        dist = out.distribution(batch["masks"])
        heads = dist.sample()
        logp_arrays = dist.log_prob(heads).cpu().numpy()
        if diagnostics is not None:
            diagnostics.observe(dist, batch, heads)
        head_arrays = heads.cpu().numpy()
        value_arrays = out.value.cpu().numpy()
        next_states = out.state.cpu().numpy() if out.state is not None else None
        for i, env in enumerate(envs):
            steps[i].append(current[i])
            actions[i].append(head_arrays[i])
            log_probs[i].append(float(logp_arrays[i]))
            values[i].append(float(value_arrays[i]))
            obs, reward, terminated, truncated, _ = env.step(head_arrays[i])
            done = terminated or truncated
            if trackers is not None:
                record = trackers[i].step(
                    reward,
                    env,
                    done,
                    env_id_offset + i,
                    env_steps + step * n + i + 1,
                    training_records=training_records,
                    worker_id=worker_id,
                    policy_version=policy_version,
                )
                if record is not None and episode_records is not None:
                    episode_records.append(record)
            rewards[i].append(reward)
            dones[i].append(done)
            if done:
                obs = env.reset()[0]
            if recurrent:
                obs = _with_target(obs, env)
                obs = with_state(
                    policy, obs, None if done or next_states is None else next_states[i]
                )
            current[i] = obs
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


def _with_target(obs: dict, env: CatanEnv) -> dict:
    """Attach the opponents' hidden cards at this decision (belief targets, training only)."""
    return {**obs, TARGET_KEY: env.aec.hidden_truth(env.learner)}


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
