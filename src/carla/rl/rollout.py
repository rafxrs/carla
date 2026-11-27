"""Experience collection.

`collect_rollouts` batches one policy forward pass across several environments, and batches
the decisions of policy-played opponent seats (a `League`) per network between learner steps.
`process_rollouts` runs this collector in synchronous CPU processes. The semi-asynchronous
`RolloutWorker` remains future work.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from carla.config.loader import from_dict
from carla.config.schema import ExperimentConfig, PPOConfig
from carla.envs.catan_env import CatanEnv, EnvSteps, ExternalAgent
from carla.envs.strategy import strategy_observation
from carla.models.base import TARGET_KEY, PolicyNetwork, batch_observations, with_state
from carla.models.factory import build_policy
from carla.rl.acting import decide
from carla.rl.buffer import RolloutBatch
from carla.rl.metrics import EpisodeTracker, RolloutDiagnostics

if TYPE_CHECKING:
    from carla.rl.opponents import League
    from carla.rl.strategies import StrategyAssigner

REFERENCE_STATE_KEY = "reference_state"
"""Observation entry holding the reference policy's own recurrent state before the decision."""
REFERENCE_TYPE_KEY = "reference_type_logp"
"""Stored observation entry: the reference policy's masked action-type log-probabilities."""
REFERENCE_LOGP_KEY = "reference_logp"
"""Stored observation entry (reference_kl_heads=all): the reference's log-probabilities of every
head (a dict keyed like `MASK_SHAPES`, flattened)."""


def load_reference(config: PPOConfig, device: torch.device | str = "cpu") -> PolicyNetwork | None:
    """The frozen reference policy of `ppo.reference_checkpoint`, or None when it is unused.

    The reference keeps its own architecture (from the checkpoint's config) and recurrent state,
    so any policy that reads the same observations can serve, e.g. a behaviour clone.
    """
    if config.reference_kl_coef <= 0:
        return None
    if config.reference_checkpoint is None:
        raise ValueError("ppo.reference_kl_coef needs ppo.reference_checkpoint")
    data = torch.load(config.reference_checkpoint, map_location="cpu", weights_only=False)
    if data.get("config") is None:
        raise ValueError(
            f"reference checkpoint has no experiment config: {config.reference_checkpoint}"
        )
    reference = build_policy(from_dict(ExperimentConfig, data["config"]).model)
    reference.load_state_dict(data["policy"])
    reference.requires_grad_(False)
    return reference.to(device).eval()


def _with_reference_state(reference: PolicyNetwork, obs: dict, state: np.ndarray | None) -> dict:
    """`obs` carrying the reference's state (zeros at an episode start or when missing)."""
    if not reference.recurrent:
        return obs
    if state is None:
        if REFERENCE_STATE_KEY in obs:
            return obs
        state = np.zeros(reference.state_dim, dtype=np.float32)
    return {**obs, REFERENCE_STATE_KEY: np.asarray(state, dtype=np.float32)}


def _strategy_inputs(env: CatanEnv, seat: int, agent=None):
    controller = env.strategy if agent is None else agent.strategy
    return controller, env.aec.stats[seat]


@torch.no_grad()
def _opponent_actions(
    envs: Sequence[CatanEnv], waiting: list[tuple[int, int]], network: PolicyNetwork
) -> list[np.ndarray]:
    """One sampled decision of `network` for each waiting (env index, seat), batched."""
    agents = []
    observations = []
    controllers, games, seats, stats = [], [], [], []
    for i, seat in waiting:
        agent = envs[i].seat_agent(seat)
        assert isinstance(agent, ExternalAgent)
        agents.append(agent)
        observations.append(with_state(network, envs[i].aec.observe(seat), agent.state))
        controllers.append(agent.strategy)
        games.append(envs[i].game)
        seats.append(seat)
        stats.append(envs[i].aec.stats[seat])
    decisions = decide(network, observations, controllers, games, seats, stats)
    out = decisions.output
    if out.state is not None:
        for agent, state in zip(agents, out.state.cpu().numpy(), strict=True):
            agent.state = state
    return list(decisions.heads)


def run_env_steps(
    envs: Sequence[CatanEnv],
    steps: Sequence[EnvSteps | None],
    policy: PolicyNetwork,
    league: "League | None" = None,
) -> list[Any]:
    """Drive every env's `reset_steps`/`step_steps` to its result (None where `steps` is None).

    Whenever envs wait on `ExternalAgent` seats, their decisions are batched per network: the
    current `policy` plays "self" seats and the `league` supplies the others.
    """
    results: list[Any] = [None] * len(envs)
    waiting: dict[int, int] = {}

    def advance(i: int, action: Any) -> None:
        generator = steps[i]
        assert generator is not None
        try:
            waiting[i] = generator.send(action)
        except StopIteration as stop:
            results[i] = stop.value

    for i, generator in enumerate(steps):
        if generator is not None:
            advance(i, None)
    while waiting:
        groups: dict[str, list[tuple[int, int]]] = {}
        for i, seat in waiting.items():
            agent = envs[i].seat_agent(seat)
            assert isinstance(agent, ExternalAgent)
            groups.setdefault(agent.key, []).append((i, seat))
        waiting = {}
        for key, group in groups.items():
            network = policy if league is None else league.network(key, policy)
            for (i, _), action in zip(group, _opponent_actions(envs, group, network), strict=True):
                advance(i, action)
    return results


def _selectors(env: CatanEnv, policy: PolicyNetwork, league: "League | None") -> dict[str, bool]:
    out = {}
    for agent in env.opponents:
        if isinstance(agent, ExternalAgent):
            network = policy if league is None else league.network(agent.key, policy)
            out[agent.key] = network.num_strategies > 0
    return out


def prepare_episode(
    env: CatanEnv,
    policy: PolicyNetwork,
    league: "League | None",
    strategies: "StrategyAssigner | None",
) -> None:
    """Sample the episode's opponents (league) and every seat's strategy (assigner)."""
    variant = None
    if league is not None:
        env.opponents = league.sample()
        variant = getattr(league, "learner_variant", lambda: None)()
    if strategies is not None:
        strategies.prepare(env, _selectors(env, policy, league), variant)


def reset_envs(
    envs: Sequence[CatanEnv],
    policy: PolicyNetwork,
    league: "League | None" = None,
    seeds: Sequence[int | None] | None = None,
    strategies: "StrategyAssigner | None" = None,
) -> list[dict]:
    """Reset every env (sampling new league opponents) and return the learner observations."""
    for env in envs:
        prepare_episode(env, policy, league, strategies)
    steps = [
        env.reset_steps(seed=None if seeds is None else seeds[i]) for i, env in enumerate(envs)
    ]
    return [result[0] for result in run_env_steps(envs, steps, policy, league)]


def needs_hidden_truth(policy: PolicyNetwork) -> bool:
    """Belief targets and privileged critics read the opponents' cards (training only)."""
    return policy.recurrent or bool(getattr(policy, "privileged", False))


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
    reference: PolicyNetwork | None = None,
    league: "League | None" = None,
    strategies: "StrategyAssigner | None" = None,
    reference_heads: str = "type",
) -> tuple[list[RolloutBatch], list[dict]]:
    """Collect `num_steps` decisions per env, retaining independent episode boundaries.

    Recurrent policies carry their state inside the returned observations (see `with_state`),
    and every stored observation also holds the belief targets (`TARGET_KEY`). With a
    `reference` policy, each stored decision also holds its action-type log-probabilities
    (`REFERENCE_TYPE_KEY`; every head with `reference_heads="all"`), and the reference carries
    its own state the same way. With a `league`, every episode samples its opponents from it.
    With `strategies`, every episode assigns the seats' strategies; stored observations then
    hold the learner's z and selections, and batches carry the strategy pseudo-rewards.
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
    truth = needs_hidden_truth(policy)
    with_strategy = policy.uses_strategy or bool(reference is not None and reference.uses_strategy)
    policy.eval()
    if league is not None:
        league.begin_rollout()
    current = (
        list(observations)
        if observations is not None
        else reset_envs(envs, policy, league, episode_seeds, strategies)
    )
    current = [
        _decision_observation(policy, obs, env, truth)
        for obs, env in zip(current, envs, strict=True)
    ]
    if reference is not None:
        current = [_with_reference_state(reference, obs, None) for obs in current]
    device = next(policy.parameters()).device
    n = len(envs)
    steps: list[list[dict]] = [[] for _ in envs]
    actions: list[list[np.ndarray]] = [[] for _ in envs]
    log_probs: list[list[float]] = [[] for _ in envs]
    values: list[list[float]] = [[] for _ in envs]
    rewards: list[list[float]] = [[] for _ in envs]
    strategy_values: list[list[float]] = [[] for _ in envs]
    strategy_rewards: list[list[float]] = [[] for _ in envs]
    dones: list[list[bool]] = [[] for _ in envs]
    for step in range(num_steps):
        decisions = decide(
            policy,
            current,
            [env.strategy for env in envs],
            [env.game for env in envs],
            [env.learner for env in envs],
            [env.aec.stats[env.learner] for env in envs],
            with_strategy=with_strategy,
        )
        out, batch = decisions.output, decisions.batch
        if diagnostics is not None:
            diagnostics.observe(decisions.distribution, batch, torch.as_tensor(decisions.heads))
        head_arrays = decisions.heads
        value_arrays = out.value.cpu().numpy()
        strategy_value_arrays = (
            out.strategy_value.cpu().numpy() if out.strategy_value is not None else None
        )
        next_states = out.state.cpu().numpy() if out.state is not None else None
        reference_logp = reference_states = None
        if reference is not None:
            ref_out = reference(
                batch, batch[REFERENCE_STATE_KEY].float() if reference.recurrent else None
            )
            ref_dist = ref_out.distribution(batch["masks"])
            keys = ("type",) if reference_heads == "type" else tuple(ref_dist.logp)
            reference_logp = {key: ref_dist.logp[key].float().cpu().numpy() for key in keys}
            if ref_out.state is not None:
                reference_states = ref_out.state.cpu().numpy()
        for i in range(n):
            stored = decisions.observations[i]
            if reference_logp is not None:
                stored = {**stored, REFERENCE_TYPE_KEY: reference_logp["type"][i]}
                if reference_heads == "all":
                    stored[REFERENCE_LOGP_KEY] = {k: v[i] for k, v in reference_logp.items()}
            steps[i].append(stored)
            actions[i].append(head_arrays[i])
            log_probs[i].append(float(decisions.log_probs[i]))
            values[i].append(float(value_arrays[i]))
            if strategy_value_arrays is not None:
                strategy_values[i].append(float(strategy_value_arrays[i]))
        results = run_env_steps(
            envs, [env.step_steps(head_arrays[i]) for i, env in enumerate(envs)], policy, league
        )
        resets: list[EnvSteps | None] = [None] * n
        for i, env in enumerate(envs):
            obs, reward, terminated, truncated, info = results[i]
            done = terminated or truncated
            if league is not None and hasattr(league, "terminal_reward"):
                reward = league.terminal_reward(env, reward, done)
            strategy_rewards[i].append(float(info.get("strategy_reward", 0.0)))
            if trackers is not None:
                record = trackers[i].step(
                    reward,
                    env,
                    done,
                    env_id_offset + i,
                    env_steps + step * n + i + 1,
                    strategy_reward=strategy_rewards[i][-1],
                )
                if record is not None and episode_records is not None:
                    episode_records.append(record)
            rewards[i].append(reward)
            dones[i].append(done)
            if done:
                if league is not None and hasattr(league, "episode_end"):
                    league.episode_end(env, record if trackers is not None else None)
                prepare_episode(env, policy, league, strategies)
                resets[i] = env.reset_steps()
            current[i] = obs
        for i, result in enumerate(run_env_steps(envs, resets, policy, league)):
            done = result is not None
            obs = result[0] if done else current[i]
            env = envs[i]
            obs = _decision_observation(policy, obs, env, truth)
            if recurrent:
                obs = with_state(
                    policy, obs, None if done or next_states is None else next_states[i]
                )
            if reference is not None:
                obs = _with_reference_state(
                    reference,
                    obs,
                    None if done or reference_states is None else reference_states[i],
                )
            current[i] = obs
    if with_strategy:
        # Bootstrap under the current z; a selection due at the next decision is not taken here.
        current_rows = [
            strategy_observation(obs, env.strategy, False)
            for obs, env in zip(current, envs, strict=True)
        ]
    else:
        current_rows = current
    last = policy(batch_observations(current_rows, device=device))
    last_values = last.value.cpu().numpy()
    last_strategy = last.strategy_value.cpu().numpy() if last.strategy_value is not None else None
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
            strategy_rewards=(
                np.asarray(strategy_rewards[i], dtype=np.float32)
                if last_strategy is not None
                else None
            ),
            strategy_values=(
                np.asarray(strategy_values[i], dtype=np.float32)
                if last_strategy is not None
                else None
            ),
            last_strategy_value=float(last_strategy[i]) if last_strategy is not None else 0.0,
        )
        for i in range(n)
    ]
    return batches, current


def _decision_observation(policy: PolicyNetwork, obs: dict, env: CatanEnv, truth: bool) -> dict:
    """Training-only additions to a learner observation: belief targets / privileged inputs."""
    if truth and TARGET_KEY not in obs:
        obs = _with_target(obs, env)
    if policy.recurrent:
        obs = with_state(policy, obs)
    return obs


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
