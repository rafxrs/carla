"""Behavioural-cloning warm start: imitate the heuristic bot before PPO.

Worker processes play `CatanEnv` games in which the heuristic bot makes the learner's decisions,
against the experiment's opponents, so observations, rewards and belief targets are exactly
those PPO trains on. The learner then minimises the negative log-likelihood of the expert's
actions, fits the value head to the expert's discounted returns, and trains the belief head.

Recurrent policies get their belief state like PPO does: one step of the GRU from a stored
state. Before each epoch the stored states are recomputed by running the current network
through every game in order, so they match the weights being trained.
"""

import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from carla.agents import make_agent
from carla.agents.heuristic import HeuristicAgent
from carla.config.loader import to_dict
from carla.config.schema import ExperimentConfig
from carla.engine.actions import ActionType
from carla.engine.rng import derive_seed
from carla.envs.action_space import HEAD_INDEX, HEAD_NAMES, decode_action, encode_action
from carla.envs.catan_env import CatanEnv
from carla.envs.trade_tokens import PAD, offer_tokens
from carla.models.base import STATE_KEY, TARGET_KEY, PolicyNetwork, batch_observations
from carla.models.factory import build_policy
from carla.rl.evaluate import evaluate_policy
from carla.rl.opponents import bot_tables

log = logging.getLogger(__name__)


@dataclass
class Demonstrations:
    """Expert decisions of many games, flattened in game order."""

    observations: dict
    """Nested numpy arrays (N, ...), including belief targets for recurrent policies."""
    actions: np.ndarray
    """(N, action width) expert actions in the policy's action format."""
    returns: np.ndarray
    """(N,) discounted returns of the expert from each decision."""
    starts: np.ndarray
    """(G,) index of each game's first decision."""
    lengths: np.ndarray
    """(G,) decisions per game."""

    def __len__(self) -> int:
        return len(self.actions)


def expert_action(action, actor: int, width: int, max_cards: int | None) -> np.ndarray:
    """The policy-format action (heads, then offer tokens if the model has a decoder)."""
    multi_card = action.type == ActionType.PROPOSE_TRADE and (
        action.trade is None or sum(action.trade.give) != 1 or sum(action.trade.receive) != 1
    )
    if multi_card and width > len(HEAD_NAMES):
        heads = np.full(len(HEAD_NAMES), -1, dtype=np.int64)
        heads[0] = ActionType.PROPOSE_TRADE
    else:
        heads = encode_action(action, actor)
    if width == len(HEAD_NAMES):
        return heads
    tokens = np.full(width - len(HEAD_NAMES), PAD, dtype=np.int64)
    if action.type == ActionType.PROPOSE_TRADE:
        assert action.trade is not None and max_cards is not None
        offer = offer_tokens(action.trade, max_cards)
        tokens[: len(offer)] = offer
        heads[HEAD_INDEX["resource"]] = PAD
        heads[HEAD_INDEX["resource2"]] = PAD
    return np.concatenate((heads, tokens))


def _play_games(args: tuple) -> list[dict]:
    """Worker: play games with the heuristic as learner; return per-game arrays."""
    config, seeds, width, recurrent = args
    torch.set_num_threads(1)
    max_cards = config.model.max_trade_cards_per_side if width > len(HEAD_NAMES) else None
    tables = bot_tables(config.opponents, 1, 1)[0][0]
    games = []
    for seed in seeds:
        env = CatanEnv(config.engine, config.env, [make_agent(kind) for kind in tables])
        obs, _ = env.reset(seed=seed)
        expert = HeuristicAgent()
        expert.reset(env.game, env.learner, derive_seed(seed, "expert"))
        observations, actions, rewards = [], [], []
        done = False
        while not done:
            game, seat = env.game, env.learner
            action = expert.act(game, seat, game.legal_actions())
            encoded = expert_action(action, seat, width, max_cards)
            if action.type != ActionType.PROPOSE_TRADE:
                assert decode_action(encoded[: len(HEAD_NAMES)], seat) == action
            if recurrent:
                obs = {**obs, TARGET_KEY: env.aec.hidden_truth(seat)}
            observations.append(obs)
            actions.append(encoded)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            done = terminated or truncated
        env.close()
        batch = batch_observations(observations)
        games.append(
            {
                "observations": _to_numpy(batch),
                "actions": np.stack(actions),
                "rewards": np.asarray(rewards, dtype=np.float32),
            }
        )
    return games


def collect_demonstrations(
    config: ExperimentConfig,
    policy: PolicyNetwork,
    games: int,
    *,
    seed: int = 0,
    workers: int = 1,
) -> Demonstrations:
    """Play `games` expert games (in `workers` processes) and flatten them."""
    with torch.no_grad():
        probe = batch_observations([_probe_observation(config)])
        out = policy.eval()(probe)
        width = int(out.distribution(probe["masks"]).sample().shape[1])
    seeds = [derive_seed(seed, "demonstration", g) for g in range(games)]
    chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
    jobs = [(config, chunk, width, policy.recurrent) for chunk in chunks]
    if workers == 1:
        results = [_play_games(job) for job in jobs]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(_play_games, jobs)
    played = [game for chunk in results for game in chunk]
    gamma = config.ppo.gamma
    returns = []
    for game in played:
        g, out = 0.0, np.zeros_like(game["rewards"])
        for t in range(len(out) - 1, -1, -1):
            g = game["rewards"][t] + gamma * g
            out[t] = g
        returns.append(out)
    lengths = np.asarray([len(game["actions"]) for game in played], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
    return Demonstrations(
        observations=_concat([game["observations"] for game in played]),
        actions=np.concatenate([game["actions"] for game in played]),
        returns=np.concatenate(returns).astype(np.float32),
        starts=starts,
        lengths=lengths,
    )


@torch.no_grad()
def recompute_states(policy: PolicyNetwork, data: Demonstrations, device) -> np.ndarray | None:
    """The belief state before every decision under the current weights (None if stateless)."""
    if not policy.recurrent:
        return None
    policy.eval()
    states = np.zeros((len(data), policy.state_dim), dtype=np.float32)
    current = policy.initial_state(len(data.starts), device)
    assert current is not None
    for t in range(int(data.lengths.max())):
        alive = np.nonzero(data.lengths > t)[0]
        index = data.starts[alive] + t
        states[index] = current[alive].cpu().numpy()
        obs = _to_tensors(_take(data.observations, index), device)
        out = policy(obs, current[alive])
        assert out.state is not None
        current[alive] = out.state
    return states


def behaviour_clone(
    config: ExperimentConfig,
    output: Path,
    *,
    games: int = 600,
    epochs: int = 8,
    learning_rate: float = 3e-4,
    minibatch_size: int = 1024,
    value_coef: float = 0.5,
    workers: int = 1,
    eval_boards: int = 10,
    seed: int = 0,
) -> Path:
    """Clone the heuristic into `config.model`; writes `output/bc.pt` loadable by --init-from."""
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.runtime.device)
    torch.manual_seed(seed)
    policy = build_policy(config.model).to(device)
    log.info("collecting %d expert games with %d workers", games, workers)
    data = collect_demonstrations(config, policy.cpu(), games, seed=seed, workers=workers)
    policy.to(device)
    log.info("%d expert decisions (%.0f per game)", len(data), len(data) / games)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    actions = torch.as_tensor(data.actions)
    returns = torch.as_tensor(data.returns)
    for epoch in range(1, epochs + 1):
        states = recompute_states(policy, data, device)
        policy.train()
        totals = {"nll": 0.0, "value": 0.0, "auxiliary": 0.0, "type_accuracy": 0.0}
        batches = 0
        for index in torch.randperm(len(data)).split(minibatch_size):
            idx = index.numpy()
            obs = _take(data.observations, idx)
            if states is not None:
                obs[STATE_KEY] = states[idx]
            mini = _to_tensors(obs, device)
            out = policy(mini)
            dist = out.distribution(mini["masks"])
            target = actions[index].to(device)
            nll = -dist.log_prob(target).mean()
            value = (out.value - returns[index].to(device)).square().mean()
            loss = nll + value_coef * value
            auxiliary = policy.auxiliary_loss(out, mini)
            if auxiliary is not None:
                loss = loss + auxiliary
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                greedy = dist.logp["type"].argmax(1)
                totals["type_accuracy"] += float((greedy == target[:, 0]).float().mean())
            totals["nll"] += float(nll.detach())
            totals["value"] += float(value.detach())
            totals["auxiliary"] += float(auxiliary.detach()) if auxiliary is not None else 0.0
            batches += 1
        log.info(
            "epoch %d | %s",
            epoch,
            " | ".join(f"{k} {v / batches:.4f}" for k, v in totals.items()),
        )
    path = output / "bc.pt"
    torch.save(
        {
            "config": to_dict(config),
            "policy": {k: v.cpu() for k, v in policy.state_dict().items()},
            "counters": {"updates": 0, "env_steps": 0},
            "imitation": {"games": games, "decisions": len(data), "epochs": epochs},
        },
        path,
    )
    log.info("saved %s", path)
    if eval_boards:
        for opponent in ("random", "heuristic"):
            result = evaluate_policy(policy, config, opponent=opponent, boards=eval_boards)
            log.info(
                "behaviour clone vs %s: %d/%d wins (%.3f), mean VP %.2f",
                opponent,
                result.wins,
                result.games,
                result.win_rate,
                result.mean_vp,
            )
    return path


def _probe_observation(config: ExperimentConfig) -> dict:
    env = CatanEnv(config.engine, config.env, [make_agent("random") for _ in range(3)])
    obs, _ = env.reset(seed=0)
    env.close()
    return obs


def _to_numpy(tree: dict) -> dict:
    return {k: _to_numpy(v) if isinstance(v, dict) else v.cpu().numpy() for k, v in tree.items()}


def _concat(trees: list[dict]) -> dict:
    return {
        k: (
            _concat([t[k] for t in trees])
            if isinstance(trees[0][k], dict)
            else np.concatenate([t[k] for t in trees])
        )
        for k in trees[0]
    }


def _take(tree: dict, index: np.ndarray) -> dict:
    return {k: _take(v, index) if isinstance(v, dict) else v[index] for k, v in tree.items()}


def _to_tensors(tree: dict, device) -> dict:
    return {
        k: _to_tensors(v, device) if isinstance(v, dict) else torch.as_tensor(v, device=device)
        for k, v in tree.items()
    }
