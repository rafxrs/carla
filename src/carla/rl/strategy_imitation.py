"""Strategy-conditioned behaviour cloning from the styled heuristic teachers (`carla strategy-bc`).

Demonstrations: in each game one teacher (`carla.agents.strategic`) plays the learner seat
against a table drawn from all teachers and the baseline heuristic. After the game, the
teacher's hindsight z (`carla.envs.strategy.encode_z` of its own public statistics) labels
every one of its decisions, like AlphaStar's z extracted from each replay. With probability
`strategy.sl_zero_prob` a game is trained without z, so the policy also plays without one.

Losses: action log-likelihood under the game's z (all heads, offer tokens included), the
outcome critic on the teacher's discounted returns, the belief auxiliary loss, and, for a model
with a selector, the selector's cross-entropy toward the teacher's archetype at the selection
points of games the teacher won (choose what wins from this situation).

Recurrent models train on chunks of `ppo.sequence_length` decisions unrolled from states
recomputed with the current weights at the start of every epoch.

Outputs (`--output`): `bc.pt` (policy, config, the z library), `strategy_library.npz`, and
`strategy_profile.csv`: the clone's behaviour under each z variant against heuristic bots.
"""

import csv
import json
import logging
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from carla.agents import TEACHERS, make_agent
from carla.config.loader import to_dict
from carla.config.schema import ExperimentConfig
from carla.engine.actions import ActionType
from carla.engine.rng import derive_seed
from carla.envs.action_space import HEAD_NAMES
from carla.envs.catan_env import CatanEnv
from carla.envs.strategy import (
    CHOICE_KEY,
    NONE,
    SELECT,
    SELECT_KEY,
    STRATEGY_KEY,
    StrategyController,
    StrategyLibrary,
    encode_z,
    zero_z,
)
from carla.models.base import STATE_KEY, TARGET_KEY, PolicyNetwork, batch_observations
from carla.models.factory import build_policy
from carla.rl.evaluate import evaluate_policy
from carla.rl.imitation import _concat, _take, _to_numpy, _to_tensors, expert_action
from carla.rl.rollout import needs_hidden_truth
from carla.rl.strategies import LIBRARY_KEY
from carla.rl.train import validate_strategy

log = logging.getLogger(__name__)


def archetype_of(teacher: str) -> str:
    return teacher.removeprefix("heuristic_")


def _play(args: tuple) -> list[dict]:
    config, jobs, width, truth = args
    torch.set_num_threads(1)
    max_cards = config.model.max_trade_cards_per_side if width > len(HEAD_NAMES) else None
    table_pool = (*TEACHERS, "heuristic")
    games = []
    for seed, teacher in jobs:
        rng = np.random.default_rng(derive_seed(seed, "table"))
        opponents = [make_agent(table_pool[int(rng.integers(len(table_pool)))]) for _ in range(3)]
        env = CatanEnv(config.engine, config.env, opponents)
        obs, _ = env.reset(seed=seed)
        expert = make_agent(teacher)
        expert.multi_card_offers = width > len(HEAD_NAMES)
        expert.reset(env.game, env.learner, derive_seed(seed, "expert"))
        schedule = StrategyController(config.strategy, SELECT)
        observations, actions, rewards, selects = [], [], [], []
        done = False
        while not done:
            game, seat = env.game, env.learner
            select = schedule.needs_selection(game, seat)
            if select:
                schedule.select(len(config.strategy.archetypes), game, env.aec.stats[seat])
            action = expert.act(game, seat, game.legal_actions())
            if truth:
                obs = {**obs, TARGET_KEY: env.aec.hidden_truth(seat)}
            observations.append(obs)
            actions.append(expert_action(action, seat, width, max_cards))
            selects.append(select)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            done = terminated or truncated
        stats = env.aec.stats[env.learner]
        won = info.get("winner") == env.learner
        env.close()
        arrays = _to_numpy(batch_observations(observations))
        # Events dominate the memory of a demonstration set; half precision holds them exactly
        # enough (one-hots and small fractions) and the belief module casts them back.
        arrays["events"] = arrays["events"].astype(np.float16)
        games.append(
            {
                "observations": arrays,
                "actions": np.stack(actions),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "selects": np.asarray(selects, dtype=bool),
                "z": encode_z(stats, reward_active=True),
                "teacher": teacher,
                "won": won,
            }
        )
    return games


def collect_strategy_demonstrations(
    config: ExperimentConfig,
    policy: PolicyNetwork,
    games: int,
    *,
    seed: int = 0,
    workers: int = 1,
) -> tuple[dict, StrategyLibrary]:
    """Teacher games (balanced over `TEACHERS`) flattened for training, and their z library."""
    with torch.no_grad():
        env = CatanEnv(config.engine, config.env, [make_agent("random") for _ in range(3)])
        probe, _ = env.reset(seed=0)
        env.close()
        probe = {**probe, STRATEGY_KEY: zero_z()}
        if needs_hidden_truth(policy):
            probe[TARGET_KEY] = env.aec.hidden_truth(0)
        batch = batch_observations([probe])
        width = int(policy.eval()(batch).distribution(batch["masks"]).sample().shape[1])
    jobs = [
        (derive_seed(seed, "strategy-demonstration", g), TEACHERS[g % len(TEACHERS)])
        for g in range(games)
    ]
    chunks = [jobs[i::workers] for i in range(workers) if jobs[i::workers]]
    args = [(config, chunk, width, needs_hidden_truth(policy)) for chunk in chunks]
    if workers == 1:
        results = [_play(a) for a in args]
    else:
        with mp.get_context("spawn").Pool(len(args)) as pool:
            results = pool.map(_play, args)
    played = [g for chunk in results for g in chunk]
    library = StrategyLibrary(
        {archetype_of(t): np.stack([g["z"] for g in played if g["teacher"] == t]) for t in TEACHERS}
    )
    archetypes = config.strategy.archetypes
    rng = np.random.default_rng(derive_seed(seed, "strategy-mask"))
    gamma = config.ppo.gamma
    for game in played:
        n = len(game["actions"])
        keep = rng.random() >= config.strategy.sl_zero_prob
        z = game["z"].copy()
        z[-1] = float(rng.random() < 0.5)  # The pseudo-reward flag is an input; vary it.
        game["observations"][STRATEGY_KEY] = np.repeat((z if keep else zero_z())[None], n, 0)
        game["observations"][SELECT_KEY] = np.zeros(n, dtype=np.float32)
        game["observations"][CHOICE_KEY] = np.full(n, -1, dtype=np.int64)
        archetype = archetype_of(game["teacher"])
        target = archetypes.index(archetype) if archetype in archetypes and game["won"] else -1
        game["select_targets"] = np.where(game["selects"], target, -1).astype(np.int64)
        returns, g = np.zeros(n, dtype=np.float32), 0.0
        for t in range(n - 1, -1, -1):
            g = game["rewards"][t] + gamma * g
            returns[t] = g
        game["returns"] = returns
    lengths = np.asarray([len(g["actions"]) for g in played], dtype=np.int64)
    data = {
        "observations": _concat([g["observations"] for g in played]),
        "actions": np.concatenate([g["actions"] for g in played]),
        "returns": np.concatenate([g["returns"] for g in played]),
        "select_targets": np.concatenate([g["select_targets"] for g in played]),
        "starts": np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64),
        "lengths": lengths,
        "teachers": [g["teacher"] for g in played],
        "won": np.asarray([g["won"] for g in played]),
    }
    return data, library


@torch.no_grad()
def recompute_states(policy: PolicyNetwork, data: dict, device) -> np.ndarray | None:
    """The recurrent state before every decision under the current weights."""
    if not policy.recurrent:
        return None
    policy.eval()
    starts, lengths = data["starts"], data["lengths"]
    total = int(lengths.sum())
    states = np.zeros((total, policy.state_dim), dtype=np.float32)
    current = torch.zeros(len(starts), policy.state_dim, device=device)
    for t in range(int(lengths.max())):
        alive = np.nonzero(lengths > t)[0]
        index = starts[alive] + t
        states[index] = current[alive].cpu().numpy()
        out = policy(_to_tensors(_take(data["observations"], index), device), current[alive])
        assert out.state is not None
        current[alive] = out.state
    return states


def _chunks(data: dict, length: int) -> list[tuple[int, int]]:
    out = []
    for start, n in zip(data["starts"], data["lengths"], strict=True):
        for offset in range(0, int(n), length):
            out.append((int(start) + offset, min(length, int(n) - offset)))
    return out


def strategy_behaviour_clone(
    config: ExperimentConfig,
    output: Path,
    *,
    games: int = 900,
    epochs: int = 8,
    learning_rate: float = 3e-4,
    minibatch_size: int = 1024,
    value_coef: float = 0.5,
    selector_coef: float = 0.5,
    workers: int = 1,
    eval_boards: int = 5,
    seed: int = 0,
) -> Path:
    """Train `config.model` on teacher demonstrations; writes `output/bc.pt`."""
    validate_strategy(config)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if config.runtime.device == "auto" and torch.cuda.is_available()
        else ("cpu" if config.runtime.device == "auto" else config.runtime.device)
    )
    torch.manual_seed(seed)
    policy = build_policy(config.model)
    log.info("collecting %d teacher games with %d workers", games, workers)
    data, library = collect_strategy_demonstrations(
        config, policy.cpu(), games, seed=seed, workers=workers
    )
    library.save(output / "strategy_library.npz")
    teachers = data["teachers"]
    for t in TEACHERS:
        won = [w for w, name in zip(data["won"], teachers, strict=True) if name == t]
        log.info("teacher %s: %d games, win rate %.3f", t, len(won), float(np.mean(won)))
    policy.to(device)
    total = len(data["actions"])
    log.info("%d teacher decisions (%.0f per game)", total, total / games)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    actions = torch.as_tensor(data["actions"])
    returns = torch.as_tensor(data["returns"])
    targets = torch.as_tensor(data["select_targets"])
    sequence = policy.recurrent and config.ppo.bptt == "sequence"
    length = config.ppo.sequence_length if sequence else 1
    for epoch in range(1, epochs + 1):
        states = recompute_states(policy, data, device)
        policy.train()
        sums = {"nll": 0.0, "value": 0.0, "auxiliary": 0.0, "selector": 0.0, "type_accuracy": 0.0}
        batches = 0
        if sequence:
            units = _chunks(data, length)
            by_size: dict[int, list] = {}
            for u in units:
                by_size.setdefault(u[1], []).append(u)
            groups = []
            per = max(1, minibatch_size // length)
            for group in by_size.values():
                perm = torch.randperm(len(group)).tolist()
                groups += [[group[j] for j in perm[i : i + per]] for i in range(0, len(group), per)]
            order = [groups[k] for k in torch.randperm(len(groups)).tolist()]
        else:
            order = [
                [(int(i), 1) for i in idx] for idx in torch.randperm(total).split(minibatch_size)
            ]
        for group in order:
            steps = group[0][1]
            starts = np.asarray([u[0] for u in group])
            index = (starts[None, :] + np.arange(steps)[:, None]).reshape(-1)
            obs = _take(data["observations"], index)
            mini = _to_tensors(obs, device)
            if sequence:
                assert states is not None
                state0 = torch.as_tensor(states[starts], device=device)
                out = policy.unroll(mini, state0, None, steps)
            else:
                if states is not None:
                    mini[STATE_KEY] = torch.as_tensor(states[index], device=device)
                out = policy(mini)
            dist = out.distribution(mini["masks"])
            target = actions[index].to(device)
            nll = -dist.log_prob(target).mean()
            value = (out.value - returns[index].to(device)).square().mean()
            loss = nll + value_coef * value
            auxiliary = policy.auxiliary_loss(out, mini)
            if auxiliary is not None:
                loss = loss + auxiliary
            selector_loss = torch.zeros((), device=device)
            choice = targets[index].to(device)
            if out.strategy_logits is not None and (choice >= 0).any():
                rows = choice >= 0
                selector_loss = torch.nn.functional.cross_entropy(
                    out.strategy_logits[rows].float(), choice[rows]
                )
                loss = loss + selector_coef * selector_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                sums["type_accuracy"] += float(
                    (dist.logp["type"].argmax(1) == target[:, 0]).float().mean()
                )
            sums["nll"] += float(nll.detach())
            sums["value"] += float(value.detach())
            sums["auxiliary"] += float(auxiliary.detach()) if auxiliary is not None else 0.0
            sums["selector"] += float(selector_loss.detach())
            batches += 1
        log.info(
            "epoch %d | %s", epoch, " | ".join(f"{k} {v / batches:.4f}" for k, v in sums.items())
        )
    path = output / "bc.pt"
    torch.save(
        {
            "config": to_dict(config),
            "policy": {k: v.cpu() for k, v in policy.state_dict().items()},
            "counters": {"updates": 0, "env_steps": 0},
            "imitation": {"games": games, "decisions": total, "epochs": epochs},
            LIBRARY_KEY: library.to_arrays(),
        },
        path,
    )
    log.info("saved %s", path)
    if policy.uses_strategy:
        sensitivity = conditioning_sensitivity(policy, data, library, config, device)
        (output / "conditioning.json").write_text(json.dumps(sensitivity, indent=2) + "\n")
        for name, row in sensitivity["archetypes"].items():
            log.info(
                "z %s on held states: %s",
                name,
                ", ".join(f"P({k}) {v:.3f}" for k, v in row.items()),
            )
        log.info("mean pairwise action-type KL between archetype z's: %.4f", sensitivity["mean_kl"])
    if eval_boards and policy.uses_strategy:
        write_profile(policy, config, library, output / "strategy_profile.csv", eval_boards)
    elif eval_boards:
        for opponent in ("random", "heuristic"):
            result = evaluate_policy(policy, config, opponent=opponent, boards=eval_boards)
            log.info(
                "clone vs %s: win rate %.3f, mean VP %.2f",
                opponent,
                result.win_rate,
                result.mean_vp,
            )
    return path


SENSITIVITY_TYPES = {
    "city": ActionType.BUILD_CITY,
    "settlement": ActionType.BUILD_SETTLEMENT,
    "road": ActionType.BUILD_ROAD,
    "dev": ActionType.BUY_DEV_CARD,
    "end_turn": ActionType.END_TURN,
}


@torch.no_grad()
def conditioning_sensitivity(
    policy: PolicyNetwork, data: dict, library: StrategyLibrary, config, device, samples: int = 4000
) -> dict:
    """How z moves the action-type choice on the same states: main-phase decisions where at
    least two purchases are legal, evaluated under each archetype's prototype z (and none).
    Reports the mean probability of each purchase type and the mean pairwise KL."""
    states = recompute_states(policy, data, device)
    masks = data["observations"]["masks"]["type"]
    purchases = [int(t) for t in SENSITIVITY_TYPES.values() if t != ActionType.END_TURN]
    rows = np.flatnonzero(masks[:, purchases].sum(1) >= 2)
    rng = np.random.default_rng(0)
    rows = rng.choice(rows, size=min(samples, len(rows)), replace=False) if len(rows) else rows
    out = {"states": len(rows), "archetypes": {}, "mean_kl": 0.0}
    if not len(rows):
        return out
    policy.eval()
    distributions = {}
    variants = {name: library.prototype(name) for name in config.strategy.archetypes}
    variants[NONE] = zero_z()
    for name, z in variants.items():
        z = z.copy()
        z[-1] = 0.0
        probs = []
        for chunk in np.array_split(rows, max(1, len(rows) // 512)):
            obs = _take(data["observations"], chunk)
            obs[STRATEGY_KEY] = np.repeat(z[None], len(chunk), 0)
            mini = _to_tensors(obs, device)
            state = torch.as_tensor(states[chunk], device=device) if states is not None else None
            result = policy(mini, state)
            probs.append(result.distribution(mini["masks"]).logp["type"].exp().cpu().numpy())
        p = np.concatenate(probs)
        distributions[name] = p
        out["archetypes"][name] = {k: float(p[:, t].mean()) for k, t in SENSITIVITY_TYPES.items()}
    names = list(config.strategy.archetypes)
    kls = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pa, pb = distributions[a] + 1e-8, distributions[b] + 1e-8
            kls.append(float((0.5 * (pa * np.log(pa / pb) + pb * np.log(pb / pa))).sum(1).mean()))
    out["mean_kl"] = float(np.mean(kls)) if kls else 0.0
    return out


def write_profile(policy, config, library, path: Path, boards: int, seed: int = 10_000) -> list:
    """Win rate and build statistics of `policy` under every z variant vs heuristic bots."""
    variants = [*config.strategy.archetypes, NONE]
    if policy.num_strategies:
        variants.append(SELECT)
    rows = []
    for variant in variants:
        for opponent in ("heuristic",):
            result = evaluate_policy(
                policy,
                replace(config, strategy=replace(config.strategy, enabled=True)),
                opponent=opponent,
                boards=boards,
                seed=seed,
                strategy=variant,
                library=library,
            )
            record = {"variant": variant, "opponent": opponent, "games": result.games}
            record["win_rate"] = result.win_rate
            record["mean_vp"] = result.mean_vp
            for key in result.records[0]:
                if key.startswith("stat_"):
                    record[key] = float(np.mean([r[key] for r in result.records]))
            if variant == SELECT:
                picks = [c for r in result.records for c in r["strategy_choices"].split("-") if c]
                for i, name in enumerate([*config.strategy.archetypes, NONE]):
                    record[f"selected_{name}"] = picks.count(str(i)) / max(1, len(picks))
            rows.append(record)
            log.info(
                "z %s vs %s: win rate %.3f, mean VP %.2f, %s",
                variant,
                opponent,
                result.win_rate,
                result.mean_vp,
                ", ".join(f"{k[5:]} {v:.2f}" for k, v in record.items() if k.startswith("stat_")),
            )
    fields = sorted(
        {k for r in rows for k in r}, key=lambda k: (k not in ("variant", "opponent"), k)
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows
