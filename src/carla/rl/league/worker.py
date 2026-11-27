"""League rollout processes.

Each process keeps, per learning player, its own environments (so episodes in flight stay
with their learner across turns), a cache of frozen networks keyed by player, and a
`TableLeague` that turns the coordinator's matchmaking plan into opponents for new episodes.
"""

import multiprocessing as mp
import traceback
from contextlib import suppress
from multiprocessing.connection import Connection

import numpy as np
import torch

from carla.agents import make_agent
from carla.config.loader import from_dict
from carla.config.schema import ExperimentConfig
from carla.engine.constants import NUM_PLAYERS
from carla.envs.catan_env import CatanEnv, ExternalAgent
from carla.models.base import PolicyNetwork
from carla.models.factory import build_policy
from carla.rl.league.core import Plan, final_scores, pairwise_result
from carla.rl.metrics import EpisodeTracker, RolloutDiagnostics
from carla.rl.process_rollouts import _random_state, _restore_random_state
from carla.rl.rollout import collect_rollouts, load_reference, reset_envs
from carla.rl.strategies import StrategyAssigner
from carla.utils.seeding import derive_seed, seed_everything

BOT_PREFIX = "bot:"


class TableLeague:
    """The `League` interface of `collect_rollouts` for one learner's environments."""

    def __init__(self, learner: str, networks: dict, rng: np.random.Generator, config) -> None:
        self.learner = learner
        self.networks = networks
        self.rng = rng
        self.config = config
        self.plan: Plan | None = None
        self.main_ids: set[str] = set()
        self.objective = "win"
        self.rank_bonus = 0.0
        self.records: list[dict] = []

    def begin_rollout(self) -> None:
        pass

    def learner_variant(self) -> str | None:
        return None if self.plan is None else self.plan.learner_variant

    def sample(self) -> list:
        assert self.plan is not None, "a plan must arrive before episodes start"
        index = int(self.rng.integers(len(self.plan.tables)))
        agents = []
        for player, variant in self.plan.tables[index]:
            if player.startswith(BOT_PREFIX):
                agent = make_agent(player.removeprefix(BOT_PREFIX))
                agent.league_key = player
            else:
                key = player if variant is None else f"{player}@{variant}"
                agent = ExternalAgent(key, f"net:{player}", variant)
                agent.league_key = key
            agents.append(agent)
        return agents

    def network(self, key: str, policy: PolicyNetwork) -> PolicyNetwork:
        name = key.removeprefix("net:")
        if name == self.learner:
            return policy
        return self.networks[name]

    def terminal_reward(self, env: CatanEnv, reward: float, done: bool) -> float:
        """Exploiter objectives (see `LeagueConfig.exploiter_objective`), at the last step."""
        if not done or self.objective == "win" or env.game.is_truncated:
            return reward
        winner = env.game.winner
        mains = self.main_seats(env)
        if self.objective == "main_loses":
            # Replace the own-outcome terminal reward by: did any main agent win?
            own = env.config.win_reward if winner == env.learner else env.config.loss_reward
            target = env.config.loss_reward if winner in mains else env.config.win_reward
            return reward - own + target
        scores = final_scores(winner, [env.game.victory_points(p) for p in range(NUM_PLAYERS)])
        ahead = sum(scores[env.learner] > scores[seat] for seat in mains)
        return reward + self.rank_bonus * ahead

    def main_seats(self, env: CatanEnv) -> list[int]:
        out = []
        for seat in range(NUM_PLAYERS):
            if seat == env.learner:
                continue
            key = getattr(env.seat_agent(seat), "league_key", "")
            if key.split("@", 1)[0] in self.main_ids:
                out.append(seat)
        return out

    def episode_end(self, env: CatanEnv, record: dict | None) -> None:
        """Pairwise results of the learner against each opponent seat, for the payoff."""
        game = env.game
        scores = final_scores(game.winner, [game.victory_points(p) for p in range(NUM_PLAYERS)])
        results = []
        for seat in range(NUM_PLAYERS):
            if seat == env.learner:
                continue
            agent = env.seat_agent(seat)
            results.append(
                {
                    "key": getattr(agent, "league_key", agent.name),
                    "result": pairwise_result(scores[env.learner], scores[seat]),
                    "stats": env.aec.stats[seat].normalized().tolist(),
                }
            )
        payload = {
            "learner": self.learner,
            "won": game.winner == env.learner,
            "truncated": game.is_truncated,
            "has_main": bool(self.main_seats(env)),
            "opponents": results,
            "learner_stats": env.aec.stats[env.learner].normalized().tolist(),
            "variant": "" if env.strategy is None else env.strategy.label,
        }
        self.records.append(payload)

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def restore(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]


def _build(config_dict: dict, weights: dict) -> PolicyNetwork:
    config = from_dict(ExperimentConfig, config_dict)
    network = build_policy(config.model)
    network.load_state_dict(weights)
    network.requires_grad_(False)
    return network.eval()


def _seated(groups: dict) -> set[str]:
    out = set()
    for group in groups.values():
        for env in group["envs"] or []:
            for agent in env.opponents:
                if isinstance(agent, ExternalAgent):
                    out.add(agent.key.removeprefix("net:"))
    return out


def _drop_retired(networks: dict, retired: set[str], groups: dict) -> None:
    seated = _seated(groups)
    for name in list(retired):
        if name not in seated:
            networks.pop(name, None)
            retired.discard(name)


def _new_group(config, worker_id: int, learner: str, networks: dict, library) -> dict:
    rng = np.random.default_rng(derive_seed(config.seed, "league", worker_id, learner))
    return {
        "envs": None,
        "observations": None,
        "trackers": None,
        "league": TableLeague(learner, networks, rng, config),
        "strategies": StrategyAssigner(
            config.strategy,
            library,
            derive_seed(config.seed, "league-strategy", worker_id, learner),
        ),
    }


def _worker(conn: Connection, worker_id: int, config: ExperimentConfig, library, saved) -> None:
    groups: dict[str, dict] = {}
    try:
        torch.set_num_threads(config.runtime.torch_threads_per_worker)
        seed_everything(derive_seed(config.seed, "league-worker", worker_id))
        networks: dict[str, PolicyNetwork] = {}
        learner_policies: dict[str, PolicyNetwork] = {}
        retired: set[str] = set()
        reference = load_reference(config.ppo)
        if saved is not None:
            _restore_random_state(saved["rng"])
            for name, restored in saved.get("groups", {}).items():
                group = groups[name] = _new_group(config, worker_id, name, networks, library)
                group["league"].restore(restored["league"])
                group["strategies"].restore(restored["strategies"])
                for key in ("envs", "observations", "trackers"):
                    group[key] = restored[key]
        conn.send(("ready", None))
        while True:
            command, payload = conn.recv()
            if command == "stop":
                break
            if command == "add_network":
                networks[payload["id"]] = _build(payload["config"], payload["weights"])
                conn.send(("ok", None))
                continue
            if command == "remove_network":
                # Episodes in flight may still seat the player; drop it once none does.
                retired.add(payload["id"])
                _drop_retired(networks, retired, groups)
                conn.send(("ok", None))
                continue
            if command == "reset_learner":
                groups.pop(payload["id"], None)
                conn.send(("ok", None))
                continue
            if command == "state":
                conn.send(
                    (
                        "ok",
                        {
                            "rng": _random_state(),
                            "groups": {
                                k: {
                                    "envs": g["envs"],
                                    "observations": g["observations"],
                                    "trackers": g["trackers"],
                                    "league": g["league"].state(),
                                    "strategies": g["strategies"].state(),
                                }
                                for k, g in groups.items()
                            },
                        },
                    )
                )
                continue
            if command != "collect":
                raise ValueError(f"unknown league worker command: {command}")
            learner = payload["learner"]
            for name, (config_dict, weights) in payload["learners"].items():
                if name not in learner_policies:
                    learner_policies[name] = build_policy(
                        from_dict(ExperimentConfig, config_dict).model
                    ).eval()
                learner_policies[name].load_state_dict(weights)
                # Current learners also play as opponents (self-play, exploiters' targets).
                networks[name] = learner_policies[name]
            policy = learner_policies[learner]
            group = groups.get(learner)
            if group is None:
                group = groups[learner] = _new_group(config, worker_id, learner, networks, library)
            league = group["league"]
            league.networks = networks
            league.plan = payload["plan"]
            league.main_ids = set(payload["main_ids"])
            league.objective = payload["objective"]
            league.rank_bonus = payload["rank_bonus"]
            league.records = []
            if group["envs"] is None:
                group["envs"] = [
                    CatanEnv(config.engine, config.env, league.sample())
                    for _ in range(config.runtime.envs_per_worker)
                ]
                group["trackers"] = [EpisodeTracker() for _ in group["envs"]]
                group["observations"] = reset_envs(
                    group["envs"],
                    policy,
                    league,
                    [
                        derive_seed(
                            config.seed, "league-env", worker_id, learner, i, payload["epoch"]
                        )
                        for i in range(len(group["envs"]))
                    ],
                    group["strategies"],
                )
            records: list[dict] = []
            diagnostics = RolloutDiagnostics()
            batches, observations = collect_rollouts(
                group["envs"],
                policy,
                payload["steps"],
                observations=group["observations"],
                policy_version=payload["version"],
                trackers=group["trackers"],
                episode_records=records,
                diagnostics=diagnostics,
                env_steps=payload["env_steps"],
                env_id_offset=worker_id * config.runtime.envs_per_worker,
                league=league,
                strategies=group["strategies"],
                reference=reference,
                reference_heads=config.ppo.reference_kl_heads,
            )
            group["observations"] = observations
            _drop_retired(networks, retired, groups)
            conn.send(("ok", (batches, records, diagnostics, league.records)))
    except BaseException:
        with suppress(BrokenPipeError, OSError):
            conn.send(("error", traceback.format_exc()))
    finally:
        for group in groups.values():
            for env in group["envs"] or []:
                env.close()
        conn.close()


class LeagueRolloutPool:
    """Synchronous league rollout processes (see the module docstring)."""

    def __init__(self, config: ExperimentConfig, library, saved: list | None = None) -> None:
        self.config = config
        self.connections: list[Connection] = []
        self.processes: list[mp.Process] = []
        self.sent_versions: dict[str, int] = {}
        context = mp.get_context("spawn")
        try:
            for i in range(config.runtime.num_workers):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker,
                    args=(child, i, config, library, None if saved is None else saved[i]),
                    name=f"carla-league-{i}",
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for connection in self.connections:
                self._receive(connection, "ready")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _receive(connection: Connection, expected: str = "ok"):
        try:
            status, value = connection.recv()
        except EOFError as exc:
            raise RuntimeError("league rollout process exited unexpectedly") from exc
        if status == "error":
            raise RuntimeError(f"league rollout process failed:\n{value}")
        if status != expected:
            raise RuntimeError(f"unexpected league rollout response: {status}")
        return value

    def broadcast(self, command: str, payload: dict) -> None:
        for connection in self.connections:
            connection.send((command, payload))
        for connection in self.connections:
            self._receive(connection)

    def add_network(self, player_id: str, config_dict: dict, weights: dict) -> None:
        self.broadcast("add_network", {"id": player_id, "config": config_dict, "weights": weights})

    def remove_network(self, player_id: str) -> None:
        self.broadcast("remove_network", {"id": player_id})

    def reset_learner(self, player_id: str) -> None:
        self.broadcast("reset_learner", {"id": player_id})

    def collect(
        self,
        learner: str,
        learners: dict,
        plan: Plan,
        *,
        version,
        steps,
        env_steps,
        main_ids,
        objective,
        rank_bonus,
        epoch,
    ):
        """`learners` maps learner id -> (version, config dict, policy); weights of changed
        learners are sent along."""
        updates = {}
        for name, (learner_version, config_dict, policy) in learners.items():
            if self.sent_versions.get(name) != learner_version or name == learner:
                weights = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
                updates[name] = (config_dict, weights)
                self.sent_versions[name] = learner_version
        payload = {
            "learner": learner,
            "learners": updates,
            "plan": plan,
            "version": version,
            "steps": steps,
            "env_steps": env_steps,
            "main_ids": list(main_ids),
            "objective": objective,
            "rank_bonus": rank_bonus,
            "epoch": epoch,
        }
        for connection in self.connections:
            connection.send(("collect", payload))
        batches, records, payoffs = [], [], []
        diagnostics = RolloutDiagnostics()
        for connection in self.connections:
            b, r, d, p = self._receive(connection)
            batches.extend(b)
            records.extend(r)
            payoffs.extend(p)
            diagnostics.merge(d)
        return batches, records, diagnostics, payoffs

    def invalidate(self, player_id: str) -> None:
        """Force the next collect to resend `player_id`'s weights (after a reset)."""
        self.sent_versions.pop(player_id, None)

    def snapshot(self) -> list[dict]:
        for connection in self.connections:
            connection.send(("state", None))
        return [self._receive(connection) for connection in self.connections]

    def close(self) -> None:
        for connection in self.connections:
            with suppress(BrokenPipeError, EOFError, OSError):
                connection.send(("stop", None))
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        for connection in self.connections:
            connection.close()
