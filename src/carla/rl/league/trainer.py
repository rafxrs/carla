"""League training coordinator (`carla league`).

One process holds every learning player (its own policy, PPO learner and optimizer) and the
league state; synchronous rollout processes (`LeagueRolloutPool`) play the games. Learning
players take turns: each turn trains one learner for `league.updates_per_turn` PPO updates,
with turns allocated in proportion to `league.schedule` (main, main exploiters, league
exploiters). Before every update the matchmaker draws a plan of four-player tables for that
learner from the current payoff; episodes that start during the update use it.

Snapshots, resets and pruning follow `carla.rl.league.core`. Every learner's newest weights are
also written as a plain checkpoint (`learners/<id>.pt`, usable as `ppo:<path>`), and the whole
league is resumable from `league_state.pt`.
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from carla.config import save_config, to_dict
from carla.config.loader import from_dict
from carla.config.schema import ExperimentConfig
from carla.envs.strategy import SELECT
from carla.models.base import PolicyNetwork
from carla.models.factory import build_policy
from carla.rl.checkpoint import restore_rng_state, rng_state, save_checkpoint
from carla.rl.evaluate import evaluate_policy
from carla.rl.league.core import (
    LEAGUE_EXPLOITER,
    MAIN,
    MAIN_EXPLOITER,
    ROLES,
    LeaguePlayer,
    Matchmaker,
    Payoff,
    elo_like,
    player_of,
    prune_candidates,
    ready_to_checkpoint,
)
from carla.rl.league.worker import BOT_PREFIX, LeagueRolloutPool
from carla.rl.metrics import EPISODE_FIELDS, EVALUATION_FIELDS, UPDATE_FIELDS
from carla.rl.ppo import PPOLearner
from carla.rl.strategies import LIBRARY_KEY, load_library
from carla.rl.train import resolve_device, validate_strategy
from carla.utils.logging import CsvJournal, write_json
from carla.utils.runs import create_run_dir, record_session
from carla.utils.seeding import derive_seed, seed_everything

log = logging.getLogger(__name__)

LEAGUE_UPDATE_FIELDS = (
    "learner",
    "role",
    "generation",
    "learner_steps",
    "branches",
    "strength",
    *UPDATE_FIELDS,
)
LEAGUE_EPISODE_FIELDS = ("learner", "role", *EPISODE_FIELDS)
LEAGUE_EVALUATION_FIELDS = ("learner", *EVALUATION_FIELDS)
LEAGUE_EVENT_FIELDS = ("total_steps", "event", "player", "parent", "role", "detail")


@dataclass
class LearnerSlot:
    player: LeaguePlayer
    policy: PolicyNetwork
    learner: PPOLearner
    steps: int = 0
    checkpoint_step: int = 0
    updates: int = 0


def validate_league(config: ExperimentConfig) -> None:
    validate_strategy(config)
    league = config.league
    if len(league.schedule) != 3 or any(w < 0 for w in league.schedule):
        raise ValueError("league.schedule needs three nonnegative weights")
    if league.main_agents < 1:
        raise ValueError("a league needs at least one main agent")
    counts = (league.main_agents, league.main_exploiters, league.league_exploiters)
    if any(c < 0 for c in counts) or any(
        w > 0 and c == 0 for w, c in zip(league.schedule, counts, strict=True)
    ):
        raise ValueError("league.schedule gives updates to a role without players")
    if len(league.main_branches) != 3 or abs(sum(league.main_branches) - 1) > 1e-6:
        raise ValueError("league.main_branches must be three shares summing to one")
    if league.updates_per_turn <= 0 or league.tables_per_plan <= 0:
        raise ValueError("league.updates_per_turn and tables_per_plan must be positive")
    if config.runtime.num_workers <= 0:
        raise ValueError("league training needs rollout processes (runtime.num_workers >= 1)")
    for name in league.teachers:
        from carla.agents import AGENTS

        if name not in AGENTS:
            raise ValueError(f"unknown league teacher {name!r}")
    for variant in league.variants:
        if variant not in ("select", "none", *config.strategy.archetypes):
            raise ValueError(f"unknown league variant {variant!r}")


class LeagueTrainer:
    def __init__(self, config: ExperimentConfig, run_dir: Path, resume: bool) -> None:
        self.config = config
        self.run_dir = run_dir
        self.device = resolve_device(config.runtime.device)
        self.library = load_library(config, config.league.init_checkpoint)
        self.extra = {LIBRARY_KEY: self.library.to_arrays()} if self.library is not None else None
        self.config_dict = to_dict(config)
        self.rng = np.random.default_rng(derive_seed(config.seed, "league-coordinator"))
        self.matchmaker = Matchmaker(
            config.league, np.random.default_rng(derive_seed(config.seed, "matchmaking"))
        )
        self.payoff = Payoff(config.league.payoff_decay)
        self.players: dict[str, LeaguePlayer] = {}
        self.learners: dict[str, LearnerSlot] = {}
        self.behaviour: dict[str, np.ndarray] = {}
        self.retired: dict[str, str] = {}
        """Pruned players (id -> weights file): episodes in flight may still seat them."""
        self.total_steps = 0
        self.epoch = 0
        self.next_evaluation = config.eval.every_env_steps or 0
        self.last_evaluation = -1
        self.init_weights = self._initial_weights()
        self.pool: LeagueRolloutPool | None = None
        self.state_path = run_dir / "league_state.pt"
        self.resumed = resume and self.state_path.exists()

    # ------------------------------------------------------------------ setup

    def _initial_weights(self) -> dict | None:
        path = self.config.league.init_checkpoint
        if path is None:
            return None
        data = torch.load(path, map_location="cpu", weights_only=False)
        saved = to_dict(from_dict(ExperimentConfig, data["config"]))["model"]
        if saved != self.config_dict["model"]:
            changed = sorted(k for k in saved if saved[k] != self.config_dict["model"].get(k))
            raise ValueError(f"league.init_checkpoint model config differs: {', '.join(changed)}")
        return data["policy"]

    def _new_learner(self, player: LeaguePlayer) -> LearnerSlot:
        policy = build_policy(self.config.model)
        if self.init_weights is not None:
            policy.load_state_dict(self.init_weights)
        strategy = self.config.strategy
        learner = PPOLearner(
            policy,
            self.config.ppo,
            str(self.device),
            strategy_coef=strategy.pseudo_reward_coef if strategy.enabled else 0.0,
            selector_entropy_coef=strategy.selector_entropy_coef,
        )
        player.strategy_aware = policy.uses_strategy
        player.selector = policy.num_strategies > 0
        return LearnerSlot(player, policy, learner)

    def _exploiter_variant(self) -> str | None:
        if not self.config.strategy.enabled:
            return None
        if (
            self.config.league.exploiter_strategy == "select"
            and self.config.model.strategy_selector
        ):
            return SELECT
        archetypes = self.config.strategy.archetypes
        return archetypes[int(self.rng.integers(len(archetypes)))]

    def setup(self) -> None:
        league = self.config.league
        for name in league.teachers:
            pid = BOT_PREFIX + name
            self.players[pid] = LeaguePlayer(pid, "bot", protected=True)
        for spec in league.seed_checkpoints:
            assert spec.path is not None
            pid = f"ckpt:{spec.name or Path(spec.path).stem}"
            data = torch.load(spec.path, map_location="cpu", weights_only=False)
            network = build_policy(from_dict(ExperimentConfig, data["config"]).model)
            self.players[pid] = LeaguePlayer(
                pid,
                "checkpoint",
                protected=True,
                path=spec.path,
                strategy_aware=network.uses_strategy,
                selector=network.num_strategies > 0,
            )
        roles = [
            (MAIN, league.main_agents),
            (MAIN_EXPLOITER, league.main_exploiters),
            (LEAGUE_EXPLOITER, league.league_exploiters),
        ]
        for role, count in roles:
            for i in range(count):
                pid = f"{role}_{i}"
                variant = None if role == MAIN else self._exploiter_variant()
                player = LeaguePlayer(pid, "learner", role=role, variant=variant)
                self.players[pid] = player
                self.learners[pid] = self._new_learner(player)
        for slot in self.learners.values():
            if slot.player.role == MAIN:
                # Like AlphaStar, the league starts with a frozen copy of each main agent.
                self._snapshot(slot, event="initial")

    def _start_pool(self, saved: list | None) -> None:
        self.pool = LeagueRolloutPool(self.config, self.library, saved)
        for player in self.players.values():
            if player.kind in ("historical", "checkpoint"):
                self._send_player(player)
        for pid, path in self.retired.items():
            data = torch.load(path, map_location="cpu", weights_only=False)
            self.pool.add_network(pid, data["config"], data["policy"])
            self.pool.remove_network(pid)

    def _send_player(self, player: LeaguePlayer) -> None:
        assert self.pool is not None and player.path is not None
        data = torch.load(player.path, map_location="cpu", weights_only=False)
        self.pool.add_network(player.id, data["config"], data["policy"])

    # ------------------------------------------------------------------ snapshots and resets

    def _snapshot(self, slot: LearnerSlot, event: str = "snapshot") -> LeaguePlayer:
        player = slot.player
        pid = f"{player.id}:{player.snapshots}"
        player.snapshots += 1
        path = self.run_dir / "league" / "players" / f"{pid.replace(':', '_')}.pt"
        save_checkpoint(
            path,
            slot.policy,
            config=self.config,
            extra=self.extra,
            env_steps=slot.steps,
            updates=slot.learner.version,
        )
        historical = LeaguePlayer(
            pid,
            "historical",
            role=player.role,
            parent=player.id,
            strategy_aware=player.strategy_aware,
            selector=player.selector,
            created_step=self.total_steps,
            path=str(path),
            variant=player.variant,
            generation=player.generation,
        )
        self.players[pid] = historical
        if self.pool is not None:
            self._send_player(historical)
        slot.checkpoint_step = slot.steps
        self._event(event, pid, player.id, player.role, f"variant={player.variant}")
        log.info("league: %s added %s (learner steps %d)", event, pid, slot.steps)
        return historical

    def _reset(self, slot: LearnerSlot) -> None:
        player = slot.player
        if self.init_weights is not None:
            slot.policy.load_state_dict(self.init_weights)
        else:
            fresh = build_policy(self.config.model)
            slot.policy.load_state_dict(fresh.state_dict())
        version = slot.learner.version
        strategy = self.config.strategy
        slot.learner = PPOLearner(
            slot.policy,
            self.config.ppo,
            str(self.device),
            strategy_coef=strategy.pseudo_reward_coef if strategy.enabled else 0.0,
            selector_entropy_coef=strategy.selector_entropy_coef,
        )
        slot.learner.version = version
        player.generation += 1
        player.variant = self._exploiter_variant()
        # Fresh payoff rows: the reset player is a new strategy.
        for table in (self.payoff.games, self.payoff.wins, self.payoff.draws):
            for key in [k for k in table if k[0] == player.id]:
                del table[key]
        for table in (self.payoff.table_games, self.payoff.table_wins):
            for key in [k for k in table if k[0] == player.id]:
                del table[key]
        if self.pool is not None:
            self.pool.reset_learner(player.id)
            self.pool.invalidate(player.id)
        self._event(
            "reset",
            player.id,
            None,
            player.role,
            f"generation={player.generation} variant={player.variant}",
        )
        log.info(
            "league: reset %s (generation %d, variant %s)",
            player.id,
            player.generation,
            player.variant,
        )

    def _after_update(self, slot: LearnerSlot) -> None:
        player = slot.player
        steps_since = slot.steps - slot.checkpoint_step
        if not ready_to_checkpoint(
            player, steps_since, self.players, self.payoff, self.config.league
        ):
            return
        self._snapshot(slot)
        if player.role == MAIN_EXPLOITER or (
            player.role == LEAGUE_EXPLOITER
            and self.rng.random() < self.config.league.league_exploiter_reset_prob
        ):
            self._reset(slot)
        for pid in prune_candidates(
            self.players, self.payoff, self.behaviour, self.config.league.max_historical
        ):
            removed = self.players.pop(pid)
            if removed.path is not None:
                self.retired[pid] = removed.path
            if self.pool is not None:
                self.pool.remove_network(pid)
            self._event("prune", pid, removed.parent, removed.role, "")
            log.info("league: pruned %s", pid)

    # ------------------------------------------------------------------ journals

    def _event(
        self, event: str, player: str, parent: str | None, role: str | None, detail: str
    ) -> None:
        self.journals["events"].write(
            {
                "total_steps": self.total_steps,
                "event": event,
                "player": player,
                "parent": parent or "",
                "role": role or "",
                "detail": detail,
            }
        )

    def _open_journals(self, committed: dict) -> None:
        self.journals = {
            name: CsvJournal(self.run_dir / f"{name}.csv", fields, committed.get(name, 0))
            for name, fields in (
                ("updates", LEAGUE_UPDATE_FIELDS),
                ("episodes", LEAGUE_EPISODE_FIELDS),
                ("evaluation_games", LEAGUE_EVALUATION_FIELDS),
                ("events", LEAGUE_EVENT_FIELDS),
            )
        }

    def write_payoff(self) -> None:
        rows = []
        opponents = sorted({k[1] for k in self.payoff.games})
        for pid in self.learners:
            for away in opponents:
                games = self.payoff.count(pid, away)
                if games:
                    rows.append(
                        {
                            "learner": pid,
                            "opponent": away,
                            "games": round(games, 2),
                            "win_rate": round(self.payoff.win_rate(pid, away), 4),
                        }
                    )
        import csv

        with (self.run_dir / "payoff.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, ["learner", "opponent", "games", "win_rate"])
            writer.writeheader()
            writer.writerows(rows)
        write_json(
            self.run_dir / "roster.json",
            {
                "total_steps": self.total_steps,
                "players": [asdict(p) for p in self.players.values()],
                "learner_steps": {k: v.steps for k, v in self.learners.items()},
                "table_win_rates": {
                    pid: {
                        "all": self.payoff.table_win_rate(pid, "all"),
                        "main": self.payoff.table_win_rate(pid, "main"),
                    }
                    for pid in self.learners
                },
                "behaviour": {
                    k: [round(float(x), 4) for x in v] for k, v in self.behaviour.items()
                },
            },
        )

    # ------------------------------------------------------------------ state

    def save(self) -> None:
        assert self.pool is not None
        for pid, slot in self.learners.items():
            save_checkpoint(
                self.run_dir / "learners" / f"{pid}.pt",
                slot.policy,
                config=self.config,
                extra=self.extra,
                env_steps=slot.steps,
                updates=slot.learner.version,
            )
        state = {
            "config": self.config_dict,
            "learners": {
                pid: {
                    "policy": slot.policy.state_dict(),
                    "optimizer": slot.learner.optimizer.state_dict(),
                    "version": slot.learner.version,
                    "steps": slot.steps,
                    "checkpoint_step": slot.checkpoint_step,
                    "updates": slot.updates,
                }
                for pid, slot in self.learners.items()
            },
            "players": {pid: asdict(p) for pid, p in self.players.items()},
            "payoff": self.payoff.state(),
            "behaviour": self.behaviour,
            "retired": self.retired,
            "total_steps": self.total_steps,
            "epoch": self.epoch,
            "next_evaluation": self.next_evaluation,
            "last_evaluation": self.last_evaluation,
            "rng": self.rng.bit_generator.state,
            "matchmaker_rng": self.matchmaker.rng.bit_generator.state,
            "torch_rng": rng_state(),
            "workers": self.pool.snapshot(),
            "log_rows": {k: v.rows for k, v in self.journals.items()},
        }
        temporary = self.state_path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(self.state_path)
        self.write_payoff()

    def load(self) -> tuple[list, dict]:
        state = torch.load(self.state_path, map_location="cpu", weights_only=False)
        saved = to_dict(from_dict(ExperimentConfig, state["config"]))
        for section in ("engine", "env", "model", "strategy"):
            if saved[section] != self.config_dict[section]:
                raise ValueError(f"league resume config mismatch: {section}")
        if saved["runtime"]["num_workers"] != self.config.runtime.num_workers:
            raise ValueError("league resume config mismatch: runtime.num_workers")
        self.players = {pid: LeaguePlayer(**p) for pid, p in state["players"].items()}
        for pid, data in state["learners"].items():
            slot = self._new_learner(self.players[pid])
            slot.policy.load_state_dict(data["policy"])
            slot.learner.optimizer.load_state_dict(data["optimizer"])
            slot.learner.version = data["version"]
            slot.steps, slot.checkpoint_step = data["steps"], data["checkpoint_step"]
            slot.updates = data["updates"]
            self.learners[pid] = slot
        self.payoff = Payoff.from_state(state["payoff"])
        self.behaviour = state["behaviour"]
        self.retired = state.get("retired", {})
        self.total_steps, self.epoch = state["total_steps"], state["epoch"] + 1
        self.next_evaluation = state["next_evaluation"]
        self.last_evaluation = state.get("last_evaluation", -1)
        self.rng.bit_generator.state = state["rng"]
        self.matchmaker.rng.bit_generator.state = state["matchmaker_rng"]
        restore_rng_state(state["torch_rng"])
        return state["workers"], state["log_rows"]

    # ------------------------------------------------------------------ main loop

    def _next_learner(self) -> LearnerSlot:
        weights = dict(zip(ROLES, self.config.league.schedule, strict=True))
        best, best_score = None, None
        for slot in self.learners.values():
            role_count = sum(1 for s in self.learners.values() if s.player.role == slot.player.role)
            share = weights[slot.player.role] / role_count
            if share <= 0:
                continue
            score = slot.updates / share
            if best_score is None or score < best_score:
                best, best_score = slot, score
        assert best is not None
        return best

    def _record_results(self, slot: LearnerSlot, payloads: list[dict]) -> None:
        for payload in payloads:
            home = payload["learner"]
            self.payoff.update_table(home, "all", payload["won"])
            if payload["has_main"]:
                self.payoff.update_table(home, "main", payload["won"])
            for opponent in payload["opponents"]:
                self.payoff.update(home, opponent["key"], opponent["result"])
                pid = player_of(opponent["key"])
                stats = np.asarray(opponent["stats"], dtype=np.float32)
                previous = self.behaviour.get(pid)
                self.behaviour[pid] = stats if previous is None else 0.98 * previous + 0.02 * stats
            learner_stats = np.asarray(payload["learner_stats"], dtype=np.float32)
            previous = self.behaviour.get(home)
            self.behaviour[home] = (
                learner_stats if previous is None else 0.98 * previous + 0.02 * learner_stats
            )

    def _evaluate(self) -> None:
        cfg = self.config
        if self.last_evaluation == self.total_steps:
            return
        self.last_evaluation = self.total_steps
        for pid, slot in self.learners.items():
            if slot.player.role != MAIN:
                continue
            for opponent in cfg.eval.opponents:
                result = evaluate_policy(
                    slot.policy,
                    cfg,
                    opponent=opponent,
                    boards=cfg.eval.boards,
                    seed=cfg.eval.seed,
                    deterministic=cfg.eval.deterministic,
                    library=self.library,
                )
                log.info(
                    "league validation %s vs %s at %d decisions: %d/%d wins (%.3f)",
                    pid,
                    opponent,
                    slot.steps,
                    result.wins,
                    result.games,
                    result.win_rate,
                )
                for record in result.records:
                    self.journals["evaluation_games"].write(
                        {
                            "learner": pid,
                            "update": slot.learner.version,
                            "env_steps": slot.steps,
                            "evaluation_target": self.total_steps,
                            "training_seconds": 0.0,
                            "wall_seconds": time.perf_counter() - self.started,
                            "checkpoint": f"learners/{pid}.pt",
                            "split": "validation",
                            **record,
                        }
                    )

    def run(self) -> None:
        cfg = self.config
        saved_workers, committed = self.load() if self.resumed else (None, {})
        if not self.resumed:
            self._open_journals({})
            self.setup()
        else:
            self._open_journals(committed)
        self._start_pool(saved_workers)
        assert self.pool is not None
        self.started = time.perf_counter()
        main_ids = [pid for pid, s in self.learners.items() if s.player.role == MAIN]
        if not self.resumed and cfg.eval.every_env_steps:
            self._evaluate()
        try:
            while self.total_steps < cfg.league.total_env_steps:
                slot = self._next_learner()
                for _ in range(cfg.league.updates_per_turn):
                    if self.total_steps >= cfg.league.total_env_steps:
                        break
                    self._train_once(slot, main_ids)
                self.save()
                if cfg.eval.every_env_steps and self.total_steps >= self.next_evaluation:
                    self._evaluate()
                    self.next_evaluation = (
                        self.total_steps // cfg.eval.every_env_steps + 1
                    ) * cfg.eval.every_env_steps
            self.save()
            if cfg.eval.every_env_steps:
                self._evaluate()
        finally:
            self.pool.close()

    def _train_once(self, slot: LearnerSlot, main_ids: list[str]) -> None:
        cfg = self.config
        assert self.pool is not None
        player = slot.player
        plan = self.matchmaker.plan(player, self.players, self.payoff, cfg.league.tables_per_plan)
        learners = {
            pid: (s.learner.version, self.config_dict, s.policy) for pid, s in self.learners.items()
        }
        objective = "win" if player.role == MAIN else cfg.league.exploiter_objective
        start = time.perf_counter()
        batches, records, diagnostics, payloads = self.pool.collect(
            player.id,
            learners,
            plan,
            version=slot.learner.version,
            steps=cfg.ppo.rollout_steps,
            env_steps=slot.steps,
            main_ids=main_ids,
            objective=objective,
            rank_bonus=cfg.league.rank_bonus,
            epoch=self.epoch,
        )
        collection = time.perf_counter() - start
        start = time.perf_counter()
        stats = slot.learner.update(batches)
        update = time.perf_counter() - start
        samples = sum(map(len, batches))
        slot.steps += samples
        slot.updates += 1
        self.total_steps += samples
        self._record_results(slot, payloads)
        for record in records:
            self.journals["episodes"].write(
                {
                    "learner": player.id,
                    "role": player.role,
                    "update": slot.learner.version,
                    **record,
                }
            )
        frozen = [p.id for p in self.players.values() if p.kind != "learner"]
        self.journals["updates"].write(
            {
                "learner": player.id,
                "role": player.role,
                "generation": player.generation,
                "learner_steps": slot.steps,
                "branches": ";".join(f"{k}:{v}" for k, v in sorted(plan.counts.items())),
                "strength": elo_like(self.payoff, player.id, frozen),
                "update": slot.learner.version,
                "env_steps": self.total_steps,
                "samples": samples,
                "optimizer_steps_total": 0,
                "episodes_completed": len(records),
                "terminal_outcomes": sum(r["outcome"] != "truncated" for r in records),
                "truncations": sum(r["outcome"] == "truncated" for r in records),
                "nonzero_rewards": sum(int((b.rewards != 0).sum()) for b in batches),
                "collection_seconds": collection,
                "update_seconds": update,
                "training_seconds": 0.0,
                "evaluation_seconds": 0.0,
                "wall_seconds": time.perf_counter() - self.started,
                "decisions_per_second": samples / (collection + update),
                **stats,
                **diagnostics.result(),
            }
        )
        if slot.updates % cfg.runtime.log_every == 0:
            log.info(
                "league %s (gen %d) | update %d | decisions %d (total %d/%d) | episodes %d | "
                "policy %.4f | value %.4f | entropy %.3f | %s",
                player.id,
                player.generation,
                slot.learner.version,
                slot.steps,
                self.total_steps,
                cfg.league.total_env_steps,
                len(records),
                stats["policy_loss"],
                stats["value_loss"],
                stats["entropy"],
                ";".join(f"{k}:{v}" for k, v in sorted(plan.counts.items())),
            )
        self._after_update(slot)


def train_league(config: ExperimentConfig, resume: Path | None = None) -> Path:
    validate_league(config)
    seed_everything(config.seed, config.runtime.deterministic_torch)
    torch.set_num_threads(max(1, config.runtime.torch_threads_per_worker))
    run_dir = Path(resume).resolve() if resume is not None else create_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    device = resolve_device(config.runtime.device)
    metadata = record_session(run_dir, str(device), resume=resume)
    file_log = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    previous = log.level
    log.setLevel(logging.INFO)
    log.addHandler(file_log)
    try:
        log.info("league run directory %s (learner device %s)", run_dir, device)
        trainer = LeagueTrainer(config, run_dir, resume is not None)
        trainer.run()
        metadata["status"] = "complete"
        metadata["env_steps"] = trainer.total_steps
        write_json(run_dir / "metadata.json", metadata)
    finally:
        log.removeHandler(file_log)
        log.setLevel(previous)
        file_log.close()
    return run_dir
