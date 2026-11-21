import csv
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from carla.cli import main
from carla.config import load_config, save_config
from carla.eval.analysis import analyze_runs, load_run, read_csv
from carla.models.factory import build_policy
from carla.rl.checkpoint import rng_state, save_checkpoint
from carla.rl.evaluate import evaluate_policy, export_evaluation
from carla.rl.experiments import run_sweep
from carla.rl.policy_tournament import run_policy_tournament
from carla.rl.train import train
from carla.utils.logging import CsvJournal


def tiny_config(output, *, evaluation=False, steps=32):
    cfg = load_config("configs/smoke.yaml")
    return replace(
        cfg,
        engine=replace(cfg.engine, max_turns=2),
        model=replace(cfg.model, hidden_dim=16, mlp_layers=1),
        ppo=replace(cfg.ppo, total_env_steps=steps, rollout_steps=4, epochs=1, minibatch_size=8),
        runtime=replace(
            cfg.runtime, device="cpu", envs_per_worker=2, checkpoint_every=1, output_dir=str(output)
        ),
        eval=replace(
            cfg.eval, every_env_steps=16 if evaluation else 0, boards=1, opponents=("random",)
        ),
    )


def weights(path):
    return torch.load(path, weights_only=False)["policy"]


def test_csv_recovery_preserves_committed_rows_and_backs_up_tail(tmp_path):
    path = tmp_path / "metrics.csv"
    log = CsvJournal(path, ("update", "value"))
    log.write({"update": 1, "value": 2})
    log.write({"update": 2, "value": 4})
    recovered = CsvJournal(path, ("update", "value"), committed_rows=1)
    assert recovered.rows == 1
    assert len(read_csv(path)) == 1
    assert len(read_csv(path.with_suffix(".csv.recovery.bak"))) == 2
    recovered.write({"update": 2, "value": 5})
    assert read_csv(path)[1]["value"] == "5"
    with pytest.raises(ValueError, match="missing committed"):
        CsvJournal(path, ("update", "value"), committed_rows=3)


@pytest.mark.parametrize("deterministic", [True, False])
def test_evaluation_preserves_weights_rng_modes_and_board_pairing(deterministic, tmp_path):
    cfg = tiny_config(tmp_path)
    policy = build_policy(cfg.model)
    policy.train()
    policy.trunk.eval()  # Preserve submodule modes too.
    modes = [m.training for m in policy.modules()]
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    state = rng_state()
    first = evaluate_policy(
        policy, cfg, boards=2, seed=20000, opponent="random", deterministic=deterministic
    )
    after = rng_state()
    assert state["python"] == after["python"] == random.getstate()
    assert np.array_equal(state["numpy"][1], after["numpy"][1])
    assert torch.equal(state["torch"], after["torch"])
    if state["torch_cuda"] is not None:
        assert all(
            torch.equal(a, b) for a, b in zip(state["torch_cuda"], after["torch_cuda"], strict=True)
        )
    assert [m.training for m in policy.modules()] == modes
    assert all(torch.equal(value, policy.state_dict()[key]) for key, value in before.items())
    second = evaluate_policy(
        policy, cfg, boards=2, seed=20000, opponent="random", deterministic=deterministic
    )
    assert first.records == second.records
    for board in range(2):
        records = [r for r in first.records if r["board"] == board]
        assert {r["seat"] for r in records} == set(range(4))
        assert len({r["game_seed"] for r in records}) == 1


def test_telemetry_and_evaluation_do_not_change_training(tmp_path):
    cfg = tiny_config(tmp_path / "evaluated", evaluation=True)
    evaluated = train(cfg)
    plain = train(
        replace(
            cfg,
            runtime=replace(cfg.runtime, output_dir=str(tmp_path / "plain")),
            eval=replace(cfg.eval, every_env_steps=0),
        )
    )
    assert all(
        torch.equal(v, weights(plain / "latest.pt")[k])
        for k, v in weights(evaluated / "latest.pt").items()
    )
    updates = read_csv(evaluated / "updates.csv")
    assert "update 4" in (evaluated / "train.log").read_text(encoding="utf-8")
    assert [int(r["env_steps"]) for r in updates] == [8, 16, 24, 32]
    assert all(float(r["collection_seconds"]) > 0 for r in updates)
    assert all("advantage_std" in r and "main_type_entropy" in r for r in updates)
    games = read_csv(evaluated / "evaluation_games.csv")
    assert {int(r["env_steps"]) for r in games} == {0, 16, 32}
    assert all((evaluated / r["checkpoint"]).exists() for r in games)
    assert len(games) == 12


def test_validation_without_retained_snapshots_keeps_resume_and_curves(tmp_path, monkeypatch):
    from importlib import import_module

    train_module = import_module("carla.rl.train")
    saved = []
    original = train_module.save_checkpoint

    def track_checkpoint(path, *args, **kwargs):
        saved.append((Path(path).name, kwargs["env_steps"]))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(train_module, "save_checkpoint", track_checkpoint)
    cfg = tiny_config(tmp_path / "compact", evaluation=True)
    cfg = replace(
        cfg,
        runtime=replace(cfg.runtime, checkpoint_every=3),
        eval=replace(cfg.eval, every_env_steps=8, retain_checkpoints=False),
    )
    run = train(cfg)
    assert [step for name, step in saved if name == "latest.pt"] == [0, 24, 32]
    assert not (run / "checkpoints").exists()
    assert (run / "latest.pt").exists()
    games = read_csv(run / "evaluation_games.csv")
    assert {row["checkpoint"] for row in games} == {""}
    assert analyze_runs([run], tmp_path / "compact_analysis")["runs"] == 1


def test_resume_preserves_episode_totals_and_reconciles_csvs(tmp_path):
    cfg = tiny_config(tmp_path / "resumed", evaluation=True, steps=16)
    run = train(cfg)
    checkpoint = run / "latest.pt"
    state = torch.load(checkpoint, weights_only=False)["training_state"]
    assert len(state["trackers"]) == 2
    rows = read_csv(run / "updates.csv")
    journal = CsvJournal(run / "updates.csv", list(rows[0]), len(rows))
    journal.write(rows[-1])  # Simulate records written after the last durable checkpoint.
    resumed_cfg = replace(cfg, ppo=replace(cfg.ppo, total_env_steps=32))
    train(resumed_cfg, resume=run / "checkpoints" / "step_000000000016.pt")
    reference = train(
        replace(resumed_cfg, runtime=replace(cfg.runtime, output_dir=str(tmp_path / "reference")))
    )
    assert all(
        torch.equal(v, weights(reference / "latest.pt")[k]) for k, v in weights(checkpoint).items()
    )
    assert read_csv(run / "episodes.csv") == read_csv(reference / "episodes.csv")
    assert len(read_csv(run / "updates.csv")) == 4
    assert (run / "updates.csv.recovery.bak").exists()
    assert len(json.loads((run / "metadata.json").read_text())["sessions"]) == 2
    # Per-game results match; timing columns intentionally differ.
    keys = ("env_steps", "checkpoint", "board", "game_seed", "seat", "outcome")
    assert [tuple(r[k] for k in keys) for r in read_csv(run / "evaluation_games.csv")] == [
        tuple(r[k] for k in keys) for r in read_csv(reference / "evaluation_games.csv")
    ]


def test_multiprocess_rollouts_resume_exactly_and_keep_workers_separate(tmp_path):
    cfg = tiny_config(tmp_path / "multi", steps=16)
    cfg = replace(cfg, runtime=replace(cfg.runtime, num_workers=2))
    run = train(cfg)
    first = torch.load(run / "latest.pt", weights_only=False)
    assert first["counters"] == {"env_steps": 16, "updates": 1}
    assert len(first["training_state"]["worker_states"]) == 2
    assert all(len(worker["envs"]) == 2 for worker in first["training_state"]["worker_states"])
    extended = replace(cfg, ppo=replace(cfg.ppo, total_env_steps=30))
    config_path = tmp_path / "multi_config.yaml"
    save_config(cfg, config_path)
    assert (
        main(
            [
                "train",
                str(config_path),
                "--resume",
                str(run / "latest.pt"),
                "ppo.total_env_steps=30",
            ]
        )
        == 0
    )
    reference = train(
        replace(extended, runtime=replace(cfg.runtime, output_dir=str(tmp_path / "reference")))
    )
    resumed = torch.load(run / "latest.pt", weights_only=False)
    assert resumed["counters"] == {"env_steps": 30, "updates": 3}
    assert all(
        torch.equal(value, weights(reference / "latest.pt")[name])
        for name, value in resumed["policy"].items()
    )
    assert read_csv(run / "episodes.csv") == read_csv(reference / "episodes.csv")
    assert {int(row["env_id"]) for row in read_csv(run / "episodes.csv")} == {0, 1, 2, 3}
    assert [int(row["samples"]) for row in read_csv(run / "updates.csv")] == [16, 12, 2]


def test_policy_tournament_rotates_checkpoints_and_bots_on_paired_boards(tmp_path):
    cfg = tiny_config(tmp_path)
    checkpoints = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        checkpoints.append(
            save_checkpoint(tmp_path / f"policy_{seed}.pt", build_policy(cfg.model), config=cfg)
        )
    players = [str(checkpoints[0]), str(checkpoints[1]), "random", "heuristic"]
    first = run_policy_tournament(players, boards=2, seed=20_000, csv_path=tmp_path / "games.csv")
    second = run_policy_tournament(players, boards=2, seed=20_000)
    assert first == second
    assert len(first.records) == 32
    assert len(read_csv(tmp_path / "games.csv")) == 32
    assert (
        main(["policy-tournament", *players, "--boards", "1", "--csv", str(tmp_path / "cli.csv")])
        == 0
    )
    assert len(read_csv(tmp_path / "cli.csv")) == 16
    assert all(entry.games == 8 for entry in first.entries)
    for slot in range(4):
        assert [r["seat"] for r in first.records if r["slot"] == slot].count(0) == 2
    for board in range(2):
        assert len({r["game_seed"] for r in first.records if r["board"] == board}) == 1


def test_sweep_analysis_and_separate_test_exports(tmp_path):
    overrides = [
        "ppo.total_env_steps=16",
        "ppo.epochs=1",
        "model.hidden_dim=16",
        "model.mlp_layers=1",
        "engine.max_turns=2",
        "runtime.device=cpu",
        "runtime.envs_per_worker=2",
        "eval.every_env_steps=8",
        "eval.boards=1",
        "eval.opponents=[random]",
    ]
    root = tmp_path / "sweep"
    options = {
        "gammas": (0.99, 0.999),
        "lambdas": (0.95, 1.0),
        "rollout_steps": (4,),
        "seeds": (0,),
        "overrides": overrides,
    }
    assert run_sweep(Path("configs/smoke.yaml"), root, **options, dry_run=True) == []
    assert not root.exists()
    paths = run_sweep(Path("configs/smoke.yaml"), root, **options)
    assert len(paths) == 4
    assert {
        (
            load_config(path / "config.yaml").ppo.gamma,
            load_config(path / "config.yaml").ppo.gae_lambda,
        )
        for path in paths
    } == {(0.99, 0.95), (0.99, 1.0), (0.999, 0.95), (0.999, 1.0)}
    paths_again = run_sweep(Path("configs/smoke.yaml"), root, **options, resume=True)
    assert paths_again == paths
    for path in paths:
        cfg = load_config(path / "config.yaml")
        policy = build_policy(cfg.model)
        checkpoint = path / "checkpoints" / "step_000000000016.pt"
        policy.load_state_dict(weights(checkpoint))
        test = evaluate_policy(policy, cfg, opponent="random", boards=1, seed=20000)
        export_evaluation(checkpoint, test, path / "evaluation_test_games.csv")
        with pytest.raises(ValueError, match="already contains"):
            export_evaluation(checkpoint, test, path / "evaluation_test_games.csv")
    report = analyze_runs([root], tmp_path / "analysis")
    assert report["runs"] == 4
    summaries = read_csv(tmp_path / "analysis" / "configuration_summary.csv")
    assert len(summaries) == 4 and all(int(r["training_seeds"]) == 1 for r in summaries)
    assert (tmp_path / "analysis" / "training_diagnostics.png").stat().st_size > 1000
    test_report = analyze_runs([root], tmp_path / "test_analysis", split="test")
    assert test_report["runs"] == 4
    # Corrupt logs must be rejected instead of silently double-counting observations.
    rows = read_csv(paths[0] / "updates.csv")
    with (paths[0] / "updates.csv").open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, list(rows[0])).writerow(rows[-1])
    with pytest.raises(ValueError, match="duplicate"):
        load_run(paths[0])
