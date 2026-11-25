import csv
from dataclasses import replace

import pytest
import torch

from carla.config import load_config
from carla.rl.train import train


def config(output, *, enabled=True, workers=1, steps=64):
    return load_config(
        "configs/smoke.yaml",
        [
            f"runtime.output_dir={output}",
            "runtime.device=cpu",
            f"runtime.num_workers={workers}",
            "runtime.envs_per_worker=2",
            "runtime.checkpoint_every=9999",
            "runtime.log_every=9999",
            f"runtime.collect_training_diagnostics={str(enabled).lower()}",
            "model.hidden_dim=16",
            "model.mlp_layers=1",
            "engine.max_turns=2",
            "env.learner_seat=0",
            "opponents.pool=[{kind: heuristic25, weight: 2}, {kind: heuristic50, weight: 1}]",
            f"ppo.total_env_steps={steps}",
            "ppo.rollout_steps=2",
            "ppo.epochs=1",
            "ppo.minibatch_size=8",
        ],
    )


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def weights(path):
    return torch.load(path / "latest.pt", weights_only=False)["policy"]


@pytest.mark.parametrize("workers", [1, 2])
def test_training_diagnostics_do_not_change_training_and_match_episode_identifiers(
    tmp_path, workers
):
    cfg = config(tmp_path / "enabled", workers=workers)
    enabled = train(cfg)
    disabled = train(
        replace(
            cfg,
            runtime=replace(
                cfg.runtime,
                collect_training_diagnostics=False,
                output_dir=str(tmp_path / "disabled"),
            ),
        )
    )
    assert not (disabled / "training_diagnostics.csv").exists()
    assert rows(enabled / "episodes.csv") == rows(disabled / "episodes.csv")
    assert all(
        torch.equal(value, weights(disabled)[name]) for name, value in weights(enabled).items()
    )
    episodes, diagnostics = rows(enabled / "episodes.csv"), rows(
        enabled / "training_diagnostics.csv"
    )
    assert len(episodes) == len(diagnostics) > 0
    assert len({(row["env_id"], row["episode_id"]) for row in diagnostics}) == len(diagnostics)
    for episode, diagnostic in zip(episodes, diagnostics, strict=True):
        for key in episode:
            assert diagnostic[key] == episode[key]
        assert diagnostic["diagnostics_complete"] == "True"
        assert diagnostic["diagnostic_learner_decisions"] == episode["learner_decisions"]
        assert diagnostic["final_victory_points"] == episode["victory_points"]
        assert int(diagnostic["worker_id"]) == int(episode["env_id"]) // cfg.runtime.envs_per_worker
        assert int(diagnostic["policy_version"]) + 1 == int(episode["update"])
        assert sum(
            int(value) for key, value in diagnostic.items() if key.startswith("action_")
        ) == int(episode["learner_decisions"])
        assert int(diagnostic["setup_settlements"]) == int(diagnostic["setup_roads"]) == 2


@pytest.mark.parametrize("workers", [1, 2])
def test_resume_keeps_active_counters_and_recovers_csv_without_duplicates(tmp_path, workers):
    cfg = config(tmp_path / "resumed", workers=workers, steps=8 * workers)
    run = train(cfg)
    paused = torch.load(run / "latest.pt", weights_only=False)["training_state"]
    envs = (
        paused["envs"]
        if workers == 1
        else [env for worker in paused["worker_states"] for env in worker["envs"]]
    )
    assert all(env.training_diagnostics.diagnostic_learner_decisions == 4 for env in envs)
    path = run / "training_diagnostics.csv"
    with path.open() as stream:
        fields = csv.DictReader(stream).fieldnames
    # Simulate a row flushed after the last committed checkpoint.
    with path.open("a", newline="") as stream:
        csv.DictWriter(stream, fields).writerow({"episode_id": 999})
    extended = replace(cfg, ppo=replace(cfg.ppo, total_env_steps=64))
    train(extended, resume=run / "latest.pt")
    reference = train(
        replace(extended, runtime=replace(extended.runtime, output_dir=str(tmp_path / "reference")))
    )
    assert rows(path) == rows(reference / "training_diagnostics.csv")
    assert rows(run / "episodes.csv") == rows(reference / "episodes.csv")
    assert all(torch.equal(value, weights(reference)[name]) for name, value in weights(run).items())
    assert path.with_suffix(".csv.recovery.bak").exists()
    assert all(row["episode_id"] != "999" for row in rows(path))
    assert all(row["diagnostics_complete"] == "True" for row in rows(path))


def test_enable_on_legacy_resume_marks_only_the_partial_episode_incomplete(tmp_path):
    cfg = config(tmp_path / "legacy", enabled=False, steps=8)
    run = train(cfg)
    checkpoint = run / "latest.pt"
    saved = torch.load(checkpoint, weights_only=False)
    saved["config"]["runtime"].pop("collect_training_diagnostics")
    for env in saved["training_state"]["envs"]:
        del env.__dict__["_collect_training_diagnostics"]
        del env.aec.__dict__["_training_diagnostics"]
    torch.save(saved, checkpoint)
    extended = replace(
        cfg,
        ppo=replace(cfg.ppo, total_env_steps=64),
        runtime=replace(cfg.runtime, collect_training_diagnostics=True),
    )
    train(extended, resume=checkpoint)
    records = rows(run / "training_diagnostics.csv")
    assert records and {row["diagnostics_complete"] for row in records} == {"True", "False"}
    partial = [row for row in records if row["diagnostics_complete"] == "False"]
    assert {row["env_id"] for row in partial} == {"0", "1"}
    assert all(row["first_city_learner_turn"] == row["vp_3_learner_turn"] == "" for row in partial)
    assert all(
        int(row["diagnostic_learner_decisions"]) < int(row["learner_decisions"]) for row in partial
    )
    # Disabling the feature preserves the committed side journal without touching it.
    before = (run / "training_diagnostics.csv").read_bytes()
    disabled = replace(
        extended,
        ppo=replace(extended.ppo, total_env_steps=72),
        runtime=replace(extended.runtime, collect_training_diagnostics=False),
    )
    train(disabled, resume=checkpoint)
    assert (run / "training_diagnostics.csv").read_bytes() == before
    state = torch.load(checkpoint, weights_only=False)["training_state"]
    assert state["log_rows"]["training_diagnostics"] == len(records)
    enabled_again = replace(extended, ppo=replace(extended.ppo, total_env_steps=96))
    train(enabled_again, resume=checkpoint)
    assert (run / "training_diagnostics.csv").read_bytes().startswith(before)


def test_init_from_uses_fresh_diagnostic_counters(tmp_path):
    cfg = config(tmp_path / "source", steps=8)
    source = train(cfg)
    initialized = train(
        replace(
            cfg,
            ppo=replace(cfg.ppo, total_env_steps=4),
            runtime=replace(cfg.runtime, output_dir=str(tmp_path / "initialized")),
        ),
        init_from=source / "latest.pt",
    )
    saved = torch.load(initialized / "latest.pt", weights_only=False)["training_state"]
    assert all(env.training_diagnostics.diagnostic_learner_decisions == 2 for env in saved["envs"])
    assert all(env.training_diagnostics.complete for env in saved["envs"])
    assert all(tracker.episode_id == 0 for tracker in saved["trackers"])
