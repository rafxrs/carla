import csv
import json
from dataclasses import replace

import pytest
import torch

from carla.cli import main
from carla.config import EnvConfig, ExperimentConfig, ModelConfig
from carla.engine.config import EngineConfig
from carla.eval.diagnostics import export_diagnostics
from carla.models.factory import build_policy
from carla.rl.checkpoint import save_checkpoint
from carla.rl.evaluate import evaluate_checkpoint, evaluate_policy, export_evaluation
from carla.rl.metrics import ENTROPY_FIELDS, EVALUATION_FIELDS
from carla.utils.logging import CsvJournal


@pytest.fixture
def tiny():
    torch.manual_seed(19)
    cfg = ExperimentConfig(
        engine=EngineConfig(max_turns=3, allow_domestic_trade=True),
        model=ModelConfig(hidden_dim=16, mlp_layers=1),
        env=EnvConfig(max_events=8),
    )
    return cfg, build_policy(cfg.model)


@pytest.mark.parametrize("deterministic", [True, False])
def test_mixed_diagnostics_preserve_games_counts_and_reproducibility(tiny, deterministic):
    cfg, policy = tiny
    opponents = ["random", "heuristic25", "heuristic50"]
    kwargs = {
        "opponents": opponents,
        "boards": 1,
        "seed": 123,
        "batch_games": 4,
        "deterministic": deterministic,
    }
    plain = evaluate_policy(policy, cfg, **kwargs)
    detailed = evaluate_policy(policy, cfg, **kwargs, diagnostics=True, trajectories=True)
    repeated = evaluate_policy(policy, cfg, **kwargs, diagnostics=True, trajectories=True)
    assert plain.records == detailed.records == repeated.records
    assert detailed.diagnostics == repeated.diagnostics
    assert plain.diagnostics is None and plain.games == 4
    assert plain.opponent == 'mixed:["random","heuristic25","heuristic50"]'
    detail = detailed.diagnostics
    assert len(detail.players) == 16 and len(detail.summary) == 4 and detail.trajectories
    for record in plain.records:
        seats = json.loads(record["opponent_seats"])
        assert seats[record["seat"]] == "learner"
        assert [kind for seat, kind in enumerate(seats) if seat != record["seat"]] == opponents
        players = [row for row in detail.players if row["learner_seat"] == record["seat"]]
        learner = next(row for row in players if row["role"] == "learner")
        assert learner["decisions"] == record["learner_decisions"]
        assert sum(row["decisions"] for row in players) == learner["game_steps"]
        assert sum(row["completed_turns"] for row in players) == record["turns"]
        assert all(row["setup_settlements"] == row["setup_roads"] == 2 for row in players)


def test_identical_triplet_matches_homogeneous_evaluation(tiny):
    cfg, policy = tiny
    single = evaluate_policy(policy, cfg, opponent="heuristic05", boards=1, seed=123)
    mixed = evaluate_policy(policy, cfg, opponents=["heuristic05"] * 3, boards=1, seed=123)
    assert (
        single.opponent == mixed.opponent
        and single.wins == mixed.wins
        and single.mean_vp == mixed.mean_vp
    )
    assert single.records == [
        {
            key: value
            for key, value in row.items()
            if key not in ("opponent_composition", "opponent_seats")
        }
        for row in mixed.records
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"opponents": []},
        {"opponents": ["random", "heuristic"]},
        {"opponents": ["random"] * 4},
        {"opponents": "random"},
        {"opponents": ["random", "random", "missing"]},
        {"opponent": "missing"},
        {"opponent": ""},
        {"opponent": "random", "opponents": ["random"] * 3},
    ],
)
def test_invalid_opponent_compositions(tiny, kwargs):
    cfg, policy = tiny
    with pytest.raises(ValueError, match="opponent"):
        evaluate_policy(policy, cfg, boards=1, **kwargs)


def test_separate_agent_instances_for_every_game_and_seat(tiny, monkeypatch):
    import carla.rl.evaluate as module

    cfg, policy = tiny
    original = module.make_agent
    agents = []

    def tracked(name):
        agent = original(name)
        agents.append(agent)
        return agent

    monkeypatch.setattr(module, "make_agent", tracked)
    evaluate_policy(policy, cfg, opponents=["heuristic25", "heuristic25", "heuristic50"], boards=1)
    assert len(agents) == len({id(agent) for agent in agents}) == 12


@pytest.mark.parametrize("encoder", ["mlp", "gnn"])
def test_legacy_checkpoint_and_mixed_diagnostic_exports(tiny, tmp_path, encoder):
    cfg, policy = tiny
    if encoder == "gnn":
        cfg = replace(
            cfg,
            model=replace(
                cfg.model,
                encoder="gnn",
                gnn_layers=1,
                set_attention=True,
                belief="gru",
                belief_dim=8,
                trade_decoder="gru",
            ),
        )
        policy = build_policy(cfg.model)
    path = save_checkpoint(tmp_path / "legacy.pt", policy, config=cfg, updates=0, env_steps=0)
    data = torch.load(path, weights_only=False)
    old_env = {
        "learner_seat",
        "win_reward",
        "loss_reward",
        "truncation_reward",
        "vp_reward",
        "shaping",
        "shaping_coef",
        "shaping_gamma",
        "max_events",
    }
    data["config"]["env"] = {
        key: value for key, value in data["config"]["env"].items() if key in old_env
    }
    data["config"]["ppo"].pop("trade_entropy_coef")
    torch.save(data, path)
    result = evaluate_checkpoint(
        path,
        opponents=["random", "heuristic25", "heuristic50"],
        boards=1,
        seed=123,
        batch_games=4,
        device="cpu",
        diagnostics=True,
        trajectories=True,
    )
    assert result.games == 4 and len(result.diagnostics.players) == 16
    assert all(row["checkpoint"] == str(path.resolve()) for row in result.diagnostics.players)
    mixed_csv = tmp_path / "mixed.csv"
    export_evaluation(path, result, mixed_csv)
    with mixed_csv.open() as stream:
        reader = csv.DictReader(stream)
        assert set(EVALUATION_FIELDS) <= set(reader.fieldnames)
        assert len(list(reader)) == 4
    paths = export_diagnostics(result.diagnostics, tmp_path / "diagnostics.csv")
    assert len(paths) == 5 and all(target.exists() for target in paths)
    before = [target.read_bytes() for target in paths]
    with pytest.raises(ValueError, match="already exists"):
        export_diagnostics(result.diagnostics, paths[0])
    assert before == [target.read_bytes() for target in paths]
    homogeneous = evaluate_checkpoint(path, opponent="random", boards=1, device="cpu")
    legacy_csv = tmp_path / "homogeneous.csv"
    export_evaluation(path, homogeneous, legacy_csv)
    with legacy_csv.open() as stream:
        assert next(csv.reader(stream)) == list(EVALUATION_FIELDS)
    original = legacy_csv.read_bytes()
    with pytest.raises(ValueError, match="separate CSV"):
        export_evaluation(path, result, legacy_csv)
    assert legacy_csv.read_bytes() == original


def test_mixed_recurrent_batches_match_single_game_inference(tiny):
    cfg, _ = tiny
    cfg = replace(
        cfg,
        model=replace(
            cfg.model,
            encoder="gnn",
            gnn_layers=1,
            set_attention=True,
            belief="gru",
            belief_dim=8,
            trade_decoder="gru",
        ),
    )
    policy = build_policy(cfg.model)
    kwargs = {"opponents": ["heuristic25", "heuristic25", "heuristic50"], "boards": 1, "seed": 123}
    assert (
        evaluate_policy(policy, cfg, batch_games=1, **kwargs).records
        == evaluate_policy(policy, cfg, batch_games=4, **kwargs).records
    )


def test_cli_mixed_diagnostics(tiny, tmp_path, capsys):
    cfg, policy = tiny
    path = save_checkpoint(tmp_path / "policy.pt", policy, config=cfg, updates=0, env_steps=0)
    assert (
        main(
            [
                "evaluate",
                str(path),
                "--opponents",
                "random",
                "heuristic25",
                "heuristic50",
                "--boards",
                "1",
                "--seed",
                "123",
                "--device",
                "cpu",
                "--batch-games",
                "4",
                "--diagnostics",
                "--diagnostics-csv",
                str(tmp_path / "diagnostics.csv"),
                "--trajectories",
            ]
        )
        == 0
    )
    assert "behavioural diagnostics:" in capsys.readouterr().out
    assert (tmp_path / "diagnostics_summary.csv").exists()
    with pytest.raises(SystemExit):
        main(
            [
                "evaluate",
                str(path),
                "--opponent",
                "random",
                "--opponents",
                "random",
                "random",
                "random",
            ]
        )


def test_legacy_training_journals_keep_their_original_columns(tmp_path):
    path = tmp_path / "updates.csv"
    legacy = CsvJournal(path, ("update", "entropy"))
    legacy.write({"update": 1, "entropy": 0.5})
    before = path.read_bytes()
    resumed = CsvJournal(
        path, ("update", "entropy", *ENTROPY_FIELDS), 1, optional_fields=ENTROPY_FIELDS
    )
    resumed.write({"update": 2, "entropy": 0.4, **dict.fromkeys(ENTROPY_FIELDS, 0.0)})
    assert path.read_bytes().startswith(before)
    with path.open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["update", "entropy"] and len(list(reader)) == 2
    with pytest.raises(ValueError, match="unknown CSV fields"):
        resumed.write({"unrecognized": 1})


def test_normal_evaluation_never_replays_for_diagnostics(tiny, monkeypatch):
    import carla.rl.evaluate as module

    cfg, policy = tiny

    def forbidden(*args, **kwargs):
        raise AssertionError("normal evaluation replayed a game")

    monkeypatch.setattr(module, "diagnose_game", forbidden)
    assert evaluate_policy(policy, cfg, boards=1).diagnostics is None
