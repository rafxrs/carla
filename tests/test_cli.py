import json
import sys
from pathlib import Path

import pytest

from carla.cli import main


def test_simulate_replay_and_config(tmp_path, capsys):
    assert (
        main(
            [
                "simulate",
                "--games",
                "2",
                "--agents",
                "heuristic",
                "random",
                "random",
                "random",
                "--save",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "2 games in" in capsys.readouterr().out
    record = tmp_path / "game_0000.json.gz"
    assert record.exists() and (tmp_path / "game_0000.txt").exists()
    assert main(["replay", str(record), "--step", "16"]) == 0
    assert "step 16" in capsys.readouterr().out
    assert main(["config", "configs/smoke.yaml", "model.hidden_dim=64"]) == 0
    assert "hidden_dim: 64" in capsys.readouterr().out


def test_tournament_command(capsys):
    assert (
        main(
            [
                "tournament",
                "--agents",
                "heuristic",
                "random",
                "random",
                "random",
                "--games",
                "4",
                "--seeds",
                "0",
            ]
        )
        == 0
    )
    assert "win rate" in capsys.readouterr().out


def test_tournament_reports_all_trade_metrics(tmp_path, capsys):
    assert (
        main(
            [
                "tournament",
                "--config",
                "configs/experiments/domestic_trade.yaml",
                "--agents",
                "heuristic",
                "random",
                "random",
                "random",
                "--games",
                "1",
                "--seeds",
                "0",
                "--records",
                str(tmp_path),
            ]
        )
        == 0
    )
    report = capsys.readouterr().out
    assert "proposals" in report
    assert "invitations" in report
    assert "counteroffers" in report
    assert "counter feasibility" in report


@pytest.mark.parametrize("encoder", ["mlp", "gnn"])
def test_train_resume_and_evaluate(tmp_path, capsys, encoder):
    torch = pytest.importorskip("torch")
    overrides = [
        f"model.encoder={encoder}",
        "model.gnn_layers=2",
        "ppo.epochs=1",
        "ppo.rollout_steps=8",
        "model.hidden_dim=16",
        "model.mlp_layers=1",
        "engine.max_turns=12",
        f"runtime.output_dir={tmp_path}",
    ]
    assert main(["train", "configs/smoke.yaml", "ppo.total_env_steps=8", *overrides]) == 0
    (run,) = tmp_path.iterdir()
    checkpoint = run / "latest.pt"
    assert {p.name for p in run.iterdir()} == {
        "config.yaml",
        "git.txt",
        "latest.pt",
        "metadata.json",
        "train.log",
        "updates.csv",
        "episodes.csv",
        "evaluation_games.csv",
    }
    first = torch.load(checkpoint, weights_only=False)
    assert first["counters"] == {"env_steps": 8, "updates": 1}
    assert first["optimizer"]["state"]
    assert all(torch.isfinite(value).all() for value in first["policy"].values())
    assert (
        main(
            [
                "evaluate",
                str(checkpoint),
                "--opponent",
                "random",
                "--boards",
                "1",
                "--device",
                "cpu",
                "--batch-games",
                "4",
            ]
        )
        == 0
    )
    assert "policy vs random:" in capsys.readouterr().out
    assert (
        main(
            [
                "train",
                "configs/smoke.yaml",
                "--resume",
                str(checkpoint),
                "ppo.total_env_steps=16",
                *overrides,
            ]
        )
        == 0
    )
    second = torch.load(checkpoint, weights_only=False)
    assert second["counters"] == {"env_steps": 16, "updates": 2}
    assert "total_env_steps: 16" in (run / "config.yaml").read_text(encoding="utf-8")
    uninterrupted = tmp_path / "uninterrupted"
    assert (
        main(
            [
                "train",
                "configs/smoke.yaml",
                "ppo.total_env_steps=16",
                *overrides[:-1],
                f"runtime.output_dir={uninterrupted}",
            ]
        )
        == 0
    )
    (reference_run,) = uninterrupted.iterdir()
    reference = torch.load(reference_run / "latest.pt", weights_only=False)
    # GNN scatter-adds on CUDA are not bitwise deterministic; allow float rounding noise.
    for name, value in second["policy"].items():
        torch.testing.assert_close(value, reference["policy"][name], rtol=1e-5, atol=1e-5)


def test_batched_training_exact_budget_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    overrides = [
        "runtime.envs_per_worker=3",
        "ppo.rollout_steps=2",
        "ppo.epochs=1",
        "model.hidden_dim=16",
        "model.mlp_layers=1",
        "engine.max_turns=12",
        f"runtime.output_dir={tmp_path}",
    ]
    assert main(["train", "configs/smoke.yaml", "ppo.total_env_steps=12", *overrides]) == 0
    (run,) = tmp_path.iterdir()
    checkpoint = run / "latest.pt"
    first = torch.load(checkpoint, weights_only=False)
    assert first["counters"] == {"env_steps": 12, "updates": 2}
    assert len(first["training_state"]["envs"]) == 3
    assert len(first["training_state"]["observations"]) == 3
    assert (
        main(
            [
                "train",
                "configs/smoke.yaml",
                "--resume",
                str(checkpoint),
                "ppo.total_env_steps=24",
                *overrides,
            ]
        )
        == 0
    )
    resumed = torch.load(checkpoint, weights_only=False)
    assert resumed["counters"] == {"env_steps": 24, "updates": 4}
    reference_dir = tmp_path / "reference"
    assert (
        main(
            [
                "train",
                "configs/smoke.yaml",
                "ppo.total_env_steps=24",
                *overrides[:-1],
                f"runtime.output_dir={reference_dir}",
            ]
        )
        == 0
    )
    (reference_run,) = reference_dir.iterdir()
    reference = torch.load(reference_run / "latest.pt", weights_only=False)
    assert all(
        torch.equal(value, reference["policy"][name]) for name, value in resumed["policy"].items()
    )
    remainder_dir = tmp_path / "remainder"
    assert (
        main(
            [
                "train",
                "configs/smoke.yaml",
                "ppo.total_env_steps=14",
                *overrides[:-1],
                f"runtime.output_dir={remainder_dir}",
            ]
        )
        == 0
    )
    (remainder_run,) = remainder_dir.iterdir()
    remainder = torch.load(remainder_run / "latest.pt", weights_only=False)
    assert remainder["counters"] == {"env_steps": 14, "updates": 3}


def test_train_rejects_unknown_option():
    with pytest.raises(SystemExit, match="2"):
        main(["train", "configs/smoke.yaml", "--not-an-option"])


@pytest.fixture
def gui_calls(monkeypatch):
    import carla.gui.app

    calls = []
    monkeypatch.setattr(carla.gui.app, "main", calls.append)
    return calls


def test_gui_command_options(gui_calls, tmp_path):
    args = ["gui", "--play", "--seat", "2", "--opponents", "heuristic, random", "--seed", "7"]
    assert (
        main([*args, "--delay", "0.2", "--set", "engine.vp_to_win=8", "runtime.output_dir=x"]) == 0
    )
    (options,) = gui_calls
    assert (options.mode, options.seat, options.opponents) == ("play", 2, ("heuristic", "random"))
    assert options.delay == 0.2 and options.record is None
    assert options.seed == 7 and options.engine.vp_to_win == 8 and options.output_dir == Path("x")
    assert main(["gui", "--watch", "--config", "configs/bots.yaml"]) == 0
    watch = gui_calls[-1]  # Watch mode defaults to the config's eval.agents
    assert watch.mode == "watch" and watch.opponents == ("heuristic", "random", "random", "random")
    assert main(["gui"]) == 0 and gui_calls[-1].mode == "play"
    assert gui_calls[-1].opponents == ("heuristic",) and gui_calls[-1].engine.max_turns is None
    assert gui_calls[-1].engine.allow_domestic_trade  # players trade by default
    record = tmp_path / "game.json"
    record.write_text("{}", encoding="utf-8")
    assert main(["gui", str(record)]) == 0
    assert gui_calls[-1].mode == "replay" and gui_calls[-1].record == record


def test_gui_trading_flag_and_limits_from_a_config(gui_calls):
    assert main(["gui", "--no-trading"]) == 0 and not gui_calls[-1].engine.allow_domestic_trade
    assert main(["gui", "--config", "configs/bots.yaml"]) == 0  # its engine does not trade
    assert gui_calls[-1].engine.allow_domestic_trade  # only the flag turns trading off
    config = "configs/experiments/domestic_trade.yaml"
    args = ["gui", "--config", config, "--set", "engine.max_trade_cards_per_side=2"]
    assert main(args) == 0
    engine = gui_calls[-1].engine
    assert engine.allow_domestic_trade and engine.max_trade_cards_per_side == 2
    assert main([*args, "--no-trading"]) == 0 and not gui_calls[-1].engine.allow_domestic_trade


def test_gui_command_errors(gui_calls, tmp_path, capsys, monkeypatch):
    for bad in (["--opponents", "alphazero"], ["--opponents", ","], ["--seat", "4"]):
        with pytest.raises(SystemExit):
            main(["gui", *bad])
    record = tmp_path / "game.json"
    record.write_text("{}", encoding="utf-8")
    assert main(["gui", str(record), "--watch"]) == 2
    assert main(["gui", str(tmp_path / "missing.json")]) == 2
    assert not gui_calls
    assert main(["gui", "--set", "engine.allow_domestic_trade=true"]) == 0
    assert gui_calls[-1].engine.allow_domestic_trade
    monkeypatch.undo()  # the real GUI from here on, without letting it open a window
    monkeypatch.setitem(sys.modules, "arcade", None)
    assert main(["gui", "--watch"]) == 1
    assert "needs Arcade" in capsys.readouterr().err
    assert main(["gui", "--set", "engine.allow_domestic_trade=true"]) == 1
    assert "needs Arcade" in capsys.readouterr().err


def test_gui_picks_a_random_seed_unless_one_is_given(gui_calls, capsys):
    assert main(["gui"]) == 0 and main(["gui"]) == 0
    first, second = gui_calls
    assert first.seed != second.seed and f"seed {first.seed}" in capsys.readouterr().out
    assert main(["gui", "--seed", "5"]) == 0 and gui_calls[-1].seed == 5
    assert main(["gui", "--config", "configs/bots.yaml"]) == 0 and gui_calls[-1].seed == 0


def test_train_init_from_starts_a_new_run_from_checkpoint_weights(tmp_path):
    torch = pytest.importorskip("torch")
    from carla.config import load_config
    from carla.models import build_policy
    from carla.rl.train import _load_initial_weights

    overrides = [
        "ppo.epochs=1",
        "ppo.rollout_steps=8",
        "ppo.total_env_steps=8",
        "model.hidden_dim=16",
        "model.mlp_layers=1",
        "engine.max_turns=12",
        "runtime.device=cpu",
    ]
    first = tmp_path / "first"
    assert main(["train", "configs/smoke.yaml", *overrides, f"runtime.output_dir={first}"]) == 0
    (source,) = first.iterdir()
    checkpoint = source / "latest.pt"

    config = load_config("configs/smoke.yaml", overrides)
    policy = build_policy(config.model)
    _load_initial_weights(checkpoint, policy, config)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["policy"]
    assert all(torch.equal(value, saved[name]) for name, value in policy.state_dict().items())

    # A new stage may change opponents and reward, but not the architecture.
    second = tmp_path / "second"
    stage = [*overrides, "opponents.pool=[{kind: heuristic}]", "env.vp_reward=0.1"]
    assert (
        main(
            [
                "train",
                "configs/smoke.yaml",
                "--init-from",
                str(checkpoint),
                *stage,
                f"runtime.output_dir={second}",
            ]
        )
        == 0
    )
    (run,) = second.iterdir()
    assert run != source
    resumed = torch.load(run / "latest.pt", weights_only=False)
    assert resumed["counters"] == {"env_steps": 8, "updates": 1}
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["sessions"][-1]["init_from"] == str(checkpoint.resolve())
    assert "initialized policy weights" in (run / "train.log").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="model config differs: hidden_dim"):
        _load_initial_weights(
            checkpoint, policy, load_config("configs/smoke.yaml", ["model.hidden_dim=32"])
        )
    with pytest.raises(ValueError, match="either resume or init_from"):
        main(
            [
                "train",
                "configs/smoke.yaml",
                "--resume",
                str(checkpoint),
                "--init-from",
                str(checkpoint),
            ]
        )
