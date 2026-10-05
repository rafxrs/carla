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


def test_train_dry_run(tmp_path):
    pytest.importorskip("torch")
    assert main(["train", "configs/smoke.yaml", f"runtime.output_dir={tmp_path}"]) == 0
    (run,) = tmp_path.iterdir()
    assert {p.name for p in run.iterdir()} == {"config.yaml", "git.txt", "checkpoint_0.pt"}
