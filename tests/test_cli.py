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


def test_train_dry_run(tmp_path):
    pytest.importorskip("torch")
    assert main(["train", "configs/smoke.yaml", f"runtime.output_dir={tmp_path}"]) == 0
    (run,) = tmp_path.iterdir()
    assert {p.name for p in run.iterdir()} == {"config.yaml", "git.txt", "checkpoint_0.pt"}


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
