from collections import Counter

import pytest

from carla.engine.config import EngineConfig
from carla.engine.record import GameRecord
from carla.eval import run_tournament, schedule, summarize, wilson_interval
from carla.eval.runner import GameResult


def test_wilson_interval():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-4) and high == pytest.approx(0.5962, abs=1e-4)
    assert wilson_interval(0, 10)[0] == pytest.approx(0.0, abs=1e-12)
    assert wilson_interval(0, 10)[1] == pytest.approx(0.2775, abs=1e-4)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(30, 60, 0.99)[1] > wilson_interval(30, 60, 0.95)[1]


def test_schedule_rotates_every_agent_through_every_seat():
    games = schedule(["a", "b", "c", "d"], 8, seed=0)
    seats = Counter((names.index("a")) for _, names in games)
    assert seats == {0: 2, 1: 2, 2: 2, 3: 2}
    assert len({seed for seed, _ in games}) == 8
    with pytest.raises(ValueError, match="need 4 agents"):
        schedule(["a"], 4, seed=0)


def test_summary_counts_seats():
    results = [
        GameResult(0, ("x", "y", "y", "y"), 0, (10, 3, 2, 2), 50, 200),
        GameResult(1, ("y", "x", "y", "y"), 2, (4, 6, 10, 2), 60, 250),
    ]
    summary = summarize(results)
    assert (summary["x"].games, summary["x"].wins) == (2, 1)
    assert (summary["y"].games, summary["y"].wins) == (6, 1)
    assert summary["x"].mean_vp == 8.0


def test_tournament_is_identical_in_parallel(tmp_path):
    kwargs = {
        "agents": ["heuristic", "random", "random", "random"],
        "num_games": 8,
        "seeds": (0, 1),
        "engine_config": EngineConfig(max_turns=300),
    }
    serial = run_tournament(**kwargs, num_workers=1)
    parallel = run_tournament(**kwargs, num_workers=2, records_dir=tmp_path)
    assert [r.winner for r in serial.results] == [r.winner for r in parallel.results]
    assert set(serial.per_seed) == {0, 1}
    assert serial.summary["heuristic"].games == 16
    saved = sorted(tmp_path.glob("*.json.gz"))
    assert len(saved) == 16
    assert GameRecord.load(saved[0]).replay().is_over


def test_strength_parity_verdicts_on_synthetic_paired_games():
    import importlib.util
    import random
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "run_strength_parity.py"
    spec = importlib.util.spec_from_file_location("run_strength_parity", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    def rows(bot, wins, boards):
        return [
            {
                "condition": f"{bot}|trading=0",
                "board": b,
                "seat": s,
                "failure": None,
                "heuristic_win": wins(b, s),
                "truncated": False,
                "heuristic_vp": 6,
                "bot_vp": 10,
                "seconds": 1.0,
            }
            for b in range(boards)
            for s in range(4)
        ]

    rng = random.Random(3)
    outcome = {(b, s): rng.random() < 0.3 for b in range(80) for s in range(4)}
    same = rows("catanatron-f", lambda b, s: outcome[b, s], 80)
    same += rows("native-f-upstream", lambda b, s: outcome[b, s], 80)
    gap = rows("catanatron-ab2", lambda b, s: True, 80) + rows(
        "native-ab2-upstream", lambda b, s: s == 0, 80
    )
    report = script.analyze(same + gap, 0.05)
    assert report["pairs"]["catanatron-f|trading=0"]["verdict"] == "equivalent"
    assert report["pairs"]["catanatron-ab2|trading=0"]["verdict"] == "different"
    few = script.analyze(
        rows("catanatron-f", lambda b, s: False, 5)
        + rows("native-f-upstream", lambda b, s: False, 5),
        0.05,
    )
    assert few["pairs"]["catanatron-f|trading=0"]["verdict"] == "insufficient boards"
    assert script.mcnemar_exact(0, 0) == 1.0 and script.mcnemar_exact(10, 0) < 0.01
