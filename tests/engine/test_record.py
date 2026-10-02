import re

import pytest

from carla.engine.game import Step
from carla.engine.record import GameRecord, ReplayError
from carla.engine.simulate import play_random_game
from carla.engine.text import describe_state


@pytest.fixture(scope="module")
def finished():
    game = play_random_game(21)
    return game, GameRecord.from_game(game, metadata={"agents": ["random"] * 4})


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_save_load_replay_roundtrip(finished, tmp_path, suffix):
    game, record = finished
    loaded = GameRecord.load(record.save(tmp_path / f"game{suffix}"))
    assert loaded.steps == record.steps and loaded.hashes == record.hashes
    replayed = loaded.replay()
    assert replayed.state_hash() == game.state_hash() and replayed.winner == game.winner
    assert replayed.rng.getstate() == game.rng.getstate()
    assert loaded.metadata["result"]["winner"] == game.winner


@pytest.mark.parametrize("k", [0, 1, 16, 17, 333, -1])
def test_replay_loads_any_position(finished, k):
    _, record = finished
    k = len(record) if k == -1 else k
    position = record.replay(k)
    assert position.num_steps == k
    expected = record.hashes[k - 1] if k else record.new_game().state_hash()
    assert position.state_hash() == expected
    if k < len(record):
        position.step(record.steps[k].action, record.steps[k].outcome)


def test_replay_detects_tampering(finished):
    _, record = finished
    roll = next(i for i, s in enumerate(record.steps) if s.action.type.name == "ROLL_DICE")
    other = (1, 2) if record.steps[roll].outcome != (1, 2) else (2, 3)
    steps = list(record.steps)
    steps[roll] = Step(steps[roll].player, steps[roll].action, other)
    tampered = GameRecord(record.seed, record.config, record.board, steps, record.hashes)
    with pytest.raises(ReplayError):
        tampered.replay()


def test_text_log_labels_match_positions(finished):
    _, record = finished
    text = record.to_text()
    assert text.startswith(f"CARLA game | seed {record.seed}")
    assert "== Setup ==" in text and "== Turn 1 | P0 ==" in text and "wins!" in text
    match = re.search(r"^#20\s+(P\d) (.*)$", text, re.MULTILINE)
    assert match
    step = record.steps[19]
    assert match.group(1) == f"P{step.player}"
    assert record.replay(20).num_steps == 20
    assert len(describe_state(record.replay(20)).splitlines()) == 11


def test_records_without_hashes_still_replay():
    game = play_random_game(4)
    record = GameRecord.from_game(game, with_hashes=False)
    assert record.hashes is None
    assert record.replay().state_hash() == game.state_hash()


def test_rejects_unknown_format():
    with pytest.raises(ValueError, match="not a CARLA game record"):
        GameRecord.from_json({"format": "other"})
