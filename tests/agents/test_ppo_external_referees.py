"""PPO checkpoints seated under the original JSettlers server (OSP v2 bridge)."""

import importlib.util
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

JAVA = pytest.mark.skipif(
    not os.environ.get("JSETTLERS_JAR") or not os.environ.get("CARLA_JSETTLERS_CLASSES"),
    reason="optional pinned JSettlers runtime required",
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from carla.config import to_dict
    from carla.config.schema import ExperimentConfig, ModelConfig
    from carla.engine.config import EngineConfig
    from carla.models import build_policy

    torch.manual_seed(0)
    model = ModelConfig(
        encoder="gnn", hidden_dim=32, gnn_layers=2, set_attention=True, belief="gru", belief_dim=16
    )
    config = ExperimentConfig(engine=EngineConfig(max_turns=500), model=model)
    path = tmp_path_factory.mktemp("policy space") / "latest.pt"  # a path with a space
    torch.save({"config": to_dict(config), "policy": build_policy(model).state_dict()}, path)
    return path


@JAVA
@pytest.mark.slow
def test_recurrent_checkpoint_completes_a_game_under_the_jsettlers_server(checkpoint, tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "run_jsettlers_benchmark", root / "scripts" / "run_jsettlers_benchmark.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    summary = script.run(1, tmp_path / "run", agent=f"ppo:{checkpoint}", timeout=400)
    assert summary["completed_records"] == 1 and not summary["timed_out"]
    seats = summary["results"][0]["result"]["seats"]
    (ours,) = [s for s in seats if s["agent"]]
    assert ours["decisions"] > 0 and ours["failures"] == [] and ours["fallbacks"] == 0


def test_osp_bridge_accepts_checkpoints_and_refuses_hidden_world_search():
    from carla.agents.osp import SUPPORTED

    assert "native-f" not in SUPPORTED and "catanatron-ab2" not in SUPPORTED
