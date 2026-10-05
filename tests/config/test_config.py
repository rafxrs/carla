from pathlib import Path

import pytest

from carla.config import (
    ConfigError,
    ExperimentConfig,
    OpponentSpec,
    config_hash,
    load_config,
    save_config,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_defaults():
    cfg = load_config()
    assert cfg == ExperimentConfig()
    assert cfg.model.encoder == "mlp" and cfg.engine.vp_to_win == 10


@pytest.mark.parametrize("path", sorted(CONFIGS.rglob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    cfg = load_config(path)
    assert isinstance(cfg, ExperimentConfig) and cfg.name == path.stem


def test_extends_chain_merges_sections():
    cfg = load_config(CONFIGS / "experiments" / "gnn_attention_belief.yaml")
    assert cfg.model.encoder == "gnn" and cfg.model.set_attention and cfg.model.belief == "gru"
    assert cfg.engine.max_turns == 500  # from base.yaml
    assert cfg.model.hidden_dim == 256


def test_overrides_are_typed():
    cfg = load_config(
        overrides=[
            "ppo.learning_rate=1e-4",
            "model.encoder=gnn",
            "engine.max_turns=null",
            "eval.seeds=[3, 4]",
            "opponents.pool=[{kind: random}, {kind: checkpoint, path: a.pt, weight: 2}]",
        ]
    )
    assert cfg.ppo.learning_rate == pytest.approx(1e-4)
    assert cfg.model.encoder == "gnn" and cfg.engine.max_turns is None
    assert cfg.eval.seeds == (3, 4)
    assert cfg.opponents.pool == (
        OpponentSpec("random"),
        OpponentSpec("checkpoint", 2.0, "a.pt"),
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("model.encoder=transformer", "not one of"),
        ("ppo.learning_rat=0.1", "unknown keys"),
        ("ppo.epochs=2.5", "expected int"),
        ("engine.allow_domestic_trade=maybe", "expected bool"),
        ("runtime.num_workers", "key=value"),
    ],
)
def test_bad_configs_are_rejected(override, message):
    with pytest.raises(ConfigError, match=message):
        load_config(overrides=[override])


def test_save_load_roundtrip(tmp_path):
    cfg = load_config(CONFIGS / "smoke.yaml", ["model.belief=lstm"])
    again = load_config(save_config(cfg, tmp_path / "resolved.yaml"))
    assert again == cfg and config_hash(again) == config_hash(cfg)
    assert config_hash(cfg) != config_hash(load_config())
