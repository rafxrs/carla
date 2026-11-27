"""Experiment configuration: typed dataclasses (`schema`) loaded from YAML (`loader`)."""

from carla.config.loader import ConfigError, config_hash, load_config, save_config, to_dict
from carla.config.schema import (
    EnvConfig,
    EvalConfig,
    ExperimentConfig,
    LeagueConfig,
    ModelConfig,
    OpponentPoolConfig,
    OpponentSpec,
    PPOConfig,
    RuntimeConfig,
    StrategyConfig,
)

__all__ = [
    "ConfigError",
    "EnvConfig",
    "EvalConfig",
    "ExperimentConfig",
    "LeagueConfig",
    "ModelConfig",
    "OpponentPoolConfig",
    "OpponentSpec",
    "PPOConfig",
    "RuntimeConfig",
    "StrategyConfig",
    "config_hash",
    "load_config",
    "save_config",
    "to_dict",
]
