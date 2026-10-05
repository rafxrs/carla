"""Experiment configuration: typed dataclasses (`schema`) loaded from YAML (`loader`)."""

from carla.config.loader import ConfigError, config_hash, load_config, save_config, to_dict
from carla.config.schema import (
    EnvConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    OpponentPoolConfig,
    OpponentSpec,
    PPOConfig,
    RuntimeConfig,
)

__all__ = [
    "ConfigError",
    "EnvConfig",
    "EvalConfig",
    "ExperimentConfig",
    "ModelConfig",
    "OpponentPoolConfig",
    "OpponentSpec",
    "PPOConfig",
    "RuntimeConfig",
    "config_hash",
    "load_config",
    "save_config",
    "to_dict",
]
