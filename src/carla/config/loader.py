"""Strict YAML <-> dataclass conversion with inheritance and command-line overrides.

A YAML file may start with `extends: other.yaml` (path relative to the file); mappings are
merged recursively, lists are replaced. Overrides use dotted keys: ``ppo.learning_rate=1e-4``.
Unknown keys and wrongly typed values raise `ConfigError`.
"""

import dataclasses
import hashlib
import json
import types
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import yaml

from carla.config.schema import ExperimentConfig

T = TypeVar("T")


class ConfigError(ValueError):
    pass


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] = (),
    cls: type[T] = ExperimentConfig,
) -> T:
    """Build a `cls` from defaults, an optional YAML file and `key=value` overrides."""
    data = _read_yaml(Path(path)) if path is not None else {}
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ConfigError(f"override {item!r} is not of the form key=value")
        _set_dotted(data, key.strip(), yaml.safe_load(raw))
    return from_dict(cls, data)


def from_dict(cls: type[T], data: Mapping[str, Any], where: str = "") -> T:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where or 'config'}: expected a mapping, got {data!r}")
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cast(Any, cls))}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where or 'config'}: unknown keys {sorted(unknown)}")
    kwargs = {k: _coerce(hints[k], v, f"{where}.{k}" if where else k) for k, v in data.items()}
    return cls(**kwargs)


def to_dict(config: Any) -> Any:
    """Plain YAML/JSON-compatible structure (tuples become lists)."""
    if dataclasses.is_dataclass(config):
        return {f.name: to_dict(getattr(config, f.name)) for f in dataclasses.fields(config)}
    if isinstance(config, list | tuple):
        return [to_dict(v) for v in config]
    return config


def save_config(config: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(to_dict(config), sort_keys=False), encoding="utf-8")
    return path


def config_hash(config: Any) -> str:
    """Short stable digest, e.g. to name runs."""
    payload = json.dumps(to_dict(config), sort_keys=True)
    return hashlib.blake2b(payload.encode(), digest_size=4).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return _merge(_read_yaml(path.parent / parent), data)


def _merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    *parents, leaf = key.split(".")
    node = data
    for part in parents:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {key!r}: {part!r} is not a section")
    node[leaf] = value


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin, args = get_origin(tp), get_args(tp)
    if dataclasses.is_dataclass(tp):
        return from_dict(cast(Any, tp), value, where)
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{where}: {value!r} is not one of {list(args)}")
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for option in (a for a in args if a is not type(None)):
            try:
                return _coerce(option, value, where)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError("; ".join(errors))
    if origin in (tuple, list):
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        items = [_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if origin is tuple else items
    if tp is float and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if tp is float and isinstance(value, str):  # YAML 1.1 reads "1e-4" as a string
        try:
            return float(value)
        except ValueError:
            pass
    if tp in (int, str, bool) and type(value) is tp:
        return value
    if tp is Any:
        return value
    raise ConfigError(f"{where}: expected {getattr(tp, '__name__', tp)}, got {value!r}")
