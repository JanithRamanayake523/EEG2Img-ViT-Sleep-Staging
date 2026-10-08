"""YAML configuration with dot-notation access and `key.sub=value` command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """A dict whose nested dicts can be read as attributes: cfg.images.size."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _set_dotted(data: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    """Load a YAML file and apply overrides such as ``classifier.epochs=5``.

    Override values are parsed with YAML, so ``true``, ``3``, ``[1,2]`` and ``null`` work.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item}")
        key, value = item.split("=", 1)
        _set_dotted(raw, key.strip(), yaml.safe_load(value))
    return _wrap(raw)


def add_common_args(parser) -> None:
    """Arguments shared by every script in `scripts/`."""
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config.")
    parser.add_argument(
        "--override", "-o", action="append", default=[],
        help="Override a config value, e.g. -o classifier.epochs=3 (repeatable).",
    )
