from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import yaml

_TOP_LEVEL_OPTIONS = {
    "artifact",
    "data",
    "engine",
    "forecast",
    "monitor",
    "project",
    "seed",
    "split",
    "system",
    "training",
}


def project_root_from_config(config_path: str | Path) -> Path:
    """Resolve all relative project paths from the config file's directory."""
    return Path(config_path).resolve().parent


def as_path(value: str | Path, root: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: YAML root must be a mapping")
    return dict(data)


def save_yaml(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def reject_unknown_keys(values: object, allowed: Collection[str], section: str) -> None:
    """Reject misspelled options without introducing a separate schema system."""
    if not isinstance(values, Mapping):
        raise ValueError(f"{section} must be a mapping")
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {section} options: {sorted(unknown)}")


def validate_config_root(cfg: object) -> None:
    reject_unknown_keys(cfg, _TOP_LEVEL_OPTIONS, "config")


def _set_by_dot_key(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = cfg
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = load_yaml(path)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as error:
            raise ValueError(f"invalid YAML value for override {key}: {raw}") from error
        _set_by_dot_key(cfg, key, value)
    validate_config_root(cfg)
    return cfg
