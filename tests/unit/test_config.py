"""Config loading, dotted CLI overrides, and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from celltemp.config import as_path, load_config, load_yaml, project_root_from_config, save_yaml


def write_cfg(tmp_path: Path, payload: dict) -> Path:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_relative_paths_use_the_config_directory(tmp_path: Path) -> None:
    path = write_cfg(tmp_path, {"a": 1})
    assert project_root_from_config(path) == path.parent.resolve()


def test_as_path_handles_relative_and_absolute_values(tmp_path: Path) -> None:
    assert as_path("data/raw", tmp_path) == tmp_path / "data" / "raw"
    absolute = (tmp_path / "elsewhere").resolve()
    assert as_path(absolute, tmp_path / "root") == absolute


def test_empty_yaml_loads_as_empty_dict(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_yaml(path) == {}


def test_save_yaml_round_trips_and_creates_parent_directories(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "cfg.yaml"
    payload = {"engine": {"integrator": "exact"}, "note": "上部Cell温度"}
    save_yaml(payload, out)
    assert load_yaml(out) == payload


# ---------------------------------------------------------------------------
# dotted overrides -- the CLI surface used throughout the README
# ---------------------------------------------------------------------------
def test_override_parses_current_config_values(tmp_path: Path) -> None:
    path = write_cfg(
        tmp_path,
        {
            "training": {"epochs": 1, "steps_per_epoch": 2},
            "forecast": {"device": "cpu"},
            "project": {"overwrite_run": True},
        },
    )
    cfg = load_config(
        path,
        [
            "training.epochs=50",
            "training.learning_rate=0.001",
            "forecast.device=cuda",
            "project.overwrite_run=false",
        ],
    )
    assert cfg["training"] == {
        "epochs": 50,
        "steps_per_epoch": 2,
        "learning_rate": pytest.approx(0.001),
    }
    assert cfg["forecast"]["device"] == "cuda"
    assert cfg["project"]["overwrite_run"] is False


def test_override_creates_missing_intermediate_keys(tmp_path: Path) -> None:
    path = write_cfg(tmp_path, {})
    cfg = load_config(path, ["monitor.observer.sensor_std=0.05"])
    assert cfg["monitor"]["observer"]["sensor_std"] == pytest.approx(0.05)


def test_override_value_may_contain_equals_signs(tmp_path: Path) -> None:
    path = write_cfg(tmp_path, {})
    cfg = load_config(path, ["project.note=a=b=c"])
    assert cfg["project"]["note"] == "a=b=c"


def test_malformed_override_raises(tmp_path: Path) -> None:
    path = write_cfg(tmp_path, {})
    with pytest.raises(ValueError, match="override must be key=value"):
        load_config(path, ["model.name"])
