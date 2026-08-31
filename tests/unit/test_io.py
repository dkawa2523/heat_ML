from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from celltemp.io import (
    load_system_spec,
    load_trajectories,
    save_system_spec,
    system_spec_from_mapping,
    trajectory_from_frame,
)


def _system_mapping() -> dict:
    return {
        "version": 3,
        "nodes": [
            {"name": "shell", "heat_capacity": 2.0},
            {"name": "core", "heat_capacity": 5.0},
        ],
        "actuators": [
            {"name": "heater", "tau": 3.0},
            {"name": "flow", "tau": 0.0, "learnable": False},
        ],
        "edges": [
            {
                "nodes": ["shell", "core"],
                "conductance": {"type": "constant", "value": 0.4},
            }
        ],
        "sources": [
            {
                "name": "heater_power",
                "node_weights": {"core": 1.0},
                "heat_rate": {
                    "type": "positive_part",
                    "control": "heater",
                    "gain": 0.2,
                },
            }
        ],
        "boundaries": [
            {
                "name": "air",
                "node_weights": {"shell": 1.0},
                "reservoir_temperature": {"intercept": 20.0},
                "conductance": {
                    "type": "power_law",
                    "control": "flow",
                    "reference": 1.0,
                    "offset": 0.1,
                    "scale": 0.4,
                    "exponent": 0.8,
                    "exponent_learnable": True,
                },
            }
        ],
        "sensors": [{"name": "tc_core", "node": "core"}],
    }


def test_system_mapping_expands_named_node_weights() -> None:
    spec = system_spec_from_mapping(_system_mapping())
    assert spec.sources[0].node_weights == (0.0, 1.0)
    assert spec.sensor_names == ("tc_core",)
    assert spec.sensor_nodes == ("core",)


def test_system_yaml_round_trip(tmp_path: Path) -> None:
    original = system_spec_from_mapping(_system_mapping())
    path = tmp_path / "system.yaml"
    save_system_spec(original, path)
    loaded = load_system_spec(path)
    assert loaded == original


def test_frame_adapter_preserves_variable_dt_and_missing_observation() -> None:
    frame = pd.DataFrame(
        {
            "time": [0.0, 0.25, 1.0],
            "tc": [20.0, np.nan, 21.0],
            "power": [0.0, 5.0, 5.0],
        }
    )
    trajectory = trajectory_from_frame(
        case_id="run-1",
        frame=frame,
        time_col="time",
        sensor_cols=["tc"],
        control_cols=["power"],
    )
    np.testing.assert_allclose(trajectory.dt, [0.25, 0.75])
    np.testing.assert_allclose(trajectory.commands[:, 0], [0.0, 5.0])
    assert not trajectory.mask[1, 0]


def test_system_loader_rejects_unknown_weight_node_and_non_mapping_root(tmp_path: Path) -> None:
    mapping = _system_mapping()
    mapping["sources"][0]["node_weights"] = {"missing": 1.0}
    with pytest.raises(ValueError, match="unknown nodes"):
        system_spec_from_mapping(mapping)

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping at its root"):
        load_system_spec(path)


def test_system_loader_rejects_unknown_schema_version() -> None:
    mapping = {**_system_mapping(), "version": 99}
    with pytest.raises(ValueError, match="unsupported system schema version 99"):
        system_spec_from_mapping(mapping)


def test_plain_sensor_names_map_to_same_named_nodes() -> None:
    mapping = _system_mapping()
    mapping["sensors"] = ["core"]
    spec = system_spec_from_mapping(mapping)
    assert spec.sensor_names == ("core",)
    assert spec.sensor_nodes == ("core",)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"time": [0.0, 1.0], "power": [1.0, 1.0]}), "missing columns"),
        (pd.DataFrame({"time": [0.0, 1.0], "tc": [20.0, 21.0]}), "missing columns"),
        (
            pd.DataFrame({"time": [0.0, 1.0], "tc": [20.0, np.inf], "power": [1.0, 1.0]}),
            "not infinite",
        ),
    ],
)
def test_frame_adapter_rejects_incomplete_or_infinite_data(
    frame: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        trajectory_from_frame(
            case_id="bad",
            frame=frame,
            time_col="time",
            sensor_cols=["tc"],
            control_cols=["power"],
        )


def test_load_trajectories_adapts_configured_cases(cae_project: Path, data_cfg: dict) -> None:
    trajectories = load_trajectories(data_cfg, cae_project)
    assert len(trajectories) == 8
    assert trajectories[0].commands.shape == (39, 3)
