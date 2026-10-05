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


def test_initial_actuator_uses_row_position_with_duplicate_dataframe_index() -> None:
    frame = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "tc": [20.0, 21.0, 22.0],
            "heater": [10.0, 10.0, 10.0],
            "initial_effective_heater": [3.0, np.nan, np.nan],
        },
        index=[0, 0, 1],
    )
    trajectory = trajectory_from_frame(
        case_id="duplicate_index",
        frame=frame,
        time_col="time",
        sensor_cols=("tc",),
        control_cols=("heater",),
    )
    np.testing.assert_array_equal(trajectory.initial_actuator, [3.0])


def _system_mapping() -> dict:
    return {
        "version": 3,
        "nodes": [
            {"name": "shell", "heat_capacity": 2.0},
            {"name": "core", "heat_capacity": 5.0},
        ],
        "actuators": [
            {"name": "heater", "tau": 3.0, "unit": "W", "role": "heat_input"},
            {
                "name": "flow",
                "tau": 0.0,
                "learnable": False,
                "unit": "m/s",
                "role": "heat_transfer",
            },
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
    assert spec.control_units == ("W", "m/s")
    assert spec.control_roles == ("heat_input", "heat_transfer")


def test_system_yaml_round_trip(tmp_path: Path) -> None:
    original = system_spec_from_mapping(_system_mapping())
    path = tmp_path / "system.yaml"
    save_system_spec(original, path)
    loaded = load_system_spec(path)
    assert loaded == original


def test_sensor_can_observe_a_weighted_node_average(tmp_path: Path) -> None:
    mapping = _system_mapping()
    mapping["sensors"] = [
        {"name": "surface_average", "node_weights": {"shell": 0.25, "core": 0.75}}
    ]
    spec = system_spec_from_mapping(mapping)
    np.testing.assert_allclose(spec.observation_matrix, [[0.25, 0.75]])

    path = tmp_path / "weighted-system.yaml"
    save_system_spec(spec, path)
    assert load_system_spec(path) == spec


def test_frame_adapter_preserves_variable_dt_and_missing_observation() -> None:
    frame = pd.DataFrame(
        {
            "time": [0.0, 0.25, 1.0],
            "tc": [20.0, np.nan, 21.0],
            "power": [0.0, 5.0, 5.0],
            "initial_effective_power": [3.0, np.nan, np.nan],
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
    assert trajectory.initial_actuator is not None
    np.testing.assert_allclose(trajectory.initial_actuator, np.array([3.0]))
    assert not trajectory.mask[1, 0]


@pytest.mark.parametrize(
    ("time_col", "sensor_cols", "control_cols"),
    [
        ("time", ["value"], ["value"]),
        ("time", ["time"], ["value"]),
    ],
)
def test_frame_adapter_rejects_column_role_collisions(
    time_col: str, sensor_cols: list[str], control_cols: list[str]
) -> None:
    frame = pd.DataFrame({"time": [0.0, 1.0], "value": [20.0, 21.0]})
    with pytest.raises(ValueError, match="must be distinct"):
        trajectory_from_frame(
            case_id="ambiguous",
            frame=frame,
            time_col=time_col,
            sensor_cols=sensor_cols,
            control_cols=control_cols,
        )


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


def test_system_loader_rejects_ambiguous_or_misspelled_physics() -> None:
    mapping = _system_mapping()
    mapping["actuators"][0]["lernable"] = True
    with pytest.raises(ValueError, match=r"unknown actuator options.*lernable"):
        system_spec_from_mapping(mapping)

    mapping = _system_mapping()
    mapping["actuators"][0]["learnable"] = "false"
    with pytest.raises(ValueError, match="must be boolean"):
        system_spec_from_mapping(mapping)

    mapping = _system_mapping()
    mapping["edges"][0]["nodes"] = ["shell", "core", "extra"]
    with pytest.raises(ValueError, match="exactly two nodes"):
        system_spec_from_mapping(mapping)


def test_plain_sensor_names_map_to_same_named_nodes() -> None:
    mapping = _system_mapping()
    mapping["sensors"] = ["core"]
    spec = system_spec_from_mapping(mapping)
    assert spec.sensor_names == ("core",)
    assert spec.sensor_nodes == ("core",)


@pytest.mark.parametrize(
    ("sensor", "message"),
    [
        ({"name": "bad", "node": "missing"}, "unknown nodes"),
        (
            {"name": "bad", "node": "core", "node_weights": {"core": 1.0}},
            "either node or node_weights",
        ),
    ],
)
def test_system_loader_rejects_invalid_sensor_mapping(sensor: dict, message: str) -> None:
    mapping = _system_mapping()
    mapping["sensors"] = [sensor]
    with pytest.raises(ValueError, match=message):
        system_spec_from_mapping(mapping)


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


@pytest.mark.parametrize("header", ["time,tc,power,power", "time,tc,power, power "])
def test_csv_rejects_duplicate_headers_before_pandas_renames_them(
    tmp_path: Path, header: str
) -> None:
    (tmp_path / "case.csv").write_text(f"{header}\n0,20,0,10\n1,21,10,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"duplicate column names.*power"):
        load_trajectories(
            {"directory": ".", "sensor_cols": ["tc"], "control_cols": ["power"]}, tmp_path
        )


@pytest.mark.parametrize("columns", [["time", "tc", "tc"], ["time", "tc", " tc "]])
def test_frame_rejects_duplicate_columns_after_stripping(columns: list[str]) -> None:
    frame = pd.DataFrame([[0.0, 20.0, 30.0], [1.0, 21.0, 31.0]], columns=columns)
    with pytest.raises(ValueError, match=r"duplicate column names.*tc"):
        trajectory_from_frame(
            case_id="ambiguous", frame=frame, time_col="time", sensor_cols=["tc"], control_cols=[]
        )


def test_initial_effective_typo_is_rejected_but_unrelated_extra_columns_are_allowed() -> None:
    frame = pd.DataFrame(
        {"time": [0.0, 1.0], "tc": [20.0, 21.0], "power": [0.0, 5.0], "truth": [3.0, 4.0]}
    )
    arguments = {
        "case_id": "case",
        "time_col": "time",
        "sensor_cols": ["tc"],
        "control_cols": ["power"],
    }
    assert trajectory_from_frame(frame=frame, **arguments).initial_actuator is None
    frame["initial_effective_powerr"] = [8.0, np.nan]
    with pytest.raises(ValueError, match=r"unknown controls.*initial_effective_powerr"):
        trajectory_from_frame(frame=frame, **arguments)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        (name, value)
        for name in ("dt", "temp_min", "temp_max")
        for value in (True, "1", np.nan, np.inf, -np.inf)
    ],
)
def test_csv_settings_reject_non_numeric_or_non_finite_limits(
    tmp_path: Path, option: str, value: object
) -> None:
    (tmp_path / "case.csv").write_text("time,tc\n0,10000\n1,10001\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"data\.{option} must be a finite number"):
        load_trajectories({"directory": ".", "sensor_cols": ["tc"], option: value}, tmp_path)


@pytest.mark.parametrize("limits", [{"dt": 0.0}, {"dt": -1.0}, {"temp_min": 30, "temp_max": 20}])
def test_csv_settings_require_positive_dt_and_ordered_temperature_bounds(
    tmp_path: Path, limits: dict
) -> None:
    (tmp_path / "case.csv").write_text("time,tc\n0,20\n1,21\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"must be positive|must not exceed"):
        load_trajectories({"directory": ".", "sensor_cols": ["tc"], **limits}, tmp_path)


def test_initial_observation_requirement_can_be_disabled_for_waveform_analysis(
    tmp_path: Path,
) -> None:
    (tmp_path / "case.csv").write_text("time,tc\n0,\n1,20\n2,21\n", encoding="utf-8")
    cfg = {"directory": ".", "sensor_cols": ["tc"], "allow_missing_temperatures": True}
    with pytest.raises(ValueError, match="initial row needs"):
        load_trajectories(cfg, tmp_path)
    trajectory = load_trajectories(cfg, tmp_path, require_initial_observation=False)[0]
    np.testing.assert_array_equal(trajectory.mask[:, 0], [False, True, True])


def test_csv_header_check_preserves_quoted_names_and_configured_separator(tmp_path: Path) -> None:
    (tmp_path / "case.csv").write_text(
        'time;"tc,center";power;truth\n0;20;1;100\n1;21;2;200\n', encoding="utf-8"
    )
    trajectory = load_trajectories(
        {"directory": ".", "sensor_cols": ["tc,center"], "control_cols": ["power"], "sep": ";"},
        tmp_path,
    )[0]
    np.testing.assert_array_equal(trajectory.temperature[:, 0], [20.0, 21.0])
    np.testing.assert_array_equal(trajectory.commands[:, 0], [1.0])


def test_all_missing_analysis_waveform_does_not_evaluate_extrema_for_bounds(tmp_path: Path) -> None:
    (tmp_path / "case.csv").write_text("time,tc\n0,\n1,\n", encoding="utf-8")
    trajectory = load_trajectories(
        {
            "directory": ".",
            "sensor_cols": ["tc"],
            "allow_missing_temperatures": True,
            "temp_min": 0.0,
            "temp_max": 100.0,
            "dt": 1.0,
        },
        tmp_path,
        require_initial_observation=False,
    )[0]
    assert not trajectory.mask.any()
