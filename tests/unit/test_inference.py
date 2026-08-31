import json
from pathlib import Path

import numpy as np
import pytest
import torch

from celltemp.artifact import load_artifact, save_artifact
from celltemp.domain import (
    ActuatorSpec,
    BoundarySpec,
    ConstantLawSpec,
    EdgeSpec,
    ReservoirTemperatureSpec,
    ThermalSystemSpec,
    Trajectory,
)
from celltemp.engine import ThermalRCModel
from celltemp.inference import estimate_state, forecast, monitor, sensor_bias_in_gauge


def _cooling_model() -> ThermalRCModel:
    spec = ThermalSystemSpec(
        node_names=("core",),
        heat_capacity=(2.0,),
        edges=(),
        actuators=(ActuatorSpec("coolant", tau=0.0),),
        boundaries=(
            BoundarySpec(
                "cooling",
                (1.0,),
                ReservoirTemperatureSpec(10.0),
                ConstantLawSpec(0.5),
            ),
        ),
    )
    return ThermalRCModel(spec)


def _duplicate_sensor_model() -> ThermalRCModel:
    base = _cooling_model().spec
    return ThermalRCModel(
        ThermalSystemSpec(
            node_names=base.node_names,
            heat_capacity=base.heat_capacity,
            edges=base.edges,
            actuators=base.actuators,
            boundaries=base.boundaries,
            sensor_names=("left", "right"),
            sensor_nodes=("core", "core"),
        )
    )


def _forecast_request(
    time: np.ndarray, initial_temperature: float, *, control_name: str = "coolant"
) -> Trajectory:
    temperature = np.full((len(time), 1), np.nan)
    temperature[0, 0] = initial_temperature
    return Trajectory(
        case_id="case",
        time=time,
        temperature=temperature,
        commands=np.zeros((len(time) - 1, 1)),
        sensor_names=("core",),
        control_names=(control_name,),
    )


def test_forecast_uses_variable_time_intervals() -> None:
    model = _cooling_model()
    request = _forecast_request(np.array([0.0, 1.0, 3.0]), 30.0)
    result = forecast(model, request)
    expected = 10.0 + 20.0 * np.exp(-0.5 / 2.0 * request.time)
    np.testing.assert_allclose(result.sensor_temperature[:, 0], expected, atol=1e-10)


def test_forecast_estimates_hidden_state_from_contiguous_history() -> None:
    spec = ThermalSystemSpec(
        node_names=("core", "shell"),
        heat_capacity=(5.0, 1.0),
        edges=(EdgeSpec("core", "shell", ConstantLawSpec(0.25, learnable=False)),),
        actuators=(ActuatorSpec("heater", tau=0.0, learnable=False),),
        boundaries=(
            BoundarySpec(
                "ambient",
                (0.0, 1.0),
                ReservoirTemperatureSpec(25.0),
                ConstantLawSpec(0.08, learnable=False),
            ),
        ),
        sensor_names=("surface",),
        sensor_nodes=("shell",),
    )
    model = ThermalRCModel(spec)
    time = np.arange(31.0)
    commands = np.zeros((len(time) - 1, 1))
    truth, _ = model.forward_trajectory(
        torch.tensor([80.0, 25.0], dtype=model.capacity.dtype),
        torch.tensor(commands, dtype=model.capacity.dtype),
        torch.tensor(np.diff(time), dtype=model.capacity.dtype),
        torch.tensor([0.0], dtype=model.capacity.dtype),
    )
    truth_array = truth.numpy()

    history_rows = 10
    warm_temperature = np.full((len(time), 1), np.nan)
    warm_temperature[:history_rows, 0] = truth_array[:history_rows, 1]
    warm_request = Trajectory(
        case_id="warm_start",
        time=time,
        temperature=warm_temperature,
        commands=commands,
        sensor_names=("surface",),
        control_names=("heater",),
    )
    cold_request = Trajectory(
        case_id="cold_start",
        time=time,
        temperature=np.concatenate([truth_array[:1, 1:2], np.full((len(time) - 1, 1), np.nan)]),
        commands=commands,
        sensor_names=("surface",),
        control_names=("heater",),
    )

    warm = forecast(model, warm_request)
    cold = forecast(model, cold_request)
    origin = history_rows - 1
    warm_core_rmse = np.sqrt(np.mean((warm.node_temperature[:, 0] - truth_array[origin:, 0]) ** 2))
    cold_core_rmse = np.sqrt(
        np.mean((cold.node_temperature[origin:, 0] - truth_array[origin:, 0]) ** 2)
    )

    assert warm.forecast_origin_index == origin
    np.testing.assert_array_equal(warm.time, time[origin:])
    assert warm_core_rmse < 0.01
    assert warm_core_rmse < cold_core_rmse / 1000.0
    state = estimate_state(model, warm_request, through_index=origin)
    np.testing.assert_allclose(state.temperature.numpy(), warm.node_temperature[0])


def test_forecast_rejects_noncontiguous_history_and_missing_future() -> None:
    model = _cooling_model()
    with_gap = Trajectory(
        case_id="gap",
        time=np.arange(4.0),
        temperature=np.array([[20.0], [np.nan], [19.0], [np.nan]]),
        commands=np.zeros((3, 1)),
        sensor_names=("core",),
        control_names=("coolant",),
    )
    with pytest.raises(ValueError, match="contiguous history prefix"):
        forecast(model, with_gap)

    no_future = Trajectory(
        case_id="no_future",
        time=np.arange(3.0),
        temperature=np.full((3, 1), 20.0),
        commands=np.zeros((2, 1)),
        sensor_names=("core",),
        control_names=("coolant",),
    )
    with pytest.raises(ValueError, match="unobserved future row"):
        forecast(model, no_future)


def test_monitor_uses_zero_mean_bias_gauge_without_a_reference() -> None:
    model = _duplicate_sensor_model()
    time = np.arange(61.0)
    forecast_temperature = np.full((len(time), 2), np.nan)
    forecast_temperature[0] = 30.0
    request = Trajectory(
        case_id="request",
        time=time,
        temperature=forecast_temperature,
        commands=np.zeros((len(time) - 1, 1)),
        sensor_names=("left", "right"),
        control_names=("coolant",),
    )
    truth = forecast(model, request).sensor_temperature
    measured = truth.copy()
    measured[10:, 0] += 0.8
    trajectory = Trajectory(
        case_id="monitor",
        time=time,
        temperature=measured,
        commands=request.commands,
        sensor_names=("left", "right"),
        control_names=("coolant",),
    )
    result = monitor(
        model,
        trajectory,
        disturbance_process_std=0.01,
        sensor_std=0.05,
        bias_process_std=0.03,
    )
    np.testing.assert_allclose(result.sensor_bias.sum(axis=1), 0.0, atol=1e-12)
    assert result.sensor_bias[-1, 0] > 0.3
    assert result.sensor_bias[-1, 1] < -0.3
    assert result.bias_gauge == "zero_mean"
    assert abs(result.innovation[-1, 0]) < abs(result.innovation[10, 0])
    assert np.isfinite(result.nis[1:]).all()


def test_monitor_reference_sensor_anchors_absolute_bias() -> None:
    model = _duplicate_sensor_model()
    time = np.arange(81.0)
    forecast_temperature = np.full((len(time), 2), np.nan)
    forecast_temperature[0] = 30.0
    request = Trajectory(
        case_id="request",
        time=time,
        temperature=forecast_temperature,
        commands=np.zeros((len(time) - 1, 1)),
        sensor_names=("left", "right"),
        control_names=("coolant",),
    )
    truth = forecast(model, request).sensor_temperature
    measured = truth.copy()
    measured[10:, 1] += 0.8
    trajectory = Trajectory(
        case_id="referenced_monitor",
        time=time,
        temperature=measured,
        commands=request.commands,
        sensor_names=request.sensor_names,
        control_names=request.control_names,
    )

    result = monitor(
        model,
        trajectory,
        bias_reference="left",
        disturbance_process_std=0.01,
        sensor_std=0.05,
        bias_process_std=0.03,
    )

    np.testing.assert_array_equal(result.sensor_bias[:, 0], 0.0)
    assert result.sensor_bias[-1, 1] > 0.6
    assert result.bias_gauge == "reference:left"
    raw_bias = np.array([[0.2, 1.0]])
    np.testing.assert_allclose(
        sensor_bias_in_gauge(raw_bias, ("left", "right"), "left"),
        [[0.0, 0.8]],
    )


def test_artifact_round_trip_preserves_forecast(tmp_path: Path) -> None:
    model = _cooling_model()
    model.boundary_laws.log_offset_multiplier.data.fill_(0.2)
    save_artifact(tmp_path / "artifact", model, metadata={"purpose": "test"})
    loaded = load_artifact(tmp_path / "artifact")
    request = _forecast_request(np.array([0.0, 1.0]), 20.0)
    expected = forecast(model, request).sensor_temperature
    actual = forecast(loaded.model, request).sensor_temperature
    np.testing.assert_allclose(actual, expected)
    assert loaded.metadata["purpose"] == "test"
    fitted_boundary = loaded.metadata["fitted_parameters"]["boundaries"][0]
    assert fitted_boundary["conductance"]["type"] == "constant"


def test_artifact_metadata_cannot_override_runtime_format(tmp_path: Path) -> None:
    target = save_artifact(
        tmp_path / "artifact",
        _cooling_model(),
        metadata={"schema_version": 999, "model_type": "stale", "dtype": "int32"},
    )
    loaded = load_artifact(target)
    assert loaded.metadata["schema_version"] == 4
    assert loaded.metadata["model_type"] == "thermal_network"
    assert loaded.metadata["dtype"] == "float64"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 99, "unsupported artifact schema"),
        ("model_type", "unknown_model", "unsupported model type"),
    ],
)
def test_artifact_rejects_unknown_formats(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    target = save_artifact(tmp_path / "artifact", _cooling_model())
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_artifact(target)


def test_artifact_rejects_unknown_dtype(tmp_path: Path) -> None:
    target = save_artifact(tmp_path / "artifact", _cooling_model())
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dtype"] = "int32"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported artifact dtype"):
        load_artifact(target)


def test_forecast_and_monitor_reject_mismatched_names() -> None:
    model = _cooling_model()
    request = _forecast_request(np.array([0.0, 1.0]), 20.0, control_name="wrong")
    with pytest.raises(ValueError, match="controls"):
        forecast(model, request)

    trajectory = Trajectory(
        case_id="bad",
        time=np.array([0.0, 1.0]),
        temperature=np.array([[20.0], [20.0]]),
        commands=np.zeros((1, 1)),
        sensor_names=("wrong",),
        control_names=("coolant",),
    )
    with pytest.raises(ValueError, match="sensors"):
        monitor(model, trajectory)
