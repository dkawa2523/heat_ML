import json
from pathlib import Path

import numpy as np
import pytest

from celltemp.artifact import load_artifact, save_artifact
from celltemp.domain import (
    ActuatorSpec,
    BoundarySpec,
    ThermalSystemSpec,
    Trajectory,
)
from celltemp.engine import ThermalRCModel
from celltemp.inference import forecast, monitor


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
                conductance=0.5,
                temperature_intercept=10.0,
            ),
        ),
    )
    return ThermalRCModel(spec)


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


def test_monitor_estimates_persistent_sensor_bias() -> None:
    model = _cooling_model()
    request = _forecast_request(np.arange(31.0), 30.0)
    truth = forecast(model, request).sensor_temperature
    measured = truth + 0.8
    trajectory = Trajectory(
        case_id="monitor",
        time=request.time,
        temperature=measured,
        commands=request.commands,
        sensor_names=("core",),
        control_names=("coolant",),
    )
    result = monitor(model, trajectory, sensor_std=0.05, bias_process_std=0.01)
    assert result.sensor_bias[-1, 0] > 0.5
    assert abs(result.residual[-1, 0]) < abs(result.residual[1, 0])


def test_artifact_round_trip_preserves_forecast(tmp_path: Path) -> None:
    model = _cooling_model()
    model.log_boundary_multiplier.data.fill_(0.2)
    save_artifact(tmp_path / "artifact", model, metadata={"purpose": "test"})
    loaded = load_artifact(tmp_path / "artifact")
    request = _forecast_request(np.array([0.0, 1.0]), 20.0)
    expected = forecast(model, request).sensor_temperature
    actual = forecast(loaded.model, request).sensor_temperature
    np.testing.assert_allclose(actual, expected)
    assert loaded.metadata["purpose"] == "test"


def test_artifact_metadata_cannot_override_runtime_format(tmp_path: Path) -> None:
    target = save_artifact(
        tmp_path / "artifact",
        _cooling_model(),
        metadata={"schema_version": 999, "model_type": "stale", "dtype": "int32"},
    )
    loaded = load_artifact(target)
    assert loaded.metadata["schema_version"] == 2
    assert loaded.metadata["model_type"] == "thermal_rc"
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
