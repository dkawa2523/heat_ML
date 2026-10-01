from __future__ import annotations

import numpy as np
import pytest

from celltemp.domain import (
    Trajectory,
)
from celltemp.inference import (
    build_observer,
    forecast,
    monitor,
    resolve_observer_settings,
    sensor_bias_in_gauge,
)
from tests.unit._inference_cases import cooling_model, duplicate_sensor_model, forecast_request


def test_monitor_uses_zero_mean_bias_gauge_without_a_reference() -> None:
    model = duplicate_sensor_model()
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
    observer = build_observer(
        model,
        resolve_observer_settings(
            "monitor",
            {
                "disturbance_process_std": 0.01,
                "sensor_std": 0.05,
                "bias_process_std": 0.03,
            },
        ),
    )
    result = monitor(model, trajectory, observer=observer)
    np.testing.assert_allclose(result.sensor_bias.sum(axis=1), 0.0, atol=1e-12)
    assert result.sensor_bias[-1, 0] > 0.3
    assert result.sensor_bias[-1, 1] < -0.3
    assert result.bias_gauge == "zero_mean"
    assert abs(result.innovation[-1, 0]) < abs(result.innovation[10, 0])
    assert np.isfinite(result.nis[1:]).all()


def test_monitor_reference_sensor_anchors_absolute_bias() -> None:
    model = duplicate_sensor_model()
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

    observer = build_observer(
        model,
        resolve_observer_settings(
            "monitor",
            {
                "bias_reference": "left",
                "disturbance_process_std": 0.01,
                "sensor_std": 0.05,
                "bias_process_std": 0.03,
            },
        ),
    )
    result = monitor(model, trajectory, observer=observer)

    np.testing.assert_array_equal(result.sensor_bias[:, 0], 0.0)
    assert result.sensor_bias[-1, 1] > 0.6
    assert result.bias_gauge == "reference:left"
    raw_bias = np.array([[0.2, 1.0]])
    np.testing.assert_allclose(
        sensor_bias_in_gauge(raw_bias, ("left", "right"), "left"),
        [[0.0, 0.8]],
    )


def test_forecast_and_monitor_reject_mismatched_names() -> None:
    model = cooling_model()
    request = forecast_request(np.array([0.0, 1.0]), 20.0, control_name="wrong")
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
