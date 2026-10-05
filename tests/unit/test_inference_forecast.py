from __future__ import annotations

import numpy as np
import pytest
import torch

from celltemp.domain import (
    ActuatorSpec,
    BoundarySpec,
    ConstantLawSpec,
    EdgeSpec,
    PositivePartLawSpec,
    ReservoirTemperatureSpec,
    SourceSpec,
    ThermalSystemSpec,
    Trajectory,
)
from celltemp.engine import KalmanObserver, ObserverState, ThermalRCModel
from celltemp.inference import (
    estimate_state,
    forecast,
)
from celltemp.inference.initialization import estimate_observer_state
from tests.unit._inference_cases import cooling_model, forecast_request


def test_forecast_uses_variable_time_intervals() -> None:
    model = cooling_model()
    request = forecast_request(np.array([0.0, 1.0, 3.0]), 30.0)
    result = forecast(model, request)
    expected = 10.0 + 20.0 * np.exp(-0.5 / 2.0 * request.time)
    np.testing.assert_allclose(result.sensor_temperature[:, 0], expected, atol=1e-10)
    assert result.sensor_temperature_std.shape == result.sensor_temperature.shape
    assert result.node_temperature_std.shape == result.node_temperature.shape
    assert np.isfinite(result.sensor_temperature_std).all()
    assert (result.sensor_temperature_std >= 0.0).all()


def test_forecast_reuses_mean_and_covariance_operators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.matrix_exp
    exponentials = 0

    def counted(value: torch.Tensor) -> torch.Tensor:
        nonlocal exponentials
        exponentials += 1
        return original(value)

    monkeypatch.setattr(torch, "matrix_exp", counted)
    result = forecast(cooling_model(), forecast_request(np.arange(21.0), 30.0))
    assert np.isfinite(result.sensor_temperature).all()
    assert exponentials == 2


@pytest.mark.parametrize("integrator", ["exact", "implicit"])
def test_forecast_shared_mean_preserves_augmented_uncertainty(integrator: str) -> None:
    model = ThermalRCModel(
        ThermalSystemSpec(
            node_names=("core", "surface"),
            heat_capacity=(3.0, 1.0),
            edges=(EdgeSpec("core", "surface", ConstantLawSpec(0.2, learnable=False)),),
            actuators=(
                ActuatorSpec("heater", tau=2.0, learnable=False),
                ActuatorSpec("coolant", tau=0.0),
            ),
            sources=(
                SourceSpec(
                    "heater",
                    (1.0, 0.0),
                    PositivePartLawSpec("heater", 0.3, threshold=2.0, learnable=False),
                ),
            ),
            boundaries=(
                BoundarySpec(
                    "coolant",
                    (0.0, 1.0),
                    ReservoirTemperatureSpec(15.0, "coolant", 1.0),
                    ConstantLawSpec(0.1, learnable=False),
                ),
            ),
            sensor_names=("weighted", "core"),
            sensor_weights=((0.25, 0.75), (1.0, 0.0)),
        ),
        integrator=integrator,
    )
    request = Trajectory(
        case_id="mixed-history",
        time=np.array([0.0, 0.5, 1.7, 4.7, 5.1, 6.5]),
        temperature=np.array([[25.0, 30.0], [np.nan, 31.0]] + [[np.nan, np.nan]] * 4),
        commands=np.array([[4.0, 5.0], [-2.0, 10.0], [8.0, 3.0], [0.0, 8.0], [4.0, 4.0]]),
        sensor_names=model.spec.sensor_names,
        control_names=model.spec.control_names,
        initial_actuator=np.array([0.0, 5.0]),
    )
    observer = KalmanObserver(
        model,
        disturbance_process_std=0.07,
        bias_process_std=0.003,
        initial_temperature_std=5.0,
        initial_disturbance_std=0.4,
        initial_bias_std=0.2,
        bias_reference="core",
    )
    with torch.no_grad():
        posterior = estimate_observer_state(model, request, through_index=1, observer=observer)
        assert posterior.heat_disturbance.abs().sum() > 0.0
        commands = torch.tensor(request.commands[1:].copy(), dtype=torch.float64)
        state = ObserverState(
            posterior.temperature,
            model.point_actuator(posterior.actuator, commands[0]),
            torch.zeros_like(posterior.heat_disturbance),
            torch.zeros_like(posterior.bias_state),
            posterior.covariance,
        )
        states = [state.temperature]
        inputs = [state.actuator]
        covariances = [state.covariance[:2, :2]]
        for index, interval in enumerate(request.dt[1:]):
            state = observer.predict(state, commands[index], float(interval))
            states.append(state.temperature)
            inputs.append(model.point_actuator(state.actuator, commands[min(index + 1, 3)]))
            covariances.append(state.covariance[:2, :2])
        expected_temperature = torch.stack(states)
        expected_covariance = torch.stack(covariances)
        expected_sensor_covariance = model.observation @ expected_covariance @ model.observation.T
        expected_node_std = torch.diagonal(expected_covariance, dim1=-2, dim2=-1).sqrt()
        expected_sensor_std = torch.diagonal(expected_sensor_covariance, dim1=-2, dim2=-1).sqrt()

    result = forecast(model, request, observer=observer)
    np.testing.assert_allclose(result.node_temperature, expected_temperature.numpy(), atol=1e-10)
    np.testing.assert_allclose(result.actuator, torch.stack(inputs).numpy(), atol=1e-10)
    np.testing.assert_allclose(result.node_temperature_std, expected_node_std.numpy(), atol=1e-12)
    np.testing.assert_allclose(
        result.sensor_temperature_std, expected_sensor_std.numpy(), atol=1e-12
    )


def test_forecast_uses_optional_effective_initial_actuator() -> None:
    spec = ThermalSystemSpec(
        node_names=("body",),
        heat_capacity=(1.0,),
        edges=(),
        actuators=(ActuatorSpec("heater", tau=2.0, learnable=False),),
        sources=(
            SourceSpec(
                "heater",
                (1.0,),
                PositivePartLawSpec("heater", 1.0, learnable=False),
            ),
        ),
    )
    model = ThermalRCModel(spec)
    temperature = np.array([[20.0], [np.nan], [np.nan]])
    warm = forecast(
        model,
        Trajectory(
            case_id="warm",
            time=np.array([0.0, 1.0, 2.0]),
            temperature=temperature,
            commands=np.zeros((2, 1)),
            sensor_names=("body",),
            control_names=("heater",),
            initial_actuator=np.array([10.0]),
        ),
    )
    settled = forecast(
        model,
        Trajectory(
            case_id="settled",
            time=np.array([0.0, 1.0, 2.0]),
            temperature=temperature,
            commands=np.zeros((2, 1)),
            sensor_names=("body",),
            control_names=("heater",),
        ),
    )

    assert warm.sensor_temperature[-1, 0] > 30.0
    np.testing.assert_allclose(settled.sensor_temperature[:, 0], 20.0)


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
    model = cooling_model()
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


@pytest.mark.parametrize("integrator", ["exact", "implicit"])
@pytest.mark.parametrize("history_rows", [1, 2])
def test_forecast_point_inputs_match_schedule_at_origin_and_command_changes(
    integrator: str,
    history_rows: int,
) -> None:
    model = ThermalRCModel(
        ThermalSystemSpec(
            node_names=("body",),
            heat_capacity=(1.0,),
            edges=(),
            actuators=(ActuatorSpec("heater"),),
            sources=(SourceSpec("heat", (1.0,), PositivePartLawSpec("heater", 1.0)),),
        ),
        integrator=integrator,
    )
    measured = np.array([[20.0], [20.0], [40.0], [40.0]])
    measured[history_rows:] = np.nan
    request = Trajectory(
        case_id="switched",
        time=np.array([0.0, 1.0, 3.0, 3.5]),
        temperature=measured,
        commands=np.array([[0.0], [10.0], [0.0]]),
        sensor_names=("body",),
        control_names=("heater",),
        initial_actuator=np.array([99.0]),
    )
    result = forecast(model, request)
    origin = history_rows - 1
    np.testing.assert_allclose(result.actuator[:, 0], np.array([0.0, 10.0, 0.0, 0.0])[origin:])
    np.testing.assert_allclose(
        result.sensor_temperature[:, 0], np.array([20.0, 20.0, 40.0, 40.0])[origin:]
    )
