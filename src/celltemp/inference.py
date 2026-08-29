"""Open-loop forecasting and causal monitoring on the same thermal state engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from celltemp.domain import Trajectory
from celltemp.engine import KalmanObserver, ThermalRCModel


@dataclass(frozen=True)
class ForecastResult:
    time: np.ndarray
    sensor_temperature: np.ndarray
    node_temperature: np.ndarray
    actuator: np.ndarray


@dataclass(frozen=True)
class MonitorResult:
    time: np.ndarray
    prior_sensor_temperature: np.ndarray
    filtered_sensor_temperature: np.ndarray
    filtered_node_temperature: np.ndarray
    actuator: np.ndarray
    sensor_bias: np.ndarray
    residual: np.ndarray
    residual_std: np.ndarray


def _model_tensor(model: ThermalRCModel, values: np.ndarray) -> torch.Tensor:
    return torch.tensor(
        np.asarray(values).copy(), dtype=model.capacity.dtype, device=model.capacity.device
    )


@torch.no_grad()
def forecast(
    model: ThermalRCModel,
    trajectory: Trajectory,
    *,
    initial_actuator: np.ndarray | None = None,
) -> ForecastResult:
    """Roll out from the initial observation and the trajectory's commands."""
    if trajectory.sensor_names != model.spec.sensor_names:
        raise ValueError("trajectory sensors do not match the system definition")
    if trajectory.control_names != model.spec.control_names:
        raise ValueError("trajectory controls do not match the system definition")
    observation = _model_tensor(model, trajectory.temperature[0])
    mask = torch.tensor(trajectory.mask[0], dtype=torch.bool, device=observation.device)
    initial_temperature = model.initialize_temperature(observation, mask)
    commands = _model_tensor(model, trajectory.commands)
    dt = _model_tensor(model, trajectory.dt)
    actuator = None if initial_actuator is None else _model_tensor(model, initial_actuator)
    states, actuators = model.forward_trajectory(initial_temperature, commands, dt, actuator)
    return ForecastResult(
        time=trajectory.time.copy(),
        sensor_temperature=model.observe(states).cpu().numpy(),
        node_temperature=states.cpu().numpy(),
        actuator=actuators.cpu().numpy(),
    )


@torch.no_grad()
def monitor(
    model: ThermalRCModel,
    trajectory: Trajectory,
    *,
    temperature_process_std: float = 0.02,
    bias_process_std: float = 0.005,
    sensor_std: float = 0.15,
) -> MonitorResult:
    """Estimate thermal state and sensor bias from a causal measurement stream."""
    if trajectory.sensor_names != model.spec.sensor_names:
        raise ValueError("trajectory sensors do not match the system definition")
    if trajectory.control_names != model.spec.control_names:
        raise ValueError("trajectory controls do not match the system definition")
    observed = _model_tensor(model, trajectory.temperature)
    commands = _model_tensor(model, trajectory.commands)
    dt = _model_tensor(model, trajectory.dt)
    mask = torch.tensor(trajectory.mask.copy(), dtype=torch.bool, device=model.capacity.device)
    observer = KalmanObserver(
        model,
        temperature_process_std=temperature_process_std,
        bias_process_std=bias_process_std,
        sensor_std=sensor_std,
    )
    state = observer.initialize(observed[0], mask=mask[0], actuator=commands[0])

    first_sensor = model.observe(state.temperature) + state.sensor_bias
    prior = [first_sensor]
    filtered = [model.observe(state.temperature)]
    nodes = [state.temperature]
    actuators = [state.actuator]
    biases = [state.sensor_bias]
    residuals = [torch.full_like(first_sensor, torch.nan)]
    residual_stds = [observer.prediction_std(state)]

    for index in range(len(commands)):
        predicted = observer.predict(state, commands[index], dt[index])
        predicted_measurement = model.observe(predicted.temperature) + predicted.sensor_bias
        residual = observed[index + 1] - predicted_measurement
        residual = torch.where(mask[index + 1], residual, torch.nan)
        state, _ = observer.update(predicted, observed[index + 1], mask[index + 1])
        prior.append(predicted_measurement)
        filtered.append(model.observe(state.temperature))
        nodes.append(state.temperature)
        actuators.append(state.actuator)
        biases.append(state.sensor_bias)
        residuals.append(residual)
        residual_stds.append(observer.prediction_std(predicted))

    def array(values: list[torch.Tensor]) -> np.ndarray:
        return torch.stack(values).cpu().numpy()

    return MonitorResult(
        time=trajectory.time.copy(),
        prior_sensor_temperature=array(prior),
        filtered_sensor_temperature=array(filtered),
        filtered_node_temperature=array(nodes),
        actuator=array(actuators),
        sensor_bias=array(biases),
        residual=array(residuals),
        residual_std=array(residual_stds),
    )
