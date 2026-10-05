"""Open-loop thermal forecast from a causally estimated history endpoint."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from celltemp.domain import Trajectory
from celltemp.engine import KalmanObserver, ThermalRCModel

from .initialization import (
    default_state_estimator,
    estimate_observer_state,
    forecast_origin_index,
    model_tensor,
)


@dataclass(frozen=True)
class ForecastResult:
    time: np.ndarray
    sensor_temperature: np.ndarray
    sensor_temperature_std: np.ndarray
    node_temperature: np.ndarray
    node_temperature_std: np.ndarray
    actuator: np.ndarray
    forecast_origin_index: int


def _state_uncertainty_std(
    model: ThermalRCModel,
    covariance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    physical_covariance = covariance[: model.n_nodes, : model.n_nodes]
    node_std = torch.sqrt(torch.clamp(torch.diag(physical_covariance), min=0.0))
    sensor_covariance = model.observation @ physical_covariance @ model.observation.T
    sensor_std = torch.sqrt(torch.clamp(torch.diag(sensor_covariance), min=0.0))
    return node_std, sensor_std


@torch.no_grad()
def forecast(
    model: ThermalRCModel,
    trajectory: Trajectory,
    *,
    initial_actuator: np.ndarray | None = None,
    observer: KalmanObserver | None = None,
) -> ForecastResult:
    """Estimate the history endpoint, then roll out the unobserved future open-loop."""
    origin = forecast_origin_index(trajectory.mask)
    estimator = default_state_estimator(model) if observer is None else observer
    posterior = estimate_observer_state(
        model,
        trajectory,
        through_index=origin,
        initial_actuator=initial_actuator,
        observer=estimator,
    )
    commands = model_tensor(model, trajectory.commands[origin:])
    dt = model_tensor(model, trajectory.dt[origin:])
    # Past disturbance and bias estimates explain measurements but are not future inputs.
    # Their covariance remains part of the open-loop uncertainty propagation.
    temperatures, actuators = model.forward_trajectory(
        posterior.temperature, commands, dt, posterior.actuator
    )
    covariance = posterior.covariance
    node_stds: list[torch.Tensor] = []
    sensor_stds: list[torch.Tensor] = []

    node_std, sensor_std = _state_uncertainty_std(model, covariance)
    node_stds.append(node_std)
    sensor_stds.append(sensor_std)
    for index in range(len(commands)):
        covariance = estimator.predict_covariance(
            covariance, actuators[index], commands[index], dt[index]
        )
        node_std, sensor_std = _state_uncertainty_std(model, covariance)
        node_stds.append(node_std)
        sensor_stds.append(sensor_std)

    return ForecastResult(
        time=trajectory.time[origin:].copy(),
        sensor_temperature=model.observe(temperatures).cpu().numpy(),
        sensor_temperature_std=torch.stack(sensor_stds).cpu().numpy(),
        node_temperature=temperatures.cpu().numpy(),
        node_temperature_std=torch.stack(node_stds).cpu().numpy(),
        actuator=actuators.cpu().numpy(),
        forecast_origin_index=origin,
    )
