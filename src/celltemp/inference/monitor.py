"""Causal thermal-state, disturbance, and sensor-bias monitoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from celltemp.domain import Trajectory
from celltemp.engine import KalmanObserver, ThermalRCModel

from .initialization import model_tensor, resolved_initial_actuator
from .settings import (
    DEFAULT_BIAS_PROCESS_STD,
    DEFAULT_BIAS_REFERENCE,
    DEFAULT_DISTURBANCE_PROCESS_STD,
    DEFAULT_INNOVATION_GATE_SIGMA,
    DEFAULT_SENSOR_STD,
    MONITOR_OBSERVER_DEFAULTS,
    build_observer,
    resolve_observer_settings,
)


@dataclass(frozen=True)
class MonitorResult:
    time: np.ndarray
    prior_physical_temperature: np.ndarray
    predicted_measurement: np.ndarray
    posterior_physical_temperature: np.ndarray
    reconstructed_measurement: np.ndarray
    posterior_node_temperature: np.ndarray
    actuator: np.ndarray
    node_heat_disturbance: np.ndarray
    sensor_bias: np.ndarray
    bias_gauge: str
    innovation: np.ndarray
    innovation_std: np.ndarray
    innovation_covariance: np.ndarray
    nis: np.ndarray
    nis_dof: np.ndarray


def sensor_bias_in_gauge(
    sensor_bias: np.ndarray,
    sensor_names: tuple[str, ...],
    bias_reference: str | None = None,
) -> np.ndarray:
    """Project sensor offsets onto the identifiable monitor bias gauge."""
    values = np.asarray(sensor_bias, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != len(sensor_names):
        raise ValueError("sensor_bias must end with one value per sensor")
    if bias_reference is None:
        return values - values.mean(axis=-1, keepdims=True)
    if bias_reference not in sensor_names:
        raise ValueError(f"unknown bias reference sensor: {bias_reference}")
    reference = sensor_names.index(bias_reference)
    return values - values[..., reference : reference + 1]


def _monitor_observer(
    model: ThermalRCModel,
    observer: KalmanObserver | None,
    overrides: dict[str, object],
) -> KalmanObserver:
    if observer is None:
        return build_observer(model, resolve_observer_settings("monitor", overrides))
    conflicts = sorted(
        name for name, value in overrides.items() if value != MONITOR_OBSERVER_DEFAULTS[name]
    )
    if conflicts:
        raise ValueError(f"observer conflicts with explicit monitor settings: {conflicts}")
    return observer


def _stack(values: list[torch.Tensor]) -> np.ndarray:
    return torch.stack(values).cpu().numpy()


@torch.no_grad()
def monitor(
    model: ThermalRCModel,
    trajectory: Trajectory,
    *,
    initial_actuator: np.ndarray | None = None,
    observer: KalmanObserver | None = None,
    disturbance_process_std: float = DEFAULT_DISTURBANCE_PROCESS_STD,
    bias_process_std: float = DEFAULT_BIAS_PROCESS_STD,
    sensor_std: float = DEFAULT_SENSOR_STD,
    innovation_gate_sigma: float = DEFAULT_INNOVATION_GATE_SIGMA,
    bias_reference: str | None = DEFAULT_BIAS_REFERENCE,
) -> MonitorResult:
    """Estimate physical temperature, unknown heat, and sensor bias causally."""
    trajectory.require_layout(model.spec.sensor_names, model.spec.control_names)
    observed = model_tensor(model, trajectory.temperature)
    commands = model_tensor(model, trajectory.commands)
    dt = model_tensor(model, trajectory.dt)
    mask = torch.tensor(trajectory.mask.copy(), dtype=torch.bool, device=model.capacity.device)
    resolved_observer = _monitor_observer(
        model,
        observer,
        {
            "disturbance_process_std": disturbance_process_std,
            "bias_process_std": bias_process_std,
            "sensor_std": sensor_std,
            "innovation_gate_sigma": innovation_gate_sigma,
            "bias_reference": bias_reference,
        },
    )
    if resolved_observer.bias_reference is not None:
        reference_index = trajectory.sensor_names.index(resolved_observer.bias_reference)
        if not trajectory.mask[:, reference_index].any():
            raise ValueError("bias reference sensor has no observations")
    actuator = resolved_initial_actuator(model, trajectory, commands, initial_actuator)
    state = resolved_observer.initialize_posterior(
        observed[0],
        mask=mask[0],
        actuator=actuator,
    )

    nan_sensor = torch.full_like(observed[0], torch.nan)
    nan_covariance = torch.full(
        (model.n_sensors, model.n_sensors),
        torch.nan,
        dtype=observed.dtype,
        device=observed.device,
    )
    prior_physical = [nan_sensor]
    predicted_measurements = [nan_sensor]
    posterior_physical = [model.observe(state.temperature)]
    reconstructed_measurements = [resolved_observer.predicted_measurement(state)]
    nodes = [state.temperature]
    actuators = [state.actuator]
    disturbances = [resolved_observer.node_heat_disturbance(state)]
    biases = [resolved_observer.sensor_bias(state)]
    innovations = [nan_sensor]
    innovation_stds = [nan_sensor]
    innovation_covariances = [nan_covariance]
    nis_values = [torch.as_tensor(torch.nan, dtype=observed.dtype, device=observed.device)]
    nis_dofs = [torch.as_tensor(0, dtype=torch.int64, device=observed.device)]

    for index in range(len(commands)):
        predicted = resolved_observer.predict(state, commands[index], dt[index])
        available = mask[index + 1]
        predicted_measurement = resolved_observer.predicted_measurement(predicted)
        innovation = torch.where(
            available,
            observed[index + 1] - predicted_measurement,
            torch.nan,
        )
        innovation_covariance = resolved_observer.innovation_covariance(predicted, available)
        full_covariance = torch.full_like(nan_covariance, torch.nan)
        available_indices = torch.nonzero(available, as_tuple=False).flatten()
        full_covariance[available_indices[:, None], available_indices[None, :]] = (
            innovation_covariance
        )
        innovation_std = torch.full_like(nan_sensor, torch.nan)
        innovation_std[available] = torch.sqrt(
            torch.clamp(torch.diag(innovation_covariance), min=0.0)
        )
        if torch.any(available):
            available_innovation = innovation[available]
            nis = available_innovation @ torch.linalg.solve(
                innovation_covariance,
                available_innovation,
            )
        else:
            nis = torch.as_tensor(torch.nan, dtype=observed.dtype, device=observed.device)
        state, _ = resolved_observer.update(predicted, observed[index + 1], available)
        prior_physical.append(model.observe(predicted.temperature))
        predicted_measurements.append(predicted_measurement)
        posterior_physical.append(model.observe(state.temperature))
        reconstructed_measurements.append(resolved_observer.predicted_measurement(state))
        nodes.append(state.temperature)
        actuators.append(state.actuator)
        disturbances.append(resolved_observer.node_heat_disturbance(state))
        biases.append(resolved_observer.sensor_bias(state))
        innovations.append(innovation)
        innovation_stds.append(innovation_std)
        innovation_covariances.append(full_covariance)
        nis_values.append(nis)
        nis_dofs.append(available.sum())

    return MonitorResult(
        time=trajectory.time.copy(),
        prior_physical_temperature=_stack(prior_physical),
        predicted_measurement=_stack(predicted_measurements),
        posterior_physical_temperature=_stack(posterior_physical),
        reconstructed_measurement=_stack(reconstructed_measurements),
        posterior_node_temperature=_stack(nodes),
        actuator=_stack(actuators),
        node_heat_disturbance=_stack(disturbances),
        sensor_bias=_stack(biases),
        bias_gauge=resolved_observer.bias_gauge,
        innovation=_stack(innovations),
        innovation_std=_stack(innovation_stds),
        innovation_covariance=_stack(innovation_covariances),
        nis=_stack(nis_values),
        nis_dof=_stack(nis_dofs),
    )
