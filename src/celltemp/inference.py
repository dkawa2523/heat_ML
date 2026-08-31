"""Open-loop forecasting and causal monitoring on the same thermal state engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from celltemp.domain import Trajectory
from celltemp.engine import KalmanObserver, ThermalRCModel, ThermalState


@dataclass(frozen=True)
class ForecastResult:
    time: np.ndarray
    sensor_temperature: np.ndarray
    node_temperature: np.ndarray
    actuator: np.ndarray
    forecast_origin_index: int


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


def _model_tensor(model: ThermalRCModel, values: np.ndarray) -> torch.Tensor:
    return torch.tensor(
        np.asarray(values).copy(), dtype=model.capacity.dtype, device=model.capacity.device
    )


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


def forecast_origin_index(observation_mask: np.ndarray) -> int:
    """Return the last row of a contiguous observation prefix.

    A wholly unobserved row starts the future suffix. Observations may be sparse
    by sensor inside the history, but cannot reappear after that boundary.
    """
    mask = np.asarray(observation_mask, dtype=np.bool_)
    if mask.ndim != 2 or len(mask) < 2:
        raise ValueError("forecast observation mask must have shape [time, sensor]")
    observed_rows: np.ndarray = np.asarray(mask.any(axis=1), dtype=np.bool_)
    future_rows = np.flatnonzero(~observed_rows)
    if not len(future_rows):
        raise ValueError("forecast input needs at least one unobserved future row")
    first_future = int(future_rows[0])
    if first_future == 0:
        raise ValueError("forecast input needs an observation on the initial row")
    if observed_rows[first_future:].any():
        raise ValueError(
            "forecast observations must be a contiguous history prefix followed by "
            "unobserved future rows"
        )
    return first_future - 1


def _default_state_estimator(model: ThermalRCModel) -> KalmanObserver:
    """Build a weak-prior observer for physical-state initialization.

    History initialization should infer hidden temperatures from their transient
    effect, rather than lock them to the first sensor average. Relative bias is a
    monitoring quantity, so it is disabled here; small heat-process freedom keeps
    the state estimate usable with mildly imperfect physical models.
    """
    return KalmanObserver(
        model,
        disturbance_process_std=0.02,
        bias_process_std=0.0,
        initial_temperature_std=100.0,
        initial_disturbance_std=0.0,
        initial_bias_std=0.0,
    )


@torch.no_grad()
def estimate_state(
    model: ThermalRCModel,
    trajectory: Trajectory,
    *,
    through_index: int | None = None,
    initial_actuator: np.ndarray | None = None,
    observer: KalmanObserver | None = None,
) -> ThermalState:
    """Causally estimate the physical state through one trajectory row.

    Missing sensors are handled by the observation mask. The returned state
    deliberately contains only node temperature and effective actuator; transient
    disturbance and sensor-bias estimates are not assumptions about the future.
    """
    trajectory.require_layout(model.spec.sensor_names, model.spec.control_names)
    final_index = len(trajectory.time) - 1 if through_index is None else through_index
    if not 0 <= final_index < len(trajectory.time):
        raise ValueError("through_index is outside the trajectory")

    observed = _model_tensor(model, trajectory.temperature)
    commands = _model_tensor(model, trajectory.commands)
    dt = _model_tensor(model, trajectory.dt)
    mask = torch.tensor(trajectory.mask.copy(), dtype=torch.bool, device=model.capacity.device)
    estimator = _default_state_estimator(model) if observer is None else observer
    actuator = commands[0] if initial_actuator is None else _model_tensor(model, initial_actuator)
    state = estimator.initialize(observed[0], mask=mask[0], actuator=actuator)
    for index in range(final_index):
        state, _ = estimator.step(
            state,
            commands[index],
            observed[index + 1],
            dt[index],
            mask[index + 1],
        )
    return ThermalState(state.temperature, state.actuator)


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
    initial_state = estimate_state(
        model,
        trajectory,
        through_index=origin,
        initial_actuator=initial_actuator,
        observer=observer,
    )
    commands = _model_tensor(model, trajectory.commands[origin:])
    dt = _model_tensor(model, trajectory.dt[origin:])
    states, actuators = model.forward_trajectory(
        initial_state.temperature,
        commands,
        dt,
        initial_state.actuator,
    )
    return ForecastResult(
        time=trajectory.time[origin:].copy(),
        sensor_temperature=model.observe(states).cpu().numpy(),
        node_temperature=states.cpu().numpy(),
        actuator=actuators.cpu().numpy(),
        forecast_origin_index=origin,
    )


@torch.no_grad()
def monitor(
    model: ThermalRCModel,
    trajectory: Trajectory,
    *,
    disturbance_process_std: float = 0.02,
    bias_process_std: float = 0.005,
    sensor_std: float = 0.15,
    innovation_gate_sigma: float = 4.0,
    bias_reference: str | None = None,
) -> MonitorResult:
    """Estimate physical temperature, unknown node heat, and sensor bias causally.

    With no ``bias_reference``, sensor bias is constrained to zero mean. Supplying
    a calibrated sensor name fixes that sensor's bias to zero and anchors all
    remaining sensor offsets to it.
    """
    trajectory.require_layout(model.spec.sensor_names, model.spec.control_names)
    observed = _model_tensor(model, trajectory.temperature)
    commands = _model_tensor(model, trajectory.commands)
    dt = _model_tensor(model, trajectory.dt)
    mask = torch.tensor(trajectory.mask.copy(), dtype=torch.bool, device=model.capacity.device)
    observer = KalmanObserver(
        model,
        disturbance_process_std=disturbance_process_std,
        bias_process_std=bias_process_std,
        sensor_std=sensor_std,
        innovation_gate_sigma=innovation_gate_sigma,
        bias_reference=bias_reference,
    )
    if bias_reference is not None:
        reference_index = trajectory.sensor_names.index(bias_reference)
        if not trajectory.mask[:, reference_index].any():
            raise ValueError("bias reference sensor has no observations")
    state = observer.initialize(observed[0], mask=mask[0], actuator=commands[0])

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
    reconstructed_measurements = [observer.predicted_measurement(state)]
    nodes = [state.temperature]
    actuators = [state.actuator]
    disturbances = [observer.node_heat_disturbance(state)]
    biases = [observer.sensor_bias(state)]
    innovations = [nan_sensor]
    innovation_stds = [nan_sensor]
    innovation_covariances = [nan_covariance]
    nis_values = [torch.as_tensor(torch.nan, dtype=observed.dtype, device=observed.device)]
    nis_dofs = [torch.as_tensor(0, dtype=torch.int64, device=observed.device)]

    for index in range(len(commands)):
        predicted = observer.predict(state, commands[index], dt[index])
        available = mask[index + 1]
        predicted_measurement = observer.predicted_measurement(predicted)
        innovation = torch.where(
            available,
            observed[index + 1] - predicted_measurement,
            torch.nan,
        )
        innovation_covariance = observer.innovation_covariance(predicted, available)
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
                innovation_covariance, available_innovation
            )
        else:
            nis = torch.as_tensor(torch.nan, dtype=observed.dtype, device=observed.device)
        state, _ = observer.update(predicted, observed[index + 1], available)
        prior_physical.append(model.observe(predicted.temperature))
        predicted_measurements.append(predicted_measurement)
        posterior_physical.append(model.observe(state.temperature))
        reconstructed_measurements.append(observer.predicted_measurement(state))
        nodes.append(state.temperature)
        actuators.append(state.actuator)
        disturbances.append(observer.node_heat_disturbance(state))
        biases.append(observer.sensor_bias(state))
        innovations.append(innovation)
        innovation_stds.append(innovation_std)
        innovation_covariances.append(full_covariance)
        nis_values.append(nis)
        nis_dofs.append(available.sum())

    def array(values: list[torch.Tensor]) -> np.ndarray:
        return torch.stack(values).cpu().numpy()

    return MonitorResult(
        time=trajectory.time.copy(),
        prior_physical_temperature=array(prior_physical),
        predicted_measurement=array(predicted_measurements),
        posterior_physical_temperature=array(posterior_physical),
        reconstructed_measurement=array(reconstructed_measurements),
        posterior_node_temperature=array(nodes),
        actuator=array(actuators),
        node_heat_disturbance=array(disturbances),
        sensor_bias=array(biases),
        bias_gauge=observer.bias_gauge,
        innovation=array(innovations),
        innovation_std=array(innovation_stds),
        innovation_covariance=array(innovation_covariances),
        nis=array(nis_values),
        nis_dof=array(nis_dofs),
    )
