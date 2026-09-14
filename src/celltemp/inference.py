"""Open-loop forecasting and causal monitoring on the same thermal state engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from celltemp.domain import Trajectory
from celltemp.engine import KalmanObserver, ObserverState, ThermalRCModel, ThermalState

_DEFAULT_DISTURBANCE_PROCESS_STD = 0.02
_DEFAULT_BIAS_PROCESS_STD = 0.005
_DEFAULT_SENSOR_STD = 0.15
_DEFAULT_INNOVATION_GATE_SIGMA = 4.0
_DEFAULT_BIAS_REFERENCE: str | None = None

_MONITOR_OBSERVER_DEFAULTS: dict[str, float | str | None] = {
    "disturbance_process_std": _DEFAULT_DISTURBANCE_PROCESS_STD,
    "bias_process_std": _DEFAULT_BIAS_PROCESS_STD,
    "sensor_std": _DEFAULT_SENSOR_STD,
    "innovation_gate_sigma": _DEFAULT_INNOVATION_GATE_SIGMA,
    "initial_temperature_std": 1.0,
    "initial_disturbance_std": 0.5,
    "initial_bias_std": 0.5,
    "bias_reference": _DEFAULT_BIAS_REFERENCE,
}
_FORECAST_OBSERVER_DEFAULTS: dict[str, float | str | None] = {
    **_MONITOR_OBSERVER_DEFAULTS,
    "bias_process_std": 0.0,
    "initial_temperature_std": 100.0,
    "initial_disturbance_std": 0.0,
    "initial_bias_std": 0.0,
}
_FORECAST_OBSERVER_OPTIONS = {
    "disturbance_process_std",
    "initial_temperature_std",
    "innovation_gate_sigma",
    "sensor_std",
}


def resolve_observer_settings(
    mode: str,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, float | str | None]:
    """Resolve one explicit observer profile for a deployment workflow."""
    if mode == "forecast":
        defaults = _FORECAST_OBSERVER_DEFAULTS
        allowed = _FORECAST_OBSERVER_OPTIONS
    elif mode == "monitor":
        defaults = _MONITOR_OBSERVER_DEFAULTS
        allowed = set(defaults)
    else:
        raise ValueError(f"unknown observer mode: {mode}")
    supplied = dict(overrides or {})
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError(f"unknown {mode}.observer options: {unknown}")
    resolved = dict(defaults)
    for name, value in supplied.items():
        if name == "bias_reference":
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{mode}.observer.bias_reference must be a sensor name or null")
            resolved[name] = value
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{mode}.observer.{name} must be numeric")
            resolved[name] = float(value)
    return resolved


def build_observer(
    model: ThermalRCModel,
    settings: Mapping[str, float | str | None],
) -> KalmanObserver:
    """Construct an observer from resolved, serializable settings."""
    return KalmanObserver(model, **dict(settings))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ForecastResult:
    time: np.ndarray
    sensor_temperature: np.ndarray
    sensor_temperature_std: np.ndarray
    node_temperature: np.ndarray
    node_temperature_std: np.ndarray
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


def _resolved_initial_actuator(
    model: ThermalRCModel,
    trajectory: Trajectory,
    commands: torch.Tensor,
    override: np.ndarray | None,
) -> torch.Tensor:
    values = override if override is not None else trajectory.initial_actuator
    if values is None:
        return commands[0]
    actuator = _model_tensor(model, values)
    if actuator.shape != (model.n_controls,) or not bool(torch.isfinite(actuator).all()):
        raise ValueError("initial_actuator must contain one finite value per control")
    return actuator


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
    return build_observer(model, resolve_observer_settings("forecast"))


@torch.no_grad()
def _estimate_observer_state(
    model: ThermalRCModel,
    trajectory: Trajectory,
    *,
    through_index: int | None = None,
    initial_actuator: np.ndarray | None = None,
    observer: KalmanObserver | None = None,
) -> ObserverState:
    """Causally estimate the complete observer state through one trajectory row."""
    trajectory.require_layout(model.spec.sensor_names, model.spec.control_names)
    final_index = len(trajectory.time) - 1 if through_index is None else through_index
    if not 0 <= final_index < len(trajectory.time):
        raise ValueError("through_index is outside the trajectory")

    observed = _model_tensor(model, trajectory.temperature)
    commands = _model_tensor(model, trajectory.commands)
    dt = _model_tensor(model, trajectory.dt)
    mask = torch.tensor(trajectory.mask.copy(), dtype=torch.bool, device=model.capacity.device)
    estimator = _default_state_estimator(model) if observer is None else observer
    actuator = _resolved_initial_actuator(model, trajectory, commands, initial_actuator)
    state = estimator.initialize_posterior(observed[0], mask=mask[0], actuator=actuator)
    for index in range(final_index):
        state, _ = estimator.step(
            state,
            commands[index],
            observed[index + 1],
            dt[index],
            mask[index + 1],
        )
    return state


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
    state = _estimate_observer_state(
        model,
        trajectory,
        through_index=through_index,
        initial_actuator=initial_actuator,
        observer=observer,
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
    estimator = _default_state_estimator(model) if observer is None else observer
    posterior = _estimate_observer_state(
        model,
        trajectory,
        through_index=origin,
        initial_actuator=initial_actuator,
        observer=estimator,
    )
    commands = _model_tensor(model, trajectory.commands[origin:])
    dt = _model_tensor(model, trajectory.dt[origin:])
    # Unknown heat and bias estimates explain the observed past but are not assumed
    # to continue. Their covariance is retained so the forecast interval still
    # reflects uncertainty about uncommanded heat and the estimated physical state.
    state = ObserverState(
        posterior.temperature,
        posterior.actuator,
        torch.zeros_like(posterior.heat_disturbance),
        torch.zeros_like(posterior.bias_state),
        posterior.covariance,
    )
    states = [state.temperature]
    actuators = [state.actuator]
    node_stds: list[torch.Tensor] = []
    sensor_stds: list[torch.Tensor] = []

    def append_uncertainty(item: ObserverState) -> None:
        physical_covariance = item.covariance[: model.n_nodes, : model.n_nodes]
        node_stds.append(torch.sqrt(torch.clamp(torch.diag(physical_covariance), min=0.0)))
        sensor_covariance = model.observation @ physical_covariance @ model.observation.T
        sensor_stds.append(torch.sqrt(torch.clamp(torch.diag(sensor_covariance), min=0.0)))

    append_uncertainty(state)
    for index in range(len(commands)):
        state = estimator.predict(state, commands[index], dt[index])
        states.append(state.temperature)
        actuators.append(state.actuator)
        append_uncertainty(state)

    temperatures = torch.stack(states)
    return ForecastResult(
        time=trajectory.time[origin:].copy(),
        sensor_temperature=model.observe(temperatures).cpu().numpy(),
        sensor_temperature_std=torch.stack(sensor_stds).cpu().numpy(),
        node_temperature=temperatures.cpu().numpy(),
        node_temperature_std=torch.stack(node_stds).cpu().numpy(),
        actuator=torch.stack(actuators).cpu().numpy(),
        forecast_origin_index=origin,
    )


@torch.no_grad()
def monitor(
    model: ThermalRCModel,
    trajectory: Trajectory,
    *,
    initial_actuator: np.ndarray | None = None,
    observer: KalmanObserver | None = None,
    disturbance_process_std: float = _DEFAULT_DISTURBANCE_PROCESS_STD,
    bias_process_std: float = _DEFAULT_BIAS_PROCESS_STD,
    sensor_std: float = _DEFAULT_SENSOR_STD,
    innovation_gate_sigma: float = _DEFAULT_INNOVATION_GATE_SIGMA,
    bias_reference: str | None = _DEFAULT_BIAS_REFERENCE,
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
    observer_overrides: dict[str, object] = {
        "disturbance_process_std": disturbance_process_std,
        "bias_process_std": bias_process_std,
        "sensor_std": sensor_std,
        "innovation_gate_sigma": innovation_gate_sigma,
        "bias_reference": bias_reference,
    }
    if observer is not None:
        conflicts = sorted(
            name
            for name, value in observer_overrides.items()
            if value != _MONITOR_OBSERVER_DEFAULTS[name]
        )
        if conflicts:
            raise ValueError(f"observer conflicts with explicit monitor settings: {conflicts}")
    else:
        observer = build_observer(model, resolve_observer_settings("monitor", observer_overrides))
    if observer.bias_reference is not None:
        reference_index = trajectory.sensor_names.index(observer.bias_reference)
        if not trajectory.mask[:, reference_index].any():
            raise ValueError("bias reference sensor has no observations")
    actuator = _resolved_initial_actuator(model, trajectory, commands, initial_actuator)
    state = observer.initialize_posterior(observed[0], mask=mask[0], actuator=actuator)

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
