"""Causal state initialization from a measured trajectory prefix."""

from __future__ import annotations

import numpy as np
import torch

from celltemp.domain import Trajectory
from celltemp.engine import KalmanObserver, ObserverState, ThermalRCModel, ThermalState

from .settings import build_observer, resolve_observer_settings


def model_tensor(model: ThermalRCModel, values: np.ndarray) -> torch.Tensor:
    """Copy numerical input onto the model dtype and device."""
    return torch.tensor(
        np.asarray(values).copy(),
        dtype=model.capacity.dtype,
        device=model.capacity.device,
    )


def resolved_initial_actuator(
    model: ThermalRCModel,
    trajectory: Trajectory,
    commands: torch.Tensor,
    override: np.ndarray | None,
) -> torch.Tensor:
    """Resolve and validate the effective actuator at the first trajectory row."""
    values = override if override is not None else trajectory.initial_actuator
    if values is None:
        return commands[0]
    actuator = model_tensor(model, values)
    if actuator.shape != (model.n_controls,) or not bool(torch.isfinite(actuator).all()):
        raise ValueError("initial_actuator must contain one finite value per control")
    return actuator


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


def default_state_estimator(model: ThermalRCModel) -> KalmanObserver:
    """Build the weak-prior observer used for physical-state initialization."""
    return build_observer(model, resolve_observer_settings("forecast"))


@torch.no_grad()
def estimate_observer_state(
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

    observed = model_tensor(model, trajectory.temperature)
    commands = model_tensor(model, trajectory.commands)
    dt = model_tensor(model, trajectory.dt)
    mask = torch.tensor(trajectory.mask.copy(), dtype=torch.bool, device=model.capacity.device)
    estimator = default_state_estimator(model) if observer is None else observer
    actuator = resolved_initial_actuator(model, trajectory, commands, initial_actuator)
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
    """Causally estimate physical temperature and effective actuator state."""
    state = estimate_observer_state(
        model,
        trajectory,
        through_index=through_index,
        initial_actuator=initial_actuator,
        observer=observer,
    )
    return ThermalState(state.temperature, state.actuator)
