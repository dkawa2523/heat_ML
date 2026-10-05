"""Causal state initialization from a measured trajectory prefix."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from celltemp.domain import Trajectory
from celltemp.domain import forecast_origin_index as forecast_origin_index
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

    observed = model_tensor(model, trajectory.temperature[: final_index + 1])
    commands = model_tensor(model, trajectory.commands[: final_index + 1])
    dt = model_tensor(model, trajectory.dt[:final_index])
    mask = torch.tensor(
        trajectory.mask[: final_index + 1].copy(), dtype=torch.bool, device=model.capacity.device
    )
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
    point_command = commands[min(final_index, len(commands) - 1)]
    return replace(state, actuator=model.point_actuator(state.actuator, point_command))


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
