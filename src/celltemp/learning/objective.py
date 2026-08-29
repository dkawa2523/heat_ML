"""Full-trajectory objectives in physical temperature units."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from celltemp.domain import Trajectory
from celltemp.engine import ThermalRCModel


def _check_names(model: ThermalRCModel, trajectory: Trajectory) -> None:
    if trajectory.sensor_names != model.spec.sensor_names:
        raise ValueError(
            f"trajectory sensors {trajectory.sensor_names} do not match "
            f"system sensors {model.spec.sensor_names}"
        )
    if trajectory.control_names != model.spec.control_names:
        raise ValueError(
            f"trajectory controls {trajectory.control_names} do not match "
            f"system controls {model.spec.control_names}"
        )


def trajectory_tensors(
    model: ThermalRCModel, trajectory: Trajectory
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_names(model, trajectory)
    device = model.capacity.device
    dtype = model.capacity.dtype
    temperature = torch.tensor(trajectory.temperature.copy(), dtype=dtype, device=device)
    commands = torch.tensor(trajectory.commands.copy(), dtype=dtype, device=device)
    dt = torch.tensor(trajectory.dt.copy(), dtype=dtype, device=device)
    mask = torch.tensor(trajectory.mask.copy(), dtype=torch.bool, device=device)
    return temperature, commands, dt, mask


def actuator_before_interval(
    model: ThermalRCModel,
    commands: torch.Tensor,
    dt: torch.Tensor,
    start: int,
) -> torch.Tensor:
    """Replay actuator dynamics to the state at ``time[start]``."""
    actuator = commands[0]
    for index in range(start):
        actuator = model.actuator_step(actuator, commands[index], dt[index])
    return actuator


def predict_trajectory(
    model: ThermalRCModel,
    trajectory: Trajectory,
    *,
    start: int = 0,
    stop: int | None = None,
) -> torch.Tensor:
    """Predict sensor temperatures from an observed shooting-point state."""
    observed, commands, dt, mask = trajectory_tensors(model, trajectory)
    stop = len(trajectory.time) - 1 if stop is None else stop
    if not 0 <= start < stop <= len(trajectory.time) - 1:
        raise ValueError("start/stop must select at least one valid interval")
    if not torch.any(mask[start]):
        raise ValueError("the shooting-point row must contain at least one observation")
    initial_temperature = model.initialize_temperature(observed[start], mask[start])
    initial_actuator = actuator_before_interval(model, commands, dt, start)
    state, _ = model.forward_trajectory(
        initial_temperature,
        commands[start:stop],
        dt[start:stop],
        initial_actuator,
    )
    return model.observe(state)


def masked_huber_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    if predicted.shape != observed.shape or mask.shape != observed.shape:
        raise ValueError("predicted, observed, and mask shapes must match")
    if not torch.any(mask):
        raise ValueError("loss window has no observations")
    return F.huber_loss(predicted[mask], observed[mask], delta=delta, reduction="mean")


def trajectory_loss(
    model: ThermalRCModel,
    trajectory: Trajectory,
    *,
    start: int = 0,
    stop: int | None = None,
    huber_delta: float = 1.0,
    include_initial: bool = False,
) -> torch.Tensor:
    """Case-balanced rollout loss measured directly in Kelvin."""
    observed, _, _, mask = trajectory_tensors(model, trajectory)
    stop = len(trajectory.time) - 1 if stop is None else stop
    predicted = predict_trajectory(model, trajectory, start=start, stop=stop)
    offset = 0 if include_initial else 1
    return masked_huber_loss(
        predicted[offset:],
        observed[start + offset : stop + 1],
        mask[start + offset : stop + 1],
        delta=huber_delta,
    )


@torch.no_grad()
def trajectory_rmse(model: ThermalRCModel, trajectory: Trajectory) -> float:
    observed, _, _, mask = trajectory_tensors(model, trajectory)
    predicted = predict_trajectory(model, trajectory)
    evaluation_mask = mask.clone()
    evaluation_mask[0] = False
    error = predicted[evaluation_mask] - observed[evaluation_mask]
    return float(torch.sqrt(torch.mean(error.square())).cpu())
