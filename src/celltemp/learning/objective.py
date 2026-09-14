"""Full-trajectory objectives in physical temperature units."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from celltemp.domain import Trajectory
from celltemp.engine import ThermalRCModel


def trajectory_tensors(
    model: ThermalRCModel, trajectory: Trajectory
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    trajectory.require_layout(model.spec.sensor_names, model.spec.control_names)
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
    initial_actuator: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replay actuator dynamics to the state at ``time[start]``."""
    actuator = commands[0] if initial_actuator is None else initial_actuator
    for index in range(start):
        actuator = model.actuator_step(actuator, commands[index], dt[index])
    return actuator


def trajectory_initial_actuator(
    model: ThermalRCModel,
    trajectory: Trajectory,
    commands: torch.Tensor,
) -> torch.Tensor:
    """Return a measured effective initial state or the settled-command default."""
    if trajectory.initial_actuator is None:
        return commands[0]
    return torch.tensor(
        trajectory.initial_actuator.copy(),
        dtype=model.capacity.dtype,
        device=model.capacity.device,
    )


def profiled_initial_temperature(
    model: ThermalRCModel,
    observed: torch.Tensor,
    commands: torch.Tensor,
    dt: torch.Tensor,
    mask: torch.Tensor,
    *,
    start: int,
    stop: int,
    prior_std: float = 50.0,
    initial_actuator: torch.Tensor | None = None,
) -> torch.Tensor:
    """Profile unobserved shooting-point temperatures as nuisance states.

    The thermal model is affine in its initial temperature. Nuisance directions
    span the null space of the available observation matrix, so profiling preserves
    the shooting-point measurements for direct, duplicate, and weighted sensors.
    This keeps hidden initial conditions from being absorbed into physical RC
    parameters without adding case-specific state to the deployed artifact. A
    Gaussian prior around the observed-temperature initialization prevents weakly
    observable modes from taking implausibly large values.
    """
    if not math.isfinite(prior_std) or prior_std <= 0.0:
        raise ValueError("initial temperature prior std must be positive and finite")
    initial = model.initialize_temperature(observed[start], mask[start])
    observation = model.observation[mask[start]]
    _, singular_values, right_vectors = torch.linalg.svd(observation, full_matrices=True)
    tolerance = torch.finfo(observation.dtype).eps * max(observation.shape) * singular_values.max()
    rank = int(torch.sum(singular_values > tolerance).detach().cpu().item())
    basis = right_vectors[rank:].T
    latent_count = basis.shape[1]
    if not latent_count:
        return initial

    profile_count = latent_count + 1
    candidate_initials = torch.cat(
        [initial.unsqueeze(0), initial.unsqueeze(0) + basis.T],
        dim=0,
    )
    initial_actuator = actuator_before_interval(
        model,
        commands,
        dt,
        start,
        initial_actuator,
    )
    states, _ = model.forward_batch(
        candidate_initials,
        commands[start:stop].unsqueeze(0).expand(profile_count, -1, -1),
        dt[start:stop].unsqueeze(0).expand(profile_count, -1),
        initial_actuator.unsqueeze(0).expand(profile_count, -1),
    )
    sensor_states = model.observe(states)
    baseline = sensor_states[0]
    sensitivity = (sensor_states[1:] - baseline).movedim(0, -1)
    selected_mask = mask[start : stop + 1]
    design = sensitivity[selected_mask]
    residual = (observed[start : stop + 1] - baseline)[selected_mask]
    gram = design.T @ design
    scale = torch.clamp(torch.trace(gram) / latent_count, min=1.0)
    numerical_ridge = torch.finfo(initial.dtype).eps ** 0.5 * scale
    prior_precision = initial.new_tensor(prior_std).reciprocal().square()
    ridge = numerical_ridge + prior_precision
    correction = torch.linalg.solve(
        gram + ridge * torch.eye(latent_count, dtype=initial.dtype, device=initial.device),
        design.T @ residual,
    )
    return initial + basis @ correction


def predict_trajectory(
    model: ThermalRCModel,
    trajectory: Trajectory,
    *,
    start: int = 0,
    stop: int | None = None,
    initial_temperature_prior_std: float = 50.0,
) -> torch.Tensor:
    """Return a conditional fit after profiling hidden shooting-point temperatures."""
    observed, commands, dt, mask = trajectory_tensors(model, trajectory)
    starting_actuator = trajectory_initial_actuator(model, trajectory, commands)
    stop = len(trajectory.time) - 1 if stop is None else stop
    if not 0 <= start < stop <= len(trajectory.time) - 1:
        raise ValueError("start/stop must select at least one valid interval")
    if not torch.any(mask[start]):
        raise ValueError("the shooting-point row must contain at least one observation")
    initial_temperature = profiled_initial_temperature(
        model,
        observed,
        commands,
        dt,
        mask,
        start=start,
        stop=stop,
        prior_std=initial_temperature_prior_std,
        initial_actuator=starting_actuator,
    )
    initial_actuator = actuator_before_interval(model, commands, dt, start, starting_actuator)
    state, _ = model.forward_trajectory(
        initial_temperature,
        commands[start:stop],
        dt[start:stop],
        initial_actuator,
    )
    return model.observe(state)


def predict_from_initial_observation(
    model: ThermalRCModel,
    trajectory: Trajectory,
    *,
    start: int = 0,
    stop: int | None = None,
) -> torch.Tensor:
    """Roll out using only observations available at the shooting-point row."""
    observed, commands, dt, mask = trajectory_tensors(model, trajectory)
    starting_actuator = trajectory_initial_actuator(model, trajectory, commands)
    stop = len(trajectory.time) - 1 if stop is None else stop
    if not 0 <= start < stop <= len(trajectory.time) - 1:
        raise ValueError("start/stop must select at least one valid interval")
    if not torch.any(mask[start]):
        raise ValueError("the shooting-point row must contain at least one observation")
    initial_temperature = model.initialize_temperature(observed[start], mask[start])
    initial_actuator = actuator_before_interval(model, commands, dt, start, starting_actuator)
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
    initial_temperature_prior_std: float = 50.0,
) -> torch.Tensor:
    """Case-balanced rollout loss measured directly in Kelvin."""
    observed, _, _, mask = trajectory_tensors(model, trajectory)
    stop = len(trajectory.time) - 1 if stop is None else stop
    predicted = predict_trajectory(
        model,
        trajectory,
        start=start,
        stop=stop,
        initial_temperature_prior_std=initial_temperature_prior_std,
    )
    offset = 0 if include_initial else 1
    return masked_huber_loss(
        predicted[offset:],
        observed[start + offset : stop + 1],
        mask[start + offset : stop + 1],
        delta=huber_delta,
    )


def _trajectory_rmse(
    observed: torch.Tensor,
    predicted: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    evaluation_mask = mask.clone()
    evaluation_mask[0] = False
    if not torch.any(evaluation_mask):
        raise ValueError("trajectory has no observations after the initial row")
    error = predicted[evaluation_mask] - observed[evaluation_mask]
    return float(torch.sqrt(torch.mean(error.square())).cpu())


@torch.no_grad()
def trajectory_rmse(
    model: ThermalRCModel,
    trajectory: Trajectory,
    *,
    initial_temperature_prior_std: float = 50.0,
) -> float:
    """Return conditional-fit RMSE after profiling hidden initial temperatures."""
    observed, _, _, mask = trajectory_tensors(model, trajectory)
    predicted = predict_trajectory(
        model,
        trajectory,
        initial_temperature_prior_std=initial_temperature_prior_std,
    )
    return _trajectory_rmse(observed, predicted, mask)


@torch.no_grad()
def initial_observation_rmse(model: ThermalRCModel, trajectory: Trajectory) -> float:
    """Return open-loop RMSE initialized only from the first observation row."""
    observed, _, _, mask = trajectory_tensors(model, trajectory)
    predicted = predict_from_initial_observation(model, trajectory)
    return _trajectory_rmse(observed, predicted, mask)
