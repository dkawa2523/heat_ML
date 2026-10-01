"""Time integration and batched rollout for the thermal RC model."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import torch

from .integrator import exact_affine_operators, exact_affine_step, implicit_euler_step
from .operators import joint_affine_system, operator_context, source_activity
from .state import ThermalState

if TYPE_CHECKING:
    from .rc import ThermalRCModel

OperatorKey: TypeAlias = tuple[float, tuple[bool, ...], tuple[float, ...]]
OperatorCache: TypeAlias = dict[OperatorKey, tuple[torch.Tensor, torch.Tensor]]


def actuator_step(
    model: ThermalRCModel,
    actuator: torch.Tensor,
    command: torch.Tensor,
    dt: float | torch.Tensor,
) -> torch.Tensor:
    """Evaluate the exact first-order actuator response over one interval."""
    if actuator.shape != command.shape or actuator.shape[-1] != model.n_controls:
        raise ValueError("actuator and command must have equal [..., n_controls] shape")
    step = torch.as_tensor(dt, dtype=actuator.dtype, device=actuator.device)
    while step.ndim < actuator.ndim:
        step = step.unsqueeze(-1)
    tau = model.actuator_tau().to(device=actuator.device, dtype=actuator.dtype)
    lagged = tau > 0.0
    safe_tau = torch.where(lagged, tau, torch.ones_like(tau))
    decay = torch.exp(-step / safe_tau)
    response = command + (actuator - command) * decay
    return torch.where(lagged, response, command)


def threshold_crossing_times(
    model: ThermalRCModel,
    actuator: torch.Tensor,
    command: torch.Tensor,
    dt: torch.Tensor,
) -> list[torch.Tensor]:
    """Return exact threshold crossing times for monotone first-order actuators."""
    if not model.spec.sources:
        return []
    tau = model.actuator_tau().to(dtype=actuator.dtype, device=actuator.device)
    step_value = float(dt.detach().cpu().item())
    crossings: list[torch.Tensor] = []
    for source_index in range(len(model.spec.sources)):
        if not bool(model.source_laws.positive_part_mask[source_index].item()):
            continue
        control_index = int(model.source_laws.control_index[source_index].item())
        if float(tau[control_index].detach().cpu().item()) <= 0.0:
            continue
        threshold = model.source_laws.threshold[source_index]
        initial = actuator[control_index]
        target = command[control_index]
        initial_side = float((initial - threshold).detach().cpu().item())
        target_side = float((target - threshold).detach().cpu().item())
        if initial_side * target_side >= 0.0:
            continue
        ratio = (threshold - target) / (initial - target)
        crossing = -tau[control_index] * torch.log(ratio)
        crossing_value = float(crossing.detach().cpu().item())
        if 0.0 < crossing_value < step_value:
            crossings.append(crossing)
    return sorted(crossings, key=lambda value: float(value.detach().cpu().item()))


def _interval_start_actuator(
    model: ThermalRCModel,
    actuator: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    tau = model.actuator_tau().to(dtype=actuator.dtype, device=actuator.device)
    return torch.where(tau > 0.0, actuator, command)


def _exact_joint_segment(
    model: ThermalRCModel,
    temperature: torch.Tensor,
    actuator: torch.Tensor,
    command: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    midpoint = actuator_step(model, actuator, command, dt * 0.5)
    active_sources = source_activity(model, midpoint)
    matrix, affine = joint_affine_system(model, command, active_sources)
    joint = torch.cat([temperature, actuator], dim=-1)
    result = exact_affine_step(joint, matrix, affine, dt)
    return result[..., : model.n_nodes], result[..., model.n_nodes :]


def exact_joint_step(
    model: ThermalRCModel,
    temperature: torch.Tensor,
    actuator: torch.Tensor,
    command: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one exact interval, splitting at positive-part source thresholds."""
    actuator = _interval_start_actuator(model, actuator, command)
    crossings = threshold_crossing_times(model, actuator, command, dt)
    start = torch.zeros((), dtype=dt.dtype, device=dt.device)
    for end in [*crossings, dt]:
        temperature, actuator = _exact_joint_segment(
            model,
            temperature,
            actuator,
            command,
            end - start,
        )
        start = end
    return temperature, actuator


def exact_joint_batch_step(
    model: ThermalRCModel,
    temperature: torch.Tensor,
    actuator: torch.Tensor,
    command: torch.Tensor,
    dt: torch.Tensor,
    cache: OperatorCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance a batch and reuse exact operators for matching interval contexts."""
    actuator = _interval_start_actuator(model, actuator, command)
    outputs: dict[int, torch.Tensor] = {}
    groups: dict[OperatorKey, list[int]] = {}

    for batch_index in range(len(temperature)):
        crossings = threshold_crossing_times(
            model,
            actuator[batch_index],
            command[batch_index],
            dt[batch_index],
        )
        if crossings:
            next_temperature, next_actuator = exact_joint_step(
                model,
                temperature[batch_index],
                actuator[batch_index],
                command[batch_index],
                dt[batch_index],
            )
            outputs[batch_index] = torch.cat([next_temperature, next_actuator])
            continue
        midpoint = actuator_step(
            model,
            actuator[batch_index],
            command[batch_index],
            dt[batch_index] * 0.5,
        )
        activity = source_activity(model, midpoint)
        pattern = tuple(bool(value) for value in activity.detach().cpu().tolist())
        key = (
            float(dt[batch_index].detach().cpu().item()),
            pattern,
            operator_context(model, actuator[batch_index]),
        )
        groups.setdefault(key, []).append(batch_index)

    for key, indices in groups.items():
        group_command = command[indices]
        group_actuator = actuator[indices]
        midpoint = actuator_step(
            model,
            group_actuator,
            group_command,
            dt[indices] * 0.5,
        )
        activity = source_activity(model, midpoint)
        matrix, affine = joint_affine_system(model, group_command, activity)
        if key not in cache:
            cache[key] = exact_affine_operators(matrix[0], dt[indices[0]])
        phi, gamma = cache[key]
        joint = torch.cat([temperature[indices], group_actuator], dim=-1)
        result = joint @ phi.T + affine @ gamma.T
        for group_index, batch_index in enumerate(indices):
            outputs[batch_index] = result[group_index]

    result = torch.stack([outputs[index] for index in range(len(temperature))])
    return result[:, : model.n_nodes], result[:, model.n_nodes :]


def temperature_transition_matrix(
    model: ThermalRCModel,
    dt: float | torch.Tensor,
    actuator: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the linear sensitivity of next temperature to current temperature."""
    system_matrix = model.system_matrix(actuator)
    step = torch.as_tensor(dt, dtype=model.capacity.dtype, device=model.capacity.device)
    if step.ndim != 0:
        raise ValueError("temperature transition dt must be scalar")
    if model.integrator == "exact":
        return torch.matrix_exp(system_matrix * step)
    identity = torch.eye(
        model.n_nodes,
        dtype=model.capacity.dtype,
        device=model.capacity.device,
    )
    return torch.linalg.solve(identity - step * system_matrix, identity)


def step(
    model: ThermalRCModel,
    state: ThermalState,
    command: torch.Tensor,
    dt: float | torch.Tensor,
) -> ThermalState:
    """Advance actuator and thermal states through one command interval."""
    command = command.to(dtype=state.temperature.dtype, device=state.temperature.device)
    interval = torch.as_tensor(
        dt,
        dtype=state.temperature.dtype,
        device=state.temperature.device,
    )
    if model.integrator == "exact":
        temperature, actuator = exact_joint_step(
            model,
            state.temperature,
            state.actuator,
            command,
            interval,
        )
        return ThermalState(temperature, actuator)

    midpoint = actuator_step(model, state.actuator, command, interval * 0.5)
    next_actuator = actuator_step(model, state.actuator, command, interval)
    next_temperature = implicit_euler_step(
        state.temperature,
        model.system_matrix(midpoint),
        model.forcing(midpoint),
        interval,
    )
    return ThermalState(next_temperature, next_actuator)


def prepare_batch_inputs(
    model: ThermalRCModel,
    initial_temperature: torch.Tensor,
    commands: torch.Tensor,
    dt: torch.Tensor,
    initial_actuator: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize devices and validate the common batched rollout shape."""
    initial_temperature = initial_temperature.to(
        dtype=model.capacity.dtype,
        device=model.capacity.device,
    )
    commands = commands.to(dtype=model.capacity.dtype, device=model.capacity.device)
    dt = dt.to(dtype=model.capacity.dtype, device=model.capacity.device)
    if initial_actuator is not None:
        initial_actuator = initial_actuator.to(
            dtype=model.capacity.dtype,
            device=model.capacity.device,
        )
    if initial_temperature.ndim != 2 or initial_temperature.shape[1] != model.n_nodes:
        raise ValueError("initial_temperature must have shape [batch, n_nodes]")
    if commands.ndim != 3 or commands.shape[2] != model.n_controls:
        raise ValueError("commands must have shape [batch, steps, n_controls]")
    if commands.shape[0] != initial_temperature.shape[0]:
        raise ValueError("initial_temperature and commands batch sizes must match")
    if dt.ndim == 1:
        if dt.shape[0] != commands.shape[1]:
            raise ValueError("shared dt must have one value per step")
        dt = dt.unsqueeze(0).expand(commands.shape[0], -1)
    elif dt.shape != commands.shape[:2]:
        raise ValueError("dt must have shape [steps] or [batch, steps]")

    actuator = commands[:, 0] if initial_actuator is None else initial_actuator
    if actuator.shape != (commands.shape[0], model.n_controls):
        raise ValueError("initial_actuator must have shape [batch, n_controls]")
    return initial_temperature, commands, dt, actuator


def forward_batch(
    model: ThermalRCModel,
    initial_temperature: torch.Tensor,
    commands: torch.Tensor,
    dt: torch.Tensor,
    initial_actuator: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate equal-length trajectories as one differentiable batch."""
    initial_temperature, commands, dt, actuator = prepare_batch_inputs(
        model,
        initial_temperature,
        commands,
        dt,
        initial_actuator,
    )
    temperature = initial_temperature
    temperatures = [temperature]
    actuators = [actuator]
    operator_cache: OperatorCache = {}

    for index in range(commands.shape[1]):
        interval_dt = dt[:, index]
        command = commands[:, index]
        if model.integrator == "exact":
            temperature, actuator = exact_joint_batch_step(
                model,
                temperature,
                actuator,
                command,
                interval_dt,
                operator_cache,
            )
        else:
            midpoint = actuator_step(model, actuator, command, interval_dt * 0.5)
            actuator = actuator_step(model, actuator, command, interval_dt)
            temperature = implicit_euler_step(
                temperature,
                model.system_matrix(midpoint),
                model.forcing(midpoint),
                interval_dt,
            )
        temperatures.append(temperature)
        actuators.append(actuator)
    return torch.stack(temperatures, dim=1), torch.stack(actuators, dim=1)


def forward_trajectory(
    model: ThermalRCModel,
    initial_temperature: torch.Tensor,
    commands: torch.Tensor,
    dt: torch.Tensor,
    initial_actuator: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate one trajectory through the shared batched rollout path."""
    actuator = None if initial_actuator is None else initial_actuator.unsqueeze(0)
    temperatures, actuators = forward_batch(
        model,
        initial_temperature.unsqueeze(0),
        commands.unsqueeze(0),
        dt.unsqueeze(0),
        actuator,
    )
    return temperatures[0], actuators[0]
