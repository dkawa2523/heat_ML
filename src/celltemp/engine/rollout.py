"""Time integration and batched rollout for the thermal RC model."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import torch

from .integrator import exact_affine_operators, exact_affine_step, implicit_euler_step
from .operators import (
    forcing,
    joint_affine_system,
    operator_context,
    source_activity,
    system_matrix,
)
from .state import ThermalState
from .validation import require_finite, require_time_step

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
    require_finite(actuator, "actuator")
    require_finite(command, "command")
    step = torch.as_tensor(dt, dtype=actuator.dtype, device=actuator.device)
    require_time_step(step)
    tau = model.actuator_tau().to(device=actuator.device, dtype=actuator.dtype)
    require_finite(tau, "actuator tau", computed=True)
    return _actuator_response(actuator, command, step, tau)


def _actuator_response(
    actuator: torch.Tensor,
    command: torch.Tensor,
    step: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Integrate validated interval inputs with the rollout's shared tau values."""
    while step.ndim < actuator.ndim:
        step = step.unsqueeze(-1)
    lagged = tau > 0.0
    safe_tau = torch.where(lagged, tau, torch.ones_like(tau))
    decay = torch.exp(-step / safe_tau)
    response = command + (actuator - command) * decay
    result = torch.where(lagged, response, command)
    require_finite(result, "actuator response", computed=True)
    return result


def threshold_crossing_times(
    model: ThermalRCModel,
    actuator: torch.Tensor,
    command: torch.Tensor,
    dt: torch.Tensor,
    tau: torch.Tensor,
) -> list[torch.Tensor]:
    """Return exact threshold crossing times for monotone first-order actuators."""
    positive = model.source_laws._positive_indices
    controls = model.source_laws._control_indices
    lagged_sources = [
        index for index in positive if model.spec.actuators[controls[index]].tau > 0.0
    ]
    if not lagged_sources:
        return []
    step_value = float(dt.detach().cpu().item())
    crossings: list[torch.Tensor] = []
    for source_index in lagged_sources:
        control_index = controls[source_index]
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


def point_actuator(
    model: ThermalRCModel,
    actuator: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """Apply instantaneous controls at a point; retain continuous lagged states."""
    if actuator.shape != command.shape or actuator.shape[-1] != model.n_controls:
        raise ValueError("actuator and command must have equal [..., n_controls] shape")
    require_finite(actuator, "actuator")
    require_finite(command, "command")
    tau = model.actuator_tau().to(dtype=actuator.dtype, device=actuator.device)
    require_finite(tau, "actuator tau", computed=True)
    return torch.where(tau > 0.0, actuator, command)


def _exact_joint_segment(
    model: ThermalRCModel,
    temperature: torch.Tensor,
    actuator: torch.Tensor,
    command: torch.Tensor,
    dt: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    midpoint = _actuator_response(actuator, command, dt * 0.5, tau)
    active_sources = source_activity(model, midpoint)
    matrix, affine = joint_affine_system(model, command, active_sources, tau)
    joint = torch.cat([temperature, actuator], dim=-1)
    result = exact_affine_step(joint, matrix, affine, dt)
    return result[..., : model.n_nodes], result[..., model.n_nodes :]


def exact_joint_step(
    model: ThermalRCModel,
    temperature: torch.Tensor,
    actuator: torch.Tensor,
    command: torch.Tensor,
    dt: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one exact interval, splitting at positive-part source thresholds."""
    actuator = torch.where(tau > 0.0, actuator, command)
    crossings = threshold_crossing_times(model, actuator, command, dt, tau)
    start = torch.zeros((), dtype=dt.dtype, device=dt.device)
    for end in [*crossings, dt]:
        temperature, actuator = _exact_joint_segment(
            model,
            temperature,
            actuator,
            command,
            end - start,
            tau,
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
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance a batch and reuse exact operators for matching interval contexts."""
    actuator = torch.where(tau > 0.0, actuator, command)
    outputs: dict[int, torch.Tensor] = {}
    groups: dict[OperatorKey, list[int]] = {}

    for batch_index in range(len(temperature)):
        crossings = threshold_crossing_times(
            model,
            actuator[batch_index],
            command[batch_index],
            dt[batch_index],
            tau,
        )
        if crossings:
            next_temperature, next_actuator = exact_joint_step(
                model,
                temperature[batch_index],
                actuator[batch_index],
                command[batch_index],
                dt[batch_index],
                tau,
            )
            outputs[batch_index] = torch.cat([next_temperature, next_actuator])
            continue
        midpoint = _actuator_response(
            actuator[batch_index],
            command[batch_index],
            dt[batch_index] * 0.5,
            tau,
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
        activity = torch.tensor(key[1], dtype=torch.bool, device=command.device).expand(
            len(indices), -1
        )
        matrix, affine = joint_affine_system(model, group_command, activity, tau)
        if key not in cache:
            cache[key] = exact_affine_operators(matrix[0], dt[indices[0]])
        phi, gamma = cache[key]
        joint = torch.cat([temperature[indices], group_actuator], dim=-1)
        result = joint @ phi.T + affine @ gamma.T
        require_finite(result, "exact batch state", computed=True)
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
    require_time_step(step)
    if step.ndim != 0:
        raise ValueError("temperature transition dt must be scalar")
    if model.integrator == "exact":
        transition = torch.matrix_exp(system_matrix * step)
        require_finite(transition, "temperature transition", computed=True)
        return transition
    identity = torch.eye(
        model.n_nodes,
        dtype=model.capacity.dtype,
        device=model.capacity.device,
    )
    transition = torch.linalg.solve(identity - step * system_matrix, identity)
    require_finite(transition, "temperature transition", computed=True)
    return transition


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
    require_time_step(interval)
    require_finite(command, "command")
    if state.temperature.shape[-1] != model.n_nodes:
        raise ValueError("temperature has the wrong number of nodes")
    if state.actuator.shape != command.shape or command.shape[-1] != model.n_controls:
        raise ValueError("actuator and command must have equal [..., n_controls] shape")
    require_finite(state.actuator, "actuator")
    tau = model.actuator_tau().to(dtype=command.dtype, device=command.device)
    require_finite(tau, "actuator tau", computed=True)
    if model.integrator == "exact":
        temperature, actuator = exact_joint_step(
            model,
            state.temperature,
            state.actuator,
            command,
            interval,
            tau,
        )
        return ThermalState._from_validated(temperature, actuator)

    midpoint = _actuator_response(state.actuator, command, interval * 0.5, tau)
    next_actuator = _actuator_response(state.actuator, command, interval, tau)
    next_temperature = implicit_euler_step(
        state.temperature,
        system_matrix(model, midpoint),
        forcing(model, midpoint),
        interval,
    )
    return ThermalState._from_validated(next_temperature, next_actuator)


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
        require_finite(initial_actuator, "initial_actuator")
    if initial_temperature.ndim != 2 or initial_temperature.shape[1] != model.n_nodes:
        raise ValueError("initial_temperature must have shape [batch, n_nodes]")
    if commands.ndim != 3 or commands.shape[2] != model.n_controls:
        raise ValueError("commands must have shape [batch, steps, n_controls]")
    if commands.shape[1] == 0 or commands.shape[0] == 0:
        raise ValueError("commands need at least one trajectory and interval")
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
    require_finite(initial_temperature, "initial_temperature")
    require_finite(commands, "commands")
    require_time_step(dt)
    return initial_temperature, commands, dt, actuator


def forward_batch(
    model: ThermalRCModel,
    initial_temperature: torch.Tensor,
    commands: torch.Tensor,
    dt: torch.Tensor,
    initial_actuator: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate trajectories and return point values aligned to starting commands.

    Zero-tau actuator values at time[k] equal commands[k]. Lagged actuator states
    remain continuous. The final point holds the last interval's command.
    """
    initial_temperature, commands, dt, actuator = prepare_batch_inputs(
        model,
        initial_temperature,
        commands,
        dt,
        initial_actuator,
    )
    temperature = initial_temperature
    temperatures = [temperature]
    operator_cache: OperatorCache = {}
    tau = model.actuator_tau()
    require_finite(tau, "actuator tau", computed=True)
    actuator = torch.where(tau > 0.0, actuator, commands[:, 0])
    actuators = [actuator]

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
                tau,
            )
        else:
            midpoint = _actuator_response(actuator, command, interval_dt * 0.5, tau)
            actuator = _actuator_response(actuator, command, interval_dt, tau)
            temperature = implicit_euler_step(
                temperature,
                system_matrix(model, midpoint),
                forcing(model, midpoint),
                interval_dt,
            )
        temperatures.append(temperature)
        point_command = commands[:, min(index + 1, commands.shape[1] - 1)]
        actuator = torch.where(tau > 0.0, actuator, point_command)
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
