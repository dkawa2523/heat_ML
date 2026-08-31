"""A compact, physical-unit thermal RC state-space model."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from celltemp.domain.topology import ConstantLawSpec, PowerLawSpec, ThermalSystemSpec

from .integrator import exact_affine_operators, exact_affine_step, implicit_euler_step
from .laws import ScalarLawSet
from .state import ThermalState


def _tensor(values: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=dtype)


class ThermalRCModel(nn.Module):
    """Symmetric RC dynamics with explicit actuator states.

    The engine works only in physical units.  Positive quantities are represented
    by positive priors times exponential multipliers, and every conductive edge has
    one shared parameter, so reciprocal heat flow and internal energy conservation
    cannot be broken by training.
    """

    spec: ThermalSystemSpec
    integrator: str
    capacity: torch.Tensor
    edge_basis: torch.Tensor
    tau_prior: torch.Tensor
    tau_learn_mask: torch.Tensor
    source_weights: torch.Tensor
    boundary_temperature_control: torch.Tensor
    boundary_weights: torch.Tensor
    boundary_temperature_intercept: torch.Tensor
    boundary_temperature_slope: torch.Tensor
    observation: torch.Tensor

    @staticmethod
    def _validate_exact_support(spec: ThermalSystemSpec) -> None:
        """Reject time-varying coefficients outside the exact affine model class."""
        actuator_tau = {item.name: item.tau for item in spec.actuators}
        incompatible: list[str] = []
        for edge in spec.edges:
            law = edge.conductance
            if not isinstance(law, ConstantLawSpec) and actuator_tau[law.control] > 0.0:
                incompatible.append(f"edge {edge.node_a}-{edge.node_b}")
        for boundary in spec.boundaries:
            law = boundary.conductance
            if not isinstance(law, ConstantLawSpec) and actuator_tau[law.control] > 0.0:
                incompatible.append(f"boundary {boundary.name}")
        for source in spec.sources:
            law = source.heat_rate
            if isinstance(law, PowerLawSpec) and actuator_tau[law.control] > 0.0:
                incompatible.append(f"source {source.name}")
        if incompatible:
            paths = ", ".join(incompatible)
            raise ValueError(
                "exact integrator requires zero-tau controls for input-dependent "
                f"conductance and nonlinear source laws: {paths}; use the implicit integrator "
                "for lagged coefficients"
            )

    def __init__(
        self,
        spec: ThermalSystemSpec,
        *,
        integrator: str = "exact",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if integrator not in {"exact", "implicit"}:
            raise ValueError("integrator must be 'exact' or 'implicit'")
        if integrator == "exact":
            self._validate_exact_support(spec)
        self.spec = spec
        self.integrator = integrator

        node_index = {name: i for i, name in enumerate(spec.node_names)}
        control_index = {name: i for i, name in enumerate(spec.control_names)}

        self.register_buffer("capacity", _tensor(spec.heat_capacity, dtype=dtype))
        edge_basis = torch.zeros(
            (len(spec.edges), len(spec.node_names), len(spec.node_names)),
            dtype=dtype,
        )
        for edge_index, edge in enumerate(spec.edges):
            source = node_index[edge.node_a]
            destination = node_index[edge.node_b]
            edge_basis[edge_index, source, source] = 1.0
            edge_basis[edge_index, destination, destination] = 1.0
            edge_basis[edge_index, source, destination] = -1.0
            edge_basis[edge_index, destination, source] = -1.0
        self.register_buffer("edge_basis", edge_basis)
        self.edge_laws = ScalarLawSet(
            [edge.conductance for edge in spec.edges],
            spec.control_names,
            dtype=dtype,
        )

        self.register_buffer("tau_prior", _tensor([a.tau for a in spec.actuators], dtype=dtype))
        self.register_buffer(
            "tau_learn_mask",
            _tensor([a.learnable and a.tau > 0.0 for a in spec.actuators], dtype=dtype),
        )
        self.log_tau_multiplier = nn.Parameter(torch.zeros(len(spec.actuators), dtype=dtype))

        self.register_buffer(
            "source_weights",
            _tensor([s.node_weights for s in spec.sources], dtype=dtype).reshape(
                len(spec.sources), len(spec.node_names)
            ),
        )
        self.source_laws = ScalarLawSet(
            [source.heat_rate for source in spec.sources],
            spec.control_names,
            dtype=dtype,
        )

        temperature_controls = [
            -1
            if boundary.reservoir_temperature.control is None
            else control_index[boundary.reservoir_temperature.control]
            for boundary in spec.boundaries
        ]
        self.register_buffer(
            "boundary_temperature_control",
            torch.tensor(temperature_controls, dtype=torch.long),
        )
        self.register_buffer(
            "boundary_weights",
            _tensor([b.node_weights for b in spec.boundaries], dtype=dtype).reshape(
                len(spec.boundaries), len(spec.node_names)
            ),
        )
        self.register_buffer(
            "boundary_temperature_intercept",
            _tensor(
                [boundary.reservoir_temperature.intercept for boundary in spec.boundaries],
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "boundary_temperature_slope",
            _tensor(
                [boundary.reservoir_temperature.slope for boundary in spec.boundaries],
                dtype=dtype,
            ),
        )
        self.boundary_laws = ScalarLawSet(
            [boundary.conductance for boundary in spec.boundaries],
            spec.control_names,
            dtype=dtype,
        )
        self.register_buffer("observation", _tensor(spec.observation_matrix, dtype=dtype))

    @property
    def n_nodes(self) -> int:
        return len(self.spec.node_names)

    @property
    def n_controls(self) -> int:
        return len(self.spec.actuators)

    @property
    def n_sensors(self) -> int:
        return len(self.spec.sensor_names)

    def _positive(self, prior: torch.Tensor, raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        learnable = mask.to(dtype=torch.bool)
        if not torch.any(learnable):
            return prior
        value = prior.clone()
        value[learnable] = prior[learnable] * torch.exp(raw[learnable])
        return value

    def conductance(self, actuator: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate internal-path conductances for the effective inputs."""
        return self.edge_laws(actuator)

    def actuator_tau(self) -> torch.Tensor:
        return self._positive(self.tau_prior, self.log_tau_multiplier, self.tau_learn_mask)

    def source_heat_rate(self, actuator: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate source heat rates for the effective inputs."""
        return self.source_laws(actuator)

    def log_parameter_multipliers(self) -> tuple[torch.Tensor, ...]:
        """Return the dimensionless parameters regularized during identification."""
        return (
            self.log_tau_multiplier,
            *self.edge_laws.log_parameter_multipliers(),
            *self.source_laws.log_parameter_multipliers(),
            *self.boundary_laws.log_parameter_multipliers(),
        )

    def boundary_conductance(self, actuator: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate every boundary conductance for the effective physical inputs."""
        return self.boundary_laws(actuator)

    def actuator_step(
        self,
        actuator: torch.Tensor,
        command: torch.Tensor,
        dt: float | torch.Tensor,
    ) -> torch.Tensor:
        """Exact first-order actuator response over one interval."""
        if actuator.shape != command.shape or actuator.shape[-1] != self.n_controls:
            raise ValueError("actuator and command must have equal [..., n_controls] shape")
        step = torch.as_tensor(dt, dtype=actuator.dtype, device=actuator.device)
        while step.ndim < actuator.ndim:
            step = step.unsqueeze(-1)
        tau = self.actuator_tau().to(device=actuator.device, dtype=actuator.dtype)
        lagged = tau > 0.0
        safe_tau = torch.where(lagged, tau, torch.ones_like(tau))
        decay = torch.exp(-step / safe_tau)
        response = command + (actuator - command) * decay
        return torch.where(lagged, response, command)

    def _laplacian(self, actuator: torch.Tensor | None = None) -> torch.Tensor:
        return torch.einsum("...e,eij->...ij", self.conductance(actuator), self.edge_basis)

    def system_matrix(self, actuator: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``A`` in ``dT/dt = A T + b`` for effective controls."""
        laplacian = self._laplacian(actuator)
        boundary_total = torch.zeros(
            self.n_nodes, dtype=self.capacity.dtype, device=self.capacity.device
        )
        if len(self.spec.boundaries):
            boundary_h = self.boundary_conductance(actuator)[..., :, None] * self.boundary_weights
            boundary_total = boundary_h.sum(dim=-2)
        return -(laplacian + torch.diag_embed(boundary_total)) / self.capacity[..., None]

    def forcing(self, actuator: torch.Tensor) -> torch.Tensor:
        """Return the actuator-dependent ``b`` in ``dT/dt = A T + b``."""
        if actuator.shape[-1] != self.n_controls:
            raise ValueError("actuator has the wrong number of controls")
        heat = torch.zeros(
            (*actuator.shape[:-1], self.n_nodes),
            dtype=actuator.dtype,
            device=actuator.device,
        )

        if len(self.spec.sources):
            heat = heat + self.source_heat_rate(actuator) @ self.source_weights

        if len(self.spec.boundaries):
            boundary_h = self.boundary_conductance(actuator)[..., :, None] * self.boundary_weights
            selected = torch.zeros(
                (*actuator.shape[:-1], len(self.spec.boundaries)),
                dtype=actuator.dtype,
                device=actuator.device,
            )
            controlled = self.boundary_temperature_control >= 0
            if torch.any(controlled):
                selected[..., controlled] = actuator[
                    ..., self.boundary_temperature_control[controlled]
                ]
            boundary_temperature = (
                self.boundary_temperature_intercept + self.boundary_temperature_slope * selected
            )
            heat = heat + (boundary_temperature[..., :, None] * boundary_h).sum(dim=-2)
        return heat / self.capacity

    def _joint_affine_system(
        self,
        command: torch.Tensor,
        active_sources: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``dx/dt = F x + c`` for ``x = [temperature, actuator]``.

        A threshold source is affine while its active state is fixed. Boundary
        temperatures are already affine in actuator values, so the complete
        thermal and first-order actuator dynamics share one exact transition.
        """
        batch_shape = command.shape[:-1]
        n_state = self.n_nodes + self.n_controls
        matrix = torch.zeros(
            (*batch_shape, n_state, n_state),
            dtype=command.dtype,
            device=command.device,
        )
        affine = torch.zeros(
            (*batch_shape, n_state),
            dtype=command.dtype,
            device=command.device,
        )
        matrix[..., : self.n_nodes, : self.n_nodes] = self.system_matrix(command)

        actuator_coefficient = torch.zeros(
            (*batch_shape, self.n_nodes, self.n_controls),
            dtype=command.dtype,
            device=command.device,
        )
        thermal_offset = torch.zeros(
            (*batch_shape, self.n_nodes),
            dtype=command.dtype,
            device=command.device,
        )

        if len(self.spec.sources):
            positive = self.source_laws.positive_part_mask
            if torch.any(positive):
                source_rate = (
                    self.source_laws.scale()[positive, None]
                    * self.source_weights[positive]
                    / self.capacity[None, :]
                )
                active_rate = (
                    active_sources[..., positive, None].to(dtype=command.dtype) * source_rate
                )
                source_controls = functional.one_hot(
                    self.source_laws.control_index[positive],
                    num_classes=self.n_controls,
                ).to(dtype=command.dtype)
                actuator_coefficient = actuator_coefficient + torch.einsum(
                    "...sn,sc->...nc", active_rate, source_controls
                )
                thermal_offset = thermal_offset - (
                    active_rate * self.source_laws.threshold[positive, None]
                ).sum(dim=-2)

            direct = ~positive
            if torch.any(direct):
                direct_heat = (
                    self.source_heat_rate(command)[..., direct] @ self.source_weights[direct]
                )
                thermal_offset = thermal_offset + direct_heat / self.capacity

        if len(self.spec.boundaries):
            boundary_rate = (
                self.boundary_conductance(command)[..., :, None]
                * self.boundary_weights
                / self.capacity[None, :]
            )
            thermal_offset = thermal_offset + (
                self.boundary_temperature_intercept[:, None] * boundary_rate
            ).sum(dim=-2)
            controlled = self.boundary_temperature_control >= 0
            if torch.any(controlled):
                boundary_controls = functional.one_hot(
                    self.boundary_temperature_control[controlled],
                    num_classes=self.n_controls,
                ).to(dtype=command.dtype)
                controlled_rate = (
                    self.boundary_temperature_slope[controlled, None]
                    * boundary_rate[..., controlled, :]
                )
                actuator_coefficient = actuator_coefficient + torch.einsum(
                    "...bn,bc->...nc", controlled_rate, boundary_controls
                )

        matrix[..., : self.n_nodes, self.n_nodes :] = actuator_coefficient
        affine[..., : self.n_nodes] = thermal_offset

        tau = self.actuator_tau().to(dtype=command.dtype, device=command.device)
        inverse_tau = torch.where(tau > 0.0, torch.reciprocal(tau), torch.zeros_like(tau))
        matrix[..., self.n_nodes :, self.n_nodes :] = torch.diag(-inverse_tau)
        affine[..., self.n_nodes :] = command * inverse_tau
        return matrix, affine

    def _interval_start_actuator(
        self, actuator: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        tau = self.actuator_tau().to(dtype=actuator.dtype, device=actuator.device)
        return torch.where(tau > 0.0, actuator, command)

    def _source_activity(self, actuator: torch.Tensor) -> torch.Tensor:
        if not len(self.spec.sources):
            return torch.empty(
                (*actuator.shape[:-1], 0),
                dtype=torch.bool,
                device=actuator.device,
            )
        activity = torch.zeros(
            (*actuator.shape[:-1], len(self.spec.sources)),
            dtype=torch.bool,
            device=actuator.device,
        )
        positive = self.source_laws.positive_part_mask
        if torch.any(positive):
            activity[..., positive] = (
                actuator[..., self.source_laws.control_index[positive]]
                > self.source_laws.threshold[positive]
            )
        return activity

    def _operator_context(self, actuator: torch.Tensor) -> tuple[float, ...]:
        """Return the instantaneous inputs that change the thermal state matrix."""
        controls = [
            laws.control_index[laws.dependent_mask]
            for laws in (self.edge_laws, self.boundary_laws)
            if torch.any(laws.dependent_mask)
        ]
        if not controls:
            return ()
        return tuple(
            float(actuator[int(index)].detach().cpu().item())
            for index in torch.unique(torch.cat(controls)).tolist()
        )

    def _threshold_crossing_times(
        self,
        actuator: torch.Tensor,
        command: torch.Tensor,
        dt: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return exact crossing times for monotone first-order actuators."""
        if not len(self.spec.sources):
            return []
        tau = self.actuator_tau().to(dtype=actuator.dtype, device=actuator.device)
        step_value = float(dt.detach().cpu().item())
        crossings: list[torch.Tensor] = []
        for source_index in range(len(self.spec.sources)):
            if not bool(self.source_laws.positive_part_mask[source_index].item()):
                continue
            control_index = int(self.source_laws.control_index[source_index].item())
            if float(tau[control_index].detach().cpu().item()) <= 0.0:
                continue
            threshold = self.source_laws.threshold[source_index]
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

    def _exact_joint_segment(
        self,
        temperature: torch.Tensor,
        actuator: torch.Tensor,
        command: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        midpoint = self.actuator_step(actuator, command, dt * 0.5)
        active_sources = self._source_activity(midpoint)
        matrix, affine = self._joint_affine_system(command, active_sources)
        joint = torch.cat([temperature, actuator], dim=-1)
        result = exact_affine_step(joint, matrix, affine, dt)
        return result[..., : self.n_nodes], result[..., self.n_nodes :]

    def _exact_joint_step(
        self,
        temperature: torch.Tensor,
        actuator: torch.Tensor,
        command: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actuator = self._interval_start_actuator(actuator, command)
        crossings = self._threshold_crossing_times(actuator, command, dt)
        start = torch.zeros((), dtype=dt.dtype, device=dt.device)
        for end in [*crossings, dt]:
            temperature, actuator = self._exact_joint_segment(
                temperature,
                actuator,
                command,
                end - start,
            )
            start = end
        return temperature, actuator

    def _exact_joint_batch_step(
        self,
        temperature: torch.Tensor,
        actuator: torch.Tensor,
        command: torch.Tensor,
        dt: torch.Tensor,
        cache: dict[
            tuple[float, tuple[bool, ...], tuple[float, ...]],
            tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actuator = self._interval_start_actuator(actuator, command)
        outputs: dict[int, torch.Tensor] = {}
        groups: dict[tuple[float, tuple[bool, ...], tuple[float, ...]], list[int]] = {}

        for batch_index in range(len(temperature)):
            crossings = self._threshold_crossing_times(
                actuator[batch_index], command[batch_index], dt[batch_index]
            )
            if crossings:
                next_temperature, next_actuator = self._exact_joint_step(
                    temperature[batch_index],
                    actuator[batch_index],
                    command[batch_index],
                    dt[batch_index],
                )
                outputs[batch_index] = torch.cat([next_temperature, next_actuator])
                continue
            midpoint = self.actuator_step(
                actuator[batch_index], command[batch_index], dt[batch_index] * 0.5
            )
            activity = self._source_activity(midpoint)
            pattern = tuple(bool(value) for value in activity.detach().cpu().tolist())
            operator_context = self._operator_context(actuator[batch_index])
            key = (float(dt[batch_index].detach().cpu().item()), pattern, operator_context)
            groups.setdefault(key, []).append(batch_index)

        for key, indices in groups.items():
            group_command = command[indices]
            group_actuator = actuator[indices]
            midpoint = self.actuator_step(group_actuator, group_command, dt[indices] * 0.5)
            activity = self._source_activity(midpoint)
            matrix, affine = self._joint_affine_system(group_command, activity)
            if key not in cache:
                cache[key] = exact_affine_operators(matrix[0], dt[indices[0]])
            phi, gamma = cache[key]
            joint = torch.cat([temperature[indices], group_actuator], dim=-1)
            result = joint @ phi.T + affine @ gamma.T
            for group_index, batch_index in enumerate(indices):
                outputs[batch_index] = result[group_index]

        result = torch.stack([outputs[index] for index in range(len(temperature))])
        return result[:, : self.n_nodes], result[:, self.n_nodes :]

    def temperature_transition_matrix(
        self,
        dt: float | torch.Tensor,
        actuator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Linear sensitivity of next temperature to current temperature."""
        system_matrix = self.system_matrix(actuator)
        step = torch.as_tensor(dt, dtype=self.capacity.dtype, device=self.capacity.device)
        if step.ndim != 0:
            raise ValueError("temperature transition dt must be scalar")
        if self.integrator == "exact":
            return torch.matrix_exp(system_matrix * step)
        identity = torch.eye(self.n_nodes, dtype=self.capacity.dtype, device=self.capacity.device)
        return torch.linalg.solve(identity - step * system_matrix, identity)

    def step(
        self,
        state: ThermalState,
        command: torch.Tensor,
        dt: float | torch.Tensor,
    ) -> ThermalState:
        """Advance actuator and thermal states through one command interval."""
        command = command.to(dtype=state.temperature.dtype, device=state.temperature.device)
        step = torch.as_tensor(
            dt,
            dtype=state.temperature.dtype,
            device=state.temperature.device,
        )
        if self.integrator == "exact":
            temperature, actuator = self._exact_joint_step(
                state.temperature,
                state.actuator,
                command,
                step,
            )
            return ThermalState(temperature, actuator)

        midpoint = self.actuator_step(state.actuator, command, step * 0.5)
        next_actuator = self.actuator_step(state.actuator, command, step)
        system_matrix = self.system_matrix(midpoint)
        forcing = self.forcing(midpoint)
        next_temperature = implicit_euler_step(state.temperature, system_matrix, forcing, step)
        return ThermalState(next_temperature, next_actuator)

    def forward_trajectory(
        self,
        initial_temperature: torch.Tensor,
        commands: torch.Tensor,
        dt: torch.Tensor,
        initial_actuator: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate one trajectory through the shared batched rollout path."""
        actuator = None if initial_actuator is None else initial_actuator.unsqueeze(0)
        temperatures, actuators = self.forward_batch(
            initial_temperature.unsqueeze(0),
            commands.unsqueeze(0),
            dt.unsqueeze(0),
            actuator,
        )
        return temperatures[0], actuators[0]

    def forward_batch(
        self,
        initial_temperature: torch.Tensor,
        commands: torch.Tensor,
        dt: torch.Tensor,
        initial_actuator: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate equal-length trajectories as one differentiable batch.

        ``dt`` may be shared as ``[steps]`` or supplied per item as
        ``[batch, steps]``. Exact joint operators are reused by timestep and
        threshold activity pattern.
        """
        initial_temperature, commands, dt, actuator = self._prepare_batch_inputs(
            initial_temperature, commands, dt, initial_actuator
        )
        temperature = initial_temperature
        temperatures = [temperature]
        actuators = [actuator]
        operator_cache: dict[
            tuple[float, tuple[bool, ...], tuple[float, ...]],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

        for index in range(commands.shape[1]):
            interval_dt = dt[:, index]
            command = commands[:, index]
            if self.integrator == "exact":
                temperature, actuator = self._exact_joint_batch_step(
                    temperature,
                    actuator,
                    command,
                    interval_dt,
                    operator_cache,
                )
            else:
                midpoint = self.actuator_step(actuator, command, interval_dt * 0.5)
                actuator = self.actuator_step(actuator, command, interval_dt)
                system_matrix = self.system_matrix(midpoint)
                forcing = self.forcing(midpoint)
                temperature = implicit_euler_step(
                    temperature,
                    system_matrix,
                    forcing,
                    interval_dt,
                )
            temperatures.append(temperature)
            actuators.append(actuator)
        return torch.stack(temperatures, dim=1), torch.stack(actuators, dim=1)

    def _prepare_batch_inputs(
        self,
        initial_temperature: torch.Tensor,
        commands: torch.Tensor,
        dt: torch.Tensor,
        initial_actuator: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        initial_temperature = initial_temperature.to(
            dtype=self.capacity.dtype, device=self.capacity.device
        )
        commands = commands.to(dtype=self.capacity.dtype, device=self.capacity.device)
        dt = dt.to(dtype=self.capacity.dtype, device=self.capacity.device)
        if initial_actuator is not None:
            initial_actuator = initial_actuator.to(
                dtype=self.capacity.dtype, device=self.capacity.device
            )
        if initial_temperature.ndim != 2 or initial_temperature.shape[1] != self.n_nodes:
            raise ValueError("initial_temperature must have shape [batch, n_nodes]")
        if commands.ndim != 3 or commands.shape[2] != self.n_controls:
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
        if actuator.shape != (commands.shape[0], self.n_controls):
            raise ValueError("initial_actuator must have shape [batch, n_controls]")
        return initial_temperature, commands, dt, actuator

    def observe(self, temperature: torch.Tensor) -> torch.Tensor:
        return temperature @ self.observation.T

    def initialize_temperature(
        self, observation: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Infer node temperatures from any observed sensor subset.

        The small ridge term supplies the mean observed temperature to unobserved
        nodes while leaving directly observed nodes effectively unchanged.
        """
        if observation.shape != (self.n_sensors,):
            raise ValueError("observation must have one value per sensor")
        if mask is None:
            mask = torch.isfinite(observation)
        mask = mask.to(dtype=torch.bool, device=observation.device)
        if not torch.any(mask):
            raise ValueError("at least one sensor is required to initialize temperature")
        observed = observation[mask]
        matrix = self.observation[mask]
        prior = torch.full(
            (self.n_nodes,),
            float(observed.mean()),
            dtype=observation.dtype,
            device=observation.device,
        )
        ridge = torch.as_tensor(1e-8, dtype=observation.dtype, device=observation.device)
        lhs = matrix.T @ matrix + ridge * torch.eye(
            self.n_nodes, dtype=observation.dtype, device=observation.device
        )
        rhs = matrix.T @ observed + ridge * prior
        return torch.linalg.solve(lhs, rhs)

    def stored_energy(self, temperature: torch.Tensor) -> torch.Tensor:
        """Capacity-weighted energy, relative to the chosen temperature origin."""
        return (temperature * self.capacity).sum(dim=-1)
