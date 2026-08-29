"""A compact, physical-unit thermal RC state-space model."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from celltemp.domain.topology import ThermalSystemSpec

from .integrator import exact_affine_operators, exact_affine_step, implicit_euler_step
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
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_prior: torch.Tensor
    edge_learn_mask: torch.Tensor
    tau_prior: torch.Tensor
    tau_learn_mask: torch.Tensor
    source_control: torch.Tensor
    source_weights: torch.Tensor
    source_threshold: torch.Tensor
    source_prior: torch.Tensor
    source_learn_mask: torch.Tensor
    boundary_control: torch.Tensor
    boundary_weights: torch.Tensor
    boundary_intercept: torch.Tensor
    boundary_slope: torch.Tensor
    boundary_prior: torch.Tensor
    boundary_learn_mask: torch.Tensor
    observation: torch.Tensor

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
        self.spec = spec
        self.integrator = integrator

        node_index = {name: i for i, name in enumerate(spec.node_names)}
        control_index = {name: i for i, name in enumerate(spec.control_names)}

        self.register_buffer("capacity", _tensor(spec.heat_capacity, dtype=dtype))
        self.register_buffer(
            "edge_src",
            torch.tensor([node_index[e.node_a] for e in spec.edges], dtype=torch.long),
        )
        self.register_buffer(
            "edge_dst",
            torch.tensor([node_index[e.node_b] for e in spec.edges], dtype=torch.long),
        )
        self.register_buffer(
            "edge_prior", _tensor([e.conductance for e in spec.edges], dtype=dtype)
        )
        self.register_buffer(
            "edge_learn_mask", _tensor([e.learnable for e in spec.edges], dtype=dtype)
        )
        self.log_edge_multiplier = nn.Parameter(torch.zeros(len(spec.edges), dtype=dtype))

        self.register_buffer("tau_prior", _tensor([a.tau for a in spec.actuators], dtype=dtype))
        self.register_buffer(
            "tau_learn_mask",
            _tensor([a.learnable and a.tau > 0.0 for a in spec.actuators], dtype=dtype),
        )
        self.log_tau_multiplier = nn.Parameter(torch.zeros(len(spec.actuators), dtype=dtype))

        self.register_buffer(
            "source_control",
            torch.tensor([control_index[s.actuator] for s in spec.sources], dtype=torch.long),
        )
        self.register_buffer(
            "source_weights",
            _tensor([s.node_weights for s in spec.sources], dtype=dtype).reshape(
                len(spec.sources), len(spec.node_names)
            ),
        )
        self.register_buffer(
            "source_threshold", _tensor([s.threshold for s in spec.sources], dtype=dtype)
        )
        self.register_buffer("source_prior", _tensor([s.gain for s in spec.sources], dtype=dtype))
        self.register_buffer(
            "source_learn_mask", _tensor([s.learnable for s in spec.sources], dtype=dtype)
        )
        self.log_source_multiplier = nn.Parameter(torch.zeros(len(spec.sources), dtype=dtype))

        boundary_controls = [
            -1 if b.actuator is None else control_index[b.actuator] for b in spec.boundaries
        ]
        self.register_buffer("boundary_control", torch.tensor(boundary_controls, dtype=torch.long))
        self.register_buffer(
            "boundary_weights",
            _tensor([b.node_weights for b in spec.boundaries], dtype=dtype).reshape(
                len(spec.boundaries), len(spec.node_names)
            ),
        )
        self.register_buffer(
            "boundary_intercept",
            _tensor([b.temperature_intercept for b in spec.boundaries], dtype=dtype),
        )
        self.register_buffer(
            "boundary_slope", _tensor([b.temperature_slope for b in spec.boundaries], dtype=dtype)
        )
        self.register_buffer(
            "boundary_prior", _tensor([b.conductance for b in spec.boundaries], dtype=dtype)
        )
        self.register_buffer(
            "boundary_learn_mask", _tensor([b.learnable for b in spec.boundaries], dtype=dtype)
        )
        self.log_boundary_multiplier = nn.Parameter(torch.zeros(len(spec.boundaries), dtype=dtype))
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
        return prior * torch.exp(raw * mask)

    def conductance(self) -> torch.Tensor:
        return self._positive(self.edge_prior, self.log_edge_multiplier, self.edge_learn_mask)

    def actuator_tau(self) -> torch.Tensor:
        return self._positive(self.tau_prior, self.log_tau_multiplier, self.tau_learn_mask)

    def source_gain(self) -> torch.Tensor:
        return self._positive(self.source_prior, self.log_source_multiplier, self.source_learn_mask)

    def boundary_conductance(self) -> torch.Tensor:
        return self._positive(
            self.boundary_prior, self.log_boundary_multiplier, self.boundary_learn_mask
        )

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

    def _laplacian(self) -> torch.Tensor:
        matrix = torch.zeros(
            (self.n_nodes, self.n_nodes), dtype=self.capacity.dtype, device=self.capacity.device
        )
        if len(self.spec.edges):
            conductance = self.conductance()
            matrix.index_put_((self.edge_src, self.edge_src), conductance, accumulate=True)
            matrix.index_put_((self.edge_dst, self.edge_dst), conductance, accumulate=True)
            matrix.index_put_((self.edge_src, self.edge_dst), -conductance, accumulate=True)
            matrix.index_put_((self.edge_dst, self.edge_src), -conductance, accumulate=True)
        return matrix

    def system_matrix(self) -> torch.Tensor:
        """Return the actuator-independent ``A`` in ``dT/dt = A T + b``."""
        laplacian = self._laplacian()
        boundary_total = torch.zeros(
            self.n_nodes, dtype=self.capacity.dtype, device=self.capacity.device
        )
        if len(self.spec.boundaries):
            boundary_h = self.boundary_conductance()[:, None] * self.boundary_weights
            boundary_total = boundary_h.sum(dim=0)
        return -(laplacian + torch.diag(boundary_total)) / self.capacity[:, None]

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
            selected = actuator[..., self.source_control]
            drive = torch.relu(selected - self.source_threshold)
            source = drive * self.source_gain()
            heat = heat + source @ self.source_weights

        if len(self.spec.boundaries):
            boundary_h = self.boundary_conductance()[:, None] * self.boundary_weights
            selected = torch.zeros(
                (*actuator.shape[:-1], len(self.spec.boundaries)),
                dtype=actuator.dtype,
                device=actuator.device,
            )
            controlled = self.boundary_control >= 0
            if torch.any(controlled):
                selected[..., controlled] = actuator[..., self.boundary_control[controlled]]
            boundary_temperature = self.boundary_intercept + self.boundary_slope * selected
            heat = heat + boundary_temperature @ boundary_h
        return heat / self.capacity

    def temperature_transition_matrix(self, dt: float | torch.Tensor) -> torch.Tensor:
        """Linear sensitivity of next temperature to current temperature."""
        system_matrix = self.system_matrix()
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
        midpoint = self.actuator_step(state.actuator, command, torch.as_tensor(dt) * 0.5)
        next_actuator = self.actuator_step(state.actuator, command, dt)
        system_matrix = self.system_matrix()
        forcing = self.forcing(midpoint)
        integrator = exact_affine_step if self.integrator == "exact" else implicit_euler_step
        next_temperature = integrator(state.temperature, system_matrix, forcing, dt)
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
        ``[batch, steps]``.  Exact transition operators are reused whenever all
        items in an interval share a timestep.
        """
        initial_temperature, commands, dt, actuator = self._prepare_batch_inputs(
            initial_temperature, commands, dt, initial_actuator
        )
        temperature = initial_temperature
        temperatures = [temperature]
        actuators = [actuator]
        system_matrix = self.system_matrix()
        operator_cache: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}

        for index in range(commands.shape[1]):
            interval_dt = dt[:, index]
            command = commands[:, index]
            midpoint = self.actuator_step(actuator, command, interval_dt * 0.5)
            actuator = self.actuator_step(actuator, command, interval_dt)
            forcing = self.forcing(midpoint)
            temperature = self._batch_temperature_step(
                temperature,
                system_matrix,
                forcing,
                interval_dt,
                operator_cache,
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

    def _batch_temperature_step(
        self,
        temperature: torch.Tensor,
        system_matrix: torch.Tensor,
        forcing: torch.Tensor,
        dt: torch.Tensor,
        cache: dict[float, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if self.integrator != "exact":
            return implicit_euler_step(temperature, system_matrix, forcing, dt)
        if not torch.all(dt == dt[0]):
            return exact_affine_step(temperature, system_matrix, forcing, dt)
        step_value = float(dt[0].detach().cpu())
        if step_value not in cache:
            cache[step_value] = exact_affine_operators(system_matrix, dt[0])
        phi, gamma = cache[step_value]
        return temperature @ phi.T + forcing @ gamma.T

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
