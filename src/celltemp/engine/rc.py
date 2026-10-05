"""Public thermal RC model and its learnable physical parameters."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from celltemp.domain.topology import ThermalSystemSpec

from .energy import HeatFlowBreakdown
from .energy import heat_flow_breakdown as _heat_flow_breakdown
from .laws import ScalarLawSet
from .operators import boundary_temperature as _boundary_temperature
from .operators import edge_laplacian_basis, validate_exact_support
from .operators import forcing as _forcing
from .operators import system_matrix as _system_matrix
from .rollout import actuator_step as _actuator_step
from .rollout import forward_batch as _forward_batch
from .rollout import forward_trajectory as _forward_trajectory
from .rollout import point_actuator as _point_actuator
from .rollout import step as _step
from .rollout import temperature_transition_matrix as _temperature_transition_matrix
from .state import ThermalState
from .validation import require_finite


def _tensor(values: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=dtype)


class ThermalRCModel(nn.Module):
    """Symmetric RC dynamics with explicit actuator states.

    The model owns physical coefficients and its stable public API. Operator
    assembly and time integration live in focused engine modules so those
    responsibilities can evolve independently without changing saved artifacts.
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
            validate_exact_support(spec)
        self.spec = spec
        self.integrator = integrator

        control_index = {name: index for index, name in enumerate(spec.control_names)}

        self.register_buffer("capacity", _tensor(spec.heat_capacity, dtype=dtype))
        self.register_buffer("edge_basis", edge_laplacian_basis(spec, dtype=dtype))
        self.edge_laws = ScalarLawSet(
            [edge.conductance for edge in spec.edges],
            spec.control_names,
            dtype=dtype,
        )

        self.register_buffer("tau_prior", _tensor([a.tau for a in spec.actuators], dtype=dtype))
        self.register_buffer(
            "tau_learn_mask",
            _tensor([bool(a.learnable) for a in spec.actuators], dtype=dtype),
        )
        self.log_tau_multiplier = nn.Parameter(torch.zeros(len(spec.actuators), dtype=dtype))

        self.register_buffer(
            "source_weights",
            _tensor([source.node_weights for source in spec.sources], dtype=dtype).reshape(
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
            _tensor([boundary.node_weights for boundary in spec.boundaries], dtype=dtype).reshape(
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
        if not any(actuator.learnable for actuator in self.spec.actuators):
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

    def learnable_log_parameter_values(self) -> tuple[torch.Tensor, ...]:
        """Return only log multipliers enabled by the system definition."""
        return (
            self.log_tau_multiplier[self.tau_learn_mask.to(dtype=torch.bool)],
            *self.edge_laws.learnable_log_parameter_values(),
            *self.source_laws.learnable_log_parameter_values(),
            *self.boundary_laws.learnable_log_parameter_values(),
        )

    def boundary_conductance(self, actuator: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate every boundary conductance for the effective physical inputs."""
        return self.boundary_laws(actuator)

    def boundary_temperature(self, actuator: torch.Tensor) -> torch.Tensor:
        """Evaluate every reservoir temperature for the effective inputs."""
        return _boundary_temperature(self, actuator)

    def actuator_step(
        self,
        actuator: torch.Tensor,
        command: torch.Tensor,
        dt: float | torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the exact first-order actuator response over one interval."""
        return _actuator_step(self, actuator, command, dt)

    def point_actuator(self, actuator: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Apply zero-tau commands at a timestamp without advancing lagged inputs.

        Point outputs use the command starting at that timestamp. At the final
        point, where no new interval exists, callers hold the last command.
        """
        return _point_actuator(self, actuator, command)

    def system_matrix(self, actuator: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``A`` in ``dT/dt = A T + b`` for effective controls."""
        if actuator is not None:
            if actuator.shape[-1] != self.n_controls:
                raise ValueError("actuator has the wrong number of controls")
            require_finite(actuator, "actuator")
        elif self.edge_laws._has_dependent or self.boundary_laws._has_dependent:
            raise ValueError("actuator is required for an input-dependent scalar law")
        matrix = _system_matrix(self, actuator)
        require_finite(matrix, "system_matrix", computed=True)
        return matrix

    def forcing(self, actuator: torch.Tensor) -> torch.Tensor:
        """Return the actuator-dependent ``b`` in ``dT/dt = A T + b``."""
        require_finite(actuator, "actuator")
        value = _forcing(self, actuator)
        require_finite(value, "forcing", computed=True)
        return value

    def heat_flow_breakdown(
        self,
        temperature: torch.Tensor,
        actuator: torch.Tensor,
    ) -> HeatFlowBreakdown:
        """Return signed edge, source, boundary, storage, and residual heat rates."""
        return _heat_flow_breakdown(self, temperature, actuator)

    def temperature_transition_matrix(
        self,
        dt: float | torch.Tensor,
        actuator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the linear sensitivity of next temperature to current temperature."""
        return _temperature_transition_matrix(self, dt, actuator)

    def step(
        self,
        state: ThermalState,
        command: torch.Tensor,
        dt: float | torch.Tensor,
    ) -> ThermalState:
        """Advance actuator and thermal states through one command interval."""
        return _step(self, state, command, dt)

    def forward_trajectory(
        self,
        initial_temperature: torch.Tensor,
        commands: torch.Tensor,
        dt: torch.Tensor,
        initial_actuator: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate one trajectory through the shared batched rollout path."""
        return _forward_trajectory(self, initial_temperature, commands, dt, initial_actuator)

    def forward_batch(
        self,
        initial_temperature: torch.Tensor,
        commands: torch.Tensor,
        dt: torch.Tensor,
        initial_actuator: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate equal-length trajectories as one differentiable batch."""
        return _forward_batch(self, initial_temperature, commands, dt, initial_actuator)

    def observe(self, temperature: torch.Tensor) -> torch.Tensor:
        return temperature @ self.observation.T

    def initialize_temperature(
        self,
        observation: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Infer node temperatures from any observed sensor subset.

        A centered, augmented QR solve supplies the mean observed temperature to
        unobserved nodes. It avoids squaring the condition number, and retains the
        small ridge even in float32 when sensors observe weighted averages.
        """
        observation = observation.to(dtype=self.capacity.dtype, device=self.capacity.device)
        if observation.shape != (self.n_sensors,):
            raise ValueError("observation must have one value per sensor")
        if mask is None:
            mask = torch.isfinite(observation)
        mask = mask.to(dtype=torch.bool, device=observation.device)
        if mask.shape != observation.shape:
            raise ValueError("observation mask must have one value per sensor")
        if not torch.any(mask):
            raise ValueError("at least one sensor is required to initialize temperature")
        observed = observation[mask]
        require_finite(observed, "observed temperatures")
        # The solve is tiny, and extra working precision keeps noisy duplicate
        # sensors from injecting rounding error into the weakly constrained modes.
        solve_dtype = torch.float64 if observed.dtype == torch.float32 else observed.dtype
        observed = observed.to(dtype=solve_dtype)
        matrix = self.observation[mask].to(dtype=solve_dtype)
        prior = observed.mean().expand(self.n_nodes)
        regularizer = 1e-4 * torch.eye(
            self.n_nodes,
            dtype=solve_dtype,
            device=observation.device,
        )
        augmented = torch.cat([matrix, regularizer], dim=0)
        residual = torch.cat([observed - matrix @ prior, torch.zeros_like(prior)])
        orthogonal, triangular = torch.linalg.qr(augmented, mode="reduced")
        correction = torch.linalg.solve_triangular(
            triangular, (orthogonal.T @ residual).unsqueeze(-1), upper=True
        ).squeeze(-1)
        temperature = (prior + correction).to(dtype=self.capacity.dtype)
        require_finite(temperature, "initial temperature", computed=True)
        return temperature

    def stored_energy(self, temperature: torch.Tensor) -> torch.Tensor:
        """Return capacity-weighted energy relative to the temperature origin."""
        return (temperature * self.capacity).sum(dim=-1)
