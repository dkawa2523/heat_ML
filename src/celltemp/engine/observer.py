"""Causal estimation of physical temperature, unknown heat, and sensor bias."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rc import ThermalRCModel
from .state import ThermalState


@dataclass(frozen=True)
class ObserverState:
    """Physical observer state.

    ``heat_disturbance`` is a signed, unknown heat rate in watts along each known
    source path. It is deliberately separate from ``bias_state`` so a physical
    load must propagate through heat capacity and the RC network before it can
    explain a measurement. ``bias_state`` contains only the identifiable sensor
    bias coordinates selected by the observer's gauge.
    """

    temperature: torch.Tensor
    actuator: torch.Tensor
    heat_disturbance: torch.Tensor
    bias_state: torch.Tensor
    covariance: torch.Tensor


class KalmanObserver:
    """Linear Kalman observer around the physical RC transition.

    The estimated state is ``[temperature, unknown source heat, sensor bias]``.
    Unknown heat and sensor bias are continuous-time random walks. Their exact
    discrete covariance is obtained from the coupled thermal dynamics, preserving
    the correlations and thermal filtering that a diagonal ``q**2 * dt`` update
    would lose. Bias uses a zero-mean gauge by default, or may be anchored to one
    explicitly calibrated reference sensor whose bias is fixed to zero.
    """

    def __init__(
        self,
        model: ThermalRCModel,
        *,
        disturbance_process_std: float = 0.02,
        bias_process_std: float = 0.005,
        sensor_std: float = 0.15,
        innovation_gate_sigma: float = 4.0,
        initial_temperature_std: float = 1.0,
        initial_disturbance_std: float = 0.5,
        initial_bias_std: float = 0.5,
        bias_reference: str | None = None,
    ) -> None:
        for name, value in (
            ("disturbance_process_std", disturbance_process_std),
            ("bias_process_std", bias_process_std),
            ("sensor_std", sensor_std),
            ("initial_temperature_std", initial_temperature_std),
            ("initial_disturbance_std", initial_disturbance_std),
            ("initial_bias_std", initial_bias_std),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if innovation_gate_sigma <= 0.0:
            raise ValueError("innovation_gate_sigma must be positive")
        self.model = model
        self.disturbance_process_std = float(disturbance_process_std)
        self.bias_process_std = float(bias_process_std)
        self.sensor_std = float(sensor_std)
        self.innovation_gate_sigma = float(innovation_gate_sigma)
        self.initial_temperature_std = float(initial_temperature_std)
        self.initial_disturbance_std = float(initial_disturbance_std)
        self.initial_bias_std = float(initial_bias_std)
        if bias_reference is not None and bias_reference not in model.spec.sensor_names:
            raise ValueError(f"unknown bias reference sensor: {bias_reference}")
        self.bias_reference = bias_reference
        self.disturbance_weights = (
            model.source_weights
            if len(model.spec.sources)
            else torch.eye(
                model.n_nodes,
                dtype=model.capacity.dtype,
                device=model.capacity.device,
            )
        )
        self.bias_basis = self._bias_basis()
        self._operator_cache: dict[
            tuple[float, tuple[float, ...]],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

    def _bias_basis(self) -> torch.Tensor:
        """Return identifiable sensor-bias coordinates for the selected gauge."""
        n_sensor = self.model.n_sensors
        if n_sensor <= 1:
            return torch.empty(
                (n_sensor, 0),
                dtype=self.model.capacity.dtype,
                device=self.model.capacity.device,
            )
        if self.bias_reference is not None:
            reference = self.model.spec.sensor_names.index(self.bias_reference)
            basis = torch.zeros(
                (n_sensor, n_sensor - 1),
                dtype=self.model.capacity.dtype,
                device=self.model.capacity.device,
            )
            estimated = [index for index in range(n_sensor) if index != reference]
            basis[estimated] = torch.eye(
                n_sensor - 1,
                dtype=self.model.capacity.dtype,
                device=self.model.capacity.device,
            )
            return basis
        contrasts = torch.zeros(
            (n_sensor, n_sensor - 1),
            dtype=self.model.capacity.dtype,
            device=self.model.capacity.device,
        )
        contrasts[:-1] = torch.eye(
            n_sensor - 1,
            dtype=self.model.capacity.dtype,
            device=self.model.capacity.device,
        )
        contrasts[-1] = -1.0
        basis, _ = torch.linalg.qr(contrasts, mode="reduced")
        return basis

    @property
    def state_size(self) -> int:
        return self.model.n_nodes + self.n_disturbances + self.n_bias_states

    @property
    def n_disturbances(self) -> int:
        return self.disturbance_weights.shape[0]

    @property
    def n_bias_states(self) -> int:
        return self.bias_basis.shape[1]

    def _measurement_matrix(self, state: ObserverState) -> torch.Tensor:
        dtype, device = state.temperature.dtype, state.temperature.device
        return torch.cat(
            [
                self.model.observation.to(dtype=dtype, device=device),
                torch.zeros(
                    (self.model.n_sensors, self.n_disturbances),
                    dtype=dtype,
                    device=device,
                ),
                self.bias_basis.to(dtype=dtype, device=device),
            ],
            dim=1,
        )

    def predicted_measurement(self, state: ObserverState) -> torch.Tensor:
        """Return sensor values implied by physical temperature and sensor bias."""
        return self.model.observe(state.temperature) + self.sensor_bias(state)

    def sensor_bias(self, state: ObserverState) -> torch.Tensor:
        """Return sensor offsets in the configured identifiable gauge."""
        return self.bias_basis @ state.bias_state

    @property
    def bias_gauge(self) -> str:
        """Describe the constraint that makes sensor bias identifiable."""
        return "zero_mean" if self.bias_reference is None else f"reference:{self.bias_reference}"

    def node_heat_disturbance(self, state: ObserverState) -> torch.Tensor:
        """Map estimated source-path heat rates to signed node heat in watts."""
        return state.heat_disturbance @ self.disturbance_weights

    def innovation_covariance(
        self, state: ObserverState, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the full covariance of the available measurement innovation."""
        if mask is None:
            mask = torch.ones(
                self.model.n_sensors,
                dtype=torch.bool,
                device=state.temperature.device,
            )
        mask = mask.to(dtype=torch.bool, device=state.temperature.device)
        matrix = self._measurement_matrix(state)[mask]
        noise = (
            torch.eye(
                int(mask.sum().item()),
                dtype=state.temperature.dtype,
                device=state.temperature.device,
            )
            * self.sensor_std**2
        )
        return matrix @ state.covariance @ matrix.T + noise

    def initialize(
        self,
        observation: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        actuator: torch.Tensor | None = None,
    ) -> ObserverState:
        temperature = self.model.initialize_temperature(observation, mask)
        if actuator is None:
            actuator = torch.zeros(
                self.model.n_controls,
                dtype=temperature.dtype,
                device=temperature.device,
            )
        disturbance = torch.zeros(
            self.n_disturbances,
            dtype=temperature.dtype,
            device=temperature.device,
        )
        bias = torch.zeros(
            self.n_bias_states,
            dtype=temperature.dtype,
            device=temperature.device,
        )
        variance = torch.cat(
            [
                torch.full_like(temperature, self.initial_temperature_std**2),
                torch.full_like(disturbance, self.initial_disturbance_std**2),
                torch.full_like(bias, self.initial_bias_std**2),
            ]
        )
        return ObserverState(
            temperature,
            actuator,
            disturbance,
            bias,
            torch.diag(variance),
        )

    def _continuous_state_matrix(self, actuator: torch.Tensor) -> torch.Tensor:
        """Return dynamics for ``[temperature, unknown source heat, sensor bias]``."""
        matrix = torch.zeros(
            (self.state_size, self.state_size),
            dtype=self.model.capacity.dtype,
            device=self.model.capacity.device,
        )
        n_node = self.model.n_nodes
        matrix[:n_node, :n_node] = self.model.system_matrix(actuator)
        matrix[:n_node, n_node : n_node + self.n_disturbances] = (
            self.disturbance_weights.T / self.model.capacity[:, None]
        )
        return matrix

    def _process_spectral_density(self) -> torch.Tensor:
        density = torch.zeros(
            self.state_size,
            dtype=self.model.capacity.dtype,
            device=self.model.capacity.device,
        )
        n_node = self.model.n_nodes
        disturbance_end = n_node + self.n_disturbances
        density[n_node:disturbance_end] = self.disturbance_process_std**2
        density[disturbance_end:] = self.bias_process_std**2
        return torch.diag(density)

    def _operators(
        self,
        dt: float | torch.Tensor,
        actuator: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step = torch.as_tensor(
            dt,
            dtype=self.model.capacity.dtype,
            device=self.model.capacity.device,
        )
        if step.ndim != 0:
            raise ValueError("observer dt must be scalar")
        step_value = float(step.detach().cpu().item())
        if step_value < 0.0:
            raise ValueError("observer dt must be non-negative")
        actuator_key = tuple(float(value) for value in actuator.detach().cpu().tolist())
        cache_key = (step_value, actuator_key)
        cached = self._operator_cache.get(cache_key)
        if cached is not None:
            return cached

        continuous = self._continuous_state_matrix(actuator)
        if self.model.integrator == "exact":
            transition = torch.matrix_exp(continuous * step)
        else:
            identity = torch.eye(
                self.state_size,
                dtype=continuous.dtype,
                device=continuous.device,
            )
            transition = torch.linalg.solve(identity - step * continuous, identity)

        # Van Loan discretization of integral exp(Fs) Q exp(F' s) ds.
        density = self._process_spectral_density()
        zero = torch.zeros_like(continuous)
        van_loan = torch.cat(
            [
                torch.cat([continuous, density], dim=1),
                torch.cat([zero, -continuous.T], dim=1),
            ],
            dim=0,
        )
        exponential = torch.matrix_exp(van_loan * step)
        continuous_transition = exponential[: self.state_size, : self.state_size]
        process_covariance = (
            exponential[: self.state_size, self.state_size :] @ continuous_transition.T
        )
        process_covariance = (process_covariance + process_covariance.T) * 0.5
        self._operator_cache[cache_key] = transition, process_covariance
        return transition, process_covariance

    def predict(
        self, state: ObserverState, command: torch.Tensor, dt: float | torch.Tensor
    ) -> ObserverState:
        interval_actuator = self.model.actuator_step(state.actuator, command, dt * 0.5)
        nominal = self.model.step(ThermalState(state.temperature, state.actuator), command, dt)
        transition, process_covariance = self._operators(dt, interval_actuator)
        n_node = self.model.n_nodes
        disturbance_end = n_node + self.n_disturbances
        disturbance_response = transition[:n_node, n_node:disturbance_end]
        temperature = nominal.temperature + disturbance_response @ state.heat_disturbance
        covariance = transition @ state.covariance @ transition.T + process_covariance
        covariance = (covariance + covariance.T) * 0.5
        return ObserverState(
            temperature,
            nominal.actuator,
            state.heat_disturbance,
            state.bias_state,
            covariance,
        )

    def _innovation_statistics(
        self,
        predicted: ObserverState,
        observation: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        matrix = self._measurement_matrix(predicted)[mask]
        combined = torch.cat(
            [
                predicted.temperature,
                predicted.heat_disturbance,
                predicted.bias_state,
            ]
        )
        innovation = observation[mask] - matrix @ combined
        covariance = self.innovation_covariance(predicted, mask)
        return matrix, innovation, covariance

    def update(
        self,
        predicted: ObserverState,
        observation: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[ObserverState, torch.Tensor]:
        if observation.shape != (self.model.n_sensors,):
            raise ValueError("observation must have one value per sensor")
        if mask is None:
            mask = torch.isfinite(observation)
        mask = mask.to(dtype=torch.bool, device=observation.device)
        if not torch.any(mask):
            return predicted, torch.empty(
                0,
                dtype=observation.dtype,
                device=observation.device,
            )

        observation_matrix, innovation, innovation_covariance = self._innovation_statistics(
            predicted, observation, mask
        )
        combined = torch.cat(
            [
                predicted.temperature,
                predicted.heat_disturbance,
                predicted.bias_state,
            ]
        )
        cross = predicted.covariance @ observation_matrix.T
        standardized = torch.abs(innovation) / torch.sqrt(
            torch.clamp(torch.diag(innovation_covariance), min=torch.finfo(observation.dtype).eps)
        )
        noise_scale = torch.clamp(standardized / self.innovation_gate_sigma, min=1.0)
        measurement_noise = torch.diag((self.sensor_std * noise_scale).square())
        robust_innovation_covariance = (
            observation_matrix @ predicted.covariance @ observation_matrix.T + measurement_noise
        )
        gain = torch.linalg.solve(robust_innovation_covariance, cross.T).T
        updated = combined + gain @ innovation

        identity = torch.eye(
            self.state_size,
            dtype=observation.dtype,
            device=observation.device,
        )
        residual_map = identity - gain @ observation_matrix
        covariance = (
            residual_map @ predicted.covariance @ residual_map.T + gain @ measurement_noise @ gain.T
        )
        covariance = (covariance + covariance.T) * 0.5
        n_node = self.model.n_nodes
        disturbance_end = n_node + self.n_disturbances
        return (
            ObserverState(
                updated[:n_node],
                predicted.actuator,
                updated[n_node:disturbance_end],
                updated[disturbance_end:],
                covariance,
            ),
            innovation,
        )

    def step(
        self,
        state: ObserverState,
        command: torch.Tensor,
        observation: torch.Tensor,
        dt: float | torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[ObserverState, torch.Tensor]:
        predicted = self.predict(state, command, dt)
        return self.update(predicted, observation, mask)
