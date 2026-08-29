"""Causal state and sensor-bias estimation for measured temperature logs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rc import ThermalRCModel
from .state import ThermalState


@dataclass(frozen=True)
class ObserverState:
    temperature: torch.Tensor
    actuator: torch.Tensor
    sensor_bias: torch.Tensor
    covariance: torch.Tensor


class KalmanObserver:
    """A linear time-varying Kalman observer around the RC transition.

    Thermal dynamics are affine in temperature for a fixed actuator state, so the
    exact transition matrix is available without numerical differentiation.  Slow
    sensor bias is an explicit random-walk state rather than an EMA added after the
    prediction.
    """

    def __init__(
        self,
        model: ThermalRCModel,
        *,
        temperature_process_std: float = 0.02,
        bias_process_std: float = 0.005,
        sensor_std: float = 0.15,
        initial_temperature_std: float = 1.0,
        initial_bias_std: float = 0.5,
    ) -> None:
        for name, value in (
            ("temperature_process_std", temperature_process_std),
            ("bias_process_std", bias_process_std),
            ("sensor_std", sensor_std),
            ("initial_temperature_std", initial_temperature_std),
            ("initial_bias_std", initial_bias_std),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.model = model
        self.temperature_process_std = float(temperature_process_std)
        self.bias_process_std = float(bias_process_std)
        self.sensor_std = float(sensor_std)
        self.initial_temperature_std = float(initial_temperature_std)
        self.initial_bias_std = float(initial_bias_std)

    def _measurement_matrix(self, state: ObserverState) -> torch.Tensor:
        return torch.cat(
            [
                self.model.observation.to(
                    dtype=state.temperature.dtype, device=state.temperature.device
                ),
                torch.eye(
                    self.model.n_sensors,
                    dtype=state.temperature.dtype,
                    device=state.temperature.device,
                ),
            ],
            dim=1,
        )

    def prediction_std(self, state: ObserverState) -> torch.Tensor:
        """Return the standard deviation of each predicted sensor measurement."""
        matrix = self._measurement_matrix(state)
        covariance = matrix @ state.covariance @ matrix.T
        covariance = (
            covariance
            + torch.eye(
                self.model.n_sensors,
                dtype=state.temperature.dtype,
                device=state.temperature.device,
            )
            * self.sensor_std**2
        )
        return torch.sqrt(torch.clamp(torch.diag(covariance), min=0.0))

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
                self.model.n_controls, dtype=temperature.dtype, device=temperature.device
            )
        bias = torch.zeros(self.model.n_sensors, dtype=temperature.dtype, device=temperature.device)
        variance = torch.cat(
            [
                torch.full_like(temperature, self.initial_temperature_std**2),
                torch.full_like(bias, self.initial_bias_std**2),
            ]
        )
        return ObserverState(temperature, actuator, bias, torch.diag(variance))

    def predict(
        self, state: ObserverState, command: torch.Tensor, dt: float | torch.Tensor
    ) -> ObserverState:
        thermal = self.model.step(ThermalState(state.temperature, state.actuator), command, dt)
        phi = self.model.temperature_transition_matrix(dt).to(
            dtype=state.temperature.dtype, device=state.temperature.device
        )
        n_node, n_sensor = self.model.n_nodes, self.model.n_sensors
        transition = torch.zeros(
            (n_node + n_sensor, n_node + n_sensor),
            dtype=state.temperature.dtype,
            device=state.temperature.device,
        )
        transition[:n_node, :n_node] = phi
        transition[n_node:, n_node:] = torch.eye(
            n_sensor, dtype=state.temperature.dtype, device=state.temperature.device
        )
        step = float(torch.as_tensor(dt).detach().cpu().item())
        process_variance = torch.cat(
            [
                torch.full_like(
                    state.temperature, self.temperature_process_std**2 * max(step, 0.0)
                ),
                torch.full_like(state.sensor_bias, self.bias_process_std**2 * max(step, 0.0)),
            ]
        )
        covariance = transition @ state.covariance @ transition.T + torch.diag(process_variance)
        return ObserverState(thermal.temperature, thermal.actuator, state.sensor_bias, covariance)

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
            return predicted, torch.empty(0, dtype=observation.dtype, device=observation.device)

        n_node, n_sensor = self.model.n_nodes, self.model.n_sensors
        observation_matrix = self._measurement_matrix(predicted)[mask]
        combined = torch.cat([predicted.temperature, predicted.sensor_bias])
        innovation = observation[mask] - observation_matrix @ combined
        noise = (
            torch.eye(int(mask.sum().item()), dtype=observation.dtype, device=observation.device)
            * self.sensor_std**2
        )
        innovation_covariance = (
            observation_matrix @ predicted.covariance @ observation_matrix.T + noise
        )
        cross = predicted.covariance @ observation_matrix.T
        gain = torch.linalg.solve(innovation_covariance, cross.T).T
        updated = combined + gain @ innovation

        identity = torch.eye(n_node + n_sensor, dtype=observation.dtype, device=observation.device)
        residual_map = identity - gain @ observation_matrix
        covariance = residual_map @ predicted.covariance @ residual_map.T + gain @ noise @ gain.T
        return (
            ObserverState(
                updated[:n_node],
                predicted.actuator,
                updated[n_node:],
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
