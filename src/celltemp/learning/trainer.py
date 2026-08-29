"""Multiple-shooting identification of stable thermal RC parameters."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from celltemp.domain import Trajectory
from celltemp.engine import ThermalRCModel

from .objective import (
    actuator_before_interval,
    masked_huber_loss,
    trajectory_rmse,
    trajectory_tensors,
)

_LOG_MULTIPLIER_LIMIT = 4.0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 80
    steps_per_epoch: int = 8
    batch_size: int = 12
    horizon: int = 60
    learning_rate: float = 0.03
    huber_delta: float = 1.0
    prior_weight: float = 1e-4
    gradient_clip: float = 10.0
    validation_every: int = 5
    patience: int = 8
    seed: int = 42

    def __post_init__(self) -> None:
        positive = {
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "batch_size": self.batch_size,
            "horizon": self.horizon,
            "learning_rate": self.learning_rate,
            "huber_delta": self.huber_delta,
            "validation_every": self.validation_every,
            "patience": self.patience,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"training values must be positive: {invalid}")
        if self.prior_weight < 0.0 or self.gradient_clip < 0.0:
            raise ValueError("prior_weight and gradient_clip must be non-negative")


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_rmse: float
    history: tuple[dict[str, float], ...]


def _log_multipliers(model: ThermalRCModel) -> tuple[torch.Tensor, ...]:
    return (
        model.log_edge_multiplier,
        model.log_tau_multiplier,
        model.log_source_multiplier,
        model.log_boundary_multiplier,
    )


def _parameter_prior(model: ThermalRCModel) -> torch.Tensor:
    nonempty = [item.square().mean() for item in _log_multipliers(model) if item.numel()]
    return torch.stack(nonempty).mean() if nonempty else model.capacity.new_zeros(())


def _valid_starts(trajectory: Trajectory, horizon: int) -> np.ndarray:
    interval_count = len(trajectory.time) - 1
    window = min(horizon, interval_count)
    latest = interval_count - window
    candidates = np.flatnonzero(trajectory.mask[:-1].any(axis=1))
    return candidates[candidates <= latest]


def _sample_batch(
    model: ThermalRCModel,
    trajectories: list[Trajectory],
    config: TrainingConfig,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    choices = rng.integers(0, len(trajectories), size=config.batch_size)
    selected: list[tuple[Trajectory, int]] = []
    horizons: list[int] = []
    for choice in choices:
        trajectory = trajectories[int(choice)]
        starts = _valid_starts(trajectory, config.horizon)
        if not len(starts):
            raise ValueError(f"trajectory {trajectory.case_id} has no valid shooting point")
        start = int(rng.choice(starts))
        selected.append((trajectory, start))
        horizons.append(min(config.horizon, len(trajectory.time) - 1 - start))
    horizon = min(horizons)

    initial_temperatures: list[torch.Tensor] = []
    initial_actuators: list[torch.Tensor] = []
    command_windows: list[torch.Tensor] = []
    dt_windows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for trajectory, start in selected:
        observed, commands, dt, mask = trajectory_tensors(model, trajectory)
        initial_temperatures.append(model.initialize_temperature(observed[start], mask[start]))
        initial_actuators.append(actuator_before_interval(model, commands, dt, start))
        command_windows.append(commands[start : start + horizon])
        dt_windows.append(dt[start : start + horizon])
        targets.append(observed[start + 1 : start + horizon + 1])
        masks.append(mask[start + 1 : start + horizon + 1])
    return (
        torch.stack(initial_temperatures),
        torch.stack(initial_actuators),
        torch.stack(command_windows),
        torch.stack(dt_windows),
        torch.stack(targets),
        torch.stack(masks),
    )


def _validation_rmse(model: ThermalRCModel, trajectories: list[Trajectory]) -> float:
    values = [trajectory_rmse(model, trajectory) for trajectory in trajectories]
    return float(np.mean(values))


def fit_thermal_model(
    model: ThermalRCModel,
    train_trajectories: list[Trajectory],
    validation_trajectories: list[Trajectory] | None = None,
    *,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Fit conductance, lag, source, and boundary multipliers by rollout loss."""
    if not train_trajectories:
        raise ValueError("at least one training trajectory is required")
    config = config or TrainingConfig()
    validation = validation_trajectories or train_trajectories
    missing_initial = sorted(
        {
            trajectory.case_id
            for trajectory in [*train_trajectories, *validation]
            if not trajectory.mask[0].any()
        }
    )
    if missing_initial:
        raise ValueError(f"training trajectories need an initial observation: {missing_initial}")
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 0
    stale_checks = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for _ in range(config.steps_per_epoch):
            initial_t, initial_u, commands, dt, target, mask = _sample_batch(
                model, train_trajectories, config, rng
            )
            optimizer.zero_grad(set_to_none=True)
            states, _ = model.forward_batch(initial_t, commands, dt, initial_u)
            predicted = model.observe(states[:, 1:])
            data_loss = masked_huber_loss(predicted, target, mask, delta=config.huber_delta)
            loss = data_loss + config.prior_weight * _parameter_prior(model)
            loss.backward()
            if config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            with torch.no_grad():
                for parameter in _log_multipliers(model):
                    parameter.clamp_(-_LOG_MULTIPLIER_LIMIT, _LOG_MULTIPLIER_LIMIT)
            epoch_losses.append(float(data_loss.detach().cpu()))

        row = {"epoch": float(epoch), "train_huber": float(np.mean(epoch_losses))}
        should_validate = (
            epoch == 1 or epoch % config.validation_every == 0 or epoch == config.epochs
        )
        if should_validate:
            model.eval()
            validation_rmse = _validation_rmse(model, validation)
            row["validation_rmse"] = validation_rmse
            if validation_rmse < best_validation:
                best_validation = validation_rmse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_checks = 0
            else:
                stale_checks += 1
        history.append(row)
        if stale_checks >= config.patience:
            break

    model.load_state_dict(best_state)
    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_rmse=best_validation,
        history=tuple(history),
    )
