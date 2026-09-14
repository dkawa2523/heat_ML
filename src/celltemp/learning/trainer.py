"""Case-balanced trajectory identification of stable thermal RC parameters."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch

from celltemp.domain import Trajectory
from celltemp.engine import ThermalRCModel

from .objective import (
    actuator_before_interval,
    initial_observation_rmse,
    masked_huber_loss,
    profiled_initial_temperature,
    trajectory_initial_actuator,
    trajectory_tensors,
)

_LOG_MULTIPLIER_LIMIT = 4.0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 80
    batch_size: int = 12
    horizon: int | None = None
    learning_rate: float = 0.03
    huber_delta: float = 1.0
    initial_temperature_prior_std: float = 50.0
    prior_weight: float = 1e-4
    gradient_clip: float = 10.0
    validation_every: int = 5
    patience: int = 8
    seed: int = 42

    def __post_init__(self) -> None:
        positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "huber_delta": self.huber_delta,
            "initial_temperature_prior_std": self.initial_temperature_prior_std,
            "validation_every": self.validation_every,
            "patience": self.patience,
        }
        invalid = [
            name for name, value in positive.items() if not math.isfinite(value) or value <= 0
        ]
        if invalid:
            raise ValueError(f"training values must be positive and finite: {invalid}")
        if self.horizon is not None and (not math.isfinite(self.horizon) or self.horizon <= 0):
            raise ValueError("training horizon must be positive and finite or null")
        non_negative = {
            "prior_weight": self.prior_weight,
            "gradient_clip": self.gradient_clip,
        }
        invalid = [
            name for name, value in non_negative.items() if not math.isfinite(value) or value < 0
        ]
        if invalid:
            raise ValueError(f"training values must be non-negative and finite: {invalid}")


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_causal_validation_rmse: float
    history: tuple[dict[str, float], ...]


def _log_multipliers(model: ThermalRCModel) -> tuple[torch.Tensor, ...]:
    return model.log_parameter_multipliers()


def _parameter_prior(model: ThermalRCModel) -> torch.Tensor:
    learned = [item for item in model.learnable_log_parameter_values() if item.numel()]
    if learned:
        return torch.cat(learned).square().mean()
    raw = _log_multipliers(model)
    return sum((item.sum() * 0.0 for item in raw), model.capacity.new_zeros(()))


def _valid_starts(trajectory: Trajectory, horizon: int) -> np.ndarray:
    interval_count = len(trajectory.time) - 1
    window = min(horizon, interval_count)
    latest = interval_count - window
    candidates = np.flatnonzero(trajectory.mask[:-1].any(axis=1))
    return np.asarray(
        [
            start
            for start in candidates
            if start <= latest and trajectory.mask[start + 1 : start + window + 1].any()
        ],
        dtype=np.int64,
    )


def _select_rollouts(
    trajectories: list[Trajectory],
    horizon: int | None,
    rng: np.random.Generator,
) -> list[tuple[Trajectory, int, int]]:
    selected: list[tuple[Trajectory, int, int]] = []
    for trajectory in trajectories:
        interval_count = len(trajectory.time) - 1
        if horizon is None:
            selected.append((trajectory, 0, interval_count))
            continue
        window = min(horizon, interval_count)
        starts = _valid_starts(trajectory, window)
        if not len(starts):
            raise ValueError(
                f"trajectory {trajectory.case_id} has no observed target in a shooting window"
            )
        start = int(rng.choice(starts))
        selected.append((trajectory, start, start + window))
    return selected


def _rollout_case_losses(
    model: ThermalRCModel,
    selected: list[tuple[Trajectory, int, int]],
    *,
    huber_delta: float,
    initial_temperature_prior_std: float,
) -> torch.Tensor:
    """Roll out equal-length groups and return one equally weighted loss per case."""
    by_length: dict[int, list[tuple[Trajectory, int, int]]] = {}
    for item in selected:
        by_length.setdefault(item[2] - item[1], []).append(item)

    case_losses: list[torch.Tensor] = []
    for group in by_length.values():
        initial_temperatures: list[torch.Tensor] = []
        initial_actuators: list[torch.Tensor] = []
        command_windows: list[torch.Tensor] = []
        dt_windows: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for trajectory, start, stop in group:
            observed, commands, dt, mask = trajectory_tensors(model, trajectory)
            starting_actuator = trajectory_initial_actuator(model, trajectory, commands)
            initial_temperatures.append(
                profiled_initial_temperature(
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
            )
            initial_actuators.append(
                actuator_before_interval(model, commands, dt, start, starting_actuator)
            )
            command_windows.append(commands[start:stop])
            dt_windows.append(dt[start:stop])
            targets.append(observed[start + 1 : stop + 1])
            masks.append(mask[start + 1 : stop + 1])

        states, _ = model.forward_batch(
            torch.stack(initial_temperatures),
            torch.stack(command_windows),
            torch.stack(dt_windows),
            torch.stack(initial_actuators),
        )
        predicted = model.observe(states[:, 1:])
        case_losses.extend(
            masked_huber_loss(predicted[index], target, mask, delta=huber_delta)
            for index, (target, mask) in enumerate(zip(targets, masks, strict=True))
        )
    return torch.stack(case_losses)


def _causal_validation_rmse(
    model: ThermalRCModel,
    trajectories: list[Trajectory],
) -> float:
    """Select deployed parameters using a future-blind initialization metric."""
    values = [initial_observation_rmse(model, trajectory) for trajectory in trajectories]
    return float(np.mean(values))


def _validate_observations(trajectories: list[Trajectory]) -> None:
    missing_initial = sorted(
        {trajectory.case_id for trajectory in trajectories if not trajectory.mask[0].any()}
    )
    if missing_initial:
        raise ValueError(f"training trajectories need an initial observation: {missing_initial}")
    missing_targets = sorted(
        {trajectory.case_id for trajectory in trajectories if not trajectory.mask[1:].any()}
    )
    if missing_targets:
        raise ValueError(
            f"training trajectories need an observation after the initial row: {missing_targets}"
        )


def _checked_backward(loss: torch.Tensor, model: ThermalRCModel, epoch: int) -> None:
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
    loss.backward()
    if any(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise FloatingPointError(f"non-finite training gradient at epoch {epoch}")


def fit_thermal_model(
    model: ThermalRCModel,
    train_trajectories: list[Trajectory],
    validation_trajectories: list[Trajectory] | None = None,
    *,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Fit physical multipliers using one case-balanced pass per epoch."""
    if not train_trajectories:
        raise ValueError("at least one training trajectory is required")
    config = config or TrainingConfig()
    validation = validation_trajectories or train_trajectories
    _validate_observations([*train_trajectories, *validation])
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
        order = rng.permutation(len(train_trajectories))
        for offset in range(0, len(order), config.batch_size):
            trajectories = [
                train_trajectories[int(index)]
                for index in order[offset : offset + config.batch_size]
            ]
            selected = _select_rollouts(trajectories, config.horizon, rng)
            optimizer.zero_grad(set_to_none=True)
            case_losses = _rollout_case_losses(
                model,
                selected,
                huber_delta=config.huber_delta,
                initial_temperature_prior_std=config.initial_temperature_prior_std,
            )
            data_loss = case_losses.mean()
            loss = data_loss + config.prior_weight * _parameter_prior(model)
            _checked_backward(loss, model, epoch)
            if config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            with torch.no_grad():
                for parameter in _log_multipliers(model):
                    parameter.clamp_(-_LOG_MULTIPLIER_LIMIT, _LOG_MULTIPLIER_LIMIT)
            epoch_losses.extend(case_losses.detach().cpu().tolist())

        row = {"epoch": float(epoch), "train_huber": float(np.mean(epoch_losses))}
        should_validate = (
            epoch == 1 or epoch % config.validation_every == 0 or epoch == config.epochs
        )
        if should_validate:
            model.eval()
            causal_validation_rmse = _causal_validation_rmse(
                model,
                validation,
            )
            if not math.isfinite(causal_validation_rmse):
                raise FloatingPointError(f"non-finite validation RMSE at epoch {epoch}")
            row["causal_validation_rmse"] = causal_validation_rmse
            if causal_validation_rmse < best_validation:
                best_validation = causal_validation_rmse
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
        best_causal_validation_rmse=best_validation,
        history=tuple(history),
    )
