"""Causal feature preparation, training, and rollout for neural baselines."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from itertools import groupby

import numpy as np
import torch
from torch.nn import functional as F

from celltemp.domain import Trajectory

from .models import SequenceModelSpec, SequenceUpdateModel, build_model


@dataclass(frozen=True)
class SequencePreprocessor:
    """Train-only scaling for temperatures, commands, and temperature rates."""

    temperature_mean: np.ndarray
    temperature_std: np.ndarray
    command_mean: np.ndarray
    command_std: np.ndarray
    rate_mean: np.ndarray
    rate_std: np.ndarray
    dt_reference: float

    @classmethod
    def fit(cls, trajectories: list[Trajectory]) -> SequencePreprocessor:
        if not trajectories:
            raise ValueError("neural preprocessing needs training trajectories")
        if any(not trajectory.mask.all() for trajectory in trajectories):
            raise ValueError("neural training trajectories must have complete temperatures")
        temperatures = np.vstack([trajectory.temperature for trajectory in trajectories])
        commands = np.vstack([trajectory.commands for trajectory in trajectories])
        rates = np.vstack(
            [
                np.diff(trajectory.temperature, axis=0) / trajectory.dt[:, None]
                for trajectory in trajectories
            ]
        )
        all_dt = np.concatenate([trajectory.dt for trajectory in trajectories])
        return cls(
            temperature_mean=np.mean(temperatures, axis=0),
            temperature_std=_safe_std(temperatures),
            command_mean=np.mean(commands, axis=0),
            command_std=_safe_std(commands),
            rate_mean=np.mean(rates, axis=0),
            rate_std=_safe_std(rates),
            dt_reference=float(np.median(all_dt)),
        )

    def temperature(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.temperature_mean) / self.temperature_std

    def command(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.command_mean) / self.command_std

    def rate(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.rate_mean) / self.rate_std

    def inverse_rate(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.rate_std + self.rate_mean

    def log_dt(self, values: np.ndarray) -> np.ndarray:
        return np.log(np.asarray(values) / self.dt_reference)

    def as_dict(self) -> dict[str, object]:
        return {
            "temperature_mean": self.temperature_mean.tolist(),
            "temperature_std": self.temperature_std.tolist(),
            "command_mean": self.command_mean.tolist(),
            "command_std": self.command_std.tolist(),
            "rate_mean": self.rate_mean.tolist(),
            "rate_std": self.rate_std.tolist(),
            "dt_reference": self.dt_reference,
        }


@dataclass(frozen=True)
class TrainingData:
    history: torch.Tensor
    query: torch.Tensor
    target_rate: torch.Tensor
    case_weight: torch.Tensor


@dataclass(frozen=True)
class NeuralTrainingConfig:
    epochs: int = 300
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    validation_every: int = 5
    patience_checks: int = 8


@dataclass(frozen=True)
class NeuralTrainingResult:
    model: SequenceUpdateModel
    best_epoch: int
    best_validation_rmse: float
    parameter_count: int
    elapsed_seconds: float
    history: tuple[dict[str, float], ...]


def _safe_std(values: np.ndarray) -> np.ndarray:
    standard_deviation = np.std(values, axis=0)
    return np.where(standard_deviation < 1e-8, 1.0, standard_deviation)


def _history_indices(index: int, history_steps: int) -> np.ndarray:
    return np.clip(np.arange(index - history_steps + 1, index + 1), 0, index)


def _history_features(
    temperature: np.ndarray,
    commands: np.ndarray,
    dt: np.ndarray,
    index: int,
    history_steps: int,
    preprocessor: SequencePreprocessor,
) -> np.ndarray:
    indices = _history_indices(index, history_steps)
    interval_indices = np.minimum(indices, len(commands) - 1)
    return np.concatenate(
        [
            preprocessor.temperature(temperature[indices]),
            preprocessor.command(commands[interval_indices]),
            preprocessor.log_dt(dt[interval_indices])[:, None],
        ],
        axis=1,
    )


def _query_features(
    command: np.ndarray,
    dt: float,
    preprocessor: SequencePreprocessor,
) -> np.ndarray:
    return np.concatenate(
        [
            preprocessor.command(command[None, :])[0],
            preprocessor.log_dt(np.array([dt], dtype=np.float64)),
        ]
    )


def prepare_training_data(
    trajectories: list[Trajectory],
    preprocessor: SequencePreprocessor,
    *,
    history_steps: int,
) -> TrainingData:
    """Create one causal sample per interval without crossing case boundaries."""
    histories: list[np.ndarray] = []
    queries: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[float] = []
    for trajectory in trajectories:
        interval_count = len(trajectory.commands)
        if not trajectory.mask.all():
            raise ValueError(f"training case {trajectory.case_id} contains missing temperatures")
        case_weight = 1.0 / interval_count
        for index in range(interval_count):
            histories.append(
                _history_features(
                    trajectory.temperature,
                    trajectory.commands,
                    trajectory.dt,
                    index,
                    history_steps,
                    preprocessor,
                )
            )
            queries.append(
                _query_features(
                    trajectory.commands[index],
                    float(trajectory.dt[index]),
                    preprocessor,
                )
            )
            rate = (
                trajectory.temperature[index + 1] - trajectory.temperature[index]
            ) / trajectory.dt[index]
            targets.append(preprocessor.rate(rate[None, :])[0])
            weights.append(case_weight)
    return TrainingData(
        history=torch.tensor(np.stack(histories), dtype=torch.float32),
        query=torch.tensor(np.stack(queries), dtype=torch.float32),
        target_rate=torch.tensor(np.stack(targets), dtype=torch.float32),
        case_weight=torch.tensor(weights, dtype=torch.float32),
    )


def _initial_history_batch(
    trajectories: list[Trajectory],
    preprocessor: SequencePreprocessor,
    history_steps: int,
) -> tuple[torch.Tensor, np.ndarray]:
    initial_temperature = np.stack([trajectory.temperature[0] for trajectory in trajectories])
    initial_command = np.stack([trajectory.commands[0] for trajectory in trajectories])
    initial_dt = np.array([trajectory.dt[0] for trajectory in trajectories])
    first_features = np.concatenate(
        [
            preprocessor.temperature(initial_temperature),
            preprocessor.command(initial_command),
            preprocessor.log_dt(initial_dt)[:, None],
        ],
        axis=1,
    )
    repeated = np.repeat(first_features[:, None, :], history_steps, axis=1)
    return torch.tensor(repeated, dtype=torch.float32), initial_temperature


@torch.no_grad()
def _group_open_loop_rmse(
    model: SequenceUpdateModel,
    trajectories: list[Trajectory],
    preprocessor: SequencePreprocessor,
    history_steps: int,
) -> np.ndarray:
    model.eval()
    history, current = _initial_history_batch(trajectories, preprocessor, history_steps)
    truth = np.stack([trajectory.temperature for trajectory in trajectories])
    commands = np.stack([trajectory.commands for trajectory in trajectories])
    dt = np.stack([trajectory.dt for trajectory in trajectories])
    squared_error = np.zeros(len(trajectories), dtype=np.float64)
    value_count = np.zeros(len(trajectories), dtype=np.float64)

    for index in range(commands.shape[1]):
        query_values = np.concatenate(
            [
                preprocessor.command(commands[:, index]),
                preprocessor.log_dt(dt[:, index])[:, None],
            ],
            axis=1,
        )
        query = torch.tensor(query_values, dtype=torch.float32)
        rate_scaled = model(history, query).cpu().numpy()
        rate = preprocessor.inverse_rate(rate_scaled)
        current = current + rate * dt[:, index, None]
        error = current - truth[:, index + 1]
        squared_error += np.sum(error**2, axis=1)
        value_count += error.shape[1]
        if index + 1 < commands.shape[1]:
            next_features = np.concatenate(
                [
                    preprocessor.temperature(current),
                    preprocessor.command(commands[:, index + 1]),
                    preprocessor.log_dt(dt[:, index + 1])[:, None],
                ],
                axis=1,
            )
            next_tensor = torch.tensor(next_features[:, None, :], dtype=torch.float32)
            history = torch.cat([history[:, 1:], next_tensor], dim=1)
    return np.sqrt(squared_error / value_count)


def causal_validation_rmse(
    model: SequenceUpdateModel,
    trajectories: list[Trajectory],
    preprocessor: SequencePreprocessor,
    history_steps: int,
) -> float:
    """Mean complete open-loop RMSE, weighted equally by validation case."""
    ordered = sorted(trajectories, key=lambda item: len(item.time))
    case_values: list[float] = []
    for _, items in groupby(ordered, key=lambda item: len(item.time)):
        group = list(items)
        case_values.extend(
            _group_open_loop_rmse(model, group, preprocessor, history_steps).tolist()
        )
    return float(np.mean(case_values))


def fit_model(
    name: str,
    spec: SequenceModelSpec,
    data: TrainingData,
    validation: list[Trajectory],
    preprocessor: SequencePreprocessor,
    *,
    config: NeuralTrainingConfig,
    seed: int,
) -> NeuralTrainingResult:
    """Fit one neural model and select it by future-blind open-loop validation."""
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = build_model(name, spec)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    best_epoch = 0
    stale_checks = 0
    history_rows: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = generator.permutation(len(data.history))
        loss_numerator = 0.0
        loss_denominator = 0.0
        for offset in range(0, len(order), config.batch_size):
            indices = torch.tensor(order[offset : offset + config.batch_size], dtype=torch.long)
            predicted = model(data.history[indices], data.query[indices])
            sample_loss = F.huber_loss(
                predicted,
                data.target_rate[indices],
                delta=1.0,
                reduction="none",
            ).mean(dim=1)
            weight = data.case_weight[indices]
            loss = torch.sum(sample_loss * weight) / torch.sum(weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            loss_numerator += float(torch.sum(sample_loss.detach() * weight))
            loss_denominator += float(torch.sum(weight))

        row = {
            "epoch": float(epoch),
            "train_huber": loss_numerator / loss_denominator,
            "validation_open_loop_rmse": float("nan"),
        }
        should_validate = epoch == 1 or epoch % config.validation_every == 0
        should_validate = should_validate or epoch == config.epochs
        if should_validate:
            validation_rmse = causal_validation_rmse(
                model,
                validation,
                preprocessor,
                spec.history_steps,
            )
            row["validation_open_loop_rmse"] = validation_rmse
            if validation_rmse < best_validation:
                best_validation = validation_rmse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_checks = 0
            else:
                stale_checks += 1
        history_rows.append(row)
        if stale_checks >= config.patience_checks:
            break

    model.load_state_dict(best_state)
    model.eval()
    return NeuralTrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_rmse=best_validation,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        elapsed_seconds=time.perf_counter() - started,
        history=tuple(history_rows),
    )


def _causal_fill_prefix(
    trajectory: Trajectory,
    origin: int,
    preprocessor: SequencePreprocessor,
) -> np.ndarray:
    """Fill only from observations at or before the forecast origin."""
    rows: list[np.ndarray] = []
    current = preprocessor.temperature_mean.copy()
    for index in range(origin + 1):
        observed = trajectory.mask[index]
        if index == 0 and observed.any():
            current[:] = float(np.mean(trajectory.temperature[index, observed]))
        current = current.copy()
        current[observed] = trajectory.temperature[index, observed]
        rows.append(current)
    return np.stack(rows)


@torch.no_grad()
def forecast_sequence(
    model: SequenceUpdateModel,
    trajectory: Trajectory,
    preprocessor: SequencePreprocessor,
    *,
    history_steps: int,
    origin: int,
) -> np.ndarray:
    """Open-loop sensor forecast using no temperature after ``origin``."""
    if not 0 <= origin < len(trajectory.time) - 1:
        raise ValueError("forecast origin must leave at least one future interval")
    model.eval()
    filled_prefix = _causal_fill_prefix(trajectory, origin, preprocessor)
    history_values = _history_features(
        filled_prefix,
        trajectory.commands,
        trajectory.dt,
        origin,
        history_steps,
        preprocessor,
    )
    history = torch.tensor(history_values[None, :, :], dtype=torch.float32)
    current = filled_prefix[-1].copy()
    predicted = [current.copy()]
    for index in range(origin, len(trajectory.commands)):
        query_values = _query_features(
            trajectory.commands[index],
            float(trajectory.dt[index]),
            preprocessor,
        )
        query = torch.tensor(query_values[None, :], dtype=torch.float32)
        rate_scaled = model(history, query).cpu().numpy()[0]
        current = current + preprocessor.inverse_rate(rate_scaled) * trajectory.dt[index]
        predicted.append(current.copy())
        if index + 1 < len(trajectory.commands):
            next_feature = np.concatenate(
                [
                    preprocessor.temperature(current[None, :])[0],
                    preprocessor.command(trajectory.commands[index + 1][None, :])[0],
                    preprocessor.log_dt(np.array([trajectory.dt[index + 1]])),
                ]
            )
            next_tensor = torch.tensor(next_feature[None, None, :], dtype=torch.float32)
            history = torch.cat([history[:, 1:], next_tensor], dim=1)
    return np.stack(predicted)
