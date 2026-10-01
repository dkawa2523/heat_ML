"""Causal evaluation on train, validation, and held-out internal test cases."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from celltemp.analysis import prediction_error_metrics
from celltemp.domain import Trajectory
from celltemp.engine import ThermalRCModel
from celltemp.inference import forecast

from .training import NeuralTrainingResult, SequencePreprocessor, forecast_sequence

SPLIT_LABELS = {"train": "train", "val": "validation", "test": "test"}


def initial_condition_request(trajectory: Trajectory) -> Trajectory:
    """Keep only the first temperature row so future truth cannot enter a forecast."""
    temperature = np.full(trajectory.temperature.shape, np.nan, dtype=float)
    mask = np.zeros(trajectory.mask.shape, dtype=bool)
    mask[0] = trajectory.mask[0]
    temperature[0, mask[0]] = trajectory.temperature[0, mask[0]]
    return Trajectory(
        case_id=trajectory.case_id,
        time=trajectory.time,
        temperature=temperature,
        commands=trajectory.commands,
        sensor_names=trajectory.sensor_names,
        control_names=trajectory.control_names,
        observation_mask=mask,
        metadata=trajectory.metadata,
        initial_actuator=trajectory.initial_actuator,
    )


def _point_rows(
    dataset: str,
    case_id: str,
    model_name: str,
    time: np.ndarray,
    sensor_names: tuple[str, ...],
    truth: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for time_index, timestamp in enumerate(time):
        for sensor_index, sensor in enumerate(sensor_names):
            rows.append(
                {
                    "dataset": dataset,
                    "split": "test",
                    "case_id": case_id,
                    "model": model_name,
                    "time": float(timestamp),
                    "sensor": sensor,
                    "truth": float(truth[time_index, sensor_index]),
                    "predicted": float(predicted[time_index, sensor_index]),
                }
            )
    return rows


def evaluate_internal_splits(
    dataset: str,
    rc_model: ThermalRCModel,
    splits: Mapping[str, list[Trajectory]],
    trained: Mapping[str, NeuralTrainingResult],
    preprocessor: SequencePreprocessor,
    history_steps: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate every split from an initial condition; retain points only for test."""
    case_rows: list[dict[str, object]] = []
    test_point_rows: list[dict[str, object]] = []
    for raw_split, split_name in SPLIT_LABELS.items():
        for trajectory in splits[raw_split]:
            request = initial_condition_request(trajectory)
            rc_result = forecast(rc_model, request)
            predictions = {"physical_rc": rc_result.sensor_temperature}
            predictions.update(
                {
                    name: forecast_sequence(
                        result.model,
                        request,
                        preprocessor,
                        history_steps=history_steps,
                        origin=0,
                    )
                    for name, result in trained.items()
                }
            )
            truth = trajectory.temperature[1:]
            for model_name, full_prediction in predictions.items():
                predicted = full_prediction[1:]
                metrics = prediction_error_metrics(truth, predicted)
                case_rows.append(
                    {
                        "dataset": dataset,
                        "split": split_name,
                        "case_id": trajectory.case_id,
                        "model": model_name,
                        "forecast_origin_index": 0,
                        **metrics,
                    }
                )
                if split_name == "test":
                    test_point_rows.extend(
                        _point_rows(
                            dataset,
                            trajectory.case_id,
                            model_name,
                            trajectory.time[1:],
                            trajectory.sensor_names,
                            truth,
                            predicted,
                        )
                    )
    return case_rows, test_point_rows
