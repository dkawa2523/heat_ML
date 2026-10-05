"""Descriptive diagnostics for aligned temperature predictions and truth."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .metrics import mae, max_abs_error, rmse


def persistence_prediction(
    temperature: np.ndarray,
    observation_mask: np.ndarray,
    origin: int,
) -> np.ndarray:
    """Repeat the latest observed temperature from a causal history.

    A sensor without any history is initialized from the mean of the sensors that
    were observed. The returned rows start at ``origin`` so they align with the
    public open-loop forecast result.
    """
    values = np.asarray(temperature, dtype=np.float64)
    mask = np.asarray(observation_mask, dtype=np.bool_)
    if values.ndim != 2 or not len(values):
        raise ValueError("temperature must have shape [time, sensor]")
    if mask.shape != values.shape:
        raise ValueError("observation_mask must have the same shape as temperature")
    if not 0 <= origin < len(values):
        raise ValueError("origin is outside the temperature history")
    if not np.isfinite(values[mask]).all():
        raise ValueError("observed temperatures must be finite")

    latest = np.full(values.shape[1], np.nan, dtype=np.float64)
    history_mask = mask[: origin + 1]
    for sensor_index in range(values.shape[1]):
        observed = np.flatnonzero(history_mask[:, sensor_index])
        if len(observed):
            latest[sensor_index] = values[int(observed[-1]), sensor_index]
    available = np.isfinite(latest)
    if not available.any():
        raise ValueError("persistence baseline needs at least one observed temperature")
    latest[~available] = float(np.mean(latest[available]))
    return np.repeat(latest[None, :], len(values) - origin, axis=0)


def validate_prediction_arrays(
    truth: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned arrays, permitting missing truth but requiring finite predictions."""
    truth_values = np.asarray(truth, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    if (
        truth_values.shape != predicted_values.shape
        or truth_values.ndim == 0
        or not len(truth_values)
    ):
        raise ValueError("truth and predicted must be equal non-empty arrays")
    if np.isinf(truth_values).any():
        raise ValueError(
            "truth may contain NaN for missing observations, but cannot contain infinity"
        )
    if not np.isfinite(predicted_values).all():
        raise ValueError(
            "predicted temperatures must all be finite, including where truth is missing"
        )
    return truth_values, predicted_values


def _validated_prediction_values(
    truth: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate finite errors only where validated truth is observed."""
    truth_values, predicted_values = validate_prediction_arrays(truth, predicted)
    observed = np.isfinite(truth_values)
    error = np.full(truth_values.shape, np.nan, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        np.subtract(predicted_values, truth_values, out=error, where=observed)
    if not np.isfinite(error[observed]).all():
        raise ValueError("prediction errors must be finite at every observed truth value")
    return truth_values, predicted_values, error


def prediction_error_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int]:
    """Score every observed truth value; reject failed predictions instead of dropping them."""
    truth_values, _, error = _validated_prediction_values(truth, predicted)
    finite_error = error[np.isfinite(error)]
    if not len(finite_error):
        raise ValueError("prediction evaluation needs at least one observed truth value")
    terminal_error = np.atleast_1d(error[-1])
    return {
        "n_points": len(finite_error),
        "n_truth_points": int(np.isfinite(truth_values).sum()),
        "prediction_coverage_fraction": 1.0,
        **_checked_error_metrics(finite_error),
        "terminal_rmse_k": _checked_error_metrics(terminal_error[np.isfinite(terminal_error)])[
            "rmse_k"
        ],
    }


def _checked_error_metrics(error: np.ndarray) -> dict[str, float]:
    """Distinguish an unavailable score from overflow of observed finite errors."""
    with np.errstate(over="ignore", invalid="ignore"):
        metrics = {
            "rmse_k": rmse(error),
            "mae_k": mae(error),
            "bias_k": float(np.mean(error)) if len(error) else float("nan"),
            "max_abs_error_k": max_abs_error(error),
        }
    if len(error) and not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("prediction error metrics must remain finite at observed truth values")
    return metrics


def _prediction_arrays(
    time: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    sensor_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_time = np.asarray(time, dtype=np.float64)
    truth_values, predicted_values, _ = _validated_prediction_values(truth, predicted)
    if sample_time.ndim != 1:
        raise ValueError("time must be one-dimensional")
    expected_shape = (len(sample_time), len(sensor_names))
    if truth_values.shape != expected_shape:
        raise ValueError("time and truth must have shapes [time] and [time, sensor]")
    if predicted_values.shape != expected_shape:
        raise ValueError("predicted must have the same shape as truth")
    if len(sample_time) < 2:
        raise ValueError("prediction diagnostics need at least two time rows")
    if not np.isfinite(sample_time).all() or not np.all(np.diff(sample_time) > 0.0):
        raise ValueError("time must be finite and strictly increasing")
    return sample_time, truth_values, predicted_values


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def prediction_sensor_rows(
    time: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    sensor_names: tuple[str, ...],
) -> list[dict[str, float | int | str]]:
    """Summarize prediction error for each sensor.

    Residuals use ``predicted - truth``. NaN truth is an unavailable observation;
    all predictions must remain finite, including at unavailable observations.
    lag-1 correlation additionally requires adjacent original samples so a missing
    interval is not treated as one time step.
    """
    sample_time, truth_values, predicted_values = _prediction_arrays(
        time, truth, predicted, sensor_names
    )
    rows: list[dict[str, float | int | str]] = []
    for index, sensor in enumerate(sensor_names):
        paired = np.isfinite(truth_values[:, index])
        paired_time = sample_time[paired]
        paired_truth = truth_values[paired, index]
        paired_predicted = predicted_values[paired, index]
        error = paired_predicted - paired_truth
        if len(error):
            truth_peak_index = int(np.argmax(paired_truth))
            predicted_peak_index = int(np.argmax(paired_predicted))
            truth_peak_at_boundary = truth_peak_index in (0, len(paired_truth) - 1)
            predicted_peak_at_boundary = predicted_peak_index in (0, len(paired_predicted) - 1)
            terminal_error = float(error[-1])
            peak_temperature_error = float(
                paired_predicted[predicted_peak_index] - paired_truth[truth_peak_index]
            )
            peak_time_error = float(
                paired_time[predicted_peak_index] - paired_time[truth_peak_index]
            )
        else:
            truth_peak_at_boundary = False
            predicted_peak_at_boundary = False
            terminal_error = float("nan")
            peak_temperature_error = float("nan")
            peak_time_error = float("nan")

        full_error = predicted_values[:, index] - truth_values[:, index]
        adjacent = np.isfinite(full_error[:-1]) & np.isfinite(full_error[1:])
        lag1 = _correlation(full_error[:-1][adjacent], full_error[1:][adjacent])
        rows.append(
            {
                "sensor": sensor,
                "n_points": len(error),
                "n_truth_points": len(error),
                "prediction_coverage_fraction": 1.0 if len(error) else float("nan"),
                **_checked_error_metrics(error),
                "terminal_error_k": terminal_error,
                "peak_temperature_error_k": peak_temperature_error,
                "peak_time_error_s": peak_time_error,
                "truth_peak_at_boundary": truth_peak_at_boundary,
                "predicted_peak_at_boundary": predicted_peak_at_boundary,
                "peak_time_qualified": bool(len(error)) and not truth_peak_at_boundary,
                "lag1_pair_count": int(adjacent.sum()),
                "residual_lag1_correlation": lag1,
            }
        )
    return rows


def _largest_finite_row(
    rows: list[dict[str, float | int | str]],
    key: str,
    *,
    absolute: bool = False,
) -> dict[str, float | int | str] | None:
    available = [row for row in rows if np.isfinite(float(row[key]))]
    if not available:
        return None
    score = (lambda row: abs(float(row[key]))) if absolute else (lambda row: float(row[key]))
    return max(available, key=score)


def prediction_comparison_rows(
    time: np.ndarray,
    truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    sensor_names: tuple[str, ...],
) -> list[dict[str, float | int | str]]:
    """Compare named predictions at one common case and forecast boundary.

    Each result is one model row. Aggregate errors cover every aligned sensor
    value; worst-sensor and peak fields preserve the most decision-relevant
    sensor-level failure without creating a second comparison table.
    """
    rows: list[dict[str, float | int | str]] = []
    for model, predicted in predictions.items():
        aggregate = prediction_error_metrics(truth, predicted)
        sensors = prediction_sensor_rows(time, truth, predicted, sensor_names)
        worst_sensor = _largest_finite_row(sensors, "rmse_k")
        worst_peak_temperature = _largest_finite_row(
            sensors, "peak_temperature_error_k", absolute=True
        )
        qualified_peak_times = [row for row in sensors if bool(row["peak_time_qualified"])]
        worst_peak_time = _largest_finite_row(
            qualified_peak_times, "peak_time_error_s", absolute=True
        )
        rows.append(
            {
                "model": model,
                **aggregate,
                "worst_sensor": "" if worst_sensor is None else str(worst_sensor["sensor"]),
                "worst_sensor_rmse_k": (
                    float("nan") if worst_sensor is None else float(worst_sensor["rmse_k"])
                ),
                "max_abs_peak_temperature_error_k": (
                    float("nan")
                    if worst_peak_temperature is None
                    else abs(float(worst_peak_temperature["peak_temperature_error_k"]))
                ),
                "max_abs_peak_temperature_error_sensor": (
                    "" if worst_peak_temperature is None else str(worst_peak_temperature["sensor"])
                ),
                "max_abs_peak_time_error_s": (
                    float("nan")
                    if worst_peak_time is None
                    else abs(float(worst_peak_time["peak_time_error_s"]))
                ),
                "max_abs_peak_time_error_sensor": (
                    "" if worst_peak_time is None else str(worst_peak_time["sensor"])
                ),
                "peak_time_qualified_sensors": len(qualified_peak_times),
            }
        )
    return rows


def _dependence_row(
    sensor: str,
    quantity: str,
    unit: str,
    values: np.ndarray,
    error: np.ndarray,
) -> dict[str, float | int | str]:
    paired = np.isfinite(values) & np.isfinite(error)
    x = values[paired]
    y = error[paired]
    if len(x):
        minimum = float(np.min(x))
        maximum = float(np.max(x))
    else:
        minimum = float("nan")
        maximum = float("nan")

    if len(x) >= 2:
        centered = x - np.mean(x)
        denominator = float(np.sum(centered**2))
        slope = (
            float(np.sum(centered * (y - np.mean(y))) / denominator)
            if denominator > 0.0
            else float("nan")
        )
        correlation = _correlation(x, y)
    else:
        slope = float("nan")
        correlation = float("nan")
    return {
        "sensor": sensor,
        "quantity": quantity,
        "quantity_unit": unit,
        "n_points": len(x),
        "quantity_min": minimum,
        "quantity_max": maximum,
        "residual_slope_k_per_unit": slope,
        "residual_correlation": correlation,
    }


def residual_dependence_rows(
    time: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    sensor_names: tuple[str, ...],
    *,
    conditions: Mapping[str, tuple[np.ndarray, str]] | None = None,
    temperature_unit: str = "K",
) -> list[dict[str, float | int | str]]:
    """Describe linear residual dependence on time, truth temperature, and conditions.

    Slopes and Pearson correlations are descriptive screening values. They do not
    establish causality, statistical significance, or a need for a new model term.
    Constant quantities have NaN slope and correlation.
    """
    sample_time, truth_values, predicted_values = _prediction_arrays(
        time, truth, predicted, sensor_names
    )
    condition_values: list[tuple[str, np.ndarray, str]] = []
    for name, (values, unit) in (conditions or {}).items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != sample_time.shape:
            raise ValueError(f"condition {name!r} must have one value per time row")
        condition_values.append((name, array, unit))

    rows: list[dict[str, float | int | str]] = []
    for index, sensor in enumerate(sensor_names):
        error = predicted_values[:, index] - truth_values[:, index]
        quantities = [
            ("time", sample_time, "s"),
            ("truth_temperature", truth_values[:, index], temperature_unit),
            *condition_values,
        ]
        rows.extend(
            _dependence_row(sensor, name, unit, values, error) for name, values, unit in quantities
        )
    return rows
