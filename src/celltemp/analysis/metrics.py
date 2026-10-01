"""Small, reusable metrics for measured and simulated thermal waveforms."""

from __future__ import annotations

import math

import numpy as np


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def rmse(error: np.ndarray) -> float:
    """Root-mean-square of finite error values, or NaN when none are available."""
    values = _finite(error)
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def mae(error: np.ndarray) -> float:
    """Mean absolute finite error, or NaN when none are available."""
    values = _finite(error)
    return float(np.mean(np.abs(values))) if values.size else float("nan")


def max_abs_error(error: np.ndarray) -> float:
    """Largest absolute finite error, or NaN when none are available."""
    values = _finite(error)
    return float(np.max(np.abs(values))) if values.size else float("nan")


def _validate_fraction(value: float, name: str) -> float:
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return fraction


def _finite_series(time: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_time = np.asarray(time, dtype=np.float64)
    sample_values = np.asarray(values, dtype=np.float64)
    if sample_time.ndim != 1 or sample_values.ndim != 1:
        raise ValueError("time and values must be one-dimensional")
    if len(sample_time) != len(sample_values) or len(sample_time) < 2:
        raise ValueError("time and values must have equal length of at least two")
    if not np.isfinite(sample_time).all() or not np.all(np.diff(sample_time) > 0.0):
        raise ValueError("time must be finite and strictly increasing")
    available = np.isfinite(sample_values)
    if available.sum() < 2:
        raise ValueError("a response needs at least two finite temperature samples")
    return sample_time[available], sample_values[available]


def _crossing_time(
    time: np.ndarray,
    values: np.ndarray,
    target: float,
    direction: str,
) -> float:
    reached = values >= target if direction == "heating" else values <= target
    indices = np.flatnonzero(reached)
    if not len(indices):
        return float("nan")
    index = int(indices[0])
    if index == 0:
        return float(time[0])
    before_value = values[index - 1]
    after_value = values[index]
    if after_value == before_value:
        return float(time[index])
    weight = (target - before_value) / (after_value - before_value)
    weight = float(np.clip(weight, 0.0, 1.0))
    return float(time[index - 1] + weight * (time[index] - time[index - 1]))


def _settling_time(
    time: np.ndarray,
    values: np.ndarray,
    final_value: float,
    band: float,
) -> float:
    inside = np.abs(values - final_value) <= band
    remains_inside = np.logical_and.accumulate(inside[::-1])[::-1]
    indices = np.flatnonzero(remains_inside)
    return float(time[int(indices[0])] - time[0]) if len(indices) else float("nan")


def _trapezoid(values: np.ndarray, time: np.ndarray) -> float:
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(time)))


def response_metrics(
    time: np.ndarray,
    temperature: np.ndarray,
    *,
    final_fraction: float = 0.1,
    settling_fraction: float = 0.02,
    minimum_change: float = 1e-9,
) -> dict[str, float | str]:
    """Summarize one thermal response without assuming a fixed sample interval.

    The 10--90 %, 63.2 %, settling, and overshoot values use the mean of the final
    portion of the record as the response endpoint. They are marked NaN for a flat
    record. For multi-step recipes, use these as whole-case descriptors and compute
    phase-specific values after segmenting the recipe.
    """
    final_fraction = _validate_fraction(final_fraction, "final_fraction")
    settling_fraction = _validate_fraction(settling_fraction, "settling_fraction")
    if not math.isfinite(minimum_change) or minimum_change < 0.0:
        raise ValueError("minimum_change must be non-negative and finite")
    sample_time, values = _finite_series(time, temperature)

    final_start = sample_time[-1] - final_fraction * (sample_time[-1] - sample_time[0])
    final_values = values[sample_time >= final_start]
    if not len(final_values):
        final_values = values[-1:]
    initial = float(values[0])
    final = float(np.mean(final_values))
    change = final - initial
    slopes = np.diff(values) / np.diff(sample_time)
    minimum_index = int(np.argmin(values))
    maximum_index = int(np.argmax(values))

    result: dict[str, float | str] = {
        "initial_temperature": initial,
        "final_temperature": final,
        "final_temperature_std": float(np.std(final_values)),
        "temperature_change": change,
        "minimum_temperature": float(values[minimum_index]),
        "time_of_minimum_s": float(sample_time[minimum_index]),
        "maximum_temperature": float(values[maximum_index]),
        "time_of_maximum_s": float(sample_time[maximum_index]),
        "max_heating_rate_per_s": float(np.max(slopes)),
        "max_cooling_rate_per_s": float(np.min(slopes)),
        "integral_change_temperature_s": _trapezoid(values - initial, sample_time),
        "integral_abs_change_temperature_s": _trapezoid(np.abs(values - initial), sample_time),
    }
    if abs(change) <= minimum_change:
        result.update(
            {
                "response_direction": "flat",
                "time_to_63_percent_s": float("nan"),
                "response_time_10_90_s": float("nan"),
                "settling_time_s": float("nan"),
                "overshoot_temperature": float("nan"),
            }
        )
        return result

    direction = "heating" if change > 0.0 else "cooling"
    targets = {fraction: initial + fraction * change for fraction in (0.1, 0.6321205588285577, 0.9)}
    crossings = {
        fraction: _crossing_time(sample_time, values, target, direction)
        for fraction, target in targets.items()
    }
    response_time = crossings[0.9] - crossings[0.1]
    band = max(abs(change) * settling_fraction, minimum_change)
    overshoot = (
        max(0.0, float(values[maximum_index]) - final)
        if direction == "heating"
        else max(0.0, final - float(values[minimum_index]))
    )
    result.update(
        {
            "response_direction": direction,
            "time_to_63_percent_s": crossings[0.6321205588285577] - sample_time[0],
            "response_time_10_90_s": response_time,
            "settling_time_s": _settling_time(sample_time, values, final, band),
            "overshoot_temperature": overshoot,
        }
    )
    return result


def uniformity_trace(temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row-wise mean, sensor span, and standard deviation.

    Span and standard deviation need at least two observed sensors and are NaN for
    single-sensor rows. This prevents a single measurement from being reported as
    perfect spatial uniformity.
    """
    values = np.asarray(temperature, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("temperature must have shape [time, sensor]")
    mean = np.full(len(values), np.nan, dtype=np.float64)
    span = np.full(len(values), np.nan, dtype=np.float64)
    standard_deviation = np.full(len(values), np.nan, dtype=np.float64)
    for index, row in enumerate(values):
        finite = row[np.isfinite(row)]
        if len(finite):
            mean[index] = float(np.mean(finite))
        if len(finite) >= 2:
            span[index] = float(np.max(finite) - np.min(finite))
            standard_deviation[index] = float(np.std(finite))
    return mean, span, standard_deviation


def uniformity_metrics(
    time: np.ndarray,
    temperature: np.ndarray,
    *,
    final_fraction: float = 0.1,
) -> dict[str, float]:
    """Summarize spatial temperature spread across the available sensors."""
    final_fraction = _validate_fraction(final_fraction, "final_fraction")
    sample_time = np.asarray(time, dtype=np.float64)
    if sample_time.ndim != 1 or not np.isfinite(sample_time).all():
        raise ValueError("time must be a finite one-dimensional array")
    if len(sample_time) != len(temperature):
        raise ValueError("time and temperature row counts must match")
    _, span, standard_deviation = uniformity_trace(temperature)
    finite_span = np.isfinite(span)
    if not finite_span.any():
        return {
            "mean_sensor_span": float("nan"),
            "maximum_sensor_span": float("nan"),
            "time_of_maximum_span_s": float("nan"),
            "final_sensor_span": float("nan"),
            "mean_sensor_std": float("nan"),
            "maximum_sensor_std": float("nan"),
        }
    maximum_index = int(np.nanargmax(span))
    final_start = sample_time[-1] - final_fraction * (sample_time[-1] - sample_time[0])
    final_span = span[(sample_time >= final_start) & finite_span]
    return {
        "mean_sensor_span": float(np.nanmean(span)),
        "maximum_sensor_span": float(span[maximum_index]),
        "time_of_maximum_span_s": float(sample_time[maximum_index]),
        "final_sensor_span": float(np.mean(final_span)) if len(final_span) else float("nan"),
        "mean_sensor_std": float(np.nanmean(standard_deviation)),
        "maximum_sensor_std": float(np.nanmax(standard_deviation)),
    }


def control_metrics(
    commands: np.ndarray,
    dt: np.ndarray,
    *,
    change_rtol: float = 1e-9,
    change_atol: float = 1e-12,
) -> dict[str, float | int]:
    """Summarize one zero-order-held command waveform."""
    values = np.asarray(commands, dtype=np.float64)
    durations = np.asarray(dt, dtype=np.float64)
    if values.ndim != 1 or durations.shape != values.shape or not len(values):
        raise ValueError("commands and dt must be equal non-empty one-dimensional arrays")
    if not np.isfinite(values).all() or not np.isfinite(durations).all():
        raise ValueError("commands and dt must be finite")
    if np.any(durations <= 0.0):
        raise ValueError("dt must be positive")
    if len(values) > 1:
        changes = ~np.isclose(values[1:], values[:-1], rtol=change_rtol, atol=change_atol)
        slew = np.diff(values) / durations[:-1]
        maximum_absolute_slew = float(np.max(np.abs(slew)))
        total_variation = float(np.sum(np.abs(np.diff(values))))
        change_count = int(changes.sum())
    else:
        maximum_absolute_slew = 0.0
        total_variation = 0.0
        change_count = 0
    duration = float(durations.sum())
    return {
        "minimum_command": float(values.min()),
        "maximum_command": float(values.max()),
        "time_weighted_mean_command": float(np.sum(values * durations) / duration),
        "command_integral_unit_s": float(np.sum(values * durations)),
        "total_variation": total_variation,
        "maximum_absolute_slew_per_s": maximum_absolute_slew,
        "change_count": change_count,
    }


def thermal_case_metrics(
    time: np.ndarray,
    temperature: np.ndarray,
    *,
    final_fraction: float = 0.1,
) -> dict[str, float | int]:
    """Return case-level duration, availability, and sensor-spread metrics."""
    sample_time = np.asarray(time, dtype=np.float64)
    values = np.asarray(temperature, dtype=np.float64)
    if sample_time.ndim != 1 or values.ndim != 2 or len(sample_time) != len(values):
        raise ValueError("time and temperature must have shapes [time] and [time, sensor]")
    if len(sample_time) < 2:
        raise ValueError("a thermal case needs at least two time rows")
    if not np.isfinite(sample_time).all() or not np.all(np.diff(sample_time) > 0.0):
        raise ValueError("time must be finite and strictly increasing")
    return {
        "rows": len(sample_time),
        "duration_s": float(sample_time[-1] - sample_time[0]),
        "observed_fraction": float(np.isfinite(values).mean()),
        **uniformity_metrics(sample_time, values, final_fraction=final_fraction),
    }


def sensor_response_rows(
    time: np.ndarray,
    temperature: np.ndarray,
    sensor_names: tuple[str, ...],
    *,
    final_fraction: float = 0.1,
    settling_fraction: float = 0.02,
) -> list[dict[str, float | str]]:
    """Apply the standard response metrics to each sensor column."""
    values = np.asarray(temperature, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(sensor_names):
        raise ValueError("temperature must have one column per sensor name")
    return [
        {
            "sensor": sensor,
            **response_metrics(
                time,
                values[:, index],
                final_fraction=final_fraction,
                settling_fraction=settling_fraction,
            ),
        }
        for index, sensor in enumerate(sensor_names)
    ]


def control_waveform_rows(
    commands: np.ndarray,
    dt: np.ndarray,
    control_names: tuple[str, ...],
    *,
    units: tuple[str | None, ...] | None = None,
    roles: tuple[str | None, ...] | None = None,
) -> list[dict[str, float | int | str | None]]:
    """Apply the standard zero-order-held metrics to each control column."""
    values = np.asarray(commands, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(control_names):
        raise ValueError("commands must have one column per control name")
    control_units = units if units is not None else (None,) * len(control_names)
    control_roles = roles if roles is not None else (None,) * len(control_names)
    if len(control_units) != len(control_names) or len(control_roles) != len(control_names):
        raise ValueError("control metadata must have one value per control name")
    return [
        {
            "control": control,
            "unit": control_units[index],
            "role": control_roles[index],
            **control_metrics(values[:, index], dt),
        }
        for index, control in enumerate(control_names)
    ]
