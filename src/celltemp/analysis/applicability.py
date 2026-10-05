"""Model-independent checks against an artifact's training envelope."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from celltemp.domain import Trajectory


def _numeric_range(value: object) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    lower, upper = value
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    bounds = float(lower), float(upper)
    return bounds if np.isfinite(bounds).all() and bounds[0] <= bounds[1] else None


def _range_row(
    saved: object,
    quantity: str,
    name: str,
    values: np.ndarray,
) -> dict[str, object]:
    saved_range = _numeric_range(saved)
    train_min = saved_range[0] if saved_range is not None else None
    train_max = saved_range[1] if saved_range is not None else None
    request_min = float(values.min())
    request_max = float(values.max())
    # The engine can land a few ulps outside a boundary after matrix exponentials.
    # This is numerical roundoff, not a physical tolerance or an expanded envelope.
    tolerance = (
        float(128.0 * np.finfo(np.float64).eps * max(1.0, abs(train_min), abs(train_max)))
        if train_min is not None and train_max is not None
        else None
    )
    within = (
        request_min >= train_min - tolerance and request_max <= train_max + tolerance
        if train_min is not None and train_max is not None and tolerance is not None
        else None
    )
    return {
        "quantity": quantity,
        "name": name,
        "train_min": train_min,
        "train_max": train_max,
        "request_min": request_min,
        "request_max": request_max,
        "comparison_tolerance": tolerance,
        "within_training_range": within,
    }


def range_coverage(
    metadata: Mapping[str, object],
    metadata_key: str,
    quantity: str,
    names: tuple[str, ...],
    values: np.ndarray,
) -> list[dict[str, object]]:
    """Compare named request values with ranges saved by model training."""
    configured = metadata.get(metadata_key)
    ranges: Mapping[object, object] = configured if isinstance(configured, Mapping) else {}
    return [
        _range_row(ranges.get(name), quantity, name, values[:, index])
        for index, name in enumerate(names)
    ]


def temporal_coverage(
    metadata: Mapping[str, object],
    request: Trajectory,
    origin: int,
) -> list[dict[str, object]]:
    """Compare forecast timestep, horizon, and control slew with training data."""
    configured = metadata.get("train_temporal_ranges")
    ranges: Mapping[object, object] = configured if isinstance(configured, Mapping) else {}
    rows = [
        _range_row(
            ranges.get("time_step_seconds"),
            "time_step_seconds",
            "dt",
            request.dt[origin:],
        ),
        _range_row(
            ranges.get("forecast_horizon_seconds"),
            "forecast_horizon_seconds",
            "horizon",
            np.asarray([request.time[-1] - request.time[origin]]),
        ),
    ]
    saved_slew = ranges.get("control_slew_per_second")
    slew_ranges: Mapping[object, object] = saved_slew if isinstance(saved_slew, Mapping) else {}
    command_rates = (
        np.abs(np.diff(request.commands, axis=0) / request.dt[:-1, None])
        if len(request.commands) > 1
        else np.empty((0, len(request.control_names)))
    )
    first_relevant_transition = max(origin - 1, 0)
    for index, name in enumerate(request.control_names):
        slew = (
            command_rates[first_relevant_transition:, index]
            if len(command_rates) > first_relevant_transition
            else np.asarray([0.0])
        )
        rows.append(
            _range_row(
                slew_ranges.get(name),
                "control_slew_per_second",
                name,
                slew,
            )
        )
    return rows


def coverage_status(rows: list[dict[str, object]]) -> tuple[str, str]:
    """Summarize coverage rows as inside, outside, unknown, or not applicable."""
    if not rows:
        return "not_applicable", ""
    if any(row["within_training_range"] is None for row in rows):
        return "unknown", ""
    outside = [str(row["name"]) for row in rows if not row["within_training_range"]]
    return ("outside" if outside else "inside"), ";".join(outside)
