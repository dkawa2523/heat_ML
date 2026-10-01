"""Configuration and CSV output for the optional power-step analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from celltemp.analysis import power_step_thermal_impedance
from celltemp.config import reject_unknown_keys
from celltemp.domain import Trajectory

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_OPTIONS = {
    "baseline_window_s",
    "max_baseline_drift_k_per_s",
    "max_baseline_std_k",
    "max_terminal_zth_drift_k_per_w_s",
    "steps",
    "terminal_window_s",
}
_STEP_OPTIONS = {
    "control",
    "heat_step_w",
    "transition_end_s",
    "transition_start_s",
}
_IMPEDANCE_COLUMNS = [
    "case_id",
    "sensor",
    "time_s",
    "time_since_step_s",
    "delta_temperature_k",
    "heat_step_w",
    "thermal_impedance_k_per_w",
]


def _plot_thermal_impedance(frame: pd.DataFrame, target: Path) -> None:
    sensors = frame["sensor"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(sensors),
        1,
        figsize=(10.0, 3.0 * len(sensors)),
        sharex=True,
        constrained_layout=True,
    )
    panels = np.atleast_1d(axes).tolist()
    for axis, sensor in zip(panels, sensors):
        sensor_frame = frame.loc[frame["sensor"] == sensor]
        for case_id, case_frame in sensor_frame.groupby("case_id", sort=False):
            positive_time = case_frame["time_since_step_s"] > 0.0
            axis.semilogx(
                case_frame.loc[positive_time, "time_since_step_s"],
                case_frame.loc[positive_time, "thermal_impedance_k_per_w"],
                linewidth=1.5,
                label=str(case_id),
            )
        axis.set_ylabel(f"{sensor}\nZth [K/W]")
        axis.grid(alpha=0.25, which="both")
        axis.legend(loc="best")
    panels[-1].set_xlabel("Time since heat-step start [s]")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180)
    plt.close(figure)


def _number(values: Mapping[str, object], name: str, section: str) -> float:
    if name not in values:
        raise ValueError(f"{section}.{name} is required")
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{section}.{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{section}.{name} must be finite")
    return result


def _global_settings(value: object, section: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a mapping")
    reject_unknown_keys(value, _OPTIONS, section)
    settings = dict(value)

    for name in ("baseline_window_s", "terminal_window_s"):
        resolved = _number(settings, name, section)
        if resolved <= 0.0:
            raise ValueError(f"{section}.{name} must be positive")
        settings[name] = resolved
    for name in (
        "max_baseline_drift_k_per_s",
        "max_baseline_std_k",
        "max_terminal_zth_drift_k_per_w_s",
    ):
        resolved = _number(settings, name, section)
        if resolved < 0.0:
            raise ValueError(f"{section}.{name} must be non-negative")
        settings[name] = resolved
    return settings


def _step_settings(steps: object, section: str) -> dict[str, dict[str, object]]:
    if not isinstance(steps, Mapping) or not steps:
        raise ValueError(f"{section}.steps must be a non-empty mapping keyed by case_id")
    normalized_steps: dict[str, dict[str, object]] = {}
    for raw_case_id, raw_step in steps.items():
        case_id = str(raw_case_id)
        step_section = f"{section}.steps.{case_id}"
        if not case_id.strip():
            raise ValueError(f"{section}.steps case_id must not be empty")
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"{step_section} must be a mapping")
        reject_unknown_keys(raw_step, _STEP_OPTIONS, step_section)
        step = dict(raw_step)
        control = step.get("control")
        if not isinstance(control, str) or not control.strip():
            raise ValueError(f"{step_section}.control must be a non-empty string")
        step["control"] = control
        for name in ("transition_start_s", "transition_end_s", "heat_step_w"):
            step[name] = _number(step, name, step_section)
        if float(step["transition_start_s"]) > float(step["transition_end_s"]):
            raise ValueError(f"{step_section}.transition_start_s must not exceed transition_end_s")
        if float(step["heat_step_w"]) == 0.0:
            raise ValueError(f"{step_section}.heat_step_w must be nonzero")
        normalized_steps[case_id] = step
    return normalized_steps


def resolve_thermal_impedance(value: object) -> dict[str, Any] | None:
    """Validate the small explicit schema without adding a second config framework."""
    if value is None:
        return None
    section = "analysis.thermal_impedance"
    settings = _global_settings(value, section)
    normalized_steps = _step_settings(settings.get("steps"), section)
    settings["steps"] = normalized_steps
    return settings


def write_thermal_impedance(
    out_dir: Path,
    trajectories: Sequence[Trajectory],
    settings: Mapping[str, Any],
    *,
    make_plot: bool,
) -> dict[str, object]:
    """Analyze only configured cases and write the two compact evidence tables."""
    by_case = {trajectory.case_id: trajectory for trajectory in trajectories}
    if len(by_case) != len(trajectories):
        raise ValueError("analysis input contains duplicate case_id values")

    impedance_rows: list[dict[str, object]] = []
    qualification_rows: list[dict[str, object]] = []
    steps = settings["steps"]
    for case_id, step in steps.items():
        if case_id not in by_case:
            raise ValueError(f"thermal impedance case not found in analysis input: {case_id}")
        trajectory = by_case[case_id]
        rows, qualifications = power_step_thermal_impedance(
            trajectory.time,
            trajectory.temperature,
            trajectory.sensor_names,
            trajectory.commands,
            trajectory.control_names,
            case_id=case_id,
            power_control=str(step["control"]),
            transition_start_s=float(step["transition_start_s"]),
            transition_end_s=float(step["transition_end_s"]),
            heat_step_w=float(step["heat_step_w"]),
            baseline_window_s=float(settings["baseline_window_s"]),
            terminal_window_s=float(settings["terminal_window_s"]),
            max_baseline_drift_k_per_s=float(settings["max_baseline_drift_k_per_s"]),
            max_baseline_std_k=float(settings["max_baseline_std_k"]),
            max_terminal_zth_drift_k_per_w_s=float(settings["max_terminal_zth_drift_k_per_w_s"]),
        )
        impedance_rows.extend(rows)
        qualification_rows.extend(qualifications)

    impedance_frame = pd.DataFrame(impedance_rows, columns=_IMPEDANCE_COLUMNS)
    impedance_frame.to_csv(out_dir / "thermal_impedance.csv", index=False)
    pd.DataFrame(qualification_rows).to_csv(
        out_dir / "thermal_impedance_qualification.csv", index=False
    )
    figure_path = None
    if make_plot and not impedance_frame.empty:
        figure_path = "figures/thermal_impedance.png"
        _plot_thermal_impedance(impedance_frame, out_dir / figure_path)
    return {
        "configured_cases": len(steps),
        "zth_qualified_sensor_responses": sum(
            row["zth_qualified"] is True for row in qualification_rows
        ),
        "rth_qualified_sensor_responses": sum(
            row["rth_qualified"] is True for row in qualification_rows
        ),
        "qualification_limits": {
            "minimum_window_samples": qualification_rows[0]["minimum_window_samples"],
            "baseline_window_s": settings["baseline_window_s"],
            "terminal_window_s": settings["terminal_window_s"],
            "max_baseline_drift_k_per_s": settings["max_baseline_drift_k_per_s"],
            "max_baseline_std_k": settings["max_baseline_std_k"],
            "max_terminal_zth_drift_k_per_w_s": settings["max_terminal_zth_drift_k_per_w_s"],
        },
        "outputs": {
            "thermal_impedance": "thermal_impedance.csv",
            "thermal_impedance_qualification": "thermal_impedance_qualification.csv",
            "thermal_impedance_figure": figure_path,
        },
    }
