"""Decision-oriented metrics and figures for thermal trajectory datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from celltemp.analysis import (
    control_waveform_rows,
    response_metrics,
    sensor_waveform_rows,
    thermal_case_metrics,
    uniformity_trace,
)
from celltemp.config import (
    DATA_OPTIONS,
    as_path,
    project_root_from_config,
    reject_unknown_keys,
    require_bool,
    require_path_value,
    temperature_unit_label,
    validate_config_root,
)
from celltemp.domain import Trajectory
from celltemp.io import load_system_spec

from .common import (
    input_file_records,
    load_runtime_trajectories,
    output_target,
    project_options,
    project_run_path,
    staged_output_directory,
    trajectory_source_paths,
)

_ANALYSIS_OPTIONS = {
    "controls",
    "control_convention",
    "dt",
    "final_fraction",
    "input_dir",
    "make_plots",
    "output_dir",
    "overwrite",
    "pattern",
    "response",
    "sep",
    "sensors",
    "time_col",
    "thermal_impedance",
}


@dataclass(frozen=True)
class _AnalysisChannels:
    sensors: tuple[str, ...]
    controls: tuple[str, ...]
    units: tuple[str | None, ...]
    roles: tuple[str | None, ...]
    system_path: Path | None


def _channel_names(value: object, option: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(name, str) or not name.strip() for name in value
    ):
        raise ValueError(f"{option} must be a list of non-empty channel names")
    names = tuple(value)
    if (not names and not allow_empty) or len(names) != len(set(names)):
        raise ValueError(f"{option} must contain distinct channel names")
    return names


def _analysis_channels(cfg: dict, values: dict[str, object], root: Path) -> _AnalysisChannels:
    if "system" in cfg:
        if "sensors" in values or "controls" in values:
            raise ValueError("analysis.sensors/controls cannot be combined with system")
        system_path = as_path(require_path_value(cfg["system"], "system"), root)
        system = load_system_spec(system_path)
        return _AnalysisChannels(
            system.sensor_names,
            system.control_names,
            system.control_units,
            system.control_roles,
            system_path,
        )
    sensors = _channel_names(values.get("sensors"), "analysis.sensors")
    controls = _channel_names(values.get("controls", ()), "analysis.controls", allow_empty=True)
    return _AnalysisChannels(
        sensors, controls, (None,) * len(controls), (None,) * len(controls), None
    )


def _response_intervals(value: object) -> dict[str, tuple[float, float]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("analysis.response must map case identifiers to {start_s, end_s}")
    intervals: dict[str, tuple[float, float]] = {}
    for case_id, interval in value.items():
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("analysis.response case identifiers must be non-empty strings")
        reject_unknown_keys(interval, {"start_s", "end_s"}, "analysis.response interval")
        start = _as_float(interval.get("start_s"), "analysis.response.start_s")
        end = _as_float(interval.get("end_s"), "analysis.response.end_s")
        if not np.isfinite([start, end]).all() or start >= end:
            raise ValueError("analysis.response intervals must have finite start_s < end_s")
        intervals[case_id] = (start, end)
    return intervals


def _response_rows(trajectory: Trajectory, interval: tuple[float, float]) -> list[dict]:
    """Record requested bounds and the observed origin used by response timings."""
    start, end = interval
    if start < trajectory.time[0] or end > trajectory.time[-1]:
        raise ValueError(f"{trajectory.case_id}: response interval lies outside recorded time")
    selected = (trajectory.time >= start) & (trajectory.time <= end)
    if selected.sum() < 2:
        raise ValueError(f"{trajectory.case_id}: response interval needs at least two timestamps")
    time = trajectory.time[selected]
    temperature = trajectory.temperature[selected]
    rows = sensor_waveform_rows(time, temperature, trajectory.sensor_names)
    unavailable = dict.fromkeys(
        (
            "final_temperature_std",
            "integral_change_temperature_s",
            "integral_abs_change_temperature_s",
            "time_to_63_percent_s",
            "response_time_10_90_s",
            "settling_time_s",
            "overshoot_temperature",
        ),
        float("nan"),
    )
    for index, row in enumerate(rows):
        observed_time = time[np.isfinite(temperature[:, index])]
        metrics = (
            response_metrics(time, temperature[:, index])
            if int(row["n_observed_points"]) >= 2
            else {**unavailable, "response_direction": "unknown"}
        )
        row.update(metrics)
        row.update(
            case_id=trajectory.case_id,
            start_s=start,
            end_s=end,
            observation_start_s=float(observed_time[0]) if len(observed_time) else float("nan"),
            observation_end_s=float(observed_time[-1]) if len(observed_time) else float("nan"),
        )
    return rows


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _resolved_values(cfg: dict) -> dict[str, object]:
    supplied = cfg.get("analysis", {})
    reject_unknown_keys(supplied, _ANALYSIS_OPTIONS, "analysis")
    analysis = dict(supplied)
    data = cfg.get("data", {})
    reject_unknown_keys(data, DATA_OPTIONS, "data")
    project = project_options(cfg)

    input_dir = analysis.get("input_dir", data.get("directory"))
    input_dir = require_path_value(input_dir, "analysis.input_dir or data.directory")
    run_path = project_run_path(project)
    defaults: dict[str, object] = {
        "input_dir": input_dir,
        "pattern": data.get("pattern", "*.csv"),
        "time_col": data.get("time_col", "time"),
        "control_convention": data.get("control_convention", "left"),
        "sep": data.get("sep", ","),
        "dt": data.get("dt"),
        "output_dir": run_path.with_name(f"{run_path.name}_analysis"),
        "overwrite": project.get("overwrite_run", False),
        "make_plots": True,
        "final_fraction": 0.1,
        "temperature_unit": project["temperature_unit"],
    }
    values = {**defaults, **analysis}
    if values.get("thermal_impedance") is not None:
        from .analysis_impedance import resolve_thermal_impedance

        values["thermal_impedance"] = resolve_thermal_impedance(values["thermal_impedance"])
    else:
        values["thermal_impedance"] = None
    values["response"] = _response_intervals(values.get("response"))
    values["make_plots"] = require_bool(values["make_plots"], "analysis.make_plots")
    values["overwrite"] = require_bool(values["overwrite"], "analysis.overwrite")
    for name in ("final_fraction",):
        value = _as_float(values[name], f"analysis.{name}")
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"analysis.{name} must be finite and in (0, 1]")
        values[name] = value
    return values


def _plot_case(
    target: Path,
    *,
    case_id: str,
    time: np.ndarray,
    temperature: np.ndarray,
    sensor_names: tuple[str, ...],
    commands: np.ndarray,
    control_names: tuple[str, ...],
    control_units: tuple[str | None, ...],
    sensor_span: np.ndarray,
    temperature_unit: str = "degC",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    show_uniformity = len(sensor_names) >= 2 and np.isfinite(sensor_span).any()
    row_count = 1 + int(bool(control_names)) + int(show_uniformity)
    figure, axes = plt.subplots(
        row_count,
        1,
        figsize=(10.0, 2.7 * row_count),
        sharex=True,
        constrained_layout=True,
    )
    panels = np.atleast_1d(axes).tolist()
    temperature_axis = panels.pop(0)
    for index, sensor in enumerate(sensor_names):
        temperature_axis.plot(time, temperature[:, index], linewidth=1.6, label=sensor)
    temperature_axis.set_ylabel(f"Temperature [{temperature_unit_label(temperature_unit)}]")
    temperature_axis.grid(alpha=0.25)
    temperature_axis.legend(loc="best", ncols=min(3, len(sensor_names)))
    temperature_axis.set_title(case_id)

    if control_names:
        control_axis = panels.pop(0)
        sampled_commands = np.concatenate([commands, commands[-1:]], axis=0)
        for index, control in enumerate(control_names):
            unit = control_units[index]
            label = control if unit is None else f"{control} [{unit}]"
            control_axis.step(
                time,
                sampled_commands[:, index],
                where="post",
                linewidth=1.35,
                label=label,
            )
        control_axis.set_ylabel("Command")
        control_axis.grid(alpha=0.25)
        control_axis.legend(loc="best", ncols=min(3, len(control_names)))

    if show_uniformity:
        uniformity_axis = panels.pop(0)
        uniformity_axis.plot(time, sensor_span, color="#7b2cbf", linewidth=1.6)
        uniformity_axis.set_ylabel("Sensor span [K]")
        uniformity_axis.grid(alpha=0.25)

    np.atleast_1d(axes)[-1].set_xlabel("Time (s)")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180)
    plt.close(figure)


def _peak_summary(sensor_rows: list[dict[str, object]]) -> dict[str, object] | None:
    finite = [
        row
        for row in sensor_rows
        if np.isfinite(_as_float(row["maximum_temperature"], "maximum_temperature"))
    ]
    if not finite:
        return None
    peak = max(
        finite,
        key=lambda row: _as_float(row["maximum_temperature"], "maximum_temperature"),
    )
    return {
        "case_id": peak["case_id"],
        "sensor": peak["sensor"],
        "temperature": peak["maximum_temperature"],
        "time_s": peak["time_of_maximum_s"],
    }


def _spread_summary(case_rows: list[dict[str, object]]) -> dict[str, object] | None:
    finite = [
        row
        for row in case_rows
        if np.isfinite(_as_float(row["maximum_sensor_span"], "maximum_sensor_span"))
    ]
    if not finite:
        return None
    widest = max(
        finite,
        key=lambda row: _as_float(row["maximum_sensor_span"], "maximum_sensor_span"),
    )
    return {
        "case_id": widest["case_id"],
        "sensor_span": widest["maximum_sensor_span"],
        "time_s": widest["time_of_maximum_span_s"],
    }


def run_analysis(cfg: dict, config_path: str | Path) -> Path:
    """Summarize measured or simulated thermal cases and render waveform panels."""
    validate_config_root(cfg)
    root = project_root_from_config(config_path)
    values = _resolved_values(cfg)
    channels = _analysis_channels(cfg, values, root)
    trajectories = load_runtime_trajectories(
        values,
        root,
        sensor_names=channels.sensors,
        control_names=channels.controls,
        require_initial_observation=False,
    )
    response_intervals = cast(dict[str, tuple[float, float]], values["response"])
    unknown_cases = response_intervals.keys() - {item.case_id for item in trajectories}
    if unknown_cases:
        raise ValueError(
            f"analysis.response contains unknown case identifiers {sorted(unknown_cases)}"
        )
    target, overwrite = output_target(
        {
            "output_dir": values["output_dir"],
            "overwrite": values["overwrite"],
        },
        root,
        protected_paths=(
            Path(config_path),
            as_path(str(values["input_dir"]), root),
            *((channels.system_path,) if channels.system_path is not None else ()),
            *trajectory_source_paths(trajectories),
        ),
    )
    final_fraction = _as_float(values["final_fraction"], "analysis.final_fraction")
    make_plots = cast(bool, values["make_plots"])

    case_rows: list[dict[str, object]] = []
    sensor_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    response_rows: list[dict[str, object]] = []
    with staged_output_directory(target, overwrite=overwrite) as out_dir:
        for trajectory in trajectories:
            mean, span, standard_deviation = uniformity_trace(trajectory.temperature)
            case_rows.append(
                {
                    "case_id": trajectory.case_id,
                    **thermal_case_metrics(
                        trajectory.time,
                        trajectory.temperature,
                        final_fraction=final_fraction,
                    ),
                }
            )
            sensor_rows.extend(
                {"case_id": trajectory.case_id, **row}
                for row in sensor_waveform_rows(
                    trajectory.time,
                    trajectory.temperature,
                    trajectory.sensor_names,
                )
            )
            if trajectory.case_id in response_intervals:
                response_rows.extend(
                    _response_rows(trajectory, response_intervals[trajectory.case_id])
                )
            control_rows.extend(
                {"case_id": trajectory.case_id, **row}
                for row in control_waveform_rows(
                    trajectory.commands,
                    trajectory.dt,
                    trajectory.control_names,
                    units=channels.units,
                    roles=channels.roles,
                )
            )

            uniformity_frame = pd.DataFrame(
                {
                    "time": trajectory.time,
                    "mean_temperature": mean,
                    "sensor_span": span,
                    "sensor_std": standard_deviation,
                }
            )
            uniformity_target = out_dir / "uniformity" / f"{trajectory.case_id}.csv"
            uniformity_target.parent.mkdir(parents=True, exist_ok=True)
            uniformity_frame.to_csv(uniformity_target, index=False)
            if make_plots:
                _plot_case(
                    out_dir / "figures" / f"{trajectory.case_id}.png",
                    case_id=trajectory.case_id,
                    time=trajectory.time,
                    temperature=trajectory.temperature,
                    sensor_names=trajectory.sensor_names,
                    commands=trajectory.commands,
                    control_names=trajectory.control_names,
                    control_units=channels.units,
                    sensor_span=span,
                    temperature_unit=str(values["temperature_unit"]),
                )

        pd.DataFrame(case_rows).to_csv(out_dir / "case_metrics.csv", index=False)
        pd.DataFrame(sensor_rows).to_csv(out_dir / "sensor_metrics.csv", index=False)
        pd.DataFrame(
            control_rows,
            columns=[
                "case_id",
                "control",
                "unit",
                "role",
                "minimum_command",
                "maximum_command",
                "time_weighted_mean_command",
                "command_integral_unit_s",
                "total_variation",
                "maximum_absolute_slew_per_s",
                "change_count",
            ],
        ).to_csv(out_dir / "control_metrics.csv", index=False)
        if response_intervals:
            pd.DataFrame(response_rows).to_csv(out_dir / "response_metrics.csv", index=False)
        impedance_summary = None
        if values["thermal_impedance"] is not None:
            from .analysis_impedance import write_thermal_impedance

            impedance_summary = write_thermal_impedance(
                out_dir,
                trajectories,
                cast(dict[str, object], values["thermal_impedance"]),
                make_plot=make_plots,
            )

        summary = {
            "schema_version": 1,
            "cases": len(trajectories),
            "inputs": input_file_records(trajectories, root),
            "sensors": list(channels.sensors),
            "controls": list(channels.controls),
            "units": {"time": "s", "temperature": values["temperature_unit"]},
            "response_availability": {
                "available_sensor_responses": sum(
                    row["response_status"] == "available" for row in sensor_rows
                ),
                "insufficient_sensor_responses": sum(
                    row["response_status"] == "insufficient_observations" for row in sensor_rows
                ),
            },
            "control_metadata": [
                {"name": name, "unit": unit, "role": role}
                for name, unit, role in zip(
                    channels.controls, channels.units, channels.roles, strict=True
                )
            ],
            "settings": {
                "final_fraction": final_fraction,
                "response": {
                    case_id: {"start_s": start, "end_s": end}
                    for case_id, (start, end) in response_intervals.items()
                },
                "figures": make_plots,
            },
            "highest_temperature": _peak_summary(sensor_rows),
            "largest_sensor_span": _spread_summary(case_rows),
            "thermal_impedance": impedance_summary,
            "outputs": {
                "case_metrics": "case_metrics.csv",
                "sensor_metrics": "sensor_metrics.csv",
                "response_metrics": "response_metrics.csv" if response_intervals else None,
                "control_metrics": "control_metrics.csv",
                "uniformity": "uniformity/",
                "figures": "figures/" if make_plots else None,
            },
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    return target
