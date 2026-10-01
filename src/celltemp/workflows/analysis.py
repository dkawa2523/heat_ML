"""Decision-oriented metrics and figures for thermal trajectory datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np
import pandas as pd

from celltemp.analysis import (
    control_waveform_rows,
    sensor_response_rows,
    thermal_case_metrics,
    uniformity_trace,
)
from celltemp.config import (
    as_path,
    project_root_from_config,
    reject_unknown_keys,
    require_bool,
    validate_config_root,
)
from celltemp.io import load_system_spec

from .analysis_impedance import resolve_thermal_impedance, write_thermal_impedance
from .common import (
    load_runtime_trajectories,
    output_target,
    project_options,
    project_run_path,
    staged_output_directory,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_ANALYSIS_OPTIONS = {
    "control_convention",
    "dt",
    "final_fraction",
    "input_dir",
    "make_plots",
    "output_dir",
    "overwrite",
    "pattern",
    "sep",
    "settling_fraction",
    "time_col",
    "thermal_impedance",
}


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _resolved_values(cfg: dict) -> dict[str, object]:
    supplied = cfg.get("analysis", {})
    reject_unknown_keys(supplied, _ANALYSIS_OPTIONS, "analysis")
    analysis = dict(supplied)
    data = cfg.get("data", {})
    if not isinstance(data, Mapping):
        raise ValueError("data must be a mapping")
    project = project_options(cfg)

    input_dir = analysis.get("input_dir", data.get("directory"))
    if not input_dir:
        raise ValueError("analysis.input_dir or data.directory is required")
    defaults: dict[str, object] = {
        "input_dir": input_dir,
        "pattern": data.get("pattern", "*.csv"),
        "time_col": data.get("time_col", "time"),
        "control_convention": data.get("control_convention", "left"),
        "sep": data.get("sep", ","),
        "dt": data.get("dt"),
        "output_dir": project_run_path(project) / "analysis",
        "overwrite": project.get("overwrite_run", False),
        "make_plots": True,
        "final_fraction": 0.1,
        "settling_fraction": 0.02,
    }
    values = {**defaults, **analysis}
    values["thermal_impedance"] = resolve_thermal_impedance(values.get("thermal_impedance"))
    values["make_plots"] = require_bool(values["make_plots"], "analysis.make_plots")
    values["overwrite"] = require_bool(values["overwrite"], "analysis.overwrite")
    for name in ("final_fraction", "settling_fraction"):
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
) -> None:
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
    temperature_axis.set_ylabel("Temperature")
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
        uniformity_axis.set_ylabel("Sensor span")
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
    if "system" not in cfg:
        raise ValueError("system is required for thermal analysis")
    values = _resolved_values(cfg)
    system = load_system_spec(as_path(str(cfg["system"]), root))
    trajectories = load_runtime_trajectories(
        values,
        root,
        sensor_names=system.sensor_names,
        control_names=system.control_names,
    )
    target, overwrite = output_target(
        {
            "output_dir": values["output_dir"],
            "overwrite": values["overwrite"],
        },
        root,
    )
    final_fraction = _as_float(values["final_fraction"], "analysis.final_fraction")
    settling_fraction = _as_float(values["settling_fraction"], "analysis.settling_fraction")
    make_plots = cast(bool, values["make_plots"])

    case_rows: list[dict[str, object]] = []
    sensor_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
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
                for row in sensor_response_rows(
                    trajectory.time,
                    trajectory.temperature,
                    trajectory.sensor_names,
                    final_fraction=final_fraction,
                    settling_fraction=settling_fraction,
                )
            )
            control_rows.extend(
                {"case_id": trajectory.case_id, **row}
                for row in control_waveform_rows(
                    trajectory.commands,
                    trajectory.dt,
                    trajectory.control_names,
                    units=system.control_units,
                    roles=system.control_roles,
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
                    control_units=system.control_units,
                    sensor_span=span,
                )

        pd.DataFrame(case_rows).to_csv(out_dir / "case_metrics.csv", index=False)
        pd.DataFrame(sensor_rows).to_csv(out_dir / "sensor_metrics.csv", index=False)
        pd.DataFrame(control_rows).to_csv(out_dir / "control_metrics.csv", index=False)
        impedance_summary = None
        if values["thermal_impedance"] is not None:
            impedance_summary = write_thermal_impedance(
                out_dir,
                trajectories,
                cast(dict[str, object], values["thermal_impedance"]),
                make_plot=make_plots,
            )

        summary = {
            "schema_version": 1,
            "cases": len(trajectories),
            "sensors": list(system.sensor_names),
            "controls": list(system.control_names),
            "control_metadata": [
                {"name": item.name, "unit": item.unit, "role": item.role}
                for item in system.actuators
            ],
            "settings": {
                "final_fraction": final_fraction,
                "settling_fraction": settling_fraction,
                "figures": make_plots,
            },
            "highest_temperature": _peak_summary(sensor_rows),
            "largest_sensor_span": _spread_summary(case_rows),
            "thermal_impedance": impedance_summary,
            "outputs": {
                "case_metrics": "case_metrics.csv",
                "sensor_metrics": "sensor_metrics.csv",
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
