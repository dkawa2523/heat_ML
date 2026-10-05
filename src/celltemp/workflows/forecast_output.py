"""Standard tables produced from one or more open-loop forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from celltemp.analysis import (
    control_waveform_rows,
    coverage_status,
    range_coverage,
    sensor_waveform_rows,
    temporal_coverage,
    thermal_case_metrics,
    uniformity_trace,
)
from celltemp.artifact import ThermalArtifact
from celltemp.domain import Trajectory
from celltemp.inference import ForecastResult

_NORMAL_95 = 1.959963984540054
_COVERAGE_COLUMNS = [
    "case_id",
    "quantity",
    "name",
    "train_min",
    "train_max",
    "request_min",
    "request_max",
    "comparison_tolerance",
    "within_training_range",
]


@dataclass(frozen=True)
class ForecastCaseSummary:
    summary: dict[str, object]
    coverage: list[dict[str, object]]
    case_metrics: dict[str, float | int | str]
    sensor_metrics: list[dict[str, float | str]]
    control_metrics: list[dict[str, float | int | str | None]]


def forecast_frame(
    artifact: ThermalArtifact,
    request: Trajectory,
    result: ForecastResult,
) -> pd.DataFrame:
    commands = request.commands[result.forecast_origin_index :]
    sampled_commands = np.concatenate([commands, commands[-1:]], axis=0)
    frame = pd.DataFrame({"time": result.time})
    for index, sensor in enumerate(artifact.sensor_names):
        temperature = result.sensor_temperature[:, index]
        standard_deviation = result.sensor_temperature_std[:, index]
        frame[f"sensor.{sensor}.temperature"] = temperature
        frame[f"sensor.{sensor}.std"] = standard_deviation
        frame[f"sensor.{sensor}.lower95"] = temperature - _NORMAL_95 * standard_deviation
        frame[f"sensor.{sensor}.upper95"] = temperature + _NORMAL_95 * standard_deviation
    for index, node in enumerate(artifact.model.spec.node_names):
        frame[f"node.{node}.temperature"] = result.node_temperature[:, index]
        frame[f"node.{node}.std"] = result.node_temperature_std[:, index]
    for index, control in enumerate(artifact.control_names):
        frame[f"control.{control}.command"] = sampled_commands[:, index]
        frame[f"control.{control}.effective"] = result.actuator[:, index]
    mean_temperature, sensor_span, sensor_std = uniformity_trace(result.sensor_temperature)
    frame["mean_temperature"] = mean_temperature
    frame["sensor_span"] = sensor_span
    frame["sensor_std"] = sensor_std
    return frame


def _coverage(
    artifact: ThermalArtifact,
    request: Trajectory,
    result: ForecastResult,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    control = range_coverage(
        artifact.metadata,
        "train_control_ranges",
        "control",
        artifact.control_names,
        request.commands[result.forecast_origin_index :],
    )
    temperature = range_coverage(
        artifact.metadata,
        "train_temperature_ranges",
        "predicted_temperature",
        artifact.sensor_names,
        result.sensor_temperature,
    )
    temporal = temporal_coverage(artifact.metadata, request, result.forecast_origin_index)
    control_status, outside_controls = coverage_status(control)
    temperature_status, outside_sensors = coverage_status(temperature)
    temporal_status, outside_temporal = coverage_status(temporal)
    status = {
        "training_range_status": control_status,
        "out_of_range_controls": outside_controls,
        "predicted_temperature_range_status": temperature_status,
        "out_of_range_sensors": outside_sensors,
        "temporal_range_status": temporal_status,
        "out_of_range_temporal": outside_temporal,
    }
    rows = [{"case_id": request.case_id, **row} for row in [*control, *temperature, *temporal]]
    return rows, status


def build_forecast_case(
    artifact: ThermalArtifact,
    request: Trajectory,
    result: ForecastResult,
    *,
    output_name: str,
) -> ForecastCaseSummary:
    """Build every standard table row for one forecast without filesystem IO."""
    origin = result.forecast_origin_index
    commands = request.commands[origin:]
    dt = request.dt[origin:]
    case_metrics = {
        "case_id": request.case_id,
        **thermal_case_metrics(result.time, result.sensor_temperature),
    }
    sensor_metrics = [
        {"case_id": request.case_id, **row}
        for row in sensor_waveform_rows(
            result.time,
            result.sensor_temperature,
            artifact.sensor_names,
        )
    ]
    control_metrics = [
        {"case_id": request.case_id, **row}
        for row in control_waveform_rows(
            commands,
            dt,
            artifact.control_names,
            units=artifact.model.spec.control_units,
            roles=artifact.model.spec.control_roles,
        )
    ]
    coverage, status = _coverage(artifact, request, result)
    peak_flat_index = int(np.nanargmax(result.sensor_temperature))
    peak_time_index, peak_sensor_index = np.unravel_index(
        peak_flat_index, result.sensor_temperature.shape
    )
    summary: dict[str, object] = {
        "case_id": request.case_id,
        "history_rows": origin + 1,
        "forecast_start_time": float(result.time[0]),
        "rows": len(result.time),
        "time_end": float(result.time[-1]),
        "initial_actuator_source": (
            "csv" if request.initial_actuator is not None else "first_command"
        ),
        **status,
        "maximum_predicted_temperature": float(
            result.sensor_temperature[peak_time_index, peak_sensor_index]
        ),
        "maximum_temperature_sensor": artifact.sensor_names[peak_sensor_index],
        "time_of_maximum_temperature_s": float(result.time[peak_time_index]),
        "maximum_sensor_span": case_metrics["maximum_sensor_span"],
        "output": output_name,
    }
    return ForecastCaseSummary(
        summary=summary,
        coverage=coverage,
        case_metrics=case_metrics,
        sensor_metrics=sensor_metrics,
        control_metrics=control_metrics,
    )


def write_forecast_tables(target: Path, cases: list[ForecastCaseSummary]) -> None:
    """Write dataset-level tables assembled from completed case forecasts."""
    pd.DataFrame([item.summary for item in cases]).to_csv(
        target / "forecast_summary.csv", index=False
    )
    pd.DataFrame([item.case_metrics for item in cases]).to_csv(
        target / "forecast_case_metrics.csv", index=False
    )
    pd.DataFrame([row for item in cases for row in item.sensor_metrics]).to_csv(
        target / "forecast_sensor_metrics.csv", index=False
    )
    pd.DataFrame([row for item in cases for row in item.control_metrics]).to_csv(
        target / "forecast_control_metrics.csv", index=False
    )
    pd.DataFrame(
        [row for item in cases for row in item.coverage],
        columns=_COVERAGE_COLUMNS,
    ).to_csv(target / "forecast_coverage.csv", index=False)


def print_coverage_warnings(cases: list[ForecastCaseSummary]) -> None:
    """Print one compact warning for each outside-envelope dimension."""
    checks = (
        (
            "training_range_status",
            "out_of_range_controls",
            "forecast commands outside training ranges",
        ),
        (
            "predicted_temperature_range_status",
            "out_of_range_sensors",
            "forecast temperatures outside training ranges",
        ),
        (
            "temporal_range_status",
            "out_of_range_temporal",
            "forecast time patterns outside training ranges",
        ),
    )
    for status_name, detail_name, message in checks:
        outside = [
            f"{item.summary['case_id']}[{item.summary[detail_name]}]"
            for item in cases
            if item.summary[status_name] == "outside"
        ]
        if outside:
            print(f"WARNING: {message}: " + ", ".join(outside))
