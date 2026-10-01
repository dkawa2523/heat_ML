"""Standard tables produced from one or more open-loop forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

from celltemp.analysis import (
    control_waveform_rows,
    coverage_status,
    range_coverage,
    sensor_response_rows,
    temporal_coverage,
    thermal_case_metrics,
    uniformity_trace,
)
from celltemp.artifact import ThermalArtifact
from celltemp.domain import Trajectory
from celltemp.engine import ThermalRCModel
from celltemp.inference import ForecastResult

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_NORMAL_95 = 1.959963984540054
_COVERAGE_COLUMNS = [
    "case_id",
    "quantity",
    "name",
    "train_min",
    "train_max",
    "request_min",
    "request_max",
    "within_training_range",
]


@dataclass(frozen=True)
class ForecastCaseTables:
    forecast: pd.DataFrame
    energy: pd.DataFrame
    summary: dict[str, object]
    coverage: list[dict[str, object]]
    case_metrics: dict[str, float | int | str]
    sensor_metrics: list[dict[str, float | str]]
    control_metrics: list[dict[str, float | int | str | None]]
    sensor_names: tuple[str, ...]
    control_names: tuple[str, ...]
    control_units: tuple[str | None, ...]


def _forecast_frame(
    artifact: ThermalArtifact,
    result: ForecastResult,
    sampled_commands: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({"time": result.time})
    for index, sensor in enumerate(artifact.sensor_names):
        temperature = result.sensor_temperature[:, index]
        standard_deviation = result.sensor_temperature_std[:, index]
        frame[f"temperature_{sensor}"] = temperature
        frame[f"temperature_std_{sensor}"] = standard_deviation
        frame[f"temperature_lower_95_{sensor}"] = temperature - _NORMAL_95 * standard_deviation
        frame[f"temperature_upper_95_{sensor}"] = temperature + _NORMAL_95 * standard_deviation
    for index, node in enumerate(artifact.model.spec.node_names):
        frame[f"state_{node}"] = result.node_temperature[:, index]
        frame[f"state_std_{node}"] = result.node_temperature_std[:, index]
    for index, control in enumerate(artifact.control_names):
        frame[f"command_{control}"] = sampled_commands[:, index]
        frame[f"effective_{control}"] = result.actuator[:, index]
    mean_temperature, sensor_span, sensor_std = uniformity_trace(result.sensor_temperature)
    frame["mean_temperature"] = mean_temperature
    frame["sensor_span"] = sensor_span
    frame["sensor_std"] = sensor_std
    return frame


def _energy_balance_frame(
    model: ThermalRCModel,
    *,
    case_id: str,
    result: ForecastResult,
) -> pd.DataFrame:
    """Build one self-describing instantaneous heat-rate table in W."""
    with torch.no_grad():
        flow = model.heat_flow_breakdown(
            torch.as_tensor(
                result.node_temperature,
                dtype=model.capacity.dtype,
                device=model.capacity.device,
            ),
            torch.as_tensor(
                result.actuator,
                dtype=model.capacity.dtype,
                device=model.capacity.device,
            ),
        )
    edge_rate = flow.edge_rate.cpu().numpy()
    internal = flow.internal_to_node.cpu().numpy()
    source = flow.source_to_node.sum(dim=-1).cpu().numpy()
    boundary = flow.boundary_to_node.sum(dim=-1).cpu().numpy()
    boundary_temperature = flow.boundary_temperature.cpu().numpy()
    storage = flow.storage_rate.cpu().numpy()
    residual = flow.balance_residual.cpu().numpy()
    frame = pd.DataFrame({"case_id": case_id, "time": result.time})
    for index, edge in enumerate(model.spec.edges):
        frame[f"edge_{edge.node_a}_to_{edge.node_b}_w"] = edge_rate[:, index]
    for index, item in enumerate(model.spec.sources):
        frame[f"source_{item.name}_w"] = source[:, index]
    for index, item in enumerate(model.spec.boundaries):
        frame[f"boundary_{item.name}_w"] = boundary[:, index]
        frame[f"reservoir_temperature_{item.name}"] = boundary_temperature[:, index]
    for index, node in enumerate(model.spec.node_names):
        frame[f"internal_net_{node}_w"] = internal[:, index]
        frame[f"storage_{node}_w"] = storage[:, index]
        frame[f"balance_residual_{node}_w"] = residual[:, index]
    frame["source_total_w"] = source.sum(axis=1)
    frame["boundary_total_w"] = boundary.sum(axis=1)
    frame["storage_total_w"] = storage.sum(axis=1)
    frame["balance_residual_total_w"] = residual.sum(axis=1)
    return frame


def _plot_energy_balance(frame: pd.DataFrame, target: Path) -> None:
    """Plot the signed external heat rates, storage, and numerical residual."""
    time = frame["time"].to_numpy(dtype=np.float64)
    source = frame["source_total_w"].to_numpy(dtype=np.float64)
    boundary = frame["boundary_total_w"].to_numpy(dtype=np.float64)
    storage = frame["storage_total_w"].to_numpy(dtype=np.float64)
    residual = frame["balance_residual_total_w"].to_numpy(dtype=np.float64)
    external = source + boundary

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, 7.5),
        sharex=True,
        constrained_layout=True,
        height_ratios=(2.0, 2.0, 1.2),
    )
    exchange_axis, storage_axis, residual_axis = axes
    exchange_axis.plot(time, source, color="#d95f02", linewidth=1.6, label="Source heat")
    exchange_axis.plot(
        time,
        boundary,
        color="#1b9e77",
        linewidth=1.6,
        label="Boundary heat",
    )
    exchange_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
    exchange_axis.set_ylabel("Heat into nodes [W]")
    exchange_axis.set_title(str(frame["case_id"].iat[0]))
    exchange_axis.grid(alpha=0.25)
    exchange_axis.legend(loc="best")

    storage_axis.plot(
        time,
        external,
        color="#7570b3",
        linewidth=1.8,
        label="Source + boundary",
    )
    storage_axis.plot(
        time,
        storage,
        color="#222222",
        linestyle="--",
        linewidth=1.5,
        label="Storage",
    )
    storage_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
    storage_axis.set_ylabel("Net heat rate [W]")
    storage_axis.grid(alpha=0.25)
    storage_axis.legend(loc="best")

    residual_axis.plot(time, residual, color="#e7298a", linewidth=1.4)
    residual_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
    residual_axis.set_ylabel("Residual [W]")
    residual_axis.set_xlabel("Time [s]")
    residual_axis.grid(alpha=0.25)

    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180)
    plt.close(figure)


def _plot_forecast_case(case: ForecastCaseTables, target: Path) -> None:
    """Plot the primary engineering view directly from the forecast table."""
    frame = case.forecast
    time = frame["time"].to_numpy(dtype=np.float64)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, 8.2),
        sharex=True,
        constrained_layout=True,
        height_ratios=(2.4, 1.2, 1.0),
    )
    temperature_axis, command_axis, span_axis = axes

    for sensor in case.sensor_names:
        temperature = frame[f"temperature_{sensor}"].to_numpy(dtype=np.float64)
        lower = frame[f"temperature_lower_95_{sensor}"].to_numpy(dtype=np.float64)
        upper = frame[f"temperature_upper_95_{sensor}"].to_numpy(dtype=np.float64)
        (line,) = temperature_axis.plot(time, temperature, linewidth=1.6, label=sensor)
        temperature_axis.fill_between(
            time,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.14,
            linewidth=0.0,
        )
    temperature_axis.set_ylabel("Temperature")
    temperature_axis.set_title("Predicted temperature with 95% latent-state intervals")
    temperature_axis.grid(alpha=0.25)
    temperature_axis.legend(loc="best", ncols=min(3, len(case.sensor_names)))

    for control, unit in zip(case.control_names, case.control_units, strict=True):
        label = control if unit is None else f"{control} [{unit}]"
        command_axis.step(
            time,
            frame[f"command_{control}"].to_numpy(dtype=np.float64),
            where="post",
            linewidth=1.35,
            label=label,
        )
    command_axis.set_ylabel("Command")
    command_axis.grid(alpha=0.25)
    if case.control_names:
        command_axis.legend(loc="best", ncols=min(3, len(case.control_names)))

    span_axis.plot(
        time,
        frame["sensor_span"].to_numpy(dtype=np.float64),
        color="#7b2cbf",
        linewidth=1.6,
    )
    span_axis.set_ylabel("Sensor span [K]")
    span_axis.set_xlabel("Time [s]")
    span_axis.grid(alpha=0.25)

    figure.suptitle(str(case.summary["case_id"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180)
    plt.close(figure)


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
) -> ForecastCaseTables:
    """Build every standard table row for one forecast without filesystem IO."""
    origin = result.forecast_origin_index
    commands = request.commands[origin:]
    dt = request.dt[origin:]
    sampled_commands = np.concatenate([commands, commands[-1:]], axis=0)
    forecast_frame = _forecast_frame(artifact, result, sampled_commands)
    case_metrics = {
        "case_id": request.case_id,
        **thermal_case_metrics(result.time, result.sensor_temperature),
    }
    sensor_metrics = [
        {"case_id": request.case_id, **row}
        for row in sensor_response_rows(
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
        "rows": len(forecast_frame),
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
    return ForecastCaseTables(
        forecast=forecast_frame,
        energy=_energy_balance_frame(artifact.model, case_id=request.case_id, result=result),
        summary=summary,
        coverage=coverage,
        case_metrics=case_metrics,
        sensor_metrics=sensor_metrics,
        control_metrics=control_metrics,
        sensor_names=artifact.sensor_names,
        control_names=artifact.control_names,
        control_units=artifact.model.spec.control_units,
    )


def write_forecast_tables(target: Path, cases: list[ForecastCaseTables]) -> None:
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
    pd.concat([item.energy for item in cases], ignore_index=True).to_csv(
        target / "energy_balance.csv", index=False
    )
    pd.DataFrame(
        [row for item in cases for row in item.coverage],
        columns=_COVERAGE_COLUMNS,
    ).to_csv(target / "forecast_coverage.csv", index=False)
    for item in cases:
        case_id = str(item.summary["case_id"])
        _plot_forecast_case(
            item,
            target / "figures" / f"forecast_{case_id}.png",
        )
        _plot_energy_balance(
            item.energy,
            target / "figures" / f"energy_balance_{case_id}.png",
        )


def print_coverage_warnings(cases: list[ForecastCaseTables]) -> None:
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
