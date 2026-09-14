"""Open-loop forecast workflow for self-contained request CSVs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from celltemp.artifact import load_artifact
from celltemp.config import project_root_from_config, validate_config_root
from celltemp.domain import Trajectory
from celltemp.inference import build_observer, forecast, resolve_observer_settings

from .common import (
    load_runtime_trajectories,
    output_target,
    resolve_artifact_path,
    staged_output_directory,
    validate_runtime_options,
    write_runtime_manifest,
)

_NORMAL_95 = 1.959963984540054


def _range_coverage(
    metadata: Mapping[str, object],
    metadata_key: str,
    quantity: str,
    names: tuple[str, ...],
    values: np.ndarray,
) -> list[dict[str, object]]:
    """Compare forecast inputs or outputs with an artifact training envelope."""
    configured = metadata.get(metadata_key)
    ranges: Mapping[object, object] = configured if isinstance(configured, Mapping) else {}
    return [
        _single_range_coverage(ranges.get(name), quantity, name, values[:, index])
        for index, name in enumerate(names)
    ]


def _single_range_coverage(
    saved: object,
    quantity: str,
    name: str,
    values: np.ndarray,
) -> dict[str, object]:
    known = (
        isinstance(saved, Sequence)
        and not isinstance(saved, (str, bytes))
        and len(saved) == 2
        and saved[0] is not None
        and saved[1] is not None
    )
    saved_range = saved if known and isinstance(saved, Sequence) else ()
    train_min = float(saved_range[0]) if known else None
    train_max = float(saved_range[1]) if known else None
    request_min = float(values.min())
    request_max = float(values.max())
    within = (
        request_min >= train_min and request_max <= train_max
        if train_min is not None and train_max is not None
        else None
    )
    return {
        "quantity": quantity,
        "name": name,
        "train_min": train_min,
        "train_max": train_max,
        "request_min": request_min,
        "request_max": request_max,
        "within_training_range": within,
    }


def _temporal_coverage(
    metadata: Mapping[str, object],
    request: Trajectory,
    origin: int,
) -> list[dict[str, object]]:
    configured = metadata.get("train_temporal_ranges")
    ranges: Mapping[object, object] = configured if isinstance(configured, Mapping) else {}
    rows = [
        _single_range_coverage(
            ranges.get("time_step_seconds"),
            "time_step_seconds",
            "dt",
            request.dt[origin:],
        ),
        _single_range_coverage(
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
            _single_range_coverage(
                slew_ranges.get(name),
                "control_slew_per_second",
                name,
                slew,
            )
        )
    return rows


def _coverage_status(rows: list[dict[str, object]]) -> tuple[str, str]:
    if not rows:
        return "not_applicable", ""
    if any(row["within_training_range"] is None for row in rows):
        return "unknown", ""
    outside = [str(row["name"]) for row in rows if not row["within_training_range"]]
    return ("outside" if outside else "inside"), ";".join(outside)


def run_forecast(cfg: dict, config_path: str | Path) -> Path:
    validate_config_root(cfg)
    root = project_root_from_config(config_path)
    values = cfg["forecast"]
    validate_runtime_options(values, "forecast", observer=True)
    artifact = load_artifact(resolve_artifact_path(cfg, root), device=values.get("device", "cpu"))
    requests = load_runtime_trajectories(
        values,
        root,
        sensor_names=artifact.sensor_names,
        control_names=artifact.control_names,
    )

    target, overwrite = output_target(cfg, root, section="forecast")
    resolved_observer = resolve_observer_settings("forecast", values.get("observer"))
    state_estimator = build_observer(artifact.model, resolved_observer)
    summaries: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    with staged_output_directory(target, overwrite=overwrite) as out_dir:
        for request in requests:
            result = forecast(artifact.model, request, observer=state_estimator)
            frame = pd.DataFrame({"time": result.time})
            for sensor_index, sensor in enumerate(artifact.sensor_names):
                frame[f"temperature_{sensor}"] = result.sensor_temperature[:, sensor_index]
                standard_deviation = result.sensor_temperature_std[:, sensor_index]
                frame[f"temperature_std_{sensor}"] = standard_deviation
                frame[f"temperature_lower_95_{sensor}"] = (
                    result.sensor_temperature[:, sensor_index] - _NORMAL_95 * standard_deviation
                )
                frame[f"temperature_upper_95_{sensor}"] = (
                    result.sensor_temperature[:, sensor_index] + _NORMAL_95 * standard_deviation
                )
            for node_index, node in enumerate(artifact.model.spec.node_names):
                frame[f"state_{node}"] = result.node_temperature[:, node_index]
                frame[f"state_std_{node}"] = result.node_temperature_std[:, node_index]
            for control_index, control in enumerate(artifact.control_names):
                frame[f"effective_{control}"] = result.actuator[:, control_index]
            output_file = out_dir / f"{request.case_id}.csv"
            frame.to_csv(output_file, index=False)
            control_coverage = _range_coverage(
                artifact.metadata,
                "train_control_ranges",
                "control",
                artifact.control_names,
                request.commands[result.forecast_origin_index :],
            )
            temperature_coverage = _range_coverage(
                artifact.metadata,
                "train_temperature_ranges",
                "predicted_temperature",
                artifact.sensor_names,
                result.sensor_temperature,
            )
            temporal_coverage = _temporal_coverage(
                artifact.metadata, request, result.forecast_origin_index
            )
            for row in [*control_coverage, *temperature_coverage, *temporal_coverage]:
                coverage_rows.append({"case_id": request.case_id, **row})
            coverage_status, outside_controls = _coverage_status(control_coverage)
            temperature_status, outside_sensors = _coverage_status(temperature_coverage)
            temporal_status, outside_temporal = _coverage_status(temporal_coverage)
            summaries.append(
                {
                    "case_id": request.case_id,
                    "history_rows": result.forecast_origin_index + 1,
                    "forecast_start_time": float(result.time[0]),
                    "rows": len(frame),
                    "time_end": float(result.time[-1]),
                    "initial_actuator_source": (
                        "csv" if request.initial_actuator is not None else "first_command"
                    ),
                    "training_range_status": coverage_status,
                    "out_of_range_controls": outside_controls,
                    "predicted_temperature_range_status": temperature_status,
                    "out_of_range_sensors": outside_sensors,
                    "temporal_range_status": temporal_status,
                    "out_of_range_temporal": outside_temporal,
                    "output": output_file.name,
                }
            )
        pd.DataFrame(summaries).to_csv(out_dir / "forecast_summary.csv", index=False)
        pd.DataFrame(
            coverage_rows,
            columns=[
                "case_id",
                "quantity",
                "name",
                "train_min",
                "train_max",
                "request_min",
                "request_max",
                "within_training_range",
            ],
        ).to_csv(out_dir / "forecast_coverage.csv", index=False)
        write_runtime_manifest(
            out_dir / "run_manifest.json",
            workflow="forecast",
            config_path=config_path,
            root=root,
            artifact=artifact,
            values=values,
            trajectories=requests,
            observer_settings=resolved_observer,
            disturbance_basis=state_estimator.disturbance_basis,
            uncertainty={
                "confidence": 0.95,
                "scope": "latent_state_and_process_only",
                "excludes": ["measurement", "parameter", "input", "model_form"],
            },
        )
    outside = [
        f"{row['case_id']}[{row['out_of_range_controls']}]"
        for row in summaries
        if row["training_range_status"] == "outside"
    ]
    if outside:
        print("WARNING: forecast commands outside training ranges: " + ", ".join(outside))
    temperature_outside = [
        f"{row['case_id']}[{row['out_of_range_sensors']}]"
        for row in summaries
        if row["predicted_temperature_range_status"] == "outside"
    ]
    if temperature_outside:
        print(
            "WARNING: forecast temperatures outside training ranges: "
            + ", ".join(temperature_outside)
        )
    temporal_outside = [
        f"{row['case_id']}[{row['out_of_range_temporal']}]"
        for row in summaries
        if row["temporal_range_status"] == "outside"
    ]
    if temporal_outside:
        print(
            "WARNING: forecast time patterns outside training ranges: "
            + ", ".join(temporal_outside)
        )
    return target
