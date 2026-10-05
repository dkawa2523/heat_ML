"""Monitor tables for the linear COMSOL causal evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from celltemp.analysis import max_abs_error as _max_abs
from celltemp.analysis import prediction_error_metrics
from celltemp.analysis import rmse as _rmse
from celltemp.artifact import load_artifact
from celltemp.config import as_path, load_config
from celltemp.inference import build_observer, resolve_observer_settings, sensor_bias_in_gauge
from celltemp.workflows.common import (
    load_runtime_trajectories,
    resolve_artifact_path,
    resolve_runtime_values,
    validate_runtime_manifest,
)
from external_tools.comsol_chip_cooling.evaluation_support import assert_aligned

SENSORS = ("chip", "sink_base", "fins")
CONTROLS = ("chip_power", "coolant_temperature")
NIS_CONFIDENCE = 0.9999
NIS_THRESHOLDS = {1: 15.1367, 2: 18.4207, 3: 21.1075}


def _bias_reference(result: pd.DataFrame) -> str | None:
    gauges = result["bias_gauge"].dropna().unique()
    if len(gauges) != 1:
        raise ValueError("monitor output must contain one bias gauge")
    gauge = str(gauges[0])
    if gauge == "zero_mean":
        return None
    prefix = "reference:"
    if not gauge.startswith(prefix):
        raise ValueError(f"unknown monitor bias gauge: {gauge}")
    return gauge.removeprefix(prefix)


def _truth_sensor_bias(source: pd.DataFrame, result: pd.DataFrame) -> np.ndarray:
    bias = source[[f"truth_bias_{sensor}" for sensor in SENSORS]].to_numpy(dtype=np.float64)
    return sensor_bias_in_gauge(bias, SENSORS, _bias_reference(result))


def _nis_threshold(dof: np.ndarray) -> np.ndarray:
    threshold = np.full(np.asarray(dof).shape, np.nan, dtype=np.float64)
    for dimension, value in NIS_THRESHOLDS.items():
        threshold[dof == dimension] = value
    return threshold


def _validate_predictions(case_id: str, source: pd.DataFrame, result: pd.DataFrame) -> None:
    """Reject failed saved estimates while retaining deliberately absent diagnostics."""
    truth = source[[f"truth_{sensor}" for sensor in SENSORS]].to_numpy(dtype=np.float64)
    measured = result[[f"sensor.{sensor}.measured" for sensor in SENSORS]].to_numpy(
        dtype=np.float64
    )
    expected = source[list(SENSORS)].to_numpy(dtype=np.float64)
    if not np.allclose(measured, expected, rtol=0.0, atol=1e-9, equal_nan=True):
        raise ValueError(f"{case_id}: saved monitor measurements differ from the source")
    for prefix, start in (
        ("prior_physical", 1),
        ("predicted_measurement", 1),
        ("posterior_physical", 0),
        ("reconstructed_measurement", 0),
    ):
        predicted = result[[f"sensor.{sensor}.{prefix}" for sensor in SENSORS]].to_numpy(
            dtype=np.float64
        )
        prediction_error_metrics(truth[start:], predicted[start:])
    estimates = [
        column
        for column in result
        if str(column).startswith(("node.", "control."))
        or (str(column).startswith("sensor.") and str(column).endswith(".bias"))
    ]
    if not np.isfinite(result[estimates].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{case_id}: saved monitor estimates must all be finite")
    observed = np.isfinite(expected)
    observed[0] = False
    for prefix in ("innovation", "innovation_std"):
        diagnostics = result[[f"sensor.{sensor}.{prefix}" for sensor in SENSORS]].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(diagnostics[observed]).all():
            raise ValueError(f"{case_id}: {prefix} must be finite at observed measurements")
    active_nis = result["nis_dof"].to_numpy(dtype=np.int64) > 0
    if not np.isfinite(result.loc[active_nis, "nis"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{case_id}: NIS must be finite at observed measurements")


def _sensor_rows(case_id: str, source: pd.DataFrame, result: pd.DataFrame) -> list[dict[str, Any]]:
    truth_sensor_bias = _truth_sensor_bias(source, result)
    rows: list[dict[str, Any]] = []
    for index, sensor in enumerate(SENSORS):
        truth = source[f"truth_{sensor}"].to_numpy(dtype=np.float64)
        measured = result[f"sensor.{sensor}.measured"].to_numpy(dtype=np.float64)
        prior_physical = result[f"sensor.{sensor}.prior_physical"].to_numpy(dtype=np.float64)
        predicted_measurement = result[f"sensor.{sensor}.predicted_measurement"].to_numpy(
            dtype=np.float64
        )
        posterior_physical = result[f"sensor.{sensor}.posterior_physical"].to_numpy(
            dtype=np.float64
        )
        reconstructed = result[f"sensor.{sensor}.reconstructed_measurement"].to_numpy(
            dtype=np.float64
        )
        sensor_bias = result[f"sensor.{sensor}.bias"].to_numpy(dtype=np.float64)
        innovation = result[f"sensor.{sensor}.innovation"].to_numpy(dtype=np.float64)
        innovation_std = result[f"sensor.{sensor}.innovation_std"].to_numpy(dtype=np.float64)
        observed = np.isfinite(measured)
        observed[0] = False
        missing = ~np.isfinite(measured)
        rows.append(
            {
                "case_id": case_id,
                "sensor": sensor,
                "measurement_rmse_k": _rmse(measured[observed] - truth[observed]),
                "prior_physical_rmse_k": _rmse(prior_physical[1:] - truth[1:]),
                "predicted_measurement_rmse_k": _rmse(
                    predicted_measurement[observed] - measured[observed]
                ),
                "posterior_physical_rmse_k": _rmse(posterior_physical[1:] - truth[1:]),
                "reconstruction_rmse_k": _rmse(reconstructed[observed] - measured[observed]),
                "sensor_bias_rmse_k": _rmse(sensor_bias[1:] - truth_sensor_bias[1:, index]),
                "final_sensor_bias_error_k": float(sensor_bias[-1] - truth_sensor_bias[-1, index]),
                "missing_posterior_rmse_k": _rmse(posterior_physical[missing] - truth[missing]),
                "max_abs_marginal_innovation": _max_abs(innovation[1:] / innovation_std[1:]),
            }
        )
    return rows


def _case_row(case_id: str, source: pd.DataFrame, result: pd.DataFrame) -> dict[str, Any]:
    truth = source[[f"truth_{sensor}" for sensor in SENSORS]].to_numpy(dtype=np.float64)
    measured = result[[f"sensor.{sensor}.measured" for sensor in SENSORS]].to_numpy(
        dtype=np.float64
    )
    prior_physical = result[[f"sensor.{sensor}.prior_physical" for sensor in SENSORS]].to_numpy(
        dtype=np.float64
    )
    posterior_physical = result[
        [f"sensor.{sensor}.posterior_physical" for sensor in SENSORS]
    ].to_numpy(dtype=np.float64)
    reconstructed = result[
        [f"sensor.{sensor}.reconstructed_measurement" for sensor in SENSORS]
    ].to_numpy(dtype=np.float64)
    sensor_bias = result[[f"sensor.{sensor}.bias" for sensor in SENSORS]].to_numpy(dtype=np.float64)
    truth_sensor_bias = _truth_sensor_bias(source, result)
    observed = np.isfinite(measured)
    observed[0] = False
    missing = ~np.isfinite(measured)
    nis = result["nis"].to_numpy(dtype=np.float64)
    dof = result["nis_dof"].to_numpy(dtype=np.int64)
    threshold = _nis_threshold(dof)
    valid_nis = np.isfinite(nis) & np.isfinite(threshold)
    alert = valid_nis & (nis >= threshold)
    hidden = source["truth_hidden_power"].to_numpy(dtype=np.float64) > 0.0
    chip_disturbance = result["node.chip.disturbance_w"].to_numpy(dtype=np.float64)
    return {
        "case_id": case_id,
        "bias_gauge": str(result["bias_gauge"].iat[0]),
        "n_rows": len(source),
        "missing_measurements": int(missing.sum()),
        "measurement_rmse_k": _rmse((measured - truth)[observed]),
        "prior_physical_rmse_k": _rmse(prior_physical[1:] - truth[1:]),
        "posterior_physical_rmse_k": _rmse(posterior_physical[1:] - truth[1:]),
        "reconstruction_rmse_k": _rmse((reconstructed - measured)[observed]),
        "sensor_bias_rmse_k": _rmse(sensor_bias[1:] - truth_sensor_bias[1:]),
        "max_final_sensor_bias_error_k": _max_abs(sensor_bias[-1] - truth_sensor_bias[-1]),
        "missing_posterior_rmse_k": _rmse((posterior_physical - truth)[missing]),
        "final_chip_disturbance_w": float(chip_disturbance[-1]),
        "peak_event_chip_disturbance_w": (
            float(np.max(chip_disturbance[hidden])) if np.any(hidden) else float("nan")
        ),
        "max_event_abs_sensor_bias_k": (
            _max_abs(sensor_bias[hidden]) if np.any(hidden) else float("nan")
        ),
        "max_nis": float(np.nanmax(nis[1:])),
        "mean_nis_per_dof": float(np.mean(nis[valid_nis] / dof[valid_nis])),
        "alert_timepoints": int(alert[1:].sum()),
        "alert_rate": float(alert[1:].mean()),
        "all_posterior_finite": bool(np.isfinite(posterior_physical).all()),
    }


def _event_metrics(
    case_id: str, source: pd.DataFrame, result: pd.DataFrame
) -> dict[str, Any] | None:
    if case_id == "M03_sensor_offset":
        target = source["truth_bias_chip"].to_numpy(dtype=np.float64) != 0.0
        event_kind = "chip_sensor_offset"
    elif case_id == "M05_uncommanded_heat":
        target = source["truth_hidden_power"].to_numpy(dtype=np.float64) > 0.0
        event_kind = "uncommanded_chip_heat"
    else:
        return None

    nis = result["nis"].to_numpy(dtype=np.float64)
    threshold = _nis_threshold(result["nis_dof"].to_numpy(dtype=np.int64))
    alert = np.isfinite(nis) & np.isfinite(threshold) & (nis >= threshold)
    event_indices = np.flatnonzero(target)
    event_start = int(event_indices[0])
    event_end = int(event_indices[-1])
    detected_indices = np.flatnonzero(
        alert & (np.arange(len(source)) >= event_start) & (np.arange(len(source)) <= event_end)
    )
    detected_index = int(detected_indices[0]) if detected_indices.size else None
    time = source["time"].to_numpy(dtype=np.float64)
    sensor_bias = result[[f"sensor.{sensor}.bias" for sensor in SENSORS]].to_numpy(dtype=np.float64)
    disturbance = result["node.chip.disturbance_w"].to_numpy(dtype=np.float64)
    return {
        "case_id": case_id,
        "event": event_kind,
        "nis_confidence": NIS_CONFIDENCE,
        "event_start_s": float(time[event_start]),
        "event_end_s": float(time[event_end]),
        "detected": detected_index is not None,
        "first_detection_s": float(time[detected_index]) if detected_index is not None else None,
        "detection_delay_s": (
            float(time[detected_index] - time[event_start]) if detected_index is not None else None
        ),
        "false_alerts_before_event": int(alert[1:event_start].sum()),
        "peak_event_nis": float(np.nanmax(nis[event_start : event_end + 1])),
        "peak_chip_disturbance_w": float(np.max(disturbance[event_start : event_end + 1])),
        "max_abs_sensor_bias_k": _max_abs(sensor_bias[event_start : event_end + 1]),
    }


def evaluate_monitoring(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate causal monitor outputs by case, sensor, and declared event."""
    config_path = root / "config.yaml"
    cfg = load_config(config_path)
    artifact = load_artifact(resolve_artifact_path(cfg, root))
    values = resolve_runtime_values(cfg, "monitor", artifact_metadata=artifact.metadata)
    result_dir = as_path(str(values["output_dir"]), root)
    trajectories = load_runtime_trajectories(
        values, root, sensor_names=artifact.sensor_names, control_names=artifact.control_names
    )
    settings = resolve_observer_settings("monitor", values.get("observer"))
    observer = build_observer(artifact.model, settings)
    validate_runtime_manifest(
        result_dir / "run_manifest.json",
        workflow="monitor",
        artifact=artifact,
        values=values,
        trajectories=trajectories,
        observer_settings=settings,
        disturbance_basis=observer.disturbance_basis,
    )
    case_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        source_path = Path(str(trajectory.metadata["path"]))
        case_id = trajectory.case_id
        source = pd.read_csv(source_path, sep=values["sep"])
        result = pd.read_csv(result_dir / "cases" / source_path.name)
        assert_aligned(source, result, case_id)
        reference = settings["bias_reference"]
        expected_gauge = "zero_mean" if reference is None else f"reference:{reference}"
        if not result["bias_gauge"].eq(expected_gauge).all():
            raise ValueError(f"{case_id}: saved monitor bias gauge differs from the settings")
        saved_commands = result[
            [f"control.{name}.command" for name in artifact.control_names]
        ].to_numpy(dtype=np.float64)
        expected_commands = np.concatenate([trajectory.commands, trajectory.commands[-1:]], axis=0)
        if not np.allclose(saved_commands, expected_commands, rtol=1e-12, atol=1e-12):
            raise ValueError(f"{case_id}: saved monitor commands differ from the source")
        _validate_predictions(case_id, source, result)
        case_rows.append(_case_row(case_id, source, result))
        sensor_rows.extend(_sensor_rows(case_id, source, result))
        event = _event_metrics(case_id, source, result)
        if event is not None:
            event_rows.append(event)
    return pd.DataFrame(case_rows), pd.DataFrame(sensor_rows), pd.DataFrame(event_rows)
