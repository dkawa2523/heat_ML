"""Monitor tables for the linear COMSOL causal evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from celltemp.analysis import max_abs_error as _max_abs
from celltemp.analysis import rmse as _rmse
from celltemp.inference import sensor_bias_in_gauge
from external_tools.comsol_chip_cooling.evaluation_support import assert_aligned

SENSORS = ("chip", "sink_base", "fins")
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


def _sensor_rows(case_id: str, source: pd.DataFrame, result: pd.DataFrame) -> list[dict[str, Any]]:
    truth_sensor_bias = _truth_sensor_bias(source, result)
    rows: list[dict[str, Any]] = []
    for index, sensor in enumerate(SENSORS):
        truth = source[f"truth_{sensor}"].to_numpy(dtype=np.float64)
        measured = result[f"measured_{sensor}"].to_numpy(dtype=np.float64)
        prior_physical = result[f"prior_physical_{sensor}"].to_numpy(dtype=np.float64)
        predicted_measurement = result[f"predicted_measurement_{sensor}"].to_numpy(dtype=np.float64)
        posterior_physical = result[f"posterior_physical_{sensor}"].to_numpy(dtype=np.float64)
        reconstructed = result[f"reconstructed_measurement_{sensor}"].to_numpy(dtype=np.float64)
        sensor_bias = result[f"sensor_bias_{sensor}"].to_numpy(dtype=np.float64)
        innovation = result[f"innovation_{sensor}"].to_numpy(dtype=np.float64)
        innovation_std = result[f"innovation_std_{sensor}"].to_numpy(dtype=np.float64)
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
    measured = result[[f"measured_{sensor}" for sensor in SENSORS]].to_numpy(dtype=np.float64)
    prior_physical = result[[f"prior_physical_{sensor}" for sensor in SENSORS]].to_numpy(
        dtype=np.float64
    )
    posterior_physical = result[[f"posterior_physical_{sensor}" for sensor in SENSORS]].to_numpy(
        dtype=np.float64
    )
    reconstructed = result[[f"reconstructed_measurement_{sensor}" for sensor in SENSORS]].to_numpy(
        dtype=np.float64
    )
    sensor_bias = result[[f"sensor_bias_{sensor}" for sensor in SENSORS]].to_numpy(dtype=np.float64)
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
    chip_disturbance = result["disturbance_chip_w"].to_numpy(dtype=np.float64)
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
    sensor_bias = result[[f"sensor_bias_{sensor}" for sensor in SENSORS]].to_numpy(dtype=np.float64)
    disturbance = result["disturbance_chip_w"].to_numpy(dtype=np.float64)
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
    source_dir = root / "data" / "eval" / "monitor"
    result_dir = root / "work" / "outputs" / "monitor"
    case_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for source_path in sorted(source_dir.glob("*.csv")):
        case_id = source_path.stem
        source = pd.read_csv(source_path)
        result = pd.read_csv(result_dir / source_path.name)
        assert_aligned(source, result, case_id)
        case_rows.append(_case_row(case_id, source, result))
        sensor_rows.extend(_sensor_rows(case_id, source, result))
        event = _event_metrics(case_id, source, result)
        if event is not None:
            event_rows.append(event)
    return pd.DataFrame(case_rows), pd.DataFrame(sensor_rows), pd.DataFrame(event_rows)
