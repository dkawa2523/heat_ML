"""Evaluate one trained celltemp artifact against the COMSOL holdout cases."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from celltemp.artifact import fitted_parameters, load_artifact
from celltemp.engine import ThermalRCModel
from celltemp.inference import forecast, sensor_bias_in_gauge
from celltemp.io import load_system_spec, trajectory_from_frame

SENSORS = ("chip", "sink_base", "fins")
CONTROLS = ("chip_power", "coolant_temperature")
NIS_CONFIDENCE = 0.9999
NIS_THRESHOLDS = {1: 15.1367, 2: 18.4207, 3: 21.1075}

# These are screening limits for this deterministic, linear COMSOL v1 problem.
# They are not product-temperature tolerances for a real chip or package.
CRITERIA = {
    "internal_test_mean_rmse_k": 0.10,
    "external_forecast_mean_rmse_k": 0.25,
    "external_forecast_worst_rmse_k": 0.75,
    "nominal_alert_rate": 0.01,
    "drift_final_sensor_bias_error_k": 0.15,
    "sensor_offset_detection_delay_s": 2.0,
    "sensor_offset_final_sensor_bias_error_k": 0.15,
    "sensor_offset_final_disturbance_w": 0.50,
    "missing_window_rmse_k": 0.50,
    "hidden_heat_detection_delay_s": 20.0,
    "hidden_heat_peak_error_w": 0.75,
    "hidden_heat_max_sensor_bias_k": 0.20,
}


def _rmse(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")


def _mae(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(np.abs(finite))) if finite.size else float("nan")


def _max_abs(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.max(np.abs(finite))) if finite.size else float("nan")


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert missing table values to JSON null instead of non-standard NaN."""
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _assert_aligned(source: pd.DataFrame, result: pd.DataFrame, case_id: str) -> None:
    if len(source) != len(result):
        raise ValueError(f"{case_id}: source and result row counts differ")
    if not np.allclose(source["time"], result["time"], rtol=0.0, atol=1e-9):
        raise ValueError(f"{case_id}: source and result times differ")


def _request_from_frame(case_id: str, frame: pd.DataFrame):
    return trajectory_from_frame(
        case_id=case_id,
        frame=frame,
        time_col="time",
        sensor_cols=SENSORS,
        control_cols=CONTROLS,
        control_convention="left",
    )


def _model_forecast(model: ThermalRCModel, case_id: str, frame: pd.DataFrame) -> np.ndarray:
    return forecast(model, _request_from_frame(case_id, frame)).sensor_temperature


def _forecast_sensor_rows(
    case_id: str,
    case_group: str,
    truth: np.ndarray,
    predicted: np.ndarray,
    prior: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sensor in enumerate(SENSORS):
        error = predicted[1:, index] - truth[1:, index]
        prior_error = prior[1:, index] - truth[1:, index]
        prior_rmse = _rmse(prior_error)
        fitted_rmse = _rmse(error)
        rows.append(
            {
                "case_id": case_id,
                "case_group": case_group,
                "sensor": sensor,
                "n_points": len(error),
                "rmse_k": fitted_rmse,
                "mae_k": _mae(error),
                "max_abs_error_k": _max_abs(error),
                "terminal_error_k": float(error[-1]),
                "prior_rmse_k": prior_rmse,
                "rmse_improvement_percent": 100.0 * (prior_rmse - fitted_rmse) / prior_rmse,
            }
        )
    return rows


def _forecast_case_row(
    case_id: str,
    case_group: str,
    source: pd.DataFrame,
    truth: np.ndarray,
    predicted: np.ndarray,
    prior: np.ndarray,
) -> dict[str, Any]:
    error = predicted[1:] - truth[1:]
    prior_error = prior[1:] - truth[1:]
    prior_rmse = _rmse(prior_error)
    fitted_rmse = _rmse(error)
    chip_truth = truth[:, 0]
    chip_prediction = predicted[:, 0]
    spatial_gap = source["truth_chip_max"].to_numpy(dtype=np.float64) - chip_truth
    hotspot_error = source["truth_chip_max"].to_numpy(dtype=np.float64) - chip_prediction
    return {
        "case_id": case_id,
        "case_group": case_group,
        "n_rows": len(source),
        "rmse_k": fitted_rmse,
        "mae_k": _mae(error),
        "max_abs_error_k": _max_abs(error),
        "prior_rmse_k": prior_rmse,
        "rmse_improvement_percent": 100.0 * (prior_rmse - fitted_rmse) / prior_rmse,
        "chip_peak_error_k": float(np.max(chip_prediction) - np.max(chip_truth)),
        "max_spatial_hotspot_gap_k": float(np.max(spatial_gap)),
        "max_hotspot_underprediction_k": float(np.max(hotspot_error)),
    }


def evaluate_forecasts(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    source_dir = root / "data" / "eval" / "forecast"
    result_dir = root / "work" / "outputs" / "forecast"
    artifact_path = root / "work" / "outputs" / "runs" / "comsol_chip_cooling" / "artifact"
    artifact = load_artifact(artifact_path)
    prior_model = ThermalRCModel(load_system_spec(root / "system.yaml"), integrator="exact")
    case_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    leakage_check = True

    for source_path in sorted(source_dir.glob("*.csv")):
        case_id = source_path.stem
        source = pd.read_csv(source_path)
        result = pd.read_csv(result_dir / source_path.name)
        _assert_aligned(source, result, case_id)
        written = result[[f"temperature_{sensor}" for sensor in SENSORS]].to_numpy()
        predicted = _model_forecast(artifact.model, case_id, source)
        if not np.allclose(written, predicted, rtol=1e-10, atol=1e-10):
            raise ValueError(f"{case_id}: saved forecast differs from the saved artifact")
        if not np.isfinite(predicted).all():
            raise ValueError(f"{case_id}: forecast contains a non-finite temperature")
        prior = _model_forecast(prior_model, case_id, source)
        truth = source[[f"truth_{sensor}" for sensor in SENSORS]].to_numpy(dtype=np.float64)
        case_group = str(source["case_group"].iloc[0])
        case_rows.append(_forecast_case_row(case_id, case_group, source, truth, predicted, prior))
        sensor_rows.extend(_forecast_sensor_rows(case_id, case_group, truth, predicted, prior))

        if case_id == "F01_power_interpolation":
            altered = source.copy()
            altered[[column for column in altered if column.startswith("truth_")]] += 10_000.0
            leakage_check = np.array_equal(
                predicted, _model_forecast(artifact.model, case_id, altered)
            )

    return pd.DataFrame(case_rows), pd.DataFrame(sensor_rows), bool(leakage_check)


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


def _monitor_sensor_rows(
    case_id: str, source: pd.DataFrame, result: pd.DataFrame
) -> list[dict[str, Any]]:
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


def _monitor_case_row(case_id: str, source: pd.DataFrame, result: pd.DataFrame) -> dict[str, Any]:
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
    source_dir = root / "data" / "eval" / "monitor"
    result_dir = root / "work" / "outputs" / "monitor"
    case_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for source_path in sorted(source_dir.glob("*.csv")):
        case_id = source_path.stem
        source = pd.read_csv(source_path)
        result = pd.read_csv(result_dir / source_path.name)
        _assert_aligned(source, result, case_id)
        case_rows.append(_monitor_case_row(case_id, source, result))
        sensor_rows.extend(_monitor_sensor_rows(case_id, source, result))
        event = _event_metrics(case_id, source, result)
        if event is not None:
            event_rows.append(event)
    return pd.DataFrame(case_rows), pd.DataFrame(sensor_rows), pd.DataFrame(event_rows)


def _parameter_rows(root: Path) -> pd.DataFrame:
    prior = ThermalRCModel(load_system_spec(root / "system.yaml"), integrator="exact")
    artifact_path = root / "work" / "outputs" / "runs" / "comsol_chip_cooling" / "artifact"
    artifact = load_artifact(artifact_path)
    before = fitted_parameters(prior)
    after = artifact.metadata["fitted_parameters"]
    rows: list[dict[str, Any]] = []
    for section, value_name in (("edges", "conductance"), ("boundaries", "conductance")):
        for prior_item, fitted_item in zip(before[section], after[section]):
            name = prior_item.get("name") or f"{prior_item['node_a']}--{prior_item['node_b']}"
            prior_value = float(prior_item[value_name]["value"])
            fitted_value = float(fitted_item[value_name]["value"])
            kind = "edge" if section == "edges" else "boundary"
            rows.append(
                {
                    "parameter": name,
                    "kind": kind,
                    "prior_value": prior_value,
                    "fitted_value": fitted_value,
                    "change_percent": 100.0 * (fitted_value - prior_value) / prior_value,
                    "unit": "W/K",
                }
            )
    return pd.DataFrame(rows)


def _checks(
    internal: dict[str, Any],
    forecast_cases: pd.DataFrame,
    monitor_cases: pd.DataFrame,
    events: pd.DataFrame,
    leakage_check: bool,
) -> dict[str, bool]:
    monitor = monitor_cases.set_index("case_id")
    event = events.set_index("case_id")
    values = {
        "internal_test_accuracy": (
            internal["test"]["mean_case_rmse"] <= CRITERIA["internal_test_mean_rmse_k"]
        ),
        "external_forecast_mean_accuracy": (
            forecast_cases["rmse_k"].mean() <= CRITERIA["external_forecast_mean_rmse_k"]
        ),
        "external_forecast_worst_accuracy": (
            forecast_cases["rmse_k"].max() <= CRITERIA["external_forecast_worst_rmse_k"]
        ),
        "external_forecast_improves_prior": bool(
            (forecast_cases["rmse_k"] < forecast_cases["prior_rmse_k"]).all()
        ),
        "forecast_truth_is_not_an_input": leakage_check,
        "nominal_monitor_false_alert_control": (
            monitor.loc["M01_noise_baseline", "alert_rate"] <= CRITERIA["nominal_alert_rate"]
        ),
        "drift_sensor_bias_attribution": (
            monitor.loc["M02_sensor_drift", "max_final_sensor_bias_error_k"]
            <= CRITERIA["drift_final_sensor_bias_error_k"]
        ),
        "sensor_offset_detection_and_attribution": (
            bool(event.loc["M03_sensor_offset", "detected"])
            and event.loc["M03_sensor_offset", "detection_delay_s"]
            <= CRITERIA["sensor_offset_detection_delay_s"]
            and monitor.loc["M03_sensor_offset", "max_final_sensor_bias_error_k"]
            <= CRITERIA["sensor_offset_final_sensor_bias_error_k"]
            and abs(monitor.loc["M03_sensor_offset", "final_chip_disturbance_w"])
            <= CRITERIA["sensor_offset_final_disturbance_w"]
        ),
        "missing_measurement_continuity": bool(
            monitor.loc["M04_missing_measurements", "all_posterior_finite"]
        )
        and monitor.loc["M04_missing_measurements", "missing_posterior_rmse_k"]
        <= CRITERIA["missing_window_rmse_k"],
        "hidden_heat_detection_and_attribution": (
            bool(event.loc["M05_uncommanded_heat", "detected"])
            and event.loc["M05_uncommanded_heat", "detection_delay_s"]
            <= CRITERIA["hidden_heat_detection_delay_s"]
            and abs(monitor.loc["M05_uncommanded_heat", "peak_event_chip_disturbance_w"] - 3.0)
            <= CRITERIA["hidden_heat_peak_error_w"]
            and monitor.loc["M05_uncommanded_heat", "max_event_abs_sensor_bias_k"]
            <= CRITERIA["hidden_heat_max_sensor_bias_k"]
        ),
    }
    return {name: bool(value) for name, value in values.items()}


def _summary(
    root: Path,
    forecast_cases: pd.DataFrame,
    monitor_cases: pd.DataFrame,
    events: pd.DataFrame,
    parameters: pd.DataFrame,
    leakage_check: bool,
) -> dict[str, Any]:
    run_dir = root / "work" / "outputs" / "runs" / "comsol_chip_cooling"
    internal = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "artifact" / "metadata.json").read_text(encoding="utf-8"))
    qa = pd.read_csv(root / "data" / "qa_summary.csv")
    checks = _checks(internal, forecast_cases, monitor_cases, events, leakage_check)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": "COMSOL 6.4 Electronic Chip Cooling",
            "fidelity": "solid_conduction_contact_constant_convection",
            "input_convention": "left_zero_order_hold",
            "published_trajectories": len(qa),
            "published_rows": int(qa["rows"].sum()),
            "forecast_cases": int((qa["role"] == "forecast").sum()),
            "monitor_cases": int((qa["role"] == "monitor").sum()),
        },
        "training": {
            "best_epoch": metadata["training"]["best_epoch"],
            "best_validation_rmse_k": metadata["training"]["best_validation_rmse"],
            "split_metrics": internal,
        },
        "forecast": {
            "mean_case_rmse_k": float(forecast_cases["rmse_k"].mean()),
            "median_case_rmse_k": float(forecast_cases["rmse_k"].median()),
            "worst_case_rmse_k": float(forecast_cases["rmse_k"].max()),
            "worst_case": str(forecast_cases.loc[forecast_cases["rmse_k"].idxmax(), "case_id"]),
            "mean_prior_rmse_k": float(forecast_cases["prior_rmse_k"].mean()),
            "mean_rmse_improvement_percent": float(
                forecast_cases["rmse_improvement_percent"].mean()
            ),
            "max_spatial_hotspot_gap_k": float(forecast_cases["max_spatial_hotspot_gap_k"].max()),
            "max_hotspot_underprediction_k": float(
                forecast_cases["max_hotspot_underprediction_k"].max()
            ),
        },
        "monitor": {
            "case_metrics": _json_records(monitor_cases),
            "events": _json_records(events),
            "bias_identifiability": (
                "Sensor bias uses the gauge written in each monitor output. A reference gauge "
                "fixes the explicitly calibrated sensor bias to zero."
            ),
        },
        "parameters": _json_records(parameters),
        "screening_criteria": CRITERIA,
        "checks": checks,
        "passed_checks": int(sum(checks.values())),
        "total_checks": len(checks),
    }


def evaluate(root: Path, output: Path) -> dict[str, Any]:
    forecast_cases, forecast_sensors, leakage_check = evaluate_forecasts(root)
    monitor_cases, monitor_sensors, events = evaluate_monitoring(root)
    parameters = _parameter_rows(root)
    summary = _summary(root, forecast_cases, monitor_cases, events, parameters, leakage_check)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "forecast_cases.csv": forecast_cases,
        "forecast_sensors.csv": forecast_sensors,
        "monitor_cases.csv": monitor_cases,
        "monitor_sensors.csv": monitor_sensors,
        "monitor_events.csv": events,
        "fitted_parameters.csv": parameters,
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False, float_format="%.10g")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parent
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "work" / "evaluation"
    summary = evaluate(root, output)
    print(json.dumps(summary["checks"], indent=2))
    print(f"saved evaluation: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
