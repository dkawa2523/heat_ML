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
from celltemp.io import load_system_spec
from celltemp.workflows.prediction_figures import write_prediction_figures
from external_tools.comsol_chip_cooling.evaluate_forecast import SENSORS, evaluate_forecasts
from external_tools.comsol_chip_cooling.evaluate_monitor import evaluate_monitoring
from external_tools.comsol_chip_cooling.evaluation_support import (
    json_records,
)

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

    def number(frame: pd.DataFrame, row: str, column: str) -> float:
        values = frame.loc[[row], [column]].to_numpy(dtype=np.float64)
        if values.size != 1:
            raise ValueError(f"expected one value for {row}.{column}")
        return float(values.item())

    nominal_alert_rate = number(monitor, "M01_noise_baseline", "alert_rate")
    drift_bias_error = number(monitor, "M02_sensor_drift", "max_final_sensor_bias_error_k")
    offset_detected = bool(number(event, "M03_sensor_offset", "detected"))
    offset_delay = number(event, "M03_sensor_offset", "detection_delay_s")
    offset_bias_error = number(monitor, "M03_sensor_offset", "max_final_sensor_bias_error_k")
    offset_disturbance = number(monitor, "M03_sensor_offset", "final_chip_disturbance_w")
    missing_finite = bool(number(monitor, "M04_missing_measurements", "all_posterior_finite"))
    missing_rmse = number(monitor, "M04_missing_measurements", "missing_posterior_rmse_k")
    hidden_heat_detected = bool(number(event, "M05_uncommanded_heat", "detected"))
    hidden_heat_delay = number(event, "M05_uncommanded_heat", "detection_delay_s")
    hidden_heat_peak = number(monitor, "M05_uncommanded_heat", "peak_event_chip_disturbance_w")
    hidden_heat_bias = number(monitor, "M05_uncommanded_heat", "max_event_abs_sensor_bias_k")
    values = {
        "internal_test_accuracy": (
            internal["test"]["mean_case_causal_rmse"] <= CRITERIA["internal_test_mean_rmse_k"]
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
        "external_forecast_improves_persistence": bool(
            (forecast_cases["rmse_k"] < forecast_cases["persistence_rmse_k"]).all()
        ),
        "forecast_truth_is_not_an_input": leakage_check,
        "nominal_monitor_false_alert_control": (
            nominal_alert_rate <= CRITERIA["nominal_alert_rate"]
        ),
        "drift_sensor_bias_attribution": (
            drift_bias_error <= CRITERIA["drift_final_sensor_bias_error_k"]
        ),
        "sensor_offset_detection_and_attribution": (
            offset_detected
            and offset_delay <= CRITERIA["sensor_offset_detection_delay_s"]
            and offset_bias_error <= CRITERIA["sensor_offset_final_sensor_bias_error_k"]
            and abs(offset_disturbance) <= CRITERIA["sensor_offset_final_disturbance_w"]
        ),
        "missing_measurement_continuity": missing_finite
        and missing_rmse <= CRITERIA["missing_window_rmse_k"],
        "hidden_heat_detection_and_attribution": (
            hidden_heat_detected
            and hidden_heat_delay <= CRITERIA["hidden_heat_detection_delay_s"]
            and abs(hidden_heat_peak - 3.0) <= CRITERIA["hidden_heat_peak_error_w"]
            and hidden_heat_bias <= CRITERIA["hidden_heat_max_sensor_bias_k"]
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
            "best_causal_validation_rmse_k": metadata["training"]["best_causal_validation_rmse"],
            "split_metrics": internal,
        },
        "forecast": {
            "mean_case_rmse_k": float(forecast_cases["rmse_k"].mean()),
            "median_case_rmse_k": float(forecast_cases["rmse_k"].median()),
            "worst_case_rmse_k": float(forecast_cases["rmse_k"].max()),
            "worst_case": str(forecast_cases.loc[forecast_cases["rmse_k"].idxmax(), "case_id"]),
            "mean_prior_rmse_k": float(forecast_cases["prior_rmse_k"].mean()),
            "mean_persistence_rmse_k": float(forecast_cases["persistence_rmse_k"].mean()),
            "mean_rmse_improvement_percent": float(
                forecast_cases["rmse_improvement_percent"].mean()
            ),
            "max_spatial_hotspot_gap_k": float(forecast_cases["max_spatial_hotspot_gap_k"].max()),
            "max_hotspot_underprediction_k": float(
                forecast_cases["max_hotspot_underprediction_k"].max()
            ),
        },
        "monitor": {
            "case_metrics": json_records(monitor_cases),
            "events": json_records(events),
            "bias_identifiability": (
                "Sensor bias uses the gauge written in each monitor output. A reference gauge "
                "fixes the explicitly calibrated sensor bias to zero."
            ),
        },
        "parameters": json_records(parameters),
        "screening_criteria": CRITERIA,
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
    }


def evaluate(root: Path, output: Path) -> dict[str, Any]:
    (
        forecast_cases,
        forecast_sensors,
        model_comparison,
        leakage_check,
        prediction_cases,
    ) = evaluate_forecasts(root)
    monitor_cases, monitor_sensors, events = evaluate_monitoring(root)
    parameters = _parameter_rows(root)
    summary = _summary(root, forecast_cases, monitor_cases, events, parameters, leakage_check)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "forecast_cases.csv": forecast_cases,
        "forecast_sensors.csv": forecast_sensors,
        "model_comparison.csv": model_comparison,
        "monitor_cases.csv": monitor_cases,
        "monitor_sensors.csv": monitor_sensors,
        "monitor_events.csv": events,
        "fitted_parameters.csv": parameters,
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False, float_format="%.10g")
    write_prediction_figures(
        prediction_cases,
        output / "figures",
        sensor_names=SENSORS,
        title="Linear COMSOL external forecast",
        file_prefix="prediction",
    )
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
    return 0 if all(summary["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
