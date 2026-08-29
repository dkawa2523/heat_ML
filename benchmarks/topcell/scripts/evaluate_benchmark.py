"""Evaluate the learned artifact on cases excluded from model fitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from celltemp.artifact import ThermalArtifact, load_artifact
from celltemp.config import as_path, load_config
from celltemp.domain import Trajectory
from celltemp.engine import ThermalRCModel
from celltemp.inference import forecast, monitor
from celltemp.io import load_system_spec, load_trajectories
from celltemp.workflows.common import resolve_artifact_path

from .definition import PARAMETER_TRUTH

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "work" / "outputs" / "benchmark"

PRIMARY_METRICS = {
    "interpolation": "learned_rmse, improvement_vs_prior_pct",
    "extrapolation": "learned_rmse, learned_max_abs",
    "dynamics": "learned_rmse, final_rmse",
    "initialization": "learned_rmse, observed_initial_sensors",
    "sampling": "learned_rmse on variable dt",
    "model_gap": "error amplification relative to matched-physics cases",
    "baseline": "residual_rmse, filtered_rmse_to_truth",
    "bias_tracking": "filtered_rmse_to_truth, bias_rmse",
    "fault_detection": "max_normalized_residual, detection_delay",
    "missing_data": "filtered_finite, filtered_rmse_to_truth",
    "disturbance_detection": "max_normalized_residual, detection_delay",
}


def _trajectory_config(
    directory: str, sensor_names: tuple[str, ...], control_names: tuple[str, ...]
) -> dict[str, object]:
    return {
        "directory": directory,
        "pattern": "*.csv",
        "time_col": "time",
        "sensor_cols": sensor_names,
        "control_cols": control_names,
        "control_convention": "left",
        "sep": ",",
        "allow_missing_temperatures": True,
    }


def _source_frame(trajectory: Trajectory) -> pd.DataFrame:
    return pd.read_csv(Path(str(trajectory.metadata["path"])))


def _error_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
        "final_rmse": float(np.sqrt(np.mean(error[-1] ** 2))),
    }


def _persistence_prediction(trajectory: Trajectory) -> np.ndarray:
    initial = trajectory.temperature[0].copy()
    observed = trajectory.mask[0]
    fill = float(np.mean(initial[observed]))
    initial[~observed] = fill
    return np.repeat(initial[None, :], len(trajectory.time), axis=0)


def evaluate_forecasts(
    artifact: ThermalArtifact, prior_model: ThermalRCModel, directory: str
) -> pd.DataFrame:
    trajectories = load_trajectories(
        _trajectory_config(directory, artifact.sensor_names, artifact.control_names), ROOT
    )
    rows: list[dict[str, object]] = []
    truth_columns = [f"truth_{sensor}" for sensor in artifact.sensor_names]
    for trajectory in trajectories:
        if trajectory.mask[1:].any():
            raise ValueError(
                f"{trajectory.case_id}: future sensor values would leak into forecast input"
            )
        frame = _source_frame(trajectory)
        truth = frame[truth_columns].to_numpy(dtype=float)
        learned = forecast(artifact.model, trajectory).sensor_temperature
        prior = forecast(prior_model, trajectory).sensor_temperature
        persistence = _persistence_prediction(trajectory)
        learned_metrics = _error_metrics(learned, truth)
        prior_metrics = _error_metrics(prior, truth)
        persistence_metrics = _error_metrics(persistence, truth)
        rows.append(
            {
                "case_id": trajectory.case_id,
                "group": str(frame["benchmark_group"].iat[0]),
                "purpose": str(frame["benchmark_purpose"].iat[0]),
                "rows": len(frame),
                "time_end": float(trajectory.time[-1]),
                "observed_initial_sensors": int(trajectory.mask[0].sum()),
                "learned_rmse": learned_metrics["rmse"],
                "learned_mae": learned_metrics["mae"],
                "learned_max_abs": learned_metrics["max_abs"],
                "final_rmse": learned_metrics["final_rmse"],
                "prior_rmse": prior_metrics["rmse"],
                "persistence_rmse": persistence_metrics["rmse"],
                "improvement_vs_prior_pct": 100.0
                * (prior_metrics["rmse"] - learned_metrics["rmse"])
                / prior_metrics["rmse"],
                "improvement_vs_persistence_pct": 100.0
                * (persistence_metrics["rmse"] - learned_metrics["rmse"])
                / persistence_metrics["rmse"],
            }
        )
    return pd.DataFrame(rows)


def summarize_forecast_groups(cases: pd.DataFrame) -> pd.DataFrame:
    return (
        cases.groupby("group", sort=False)
        .agg(
            n_cases=("case_id", "count"),
            mean_learned_rmse=("learned_rmse", "mean"),
            worst_learned_rmse=("learned_rmse", "max"),
            mean_prior_rmse=("prior_rmse", "mean"),
            mean_persistence_rmse=("persistence_rmse", "mean"),
        )
        .reset_index()
    )


def _rmse(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    return float(np.sqrt(np.mean(values[finite] ** 2)))


def _detection_delay(
    time: np.ndarray, normalized_residual: np.ndarray, event_start: float
) -> float:
    finite = np.isfinite(normalized_residual)
    signal = np.where(finite, normalized_residual, -np.inf).max(axis=1)
    detected = np.flatnonzero((time >= event_start) & (signal >= 4.0))
    return float("nan") if len(detected) == 0 else float(time[detected[0]] - event_start)


def evaluate_monitors(
    artifact: ThermalArtifact, directory: str, observer_config: dict
) -> pd.DataFrame:
    trajectories = load_trajectories(
        _trajectory_config(directory, artifact.sensor_names, artifact.control_names), ROOT
    )
    rows: list[dict[str, object]] = []
    truth_columns = [f"truth_{sensor}" for sensor in artifact.sensor_names]
    bias_columns = [f"truth_bias_{sensor}" for sensor in artifact.sensor_names]
    for trajectory in trajectories:
        frame = _source_frame(trajectory)
        truth = frame[truth_columns].to_numpy(dtype=float)
        truth_bias = frame[bias_columns].to_numpy(dtype=float)
        result = monitor(artifact.model, trajectory, **observer_config)
        residual_valid = np.isfinite(result.residual) & (result.residual_std > 0.0)
        normalized = np.full_like(result.residual, np.nan)
        normalized[residual_valid] = np.abs(
            result.residual[residual_valid] / result.residual_std[residual_valid]
        )
        burn_in = trajectory.time >= 30.0
        event_start = (
            float(frame["event_start"].iat[0]) if "event_start" in frame.columns else float("nan")
        )
        rows.append(
            {
                "case_id": trajectory.case_id,
                "group": str(frame["benchmark_group"].iat[0]),
                "purpose": str(frame["benchmark_purpose"].iat[0]),
                "measured_rmse_to_truth": _rmse(trajectory.temperature - truth),
                "filtered_rmse_to_truth": _rmse(result.filtered_sensor_temperature - truth),
                "bias_rmse_after_burn_in": _rmse(result.sensor_bias[burn_in] - truth_bias[burn_in]),
                "residual_rmse": _rmse(result.residual),
                "max_normalized_residual": float(np.nanmax(normalized)),
                "normalized_residual_over_4_fraction": float(
                    np.mean(normalized[residual_valid] >= 4.0)
                ),
                "missing_fraction": float(np.mean(~trajectory.mask)),
                "filtered_finite": bool(np.isfinite(result.filtered_sensor_temperature).all()),
                "event_start": event_start,
                "detection_delay": (
                    _detection_delay(trajectory.time, normalized, event_start)
                    if np.isfinite(event_start)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def parameter_recovery(artifact: ThermalArtifact) -> pd.DataFrame:
    fitted = artifact.metadata["fitted_parameters"]
    values: dict[tuple[str, str], float] = {}
    for item in fitted["edges"]:
        values[("edge", f"{item['node_a']}-{item['node_b']}")] = float(item["conductance"])
    for item in fitted["actuators"]:
        values[("actuator", str(item["name"]))] = float(item["tau"])
    for item in fitted["sources"]:
        values[("source", str(item["name"]))] = float(item["gain"])
    for item in fitted["boundaries"]:
        values[("boundary", str(item["name"]))] = float(item["conductance"])

    rows = []
    for (parameter_type, name), truth_value in PARAMETER_TRUTH.items():
        fitted_value = values[(parameter_type, name)]
        rows.append(
            {
                "parameter_type": parameter_type,
                "name": name,
                "truth": truth_value,
                "fitted": fitted_value,
                "absolute_relative_error": abs(fitted_value - truth_value) / truth_value,
            }
        )
    return pd.DataFrame(rows)


def _row_by_group(frame: pd.DataFrame, group: str) -> pd.Series:
    return frame.loc[frame["group"] == group].iloc[0]


def acceptance_checks(forecast_cases: pd.DataFrame, monitor_cases: pd.DataFrame) -> dict[str, bool]:
    core = forecast_cases.loc[forecast_cases["group"] != "model_gap"]
    gap = _row_by_group(forecast_cases, "model_gap")
    nominal = _row_by_group(monitor_cases, "baseline")
    drift = _row_by_group(monitor_cases, "bias_tracking")
    fault = _row_by_group(monitor_cases, "fault_detection")
    missing = _row_by_group(monitor_cases, "missing_data")
    disturbance = _row_by_group(monitor_cases, "disturbance_detection")
    return {
        "forecast_core_all_finite": bool(np.isfinite(core["learned_rmse"].to_numpy()).all()),
        "forecast_core_mean_rmse_below_1K": bool(core["learned_rmse"].mean() < 1.0),
        "forecast_core_worst_rmse_below_1_5K": bool(core["learned_rmse"].max() < 1.5),
        "forecast_core_beats_engineering_prior": bool(
            core["learned_rmse"].mean() < core["prior_rmse"].mean()
        ),
        "nonlinear_model_gap_is_visible": bool(
            gap["learned_rmse"] > max(0.5, 2.0 * core["learned_rmse"].mean())
        ),
        "monitor_nominal_residual_below_0_35K": bool(nominal["residual_rmse"] < 0.35),
        "monitor_bias_filter_improves_measurement": bool(
            drift["filtered_rmse_to_truth"] < drift["measured_rmse_to_truth"]
        ),
        "monitor_sensor_fault_detected_within_2s": bool(fault["detection_delay"] <= 2.0),
        "monitor_missing_data_remains_finite": bool(missing["filtered_finite"]),
        "monitor_missing_data_rmse_below_0_25K": bool(missing["filtered_rmse_to_truth"] < 0.25),
        "monitor_heat_load_detected_within_5s": bool(disturbance["detection_delay"] <= 5.0),
    }


def write_case_catalog(
    forecast_cases: pd.DataFrame, monitor_cases: pd.DataFrame, output_dir: Path
) -> None:
    forecast_catalog = forecast_cases[["case_id", "group", "purpose"]].copy()
    forecast_catalog.insert(0, "workflow", "forecast")
    monitor_catalog = monitor_cases[["case_id", "group", "purpose"]].copy()
    monitor_catalog.insert(0, "workflow", "monitor")
    catalog = pd.concat([forecast_catalog, monitor_catalog], ignore_index=True)
    catalog["primary_metrics"] = catalog["group"].map(PRIMARY_METRICS)
    catalog.to_csv(output_dir / "case_catalog.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "config.yaml")
    artifact = load_artifact(resolve_artifact_path(config, ROOT))
    prior_model = ThermalRCModel(
        load_system_spec(as_path(str(config["system"]), ROOT)),
        integrator=str(config.get("engine", {}).get("integrator", "exact")),
    )
    prior_model.eval()

    forecasts = evaluate_forecasts(artifact, prior_model, str(config["forecast"]["input_dir"]))
    forecast_groups = summarize_forecast_groups(forecasts)
    monitors = evaluate_monitors(
        artifact,
        str(config["monitor"]["input_dir"]),
        dict(config["monitor"].get("observer", {})),
    )
    parameters = parameter_recovery(artifact)
    checks = acceptance_checks(forecasts, monitors)

    forecasts.to_csv(OUTPUT_DIR / "forecast_by_case.csv", index=False)
    forecast_groups.to_csv(OUTPUT_DIR / "forecast_by_group.csv", index=False)
    monitors.to_csv(OUTPUT_DIR / "monitor_by_case.csv", index=False)
    parameters.to_csv(OUTPUT_DIR / "parameter_recovery.csv", index=False)
    write_case_catalog(forecasts, monitors, OUTPUT_DIR)

    core = forecasts.loc[forecasts["group"] != "model_gap"]
    summary = {
        "benchmark_version": 1,
        "evaluation_boundary": "external cases excluded from training discovery",
        "status": "pass" if all(checks.values()) else "review",
        "checks": checks,
        "forecast": {
            "n_core_cases": len(core),
            "mean_core_rmse": float(core["learned_rmse"].mean()),
            "worst_core_rmse": float(core["learned_rmse"].max()),
            "mean_prior_rmse": float(core["prior_rmse"].mean()),
            "model_gap_rmse": float(_row_by_group(forecasts, "model_gap")["learned_rmse"]),
        },
        "monitor": {
            "n_cases": len(monitors),
            "max_detection_delay": float(monitors["detection_delay"].dropna().max()),
        },
    }
    (OUTPUT_DIR / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"benchmark status: {summary['status']}")
    print(f"saved benchmark evidence: {OUTPUT_DIR}")
    if summary["status"] != "pass":
        raise SystemExit("benchmark acceptance checks failed")


if __name__ == "__main__":
    main()
