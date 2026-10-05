"""Evaluate the learned artifact on cases excluded from model fitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from benchmarks.validation_figures import (
    VALIDATION_FIGURE_DIRECTORY,
    validation_figure_reference,
)
from celltemp.analysis import (
    persistence_prediction,
    prediction_comparison_rows,
    prediction_error_metrics,
    rmse,
)
from celltemp.artifact import ThermalArtifact, load_artifact
from celltemp.config import as_path, load_config
from celltemp.domain import Trajectory
from celltemp.engine import ThermalRCModel
from celltemp.inference import (
    build_observer,
    forecast,
    forecast_origin_index,
    resolve_observer_settings,
    sensor_bias_in_gauge,
)
from celltemp.io import load_system_spec, load_trajectories
from celltemp.workflows.common import (
    resolve_artifact_path,
    resolve_runtime_values,
    validate_runtime_manifest,
)
from celltemp.workflows.prediction_figures import PredictionCase, write_prediction_figures

from .definition import PARAMETER_TRUTH

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "work" / "outputs" / "benchmark"
NIS_THRESHOLDS = {1: 15.1367, 2: 18.4207, 3: 21.1075, 4: 23.5127}

PRIMARY_METRICS = {
    "interpolation": "learned_rmse, improvement_vs_prior_pct",
    "extrapolation": "learned_rmse, learned_max_abs",
    "dynamics": "learned_rmse, final_rmse",
    "initialization": "learned_rmse, history_rows, observed_history_sensors",
    "sampling": "learned_rmse on variable dt",
    "model_gap": "error amplification relative to matched-physics cases",
    "baseline": "innovation_rmse, mean_nis_per_dof",
    "bias_tracking": "sensor_bias_rmse, posterior_physical_rmse",
    "fault_detection": "NIS detection delay, first-alert sensor localization",
    "missing_data": "posterior_finite, posterior_physical_rmse",
    "disturbance_detection": "NIS detection delay, estimated physical heat",
}


def _json_bool(value: object) -> bool:
    """Normalize NumPy/pandas comparison results for JSON output."""
    return bool(value)


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


def _workflow_inputs(
    artifact: ThermalArtifact, values: dict, workflow: str, config_path: Path
) -> tuple[list[Trajectory], Path, dict[str, float | str | None]]:
    """Bind saved workflow output to its current inputs, artifact, and settings once."""
    root = config_path.resolve().parent
    data_config = _trajectory_config(
        str(values["input_dir"]), artifact.sensor_names, artifact.control_names
    )
    data_config.update(
        {
            name: values[name]
            for name in ("pattern", "time_col", "control_convention", "sep", "dt")
            if name in values
        }
    )
    trajectories = load_trajectories(data_config, root)
    output_dir = as_path(str(values["output_dir"]), root)
    observer_settings = resolve_observer_settings(workflow, values.get("observer"))
    observer = build_observer(artifact.model, observer_settings)
    validate_runtime_manifest(
        output_dir / "run_manifest.json",
        workflow=workflow,
        artifact=artifact,
        values=values,
        trajectories=trajectories,
        observer_settings=observer_settings,
        disturbance_basis=observer.disturbance_basis,
    )
    return trajectories, output_dir, observer_settings


def _saved_case(
    output_dir: Path, trajectory: Trajectory, columns: list[str], *, origin: int = 0
) -> pd.DataFrame:
    frame = pd.read_csv(output_dir / "cases" / f"{trajectory.case_id}.csv")
    if not {"time", *columns} <= set(frame):
        raise ValueError(f"{trajectory.case_id}: saved predictions are missing required columns")
    expected_time = trajectory.time[origin:]
    if len(frame) != len(expected_time) or not np.allclose(
        frame["time"].to_numpy(dtype=float), expected_time, rtol=1e-12, atol=1e-12
    ):
        raise ValueError(f"{trajectory.case_id}: saved prediction times differ from the inputs")
    return frame


def evaluate_forecasts(
    artifact: ThermalArtifact, prior_model: ThermalRCModel, values: dict, config_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[PredictionCase]]:
    trajectories, output_dir, observer_settings = _workflow_inputs(
        artifact, values, "forecast", config_path
    )
    prior_observer = build_observer(prior_model, observer_settings)
    rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    prediction_cases: list[PredictionCase] = []
    truth_columns = [f"truth_{sensor}" for sensor in artifact.sensor_names]
    for trajectory in trajectories:
        frame = _source_frame(trajectory)
        origin = forecast_origin_index(trajectory.mask)
        learned_columns = [f"sensor.{sensor}.temperature" for sensor in artifact.sensor_names]
        saved = _saved_case(output_dir, trajectory, learned_columns, origin=origin)
        learned = saved[learned_columns].to_numpy(dtype=float)
        truth = frame[truth_columns].to_numpy(dtype=float)[origin:]
        learned_metrics = prediction_error_metrics(truth, learned)
        prior_result = forecast(prior_model, trajectory, observer=prior_observer)
        if prior_result.forecast_origin_index != origin:
            raise RuntimeError("forecast implementations disagree on the history boundary")
        prior = prior_result.sensor_temperature
        persistence = persistence_prediction(trajectory.temperature, trajectory.mask, origin)
        prior_metrics = prediction_error_metrics(truth, prior)
        persistence_metrics = prediction_error_metrics(truth, persistence)
        case_group = str(frame["benchmark_group"].iat[0])
        purpose = str(frame["benchmark_purpose"].iat[0])
        rows.append(
            {
                "case_id": trajectory.case_id,
                "group": case_group,
                "purpose": purpose,
                "input_rows": len(frame),
                "history_rows": origin + 1,
                "forecast_rows": len(learned),
                "forecast_start_time": float(trajectory.time[origin]),
                "time_end": float(trajectory.time[-1]),
                "observed_initial_sensors": int(trajectory.mask[0].sum()),
                "observed_history_sensors": int(trajectory.mask[: origin + 1].any(axis=0).sum()),
                "predictions_finite": bool(np.isfinite(learned).all()),
                "n_evaluation_points": learned_metrics["n_points"],
                "expected_evaluation_points": truth.size,
                "prediction_coverage_fraction": learned_metrics["prediction_coverage_fraction"],
                "learned_rmse": learned_metrics["rmse_k"],
                "learned_mae": learned_metrics["mae_k"],
                "learned_max_abs": learned_metrics["max_abs_error_k"],
                "final_rmse": learned_metrics["terminal_rmse_k"],
                "prior_rmse": prior_metrics["rmse_k"],
                "persistence_rmse": persistence_metrics["rmse_k"],
                "improvement_vs_prior_pct": 100.0
                * (prior_metrics["rmse_k"] - learned_metrics["rmse_k"])
                / prior_metrics["rmse_k"],
                "improvement_vs_persistence_pct": 100.0
                * (persistence_metrics["rmse_k"] - learned_metrics["rmse_k"])
                / persistence_metrics["rmse_k"],
            }
        )
        prediction_cases.append(
            PredictionCase(
                case_id=trajectory.case_id,
                time=trajectory.time[origin:],
                truth=truth,
                predicted=learned,
            )
        )
        comparison_rows.extend(
            {
                "case_id": trajectory.case_id,
                "case_group": case_group,
                "purpose": purpose,
                **comparison,
            }
            for comparison in prediction_comparison_rows(
                trajectory.time[origin:],
                truth,
                {
                    "fitted_rc": learned,
                    "engineering_prior_rc": prior,
                    "persistence": persistence,
                },
                artifact.sensor_names,
            )
        )
    return pd.DataFrame(rows), pd.DataFrame(comparison_rows), prediction_cases


def _write_evidence_figures(
    cases: list[PredictionCase], forecasts: pd.DataFrame, sensor_names: tuple[str, ...]
) -> None:
    groups = forecasts.set_index("case_id")["group"].astype(str).to_dict()
    core = [case for case in cases if groups[case.case_id] != "model_gap"]
    model_gap = [case for case in cases if groups[case.case_id] == "model_gap"]
    write_prediction_figures(
        core,
        VALIDATION_FIGURE_DIRECTORY,
        sensor_names=sensor_names,
        title="TopCell external forecast — core cases",
        file_prefix="topcell_core_prediction",
    )
    if model_gap:
        write_prediction_figures(
            model_gap,
            VALIDATION_FIGURE_DIRECTORY,
            sensor_names=sensor_names,
            title="TopCell external forecast — nonlinear model gap",
            file_prefix="topcell_model_gap_prediction",
        )


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


def _nis_threshold(dof: np.ndarray) -> np.ndarray:
    threshold = np.full(np.asarray(dof).shape, np.nan, dtype=np.float64)
    for dimension, value in NIS_THRESHOLDS.items():
        threshold[dof == dimension] = value
    return threshold


def _detection_delay(time: np.ndarray, alert: np.ndarray, event_start: float) -> float:
    detected = np.flatnonzero((time >= event_start) & alert)
    return float("nan") if len(detected) == 0 else float(time[detected[0]] - event_start)


def evaluate_monitors(artifact: ThermalArtifact, values: dict, config_path: Path) -> pd.DataFrame:
    trajectories, output_dir, observer_config = _workflow_inputs(
        artifact, values, "monitor", config_path
    )
    rows: list[dict[str, object]] = []
    truth_columns = [f"truth_{sensor}" for sensor in artifact.sensor_names]
    bias_columns = [f"truth_bias_{sensor}" for sensor in artifact.sensor_names]
    disturbance_columns = [f"truth_disturbance_{node}_w" for node in artifact.model.spec.node_names]
    bias_reference = cast(str | None, observer_config["bias_reference"])
    bias_gauge = "zero_mean" if bias_reference is None else f"reference:{bias_reference}"
    saved_columns = (
        [
            f"sensor.{sensor}.{quantity}"
            for quantity in (
                "measured",
                "prior_physical",
                "predicted_measurement",
                "posterior_physical",
                "reconstructed_measurement",
                "innovation",
                "innovation_std",
                "bias",
            )
            for sensor in artifact.sensor_names
        ]
        + [
            f"node.{node}.{quantity}"
            for quantity in ("temperature", "disturbance_w")
            for node in artifact.model.spec.node_names
        ]
        + [
            f"control.{control}.{quantity}"
            for quantity in ("command", "effective")
            for control in artifact.control_names
        ]
        + ["nis", "nis_dof", "bias_gauge"]
    )
    for trajectory in trajectories:
        frame = _source_frame(trajectory)
        saved = _saved_case(output_dir, trajectory, saved_columns)
        measured = saved[
            [f"sensor.{sensor}.measured" for sensor in artifact.sensor_names]
        ].to_numpy(dtype=float)
        if not np.allclose(
            measured, trajectory.temperature, rtol=1e-12, atol=1e-12, equal_nan=True
        ):
            raise ValueError(f"{trajectory.case_id}: saved measurements differ from the inputs")
        if not saved["bias_gauge"].eq(bias_gauge).all():
            raise ValueError(f"{trajectory.case_id}: saved bias gauge differs from the settings")
        commands = saved[[f"control.{name}.command" for name in artifact.control_names]].to_numpy(
            dtype=float
        )
        expected_commands = np.concatenate([trajectory.commands, trajectory.commands[-1:]], axis=0)
        if not np.allclose(commands, expected_commands, rtol=1e-12, atol=1e-12):
            raise ValueError(f"{trajectory.case_id}: saved commands differ from the inputs")
        estimates = [
            column
            for column in saved_columns
            if column.rsplit(".", 1)[-1]
            in (
                "posterior_physical",
                "reconstructed_measurement",
                "bias",
                "temperature",
                "disturbance_w",
                "command",
                "effective",
            )
        ]
        prior_estimates = [
            column
            for column in saved_columns
            if column.rsplit(".", 1)[-1] in ("prior_physical", "predicted_measurement")
        ]
        if not (
            np.isfinite(saved[estimates].to_numpy(dtype=float)).all()
            and np.isfinite(saved[prior_estimates].to_numpy(dtype=float)[1:]).all()
        ):
            raise ValueError(f"{trajectory.case_id}: saved monitor estimates must be finite")
        posterior = saved[
            [f"sensor.{sensor}.posterior_physical" for sensor in artifact.sensor_names]
        ].to_numpy(dtype=float)
        sensor_bias = saved[[f"sensor.{sensor}.bias" for sensor in artifact.sensor_names]].to_numpy(
            dtype=float
        )
        disturbance = saved[
            [f"node.{node}.disturbance_w" for node in artifact.model.spec.node_names]
        ].to_numpy(dtype=float)
        innovation = saved[
            [f"sensor.{sensor}.innovation" for sensor in artifact.sensor_names]
        ].to_numpy(dtype=float)
        innovation_std = saved[
            [f"sensor.{sensor}.innovation_std" for sensor in artifact.sensor_names]
        ].to_numpy(dtype=float)
        nis = saved["nis"].to_numpy(dtype=float)
        dof = saved["nis_dof"].to_numpy(dtype=float)
        expected_dof = trajectory.mask.sum(axis=1)
        expected_dof[0] = 0
        observed = trajectory.mask.copy()
        observed[0] = False
        if (
            not np.array_equal(dof, expected_dof)
            or not np.isfinite(nis[dof > 0]).all()
            or not np.isfinite(innovation[observed]).all()
            or not np.isfinite(innovation_std[observed]).all()
            or not (innovation_std[observed] > 0).all()
        ):
            raise ValueError(
                f"{trajectory.case_id}: saved observed monitor diagnostics must be finite"
            )
        truth = frame[truth_columns].to_numpy(dtype=float)
        truth_bias = frame[bias_columns].to_numpy(dtype=float)
        truth_sensor_bias = sensor_bias_in_gauge(
            truth_bias,
            artifact.sensor_names,
            bias_reference,
        )
        truth_disturbance = frame[disturbance_columns].to_numpy(dtype=float)
        prediction_error_metrics(truth, posterior)
        threshold = _nis_threshold(dof)
        valid_nis = np.isfinite(nis) & np.isfinite(threshold)
        alert = valid_nis & (nis >= threshold)
        burn_in = trajectory.time >= 30.0
        event_start = (
            float(frame["event_start"].iat[0]) if "event_start" in frame.columns else float("nan")
        )
        event = np.linalg.norm(truth_disturbance, axis=1) > 0.0
        if not np.any(event) and np.isfinite(event_start):
            event = trajectory.time >= event_start
        disturbance_norm = np.linalg.norm(disturbance, axis=1)
        event_alerts = np.flatnonzero(event & alert)
        first_alert_sensor = ""
        if event_alerts.size:
            alert_index = int(event_alerts[0])
            marginal = np.where(
                trajectory.mask[alert_index],
                np.abs(innovation[alert_index] / innovation_std[alert_index]),
                np.nan,
            )
            first_alert_sensor = artifact.sensor_names[int(np.nanargmax(marginal))]
        rows.append(
            {
                "case_id": trajectory.case_id,
                "group": str(frame["benchmark_group"].iat[0]),
                "purpose": str(frame["benchmark_purpose"].iat[0]),
                "bias_gauge": bias_gauge,
                "measured_rmse_to_truth": rmse(trajectory.temperature - truth),
                "posterior_physical_rmse": rmse(posterior - truth),
                "sensor_bias_rmse_after_burn_in": rmse(
                    sensor_bias[burn_in] - truth_sensor_bias[burn_in]
                ),
                "max_final_sensor_bias_error": float(
                    np.max(np.abs(sensor_bias[-1] - truth_sensor_bias[-1]))
                ),
                "innovation_rmse": rmse(innovation[observed]),
                "max_nis": float(np.max(nis[valid_nis])) if valid_nis.any() else float("nan"),
                "mean_nis_per_dof": float(np.mean(nis[valid_nis] / dof[valid_nis])),
                "nis_alert_fraction": float(np.mean(alert[1:])),
                "missing_fraction": float(np.mean(~trajectory.mask)),
                "posterior_finite": bool(np.isfinite(posterior).all()),
                "peak_event_disturbance_w": (
                    float(np.max(disturbance_norm[event])) if np.any(event) else float("nan")
                ),
                "event_disturbance_rmse_w": rmse(disturbance[event] - truth_disturbance[event]),
                "max_event_sensor_bias_k": (
                    float(np.max(np.abs(sensor_bias[event]))) if np.any(event) else float("nan")
                ),
                "first_alert_sensor": first_alert_sensor,
                "event_start": event_start,
                "detection_delay": (
                    _detection_delay(trajectory.time, alert, event_start)
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
        values[("edge", f"{item['node_a']}-{item['node_b']}")] = float(item["conductance"]["value"])
    for item in fitted["actuators"]:
        values[("actuator", str(item["name"]))] = float(item["tau"])
    for item in fitted["sources"]:
        values[("source", str(item["name"]))] = float(item["heat_rate"]["gain"])
    for item in fitted["boundaries"]:
        values[("boundary", str(item["name"]))] = float(item["conductance"]["value"])

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


def _row_by_case(frame: pd.DataFrame, case_id: str) -> pd.Series:
    return frame.loc[frame["case_id"] == case_id].iloc[0]


def acceptance_checks(forecast_cases: pd.DataFrame, monitor_cases: pd.DataFrame) -> dict[str, bool]:
    core = forecast_cases.loc[forecast_cases["group"] != "model_gap"]
    gap = _row_by_group(forecast_cases, "model_gap")
    nominal = _row_by_group(monitor_cases, "baseline")
    drift = _row_by_group(monitor_cases, "bias_tracking")
    fault = _row_by_group(monitor_cases, "fault_detection")
    missing = _row_by_group(monitor_cases, "missing_data")
    disturbance = _row_by_group(monitor_cases, "disturbance_detection")
    sparse_initial = _row_by_case(forecast_cases, "F09_sparse_initial_observation")
    sparse_history = _row_by_case(forecast_cases, "F12_history_initialized_sparse")
    complete_predictions = (
        core["predictions_finite"].eq(True)
        & core["prediction_coverage_fraction"].eq(1.0)
        & core["n_evaluation_points"].gt(0)
        & core["n_evaluation_points"].eq(core["expected_evaluation_points"])
        & np.isfinite(core["learned_rmse"])
    )
    return {
        "forecast_core_all_finite": bool(len(core) and complete_predictions.all()),
        "forecast_core_mean_rmse_below_1K": _json_bool(core["learned_rmse"].mean() < 1.0),
        "forecast_core_worst_rmse_below_1_5K": bool(core["learned_rmse"].max() < 1.5),
        "forecast_core_beats_engineering_prior": _json_bool(
            core["learned_rmse"].mean() < core["prior_rmse"].mean()
        ),
        "forecast_core_beats_persistence": _json_bool(
            core["learned_rmse"].mean() < core["persistence_rmse"].mean()
        ),
        "forecast_history_reduces_sparse_initialization_error": bool(
            sparse_history["learned_rmse"] < 0.25 * sparse_initial["learned_rmse"]
        ),
        "nonlinear_model_gap_is_visible": bool(
            gap["learned_rmse"] > max(0.5, 2.0 * core["learned_rmse"].mean())
        ),
        "monitor_nominal_innovation_below_0_35K": bool(nominal["innovation_rmse"] < 0.35),
        "monitor_sensor_bias_rmse_below_0_15K": bool(
            drift["sensor_bias_rmse_after_burn_in"] < 0.15
        ),
        "monitor_sensor_fault_detected_within_2s": bool(fault["detection_delay"] <= 2.0),
        "monitor_sensor_fault_localized": bool(fault["first_alert_sensor"] == "CP"),
        "monitor_missing_data_remains_finite": bool(missing["posterior_finite"]),
        "monitor_missing_data_rmse_below_0_25K": bool(missing["posterior_physical_rmse"] < 0.25),
        "monitor_heat_load_detected_within_5s": bool(disturbance["detection_delay"] <= 5.0),
        "monitor_heat_load_attributed_to_physical_disturbance": bool(
            disturbance["event_disturbance_rmse_w"] < 0.25
        ),
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

    forecasts, model_comparison, prediction_cases = evaluate_forecasts(
        artifact,
        prior_model,
        resolve_runtime_values(config, "forecast", artifact_metadata=artifact.metadata),
        ROOT / "config.yaml",
    )
    forecast_groups = summarize_forecast_groups(forecasts)
    monitors = evaluate_monitors(
        artifact,
        resolve_runtime_values(config, "monitor", artifact_metadata=artifact.metadata),
        ROOT / "config.yaml",
    )
    parameters = parameter_recovery(artifact)
    checks = acceptance_checks(forecasts, monitors)

    forecasts.to_csv(OUTPUT_DIR / "forecast_by_case.csv", index=False)
    forecast_groups.to_csv(OUTPUT_DIR / "forecast_by_group.csv", index=False)
    model_comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    monitors.to_csv(OUTPUT_DIR / "monitor_by_case.csv", index=False)
    parameters.to_csv(OUTPUT_DIR / "parameter_recovery.csv", index=False)
    write_case_catalog(forecasts, monitors, OUTPUT_DIR)
    _write_evidence_figures(prediction_cases, forecasts, artifact.sensor_names)

    core = forecasts.loc[forecasts["group"] != "model_gap"]
    summary = {
        "benchmark_version": 2,
        "evaluation_boundary": "external cases excluded from training discovery",
        "status": "pass" if all(checks.values()) else "review",
        "checks": checks,
        "forecast": {
            "n_core_cases": len(core),
            "mean_core_rmse": float(core["learned_rmse"].mean()),
            "worst_core_rmse": float(core["learned_rmse"].max()),
            "mean_prior_rmse": float(core["prior_rmse"].mean()),
            "mean_persistence_rmse": float(core["persistence_rmse"].mean()),
            "model_gap_rmse": float(_row_by_group(forecasts, "model_gap")["learned_rmse"]),
            "prediction_figures": {
                "core_timeseries": validation_figure_reference(
                    "topcell_core_prediction_timeseries.png"
                ),
                "core_parity": validation_figure_reference("topcell_core_prediction_parity.png"),
                "model_gap_timeseries": validation_figure_reference(
                    "topcell_model_gap_prediction_timeseries.png"
                ),
                "model_gap_parity": validation_figure_reference(
                    "topcell_model_gap_prediction_parity.png"
                ),
            },
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
