"""Forecast tables for the linear COMSOL holdout evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from celltemp.analysis import (
    persistence_prediction,
    prediction_comparison_rows,
    prediction_error_metrics,
)
from celltemp.artifact import load_artifact
from celltemp.engine import ThermalRCModel
from celltemp.inference import forecast
from celltemp.io import load_system_spec, trajectory_from_frame
from celltemp.workflows.prediction_figures import PredictionCase
from external_tools.comsol_chip_cooling.evaluation_support import (
    assert_aligned,
    percent_improvement,
)

SENSORS = ("chip", "sink_base", "fins")
CONTROLS = ("chip_power", "coolant_temperature")


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
    predicted = forecast(model, _request_from_frame(case_id, frame)).sensor_temperature
    if not np.isfinite(predicted).all():
        raise ValueError(f"{case_id}: forecast contains a non-finite temperature")
    return predicted


def _sensor_rows(
    case_id: str,
    case_group: str,
    truth: np.ndarray,
    predicted: np.ndarray,
    prior: np.ndarray,
    persistence: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sensor in enumerate(SENSORS):
        truth_values = truth[1:, index]
        fitted = prediction_error_metrics(truth_values, predicted[1:, index])
        prior_metrics = prediction_error_metrics(truth_values, prior[1:, index])
        persistence_metrics = prediction_error_metrics(truth_values, persistence[1:, index])
        fitted_rmse = float(fitted["rmse_k"])
        prior_rmse = float(prior_metrics["rmse_k"])
        persistence_rmse = float(persistence_metrics["rmse_k"])
        rows.append(
            {
                "case_id": case_id,
                "case_group": case_group,
                "sensor": sensor,
                "n_points": fitted["n_points"],
                "rmse_k": fitted_rmse,
                "mae_k": fitted["mae_k"],
                "max_abs_error_k": fitted["max_abs_error_k"],
                "terminal_error_k": float(predicted[-1, index] - truth[-1, index]),
                "prior_rmse_k": prior_rmse,
                "persistence_rmse_k": persistence_rmse,
                "rmse_improvement_percent": percent_improvement(prior_rmse, fitted_rmse),
                "rmse_improvement_vs_persistence_percent": percent_improvement(
                    persistence_rmse, fitted_rmse
                ),
            }
        )
    return rows


def _case_row(
    case_id: str,
    case_group: str,
    source: pd.DataFrame,
    truth: np.ndarray,
    predicted: np.ndarray,
    prior: np.ndarray,
    persistence: np.ndarray,
) -> dict[str, Any]:
    fitted = prediction_error_metrics(truth[1:], predicted[1:])
    prior_metrics = prediction_error_metrics(truth[1:], prior[1:])
    persistence_metrics = prediction_error_metrics(truth[1:], persistence[1:])
    fitted_rmse = float(fitted["rmse_k"])
    prior_rmse = float(prior_metrics["rmse_k"])
    persistence_rmse = float(persistence_metrics["rmse_k"])
    chip_truth = truth[:, 0]
    chip_prediction = predicted[:, 0]
    spatial_gap = source["truth_chip_max"].to_numpy(dtype=np.float64) - chip_truth
    hotspot_error = source["truth_chip_max"].to_numpy(dtype=np.float64) - chip_prediction
    return {
        "case_id": case_id,
        "case_group": case_group,
        "n_rows": len(source),
        "rmse_k": fitted_rmse,
        "mae_k": fitted["mae_k"],
        "max_abs_error_k": fitted["max_abs_error_k"],
        "prior_rmse_k": prior_rmse,
        "persistence_rmse_k": persistence_rmse,
        "rmse_improvement_percent": percent_improvement(prior_rmse, fitted_rmse),
        "rmse_improvement_vs_persistence_percent": percent_improvement(
            persistence_rmse, fitted_rmse
        ),
        "chip_peak_error_k": float(np.max(chip_prediction) - np.max(chip_truth)),
        "max_spatial_hotspot_gap_k": float(np.max(spatial_gap)),
        "max_hotspot_underprediction_k": float(np.max(hotspot_error)),
    }


def evaluate_forecasts(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, list[PredictionCase]]:
    """Evaluate saved forecasts against truth and the two standard baselines."""
    source_dir = root / "data" / "eval" / "forecast"
    result_dir = root / "work" / "outputs" / "forecast"
    artifact_path = root / "work" / "outputs" / "runs" / "comsol_chip_cooling" / "artifact"
    artifact = load_artifact(artifact_path)
    prior_model = ThermalRCModel(load_system_spec(root / "system.yaml"), integrator="exact")
    case_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    prediction_cases: list[PredictionCase] = []
    leakage_check = True

    for source_path in sorted(source_dir.glob("*.csv")):
        case_id = source_path.stem
        source = pd.read_csv(source_path)
        result = pd.read_csv(result_dir / "cases" / source_path.name)
        assert_aligned(source, result, case_id)
        written = result[[f"sensor.{sensor}.temperature" for sensor in SENSORS]].to_numpy()
        predicted = _model_forecast(artifact.model, case_id, source)
        if not np.allclose(written, predicted, rtol=1e-10, atol=1e-10):
            raise ValueError(f"{case_id}: saved forecast differs from the saved artifact")
        if not np.isfinite(predicted).all():
            raise ValueError(f"{case_id}: forecast contains a non-finite temperature")
        prior = _model_forecast(prior_model, case_id, source)
        truth = source[[f"truth_{sensor}" for sensor in SENSORS]].to_numpy(dtype=np.float64)
        observed = source[list(SENSORS)].to_numpy(dtype=np.float64)
        persistence = persistence_prediction(observed, np.isfinite(observed), origin=0)
        case_group = str(source["case_group"].iloc[0])
        case_rows.append(
            _case_row(case_id, case_group, source, truth, predicted, prior, persistence)
        )
        sensor_rows.extend(_sensor_rows(case_id, case_group, truth, predicted, prior, persistence))
        comparison_rows.extend(
            {
                "case_id": case_id,
                "case_group": case_group,
                **comparison,
            }
            for comparison in prediction_comparison_rows(
                source["time"].to_numpy(dtype=np.float64)[1:],
                truth[1:],
                {
                    "fitted_rc": predicted[1:],
                    "engineering_prior_rc": prior[1:],
                    "persistence": persistence[1:],
                },
                SENSORS,
            )
        )
        prediction_cases.append(
            PredictionCase(
                case_id=case_id,
                time=source["time"].to_numpy(dtype=np.float64),
                truth=truth,
                predicted=predicted,
            )
        )

        if case_id == "F01_power_interpolation":
            altered = source.copy()
            truth_columns = [column for column in altered.columns if column.startswith("truth_")]
            for column in truth_columns:
                altered[column] = source[column].to_numpy(dtype=np.float64) + 10_000.0
            leakage_check = np.array_equal(
                predicted, _model_forecast(artifact.model, case_id, altered)
            )

    return (
        pd.DataFrame(case_rows),
        pd.DataFrame(sensor_rows),
        pd.DataFrame(comparison_rows),
        leakage_check,
        prediction_cases,
    )
