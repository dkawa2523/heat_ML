"""Evaluate the fitted thermal network across the nonlinear CAE screening set."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from celltemp.analysis import max_abs_error as _max_abs
from celltemp.analysis import (
    persistence_prediction,
    prediction_comparison_rows,
    prediction_error_metrics,
    prediction_sensor_rows,
    residual_dependence_rows,
)
from celltemp.analysis import rmse as _rmse
from celltemp.artifact import fitted_parameters, load_artifact
from celltemp.config import as_path, load_config
from celltemp.engine import ThermalRCModel
from celltemp.inference import forecast
from celltemp.io import load_system_spec, trajectory_from_frame
from celltemp.workflows import run_train
from external_tools.comsol_chip_cooling.benchmark_nonlinear_output import (
    publish_benchmark,
)
from external_tools.comsol_chip_cooling.evaluation_support import (
    json_records,
    percent_improvement,
)

SENSORS = ("chip", "sink_base", "fins")
CONTROLS = ("chip_power", "coolant_temperature", "inlet_air_velocity")
CONTROL_UNITS = {
    "chip_power": "W",
    "coolant_temperature": "degC",
    "inlet_air_velocity": "m/s",
}


def _truth(frame: pd.DataFrame) -> np.ndarray:
    truth_columns = [f"truth_{sensor}" for sensor in SENSORS]
    columns = truth_columns if set(truth_columns) <= set(frame.columns) else list(SENSORS)
    return frame[columns].to_numpy(dtype=np.float64)


def _request(case_id: str, frame: pd.DataFrame):
    request = frame[["time", *SENSORS, *CONTROLS]].copy()
    request.loc[request.index[1:], list(SENSORS)] = np.nan
    return trajectory_from_frame(
        case_id=case_id,
        frame=request,
        time_col="time",
        sensor_cols=SENSORS,
        control_cols=CONTROLS,
        control_convention="left",
    )


def _predict(model: ThermalRCModel, case_id: str, frame: pd.DataFrame) -> np.ndarray:
    result = forecast(model, _request(case_id, frame))
    time = frame["time"].to_numpy(dtype=np.float64)
    if not np.allclose(result.time, time, rtol=0.0, atol=1e-10):
        raise ValueError(f"{case_id}: forecast output is not time-aligned")
    if not np.isfinite(result.sensor_temperature).all():
        raise ValueError(f"{case_id}: forecast contains non-finite values")
    return result.sensor_temperature


def _load_frames(directory: Path, *, evaluation: bool) -> dict[str, pd.DataFrame]:
    paths = sorted(directory.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no nonlinear CAE cases under {directory}")
    required = {
        "time",
        *SENSORS,
        *CONTROLS,
        "truth_pressure_drop",
        "truth_radiative_heat_rate",
        "case_id",
        "case_group",
        "radiation_enabled",
    }
    if evaluation:
        required.update(f"truth_{sensor}" for sensor in SENSORS)
    frames: dict[str, pd.DataFrame] = {}
    for path in paths:
        frame = pd.read_csv(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path.stem}: missing benchmark columns {missing}")
        case_id = str(frame["case_id"].iat[0])
        if case_id != path.stem:
            raise ValueError(f"{path.name}: case_id does not match the file name")
        if not frame[list(SENSORS)].iloc[0].notna().all():
            raise ValueError(f"{case_id}: initial observations must be complete")
        if evaluation and frame[list(SENSORS)].iloc[1:].notna().any().any():
            raise ValueError(f"{case_id}: future observations would leak CAE truth")
        finite_columns = [*CONTROLS, "truth_pressure_drop", "truth_radiative_heat_rate"]
        if not np.isfinite(frame[finite_columns]).all().all():
            raise ValueError(f"{case_id}: controls and physical diagnostics must be finite")
        if not np.isfinite(_truth(frame)).all():
            raise ValueError(f"{case_id}: truth temperatures must be finite")
        frames[case_id] = frame
    return frames


def _prediction_frame(
    source: pd.DataFrame,
    predicted: np.ndarray,
    prior: np.ndarray,
    persistence: np.ndarray,
) -> pd.DataFrame:
    result = source[["time", *CONTROLS, "truth_pressure_drop", "truth_radiative_heat_rate"]].copy()
    truth = _truth(source)
    for index, sensor in enumerate(SENSORS):
        result[f"truth_{sensor}"] = truth[:, index]
        result[f"predicted_{sensor}"] = predicted[:, index]
        result[f"error_{sensor}"] = predicted[:, index] - truth[:, index]
        result[f"prior_{sensor}"] = prior[:, index]
        result[f"persistence_{sensor}"] = persistence[:, index]
    return result


def _case_row(
    case_id: str,
    source: pd.DataFrame,
    truth: np.ndarray,
    predicted: np.ndarray,
    prior: np.ndarray,
    persistence: np.ndarray,
) -> dict[str, Any]:
    error = predicted[1:] - truth[1:]
    fitted = prediction_error_metrics(truth[1:], predicted[1:])
    prior_metrics = prediction_error_metrics(truth[1:], prior[1:])
    persistence_metrics = prediction_error_metrics(truth[1:], persistence[1:])
    fitted_rmse = float(fitted["rmse_k"])
    prior_rmse = float(prior_metrics["rmse_k"])
    persistence_rmse = float(persistence_metrics["rmse_k"])
    chip_hotspot = (
        source["truth_chip_max"].to_numpy(dtype=np.float64)[1:]
        if "truth_chip_max" in source
        else truth[1:, 0]
    )
    return {
        "case_id": case_id,
        "case_group": str(source["case_group"].iat[0]),
        "radiation_enabled": bool(source["radiation_enabled"].iat[0]),
        "n_forecast_timepoints": len(error),
        "rmse_k": fitted_rmse,
        "mae_k": fitted["mae_k"],
        "bias_k": fitted["bias_k"],
        "max_abs_error_k": fitted["max_abs_error_k"],
        "prior_rmse_k": prior_rmse,
        "persistence_rmse_k": persistence_rmse,
        "rmse_improvement_percent": percent_improvement(prior_rmse, fitted_rmse),
        "rmse_improvement_vs_persistence_percent": percent_improvement(
            persistence_rmse, fitted_rmse
        ),
        "chip_peak_error_k": float(np.max(predicted[:, 0]) - np.max(truth[:, 0])),
        "max_hotspot_underprediction_k": float(max(0.0, np.max(chip_hotspot - predicted[1:, 0]))),
    }


def _sensor_rows(
    case_id: str,
    source: pd.DataFrame,
    truth: np.ndarray,
    predicted: np.ndarray,
    prior: np.ndarray,
    persistence: np.ndarray,
) -> list[dict[str, Any]]:
    time = source["time"].to_numpy(dtype=np.float64)[1:]
    rows: list[dict[str, Any]] = []
    for index, base in enumerate(prediction_sensor_rows(time, truth[1:], predicted[1:], SENSORS)):
        prior_rmse = _rmse(prior[1:, index] - truth[1:, index])
        persistence_rmse = _rmse(persistence[1:, index] - truth[1:, index])
        fitted_rmse = float(base["rmse_k"])
        rows.append(
            {
                "case_id": case_id,
                "case_group": str(source["case_group"].iat[0]),
                **base,
                "prior_rmse_k": prior_rmse,
                "persistence_rmse_k": persistence_rmse,
                "rmse_improvement_percent": percent_improvement(prior_rmse, fitted_rmse),
                "rmse_improvement_vs_persistence_percent": percent_improvement(
                    persistence_rmse, fitted_rmse
                ),
            }
        )
    return rows


def _residual_rows(
    case_id: str,
    source: pd.DataFrame,
    truth: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, Any]]:
    conditions = {
        control: (source[control].to_numpy(dtype=np.float64)[1:], CONTROL_UNITS[control])
        for control in CONTROLS
    }
    conditions["pressure_drop"] = (
        source["truth_pressure_drop"].to_numpy(dtype=np.float64)[1:],
        "Pa",
    )
    conditions["radiative_heat_rate"] = (
        source["truth_radiative_heat_rate"].to_numpy(dtype=np.float64)[1:],
        "W",
    )
    return [
        {
            "case_id": case_id,
            "case_group": str(source["case_group"].iat[0]),
            **row,
        }
        for row in residual_dependence_rows(
            source["time"].to_numpy(dtype=np.float64)[1:],
            truth[1:],
            predicted[1:],
            SENSORS,
            conditions=conditions,
            temperature_unit="degC",
        )
    ]


def _group_rows(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, values in cases.groupby("case_group", sort=True):
        rows.append(
            {
                "case_group": group,
                "n_cases": len(values),
                "mean_rmse_k": float(values["rmse_k"].mean()),
                "worst_rmse_k": float(values["rmse_k"].max()),
                "mean_prior_rmse_k": float(values["prior_rmse_k"].mean()),
                "mean_persistence_rmse_k": float(values["persistence_rmse_k"].mean()),
                "worst_case": str(values.loc[values["rmse_k"].idxmax(), "case_id"]),
            }
        )
    return pd.DataFrame(rows)


def _pair_rows(
    pair_table: pd.DataFrame,
    evaluation_frames: dict[str, pd.DataFrame],
    base_frames: dict[str, pd.DataFrame],
    predictions: dict[str, np.ndarray],
    model: ThermalRCModel,
) -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []
    aligned = True
    for item in pair_table.itertuples(index=False):
        radiation_id = str(item.radiation_case_id)
        base_id = str(item.base_case_id)
        radiation = evaluation_frames[radiation_id]
        base = base_frames[base_id]
        columns = ["time", *CONTROLS]
        inputs_match = np.allclose(
            radiation[columns].to_numpy(dtype=np.float64),
            base[columns].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-10,
        )
        aligned &= inputs_match
        if not inputs_match:
            raise ValueError(f"{radiation_id}: radiation/base time or controls do not match")
        radiation_truth = _truth(radiation)
        base_truth = _truth(base)
        radiation_prediction = predictions[radiation_id]
        base_prediction = _predict(model, base_id, base)
        truth_delta = radiation_truth - base_truth
        predicted_delta = radiation_prediction - base_prediction
        for index, sensor in enumerate(SENSORS):
            delta_error = predicted_delta[1:, index] - truth_delta[1:, index]
            terminal_truth = float(truth_delta[-1, index])
            terminal_predicted = float(predicted_delta[-1, index])
            rows.append(
                {
                    "radiation_case_id": radiation_id,
                    "base_case_id": base_id,
                    "sensor": sensor,
                    "initial_truth_delta_k": float(truth_delta[0, index]),
                    "truth_terminal_delta_k": terminal_truth,
                    "predicted_terminal_delta_k": terminal_predicted,
                    "max_abs_truth_delta_k": _max_abs(truth_delta[:, index]),
                    "max_abs_predicted_delta_k": _max_abs(predicted_delta[:, index]),
                    "pair_delta_rmse_k": _rmse(delta_error),
                    "pair_delta_max_abs_error_k": _max_abs(delta_error),
                    "terminal_predicted_to_truth_ratio": (
                        terminal_predicted / terminal_truth
                        if abs(terminal_truth) > 1e-12
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows), aligned


def _summary(
    artifact,
    artifact_source: str,
    cases: pd.DataFrame,
    groups: pd.DataFrame,
    pair: pd.DataFrame,
    pair_aligned: bool,
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    controls_complete = artifact.control_names == CONTROLS
    workflow_pass = (
        controls_complete
        and pair_aligned
        and all(np.isfinite(value).all() for value in predictions.values())
    )
    non_radiation = cases[~cases["radiation_enabled"]]
    radiation = cases[cases["radiation_enabled"]]

    def subset_metrics(values: pd.DataFrame) -> dict[str, Any]:
        return {
            "n_cases": len(values),
            "mean_rmse_k": float(values["rmse_k"].mean()),
            "worst_case": str(values.loc[values["rmse_k"].idxmax(), "case_id"]),
            "worst_rmse_k": float(values["rmse_k"].max()),
            "mean_prior_rmse_k": float(values["prior_rmse_k"].mean()),
            "mean_persistence_rmse_k": float(values["persistence_rmse_k"].mean()),
        }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Open-loop model-form screening across nonlinear interpolation, extrapolation, "
            "pulse, hot-start, and paired-radiation CAE cases."
        ),
        "data_quality": {
            "mesh_profile": "global-8",
            "mesh_qualified": False,
            "temporal_qualified": False,
            "experiment_compared": False,
            "permitted_use": "model-form screening",
        },
        "workflow": {
            "status": "pass" if workflow_pass else "fail",
            "artifact_controls": list(artifact.control_names),
            "complete_external_control_boundary": controls_complete,
            "forecast_truth_columns_excluded_by_request": True,
            "all_predictions_finite": all(
                np.isfinite(value).all() for value in predictions.values()
            ),
            "radiation_pair_time_and_controls_identical": pair_aligned,
        },
        "model": {
            "type": "three-node thermal network using shared scalar response laws",
            "artifact": artifact_source,
            "fitted_parameters": fitted_parameters(artifact.model),
        },
        "evaluation": {
            "n_cases": len(cases),
            "groups": json_records(groups),
            "non_radiation_holdouts": subset_metrics(non_radiation),
            "radiation_model_gap": subset_metrics(radiation),
            "mean_case_rmse_k": float(cases["rmse_k"].mean()),
            "worst_case": str(cases.loc[cases["rmse_k"].idxmax(), "case_id"]),
            "worst_case_rmse_k": float(cases["rmse_k"].max()),
            "mean_prior_rmse_k": float(cases["prior_rmse_k"].mean()),
            "mean_persistence_rmse_k": float(cases["persistence_rmse_k"].mean()),
            "residual_diagnostics": {
                "sign_convention": "predicted_minus_truth",
                "forecast_origin_excluded": True,
                "interpretation": (
                    "descriptive screening only; protocol covariation prevents causal attribution"
                ),
            },
            "prediction_figures": {
                "non_radiation_timeseries": "figures/core_prediction_timeseries.png",
                "non_radiation_parity": "figures/core_prediction_parity.png",
                "radiation_model_gap_timeseries": "figures/model_gap_prediction_timeseries.png",
                "radiation_model_gap_parity": "figures/model_gap_prediction_parity.png",
            },
        },
        "radiation_pairs": {
            "n_pairs": int(pair["radiation_case_id"].nunique()),
            "max_abs_truth_effect_k": float(pair["max_abs_truth_delta_k"].max()),
            "mean_pair_delta_rmse_k": float(pair["pair_delta_rmse_k"].mean()),
            "interpretation": "directional screening only on a non-converged mesh",
        },
    }


def run_benchmark(
    config_path: Path,
    output: Path,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parent
    cfg = load_config(config_path)
    if artifact_path is None:
        run_dir = run_train(cfg, config_path)
        artifact_source = run_dir / "artifact"
    else:
        artifact_source = artifact_path.resolve()
    artifact = load_artifact(artifact_source)
    if artifact.sensor_names != SENSORS or artifact.control_names != CONTROLS:
        raise ValueError("nonlinear benchmark artifact does not match the three-sensor boundary")
    prior = ThermalRCModel(
        load_system_spec(as_path(str(cfg["system"]), root)),
        integrator=str(cfg.get("engine", {}).get("integrator", "exact")),
    )

    forecast_frames = _load_frames(root / "data/nonlinear/eval/forecast", evaluation=True)
    radiation_frames = _load_frames(root / "data/nonlinear/eval/model_gap", evaluation=True)
    evaluation_frames = {**forecast_frames, **radiation_frames}
    train_frames = _load_frames(root / "data/nonlinear/train", evaluation=False)
    base_frames = {**train_frames, **forecast_frames}
    pair_table = pd.read_csv(root / "data/nonlinear/radiation_pairs.csv")

    predictions: dict[str, np.ndarray] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    case_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for case_id, source in evaluation_frames.items():
        truth = _truth(source)
        predicted = _predict(artifact.model, case_id, source)
        prior_prediction = _predict(prior, case_id, source)
        observed = source[list(SENSORS)].to_numpy(dtype=np.float64)
        persistence = persistence_prediction(observed, np.isfinite(observed), origin=0)
        predictions[case_id] = predicted
        prediction_frames[case_id] = _prediction_frame(
            source, predicted, prior_prediction, persistence
        )
        case_rows.append(
            _case_row(case_id, source, truth, predicted, prior_prediction, persistence)
        )
        sensor_rows.extend(
            _sensor_rows(case_id, source, truth, predicted, prior_prediction, persistence)
        )
        comparison_rows.extend(
            {
                "case_id": case_id,
                "case_group": str(source["case_group"].iat[0]),
                "radiation_enabled": bool(source["radiation_enabled"].iat[0]),
                **comparison,
            }
            for comparison in prediction_comparison_rows(
                source["time"].to_numpy(dtype=np.float64)[1:],
                truth[1:],
                {
                    "fitted_rc": predicted[1:],
                    "engineering_prior_rc": prior_prediction[1:],
                    "persistence": persistence[1:],
                },
                SENSORS,
            )
        )
        residual_rows.extend(_residual_rows(case_id, source, truth, predicted))

    cases = pd.DataFrame(case_rows)
    sensors = pd.DataFrame(sensor_rows)
    model_comparison = pd.DataFrame(comparison_rows)
    residuals = pd.DataFrame(residual_rows)
    groups = _group_rows(cases)
    pair, pair_aligned = _pair_rows(
        pair_table, evaluation_frames, base_frames, predictions, artifact.model
    )
    summary = _summary(
        artifact,
        (
            artifact_source.relative_to(root).as_posix()
            if artifact_source.is_relative_to(root)
            else str(artifact_source)
        ),
        cases,
        groups,
        pair,
        pair_aligned,
        predictions,
    )

    publish_benchmark(
        output,
        case_metrics=cases,
        group_metrics=groups,
        sensor_metrics=sensors,
        model_comparison=model_comparison,
        residual_dependence=residuals,
        radiation_pair_metrics=pair,
        prediction_frames=prediction_frames,
        summary=summary,
        sensor_names=SENSORS,
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_root / "high_fidelity_benchmark.yaml",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        help="Reuse an existing fitted artifact; otherwise train from the configured split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "data/nonlinear/benchmark",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_benchmark(args.config, args.output, args.artifact)
    print(json.dumps(summary["workflow"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["evaluation"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["radiation_pairs"], indent=2, ensure_ascii=False))
    print(f"saved benchmark: {args.output.resolve()}")
    return 0 if summary["workflow"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
