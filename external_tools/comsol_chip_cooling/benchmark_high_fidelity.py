"""Evaluate the shared thermal network against the local-mesh nonlinear CAE pair."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.validation_figures import validation_figure_reference
from celltemp.analysis import mae as _mae
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
from celltemp.workflows import run_forecast, run_train
from external_tools.comsol_chip_cooling.benchmark_high_fidelity_output import publish_benchmark
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
CASE_LOCATIONS = {
    "HV01_composite_conjugate": Path(
        "data/nonlinear_high_fidelity/dynamic/eval/forecast/HV01_composite_conjugate.csv"
    ),
    "HV02_composite_radiation": Path(
        "data/nonlinear_high_fidelity/dynamic/eval/model_gap/HV02_composite_radiation.csv"
    ),
}
RESPONSE_PHASES = (
    ("initialization", 0.0, 60.0),
    ("power_excitation", 60.0, 120.0),
    ("airflow_excitation", 120.0, 160.0),
    ("coupled_hot_low_flow", 160.0, 220.0),
)


def _request(case_id: str, frame: pd.DataFrame):
    return trajectory_from_frame(
        case_id=case_id,
        frame=frame,
        time_col="time",
        sensor_cols=SENSORS,
        control_cols=CONTROLS,
        control_convention="left",
    )


def _predict(model: ThermalRCModel, case_id: str, frame: pd.DataFrame) -> np.ndarray:
    return forecast(model, _request(case_id, frame)).sensor_temperature


def _load_cases(root: Path) -> dict[str, pd.DataFrame]:
    frames = {case_id: pd.read_csv(root / relative) for case_id, relative in CASE_LOCATIONS.items()}
    required = {
        "time",
        *SENSORS,
        *CONTROLS,
        *(f"truth_{sensor}" for sensor in SENSORS),
        "truth_chip_max",
        "truth_pressure_drop",
        "truth_radiative_heat_rate",
    }
    for case_id, frame in frames.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{case_id}: missing benchmark columns {missing}")
        if not frame[list(SENSORS)].iloc[0].notna().all():
            raise ValueError(f"{case_id}: initial observations must be complete")
        if frame[list(SENSORS)].iloc[1:].notna().any().any():
            raise ValueError(f"{case_id}: future observations would leak CAE truth")
        if not np.isfinite(frame[[*(f"truth_{name}" for name in SENSORS), *CONTROLS]]).all().all():
            raise ValueError(f"{case_id}: truth and controls must be finite")
    return frames


def _validate_pair(frames: dict[str, pd.DataFrame]) -> bool:
    base = frames["HV01_composite_conjugate"]
    radiation = frames["HV02_composite_radiation"]
    columns = ["time", *CONTROLS]
    return np.allclose(
        base[columns].to_numpy(dtype=np.float64),
        radiation[columns].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-10,
    )


def _experiment_status(root: Path) -> tuple[bool, bool]:
    path = root / "data/nonlinear_high_fidelity/experiment/result/validation_status.json"
    if not path.exists():
        return False, False
    status = json.loads(path.read_text(encoding="utf-8"))
    compared = status.get("status") == "evaluated"
    accepted = compared and status.get("acceptance_passed") is True
    return compared, accepted


def _reference_quality(root: Path, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    path = root / "data/nonlinear_high_fidelity/dynamic/cae_reference.csv"
    reference = pd.read_csv(path)
    radiation = frames["HV02_composite_radiation"]
    columns = ["time", *SENSORS, *CONTROLS]
    expected = radiation[["time", *(f"truth_{name}" for name in SENSORS), *CONTROLS]].copy()
    expected.columns = columns
    reference_matches = np.allclose(
        reference[columns].to_numpy(dtype=np.float64),
        expected.to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-9,
    )
    uncertainty = {
        sensor: float(reference[f"mesh_uncertainty_{sensor}"].max()) for sensor in SENSORS
    }
    experiment_compared, experiment_validated = _experiment_status(root)
    return {
        "dynamic_reference_matches_hv02": reference_matches,
        "mesh_profile": str(reference["mesh_profile"].iat[0]),
        "mesh_qualified": bool(reference["mesh_qualified"].all()),
        "benchmark_qualified": bool(reference["benchmark_qualified"].all()),
        "temporal_qualified": bool(reference["temporal_qualified"].all()),
        "experiment_compared": experiment_compared,
        "experiment_validated": experiment_validated,
        "mesh_difference_k": uncertainty,
    }


def _forecast_directories(cfg: dict[str, Any], root: Path) -> dict[str, Path]:
    conjugate = as_path(str(cfg["forecast"]["output_dir"]), root)
    radiation = root / "work/high_fidelity_benchmark/forecast/radiation"
    return {
        "HV01_composite_conjugate": conjugate,
        "HV02_composite_radiation": radiation,
    }


def _run_workflows(cfg: dict[str, Any], config_path: Path) -> tuple[Path, dict[str, Path]]:
    root = config_path.parent
    run_dir = run_train(cfg, config_path)
    output_dirs = _forecast_directories(cfg, root)
    run_forecast(cfg, config_path)
    radiation_cfg = copy.deepcopy(cfg)
    radiation_cfg["forecast"]["input_dir"] = str(CASE_LOCATIONS["HV02_composite_radiation"].parent)
    radiation_cfg["forecast"]["output_dir"] = str(
        output_dirs["HV02_composite_radiation"].relative_to(root)
    )
    run_forecast(radiation_cfg, config_path)
    return run_dir, output_dirs


def _saved_prediction(
    case_id: str,
    frame: pd.DataFrame,
    directory: Path,
    model: ThermalRCModel,
) -> np.ndarray:
    result = pd.read_csv(directory / f"{case_id}.csv")
    if not np.allclose(result["time"], frame["time"], rtol=0.0, atol=1e-10):
        raise ValueError(f"{case_id}: forecast output is not time-aligned")
    columns = [f"temperature_{sensor}" for sensor in SENSORS]
    saved = result[columns].to_numpy(dtype=np.float64)
    direct = _predict(model, case_id, frame)
    if not np.allclose(saved, direct, rtol=1e-10, atol=1e-10):
        raise ValueError(f"{case_id}: public forecast differs from artifact replay")
    if not np.isfinite(saved).all():
        raise ValueError(f"{case_id}: forecast contains non-finite values")
    return saved


def _prediction_frame(
    source: pd.DataFrame,
    predicted: np.ndarray,
    prior: np.ndarray,
    persistence: np.ndarray,
    uncertainty: dict[str, float],
) -> pd.DataFrame:
    result = source[["time", *CONTROLS]].copy()
    for index, sensor in enumerate(SENSORS):
        truth = source[f"truth_{sensor}"].to_numpy(dtype=np.float64)
        result[f"truth_{sensor}"] = truth
        result[f"predicted_{sensor}"] = predicted[:, index]
        result[f"error_{sensor}"] = predicted[:, index] - truth
        result[f"prior_{sensor}"] = prior[:, index]
        result[f"persistence_{sensor}"] = persistence[:, index]
        result[f"mesh_difference_{sensor}"] = uncertainty[sensor]
    result["truth_chip_max"] = source["truth_chip_max"]
    result["hotspot_underprediction"] = source["truth_chip_max"] - predicted[:, 0]
    return result


def _sensor_rows(
    case_id: str,
    time: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    prior: np.ndarray,
    persistence: np.ndarray,
    uncertainty: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_rows = prediction_sensor_rows(time[1:], truth[1:], predicted[1:], SENSORS)
    for index, base in enumerate(base_rows):
        sensor = SENSORS[index]
        error = predicted[1:, index] - truth[1:, index]
        prior_error = prior[1:, index] - truth[1:, index]
        persistence_error = persistence[1:, index] - truth[1:, index]
        fitted_rmse = float(base["rmse_k"])
        prior_rmse = _rmse(prior_error)
        persistence_rmse = _rmse(persistence_error)
        mesh_difference = uncertainty[sensor]
        rows.append(
            {
                "case_id": case_id,
                **base,
                "prior_rmse_k": prior_rmse,
                "persistence_rmse_k": persistence_rmse,
                "rmse_improvement_percent": percent_improvement(prior_rmse, fitted_rmse),
                "rmse_improvement_vs_persistence_percent": percent_improvement(
                    persistence_rmse, fitted_rmse
                ),
                "mesh_difference_k": mesh_difference,
                "rmse_over_mesh_difference": fitted_rmse / mesh_difference,
                "max_abs_error_over_mesh_difference": _max_abs(error) / mesh_difference,
                "fraction_abs_error_within_mesh_difference": float(
                    np.mean(np.abs(error) <= mesh_difference)
                ),
                "model_error_resolved_above_mesh_difference": fitted_rmse > mesh_difference,
                "pointwise_error_resolved_above_mesh_difference": (
                    _max_abs(error) > mesh_difference
                ),
            }
        )
    return rows


def _residual_dependence(
    case_id: str,
    source: pd.DataFrame,
    truth: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, Any]]:
    time = source["time"].to_numpy(dtype=np.float64)[1:]
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
        {"case_id": case_id, **row}
        for row in residual_dependence_rows(
            time,
            truth[1:],
            predicted[1:],
            SENSORS,
            conditions=conditions,
            temperature_unit="degC",
        )
    ]


def _case_row(
    case_id: str,
    source: pd.DataFrame,
    truth: np.ndarray,
    predicted: np.ndarray,
    prior: np.ndarray,
    persistence: np.ndarray,
    uncertainty: dict[str, float],
) -> dict[str, Any]:
    error = predicted[1:] - truth[1:]
    fitted = prediction_error_metrics(truth[1:], predicted[1:])
    prior_metrics = prediction_error_metrics(truth[1:], prior[1:])
    persistence_metrics = prediction_error_metrics(truth[1:], persistence[1:])
    fitted_rmse = float(fitted["rmse_k"])
    prior_rmse = float(prior_metrics["rmse_k"])
    persistence_rmse = float(persistence_metrics["rmse_k"])
    mesh = np.asarray([uncertainty[name] for name in SENSORS])
    hotspot_error = source["truth_chip_max"].to_numpy(dtype=np.float64)[1:] - predicted[1:, 0]
    return {
        "case_id": case_id,
        "radiation_enabled": bool(source["radiation_enabled"].iat[0]),
        "n_forecast_timepoints": len(error),
        "n_scalar_predictions": int(error.size),
        "rmse_k": fitted_rmse,
        "mae_k": fitted["mae_k"],
        "max_abs_error_k": fitted["max_abs_error_k"],
        "prior_rmse_k": prior_rmse,
        "persistence_rmse_k": persistence_rmse,
        "rmse_improvement_percent": percent_improvement(prior_rmse, fitted_rmse),
        "rmse_improvement_vs_persistence_percent": percent_improvement(
            persistence_rmse, fitted_rmse
        ),
        "rms_error_over_mesh_difference": _rmse(error / mesh),
        "max_abs_error_over_mesh_difference": float(np.max(np.abs(error) / mesh)),
        "fraction_abs_error_within_mesh_difference": float(np.mean(np.abs(error) <= mesh)),
        "chip_peak_error_k": float(np.max(predicted[:, 0]) - np.max(truth[:, 0])),
        "max_hotspot_underprediction_k": float(max(0.0, np.max(hotspot_error))),
    }


def _phase_rows(
    case_id: str,
    source: pd.DataFrame,
    truth: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, Any]]:
    time = source["time"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for phase, start, end in RESPONSE_PHASES:
        mask = (time > start) & (time <= end)
        error = predicted[mask] - truth[mask]
        rows.append(
            {
                "case_id": case_id,
                "phase": phase,
                "response_time_start_exclusive_s": start,
                "response_time_end_inclusive_s": end,
                "n_timepoints": int(mask.sum()),
                "rmse_k": _rmse(error),
                "mae_k": _mae(error),
                "max_abs_error_k": _max_abs(error),
            }
        )
    return rows


def _pair_rows(
    frames: dict[str, pd.DataFrame],
    predictions: dict[str, np.ndarray],
    uncertainty: dict[str, float],
) -> pd.DataFrame:
    base_id = "HV01_composite_conjugate"
    radiation_id = "HV02_composite_radiation"
    base_truth = frames[base_id][[f"truth_{name}" for name in SENSORS]].to_numpy(dtype=np.float64)
    radiation_truth = frames[radiation_id][[f"truth_{name}" for name in SENSORS]].to_numpy(
        dtype=np.float64
    )
    truth_delta = radiation_truth - base_truth
    predicted_delta = predictions[radiation_id] - predictions[base_id]
    rows: list[dict[str, Any]] = []
    for index, sensor in enumerate(SENSORS):
        delta_error = predicted_delta[1:, index] - truth_delta[1:, index]
        max_effect = _max_abs(truth_delta[:, index])
        rows.append(
            {
                "sensor": sensor,
                "truth_terminal_radiation_delta_k": float(truth_delta[-1, index]),
                "predicted_terminal_pair_delta_k": float(predicted_delta[-1, index]),
                "max_abs_truth_radiation_delta_k": max_effect,
                "max_abs_predicted_pair_delta_k": _max_abs(predicted_delta[:, index]),
                "pair_delta_rmse_k": _rmse(delta_error),
                "pair_delta_max_abs_error_k": _max_abs(delta_error),
                "mesh_difference_k": uncertainty[sensor],
                "radiation_effect_over_mesh_difference": max_effect / uncertainty[sensor],
                "terminal_predicted_to_truth_delta_ratio_abs": float(
                    abs(predicted_delta[-1, index] / truth_delta[-1, index])
                ),
                "terminal_direction_matches": bool(
                    np.sign(predicted_delta[-1, index]) == np.sign(truth_delta[-1, index])
                ),
            }
        )
    return pd.DataFrame(rows)


def _truth_leakage_check(frames: dict[str, pd.DataFrame], model: ThermalRCModel) -> bool:
    for case_id, source in frames.items():
        baseline = _predict(model, case_id, source)
        altered = source.copy()
        truth_columns = [column for column in altered.columns if column.startswith("truth_")]
        for column in truth_columns:
            altered[column] = source[column].to_numpy(dtype=np.float64) + 10_000.0
        if not np.array_equal(baseline, _predict(model, case_id, altered)):
            return False
    return True


def _input_coverage(
    frames: dict[str, pd.DataFrame], metadata: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    train_ranges = metadata["train_control_ranges"]
    result: dict[str, dict[str, Any]] = {}
    for control in CONTROLS:
        values = np.concatenate(
            [frame[control].to_numpy(dtype=np.float64) for frame in frames.values()]
        )
        evaluation_range = [float(values.min()), float(values.max())]
        training_range = [float(value) for value in train_ranges[control]]
        result[control] = {
            "training_range": training_range,
            "evaluation_range": evaluation_range,
            "within_training_range": (
                evaluation_range[0] >= training_range[0]
                and evaluation_range[1] <= training_range[1]
            ),
        }
    return result


def _is_control_coupled(model: ThermalRCModel, control: str) -> bool:
    edge_coupled = any(
        getattr(item.conductance, "control", None) == control for item in model.spec.edges
    )
    source_coupled = any(
        getattr(item.heat_rate, "control", None) == control for item in model.spec.sources
    )
    boundary_coupled = any(
        item.reservoir_temperature.control == control
        or getattr(item.conductance, "control", None) == control
        for item in model.spec.boundaries
    )
    return edge_coupled or source_coupled or boundary_coupled


def _summary(
    run_dir: Path,
    artifact,
    frames: dict[str, pd.DataFrame],
    cases: pd.DataFrame,
    sensors: pd.DataFrame,
    phases: pd.DataFrame,
    residual_dependence: pd.DataFrame,
    pair: pd.DataFrame,
    quality: dict[str, Any],
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    training_metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    split = pd.read_csv(run_dir / "split.csv")
    leakage_free = _truth_leakage_check(frames, artifact.model)
    controls_complete = artifact.control_names == CONTROLS
    pair_aligned = _validate_pair(frames)
    workflow_pass = bool(
        controls_complete
        and leakage_free
        and pair_aligned
        and quality["dynamic_reference_matches_hv02"]
    )
    rms_model_error = bool(sensors["model_error_resolved_above_mesh_difference"].any())
    pointwise_model_error = bool(sensors["pointwise_error_resolved_above_mesh_difference"].any())
    if rms_model_error:
        adequacy = "sensor_rmse_resolved_above_mesh_difference"
    elif pointwise_model_error:
        adequacy = "pointwise_error_resolved_but_sensor_rmse_within_mesh_difference"
    else:
        adequacy = "not_resolved_beyond_mesh_difference"
    strict_qualified = bool(
        quality["mesh_qualified"]
        and quality["temporal_qualified"]
        and quality["experiment_validated"]
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "External open-loop evaluation of the flow-dependent three-node thermal network "
            "against local-mesh conjugate-flow CAE, with radiation as a paired model gap."
        ),
        "data_quality": {
            **quality,
            "pair_time_and_controls_identical": pair_aligned,
            "strict_validation_status": "qualified" if strict_qualified else "not_qualified",
            "permitted_use": (
                "validated engineering evaluation" if strict_qualified else "model-form screening"
            ),
        },
        "workflow": {
            "status": "pass" if workflow_pass else "fail",
            "artifact_controls": list(artifact.control_names),
            "complete_external_control_boundary": controls_complete,
            "forecast_truth_is_not_an_input": leakage_free,
            "all_predictions_finite": all(
                np.isfinite(value).all() for value in predictions.values()
            ),
        },
        "model": {
            "type": "three-node thermal network using shared scalar response laws",
            "inlet_air_velocity_coupled_to_physics": _is_control_coupled(
                artifact.model, "inlet_air_velocity"
            ),
            "fitted_parameters": fitted_parameters(artifact.model),
            "input_coverage": _input_coverage(frames, artifact.metadata),
        },
        "training": {
            "data": "data/nonlinear/train/*.csv",
            "split_assignment": json_records(split[["case_id", "split"]]),
            "best_epoch": artifact.metadata["training"]["best_epoch"],
            "best_causal_validation_rmse_k": artifact.metadata["training"][
                "best_causal_validation_rmse"
            ],
            "metrics": training_metrics,
        },
        "evaluation": {
            "cases": json_records(cases),
            "phases": json_records(phases),
            "mean_case_rmse_k": float(cases["rmse_k"].mean()),
            "worst_case": str(cases.loc[cases["rmse_k"].idxmax(), "case_id"]),
            "worst_case_rmse_k": float(cases["rmse_k"].max()),
            "mean_prior_rmse_k": float(cases["prior_rmse_k"].mean()),
            "mean_persistence_rmse_k": float(cases["persistence_rmse_k"].mean()),
            "model_adequacy_status": adequacy,
            "residual_diagnostics": {
                "sign_convention": "predicted_minus_truth",
                "forecast_origin_excluded": True,
                "quantities": list(residual_dependence["quantity"].drop_duplicates()),
                "interpretation": (
                    "descriptive screening only; correlation does not identify a missing law"
                ),
            },
            "prediction_figures": {
                "timeseries": validation_figure_reference(
                    "high_fidelity_comsol_prediction_timeseries.png"
                ),
                "parity": validation_figure_reference("high_fidelity_comsol_prediction_parity.png"),
            },
        },
        "radiation_pair": {
            "sensors": json_records(pair),
            "max_abs_truth_effect_k": float(pair["max_abs_truth_radiation_delta_k"].max()),
            "max_effect_over_mesh_difference": float(
                pair["radiation_effect_over_mesh_difference"].max()
            ),
            "interpretation": "directional screening only",
        },
    }


def run_benchmark(config_path: Path, output: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parent
    cfg = load_config(config_path)
    frames = _load_cases(root)
    quality = _reference_quality(root, frames)
    run_dir, forecast_dirs = _run_workflows(cfg, config_path)
    artifact = load_artifact(run_dir / "artifact")
    prior = ThermalRCModel(
        load_system_spec(as_path(str(cfg["system"]), root)),
        integrator=str(cfg.get("engine", {}).get("integrator", "exact")),
    )

    predictions: dict[str, np.ndarray] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    case_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    residual_dependence_rows_output: list[dict[str, Any]] = []
    uncertainty = quality["mesh_difference_k"]
    for case_id, source in frames.items():
        predicted = _saved_prediction(case_id, source, forecast_dirs[case_id], artifact.model)
        prior_prediction = _predict(prior, case_id, source)
        truth = source[[f"truth_{name}" for name in SENSORS]].to_numpy(dtype=np.float64)
        observed = source[list(SENSORS)].to_numpy(dtype=np.float64)
        persistence = persistence_prediction(observed, np.isfinite(observed), origin=0)
        predictions[case_id] = predicted
        prediction_frames[case_id] = _prediction_frame(
            source, predicted, prior_prediction, persistence, uncertainty
        )
        case_rows.append(
            _case_row(
                case_id,
                source,
                truth,
                predicted,
                prior_prediction,
                persistence,
                uncertainty,
            )
        )
        time = source["time"].to_numpy(dtype=np.float64)
        sensor_rows.extend(
            _sensor_rows(
                case_id,
                time,
                truth,
                predicted,
                prior_prediction,
                persistence,
                uncertainty,
            )
        )
        comparison_rows.extend(
            {
                "case_id": case_id,
                "case_group": (
                    "radiation" if bool(source["radiation_enabled"].iat[0]) else "conjugate_flow"
                ),
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
        phase_rows.extend(_phase_rows(case_id, source, truth, predicted))
        residual_dependence_rows_output.extend(
            _residual_dependence(case_id, source, truth, predicted)
        )

    cases = pd.DataFrame(case_rows)
    sensors = pd.DataFrame(sensor_rows)
    model_comparison = pd.DataFrame(comparison_rows)
    phases = pd.DataFrame(phase_rows)
    residual_dependence = pd.DataFrame(residual_dependence_rows_output)
    pair = _pair_rows(frames, predictions, uncertainty)
    summary = _summary(
        run_dir,
        artifact,
        frames,
        cases,
        sensors,
        phases,
        residual_dependence,
        pair,
        quality,
        predictions,
    )
    publish_benchmark(
        output,
        case_metrics=cases,
        sensor_metrics=sensors,
        model_comparison=model_comparison,
        phase_metrics=phases,
        residual_dependence=residual_dependence,
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
        "--output",
        type=Path,
        default=default_root / "data/nonlinear_high_fidelity/benchmark",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_benchmark(args.config, args.output)
    print(json.dumps(summary["workflow"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["evaluation"], indent=2, ensure_ascii=False))
    print(f"saved benchmark: {args.output.resolve()}")
    return 0 if summary["workflow"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
