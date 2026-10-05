"""Evaluation evidence and persisted outputs for one fitted training run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from celltemp.analysis import (
    persistence_prediction,
    prediction_comparison_rows,
    representative_command,
    thermal_mode_rows,
    thermal_path_rows,
)
from celltemp.artifact import save_artifact
from celltemp.config import save_yaml, temperature_unit_label
from celltemp.domain import Trajectory
from celltemp.engine import ThermalRCModel
from celltemp.learning import (
    TrainingConfig,
    TrainingResult,
    predict_from_initial_observation,
    predict_trajectory,
)

from .common import input_file_records, project_options

_OPERATING_POINT = "representative_training_command"


@dataclass(frozen=True)
class _CasePrediction:
    trajectory: Trajectory
    split: str
    conditional: np.ndarray
    causal: np.ndarray


@dataclass(frozen=True)
class _TrainingEvidence:
    """Tables and predictions derived from a fitted model without filesystem IO."""

    metrics: pd.DataFrame
    sensor_metrics: pd.DataFrame
    summary: dict[str, dict[str, float | int]]
    predictions: tuple[_CasePrediction, ...]
    operating_point: np.ndarray
    temperature_unit: str = "degC"


def _rollout_errors(
    model: ThermalRCModel,
    trajectory: Trajectory,
    initial_temperature_prior_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        predicted = (
            predict_trajectory(
                model,
                trajectory,
                initial_temperature_prior_std=initial_temperature_prior_std,
            )
            .detach()
            .cpu()
            .numpy()
        )
        initial_prediction = (
            predict_from_initial_observation(model, trajectory).detach().cpu().numpy()
        )
    observed = trajectory.temperature
    mask = trajectory.mask.copy()
    mask[0] = False
    return predicted, predicted - observed, initial_prediction, initial_prediction - observed, mask


def _evaluate_case(
    trajectory: Trajectory,
    split_name: str,
    error: np.ndarray,
    initial_error: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    values = error[mask]
    initial_values = initial_error[mask]
    return {
        "split": split_name,
        "case_id": trajectory.case_id,
        "n_observations": int(mask.sum()),
        "conditional_rmse": float(np.sqrt(np.mean(values**2))),
        "conditional_mae": float(np.mean(np.abs(values))),
        "conditional_max_abs_error": float(np.max(np.abs(values))),
        "causal_rmse": float(np.sqrt(np.mean(initial_values**2))),
        "causal_mae": float(np.mean(np.abs(initial_values))),
        "causal_max_abs_error": float(np.max(np.abs(initial_values))),
    }


def _sensor_metrics(
    trajectory: Trajectory,
    split_name: str,
    error: np.ndarray,
    initial_error: np.ndarray,
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sensor in enumerate(trajectory.sensor_names):
        values = error[:, index][mask[:, index]]
        if not len(values):
            continue
        initial_values = initial_error[:, index][mask[:, index]]
        rows.append(
            {
                "split": split_name,
                "case_id": trajectory.case_id,
                "sensor": sensor,
                "conditional_rmse": float(np.sqrt(np.mean(values**2))),
                "conditional_mae": float(np.mean(np.abs(values))),
                "conditional_max_abs_error": float(np.max(np.abs(values))),
                "causal_rmse": float(np.sqrt(np.mean(initial_values**2))),
            }
        )
    return rows


def _summary(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for split_name, part in frame.groupby("split", sort=False):
        result[str(split_name)] = {
            "n_cases": len(part),
            "mean_case_conditional_rmse": float(part["conditional_rmse"].mean()),
            "median_case_conditional_rmse": float(part["conditional_rmse"].median()),
            "worst_case_conditional_rmse": float(part["conditional_rmse"].max()),
            "mean_case_conditional_mae": float(part["conditional_mae"].mean()),
            "mean_case_causal_rmse": float(part["causal_rmse"].mean()),
            "worst_case_causal_rmse": float(part["causal_rmse"].max()),
        }
    return result


def _build_training_evidence(
    model: ThermalRCModel,
    trajectories: dict[str, list[Trajectory]],
    *,
    initial_temperature_prior_std: float,
    keep_predictions: bool = False,
    temperature_unit: str = "degC",
) -> _TrainingEvidence:
    """Evaluate the selected model, retaining held-out arrays only for diagnostics."""
    case_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    predictions: list[_CasePrediction] = []
    for split_name, items in trajectories.items():
        for trajectory in items:
            predicted, error, initial_prediction, initial_error, mask = _rollout_errors(
                model,
                trajectory,
                initial_temperature_prior_std,
            )
            case_rows.append(_evaluate_case(trajectory, split_name, error, initial_error, mask))
            sensor_rows.extend(_sensor_metrics(trajectory, split_name, error, initial_error, mask))
            if keep_predictions and split_name in {"val", "test"}:
                predictions.append(
                    _CasePrediction(trajectory, split_name, predicted, initial_prediction)
                )

    metrics = pd.DataFrame(case_rows)
    operating_point = representative_command(trajectories["train"])
    return _TrainingEvidence(
        metrics=metrics,
        sensor_metrics=pd.DataFrame(sensor_rows),
        summary=_summary(metrics),
        predictions=tuple(predictions),
        operating_point=operating_point,
        temperature_unit=temperature_unit,
    )


def _model_comparison(
    model: ThermalRCModel, predictions: tuple[_CasePrediction, ...]
) -> pd.DataFrame:
    prior_model = ThermalRCModel(
        model.spec, integrator=model.integrator, dtype=model.capacity.dtype
    ).to(model.capacity.device)
    prior_model.eval()
    rows: list[dict[str, Any]] = []
    for item in predictions:
        trajectory = item.trajectory
        with torch.no_grad():
            prior = predict_from_initial_observation(prior_model, trajectory).cpu().numpy()
        truth = np.where(trajectory.mask, trajectory.temperature, np.nan)
        truth[0] = np.nan
        rows.extend(
            {"split": item.split, "case_id": trajectory.case_id, **row}
            for row in prediction_comparison_rows(
                trajectory.time,
                truth,
                {
                    "fitted_rc": item.causal,
                    "engineering_prior_rc": prior,
                    "persistence": persistence_prediction(
                        trajectory.temperature, trajectory.mask, origin=0
                    ),
                },
                trajectory.sensor_names,
            )
        )
    return pd.DataFrame(rows)


def _write_test_prediction_figures(evidence: _TrainingEvidence, target: Path) -> None:
    from .prediction_figures import PredictionCase, write_prediction_figures

    predictions = [item for item in evidence.predictions if item.split == "test"]
    if not predictions:
        return
    first_trajectory = predictions[0].trajectory
    cases = [
        PredictionCase(
            case_id=item.trajectory.case_id,
            time=item.trajectory.time,
            truth=np.where(item.trajectory.mask, item.trajectory.temperature, np.nan),
            predicted=item.causal,
        )
        for item in predictions
    ]
    write_prediction_figures(
        cases,
        target,
        sensor_names=first_trajectory.sensor_names,
        title="Held-out test prediction",
        file_prefix="test_prediction",
        temperature_unit=temperature_unit_label(evidence.temperature_unit),
    )


def _ranges(
    trajectories: list[Trajectory], field: str, names: tuple[str, ...]
) -> dict[str, list[float | None]]:
    arrays = [getattr(item, field) for item in trajectories]
    values = np.concatenate(arrays, axis=0)
    result: dict[str, list[float | None]] = {}
    for index, name in enumerate(names):
        finite = values[:, index][np.isfinite(values[:, index])]
        result[name] = [float(finite.min()), float(finite.max())] if finite.size else [None, None]
    return result


def _temporal_ranges(
    trajectories: list[Trajectory], control_names: tuple[str, ...]
) -> dict[str, object]:
    """Summarize the time-domain envelope relevant to open-loop use."""
    time_steps = np.concatenate([trajectory.dt for trajectory in trajectories])
    durations = np.asarray(
        [trajectory.time[-1] - trajectory.time[0] for trajectory in trajectories]
    )
    slew_ranges: dict[str, list[float]] = {}
    for index, name in enumerate(control_names):
        rates = [
            np.abs(np.diff(trajectory.commands[:, index]) / trajectory.dt[:-1])
            for trajectory in trajectories
            if len(trajectory.commands) > 1
        ]
        maximum = max((float(values.max()) for values in rates if values.size), default=0.0)
        slew_ranges[name] = [0.0, maximum]
    return {
        "time_step_seconds": [float(time_steps.min()), float(time_steps.max())],
        "forecast_horizon_seconds": [0.0, float(durations.max())],
        "control_slew_per_second": slew_ranges,
    }


def _artifact_metadata(
    *,
    config_path: str | Path,
    run_name: str,
    seed: int,
    control_convention: str,
    model: ThermalRCModel,
    trajectories: dict[str, list[Trajectory]],
    training: TrainingConfig,
    result: TrainingResult,
    evidence: _TrainingEvidence,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "seed": seed,
        "control_convention": control_convention,
        "temperature_unit": evidence.temperature_unit,
        "training_inputs": [
            {**record, "split": name}
            for name, cases in trajectories.items()
            for record in input_file_records(cases, Path(config_path).resolve().parent)
        ],
        "config_source": {
            "path": Path(config_path).resolve().name,
            "relative_paths_from": "config_directory",
        },
        "training": {
            "best_epoch": result.best_epoch,
            "best_causal_validation_rmse": result.best_causal_validation_rmse,
            "selected_model_conditional_validation_rmse": evidence.summary.get(
                "val", evidence.summary["train"]
            )["mean_case_conditional_rmse"],
            "model_selection_metric": "causal_rmse",
            "rollout": "full_trajectory" if training.horizon is None else "window",
            "horizon": training.horizon,
            "initial_temperature_prior_std": training.initial_temperature_prior_std,
        },
        "evaluation_protocols": {
            "conditional": "fit with regularized hidden initial temperatures",
            "causal": "open loop from observations on the first row only",
        },
        "evaluation": evidence.summary,
        "train_temperature_ranges": _ranges(
            trajectories["train"], "temperature", model.spec.sensor_names
        ),
        "train_control_ranges": _ranges(
            trajectories["train"], "commands", model.spec.control_names
        ),
        "train_temporal_ranges": _temporal_ranges(trajectories["train"], model.spec.control_names),
        "model_characterization": {
            "operating_point": _OPERATING_POINT,
            "selection": "observed training command nearest the component-wise median",
            "steady_actuator": {
                name: {
                    "value": float(evidence.operating_point[index]),
                    "unit": model.spec.control_units[index],
                    "role": model.spec.control_roles[index],
                }
                for index, name in enumerate(model.spec.control_names)
            },
        },
    }


def _write_test_predictions(predictions: tuple[_CasePrediction, ...], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in predictions:
        if item.split != "test":
            continue
        trajectory = item.trajectory
        frame = pd.DataFrame({"time": trajectory.time})
        for index, sensor in enumerate(trajectory.sensor_names):
            observed = np.where(trajectory.mask[:, index], trajectory.temperature[:, index], np.nan)
            frame[f"sensor.{sensor}.observed"] = observed
            frame[f"sensor.{sensor}.conditional"] = item.conditional[:, index]
            frame[f"sensor.{sensor}.causal"] = item.causal[:, index]
            frame[f"sensor.{sensor}.error"] = item.conditional[:, index] - observed
        frame.to_csv(target / f"{trajectory.case_id}.csv", index=False)


def _write_split(split: dict[str, list[Trajectory]], target: Path) -> None:
    rows = [
        {**dict(trajectory.metadata), "case_id": trajectory.case_id, "split": split_name}
        for split_name, trajectories in split.items()
        for trajectory in trajectories
    ]
    pd.DataFrame(rows).to_csv(target, index=False)


def _write_data_summary(trajectories: list[Trajectory], target: Path) -> None:
    rows: list[dict[str, object]] = []
    for trajectory in trajectories:
        observed = trajectory.temperature[trajectory.mask]
        rows.append(
            {
                **dict(trajectory.metadata),
                "case_id": trajectory.case_id,
                "n_rows": len(trajectory.time),
                "time_start": float(trajectory.time[0]),
                "time_end": float(trajectory.time[-1]),
                "temp_min": float(observed.min()),
                "temp_max": float(observed.max()),
            }
        )
    pd.DataFrame(rows).to_csv(target, index=False)


def write_training_outputs(
    target: Path,
    *,
    cfg: dict,
    config_path: str | Path,
    run_name: str,
    seed: int,
    control_convention: str,
    model: ThermalRCModel,
    all_trajectories: list[Trajectory],
    trajectories: dict[str, list[Trajectory]],
    training: TrainingConfig,
    result: TrainingResult,
    diagnostics: bool = False,
) -> _TrainingEvidence:
    """Write core scores, provenance, and a portable model independently of diagnostics."""
    evidence = _build_training_evidence(
        model,
        trajectories,
        initial_temperature_prior_std=training.initial_temperature_prior_std,
        keep_predictions=diagnostics,
        temperature_unit=project_options(cfg)["temperature_unit"],
    )
    metadata = _artifact_metadata(
        config_path=config_path,
        run_name=run_name,
        seed=seed,
        control_convention=control_convention,
        model=model,
        trajectories=trajectories,
        training=training,
        result=result,
        evidence=evidence,
    )
    evidence.metrics.to_csv(target / "metrics_by_case.csv", index=False)
    evidence.sensor_metrics.to_csv(target / "metrics_by_sensor.csv", index=False)
    pd.DataFrame(result.history).to_csv(target / "training_history.csv", index=False)
    (target / "metrics_summary.json").write_text(
        json.dumps(evidence.summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_split(trajectories, target / "split.csv")
    _write_data_summary(all_trajectories, target / "data_summary.csv")
    save_yaml(cfg, target / "config.snapshot.yaml")
    save_artifact(target / "artifact", model, metadata=metadata)
    return evidence


def write_training_diagnostics(
    target: Path, model: ThermalRCModel, evidence: _TrainingEvidence
) -> None:
    """Write optional comparisons and model inspection after the core run is saved."""
    _model_comparison(model, evidence.predictions).to_csv(
        target / "model_comparison.csv", index=False
    )
    pd.DataFrame(
        thermal_path_rows(model, evidence.operating_point, operating_point=_OPERATING_POINT)
    ).to_csv(target / "thermal_paths.csv", index=False)
    pd.DataFrame(
        thermal_mode_rows(model, evidence.operating_point, operating_point=_OPERATING_POINT)
    ).to_csv(target / "thermal_modes.csv", index=False)
    _write_test_predictions(evidence.predictions, target / "test_predictions")
    _write_test_prediction_figures(evidence, target / "figures")
