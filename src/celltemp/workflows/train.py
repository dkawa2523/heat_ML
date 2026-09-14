"""End-to-end identification and held-out trajectory evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from celltemp.artifact import save_artifact
from celltemp.config import (
    as_path,
    project_root_from_config,
    reject_unknown_keys,
    save_yaml,
    validate_config_root,
)
from celltemp.domain import Trajectory
from celltemp.engine import ThermalRCModel
from celltemp.io import load_split_assignments, load_system_spec, load_trajectories
from celltemp.learning import (
    TrainingConfig,
    fit_thermal_model,
    predict_from_initial_observation,
    predict_trajectory,
    split_trajectories,
)

from .common import output_target, project_options, project_run_path, staged_output_directory

_DATA_OPTIONS = {
    "allow_missing_temperatures",
    "control_convention",
    "directory",
    "dt",
    "pattern",
    "sep",
    "temp_max",
    "temp_min",
    "time_col",
}
_SPLIT_OPTIONS = {
    "method",
    "recipe_atol",
    "recipe_rtol",
    "table",
    "train_ratio",
    "val_ratio",
}


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


def _summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
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


def _write_test_predictions(
    predictions: list[tuple[Trajectory, np.ndarray, np.ndarray]], target: Path
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for trajectory, predicted, initial_prediction in predictions:
        frame = pd.DataFrame({"time": trajectory.time})
        for index, sensor in enumerate(trajectory.sensor_names):
            frame[f"observed_{sensor}"] = trajectory.temperature[:, index]
            frame[f"predicted_{sensor}"] = predicted[:, index]
            frame[f"predicted_from_initial_observation_{sensor}"] = initial_prediction[:, index]
            frame[f"error_{sensor}"] = predicted[:, index] - trajectory.temperature[:, index]
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


def _configured_training(values: dict) -> TrainingConfig:
    fields = TrainingConfig.__dataclass_fields__
    reject_unknown_keys(values, fields, "training")
    return TrainingConfig(**values)


def run_train(cfg: dict, config_path: str | Path) -> Path:
    """Fit one shared physical model and save held-out evidence plus an artifact."""
    validate_config_root(cfg)
    root = project_root_from_config(config_path)
    project_cfg = project_options(cfg)
    reject_unknown_keys(cfg["data"], _DATA_OPTIONS, "data")
    reject_unknown_keys(cfg.get("engine", {}), {"integrator"}, "engine")
    run_name = str(project_cfg.get("run_name", "thermal_network"))
    run_target, overwrite = output_target(
        {
            "output_dir": project_run_path(project_cfg),
            "overwrite": project_cfg.get("overwrite_run", False),
        },
        root,
    )
    seed = int(cfg.get("seed", 42))
    system_path = str(cfg["system"])
    model = ThermalRCModel(
        load_system_spec(as_path(system_path, root)),
        integrator=str(cfg.get("engine", {}).get("integrator", "exact")),
    )
    data_cfg = {
        **cfg["data"],
        "sensor_cols": model.spec.sensor_names,
        "control_cols": model.spec.control_names,
    }
    all_trajectories = load_trajectories(data_cfg, root)
    unevaluable = sorted(
        trajectory.case_id for trajectory in all_trajectories if not trajectory.mask[1:].any()
    )
    if unevaluable:
        raise ValueError(f"training data needs an observation after the initial row: {unevaluable}")
    split_cfg = cfg.get("split", {})
    reject_unknown_keys(split_cfg, _SPLIT_OPTIONS, "split")
    assignments = None
    if str(split_cfg.get("method", "random")) == "explicit":
        table = split_cfg.get("table")
        if not table:
            raise ValueError("split.method=explicit requires split.table")
        assignments = load_split_assignments(as_path(str(table), root))
    trajectories = split_trajectories(
        all_trajectories,
        split_cfg,
        seed=seed,
        assignments=assignments,
    )
    training_values = dict(cfg.get("training", {}))
    if "seed" in training_values:
        raise ValueError("use the top-level seed instead of training.seed")
    training = _configured_training({**training_values, "seed": seed})
    result = fit_thermal_model(
        model,
        trajectories["train"],
        trajectories.get("val"),
        config=training,
    )

    rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    test_predictions: list[tuple[Trajectory, np.ndarray, np.ndarray]] = []
    for split_name, items in trajectories.items():
        for trajectory in items:
            predicted, error, initial_prediction, initial_error, mask = _rollout_errors(
                model,
                trajectory,
                training.initial_temperature_prior_std,
            )
            rows.append(_evaluate_case(trajectory, split_name, error, initial_error, mask))
            sensor_rows.extend(_sensor_metrics(trajectory, split_name, error, initial_error, mask))
            if split_name == "test":
                test_predictions.append((trajectory, predicted, initial_prediction))
    metrics = pd.DataFrame(rows)
    summary = _summary(metrics)
    metadata = {
        "run_name": run_name,
        "seed": seed,
        "control_convention": str(data_cfg.get("control_convention", "left")),
        "config_source": {
            "path": Path(config_path).resolve().name,
            "relative_paths_from": "config_directory",
        },
        "training": {
            "best_epoch": result.best_epoch,
            "best_causal_validation_rmse": result.best_causal_validation_rmse,
            "selected_model_conditional_validation_rmse": summary.get("val", summary["train"])[
                "mean_case_conditional_rmse"
            ],
            "model_selection_metric": "causal_rmse",
            "rollout": "full_trajectory" if training.horizon is None else "window",
            "horizon": training.horizon,
            "initial_temperature_prior_std": training.initial_temperature_prior_std,
        },
        "evaluation_protocols": {
            "conditional": "fit with regularized hidden initial temperatures",
            "causal": "open loop from observations on the first row only",
        },
        "evaluation": summary,
        "train_temperature_ranges": _ranges(
            trajectories["train"], "temperature", model.spec.sensor_names
        ),
        "train_control_ranges": _ranges(
            trajectories["train"], "commands", model.spec.control_names
        ),
        "train_temporal_ranges": _temporal_ranges(trajectories["train"], model.spec.control_names),
    }
    with staged_output_directory(run_target, overwrite=overwrite) as run_dir:
        metrics.to_csv(run_dir / "metrics_by_case.csv", index=False)
        pd.DataFrame(sensor_rows).to_csv(run_dir / "metrics_by_sensor.csv", index=False)
        pd.DataFrame(result.history).to_csv(run_dir / "training_history.csv", index=False)
        (run_dir / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        _write_split(trajectories, run_dir / "split.csv")
        _write_data_summary(all_trajectories, run_dir / "data_summary.csv")
        save_yaml(cfg, run_dir / "config.snapshot.yaml")
        _write_test_predictions(test_predictions, run_dir / "test_predictions")
        save_artifact(run_dir / "artifact", model, metadata=metadata)
    return run_target
