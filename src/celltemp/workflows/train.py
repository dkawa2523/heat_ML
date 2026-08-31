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
    predict_trajectory,
    split_trajectories,
)

from .common import output_target, project_options, staged_output_directory

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
_SPLIT_OPTIONS = {"method", "table", "train_ratio", "val_ratio"}


def _rollout_errors(
    model: ThermalRCModel, trajectory: Trajectory
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        predicted = predict_trajectory(model, trajectory).detach().cpu().numpy()
    observed = trajectory.temperature
    mask = trajectory.mask.copy()
    mask[0] = False
    return predicted, predicted - observed, mask


def _evaluate_case(
    trajectory: Trajectory, split_name: str, error: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    values = error[mask]
    return {
        "split": split_name,
        "case_id": trajectory.case_id,
        "n_observations": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mae": float(np.mean(np.abs(values))),
        "max_abs_error": float(np.max(np.abs(values))),
    }


def _sensor_metrics(
    trajectory: Trajectory, split_name: str, error: np.ndarray, mask: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sensor in enumerate(trajectory.sensor_names):
        values = error[:, index][mask[:, index]]
        if not len(values):
            continue
        rows.append(
            {
                "split": split_name,
                "case_id": trajectory.case_id,
                "sensor": sensor,
                "rmse": float(np.sqrt(np.mean(values**2))),
                "mae": float(np.mean(np.abs(values))),
                "max_abs_error": float(np.max(np.abs(values))),
            }
        )
    return rows


def _summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for split_name, part in frame.groupby("split", sort=False):
        result[str(split_name)] = {
            "n_cases": len(part),
            "mean_case_rmse": float(part["rmse"].mean()),
            "median_case_rmse": float(part["rmse"].median()),
            "worst_case_rmse": float(part["rmse"].max()),
            "mean_case_mae": float(part["mae"].mean()),
        }
    return result


def _ranges(
    trajectories: list[Trajectory], field: str, names: tuple[str, ...]
) -> dict[str, list[float]]:
    arrays = [getattr(item, field) for item in trajectories]
    values = np.concatenate(arrays, axis=0)
    return {
        name: [float(np.nanmin(values[:, index])), float(np.nanmax(values[:, index]))]
        for index, name in enumerate(names)
    }


def _write_test_predictions(predictions: list[tuple[Trajectory, np.ndarray]], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for trajectory, predicted in predictions:
        frame = pd.DataFrame({"time": trajectory.time})
        for index, sensor in enumerate(trajectory.sensor_names):
            frame[f"observed_{sensor}"] = trajectory.temperature[:, index]
            frame[f"predicted_{sensor}"] = predicted[:, index]
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
    base = as_path(project_cfg.get("output_dir", "outputs/runs"), root)
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
    run_target, overwrite = output_target(
        {
            "output_dir": str(base / run_name),
            "overwrite": bool(project_cfg.get("overwrite_run", False)),
        },
        root,
    )
    result = fit_thermal_model(
        model,
        trajectories["train"],
        trajectories.get("val"),
        config=training,
    )

    rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    test_predictions: list[tuple[Trajectory, np.ndarray]] = []
    for split_name, items in trajectories.items():
        for trajectory in items:
            predicted, error, mask = _rollout_errors(model, trajectory)
            rows.append(_evaluate_case(trajectory, split_name, error, mask))
            sensor_rows.extend(_sensor_metrics(trajectory, split_name, error, mask))
            if split_name == "test":
                test_predictions.append((trajectory, predicted))
    metrics = pd.DataFrame(rows)
    summary = _summary(metrics)
    metadata = {
        "run_name": run_name,
        "seed": seed,
        "control_convention": str(data_cfg.get("control_convention", "left")),
        "training": {
            "best_epoch": result.best_epoch,
            "best_validation_rmse": result.best_validation_rmse,
            "rollout": "full_trajectory" if training.horizon is None else "window",
            "horizon": training.horizon,
        },
        "evaluation": summary,
        "train_temperature_ranges": _ranges(
            trajectories["train"], "temperature", model.spec.sensor_names
        ),
        "train_control_ranges": _ranges(
            trajectories["train"], "commands", model.spec.control_names
        ),
    }
    with staged_output_directory(run_target, overwrite=overwrite) as run_dir:
        metrics.to_csv(run_dir / "metrics_by_case.csv", index=False)
        pd.DataFrame(sensor_rows).to_csv(run_dir / "metrics_by_sensor.csv", index=False)
        pd.DataFrame(result.history).to_csv(run_dir / "training_history.csv", index=False)
        (run_dir / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_split(trajectories, run_dir / "split.csv")
        _write_data_summary(all_trajectories, run_dir / "data_summary.csv")
        save_yaml(cfg, run_dir / "config.yaml")
        _write_test_predictions(test_predictions, run_dir / "test_predictions")
        save_artifact(run_dir / "artifact", model, metadata=metadata)
    return run_target
