"""Orchestrate thermal model identification from configured trajectories."""

from __future__ import annotations

from pathlib import Path

from celltemp.config import (
    DATA_OPTIONS,
    as_path,
    project_root_from_config,
    reject_unknown_keys,
    require_path_value,
    validate_config_root,
)
from celltemp.engine import ThermalRCModel
from celltemp.io import load_split_assignments, load_system_spec, load_trajectories
from celltemp.learning import (
    TrainingConfig,
    fit_thermal_model,
    split_trajectories,
)

from .common import (
    output_target,
    project_options,
    project_run_path,
    staged_output_directory,
    trajectory_source_paths,
)
from .train_output import write_training_diagnostics, write_training_outputs

_SPLIT_OPTIONS = {
    "method",
    "recipe_atol",
    "recipe_rtol",
    "table",
    "train_ratio",
    "val_ratio",
}


def _configured_training(values: dict) -> TrainingConfig:
    fields = TrainingConfig.__dataclass_fields__
    reject_unknown_keys(values, fields, "training")
    return TrainingConfig(**values)


def run_train(cfg: dict, config_path: str | Path) -> Path:
    """Fit one shared physical model and save held-out evidence plus an artifact."""
    validate_config_root(cfg)
    root = project_root_from_config(config_path)
    project_cfg = project_options(cfg)
    reject_unknown_keys(cfg["data"], DATA_OPTIONS, "data")
    reject_unknown_keys(cfg.get("engine", {}), {"integrator"}, "engine")
    split_cfg = cfg.get("split", {})
    reject_unknown_keys(split_cfg, _SPLIT_OPTIONS, "split")
    if "table" in split_cfg:
        require_path_value(split_cfg["table"], "split.table")
    split_path = None
    if str(split_cfg.get("method", "random")) == "explicit":
        if "table" not in split_cfg:
            raise ValueError("split.method=explicit requires split.table")
        split_path = as_path(split_cfg["table"], root)
    run_name = str(project_cfg.get("run_name", "thermal_network"))
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
    run_target, overwrite = output_target(
        {
            "output_dir": project_run_path(project_cfg),
            "overwrite": project_cfg.get("overwrite_run", False),
        },
        root,
        protected_paths=(
            Path(config_path),
            as_path(str(cfg["data"]["directory"]), root),
            as_path(str(cfg["system"]), root),
            *trajectory_source_paths(all_trajectories),
            *((split_path,) if split_path is not None else ()),
        ),
    )
    unevaluable = sorted(
        trajectory.case_id for trajectory in all_trajectories if not trajectory.mask[1:].any()
    )
    if unevaluable:
        raise ValueError(f"training data needs an observation after the initial row: {unevaluable}")
    assignments = load_split_assignments(split_path) if split_path is not None else None
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

    with staged_output_directory(run_target, overwrite=overwrite) as run_dir:
        evidence = write_training_outputs(
            run_dir,
            cfg=cfg,
            config_path=config_path,
            run_name=run_name,
            seed=seed,
            control_convention=str(data_cfg.get("control_convention", "left")),
            model=model,
            all_trajectories=all_trajectories,
            trajectories=trajectories,
            training=training,
            result=result,
            diagnostics=project_cfg["diagnostics"],
        )
    if project_cfg["diagnostics"]:
        with staged_output_directory(run_target / "diagnostics", overwrite=overwrite) as target:
            write_training_diagnostics(target, model, evidence)
    return run_target
