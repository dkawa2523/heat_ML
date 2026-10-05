"""Causal monitoring workflow for self-contained measurement CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from celltemp.artifact import load_artifact
from celltemp.config import (
    as_path,
    project_root_from_config,
    require_path_value,
    validate_config_root,
)
from celltemp.inference import build_observer, monitor, resolve_observer_settings

from .common import (
    load_runtime_trajectories,
    output_target,
    resolve_artifact_path,
    resolve_runtime_values,
    staged_output_directory,
    trajectory_source_paths,
    validate_runtime_options,
    write_runtime_manifest,
)


def run_monitor(cfg: dict, config_path: str | Path) -> Path:
    validate_config_root(cfg)
    root = project_root_from_config(config_path)
    values = cfg.get("monitor", {})
    validate_runtime_options(values, "monitor", observer=True)
    require_path_value(values.get("input_dir"), "monitor.input_dir")
    require_path_value(values.get("output_dir"), "monitor.output_dir")
    artifact = load_artifact(resolve_artifact_path(cfg, root), device=values.get("device", "cpu"))
    values = resolve_runtime_values(cfg, "monitor", artifact_metadata=artifact.metadata)
    trajectories = load_runtime_trajectories(
        values,
        root,
        sensor_names=artifact.sensor_names,
        control_names=artifact.control_names,
    )
    target, overwrite = output_target(
        cfg,
        root,
        section="monitor",
        protected_paths=(
            artifact.path,
            Path(config_path),
            as_path(values["input_dir"], root),
            *trajectory_source_paths(trajectories),
        ),
    )
    resolved_observer = resolve_observer_settings("monitor", values.get("observer"))
    state_estimator = build_observer(artifact.model, resolved_observer)
    summaries: list[dict[str, object]] = []
    with staged_output_directory(target, overwrite=overwrite) as out_dir:
        (out_dir / "cases").mkdir()
        for trajectory in trajectories:
            result = monitor(artifact.model, trajectory, observer=state_estimator)
            output = pd.DataFrame({"time": result.time})
            for sensor_index, sensor in enumerate(artifact.sensor_names):
                output[f"sensor.{sensor}.measured"] = trajectory.temperature[:, sensor_index]
                output[f"sensor.{sensor}.prior_physical"] = result.prior_physical_temperature[
                    :, sensor_index
                ]
                output[f"sensor.{sensor}.predicted_measurement"] = result.predicted_measurement[
                    :, sensor_index
                ]
                output[f"sensor.{sensor}.posterior_physical"] = (
                    result.posterior_physical_temperature[:, sensor_index]
                )
                output[f"sensor.{sensor}.reconstructed_measurement"] = (
                    result.reconstructed_measurement[:, sensor_index]
                )
                output[f"sensor.{sensor}.innovation"] = result.innovation[:, sensor_index]
                output[f"sensor.{sensor}.innovation_std"] = result.innovation_std[:, sensor_index]
                output[f"sensor.{sensor}.bias"] = result.sensor_bias[:, sensor_index]
            for node_index, node in enumerate(artifact.model.spec.node_names):
                output[f"node.{node}.temperature"] = result.posterior_node_temperature[
                    :, node_index
                ]
                output[f"node.{node}.disturbance_w"] = result.node_heat_disturbance[:, node_index]
            sampled_commands = np.concatenate(
                [trajectory.commands, trajectory.commands[-1:]], axis=0
            )
            for control_index, control in enumerate(artifact.control_names):
                output[f"control.{control}.command"] = sampled_commands[:, control_index]
                output[f"control.{control}.effective"] = result.actuator[:, control_index]
            output["nis"] = result.nis
            output["nis_dof"] = result.nis_dof
            output["bias_gauge"] = result.bias_gauge
            output_file = out_dir / "cases" / f"{trajectory.case_id}.csv"
            output.to_csv(output_file, index=False)
            valid = np.isfinite(result.innovation)
            innovation_values = result.innovation[valid]
            valid_nis = np.isfinite(result.nis) & (result.nis_dof > 0)
            summaries.append(
                {
                    "case_id": trajectory.case_id,
                    "bias_gauge": result.bias_gauge,
                    "initial_actuator_source": (
                        "csv" if trajectory.initial_actuator is not None else "first_command"
                    ),
                    "innovation_rmse": (
                        float(np.sqrt(np.mean(innovation_values**2)))
                        if innovation_values.size
                        else float("nan")
                    ),
                    "max_abs_innovation": (
                        float(np.max(np.abs(innovation_values)))
                        if innovation_values.size
                        else float("nan")
                    ),
                    "mean_nis_per_dof": (
                        float(np.mean(result.nis[valid_nis] / result.nis_dof[valid_nis]))
                        if np.any(valid_nis)
                        else float("nan")
                    ),
                    "output": output_file.relative_to(out_dir).as_posix(),
                }
            )
        pd.DataFrame(summaries).to_csv(out_dir / "monitor_summary.csv", index=False)
        write_runtime_manifest(
            out_dir / "run_manifest.json",
            workflow="monitor",
            config_path=config_path,
            root=root,
            artifact=artifact,
            values=values,
            trajectories=trajectories,
            observer_settings=resolved_observer,
            disturbance_basis=state_estimator.disturbance_basis,
        )
    return target
