"""Causal monitoring workflow for self-contained measurement CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from celltemp.artifact import load_artifact
from celltemp.config import project_root_from_config, validate_config_root
from celltemp.inference import build_observer, monitor, resolve_observer_settings

from .common import (
    load_runtime_trajectories,
    output_target,
    resolve_artifact_path,
    staged_output_directory,
    validate_runtime_options,
    write_runtime_manifest,
)


def run_monitor(cfg: dict, config_path: str | Path) -> Path:
    validate_config_root(cfg)
    root = project_root_from_config(config_path)
    values = cfg["monitor"]
    validate_runtime_options(values, "monitor", observer=True)
    artifact = load_artifact(resolve_artifact_path(cfg, root), device=values.get("device", "cpu"))
    trajectories = load_runtime_trajectories(
        values,
        root,
        sensor_names=artifact.sensor_names,
        control_names=artifact.control_names,
    )
    target, overwrite = output_target(cfg, root, section="monitor")
    resolved_observer = resolve_observer_settings("monitor", values.get("observer"))
    state_estimator = build_observer(artifact.model, resolved_observer)
    summaries: list[dict[str, object]] = []
    with staged_output_directory(target, overwrite=overwrite) as out_dir:
        for trajectory in trajectories:
            result = monitor(artifact.model, trajectory, observer=state_estimator)
            output = pd.DataFrame({"time": result.time})
            for sensor_index, sensor in enumerate(artifact.sensor_names):
                output[f"measured_{sensor}"] = trajectory.temperature[:, sensor_index]
                output[f"prior_physical_{sensor}"] = result.prior_physical_temperature[
                    :, sensor_index
                ]
                output[f"predicted_measurement_{sensor}"] = result.predicted_measurement[
                    :, sensor_index
                ]
                output[f"posterior_physical_{sensor}"] = result.posterior_physical_temperature[
                    :, sensor_index
                ]
                output[f"reconstructed_measurement_{sensor}"] = result.reconstructed_measurement[
                    :, sensor_index
                ]
                output[f"innovation_{sensor}"] = result.innovation[:, sensor_index]
                output[f"innovation_std_{sensor}"] = result.innovation_std[:, sensor_index]
                output[f"sensor_bias_{sensor}"] = result.sensor_bias[:, sensor_index]
            for node_index, node in enumerate(artifact.model.spec.node_names):
                output[f"state_{node}"] = result.posterior_node_temperature[:, node_index]
                output[f"disturbance_{node}_w"] = result.node_heat_disturbance[:, node_index]
            for control_index, control in enumerate(artifact.control_names):
                output[f"effective_{control}"] = result.actuator[:, control_index]
            output["nis"] = result.nis
            output["nis_dof"] = result.nis_dof
            output["bias_gauge"] = result.bias_gauge
            output_file = out_dir / f"{trajectory.case_id}.csv"
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
                    "output": output_file.name,
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
