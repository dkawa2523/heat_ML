"""Causal monitoring workflow for self-contained measurement CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from celltemp.artifact import load_artifact
from celltemp.config import project_root_from_config
from celltemp.inference import monitor

from .common import (
    load_runtime_trajectories,
    output_target,
    resolve_artifact_path,
    staged_output_directory,
)


def run_monitor(cfg: dict, config_path: str | Path) -> Path:
    root = project_root_from_config(config_path)
    values = cfg["monitor"]
    artifact = load_artifact(resolve_artifact_path(cfg, root), device=values.get("device", "cpu"))
    trajectories = load_runtime_trajectories(
        values,
        root,
        sensor_names=artifact.sensor_names,
        control_names=artifact.control_names,
    )
    target, overwrite = output_target(cfg, root, section="monitor")
    observer_cfg = values.get("observer", {})
    summaries: list[dict[str, object]] = []
    with staged_output_directory(target, overwrite=overwrite) as out_dir:
        for trajectory in trajectories:
            result = monitor(artifact.model, trajectory, **observer_cfg)
            output = pd.DataFrame({"time": result.time})
            for sensor_index, sensor in enumerate(artifact.sensor_names):
                output[f"measured_{sensor}"] = trajectory.temperature[:, sensor_index]
                output[f"prior_{sensor}"] = result.prior_sensor_temperature[:, sensor_index]
                output[f"filtered_{sensor}"] = result.filtered_sensor_temperature[:, sensor_index]
                output[f"residual_{sensor}"] = result.residual[:, sensor_index]
                output[f"residual_std_{sensor}"] = result.residual_std[:, sensor_index]
                output[f"bias_{sensor}"] = result.sensor_bias[:, sensor_index]
            for node_index, node in enumerate(artifact.model.spec.node_names):
                output[f"state_{node}"] = result.filtered_node_temperature[:, node_index]
            for control_index, control in enumerate(artifact.control_names):
                output[f"effective_{control}"] = result.actuator[:, control_index]
            output_file = out_dir / f"{trajectory.case_id}.csv"
            output.to_csv(output_file, index=False)
            valid = np.isfinite(result.residual)
            residual_values = result.residual[valid]
            summaries.append(
                {
                    "case_id": trajectory.case_id,
                    "residual_rmse": (
                        float(np.sqrt(np.mean(residual_values**2)))
                        if residual_values.size
                        else float("nan")
                    ),
                    "max_abs_residual": (
                        float(np.max(np.abs(residual_values)))
                        if residual_values.size
                        else float("nan")
                    ),
                    "output": str(target / output_file.name),
                }
            )
        pd.DataFrame(summaries).to_csv(out_dir / "monitor_summary.csv", index=False)
    return target
