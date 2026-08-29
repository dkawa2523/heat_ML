"""Open-loop forecast workflow for self-contained request CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from celltemp.artifact import load_artifact
from celltemp.config import project_root_from_config, validate_config_root
from celltemp.inference import forecast

from .common import (
    load_runtime_trajectories,
    output_target,
    resolve_artifact_path,
    staged_output_directory,
    validate_runtime_options,
)


def run_forecast(cfg: dict, config_path: str | Path) -> Path:
    validate_config_root(cfg)
    root = project_root_from_config(config_path)
    values = cfg["forecast"]
    validate_runtime_options(values, "forecast")
    artifact = load_artifact(resolve_artifact_path(cfg, root), device=values.get("device", "cpu"))
    requests = load_runtime_trajectories(
        values,
        root,
        sensor_names=artifact.sensor_names,
        control_names=artifact.control_names,
    )
    for request in requests:
        if request.mask[1:].any():
            raise ValueError(
                f"{request.case_id}: forecast temperatures are only allowed on the initial row"
            )

    target, overwrite = output_target(cfg, root, section="forecast")
    summaries: list[dict[str, object]] = []
    with staged_output_directory(target, overwrite=overwrite) as out_dir:
        for request in requests:
            result = forecast(artifact.model, request)
            frame = pd.DataFrame({"time": result.time})
            for sensor_index, sensor in enumerate(artifact.sensor_names):
                frame[f"temperature_{sensor}"] = result.sensor_temperature[:, sensor_index]
            for node_index, node in enumerate(artifact.model.spec.node_names):
                frame[f"state_{node}"] = result.node_temperature[:, node_index]
            for control_index, control in enumerate(artifact.control_names):
                frame[f"effective_{control}"] = result.actuator[:, control_index]
            output_file = out_dir / f"{request.case_id}.csv"
            frame.to_csv(output_file, index=False)
            summaries.append(
                {
                    "case_id": request.case_id,
                    "rows": len(frame),
                    "time_end": float(result.time[-1]),
                    "output": output_file.name,
                }
            )
        pd.DataFrame(summaries).to_csv(out_dir / "forecast_summary.csv", index=False)
    return target
