"""Open-loop forecast workflow for self-contained request CSVs."""

from __future__ import annotations

from pathlib import Path

from celltemp.artifact import load_artifact
from celltemp.config import (
    as_path,
    project_root_from_config,
    require_path_value,
    validate_config_root,
)
from celltemp.inference import build_observer, forecast, resolve_observer_settings

from .common import (
    load_runtime_trajectories,
    output_target,
    project_options,
    resolve_artifact_path,
    resolve_runtime_values,
    staged_output_directory,
    trajectory_source_paths,
    validate_runtime_options,
    write_runtime_manifest,
)
from .forecast_output import (
    ForecastCaseSummary,
    build_forecast_case,
    forecast_frame,
    print_coverage_warnings,
    write_forecast_tables,
)


def run_forecast(cfg: dict, config_path: str | Path) -> Path:
    """Estimate history endpoints and save open-loop thermal predictions."""
    validate_config_root(cfg)
    root = project_root_from_config(config_path)
    values = cfg.get("forecast", {})
    validate_runtime_options(values, "forecast", observer=True)
    require_path_value(values.get("input_dir"), "forecast.input_dir")
    require_path_value(values.get("output_dir"), "forecast.output_dir")
    artifact = load_artifact(resolve_artifact_path(cfg, root), device=values.get("device", "cpu"))
    values = resolve_runtime_values(cfg, "forecast", artifact_metadata=artifact.metadata)
    requests = load_runtime_trajectories(
        values,
        root,
        sensor_names=artifact.sensor_names,
        control_names=artifact.control_names,
    )

    target, overwrite = output_target(
        cfg,
        root,
        section="forecast",
        protected_paths=(
            artifact.path,
            Path(config_path),
            as_path(values["input_dir"], root),
            *trajectory_source_paths(requests),
        ),
    )
    resolved_observer = resolve_observer_settings("forecast", values.get("observer"))
    state_estimator = build_observer(artifact.model, resolved_observer)
    case_tables: list[ForecastCaseSummary] = []
    with staged_output_directory(target, overwrite=overwrite) as out_dir:
        (out_dir / "cases").mkdir()
        for request in requests:
            result = forecast(artifact.model, request, observer=state_estimator)
            output_name = f"cases/{request.case_id}.csv"
            tables = build_forecast_case(
                artifact,
                request,
                result,
                output_name=output_name,
            )
            forecast_frame(artifact, request, result).to_csv(out_dir / output_name, index=False)
            case_tables.append(tables)
        write_forecast_tables(out_dir, case_tables)
        write_runtime_manifest(
            out_dir / "run_manifest.json",
            workflow="forecast",
            config_path=config_path,
            root=root,
            artifact=artifact,
            values=values,
            trajectories=requests,
            observer_settings=resolved_observer,
            disturbance_basis=state_estimator.disturbance_basis,
            uncertainty={
                "confidence": 0.95,
                "scope": "latent_state_and_process_only",
                "excludes": ["measurement", "parameter", "input", "model_form"],
            },
        )
    print_coverage_warnings(case_tables)
    if project_options(cfg)["diagnostics"]:
        from .forecast_diagnostics import write_forecast_diagnostics

        with staged_output_directory(target / "diagnostics", overwrite=overwrite) as diagnostics:
            write_forecast_diagnostics(diagnostics, target, artifact)
    return target
