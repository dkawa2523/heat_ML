"""Workflow paths, atomic output publication, and input/output provenance."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

from celltemp.config import (
    DATA_OPTIONS,
    as_path,
    reject_unknown_keys,
    require_bool,
    require_path_value,
    temperature_unit_label,
)
from celltemp.domain import Trajectory
from celltemp.io import load_trajectories

if TYPE_CHECKING:
    from celltemp.artifact import ThermalArtifact

_RUNTIME_OPTIONS = {
    "control_convention",
    "device",
    "dt",
    "input_dir",
    "output_dir",
    "overwrite",
    "pattern",
    "sep",
    "time_col",
}


def project_options(cfg: dict) -> dict:
    values = cfg.get("project", {})
    reject_unknown_keys(
        values,
        {"diagnostics", "output_dir", "overwrite_run", "run_name", "temperature_unit"},
        "project",
    )
    result = dict(values)
    for option in ("output_dir", "run_name"):
        if option in result:
            require_path_value(result[option], f"project.{option}")
    result["temperature_unit"] = result.get("temperature_unit", "degC")
    temperature_unit_label(result["temperature_unit"])
    if "overwrite_run" in result:
        result["overwrite_run"] = require_bool(result["overwrite_run"], "project.overwrite_run")
    result["diagnostics"] = require_bool(result.get("diagnostics", False), "project.diagnostics")
    return result


def project_run_path(project: dict) -> Path:
    """Return the configured run path without losing relative-path provenance."""
    run_name = str(
        require_path_value(project.get("run_name", "thermal_network"), "project.run_name")
    )
    if not run_name.strip() or run_name in {".", ".."} or Path(run_name).name != run_name:
        raise ValueError("project.run_name must be a single directory name")
    return (
        Path(require_path_value(project.get("output_dir", "outputs/runs"), "project.output_dir"))
        / run_name
    )


def validate_runtime_options(values: object, section: str, *, observer: bool = False) -> None:
    allowed = _RUNTIME_OPTIONS | ({"observer"} if observer else set())
    reject_unknown_keys(values, allowed, section)
    values = cast(Mapping[str, object], values)
    for option in ("input_dir", "output_dir"):
        if option in values:
            require_path_value(values[option], f"{section}.{option}")


def resolve_runtime_values(
    cfg: dict, section: str, *, artifact_metadata: Mapping[str, object]
) -> dict:
    """Share CSV conventions while retaining explicit runtime overrides."""
    values = cfg[section]
    validate_runtime_options(values, section, observer=True)
    data = cfg.get("data", {})
    reject_unknown_keys(data, DATA_OPTIONS, "data")
    defaults: dict[str, object] = {
        "time_col": "time",
        "sep": ",",
        "dt": None,
        "control_convention": "left",
    }
    if "data" not in cfg:
        defaults["control_convention"] = artifact_metadata.get("control_convention", "left")
    for option in defaults:
        if option in data:
            defaults[option] = data[option]
    result = {**defaults, **values}
    project = project_options(cfg)
    unit = artifact_metadata.get("temperature_unit", "degC")
    temperature_unit_label(unit)
    if "temperature_unit" in cfg.get("project", {}) and project["temperature_unit"] != unit:
        raise ValueError("project.temperature_unit must match the artifact temperature_unit")
    result["temperature_unit"] = unit
    return result


def resolve_artifact_path(cfg: dict, root: Path) -> Path:
    """Use an explicit artifact or derive the training run's artifact path."""
    project = project_options(cfg)
    if "artifact" in cfg:
        return as_path(require_path_value(cfg["artifact"], "artifact"), root)
    return _resolve_output_path(project_run_path(project), root) / "artifact"


def _resolve_output_path(configured: Path, root: Path) -> Path:
    target = as_path(configured, root).resolve()
    project = root.resolve()
    if target == Path(target.anchor) or target == project or target in project.parents:
        raise ValueError(f"output_dir must not be the project root or an ancestor: {target}")
    if not configured.is_absolute() and project not in target.parents:
        raise ValueError(
            f"relative output_dir must remain inside the project; use an absolute path: {target}"
        )
    return target


def output_target(
    cfg: dict,
    root: Path,
    *,
    section: str | None = None,
    protected_paths: Sequence[Path] = (),
) -> tuple[Path, bool]:
    """Resolve and validate an output target without changing the filesystem."""
    values = cfg if section is None else cfg[section]
    configured = Path(
        require_path_value(values["output_dir"], f"{section or 'project'}.output_dir")
    )
    target = _resolve_output_path(configured, root)
    for protected in protected_paths:
        source = protected.resolve()
        if target == source or target in source.parents or source in target.parents:
            raise ValueError(
                f"output_dir overlaps an input or artifact path: {target} and {source}"
            )
    overwrite = require_bool(
        values.get("overwrite", project_options(cfg).get("overwrite_run", False)),
        f"{section}.overwrite" if section is not None else "overwrite",
    )
    if target.exists() and not target.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {target}")
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory already exists and is not empty: {target}")
    return target, overwrite


@contextmanager
def staged_output_directory(target: Path, *, overwrite: bool) -> Generator[Path, None, None]:
    """Write a complete result beside the target and replace it only on success."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    backup: Path | None = None
    try:
        yield staging
        if target.exists():
            if not target.is_dir():
                raise FileExistsError(f"output path exists and is not a directory: {target}")
            if any(target.iterdir()) and not overwrite:
                raise FileExistsError(f"output directory already exists and is not empty: {target}")
            backup = target.with_name(f".{target.name}-backup-{uuid.uuid4().hex}")
            target.replace(backup)
        try:
            staging.replace(target)
        except BaseException:
            if backup is not None and backup.exists() and not target.exists():
                backup.replace(target)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_runtime_trajectories(
    values: dict,
    root: Path,
    *,
    sensor_names: tuple[str, ...],
    control_names: tuple[str, ...],
    require_initial_observation: bool = True,
) -> list[Trajectory]:
    """Load forecast requests or monitor logs through the shared CSV format."""
    data_cfg = {
        "directory": require_path_value(values["input_dir"], "input_dir"),
        "pattern": values.get("pattern", "*.csv"),
        "time_col": values.get("time_col", "time"),
        "sensor_cols": sensor_names,
        "control_cols": control_names,
        "control_convention": values.get("control_convention", "left"),
        "sep": values.get("sep", ","),
        "dt": values.get("dt"),
        "allow_missing_temperatures": True,
    }
    return load_trajectories(
        data_cfg, root, require_initial_observation=require_initial_observation
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path, root: Path) -> str:
    """Prefer config-relative provenance while retaining external absolute paths."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def input_file_records(trajectories: Sequence[Trajectory], root: Path) -> list[dict[str, str]]:
    """Identify the actual CSV bytes loaded, rejecting changes during a workflow."""
    records: list[dict[str, str]] = []
    for trajectory in trajectories:
        source = Path(str(trajectory.metadata["path"])).resolve()
        actual = _sha256(source)
        loaded = trajectory.metadata.get("sha256", actual)
        if loaded != actual:
            raise ValueError(f"input file changed during workflow: {source}")
        records.append(
            {"case_id": trajectory.case_id, "path": _portable_path(source, root), "sha256": actual}
        )
    return records


def trajectory_source_paths(trajectories: Sequence[Trajectory]) -> tuple[Path, ...]:
    """Protect actual sources, including files selected outside the input directory."""
    return tuple(Path(str(trajectory.metadata["path"])).resolve() for trajectory in trajectories)


def validate_runtime_manifest(
    path: Path,
    *,
    workflow: str,
    artifact: ThermalArtifact,
    values: Mapping[str, object],
    trajectories: Sequence[Trajectory],
    observer_settings: Mapping[str, object],
    disturbance_basis: str,
) -> None:
    """Bind saved case tables to the inputs, model, and conditions that produced them."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3:
        raise ValueError(f"saved {workflow} output schema is outdated; rerun {workflow}")
    if manifest.get("workflow") != workflow:
        raise ValueError(f"saved {workflow} manifest identifies a different workflow")
    saved_artifact = manifest["artifact"]
    if (
        saved_artifact["metadata_sha256"] != _sha256(artifact.path / "metadata.json")
        or saved_artifact["file_sha256"] != artifact.metadata["file_sha256"]
    ):
        raise ValueError(f"saved {workflow} artifact differs from the benchmark")
    records = manifest["input"]["files"]
    expected = {(item.case_id, item.metadata["sha256"]) for item in trajectories}
    actual = {(item["case_id"], item["sha256"]) for item in records}
    if len(records) != len(trajectories) or actual != expected:
        raise ValueError(f"saved {workflow} inputs differ from the benchmark")
    expected_settings = {
        "device": str(values.get("device", "cpu")),
        "control_convention": str(values.get("control_convention", "left")),
        "time_column": str(values.get("time_col", "time")),
        "csv_separator": str(values.get("sep", ",")),
        "expected_dt_seconds": values.get("dt"),
        "observer": {**observer_settings, "disturbance_basis": disturbance_basis},
    }
    if any(manifest["settings"].get(name) != value for name, value in expected_settings.items()):
        raise ValueError(f"saved {workflow} settings differ from the benchmark")
    if manifest.get("outputs", {}).get("case_directory") != "cases":
        raise ValueError(f"saved {workflow} must identify the cases output directory")


def write_runtime_manifest(
    target: Path,
    *,
    workflow: str,
    config_path: str | Path,
    root: Path,
    artifact: ThermalArtifact,
    values: Mapping[str, object],
    trajectories: Sequence[Trajectory],
    observer_settings: Mapping[str, object],
    disturbance_basis: str,
    uncertainty: Mapping[str, object] | None = None,
) -> None:
    """Write compact provenance shared by forecast and monitor outputs."""
    from celltemp.artifact import package_version

    inputs = input_file_records(trajectories, root)
    source_config = Path(config_path).resolve()
    manifest: dict[str, object] = {
        "schema_version": 3,
        "workflow": workflow,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package": {
            "distribution": "thermal-cell-practical",
            "version": package_version(),
        },
        "config": {
            "path": _portable_path(source_config, root),
            "relative_paths_from": "config_directory",
            "sha256": _sha256(source_config),
        },
        "artifact": {
            "path": _portable_path(artifact.path, root),
            "schema_version": artifact.metadata["schema_version"],
            "model_type": artifact.metadata["model_type"],
            "metadata_sha256": _sha256(artifact.path / "metadata.json"),
            "file_sha256": artifact.metadata["file_sha256"],
        },
        "input": {
            "directory": _portable_path(as_path(str(values["input_dir"]), root), root),
            "pattern": str(values.get("pattern", "*.csv")),
            "files": inputs,
        },
        "outputs": {
            "case_directory": "cases",
            "case_paths_relative_to": "output_directory",
            "column_layout": {
                "sensor": "sensor.<name>.<quantity>",
                "node": "node.<name>.<quantity>",
                "control": "control.<name>.<quantity>",
                "name_parsing": (
                    "fixed entity prefix and final quantity suffix; dots in names are preserved"
                ),
            },
        },
        "units": {
            "time": "s",
            "temperature": values.get(
                "temperature_unit", artifact.metadata.get("temperature_unit", "degC")
            ),
            "temperature_difference": "K",
            "disturbance": "W",
            "control": dict(
                zip(artifact.control_names, artifact.model.spec.control_units, strict=True)
            ),
        },
        "settings": {
            "device": str(values.get("device", "cpu")),
            "control_convention": str(values.get("control_convention", "left")),
            "time_column": str(values.get("time_col", "time")),
            "csv_separator": str(values.get("sep", ",")),
            "expected_dt_seconds": values.get("dt"),
            "observer": {
                **observer_settings,
                "disturbance_basis": disturbance_basis,
            },
        },
    }
    if uncertainty is not None:
        settings = manifest["settings"]
        if isinstance(settings, dict):
            settings["uncertainty"] = dict(uncertainty)
    target.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
