"""Small filesystem and path helpers shared by command workflows."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from celltemp.artifact import ThermalArtifact, package_version
from celltemp.config import as_path, reject_unknown_keys, require_bool
from celltemp.domain import Trajectory
from celltemp.io import load_trajectories

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
    reject_unknown_keys(values, {"output_dir", "overwrite_run", "run_name"}, "project")
    result = dict(values)
    if "overwrite_run" in result:
        result["overwrite_run"] = require_bool(result["overwrite_run"], "project.overwrite_run")
    return result


def project_run_path(project: dict) -> Path:
    """Return the configured run path without losing relative-path provenance."""
    run_name = str(project.get("run_name", "thermal_network"))
    if not run_name.strip() or run_name in {".", ".."} or Path(run_name).name != run_name:
        raise ValueError("project.run_name must be a single directory name")
    return Path(str(project.get("output_dir", "outputs/runs"))) / run_name


def validate_runtime_options(values: object, section: str, *, observer: bool = False) -> None:
    allowed = _RUNTIME_OPTIONS | ({"observer"} if observer else set())
    reject_unknown_keys(values, allowed, section)


def resolve_artifact_path(cfg: dict, root: Path) -> Path:
    """Use an explicit artifact or derive the training run's artifact path."""
    project = project_options(cfg)
    if "artifact" in cfg:
        return as_path(str(cfg["artifact"]), root)
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


def output_target(cfg: dict, root: Path, *, section: str | None = None) -> tuple[Path, bool]:
    """Resolve and validate an output target without changing the filesystem."""
    values = cfg if section is None else cfg[section]
    configured = Path(str(values["output_dir"]))
    target = _resolve_output_path(configured, root)
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
def staged_output_directory(target: Path, *, overwrite: bool) -> Iterator[Path]:
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
) -> list[Trajectory]:
    """Load forecast requests or monitor logs through the shared CSV format."""
    data_cfg = {
        "directory": values["input_dir"],
        "pattern": values.get("pattern", "*.csv"),
        "time_col": values.get("time_col", "time"),
        "sensor_cols": sensor_names,
        "control_cols": control_names,
        "control_convention": values.get("control_convention", "left"),
        "sep": values.get("sep", ","),
        "dt": values.get("dt"),
        "allow_missing_temperatures": True,
    }
    return load_trajectories(data_cfg, root)


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
    inputs: list[dict[str, str]] = []
    for trajectory in trajectories:
        source = Path(str(trajectory.metadata["path"])).resolve()
        inputs.append(
            {
                "case_id": trajectory.case_id,
                "path": _portable_path(source, root),
                "sha256": _sha256(source),
            }
        )
    source_config = Path(config_path).resolve()
    manifest: dict[str, object] = {
        "schema_version": 1,
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
        "settings": {
            "device": str(values.get("device", "cpu")),
            "control_convention": str(values.get("control_convention", "left")),
            "time_column": str(values.get("time_col", "time")),
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
