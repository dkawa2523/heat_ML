"""Small filesystem and path helpers shared by command workflows."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from celltemp.config import as_path, reject_unknown_keys
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
    return dict(values)


def validate_runtime_options(values: object, section: str, *, observer: bool = False) -> None:
    allowed = _RUNTIME_OPTIONS | ({"observer"} if observer else set())
    reject_unknown_keys(values, allowed, section)


def resolve_artifact_path(cfg: dict, root: Path) -> Path:
    """Use an explicit artifact or derive the training run's artifact path."""
    project = project_options(cfg)
    if "artifact" in cfg:
        return as_path(str(cfg["artifact"]), root)
    base = as_path(str(project.get("output_dir", "outputs/runs")), root)
    run_name = str(project.get("run_name", "thermal_rc"))
    return base / run_name / "artifact"


def output_target(cfg: dict, root: Path, *, section: str | None = None) -> tuple[Path, bool]:
    """Resolve and validate an output target without changing the filesystem."""
    values = cfg if section is None else cfg[section]
    configured = Path(str(values["output_dir"]))
    target = as_path(configured, root).resolve()
    project = root.resolve()
    if target == Path(target.anchor) or target == project or target in project.parents:
        raise ValueError(f"output_dir must not be the project root or an ancestor: {target}")
    if not configured.is_absolute() and project not in target.parents:
        raise ValueError(
            f"relative output_dir must remain inside the project; use an absolute path: {target}"
        )
    overwrite = bool(values.get("overwrite", project_options(cfg).get("overwrite_run", False)))
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
