"""Small filesystem and path helpers shared by command workflows."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from celltemp.config import as_path
from celltemp.domain import Trajectory
from celltemp.io import load_trajectories


def resolve_artifact_path(cfg: dict, root: Path) -> Path:
    """Use an explicit artifact or derive the training run's artifact path."""
    if "artifact" in cfg:
        return as_path(str(cfg["artifact"]), root)
    project = cfg.get("project", {})
    base = as_path(str(project.get("output_dir", "outputs/runs")), root)
    run_name = str(project.get("run_name", "thermal_rc"))
    return base / run_name / "artifact"


def output_target(cfg: dict, root: Path, *, section: str | None = None) -> tuple[Path, bool]:
    """Resolve and validate an output target without changing the filesystem."""
    values = cfg if section is None else cfg[section]
    target = as_path(values["output_dir"], root).resolve()
    project = root.resolve()
    if target == project or project not in target.parents:
        raise ValueError(f"output_dir must be a child of the project root: {target}")
    overwrite = bool(values.get("overwrite", cfg.get("project", {}).get("overwrite_run", False)))
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
    try:
        yield staging
        if target.exists():
            if any(target.iterdir()) and not overwrite:
                raise FileExistsError(f"output directory already exists and is not empty: {target}")
            shutil.rmtree(target)
        staging.replace(target)
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
