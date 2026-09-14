"""CSV adapters for self-contained thermal trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from celltemp.config import as_path, require_bool
from celltemp.domain import Trajectory

_INITIAL_ACTUATOR_PREFIX = "initial_effective_"


def _initial_actuator(
    frame: pd.DataFrame,
    control_names: tuple[str, ...],
    default: np.ndarray,
) -> np.ndarray | None:
    """Read an optional effective actuator state from the first CSV row."""
    columns = tuple(f"{_INITIAL_ACTUATOR_PREFIX}{name}" for name in control_names)
    present = tuple(name in frame.columns for name in columns)
    if not any(present):
        return None
    values = np.array(default, dtype=np.float64, copy=True)
    for index, (column, exists) in enumerate(zip(columns, present, strict=True)):
        if exists:
            values[index] = float(frame.loc[frame.index[0], column])
    if not np.isfinite(values).all():
        raise ValueError("initial actuator values on the first row must be finite")
    return values


def trajectory_from_frame(
    *,
    case_id: str,
    frame: pd.DataFrame,
    time_col: str,
    sensor_cols: Sequence[str],
    control_cols: Sequence[str],
    control_convention: str = "left",
    metadata: Mapping[str, object] | None = None,
) -> Trajectory:
    """Convert one self-contained timestamped table."""
    sensor_names = tuple(sensor_cols)
    control_names = tuple(control_cols)
    required = [time_col, *sensor_names, *control_names]
    if len(required) != len(set(required)):
        raise ValueError("time, sensor, and control column names must be distinct")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"trajectory table is missing columns {missing}")

    temperature = frame[list(sensor_names)].to_numpy(dtype=np.float64)
    if np.isinf(temperature).any():
        raise ValueError("temperature observations may be missing, but not infinite")
    sampled_controls = frame[list(control_names)].to_numpy(dtype=np.float64)
    default_initial = (
        sampled_controls[0] if control_convention.lower() == "left" else sampled_controls[1]
    )
    return Trajectory.from_sampled_controls(
        case_id=case_id,
        time=frame[time_col].to_numpy(dtype=np.float64),
        temperature=temperature,
        sampled_controls=sampled_controls,
        sensor_names=sensor_names,
        control_names=control_names,
        convention=control_convention,
        observation_mask=np.isfinite(temperature),
        metadata=metadata,
        initial_actuator=_initial_actuator(frame, control_names, default_initial),
    )


def _validate_loaded_trajectory(
    trajectory: Trajectory,
    data_cfg: Mapping[str, Any],
    source: Path,
) -> None:
    allow_missing = require_bool(
        data_cfg.get("allow_missing_temperatures", False),
        "data.allow_missing_temperatures",
    )
    if not allow_missing and not trajectory.mask.all():
        raise ValueError(f"{source.name}: missing temperature observations are not enabled")
    if not trajectory.mask[0].any():
        raise ValueError(f"{source.name}: the initial row needs at least one temperature")
    expected_dt = data_cfg.get("dt")
    if expected_dt is not None and not np.allclose(
        trajectory.dt, float(expected_dt), rtol=1e-4, atol=1e-8
    ):
        raise ValueError(f"{source.name}: dt is not constant {expected_dt}")
    observed = trajectory.temperature[trajectory.mask]
    low = data_cfg.get("temp_min")
    high = data_cfg.get("temp_max")
    if low is not None and observed.min() < float(low):
        raise ValueError(f"{source.name}: temperature below temp_min")
    if high is not None and observed.max() > float(high):
        raise ValueError(f"{source.name}: temperature above temp_max")


def load_trajectories(data_cfg: Mapping[str, Any], root: str | Path) -> list[Trajectory]:
    """Discover self-contained trajectory CSVs under one configured directory."""
    project_root = Path(root)
    directory_value = data_cfg.get("directory")
    if not directory_value:
        raise ValueError("data.directory is required")
    directory = as_path(str(directory_value), project_root)
    pattern = str(data_cfg.get("pattern", "*.csv"))
    paths = sorted(path for path in directory.glob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"no trajectory CSV files found under: {directory}")
    case_ids = [path.stem for path in paths]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("trajectory file stems must be unique case identifiers")

    time_col = str(data_cfg.get("time_col", "time"))
    sensors = tuple(str(name) for name in data_cfg["sensor_cols"])
    controls = tuple(str(name) for name in data_cfg.get("control_cols", ()))
    separator = str(data_cfg.get("sep", ","))
    trajectories: list[Trajectory] = []
    for source in paths:
        frame = pd.read_csv(source, sep=separator)
        frame.columns = [str(name).strip() for name in frame.columns]
        trajectory = trajectory_from_frame(
            case_id=source.stem,
            frame=frame,
            time_col=time_col,
            sensor_cols=sensors,
            control_cols=controls,
            control_convention=str(data_cfg.get("control_convention", "left")),
            metadata={"path": str(source)},
        )
        _validate_loaded_trajectory(trajectory, data_cfg, source)
        trajectories.append(trajectory)
    return trajectories


def load_split_assignments(path: str | Path) -> dict[str, str]:
    """Read an optional two-column ``case_id,split`` evaluation table."""
    table = pd.read_csv(path)
    required = {"case_id", "split"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"split table is missing columns {missing}")
    if table.empty or table["case_id"].isna().any():
        raise ValueError("split table must contain case identifiers")
    case_ids = table["case_id"].astype(str)
    if case_ids.duplicated().any():
        raise ValueError("split table case_id values must be unique")
    return dict(zip(case_ids, table["split"].astype(str)))
