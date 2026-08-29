"""Simple whole-trajectory train/validation/test splitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np

from celltemp.domain import Trajectory

Split = dict[str, list[Trajectory]]


def _control_key(trajectory: Trajectory) -> tuple[tuple[int, ...], bytes, bytes]:
    """Identify trajectories with the same time grid and applied command history."""
    commands = np.ascontiguousarray(trajectory.commands, dtype=np.float64)
    dt = np.ascontiguousarray(trajectory.dt, dtype=np.float64)
    return commands.shape, dt.tobytes(), commands.tobytes()


def _groups(
    trajectories: Sequence[Trajectory], *, group_by_controls: bool
) -> list[list[Trajectory]]:
    if not group_by_controls:
        return [[trajectory] for trajectory in trajectories]
    grouped: dict[tuple[tuple[int, ...], bytes, bytes], list[Trajectory]] = {}
    for trajectory in trajectories:
        grouped.setdefault(_control_key(trajectory), []).append(trajectory)
    return list(grouped.values())


def _random_split(
    trajectories: Sequence[Trajectory], split_cfg: Mapping[str, object], seed: int
) -> Split:
    groups = _groups(
        trajectories,
        group_by_controls=bool(split_cfg.get("group_by_controls", True)),
    )
    order = np.arange(len(groups))
    np.random.default_rng(seed).shuffle(order)
    shuffled = [groups[index] for index in order]
    count = len(shuffled)
    train_ratio = float(cast(int | float | str, split_cfg.get("train_ratio", 0.7)))
    validation_ratio = float(cast(int | float | str, split_cfg.get("val_ratio", 0.15)))
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("split.train_ratio must be between 0 and 1")
    if not 0.0 <= validation_ratio < 1.0 or train_ratio + validation_ratio >= 1.0:
        raise ValueError("split.val_ratio must be non-negative and leave a test fraction")
    train_count = max(1, round(count * train_ratio))
    validation_count = round(count * validation_ratio)
    if count >= 3:
        validation_count = max(1, validation_count)
    if train_count + validation_count >= count and count >= 3:
        train_count, validation_count = max(1, count - 2), 1

    def flatten(parts: Sequence[Sequence[Trajectory]]) -> list[Trajectory]:
        return [trajectory for part in parts for trajectory in part]

    return {
        "train": flatten(shuffled[:train_count]),
        "val": flatten(shuffled[train_count : train_count + validation_count]),
        "test": flatten(shuffled[train_count + validation_count :]),
    }


def _explicit_split(
    trajectories: Sequence[Trajectory], assignments: Mapping[str, str] | None
) -> Split:
    if assignments is None:
        raise ValueError("split.method=explicit requires split.table")
    known = {trajectory.case_id for trajectory in trajectories}
    if set(assignments) != known:
        missing = sorted(known - set(assignments))
        extra = sorted(set(assignments) - known)
        raise ValueError(
            f"split table must cover every trajectory; missing={missing}, extra={extra}"
        )
    result: Split = {"train": [], "val": [], "test": []}
    for trajectory in trajectories:
        value = assignments[trajectory.case_id].strip().lower()
        if value not in result:
            raise ValueError("split table values must be train, val, or test")
        result[value].append(trajectory)
    if not result["train"]:
        raise ValueError("explicit split must contain at least one training trajectory")
    return result


def split_trajectories(
    trajectories: Sequence[Trajectory],
    split_cfg: Mapping[str, object] | None = None,
    *,
    seed: int = 42,
    assignments: Mapping[str, str] | None = None,
) -> Split:
    """Split whole trajectories, grouping identical command histories by default."""
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    values = split_cfg or {}
    method = str(values.get("method", "random"))
    if method == "random":
        return _random_split(trajectories, values, seed)
    if method == "explicit":
        return _explicit_split(trajectories, assignments)
    raise ValueError("split.method must be 'random' or 'explicit'")
