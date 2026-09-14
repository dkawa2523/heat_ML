"""Simple whole-trajectory train/validation/test splitting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np

from celltemp.domain import Trajectory

Split = dict[str, list[Trajectory]]

_DEFAULT_RECIPE_RTOL = 1e-9
_DEFAULT_RECIPE_ATOL = 1e-9


def _same_recipe(
    first: Trajectory,
    second: Trajectory,
    *,
    rtol: float,
    atol: float,
) -> bool:
    return (
        first.commands.shape == second.commands.shape
        and np.allclose(first.dt, second.dt, rtol=rtol, atol=atol)
        and np.allclose(first.commands, second.commands, rtol=rtol, atol=atol)
    )


def _control_groups(
    trajectories: Sequence[Trajectory], *, rtol: float, atol: float
) -> list[list[Trajectory]]:
    """Keep numerically equivalent command recipes in one evaluation partition."""
    groups: list[list[Trajectory]] = []
    for trajectory in trajectories:
        matching = next(
            (group for group in groups if _same_recipe(group[0], trajectory, rtol=rtol, atol=atol)),
            None,
        )
        if matching is None:
            groups.append([trajectory])
        else:
            matching.append(trajectory)
    return groups


def _random_split(
    trajectories: Sequence[Trajectory], split_cfg: Mapping[str, object], seed: int
) -> Split:
    recipe_rtol = float(cast(int | float | str, split_cfg.get("recipe_rtol", _DEFAULT_RECIPE_RTOL)))
    recipe_atol = float(cast(int | float | str, split_cfg.get("recipe_atol", _DEFAULT_RECIPE_ATOL)))
    if not math.isfinite(recipe_rtol) or recipe_rtol < 0.0:
        raise ValueError("split.recipe_rtol must be non-negative and finite")
    if not math.isfinite(recipe_atol) or recipe_atol < 0.0:
        raise ValueError("split.recipe_atol must be non-negative and finite")
    groups = _control_groups(trajectories, rtol=recipe_rtol, atol=recipe_atol)
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
