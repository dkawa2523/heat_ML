"""Folder discovery and whole-trajectory splitting."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from celltemp.io import load_split_assignments, load_trajectories
from celltemp.learning import split_trajectories


def _sources(root: Path) -> list[Path]:
    return sorted((root / "data" / "raw").glob("*.csv"))


def _first_source(root: Path) -> Path:
    return _sources(root)[0]


def test_folder_discovery_does_not_interpret_file_names(cae_project: Path, data_cfg: dict) -> None:
    source = _first_source(cae_project)
    renamed = source.with_name("arbitrary-cae-export.csv")
    source.rename(renamed)
    trajectories = load_trajectories(data_cfg, cae_project)
    arbitrary = next(item for item in trajectories if item.case_id == "arbitrary-cae-export")
    frame = pd.read_csv(renamed)
    np.testing.assert_allclose(
        arbitrary.commands[0], frame.loc[0, ["brine", "heater", "plasma"]].to_numpy(float)
    )


def test_time_varying_controls_are_read_directly(cae_project: Path, data_cfg: dict) -> None:
    source = _first_source(cae_project)
    frame = pd.read_csv(source)
    frame["heater"] = np.linspace(80.0, 160.0, len(frame))
    frame.to_csv(source, index=False)
    trajectory = next(
        item for item in load_trajectories(data_cfg, cae_project) if item.case_id == source.stem
    )
    assert np.ptp(trajectory.commands[:, 1]) > 0.0


def test_directory_and_pattern_are_the_only_discovery_settings(
    cae_project: Path, data_cfg: dict
) -> None:
    with pytest.raises(ValueError, match=r"data\.directory"):
        load_trajectories({**data_cfg, "directory": ""}, cae_project)
    with pytest.raises(FileNotFoundError, match="no trajectory CSV"):
        load_trajectories({**data_cfg, "pattern": "none-*.csv"}, cae_project)


@pytest.mark.parametrize("missing", ["time", "Center", "plasma"])
def test_every_csv_is_self_contained(cae_project: Path, data_cfg: dict, missing: str) -> None:
    source = _first_source(cae_project)
    pd.read_csv(source).drop(columns=[missing]).to_csv(source, index=False)
    with pytest.raises(ValueError, match=r"missing columns"):
        load_trajectories(data_cfg, cae_project)


def _rewrite_first(root: Path, transform) -> None:
    source = _first_source(root)
    transform(pd.read_csv(source)).to_csv(source, index=False)


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (lambda frame: frame.assign(time=lambda value: value["time"] ** 1.2), "dt is not"),
        (
            lambda frame: frame.assign(time=lambda value: value["time"].mask(value.index == 5, 4)),
            "strictly increasing",
        ),
        (
            lambda frame: frame.assign(Center=lambda value: value["Center"].mask(value.index == 5)),
            "missing temperature",
        ),
        (
            lambda frame: frame.assign(
                Center=lambda value: value["Center"].mask(value.index == 5, np.inf)
            ),
            "not infinite",
        ),
        (
            lambda frame: frame.assign(
                edge=lambda value: value["edge"].mask(value.index == 5, 9999)
            ),
            "above temp_max",
        ),
    ],
)
def test_trajectory_values_are_validated(
    cae_project: Path, data_cfg: dict, transform, message: str
) -> None:
    _rewrite_first(cae_project, transform)
    with pytest.raises(ValueError, match=message):
        load_trajectories(data_cfg, cae_project)


def test_enabled_missing_observation_is_preserved(cae_project: Path, data_cfg: dict) -> None:
    _rewrite_first(
        cae_project,
        lambda frame: frame.assign(Center=lambda value: value["Center"].mask(value.index == 5)),
    )
    source_id = _first_source(cae_project).stem
    trajectory = next(
        item
        for item in load_trajectories({**data_cfg, "allow_missing_temperatures": True}, cae_project)
        if item.case_id == source_id
    )
    assert not trajectory.mask[5, 1]


def test_identical_control_histories_never_leak_between_splits(
    cae_project: Path, data_cfg: dict
) -> None:
    source = _first_source(cae_project)
    replica = source.with_name("same-controls-different-initial-state.csv")
    shutil.copyfile(source, replica)
    frame = pd.read_csv(replica)
    frame[["CP", "Center", "middle", "edge"]] += 2.0
    frame.to_csv(replica, index=False)

    trajectories = load_trajectories(data_cfg, cae_project)
    split = split_trajectories(trajectories, {"method": "random"}, seed=7)
    location = {trajectory.case_id: name for name, items in split.items() for trajectory in items}
    assert location[source.stem] == location[replica.stem]
    assert len(location) == len(trajectories)


def test_random_split_is_deterministic(cae_project: Path, data_cfg: dict) -> None:
    trajectories = load_trajectories(data_cfg, cae_project)
    first = split_trajectories(trajectories, {"method": "random"}, seed=11)
    second = split_trajectories(trajectories, {"method": "random"}, seed=11)
    assert {name: [item.case_id for item in values] for name, values in first.items()} == {
        name: [item.case_id for item in values] for name, values in second.items()
    }


def test_optional_split_table_is_only_case_id_and_partition(
    cae_project: Path, data_cfg: dict
) -> None:
    trajectories = load_trajectories(data_cfg, cae_project)
    rows = [
        {
            "case_id": trajectory.case_id,
            "split": "train" if index < 5 else "val" if index < 7 else "test",
        }
        for index, trajectory in enumerate(trajectories)
    ]
    path = cae_project / "split.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    assignments = load_split_assignments(path)
    split = split_trajectories(
        trajectories,
        {"method": "explicit"},
        assignments=assignments,
    )
    assert {name: len(items) for name, items in split.items()} == {
        "train": 5,
        "val": 2,
        "test": 1,
    }


def test_split_table_must_match_discovered_cases(cae_project: Path, data_cfg: dict) -> None:
    trajectories = load_trajectories(data_cfg, cae_project)
    assignments = {trajectory.case_id: "train" for trajectory in trajectories[:-1]}
    with pytest.raises(ValueError, match="cover every trajectory"):
        split_trajectories(
            trajectories,
            {"method": "explicit"},
            assignments=assignments,
        )


def test_split_rejects_unknown_method_or_invalid_label(cae_project: Path, data_cfg: dict) -> None:
    trajectories = load_trajectories(data_cfg, cae_project)
    with pytest.raises(ValueError, match=r"random.*explicit"):
        split_trajectories(trajectories, {"method": "holdout"})
    assignments = {trajectory.case_id: "train" for trajectory in trajectories}
    assignments[trajectories[-1].case_id] = "development"
    with pytest.raises(ValueError, match="train, val, or test"):
        split_trajectories(
            trajectories,
            {"method": "explicit"},
            assignments=assignments,
        )


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"case_id": ["a"]}), "missing columns"),
        (pd.DataFrame({"case_id": ["a", "a"], "split": ["train", "test"]}), "unique"),
    ],
)
def test_split_table_reader_rejects_ambiguous_rows(
    tmp_path: Path, frame: pd.DataFrame, message: str
) -> None:
    path = tmp_path / "split.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_split_assignments(path)
