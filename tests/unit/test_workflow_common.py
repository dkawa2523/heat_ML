"""Output boundaries shared by the three filesystem workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from celltemp.io import load_trajectories
from celltemp.workflows.common import (
    input_file_records,
    load_runtime_trajectories,
    output_target,
    project_options,
    project_run_path,
    resolve_runtime_values,
    staged_output_directory,
)


def test_output_target_requires_an_explicit_absolute_path_outside_project(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external-results"
    target, _ = output_target({"output_dir": str(external)}, project)
    assert target == external.resolve()

    with pytest.raises(ValueError, match="relative output_dir"):
        output_target({"output_dir": "../external-results"}, project)
    with pytest.raises(ValueError, match="root or an ancestor"):
        output_target({"output_dir": str(tmp_path)}, project)


def test_project_run_path_cannot_hide_relative_escape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    configured = project_run_path({"output_dir": "../external-results", "run_name": "run"})
    with pytest.raises(ValueError, match="relative output_dir"):
        output_target({"output_dir": configured}, project)

    with pytest.raises(ValueError, match="single directory name"):
        project_run_path({"output_dir": str(tmp_path), "run_name": "../external-results"})


def test_output_target_rejects_string_boolean(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overwrite must be boolean"):
        output_target({"output_dir": "results", "overwrite": "false"}, tmp_path)
    with pytest.raises(ValueError, match=r"project\.overwrite_run must be boolean"):
        output_target(
            {
                "output_dir": "results",
                "project": {"overwrite_run": "false"},
            },
            tmp_path,
        )


@pytest.mark.parametrize("output", ["data", "data/results", "shared"])
def test_output_target_protects_inputs_from_replacement(tmp_path: Path, output: str) -> None:
    source = tmp_path / ("shared/data" if output == "shared" else "data")
    source.mkdir(parents=True)
    (source / "measurement.csv").write_text("original", encoding="utf-8")

    with pytest.raises(ValueError, match="overlaps an input or artifact"):
        output_target(
            {"output_dir": output, "overwrite": True},
            tmp_path,
            protected_paths=(source,),
        )
    assert (source / "measurement.csv").read_text(encoding="utf-8") == "original"


def test_staged_output_replaces_only_after_success(tmp_path: Path) -> None:
    target = tmp_path / "results"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")

    def fail_during_staging() -> None:
        with staged_output_directory(target, overwrite=True) as staging:
            (staging / "new.txt").write_text("incomplete", encoding="utf-8")
            raise RuntimeError("processing failed")

    with pytest.raises(RuntimeError, match="processing failed"):
        fail_during_staging()
    assert (target / "old.txt").read_text(encoding="utf-8") == "old"

    with staged_output_directory(target, overwrite=True) as staging:
        (staging / "new.txt").write_text("complete", encoding="utf-8")
    assert not (target / "old.txt").exists()
    assert (target / "new.txt").read_text(encoding="utf-8") == "complete"
    assert not list(tmp_path.glob(".results-backup-*"))


def test_input_manifest_rejects_measurements_changed_after_loading(tmp_path: Path) -> None:
    source = tmp_path / "case.csv"
    source.write_text("time,tc\n0,20\n1,21\n", encoding="utf-8")
    cases = load_trajectories({"directory": ".", "sensor_cols": ["tc"]}, tmp_path)
    records = input_file_records(cases, tmp_path)
    assert records[0]["sha256"] == cases[0].metadata["sha256"]
    source.write_text("time,tc\n0,20\n1,99\n", encoding="utf-8")

    with pytest.raises(ValueError, match="input file changed during workflow"):
        input_file_records(cases, tmp_path)


@pytest.mark.parametrize("section", ["forecast", "monitor"])
def test_runtime_csv_conventions_inherit_from_data_and_load_consistently(
    tmp_path: Path, section: str
) -> None:
    (tmp_path / "case.csv").write_text(
        "elapsed;tc;power\n10;20;0\n12;21;5\n14;22;10\n", encoding="utf-8"
    )
    cfg = {
        "data": {"time_col": "elapsed", "sep": ";", "control_convention": "right", "dt": 2},
        section: {"input_dir": ".", "output_dir": "results"},
    }
    values = resolve_runtime_values(cfg, section, artifact_metadata={})
    cases = load_runtime_trajectories(
        values, tmp_path, sensor_names=("tc",), control_names=("power",)
    )
    np.testing.assert_array_equal(cases[0].commands[:, 0], [5, 10])
    assert values["time_col"] == "elapsed"
    assert values["sep"] == ";"
    assert values["control_convention"] == "right"
    assert values["dt"] == 2


def test_explicit_runtime_csv_values_override_data_including_null_dt() -> None:
    cfg = {
        "data": {"time_col": "elapsed", "sep": ";", "control_convention": "right", "dt": 2},
        "forecast": {"time_col": "time", "sep": ",", "control_convention": "left", "dt": None},
    }
    values = resolve_runtime_values(cfg, "forecast", artifact_metadata={})
    assert {key: values[key] for key in cfg["forecast"]} == cfg["forecast"]


def test_artifact_only_runtime_uses_saved_control_convention_and_temperature_unit() -> None:
    values = resolve_runtime_values(
        {"monitor": {}},
        "monitor",
        artifact_metadata={"control_convention": "right", "temperature_unit": "K"},
    )
    assert values["control_convention"] == "right"
    assert values["temperature_unit"] == "K"
    assert values["time_col"] == "time"
    assert values["sep"] == ","
    assert values["dt"] is None
    # With a data section, training's usual left default takes precedence.
    values = resolve_runtime_values(
        {"data": {}, "monitor": {}}, "monitor", artifact_metadata={"control_convention": "right"}
    )
    assert values["control_convention"] == "left"


def test_project_temperature_unit_must_match_runtime_artifact() -> None:
    assert project_options({})["temperature_unit"] == "degC"
    cfg = {"project": {"temperature_unit": "K"}, "forecast": {}}
    assert (
        resolve_runtime_values(cfg, "forecast", artifact_metadata={"temperature_unit": "K"})[
            "temperature_unit"
        ]
        == "K"
    )
    with pytest.raises(ValueError, match="must match the artifact"):
        resolve_runtime_values(cfg, "forecast", artifact_metadata={})
    with pytest.raises(ValueError, match="unknown forecast options"):
        resolve_runtime_values(
            {"forecast": {"temperature_unit": "K"}}, "forecast", artifact_metadata={}
        )


@pytest.mark.parametrize("value", [None, "", " ", True])
def test_project_paths_reject_bad_supplied_values(value: object) -> None:
    for option in ("output_dir", "run_name"):
        with pytest.raises(ValueError, match=rf"project\.{option} must be a non-empty path"):
            project_options({"project": {option: value}})


@pytest.mark.parametrize("option", ["input_dir", "output_dir"])
def test_runtime_paths_reject_supplied_null(option: str) -> None:
    with pytest.raises(ValueError, match=rf"forecast\.{option} must be a non-empty path"):
        resolve_runtime_values({"forecast": {option: None}}, "forecast", artifact_metadata={})
