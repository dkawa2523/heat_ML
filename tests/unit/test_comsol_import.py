"""COMSOL drafts remain ordinary validated system and trajectory inputs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from celltemp.domain import ConstantLawSpec
from celltemp.io import load_system_spec, load_trajectories
from external_tools import comsol_import


def _template() -> dict[str, Any]:
    return {
        "version": 3,
        "nodes": [{"name": "body", "heat_capacity": 999.0}, {"name": "shell.part"}],
        "actuators": [{"name": "power.command", "tau": 0.0, "unit": "W"}],
        "edges": [
            {
                "nodes": ["body", "shell.part"],
                "conductance": {"type": "constant", "value": 0.4, "learnable": False},
            }
        ],
        "sources": [
            {
                "name": "heater",
                "node_weights": {"body": 1.0},
                "heat_rate": {
                    "type": "positive_part",
                    "control": "power.command",
                    "gain": 1.0,
                },
            }
        ],
        "sensors": [
            {"name": "tc_body", "node": "body"},
            {"name": "mean_tc", "node_weights": {"body": 0.25, "shell.part": 0.75}},
        ],
    }


def _extracted() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0.0, 0.5, 2.0],
            "node.body.heat_capacity": [2.0, 99.0, 101.0],
            "node.shell.part.heat_capacity": [6.0, 200.0, 300.0],
            "node.body.temperature": [20.0, 25.0, 30.0],
            "node.shell.part.temperature": [40.0, 45.0, 50.0],
            "control.power.command.command": [0.0, 1.0, 7.0],
        }
    )


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "solved.mph"
    model.write_bytes(b"mock solved model")
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump(_template(), sort_keys=False), encoding="utf-8")
    return model, template


def test_imported_draft_loads_alias_weighted_sensors_controls_and_first_capacity(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, template = inputs
    original_template = template.read_bytes()
    calls: dict[str, Any] = {}

    def extract(model_path: Path, **kwargs: Any) -> pd.DataFrame:
        assert model_path == model
        calls.update(kwargs)
        (kwargs["work_dir"] / "temporary.csv").write_text("raw", encoding="utf-8")
        return _extracted()

    monkeypatch.setattr(comsol_import, "extract_model", extract)
    target = comsol_import.import_comsol_model(
        model,
        template,
        tmp_path / "draft",
        regions={"body": "body_selection", "shell.part": "2,3"},
        controls={"power.command": "source_power(t)"},
    )

    assert {path.name for path in target.iterdir()} == {"system.yaml", "trajectory.csv"}
    assert template.read_bytes() == original_template
    assert calls["control_units"] == {"power.command": "W"}
    assert not calls["work_dir"].exists()
    spec = load_system_spec(target / "system.yaml")
    assert spec.heat_capacity == (2.0, 6.0)
    assert isinstance(spec.edges[0].conductance, ConstantLawSpec)
    assert spec.edges[0].conductance.value == 0.4
    assert spec.actuators[0].unit == "W"
    trajectory = load_trajectories(
        {
            "directory": target,
            "sensor_cols": spec.sensor_names,
            "control_cols": spec.control_names,
        },
        tmp_path,
    )[0]
    np.testing.assert_array_equal(trajectory.time, [0.0, 0.5, 2.0])
    np.testing.assert_array_equal(
        trajectory.temperature, [[20.0, 35.0], [25.0, 40.0], [30.0, 45.0]]
    )
    np.testing.assert_array_equal(trajectory.commands[:, 0], [0.0, 1.0])
    frame = pd.read_csv(target / "trajectory.csv")
    np.testing.assert_array_equal(frame["power.command"], [0.0, 1.0, 7.0])


def test_capacity_replacement_does_not_mutate_the_template_mapping() -> None:
    template = _template()
    original = deepcopy(template)
    comsol_import._draft_outputs(template, _extracted())
    assert template == original


def test_completed_interval_commands_remain_compatible_with_right_convention(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, template = inputs
    extracted = _extracted()
    extracted["control.power.command.command"] = [0.0, 0.0, 1.0]
    monkeypatch.setattr(comsol_import, "extract_model", lambda *_args, **_kwargs: extracted)
    output = comsol_import.import_comsol_model(
        model,
        template,
        tmp_path / "draft",
        regions={"body": "1", "shell.part": "2"},
        controls={"power.command": "piecewise_power(t)"},
    )
    spec = load_system_spec(output / "system.yaml")
    trajectory = load_trajectories(
        {
            "directory": output,
            "sensor_cols": spec.sensor_names,
            "control_cols": spec.control_names,
            "control_convention": "right",
        },
        tmp_path,
    )[0]
    np.testing.assert_array_equal(trajectory.commands[:, 0], [0.0, 1.0])
    np.testing.assert_array_equal(
        pd.read_csv(output / "trajectory.csv")["power.command"], [0.0, 0.0, 1.0]
    )


@pytest.mark.parametrize(
    ("regions", "controls"),
    [
        ({"body": "1"}, {"power.command": "P"}),
        ({"body": "1", "shell.part": "2", "extra": "3"}, {"power.command": "P"}),
        ({"body": "1", "shell.part": "2"}, {}),
        ({"body": "1", "shell.part": "2"}, {"power.command": "P", "extra": "Q"}),
    ],
)
def test_regions_and_controls_match_every_template_entity_before_extraction(
    regions: dict[str, str],
    controls: dict[str, str],
    inputs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, template = inputs

    def extract(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        pytest.fail("entity bindings must be checked before extracting")

    monkeypatch.setattr(comsol_import, "extract_model", extract)
    with pytest.raises(ValueError, match="every template entity exactly once"):
        comsol_import.import_comsol_model(
            model, template, tmp_path / "draft", regions=regions, controls=controls
        )


@pytest.mark.parametrize("capacity", [0.0, -2.0, np.nan, np.inf])
def test_invalid_extracted_capacity_is_rejected(capacity: float) -> None:
    extracted = _extracted()
    extracted.loc[0, "node.body.heat_capacity"] = capacity
    with pytest.raises(ValueError, match=r"heat_capacity|finite"):
        comsol_import._draft_outputs(_template(), extracted)


@pytest.mark.parametrize("failure", ["extraction", "capacity", "save"])
def test_failed_import_keeps_existing_output(
    failure: str,
    inputs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, template = inputs
    target = tmp_path / "draft"
    target.mkdir()
    (target / "previous.csv").write_text("previous", encoding="utf-8")

    def extract(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        if failure == "extraction":
            raise RuntimeError("extraction failed")
        frame = _extracted()
        if failure == "capacity":
            frame.loc[0, "node.body.heat_capacity"] = -1.0
        return frame

    def fail_save(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("save failed")

    monkeypatch.setattr(comsol_import, "extract_model", extract)
    if failure == "save":
        monkeypatch.setattr(comsol_import, "save_system_spec", fail_save)
    with pytest.raises((RuntimeError, ValueError, OSError)):
        comsol_import.import_comsol_model(
            model,
            template,
            target,
            regions={"body": "body_selection", "shell.part": "2,3"},
            controls={"power.command": "source_power(t)"},
            overwrite=True,
        )
    assert {path.name for path in target.iterdir()} == {"previous.csv"}
    assert (target / "previous.csv").read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".draft-*"))


@pytest.mark.parametrize("protected", ["model", "template"])
def test_import_cannot_replace_an_input_or_its_parent(
    protected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_parent = tmp_path / "models"
    template_parent = tmp_path / "templates"
    model_parent.mkdir()
    template_parent.mkdir()
    model = model_parent / "solved.mph"
    model.write_bytes(b"model")
    template = template_parent / "template.yaml"
    template.write_text(yaml.safe_dump(_template()), encoding="utf-8")

    def extract(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        pytest.fail("an overlapping output must be rejected before extracting")

    monkeypatch.setattr(comsol_import, "extract_model", extract)
    target = model_parent if protected == "model" else template_parent
    with pytest.raises(ValueError, match="overlaps an input"):
        comsol_import.import_comsol_model(
            model,
            template,
            target,
            regions={"body": "1", "shell.part": "2"},
            controls={"power.command": "P"},
            overwrite=True,
        )
    assert model.read_bytes() == b"model"
    assert "shell.part" in template.read_text(encoding="utf-8")


def test_nodes_without_controls_need_no_control_bindings(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, template = inputs
    mapping = _template()
    mapping.pop("actuators")
    mapping.pop("sources")
    template.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    monkeypatch.setattr(comsol_import, "extract_model", lambda *_args, **_kwargs: _extracted())
    output = comsol_import.import_comsol_model(
        model, template, tmp_path / "draft", regions={"body": "1", "shell.part": "2"}, controls={}
    )
    assert load_system_spec(output / "system.yaml").control_names == ()
    assert list(pd.read_csv(output / "trajectory.csv")) == ["time", "tc_body", "mean_tc"]


def test_cli_forwards_explicit_extraction_options_and_keeps_kelvin(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, template = inputs
    calls: dict[str, Any] = {}

    def extract(_model_path: Path, **kwargs: Any) -> pd.DataFrame:
        calls.update(kwargs)
        frame = _extracted()
        frame[["node.body.temperature", "node.shell.part.temperature"]] += 273.15
        return frame

    monkeypatch.setattr(comsol_import, "extract_model", extract)
    target = tmp_path / "draft"
    code = comsol_import.main(
        [
            "--model",
            str(model),
            "--template",
            str(template),
            "--output-dir",
            str(target),
            "--region",
            "body=body_selection",
            "--region",
            "shell.part=2,3",
            "--control",
            "power.command=source_power(t)",
            "--dataset",
            "dset3",
            "--component",
            "solid",
            "--capacity-expression",
            "solid.rho*solid.Cp",
            "--temperature-unit",
            "K",
            "--comsol-root",
            str(tmp_path / "comsol"),
        ]
    )
    assert code == 0
    assert calls["dataset"] == "dset3"
    assert calls["component"] == "solid"
    assert calls["capacity_expression"] == "solid.rho*solid.Cp"
    assert calls["temperature_unit"] == "K"
    assert calls["comsol_root"] == str(tmp_path / "comsol")
    np.testing.assert_allclose(
        pd.read_csv(target / "trajectory.csv")["tc_body"], [293.15, 298.15, 303.15]
    )


@pytest.mark.parametrize("values", [["body=1", "body=2"], ["body"], ["body="]])
def test_cli_requires_each_assignment_once(values: list[str]) -> None:
    with pytest.raises(ValueError, match=r"repeats|name=value"):
        comsol_import._assignments(values, "--region")


@pytest.mark.parametrize(("unit", "requested"), [("degC", "K"), ("K", "degC")])
def test_temperature_control_scale_mismatch_is_rejected_before_extraction(
    unit: str,
    requested: str,
    inputs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, template = inputs
    mapping = _template()
    mapping["actuators"][0]["unit"] = unit
    mapping["boundaries"] = [
        {
            "name": "coolant",
            "node_weights": {"body": 1.0},
            "reservoir_temperature": {"control": "power.command", "intercept": 0.0, "slope": 1.0},
            "conductance": {"type": "constant", "value": 0.2},
        }
    ]
    template.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    def extract(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        pytest.fail("a temperature unit mismatch must be rejected before extracting")

    monkeypatch.setattr(comsol_import, "extract_model", extract)
    with pytest.raises(ValueError, match="same temperature unit"):
        comsol_import.import_comsol_model(
            model,
            template,
            tmp_path / "draft",
            regions={"body": "1", "shell.part": "2"},
            controls={"power.command": "coolant_temperature"},
            temperature_unit=requested,
        )
