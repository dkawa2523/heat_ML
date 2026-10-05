"""CSV defaults and recorded runtime conditions use the same resolved values."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from celltemp.artifact import save_artifact
from celltemp.config import save_yaml
from celltemp.domain import ActuatorSpec, PositivePartLawSpec, SourceSpec, ThermalSystemSpec
from celltemp.engine import ThermalRCModel
from celltemp.workflows.forecast import run_forecast
from celltemp.workflows.monitor import run_monitor
from celltemp.workflows.train import run_train

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("with_data", [True, False])
def test_runtime_csv_defaults_are_used_by_predictions_and_manifest(
    tmp_path: Path, with_data: bool
) -> None:
    model = ThermalRCModel(
        ThermalSystemSpec(
            node_names=("body",),
            heat_capacity=(2.0,),
            edges=(),
            actuators=(ActuatorSpec("power", tau=0.0, learnable=False, unit="W"),),
            sources=(SourceSpec("heater", (1.0,), PositivePartLawSpec("power", 1.0)),),
        )
    )
    save_artifact(
        tmp_path / "artifact",
        model,
        metadata={"control_convention": "right", "temperature_unit": "K"},
    )
    (tmp_path / "request").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "request" / "case.csv").write_text(
        "elapsed;body;power\n10;293;0\n12;298;5\n14;;10\n", encoding="utf-8"
    )
    (tmp_path / "logs" / "case.csv").write_text(
        "elapsed;body;power\n10;293;0\n12;298;5\n14;308;10\n", encoding="utf-8"
    )
    cfg: dict = {
        "artifact": "artifact",
        "project": {"temperature_unit": "K"},
        "forecast": {"input_dir": "request", "output_dir": "forecast"},
        "monitor": {"input_dir": "logs", "output_dir": "monitor"},
    }
    csv_values = {"time_col": "elapsed", "sep": ";", "dt": 2}
    if with_data:
        cfg["data"] = {**csv_values, "control_convention": "right"}
    else:
        # Standalone artifacts retain the saved command convention; the other
        # CSV fields can still be explicitly configured without a training section.
        cfg["forecast"].update(csv_values)
        cfg["monitor"].update(csv_values)
    config_path = tmp_path / "config.yaml"
    save_yaml(cfg, config_path)

    run_forecast(cfg, config_path)
    run_monitor(cfg, config_path)

    predicted = pd.read_csv(tmp_path / "forecast" / "cases" / "case.csv")
    assert predicted["time"].tolist() == [12, 14]
    assert predicted["sensor.body.temperature"].diff().iloc[-1] == pytest.approx(10)
    monitored = pd.read_csv(tmp_path / "monitor" / "cases" / "case.csv")
    assert monitored["control.power.command"].tolist() == [5, 10, 10]
    for workflow in ("forecast", "monitor"):
        manifest = json.loads((tmp_path / workflow / "run_manifest.json").read_text())
        assert manifest["settings"]["control_convention"] == "right"
        assert manifest["settings"]["time_column"] == "elapsed"
        assert manifest["settings"]["csv_separator"] == ";"
        assert manifest["settings"]["expected_dt_seconds"] == 2
        assert manifest["units"]["temperature"] == "K"


@pytest.mark.parametrize("value", [None, True, " "])
def test_training_rejects_bad_split_table_before_opening_input_files(
    tmp_path: Path, value: object
) -> None:
    cfg = {"system": "system.yaml", "data": {}, "split": {"method": "explicit", "table": value}}
    with pytest.raises(ValueError, match=r"split\.table must be a non-empty path"):
        run_train(cfg, tmp_path / "config.yaml")


@pytest.mark.parametrize("section", ["forecast", "monitor"])
@pytest.mark.parametrize("options", [None, {}, {"input_dir": "request"}])
def test_runtime_reports_required_paths_before_loading_artifact(
    tmp_path: Path, section: str, options: dict | None
) -> None:
    cfg = {} if options is None else {section: options}
    required = "output_dir" if options else "input_dir"
    workflow = run_forecast if section == "forecast" else run_monitor
    with pytest.raises(ValueError, match=rf"{section}\.{required} must be a non-empty path"):
        workflow(cfg, tmp_path / "config.yaml")
