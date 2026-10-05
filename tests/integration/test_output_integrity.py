"""Public workflows preserve measurements and separate cases from aggregate tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from celltemp.artifact import save_artifact
from celltemp.config import load_yaml, save_yaml
from celltemp.domain import ActuatorSpec, PositivePartLawSpec, SourceSpec, ThermalSystemSpec
from celltemp.engine import ThermalRCModel
from celltemp.workflows import run_analysis, run_forecast, run_monitor, run_train

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("overwrite", [False, True])
def test_default_analysis_output_survives_training(cae_project: Path, overwrite: bool) -> None:
    path = cae_project / "config.yaml"
    cfg = load_yaml(path)
    cfg["project"].update(overwrite_run=overwrite, diagnostics=False)
    cfg["training"]["epochs"] = 1
    cfg["analysis"] = {"make_plots": False}
    save_yaml(cfg, path)

    analysis = run_analysis(cfg, path)
    before = {item.name: item.read_bytes() for item in analysis.iterdir() if item.is_file()}
    trained = run_train(cfg, path)

    assert analysis == trained.with_name(f"{trained.name}_analysis")
    assert before == {item.name: item.read_bytes() for item in analysis.iterdir() if item.is_file()}
    assert (trained / "artifact" / "model.pt").is_file()
    assert not (trained / "diagnostics").exists()


def test_failed_training_diagnostics_preserve_saved_model(
    cae_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from celltemp.workflows import train

    path = cae_project / "config.yaml"
    cfg = load_yaml(path)
    cfg["training"]["epochs"] = 1

    def fail(*_args: object) -> None:
        raise RuntimeError("diagnostic plot failed")

    monkeypatch.setattr(train, "write_training_diagnostics", fail)
    with pytest.raises(RuntimeError, match="diagnostic plot failed"):
        run_train(cfg, path)

    target = cae_project / "outputs/runs/test_run"
    assert (target / "artifact/model.pt").is_file()
    assert (target / "metrics_summary.json").is_file()
    assert not (target / "diagnostics").exists()


def _runtime_project(root: Path, case_id: str) -> tuple[dict, Path]:
    model = ThermalRCModel(
        ThermalSystemSpec(
            node_names=("body",),
            heat_capacity=(1.0,),
            edges=(),
            sensor_names=("tc",),
            sensor_nodes=("body",),
            actuators=(ActuatorSpec("power", tau=0.0, learnable=False),),
            sources=(SourceSpec("heater", (1.0,), PositivePartLawSpec("power", 1.0)),),
        )
    )
    save_artifact(root / "artifact", model)
    inputs = root / "data"
    inputs.mkdir()
    pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0],
            "tc": [20.0, np.nan, np.nan, np.nan],
            "power": [0.0, 10.0, 0.0, 0.0],
        }
    ).to_csv(inputs / f"{case_id}.csv", index=False)
    cfg = {
        "artifact": "artifact",
        "project": {"diagnostics": True},
        "forecast": {"input_dir": "data", "output_dir": "forecast", "overwrite": True},
        "monitor": {"input_dir": "data", "output_dir": "monitor", "overwrite": True},
    }
    config_path = root / "config.yaml"
    save_yaml(cfg, config_path)
    return cfg, config_path


@pytest.mark.parametrize("case_id", ["forecast_summary", "energy_balance", "forecast_coverage"])
def test_forecast_case_names_cannot_overwrite_aggregate_tables(
    tmp_path: Path, case_id: str
) -> None:
    cfg, config_path = _runtime_project(tmp_path, case_id)

    target = run_forecast(cfg, config_path)

    summary = pd.read_csv(target / "forecast_summary.csv")
    assert summary.loc[0, "output"] == f"cases/{case_id}.csv"
    case = pd.read_csv(target / str(summary.loc[0, "output"]))
    np.testing.assert_allclose(case["sensor.tc.temperature"], [20.0, 20.0, 30.0, 30.0], atol=1e-5)
    np.testing.assert_allclose(case["control.power.effective"], case["control.power.command"])
    energy = pd.read_csv(target / "diagnostics" / "energy_balance.csv")
    np.testing.assert_allclose(energy["source.heater.heat_w"], [0.0, 10.0, 0.0, 0.0])


def test_monitor_case_name_cannot_overwrite_summary(tmp_path: Path) -> None:
    cfg, config_path = _runtime_project(tmp_path, "monitor_summary")
    source = tmp_path / "data" / "monitor_summary.csv"
    frame = pd.read_csv(source)
    frame["tc"] = [20.0, 20.0, 30.0, 30.0]
    frame.to_csv(source, index=False)

    target = run_monitor(cfg, config_path)

    summary = pd.read_csv(target / "monitor_summary.csv")
    assert summary.loc[0, "output"] == "cases/monitor_summary.csv"
    case = pd.read_csv(target / str(summary.loc[0, "output"]))
    assert "sensor.tc.posterior_physical" in case
    np.testing.assert_allclose(case["control.power.effective"], [0.0, 10.0, 0.0, 0.0])


def test_default_forecast_saves_core_results_without_diagnostics(tmp_path: Path) -> None:
    cfg, path = _runtime_project(tmp_path, "case")
    cfg.pop("project")
    target = run_forecast(cfg, path)
    assert (target / "cases/case.csv").is_file()
    assert (target / "forecast_coverage.csv").is_file()
    assert not (target / "diagnostics").exists()


def test_failed_forecast_diagnostics_preserve_saved_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from celltemp.workflows import forecast_diagnostics

    cfg, path = _runtime_project(tmp_path, "case")

    def fail(*_args: object) -> None:
        raise RuntimeError("diagnostic plot failed")

    monkeypatch.setattr(forecast_diagnostics, "_plot_forecast_case", fail)
    with pytest.raises(RuntimeError, match="diagnostic plot failed"):
        run_forecast(cfg, path)

    target = tmp_path / "forecast"
    assert (target / "cases/case.csv").is_file()
    assert (target / "run_manifest.json").is_file()
    assert not (target / "diagnostics").exists()


@pytest.mark.parametrize("workflow", ["forecast", "monitor"])
def test_runtime_output_cannot_replace_input_or_artifact(
    tmp_path: Path,
    workflow: str,
) -> None:
    cfg, config_path = _runtime_project(tmp_path, "case")
    runner = run_forecast if workflow == "forecast" else run_monitor
    for protected in ("data", "artifact"):
        before = {p.name: p.read_bytes() for p in (tmp_path / protected).iterdir() if p.is_file()}
        cfg[workflow]["output_dir"] = protected
        with pytest.raises(ValueError, match="overlaps an input or artifact"):
            runner(cfg, config_path)
        after = {p.name: p.read_bytes() for p in (tmp_path / protected).iterdir() if p.is_file()}
        assert before == after


@pytest.mark.parametrize("workflow", ["forecast", "monitor", "analysis"])
@pytest.mark.parametrize("selection", ["symlink", "parent_glob"])
def test_workflows_protect_actual_csv_sources_outside_input_directory(
    tmp_path: Path, workflow: str, selection: str
) -> None:
    cfg, config_path = _runtime_project(tmp_path, "case")
    cfg["system"] = "artifact/system.yaml"
    cfg["analysis"] = {"input_dir": "data", "output_dir": "replacement", "overwrite": True}
    cfg[workflow]["output_dir"] = "replacement"
    target = tmp_path / "replacement"
    target.mkdir()
    source = target / "case.csv"
    link = tmp_path / "data" / "case.csv"
    link.replace(source)
    if selection == "symlink":
        try:
            link.symlink_to(source)
        except OSError:
            pytest.skip("symlink creation is unavailable on this host")
    else:
        cfg[workflow]["pattern"] = "../replacement/*.csv"
    before = source.read_bytes()
    runner = {"forecast": run_forecast, "monitor": run_monitor, "analysis": run_analysis}[workflow]

    with pytest.raises(ValueError, match="overlaps an input or artifact"):
        runner(cfg, config_path)

    assert source.read_bytes() == before


def test_train_output_cannot_replace_explicit_split_table(cae_project: Path) -> None:
    config_path = cae_project / "config.yaml"
    cfg = load_yaml(config_path)
    target = cae_project / "outputs" / "runs" / "test_run"
    target.mkdir(parents=True)
    table = target / "manual_assignments.csv"
    pd.DataFrame({"case_id": ["example"], "split": ["train"]}).to_csv(table, index=False)
    before = table.read_bytes()
    cfg["split"] = {"method": "explicit", "table": str(table)}

    with pytest.raises(ValueError, match="overlaps an input or artifact"):
        run_train(cfg, config_path)

    assert table.read_bytes() == before


def test_train_output_cannot_replace_input_dataset(cae_project: Path) -> None:
    config_path = cae_project / "config.yaml"
    cfg = load_yaml(config_path)
    target = cae_project / "outputs" / "runs" / "test_run"
    target.mkdir(parents=True)
    data = target / "input"
    (cae_project / "data" / "raw").replace(data)
    cfg["data"]["directory"] = str(data)
    before = {path.name: path.read_bytes() for path in data.glob("*.csv")}

    with pytest.raises(ValueError, match="overlaps an input or artifact"):
        run_train(cfg, config_path)

    assert {path.name: path.read_bytes() for path in data.glob("*.csv")} == before


def test_analysis_retains_insufficient_sensor_channels(tmp_path: Path) -> None:
    save_yaml(
        {
            "version": 3,
            "nodes": [{"name": "body", "heat_capacity": 1.0}],
            "sensors": [
                {"name": name, "node": "body"} for name in ("working", "single", "missing")
            ],
        },
        tmp_path / "system.yaml",
    )
    (tmp_path / "data").mkdir()
    pd.DataFrame(
        {
            "time": [0.0, 1.0, 3.0],
            "working": [20.0, 22.0, 24.0],
            "single": [np.nan, 30.0, np.nan],
            "missing": [np.nan, np.nan, np.nan],
        }
    ).to_csv(tmp_path / "data" / "sparse.csv", index=False)
    cfg = {
        "system": "system.yaml",
        "data": {"directory": "data"},
        "analysis": {"output_dir": "analysis", "make_plots": True},
    }
    config_path = tmp_path / "config.yaml"
    save_yaml(cfg, config_path)

    target = run_analysis(cfg, config_path)

    rows = pd.read_csv(target / "sensor_metrics.csv").set_index("sensor")
    assert rows.loc["missing", "n_observed_points"] == 0
    assert rows.loc["single", "response_status"] == "insufficient_observations"
    summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    assert summary["response_availability"]["insufficient_sensor_responses"] == 2
    assert summary["highest_temperature"]["temperature"] == 30.0
    assert (target / "figures" / "sparse.png").stat().st_size > 0
