"""The public workflow: train one artifact, then forecast and monitor with it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from celltemp.artifact import save_artifact
from celltemp.cli import main
from celltemp.config import save_yaml
from celltemp.domain import (
    ActuatorSpec,
    BoundarySpec,
    ConstantLawSpec,
    EdgeSpec,
    PositivePartLawSpec,
    ReservoirTemperatureSpec,
    SourceSpec,
    ThermalSystemSpec,
)
from celltemp.engine import ThermalRCModel
from tests.conftest import SENSORS, write_forecast_request, write_monitor_log

pytestmark = pytest.mark.integration


def _run_cli(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["celltemp", *arguments])
    main()


def test_end_to_end_workflow(cae_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = cae_project / "config.yaml"
    _run_cli(monkeypatch, "train", "--config", str(config_path))
    run_dir = cae_project / "outputs" / "runs" / "test_run"

    assert (run_dir / "artifact" / "model.pt").exists()
    assert (run_dir / "artifact" / "system.yaml").exists()
    summary = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"train", "val", "test"}
    assert np.isfinite(summary["test"]["mean_case_conditional_rmse"])
    assert np.isfinite(summary["test"]["mean_case_causal_rmse"])
    assert len(pd.read_csv(run_dir / "split.csv")) == 8
    metadata = json.loads((run_dir / "artifact" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["training"]["model_selection_metric"] == "causal_rmse"
    assert "train_temporal_ranges" in metadata

    write_forecast_request(cae_project)
    write_monitor_log(cae_project, noise=0.05)
    _run_cli(monkeypatch, "forecast", "--config", str(config_path))
    _run_cli(monkeypatch, "monitor", "--config", str(config_path))
    # A repeated run takes the explicit overwrite path without retaining stale files.
    _run_cli(monkeypatch, "forecast", "--config", str(config_path))
    forecast_dir = cae_project / "outputs" / "forecast"
    monitor_dir = cae_project / "outputs" / "monitor"

    forecast = pd.read_csv(forecast_dir / "const_case.csv")
    assert len(forecast) == 11
    assert np.isfinite(forecast[[f"temperature_{name}" for name in SENSORS]]).all().all()
    assert np.isfinite(forecast[[f"temperature_std_{name}" for name in SENSORS]]).all().all()
    assert (forecast[[f"temperature_std_{name}" for name in SENSORS]] >= 0.0).all().all()
    forecast_manifest = json.loads((forecast_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert forecast_manifest["settings"]["observer"]["initial_temperature_std"] == 100.0
    assert forecast_manifest["settings"]["uncertainty"]["scope"] == "latent_state_and_process_only"
    assert forecast_manifest["input"]["files"][0]["sha256"]
    coverage = pd.read_csv(forecast_dir / "forecast_coverage.csv")
    assert coverage["within_training_range"].all()
    monitored = pd.read_csv(monitor_dir / "monitor_case.csv")
    assert np.isfinite(monitored[[f"posterior_physical_{name}" for name in SENSORS]]).all().all()
    assert set(monitored["bias_gauge"]) == {f"reference:{SENSORS[-1]}"}
    assert np.equal(monitored[f"sensor_bias_{SENSORS[-1]}"], 0.0).all()
    summary = pd.read_csv(monitor_dir / "monitor_summary.csv")
    assert set(summary["bias_gauge"]) == {f"reference:{SENSORS[-1]}"}
    assert np.isfinite(summary["innovation_rmse"]).all()
    assert np.isfinite(summary["mean_nis_per_dof"]).all()
    monitor_manifest = json.loads((monitor_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert monitor_manifest["workflow"] == "monitor"
    assert monitor_manifest["settings"]["observer"]["bias_reference"] == SENSORS[-1]


def test_forecast_uses_history_without_accepting_measurements_after_the_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = ThermalSystemSpec(
        node_names=("core", "shell"),
        heat_capacity=(3.0, 1.0),
        edges=(EdgeSpec("core", "shell", ConstantLawSpec(0.4, learnable=False)),),
        actuators=(ActuatorSpec("heater", tau=0.0, learnable=False),),
        sources=(
            SourceSpec(
                "core_heating",
                (1.0, 0.0),
                PositivePartLawSpec("heater", 0.2, learnable=False),
            ),
        ),
        boundaries=(
            BoundarySpec(
                "ambient",
                (0.0, 1.0),
                ReservoirTemperatureSpec(25.0),
                ConstantLawSpec(0.1, learnable=False),
            ),
        ),
        sensor_names=("surface_tc",),
        sensor_nodes=("shell",),
    )
    save_artifact(tmp_path / "artifact", ThermalRCModel(spec))

    request_dir = tmp_path / "data" / "forecast"
    request_dir.mkdir(parents=True)
    request = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "surface_tc": [25.0, np.nan, np.nan],
            "heater": [4.0, 4.0, 4.0],
        }
    )
    request_path = request_dir / "hidden_case.csv"
    request.to_csv(request_path, index=False)
    config_path = tmp_path / "runtime.yaml"
    save_yaml(
        {
            "artifact": "artifact",
            "forecast": {
                "input_dir": "data/forecast",
                "output_dir": "outputs/forecast",
                "overwrite": True,
            },
        },
        config_path,
    )

    _run_cli(monkeypatch, "forecast", "--config", str(config_path))
    output_path = tmp_path / "outputs" / "forecast" / "hidden_case.csv"
    output = pd.read_csv(output_path)
    assert "temperature_surface_tc" in output
    assert "state_core" in output
    assert "state_shell" in output
    assert np.isfinite(output[["state_core", "state_shell"]]).all().all()

    request.loc[1, "surface_tc"] = 26.0
    request.to_csv(request_path, index=False)
    _run_cli(monkeypatch, "forecast", "--config", str(config_path))
    history_initialized = pd.read_csv(output_path)
    assert history_initialized["time"].tolist() == [1.0, 2.0]
    summary = pd.read_csv(tmp_path / "outputs" / "forecast" / "forecast_summary.csv")
    assert summary.loc[0, "history_rows"] == 2
    assert summary.loc[0, "forecast_start_time"] == 1.0

    previous_output = output_path.read_bytes()
    request.loc[1, "surface_tc"] = np.nan
    request.loc[2, "surface_tc"] = 26.0
    request.to_csv(request_path, index=False)
    with pytest.raises(ValueError, match="contiguous history prefix"):
        _run_cli(monkeypatch, "forecast", "--config", str(config_path))
    assert output_path.read_bytes() == previous_output


@pytest.mark.parametrize("command", ["train", "forecast", "monitor"])
def test_cli_exposes_only_product_workflows(command: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["celltemp", command, "--help"])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0


def test_workflow_rejects_a_misspelled_section_option(
    cae_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match=r"unknown engine options.*integrtor"):
        _run_cli(
            monkeypatch,
            "train",
            "--config",
            str(cae_project / "config.yaml"),
            "engine.integrtor=exact",
        )
