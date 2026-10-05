from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from celltemp.artifact import save_artifact
from celltemp.config import load_yaml, save_yaml
from celltemp.engine import ThermalRCModel
from celltemp.io import load_system_spec
from celltemp.workflows import run_monitor
from external_tools.comsol_chip_cooling.evaluate_monitor import (
    SENSORS,
    _validate_predictions,
    evaluate_monitoring,
)


def _monitor_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.DataFrame({"time": [0.0, 1.0, 2.0]})
    result = source.copy()
    for sensor in SENSORS:
        source[sensor] = [20.0, 21.0, np.nan]
        source[f"truth_{sensor}"] = [20.0, 21.0, 22.0]
        result[f"sensor.{sensor}.measured"] = source[sensor]
        for prefix in ("prior_physical", "predicted_measurement"):
            result[f"sensor.{sensor}.{prefix}"] = [np.nan, 21.0, 22.0]
        for prefix in ("posterior_physical", "reconstructed_measurement"):
            result[f"sensor.{sensor}.{prefix}"] = [20.0, 21.0, 22.0]
        result[f"node.{sensor}.temperature"] = [20.0, 21.0, 22.0]
        result[f"sensor.{sensor}.bias"] = 0.0
        result[f"node.{sensor}.disturbance_w"] = 0.0
        result[f"sensor.{sensor}.innovation"] = [np.nan, 0.0, np.nan]
        result[f"sensor.{sensor}.innovation_std"] = [np.nan, 0.15, np.nan]
    result["nis"] = [np.nan, 0.0, np.nan]
    result["nis_dof"] = [0, 3, 0]
    return source, result


def test_saved_comsol_monitor_allows_deliberately_unavailable_diagnostics() -> None:
    source, result = _monitor_frames()

    _validate_predictions("monitor", source, result)


@pytest.mark.parametrize(
    "column",
    [
        "sensor.chip.posterior_physical",
        "sensor.chip.bias",
        "node.chip.disturbance_w",
        "sensor.chip.innovation",
        "nis",
    ],
)
@pytest.mark.parametrize("failed_value", [np.nan, np.inf])
def test_saved_comsol_monitor_rejects_failed_estimates(column: str, failed_value: float) -> None:
    source, result = _monitor_frames()
    result.loc[1, column] = failed_value

    with pytest.raises(ValueError, match="finite"):
        _validate_predictions("monitor", source, result)


def test_saved_comsol_monitor_rejects_outputs_from_a_different_input() -> None:
    source, result = _monitor_frames()
    result.loc[1, "sensor.chip.measured"] = 999.0

    with pytest.raises(ValueError, match="measurements differ"):
        _validate_predictions("monitor", source, result)


def test_comsol_monitor_rejects_an_old_output_schema(comsol_monitor_output: Path) -> None:
    root = comsol_monitor_output
    path = root / "monitor/run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outdated; rerun monitor"):
        evaluate_monitoring(root)


@pytest.fixture
def comsol_monitor_output(tmp_path: Path) -> Path:
    system = load_system_spec(
        Path(__file__).parents[2] / "external_tools/comsol_chip_cooling/system.yaml"
    )
    save_artifact(tmp_path / "artifact", ThermalRCModel(system))
    source, _ = _monitor_frames()
    for sensor in SENSORS:
        source[f"truth_bias_{sensor}"] = 0.0
    source["chip_power"] = 0.0
    source["coolant_temperature"] = 25.0
    source["truth_hidden_power"] = 0.0
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source.to_csv(inputs / "M01_noise_baseline.csv", index=False)
    cfg = {
        "artifact": "artifact",
        "monitor": {"input_dir": "inputs", "output_dir": "monitor"},
    }
    save_yaml(cfg, tmp_path / "config.yaml")
    run_monitor(cfg, tmp_path / "config.yaml")
    return tmp_path


@pytest.mark.parametrize("changed", ["input", "artifact", "settings"])
def test_comsol_monitor_rejects_stale_provenance(comsol_monitor_output: Path, changed: str) -> None:
    root = comsol_monitor_output
    if changed == "input":
        path = root / "inputs/M01_noise_baseline.csv"
        source = pd.read_csv(path)
        source["chip_power"] = 1000.0
        source.to_csv(path, index=False)
    elif changed == "artifact":
        path = root / "artifact/metadata.json"
        metadata = json.loads(path.read_text())
        metadata["review_marker"] = "another run"
        path.write_text(json.dumps(metadata))
    else:
        path = root / "config.yaml"
        cfg = load_yaml(path)
        cfg["monitor"]["observer"] = {"sensor_std": 0.3}
        save_yaml(cfg, path)

    with pytest.raises(ValueError, match=r"inputs differ|artifact differs|settings differ"):
        evaluate_monitoring(root)


def test_comsol_monitor_accepts_noncompute_config_changes(comsol_monitor_output: Path) -> None:
    root = comsol_monitor_output
    cfg = load_yaml(root / "config.yaml")
    cfg["project"] = {"diagnostics": True}
    cfg["analysis"] = {"output_dir": "another analysis directory"}
    cfg["training"] = {"epochs": 500}
    save_yaml(cfg, root / "config.yaml")

    cases, sensors, _events = evaluate_monitoring(root)

    assert cases["case_id"].tolist() == ["M01_noise_baseline"]
    assert len(sensors) == 3


def test_comsol_monitor_rejects_commands_changed_in_saved_output(
    comsol_monitor_output: Path,
) -> None:
    root = comsol_monitor_output
    path = root / "monitor/cases/M01_noise_baseline.csv"
    saved = pd.read_csv(path)
    saved["control.chip_power.command"] = 1000.0
    saved.to_csv(path, index=False)

    with pytest.raises(ValueError, match="commands differ"):
        evaluate_monitoring(root)


def test_comsol_monitor_rejects_a_saved_gauge_that_differs_from_settings(
    comsol_monitor_output: Path,
) -> None:
    root = comsol_monitor_output
    path = root / "monitor/cases/M01_noise_baseline.csv"
    saved = pd.read_csv(path)
    saved["bias_gauge"] = "reference:fins"
    saved.to_csv(path, index=False)
    with pytest.raises(ValueError, match="bias gauge differs"):
        evaluate_monitoring(root)
