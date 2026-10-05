"""TopCell scores the workflow's published predictions and provenance."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from benchmarks.topcell.scripts import evaluate_benchmark as evaluator
from celltemp.artifact import load_artifact, save_artifact
from celltemp.config import save_yaml
from celltemp.inference import build_observer, resolve_observer_settings
from celltemp.io import load_trajectories
from celltemp.workflows.common import write_runtime_manifest

from ._engine_cases import conduction_model


def _prediction_frame(source, model, truth, workflow):
    saved = pd.DataFrame({"time": source["time"]})
    for index, sensor in enumerate(model.spec.sensor_names):
        if workflow == "forecast":
            saved[f"sensor.{sensor}.temperature"] = truth[:, index]
        else:
            saved[f"sensor.{sensor}.measured"] = source[sensor]
            for prefix in ("posterior_physical", "reconstructed_measurement"):
                saved[f"sensor.{sensor}.{prefix}"] = truth[:, index] + 0.1
            for prefix in ("prior_physical", "predicted_measurement"):
                saved[f"sensor.{sensor}.{prefix}"] = truth[:, index]
                saved.loc[0, f"sensor.{sensor}.{prefix}"] = np.nan
            saved[f"sensor.{sensor}.bias"] = 0.0
            saved[f"sensor.{sensor}.innovation"] = [np.nan, 0.2, 0.1 if index == 0 else np.nan]
            saved[f"sensor.{sensor}.innovation_std"] = [
                np.nan,
                0.15,
                0.15 if index == 0 else np.nan,
            ]
    if workflow == "monitor":
        for index, node in enumerate(model.spec.node_names):
            saved[f"node.{node}.temperature"] = truth[:, index] + 0.1
            saved[f"node.{node}.disturbance_w"] = 0.0
        saved["nis"] = [np.nan, 2.0, 1.0]
        saved["nis_dof"] = [0, 2, 1]
        saved["bias_gauge"] = "zero_mean"
    return saved


@pytest.fixture
def saved_outputs(tmp_path):
    model = conduction_model()
    save_artifact(tmp_path / "artifact", model)
    artifact = load_artifact(tmp_path / "artifact")
    config = {
        "artifact": "artifact",
        "forecast": {"input_dir": "forecast_inputs", "output_dir": "forecast_output"},
        "monitor": {"input_dir": "monitor_inputs", "output_dir": "monitor_output"},
    }
    config_path = tmp_path / "config.yaml"
    save_yaml(config, config_path)
    truth = np.array([[10.0, 20.0], [11.0, 19.0], [12.0, 18.0]])
    for workflow in ("forecast", "monitor"):
        case_id = f"{workflow}_case"
        values = config[workflow]
        input_dir = tmp_path / values["input_dir"]
        output_dir = tmp_path / values["output_dir"]
        input_dir.mkdir()
        (output_dir / "cases").mkdir(parents=True)
        source = pd.DataFrame({"time": [0.0, 30.0, 60.0]})
        for index, sensor in enumerate(model.spec.sensor_names):
            source[sensor] = truth[:, index]
            source[f"truth_{sensor}"] = truth[:, index]
            source[f"truth_bias_{sensor}"] = 0.0
        source["benchmark_group"] = "interpolation" if workflow == "forecast" else "baseline"
        source["benchmark_purpose"] = "published output regression"
        if workflow == "forecast":
            source.loc[1:, list(model.spec.sensor_names)] = np.nan
        else:
            source.loc[2, model.spec.sensor_names[1]] = np.nan
            for node in model.spec.node_names:
                source[f"truth_disturbance_{node}_w"] = 0.0
        source.to_csv(input_dir / f"{case_id}.csv", index=False)
        trajectories = load_trajectories(
            evaluator._trajectory_config(
                str(input_dir), model.spec.sensor_names, model.spec.control_names
            ),
            tmp_path,
        )
        settings = resolve_observer_settings(workflow)
        observer = build_observer(artifact.model, settings)
        write_runtime_manifest(
            output_dir / "run_manifest.json",
            workflow=workflow,
            config_path=config_path,
            root=tmp_path,
            artifact=artifact,
            values=values,
            trajectories=trajectories,
            observer_settings=settings,
            disturbance_basis=observer.disturbance_basis,
        )
        saved = _prediction_frame(source, model, truth, workflow)
        saved.to_csv(output_dir / "cases" / f"{case_id}.csv", index=False)
    return artifact, model, config, config_path


def _case_path(saved_outputs, workflow):
    _, _, config, path = saved_outputs
    return path.parent / config[workflow]["output_dir"] / "cases" / f"{workflow}_case.csv"


def _evaluate_saved(saved_outputs, workflow):
    artifact, prior, config, path = saved_outputs
    if workflow == "forecast":
        return evaluator.evaluate_forecasts(artifact, prior, config[workflow], path)
    return evaluator.evaluate_monitors(artifact, config[workflow], path)


def test_forecast_scores_saved_values_and_only_runs_the_prior(saved_outputs, monkeypatch):
    artifact, prior, config, path = saved_outputs
    real_forecast = evaluator.forecast
    calls = []

    def prior_only(model, trajectory, **kwargs):
        assert model is prior
        calls.append(trajectory.case_id)
        return real_forecast(model, trajectory, **kwargs)

    monkeypatch.setattr(evaluator, "forecast", prior_only)
    saved_path = _case_path(saved_outputs, "forecast")
    frame = pd.read_csv(saved_path)
    temperature = frame["sensor.a.temperature"].to_numpy(dtype=float, copy=True)
    temperature[1] += 6.0
    frame["sensor.a.temperature"] = temperature
    frame.to_csv(saved_path, index=False)

    cases, _, _ = evaluator.evaluate_forecasts(artifact, prior, config["forecast"], path)

    assert calls == ["forecast_case"]
    assert cases["learned_rmse"].iat[0] == pytest.approx(np.sqrt(6.0))
    assert cases["n_evaluation_points"].iat[0] == 6


def test_monitor_scores_saved_values_without_running_inference(saved_outputs, monkeypatch):
    artifact, _, config, path = saved_outputs

    def no_rollout(*args, **kwargs):
        pytest.fail("monitor evaluation must use saved output")

    monkeypatch.setattr(artifact.model, "step", no_rollout)
    saved_path = _case_path(saved_outputs, "monitor")
    frame = pd.read_csv(saved_path)
    temperature = frame["sensor.a.posterior_physical"].to_numpy(dtype=float, copy=True)
    temperature[1] += 6.0
    frame["sensor.a.posterior_physical"] = temperature
    frame.to_csv(saved_path, index=False)

    cases = evaluator.evaluate_monitors(artifact, config["monitor"], path)

    assert cases["posterior_physical_rmse"].iat[0] == pytest.approx(
        np.sqrt((6.1**2 + 5 * 0.1**2) / 6)
    )
    assert cases["posterior_finite"].iat[0]


@pytest.mark.parametrize("workflow", ["forecast", "monitor"])
@pytest.mark.parametrize("field", ["artifact", "input", "observer"])
def test_saved_output_requires_matching_provenance(saved_outputs, workflow, field):
    _, _, config, path = saved_outputs
    manifest_path = path.parent / config[workflow]["output_dir"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if field == "artifact":
        manifest["artifact"]["metadata_sha256"] = "stale"
    elif field == "input":
        manifest["input"]["files"][0]["sha256"] = "stale"
    else:
        manifest["settings"]["observer"]["sensor_std"] = 100.0
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=r"differs? from the benchmark"):
        _evaluate_saved(saved_outputs, workflow)


@pytest.mark.parametrize("workflow", ["forecast", "monitor"])
def test_nonprediction_config_changes_keep_saved_output_scorable(saved_outputs, workflow):
    _, _, config, path = saved_outputs
    config["project"] = {"diagnostics": False}
    config["analysis"] = {"output_dir": "new-analysis-output"}
    config["training"] = {"epochs": 999}
    save_yaml(config, path)

    result = _evaluate_saved(saved_outputs, workflow)

    if workflow == "forecast":
        assert isinstance(result, tuple)
        assert result[0]["learned_rmse"].iat[0] == pytest.approx(0.0)
    else:
        assert isinstance(result, pd.DataFrame)
        assert result["posterior_physical_rmse"].iat[0] == pytest.approx(0.1)


@pytest.mark.parametrize("workflow", ["forecast", "monitor"])
@pytest.mark.parametrize("failure", ["time", "prediction"])
def test_saved_output_rejects_misaligned_or_failed_predictions(saved_outputs, workflow, failure):
    saved_path = _case_path(saved_outputs, workflow)
    frame = pd.read_csv(saved_path)
    if failure == "time":
        sample_time = frame["time"].to_numpy(dtype=float, copy=True)
        sample_time[1] += 1.0
        frame["time"] = sample_time
    else:
        column = "sensor.a.temperature" if workflow == "forecast" else "sensor.a.posterior_physical"
        frame.loc[1, column] = np.nan
    frame.to_csv(saved_path, index=False)

    with pytest.raises(ValueError, match=r"times differ|must.*finite"):
        _evaluate_saved(saved_outputs, workflow)


@pytest.mark.parametrize("column", ["sensor.a.measured", "sensor.a.innovation", "nis", "nis_dof"])
def test_saved_monitor_rejects_invalid_observed_diagnostics(saved_outputs, column):
    artifact, _, config, path = saved_outputs
    saved_path = _case_path(saved_outputs, "monitor")
    frame = pd.read_csv(saved_path)
    frame.loc[1, column] = np.nan
    frame.to_csv(saved_path, index=False)

    with pytest.raises(ValueError, match=r"measurements differ|diagnostics must be finite"):
        evaluator.evaluate_monitors(artifact, config["monitor"], path)
