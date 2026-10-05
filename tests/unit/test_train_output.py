"""Regression tests for held-out evidence written by the training workflow."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from celltemp.domain import Trajectory
from celltemp.io import load_trajectories
from celltemp.learning import TrainingConfig, TrainingResult
from celltemp.workflows.train_output import (
    _build_training_evidence,
    _model_comparison,
    _write_test_prediction_figures,
    write_training_outputs,
)

from ._engine_cases import conduction_model


def test_model_comparison_excludes_initial_and_unobserved_values() -> None:
    model = conduction_model()
    training_case = Trajectory(
        case_id="train",
        time=np.array([0.0, 1.0]),
        temperature=np.array([[10.0, 20.0], [11.0, 19.0]]),
        commands=np.empty((1, 0)),
        sensor_names=model.spec.sensor_names,
        control_names=model.spec.control_names,
    )
    held_out_case = Trajectory(
        case_id="held_out",
        time=np.array([0.0, 1.0]),
        temperature=np.array([[10.0, 20.0], [11.0, 999.0]]),
        commands=np.empty((1, 0)),
        sensor_names=model.spec.sensor_names,
        control_names=model.spec.control_names,
        observation_mask=np.array([[True, True], [True, False]]),
    )

    evidence = _build_training_evidence(
        model,
        {"train": [training_case], "val": [held_out_case], "test": []},
        initial_temperature_prior_std=1.0,
        keep_predictions=True,
    )

    comparison = _model_comparison(model, evidence.predictions)
    assert set(comparison["n_points"]) == {1}
    fitted_rmse = comparison.loc[comparison["model"] == "fitted_rc", "rmse_k"].item()
    causal_rmse = evidence.metrics.loc[evidence.metrics["split"] == "val", "causal_rmse"].item()
    assert fitted_rmse == causal_rmse
    assert comparison.loc[comparison["model"] == "fitted_rc", "worst_sensor"].item() == "a"


def test_training_diagnostic_uses_the_project_temperature_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = conduction_model()
    case = Trajectory(
        case_id="kelvin",
        time=np.array([0.0, 1.0]),
        temperature=np.array([[300.0, 310.0], [301.0, 309.0]]),
        commands=np.empty((1, 0)),
        sensor_names=model.spec.sensor_names,
        control_names=model.spec.control_names,
    )
    evidence = _build_training_evidence(
        model,
        {"train": [case], "test": [case]},
        initial_temperature_prior_std=1.0,
        keep_predictions=True,
        temperature_unit="K",
    )
    recorded = {}

    def capture(cases, target, **options):
        recorded.update(options)
        np.testing.assert_array_equal(cases[0].truth, case.temperature)

    monkeypatch.setattr("celltemp.workflows.prediction_figures.write_prediction_figures", capture)
    _write_test_prediction_figures(evidence, tmp_path)
    assert recorded["temperature_unit"] == "K"


def test_training_artifact_records_the_project_temperature_unit(tmp_path: Path) -> None:
    model = conduction_model()
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    pd.DataFrame({"time": [0.0, 1.0], "a": [300.0, 301.0], "b": [310.0, 309.0]}).to_csv(
        input_dir / "kelvin.csv", index=False
    )
    cases = load_trajectories(
        {"directory": "data", "sensor_cols": model.spec.sensor_names}, tmp_path
    )
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    evidence = write_training_outputs(
        output_dir,
        cfg={"project": {"temperature_unit": "K"}},
        config_path=tmp_path / "config.yaml",
        run_name="kelvin",
        seed=42,
        control_convention="left",
        model=model,
        all_trajectories=cases,
        trajectories={"train": cases},
        training=TrainingConfig(epochs=1),
        result=TrainingResult(best_epoch=1, best_causal_validation_rmse=1.0, history=()),
    )
    metadata = json.loads((output_dir / "artifact" / "metadata.json").read_text())
    assert metadata["temperature_unit"] == "K"
    assert evidence.temperature_unit == "K"
