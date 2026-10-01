"""Regression tests for held-out evidence written by the training workflow."""

import numpy as np

from celltemp.domain import Trajectory
from celltemp.workflows.train_output import _build_training_evidence

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
    )

    comparison = evidence.model_comparison
    assert set(comparison["n_points"]) == {1}
    fitted_rmse = comparison.loc[comparison["model"] == "fitted_rc", "rmse_k"].item()
    causal_rmse = evidence.metrics.loc[evidence.metrics["split"] == "val", "causal_rmse"].item()
    assert fitted_rmse == causal_rmse
    assert comparison.loc[comparison["model"] == "fitted_rc", "worst_sensor"].item() == "a"
