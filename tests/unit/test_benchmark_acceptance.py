"""Regression guards against favorable scores obtained by dropping failed forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.topcell.scripts.evaluate_benchmark import acceptance_checks
from celltemp.analysis import prediction_error_metrics


def _passing_forecasts() -> pd.DataFrame:
    definitions = (
        ("F01_level_interpolation", "interpolation", 0.1),
        ("F09_sparse_initial_observation", "initialization", 1.0),
        ("F12_history_initialized_sparse", "initialization", 0.1),
        ("F11_nonlinear_heat_loss", "model_gap", 2.0),
    )
    return pd.DataFrame(
        [
            {
                "case_id": case_id,
                "group": group,
                "learned_rmse": error,
                "prior_rmse": 3.0,
                "persistence_rmse": 4.0,
                "predictions_finite": True,
                "n_evaluation_points": 20,
                "expected_evaluation_points": 20,
                "prediction_coverage_fraction": 1.0,
            }
            for case_id, group, error in definitions
        ]
    )


def _passing_monitors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group": group,
                "innovation_rmse": 0.1,
                "sensor_bias_rmse_after_burn_in": 0.1,
                "detection_delay": 0.0,
                "first_alert_sensor": "CP",
                "posterior_finite": True,
                "posterior_physical_rmse": 0.1,
                "event_disturbance_rmse_w": 0.1,
            }
            for group in (
                "baseline",
                "bias_tracking",
                "fault_detection",
                "missing_data",
                "disturbance_detection",
            )
        ]
    )


def test_topcell_complete_predictions_pass_the_existing_acceptance_contract() -> None:
    assert all(acceptance_checks(_passing_forecasts(), _passing_monitors()).values())


@pytest.mark.parametrize(
    ("field", "failed_value"),
    [
        ("predictions_finite", False),
        ("n_evaluation_points", 4),
        ("n_evaluation_points", 0),
        ("prediction_coverage_fraction", 0.2),
        ("prediction_coverage_fraction", np.nan),
    ],
)
def test_topcell_rejects_incomplete_predictions_even_with_zero_rmse(
    field: str, failed_value: float | int | bool
) -> None:
    forecasts = _passing_forecasts()
    forecasts.loc[0, "learned_rmse"] = 0.0
    forecasts.loc[0, field] = failed_value

    checks = acceptance_checks(forecasts, _passing_monitors())

    assert not checks["forecast_core_all_finite"]
    assert not all(checks.values())


def test_forecast_origin_cannot_hide_nan_in_every_future_row() -> None:
    truth = np.arange(20, dtype=float).reshape(5, 4)
    predicted = truth.copy()
    predicted[1:] = np.nan

    with pytest.raises(ValueError, match="must all be finite"):
        prediction_error_metrics(truth, predicted)
