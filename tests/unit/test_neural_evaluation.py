from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.neural_comparison.evaluation import (
    validate_case_metrics,
    validate_evaluation_points,
    validate_matching_splits,
    validate_prediction_points,
    validate_training_inputs,
)


def _case_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rmse_k": 0.1,
                "mae_k": 0.1,
                "max_abs_error_k": 0.2,
                "n_points": 4,
                "n_truth_points": 4,
                "prediction_coverage_fraction": 1.0,
            }
        ]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rmse_k", np.nan),
        ("rmse_k", np.inf),
        ("n_points", 0),
        ("n_points", 1.5),
        ("n_truth_points", 8),
        ("prediction_coverage_fraction", 0.5),
    ],
)
def test_neural_summary_cannot_drop_failed_or_unscored_cases(field: str, value: float) -> None:
    metrics = _case_metrics()
    metrics[field] = metrics[field].astype(float)
    metrics.loc[0, field] = value

    with pytest.raises(ValueError, match=r"case|truth value|coverage"):
        validate_case_metrics(metrics)


def test_neural_summary_accepts_complete_case_metrics_and_missing_truth() -> None:
    validate_case_metrics(_case_metrics())
    validate_prediction_points(pd.DataFrame({"truth": [20.0, np.nan], "predicted": [20.0, 21.0]}))


def test_neural_summary_verifies_waveform_point_coverage() -> None:
    cases = pd.DataFrame({"case_id": ["case_a"], "n_points": [2]})
    points = pd.DataFrame(
        {
            "case_id": ["case_a", "case_a"],
            "time": [0.0, 1.0],
            "sensor": ["chip", "chip"],
            "truth": [20.0, 21.0],
            "predicted": [20.0, 21.0],
        }
    )
    validate_evaluation_points(cases, points, ("case_id",))
    with pytest.raises(ValueError, match="coverage differs"):
        validate_evaluation_points(cases, points.iloc[:1], ("case_id",))
    with pytest.raises(ValueError, match="must be unique"):
        validate_evaluation_points(cases, pd.concat([points, points]), ("case_id",))


@pytest.mark.parametrize("prediction", [np.nan, np.inf, -np.inf])
def test_neural_parity_cannot_drop_failed_predictions(prediction: float) -> None:
    with pytest.raises(ValueError, match="must all be finite"):
        validate_prediction_points(
            pd.DataFrame({"truth": [20.0, np.nan], "predicted": [20.0, prediction]})
        )


def test_neural_boundary_guard_rejects_a_case_rc_already_fitted() -> None:
    expected = {"train": ["case_b"], "val": ["case_c"], "test": ["case_a"]}
    saved = pd.DataFrame(
        {"case_id": ["case_a", "case_b", "case_c"], "split": ["train", "test", "val"]}
    )

    with pytest.raises(ValueError, match="split differs"):
        validate_matching_splits(expected, saved)


def test_neural_boundary_guard_accepts_identical_assignments_in_a_different_order() -> None:
    expected = {"train": ["case_b"], "val": ["case_c"], "test": ["case_a"]}
    saved = pd.DataFrame(
        {"case_id": ["case_a", "case_c", "case_b"], "split": ["test", "val", "train"]}
    )

    validate_matching_splits(expected, saved)


def test_neural_boundary_guard_rejects_duplicate_saved_case_identifiers() -> None:
    saved = pd.DataFrame({"case_id": ["case_a", "case_a"], "split": ["test", "train"]})

    with pytest.raises(ValueError, match="unique case_id"):
        validate_matching_splits({"train": ["case_a"]}, saved)


def test_neural_boundary_guard_rejects_changed_input_bytes_and_legacy_artifacts() -> None:
    expected = {"case_a": ("train", "new_hash")}
    with pytest.raises(ValueError, match="input bytes or splits changed"):
        validate_training_inputs(
            expected, [{"case_id": "case_a", "split": "train", "sha256": "old_hash"}]
        )
    with pytest.raises(ValueError, match="lacks hashed training inputs"):
        validate_training_inputs(expected, None)


def test_neural_boundary_guard_accepts_identical_input_bytes_and_splits() -> None:
    validate_training_inputs(
        {"case_a": ("train", "hash_a")},
        [{"case_id": "case_a", "split": "train", "sha256": "hash_a"}],
    )
