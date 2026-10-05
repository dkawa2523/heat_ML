"""Array validation stays usable without allocating prediction errors."""

import numpy as np
import pytest

from celltemp.analysis import validate_prediction_arrays


def test_validation_allows_missing_truth_without_calculating_errors(monkeypatch):
    truth = np.array([[np.nan, 21.0], [22.0, np.nan]])
    predicted = np.array([[20.0, 21.0], [22.0, 23.0]])

    def no_error_array(*args, **kwargs):
        pytest.fail("validation must not allocate an error array")

    monkeypatch.setattr(np, "full", no_error_array)
    actual_truth, actual_prediction = validate_prediction_arrays(truth, predicted)

    assert actual_truth is truth
    assert actual_prediction is predicted


@pytest.mark.parametrize("invalid_prediction", [np.nan, np.inf, -np.inf])
def test_validation_rejects_failed_predictions_at_missing_truth(invalid_prediction):
    with pytest.raises(ValueError, match="must all be finite"):
        validate_prediction_arrays(np.array([np.nan]), np.array([invalid_prediction]))


def test_validation_rejects_infinite_truth():
    with pytest.raises(ValueError, match="cannot contain infinity"):
        validate_prediction_arrays(np.array([np.inf]), np.array([20.0]))
