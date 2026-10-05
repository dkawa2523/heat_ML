from __future__ import annotations

import numpy as np
import pytest

from celltemp.workflows.prediction_figures import (
    PredictionCase,
    coefficient_of_determination,
    write_prediction_figures,
)


def test_prediction_figures_use_held_out_rows_and_select_worst_case(tmp_path) -> None:
    truth = np.array([[20.0, 21.0], [21.0, 22.0], [22.0, 23.0]])
    cases = [
        PredictionCase("accurate", np.array([0.0, 1.0, 2.0]), truth, truth.copy()),
        PredictionCase(
            "worst",
            np.array([0.0, 1.0, 2.0]),
            truth,
            truth + np.array([[0.0, 0.0], [0.2, -0.2], [0.3, -0.3]]),
        ),
    ]

    result = write_prediction_figures(
        cases,
        tmp_path,
        sensor_names=("chip", "sink"),
        title="Test prediction",
    )

    evaluated_truth = np.concatenate([case.truth[1:].ravel() for case in cases])
    evaluated_prediction = np.concatenate([case.predicted[1:].ravel() for case in cases])
    assert result.representative_case == "worst"
    assert result.n_points == 8
    assert result.r_squared == pytest.approx(
        coefficient_of_determination(evaluated_truth, evaluated_prediction)
    )
    assert result.timeseries_path.stat().st_size > 0
    assert result.parity_path.stat().st_size > 0


@pytest.mark.parametrize("failed_prediction", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("missing_truth", [False, True])
def test_r_squared_rejects_failed_predictions_even_at_missing_truth(
    failed_prediction: float, missing_truth: bool
) -> None:
    truth = np.array([20.0, np.nan if missing_truth else 21.0, 22.0])
    predicted = np.array([20.0, failed_prediction, 22.0])

    with pytest.raises(ValueError, match="predicted temperatures must all be finite"):
        coefficient_of_determination(truth, predicted)


def test_r_squared_scores_missing_truth_without_rejecting_finite_predictions() -> None:
    truth = np.array([20.0, np.nan, 22.0])
    predicted = np.array([20.0, 1000.0, 22.0])

    assert coefficient_of_determination(truth, predicted) == pytest.approx(1.0)


@pytest.mark.parametrize("failed_prediction", [np.nan, np.inf, -np.inf])
def test_prediction_case_rejects_failed_predictions_at_missing_truth(
    failed_prediction: float,
) -> None:
    truth = np.array([[20.0], [np.nan], [22.0]])
    predicted = np.array([[20.0], [failed_prediction], [22.0]])

    with pytest.raises(ValueError, match="predicted temperatures must all be finite"):
        PredictionCase("failed", np.array([0.0, 1.0, 2.0]), truth, predicted)


@pytest.mark.parametrize("failed_prediction", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("failed_row", [0, 1])
def test_writer_checks_mutated_predictions_before_creating_any_figures(
    tmp_path, failed_prediction: float, failed_row: int
) -> None:
    truth = np.array([[np.nan], [21.0], [22.0]])
    predicted = np.array([[20.0], [21.0], [22.0]])
    case = PredictionCase("mutated", np.array([0.0, 1.0, 2.0]), truth, predicted)
    predicted[failed_row, 0] = failed_prediction
    target = tmp_path / "figures"

    with pytest.raises(ValueError, match="predicted temperatures must all be finite"):
        write_prediction_figures([case], target, sensor_names=("chip",), title="Failed prediction")

    assert not target.exists()


def test_writer_keeps_missing_truth_and_sensors_without_evaluation_truth(tmp_path) -> None:
    truth = np.array([[20.0, 21.0], [np.nan, np.nan], [22.0, np.nan]])
    predicted = np.array([[20.0, 21.0], [21.0, 22.0], [22.0, 23.0]])
    case = PredictionCase("missing", np.array([0.0, 1.0, 2.0]), truth, predicted)

    result = write_prediction_figures(
        [case], tmp_path, sensor_names=("chip", "sink"), title="Missing truth"
    )

    assert result.n_points == 1
    assert np.isnan(result.r_squared)
    assert result.timeseries_path.is_file()
    assert result.parity_path.is_file()


@pytest.mark.parametrize("evaluation_start_index", [0, 2])
def test_writer_preserves_requested_evaluation_start_index(
    tmp_path, evaluation_start_index: int
) -> None:
    truth = np.array([[20.0], [21.0], [22.0]])
    case = PredictionCase(
        "boundary", np.array([0.0, 1.0, 2.0]), truth, truth.copy(), evaluation_start_index
    )

    result = write_prediction_figures(
        [case], tmp_path, sensor_names=("chip",), title="Evaluation boundary"
    )

    assert result.n_points == 3 - evaluation_start_index


@pytest.mark.parametrize("evaluation_start_index", [-1, 3])
def test_prediction_case_rejects_invalid_evaluation_start_index(
    evaluation_start_index: int,
) -> None:
    truth = np.array([[20.0], [21.0], [22.0]])

    with pytest.raises(ValueError, match="evaluation_start_index"):
        PredictionCase(
            "boundary", np.array([0.0, 1.0, 2.0]), truth, truth.copy(), evaluation_start_index
        )


def test_writer_rejects_empty_cases_without_writing_figures(tmp_path) -> None:
    target = tmp_path / "figures"

    with pytest.raises(ValueError, match="at least one prediction case"):
        write_prediction_figures([], target, sensor_names=("chip",), title="Empty")

    assert not target.exists()


def test_r_squared_rejects_no_observed_truth() -> None:
    with pytest.raises(ValueError, match="no finite truth/prediction pairs"):
        coefficient_of_determination(np.full(3, np.nan), np.arange(3.0))


def test_r_squared_rejects_overflow_in_observed_truth_variation() -> None:
    truth = np.array([-1e308, 1e308])

    with pytest.raises(ValueError, match="R-squared sums must remain finite"):
        coefficient_of_determination(truth, truth.copy())


def test_writer_rejects_error_overflow_before_creating_figures(tmp_path) -> None:
    truth = np.array([[-1e308], [-1e308]])
    case = PredictionCase("overflow", np.array([0.0, 1.0]), truth, -truth)
    target = tmp_path / "figures"

    with pytest.raises(ValueError, match="prediction errors must be finite"):
        write_prediction_figures([case], target, sensor_names=("chip",), title="Overflow")

    assert not target.exists()
