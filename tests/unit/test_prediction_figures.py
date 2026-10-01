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
