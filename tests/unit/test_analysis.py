from __future__ import annotations

import numpy as np
import pytest

from celltemp.analysis import (
    control_metrics,
    control_waveform_rows,
    coverage_status,
    mae,
    max_abs_error,
    persistence_prediction,
    prediction_comparison_rows,
    prediction_error_metrics,
    prediction_sensor_rows,
    range_coverage,
    residual_dependence_rows,
    response_metrics,
    rmse,
    sensor_response_rows,
    temporal_coverage,
    thermal_case_metrics,
    uniformity_metrics,
    uniformity_trace,
)
from celltemp.domain import Trajectory


def test_error_metrics_ignore_missing_values() -> None:
    error = np.array([1.0, -2.0, np.nan])

    assert rmse(error) == pytest.approx(np.sqrt(2.5))
    assert mae(error) == pytest.approx(1.5)
    assert max_abs_error(error) == pytest.approx(2.0)


def test_prediction_error_metrics_use_aligned_finite_pairs() -> None:
    truth = np.array([[20.0, 30.0], [22.0, np.nan], [24.0, 36.0]])
    predicted = np.array([[21.0, 28.0], [21.0, 34.0], [26.0, 35.0]])

    metrics = prediction_error_metrics(truth, predicted)

    assert metrics["n_points"] == 5
    assert metrics["rmse_k"] == pytest.approx(np.sqrt(11.0 / 5.0))
    assert metrics["mae_k"] == pytest.approx(1.4)
    assert metrics["bias_k"] == pytest.approx(-0.2)
    assert metrics["max_abs_error_k"] == pytest.approx(2.0)
    assert metrics["terminal_rmse_k"] == pytest.approx(np.sqrt(2.5))


def test_prediction_error_metrics_reject_empty_arrays() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        prediction_error_metrics(np.empty((0, 2)), np.empty((0, 2)))


def test_persistence_prediction_uses_latest_causal_observation() -> None:
    temperature = np.array(
        [
            [20.0, np.nan, 40.0],
            [21.0, np.nan, np.nan],
            [np.nan, np.nan, 42.0],
            [np.nan, np.nan, np.nan],
        ]
    )
    mask = np.isfinite(temperature)

    predicted = persistence_prediction(temperature, mask, origin=2)

    np.testing.assert_allclose(predicted, [[21.0, 31.5, 42.0], [21.0, 31.5, 42.0]])


def test_persistence_prediction_requires_observed_history() -> None:
    temperature = np.full((3, 2), np.nan)

    with pytest.raises(ValueError, match="at least one observed"):
        persistence_prediction(temperature, np.zeros_like(temperature, dtype=bool), origin=0)


def test_prediction_comparison_rows_report_worst_sensor_and_peak_errors() -> None:
    time = np.array([0.0, 1.0, 2.0])
    truth = np.array([[0.0, 0.0], [2.0, 1.0], [1.0, 3.0]])
    shifted = np.array([[0.0, 1.0], [1.0, 1.0], [3.0, 2.0]])

    rows = prediction_comparison_rows(
        time,
        truth,
        {"fitted_rc": shifted, "persistence": np.zeros_like(truth)},
        ("chip", "sink"),
    )

    assert [row["model"] for row in rows] == ["fitted_rc", "persistence"]
    assert rows[0]["worst_sensor"] == "chip"
    assert rows[0]["worst_sensor_rmse_k"] == pytest.approx(np.sqrt(5.0 / 3.0))
    assert rows[0]["max_abs_peak_temperature_error_k"] == pytest.approx(1.0)
    assert rows[0]["max_abs_peak_time_error_s"] == pytest.approx(1.0)
    assert rows[0]["peak_time_qualified_sensors"] == 1
    assert rows[1]["rmse_k"] == pytest.approx(np.sqrt(15.0 / 6.0))


def test_response_metrics_describe_heating_dynamics() -> None:
    time = np.arange(12, dtype=np.float64)
    temperature = np.minimum(time * 10.0, 100.0)

    metrics = response_metrics(time, temperature, final_fraction=0.1)

    assert metrics["response_direction"] == "heating"
    assert metrics["initial_temperature"] == 0.0
    assert metrics["final_temperature"] == 100.0
    assert metrics["time_to_63_percent_s"] == pytest.approx(6.3212055883)
    assert metrics["response_time_10_90_s"] == pytest.approx(8.0)
    assert metrics["settling_time_s"] == pytest.approx(10.0)
    assert metrics["overshoot_temperature"] == 0.0
    assert metrics["max_heating_rate_per_s"] == 10.0


def test_response_metrics_marks_flat_records_without_step_metrics() -> None:
    metrics = response_metrics(np.array([0.0, 1.0, 2.0]), np.array([25.0, 25.0, 25.0]))

    assert metrics["response_direction"] == "flat"
    assert np.isnan(float(metrics["time_to_63_percent_s"]))


def test_uniformity_trace_and_summary_require_multiple_sensors() -> None:
    time = np.array([0.0, 1.0, 2.0])
    temperature = np.array([[20.0, 22.0], [21.0, np.nan], [23.0, 27.0]])

    mean, span, standard_deviation = uniformity_trace(temperature)
    summary = uniformity_metrics(time, temperature, final_fraction=0.5)

    np.testing.assert_allclose(mean, [21.0, 21.0, 25.0])
    np.testing.assert_allclose(span[[0, 2]], [2.0, 4.0])
    assert np.isnan(span[1])
    np.testing.assert_allclose(standard_deviation[[0, 2]], [1.0, 2.0])
    assert summary["maximum_sensor_span"] == 4.0
    assert summary["time_of_maximum_span_s"] == 2.0


def test_control_metrics_use_interval_durations() -> None:
    metrics = control_metrics(
        np.array([0.0, 2.0, 2.0]),
        np.array([1.0, 2.0, 1.0]),
    )

    assert metrics["time_weighted_mean_command"] == pytest.approx(1.5)
    assert metrics["command_integral_unit_s"] == pytest.approx(6.0)
    assert metrics["total_variation"] == 2.0
    assert metrics["maximum_absolute_slew_per_s"] == 2.0
    assert metrics["change_count"] == 1


def test_trajectory_metric_rows_share_names_and_control_metadata() -> None:
    time = np.array([0.0, 1.0, 3.0])
    temperature = np.array([[20.0, 21.0], [24.0, 23.0], [28.0, 26.0]])
    commands = np.array([[2.0], [4.0]])
    dt = np.diff(time)

    case = thermal_case_metrics(time, temperature)
    sensors = sensor_response_rows(time, temperature, ("core", "shell"))
    controls = control_waveform_rows(
        commands,
        dt,
        ("heater",),
        units=("W",),
        roles=("heat_input",),
    )

    assert case["rows"] == 3
    assert case["maximum_sensor_span"] == 2.0
    assert [row["sensor"] for row in sensors] == ["core", "shell"]
    assert controls[0]["unit"] == "W"
    assert controls[0]["role"] == "heat_input"
    assert controls[0]["command_integral_unit_s"] == pytest.approx(10.0)


def test_range_coverage_reports_known_and_unknown_envelopes() -> None:
    rows = range_coverage(
        {"ranges": {"heater": [0.0, 10.0]}},
        "ranges",
        "control",
        ("heater", "flow"),
        np.array([[2.0, 1.0], [12.0, 2.0]]),
    )

    assert rows[0]["within_training_range"] is False
    assert rows[1]["within_training_range"] is None
    assert coverage_status(rows) == ("unknown", "")
    assert coverage_status([rows[0]]) == ("outside", "heater")


def test_temporal_coverage_uses_only_the_forecast_interval() -> None:
    trajectory = Trajectory(
        case_id="variable",
        time=np.array([0.0, 1.0, 3.0, 4.0]),
        temperature=np.full((4, 1), 20.0),
        commands=np.array([[0.0], [8.0], [10.0]]),
        sensor_names=("temperature",),
        control_names=("heater",),
    )
    metadata = {
        "train_temporal_ranges": {
            "time_step_seconds": [0.5, 0.75],
            "forecast_horizon_seconds": [0.0, 3.0],
            "control_slew_per_second": {"heater": [0.0, 3.0]},
        }
    }

    rows = temporal_coverage(metadata, trajectory, origin=2)

    assert [row["name"] for row in rows] == ["dt", "horizon", "heater"]
    assert rows[0]["within_training_range"] is False
    assert rows[1]["within_training_range"] is True
    assert rows[2]["request_max"] == pytest.approx(1.0)


def test_prediction_diagnostics_report_peak_timing_and_residual_memory() -> None:
    time = np.arange(5, dtype=np.float64)
    truth = np.array([[20.0], [21.0], [23.0], [25.0], [24.5]])
    predicted = truth + np.arange(5, dtype=np.float64)[:, None]

    row = prediction_sensor_rows(time, truth, predicted, ("chip",))[0]

    assert row["sensor"] == "chip"
    assert row["bias_k"] == pytest.approx(2.0)
    assert row["peak_temperature_error_k"] == pytest.approx(3.5)
    assert row["peak_time_error_s"] == pytest.approx(1.0)
    assert row["truth_peak_at_boundary"] is False
    assert row["predicted_peak_at_boundary"] is True
    assert row["peak_time_qualified"] is True
    assert row["lag1_pair_count"] == 4
    assert row["residual_lag1_correlation"] == pytest.approx(1.0)


def test_prediction_diagnostics_do_not_bridge_missing_intervals() -> None:
    time = np.arange(5, dtype=np.float64)
    truth = np.arange(5, dtype=np.float64)[:, None]
    predicted = truth + np.array([[0.0], [1.0], [np.nan], [3.0], [4.0]])

    row = prediction_sensor_rows(time, truth, predicted, ("chip",))[0]

    assert row["n_points"] == 4
    assert row["lag1_pair_count"] == 2
    assert row["residual_lag1_correlation"] == pytest.approx(1.0)


def test_prediction_peak_time_is_unqualified_at_record_boundary() -> None:
    time = np.array([0.0, 1.0, 2.0])
    truth = np.array([[20.0], [21.0], [22.0]])
    predicted = np.array([[20.0], [22.0], [21.0]])

    sensor = prediction_sensor_rows(time, truth, predicted, ("chip",))[0]
    comparison = prediction_comparison_rows(time, truth, {"fitted_rc": predicted}, ("chip",))[0]

    assert sensor["truth_peak_at_boundary"] is True
    assert sensor["peak_time_qualified"] is False
    assert comparison["peak_time_qualified_sensors"] == 0
    assert np.isnan(float(comparison["max_abs_peak_time_error_s"]))


def test_residual_dependence_is_descriptive_and_marks_constant_conditions() -> None:
    time = np.arange(5, dtype=np.float64)
    truth = np.column_stack((20.0 + time, 30.0 + 2.0 * time))
    predicted = truth + (2.0 * time + 1.0)[:, None]

    rows = residual_dependence_rows(
        time,
        truth,
        predicted,
        ("chip", "fins"),
        conditions={
            "power": (time * 10.0, "W"),
            "coolant": (np.full_like(time, 25.0), "degC"),
        },
        temperature_unit="degC",
    )
    chip = {row["quantity"]: row for row in rows if row["sensor"] == "chip"}

    assert chip["time"]["residual_slope_k_per_unit"] == pytest.approx(2.0)
    assert chip["time"]["residual_correlation"] == pytest.approx(1.0)
    assert chip["power"]["residual_slope_k_per_unit"] == pytest.approx(0.2)
    assert np.isnan(float(chip["coolant"]["residual_slope_k_per_unit"]))
    assert np.isnan(float(chip["coolant"]["residual_correlation"]))
