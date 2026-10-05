"""Missing channels retain explicit availability without inventing a response."""

import numpy as np
import pytest

from celltemp.analysis import sensor_response_rows, sensor_waveform_rows


def test_response_rows_retain_missing_and_single_observation_channels() -> None:
    time = np.array([0.0, 1.0, 3.0])
    values = np.array([[20.0, np.nan, np.nan], [22.0, 30.0, np.nan], [24.0, np.nan, np.nan]])

    rows = sensor_response_rows(time, values, ("working", "single", "missing"))

    assert [row["n_observed_points"] for row in rows] == [3, 1, 0]
    assert rows[0]["response_status"] == "available"
    assert rows[0]["maximum_temperature"] == 24.0
    assert rows[1]["response_status"] == "insufficient_observations"
    assert rows[1]["initial_temperature"] == 30.0
    assert rows[1]["final_temperature"] == 30.0
    assert rows[1]["maximum_temperature"] == 30.0
    assert rows[1]["time_of_maximum_s"] == 1.0
    assert np.isnan(float(rows[1]["temperature_change"]))
    assert "time_to_63_percent_s" not in rows[1]
    assert "integral_change_temperature_s" not in rows[1]
    assert rows[2]["response_status"] == "insufficient_observations"
    assert np.isnan(float(rows[2]["maximum_temperature"]))


def test_multi_step_waveform_reports_last_observation_without_step_descriptors() -> None:
    time = np.arange(6, dtype=float)
    temperature = np.array([[20.0], [80.0], [30.0], [40.0], [50.0], [np.nan]])
    row = sensor_waveform_rows(time, temperature, ("tc",))[0]

    assert sensor_response_rows is sensor_waveform_rows
    assert row["final_temperature"] == 50.0
    assert row["temperature_change"] == 30.0
    assert row["maximum_temperature"] == 80.0
    assert row["time_of_maximum_s"] == 1.0
    assert row["max_heating_rate_per_s"] == 60.0
    assert row["max_cooling_rate_per_s"] == -50.0
    assert not {
        "response_direction",
        "time_to_63_percent_s",
        "response_time_10_90_s",
        "settling_time_s",
        "overshoot_temperature",
        "final_temperature_std",
        "integral_change_temperature_s",
        "integral_abs_change_temperature_s",
    }.intersection(row)


def test_missing_channels_do_not_hide_invalid_timestamps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        sensor_response_rows(np.array([0.0, 0.0]), np.full((2, 1), np.nan), ("missing",))


def test_missing_channels_do_not_hide_infinite_temperatures() -> None:
    with pytest.raises(ValueError, match="not infinite"):
        sensor_response_rows(np.array([0.0, 1.0]), np.array([[20.0], [np.inf]]), ("fault",))
