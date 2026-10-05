"""Waveform analysis accepts channel-only input and explicit response intervals."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from celltemp.analysis import response_metrics
from celltemp.config import save_yaml
from celltemp.workflows import run_analysis

pytestmark = pytest.mark.integration


def _analysis_project(root: Path, *, controls: bool = False) -> tuple[dict, Path]:
    directory = root / "data"
    directory.mkdir()
    frame = pd.DataFrame({"time": np.arange(5, dtype=float), "tc": [np.nan, 20, 30, 40, 41]})
    if controls:
        frame["power"] = [0, 10, 0, 20, 0]
    frame.to_csv(directory / "case.csv", index=False)
    cfg = {
        "analysis": {
            "input_dir": "data",
            "output_dir": "analysis",
            "sensors": ["tc"],
            "make_plots": False,
        },
    }
    if controls:
        cfg["analysis"]["controls"] = ["power"]
    path = root / "config.yaml"
    save_yaml(cfg, path)
    return cfg, path


@pytest.mark.parametrize("controls", [False, True])
def test_waveform_analysis_needs_no_system_or_initial_observation(
    tmp_path: Path, controls: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, path = _analysis_project(tmp_path, controls=controls)
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    target = run_analysis(cfg, path)

    summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    row = pd.read_csv(target / "sensor_metrics.csv").iloc[0]
    control_frame = pd.read_csv(target / "control_metrics.csv")
    assert summary["sensors"] == ["tc"]
    assert summary["controls"] == (["power"] if controls else [])
    assert row["n_observed_points"] == 4
    assert row["initial_temperature"] == 20
    assert row["final_temperature"] == 41
    assert row["maximum_temperature"] == 41
    assert "settling_time_s" not in row.index
    assert len(control_frame) == int(controls)
    assert summary["outputs"]["response_metrics"] is None
    assert not (target / "figures").exists()


def test_explicit_response_interval_is_separate_from_basic_waveform_metrics(tmp_path: Path) -> None:
    cfg, path = _analysis_project(tmp_path, controls=True)
    cfg["analysis"]["response"] = {"case": {"start_s": 1.0, "end_s": 4.0}}

    target = run_analysis(cfg, path)

    frame = pd.read_csv(target / "response_metrics.csv")
    expected = response_metrics(np.arange(1.0, 5.0), np.array([20.0, 30.0, 40.0, 41.0]))
    assert len(frame) == 1
    assert frame.loc[0, "start_s"] == 1.0
    assert frame.loc[0, "end_s"] == 4.0
    assert frame.loc[0, "response_direction"] == "heating"
    for name in ("time_to_63_percent_s", "response_time_10_90_s", "settling_time_s"):
        assert frame.loc[0, name] == pytest.approx(expected[name])
    assert "settling_time_s" not in pd.read_csv(target / "sensor_metrics.csv").columns


@pytest.mark.parametrize("option", ["sensors", "controls"])
def test_system_and_explicit_channel_definitions_are_ambiguous(tmp_path: Path, option: str) -> None:
    cfg, path = _analysis_project(tmp_path)
    cfg["system"] = "unused.yaml"
    cfg["analysis"] = {"input_dir": "data", option: ["tc"], "make_plots": False}
    with pytest.raises(ValueError, match="cannot be combined with system"):
        run_analysis(cfg, path)


@pytest.mark.parametrize("sensors", [None, [], "tc", [True], [""], ["tc", "tc"]])
def test_channel_only_analysis_requires_unambiguous_sensor_names(
    tmp_path: Path, sensors: object
) -> None:
    cfg, path = _analysis_project(tmp_path)
    cfg["analysis"]["sensors"] = sensors
    with pytest.raises(ValueError, match=r"analysis\.sensors"):
        run_analysis(cfg, path)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"missing": {"start_s": 1, "end_s": 4}}, "unknown case"),
        ({"case": {"start_s": 1, "end_s": 5}}, "outside recorded time"),
        ({"case": {"start_s": 1.1, "end_s": 1.9}}, "at least two timestamps"),
        ({"case": {"start_s": 3, "end_s": 1}}, "start_s < end_s"),
        ({"case": {"start_s": True, "end_s": 4}}, "must be numeric"),
        ({"case": {"start_s": np.nan, "end_s": 4}}, "start_s < end_s"),
        ({"case": {"start_s": 1, "end_s": 4, "unknown": 1}}, "unknown.*interval"),
    ],
)
def test_explicit_response_rejects_invalid_intervals(
    tmp_path: Path, response: dict, message: str
) -> None:
    cfg, path = _analysis_project(tmp_path)
    cfg["analysis"]["response"] = response
    with pytest.raises(ValueError, match=message):
        run_analysis(cfg, path)
    assert not (tmp_path / "analysis").exists()


def test_response_interval_retains_missing_channel_availability(tmp_path: Path) -> None:
    cfg, path = _analysis_project(tmp_path)
    source = tmp_path / "data" / "case.csv"
    frame = pd.read_csv(source)
    frame["missing"] = np.nan
    frame["single"] = [np.nan, np.nan, 100.0, np.nan, np.nan]
    frame.to_csv(source, index=False)
    cfg["analysis"]["sensors"] = ["tc", "missing", "single"]
    cfg["analysis"]["response"] = {"case": {"start_s": 1, "end_s": 4}}

    target = run_analysis(cfg, path)

    rows = pd.read_csv(target / "response_metrics.csv").set_index("sensor")
    assert rows.loc["missing", "n_observed_points"] == 0
    assert rows.loc["single", "n_observed_points"] == 1
    assert rows.loc["single", "maximum_temperature"] == 100.0
    assert pd.isna(rows.loc["missing", "observation_start_s"])
    assert pd.isna(rows.loc["missing", "observation_end_s"])
    assert rows.loc["single", "observation_start_s"] == 2.0
    assert rows.loc["single", "observation_end_s"] == 2.0
    assert pd.isna(rows.loc["missing", "time_to_63_percent_s"])
    assert pd.isna(rows.loc["single", "settling_time_s"])


def test_system_based_analysis_also_accepts_missing_initial_measurements(tmp_path: Path) -> None:
    cfg, path = _analysis_project(tmp_path)
    del cfg["analysis"]["sensors"]
    cfg["system"] = "system.yaml"
    save_yaml(
        {
            "nodes": [{"name": "body", "heat_capacity": 1.0}],
            "sensors": [{"name": "tc", "node": "body"}],
        },
        tmp_path / "system.yaml",
    )

    target = run_analysis(cfg, path)

    row = pd.read_csv(target / "sensor_metrics.csv").iloc[0]
    assert row["n_observed_points"] == 4
    assert row["initial_temperature"] == 20.0


@pytest.mark.parametrize(
    ("temperature", "interval", "observed_end", "time_to_63"),
    [
        ([0.0, 10.0, 20.0, 30.0], {"start_s": 5.0, "end_s": 25.0}, 20.0, 6.3212055883),
        ([np.nan, 10.0, 20.0, 30.0], {"start_s": 0.0, "end_s": 30.0}, 30.0, 12.6424111766),
    ],
)
def test_response_timings_record_the_observed_origin_within_requested_bounds(
    tmp_path: Path,
    temperature: list[float],
    interval: dict[str, float],
    observed_end: float,
    time_to_63: float,
) -> None:
    cfg, path = _analysis_project(tmp_path)
    pd.DataFrame({"time": [0.0, 10.0, 20.0, 30.0], "tc": temperature}).to_csv(
        tmp_path / "data" / "case.csv", index=False
    )
    cfg["analysis"]["response"] = {"case": interval}

    target = run_analysis(cfg, path)

    row = pd.read_csv(target / "response_metrics.csv").iloc[0]
    assert row["start_s"] == interval["start_s"]
    assert row["end_s"] == interval["end_s"]
    assert row["observation_start_s"] == 10.0
    assert row["observation_end_s"] == observed_end
    assert row["time_to_63_percent_s"] == pytest.approx(time_to_63)
    assert row["response_time_10_90_s"] == pytest.approx(0.8 * (observed_end - 10.0))
    assert row["settling_time_s"] == observed_end - 10.0
