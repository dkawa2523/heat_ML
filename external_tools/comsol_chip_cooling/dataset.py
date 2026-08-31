"""Convert COMSOL tables into self-contained celltemp trajectory CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from cases import Case

from celltemp.inference import forecast_origin_index

SENSORS = ("chip", "sink_base", "fins")
CONTROLS = ("chip_power", "coolant_temperature")
EXPECTED_VOLUMES_M3 = {
    "chip_volume_m3": 6.4e-6,
    "sink_base_volume_m3": 1.0e-5,
    "fins_volume_m3": 1.9e-5,
}
RAW_COLUMNS = (
    "time",
    "chip_kelvin",
    "sink_base_kelvin",
    "fins_kelvin",
    "chip_max_kelvin",
    "raw_chip_power",
    "raw_coolant_kelvin",
    "raw_hidden_power",
    "chip_volume_m3",
    "sink_base_volume_m3",
    "fins_volume_m3",
)


def schedule_frame(case: Case) -> pd.DataFrame:
    """Return the exact input table embedded in the COMSOL batch invocation."""
    return pd.DataFrame(
        {
            "time_s": case.time,
            "chip_power_w": case.chip_power,
            "coolant_temperature_c": case.coolant_temperature,
            "hidden_power_w": case.hidden_power,
            "initial_temperature_c": np.full_like(case.time, case.initial_temperature),
        }
    )


def parse_comsol_table(path: Path, case: Case) -> pd.DataFrame:
    """Parse the stable numeric portion of a COMSOL text table and verify inputs."""
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        values = [float(value) for value in stripped.split()]
        if len(values) != len(RAW_COLUMNS):
            raise ValueError(
                f"{path.name}: expected {len(RAW_COLUMNS)} numeric columns, got {len(values)}"
            )
        rows.append(values)
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    if len(frame) != len(case.time) or not np.allclose(frame["time"], case.time, atol=1e-9):
        raise ValueError(f"{case.case_id}: COMSOL output times differ from the schedule")
    # COMSOL evaluates a discontinuous Piecewise function at an output knot
    # from the completed interval (left limit). Row k still governs the next
    # interval, so the raw endpoint sequence is [u0, u0, ..., u[n-2]].
    if not np.allclose(frame["raw_chip_power"], _left_limits(case.chip_power), atol=1e-8):
        raise ValueError(f"{case.case_id}: evaluated chip power differs from the schedule")
    if not np.allclose(
        frame["raw_coolant_kelvin"] - 273.15,
        _left_limits(case.coolant_temperature),
    ):
        raise ValueError(f"{case.case_id}: evaluated coolant temperature differs from schedule")
    if not np.allclose(frame["raw_hidden_power"], _left_limits(case.hidden_power), atol=1e-8):
        raise ValueError(f"{case.case_id}: evaluated disturbance power differs from schedule")
    for column, expected in EXPECTED_VOLUMES_M3.items():
        if not np.allclose(frame[column], expected, rtol=1e-8, atol=1e-14):
            raise ValueError(
                f"{case.case_id}: {column} changed; verify the Application Library geometry"
            )
    return frame


def _left_limits(values: np.ndarray) -> np.ndarray:
    sampled = np.asarray(values, dtype=np.float64)
    return np.concatenate([sampled[:1], sampled[:-1]])


def truth_frame(raw: pd.DataFrame, case: Case) -> pd.DataFrame:
    """Create common truth, controls, diagnostics, and provenance columns."""
    frame = pd.DataFrame(
        {
            "time": raw["time"],
            "truth_chip": raw["chip_kelvin"] - 273.15,
            "truth_sink_base": raw["sink_base_kelvin"] - 273.15,
            "truth_fins": raw["fins_kelvin"] - 273.15,
            # COMSOL and celltemp both apply row k over [time[k], time[k+1]).
            "chip_power": case.chip_power,
            "coolant_temperature": case.coolant_temperature,
            "truth_chip_max": raw["chip_max_kelvin"] - 273.15,
            "truth_hidden_power": case.hidden_power,
            "case_id": case.case_id,
            "case_group": case.group,
            "fidelity": "solid_conduction_contact_constant_convection",
            "input_convention": "left_zero_order_hold",
            "source_model": "COMSOL Electronic Chip Cooling",
        }
    )
    return frame


def training_frame(truth: pd.DataFrame) -> pd.DataFrame:
    frame = truth.copy()
    for sensor in SENSORS:
        frame[sensor] = frame[f"truth_{sensor}"]
    return _ordered(frame, include_truth=False)


def forecast_frame(truth: pd.DataFrame) -> pd.DataFrame:
    frame = truth.copy()
    for sensor in SENSORS:
        frame[sensor] = np.nan
        frame.loc[0, sensor] = frame.loc[0, f"truth_{sensor}"]
    return _ordered(frame, include_truth=True)


def _monitor_frame(
    truth: pd.DataFrame,
    *,
    seed: int,
    bias: dict[str, np.ndarray] | None = None,
    missing: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    frame = truth.copy()
    rng = np.random.default_rng(seed)
    bias = bias or {}
    missing = missing or {}
    for sensor in SENSORS:
        sensor_bias = bias.get(sensor, np.zeros(len(frame), dtype=np.float64))
        measured = frame[f"truth_{sensor}"].to_numpy() + sensor_bias
        measured = measured + rng.normal(0.0, 0.15, len(frame))
        if sensor in missing:
            measured = measured.copy()
            measured[missing[sensor]] = np.nan
        frame[sensor] = measured
        frame[f"truth_bias_{sensor}"] = sensor_bias
    return _ordered(frame, include_truth=True, include_bias=True)


def monitor_frames(reference: pd.DataFrame, disturbance: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build measurement scenarios without rerunning identical physical truth."""
    time = reference["time"].to_numpy()
    drift = np.clip((time - 180.0) / 300.0, 0.0, 1.0)
    fault = time >= 300.0
    zeros = np.zeros(len(time), dtype=np.float64)

    baseline = _monitor_frame(reference, seed=101)
    baseline["case_id"] = "M01_noise_baseline"

    drifted = _monitor_frame(
        reference,
        seed=102,
        bias={"chip": 1.5 * drift, "sink_base": 0.7 * drift, "fins": zeros},
    )
    drifted["case_id"] = "M02_sensor_drift"

    faulted = _monitor_frame(
        reference,
        seed=103,
        bias={"chip": np.where(fault, 3.0, 0.0), "sink_base": zeros, "fins": zeros},
    )
    faulted["case_id"] = "M03_sensor_offset"

    missing = _monitor_frame(
        reference,
        seed=104,
        missing={
            "sink_base": (time >= 220.0) & (time <= 360.0),
            "fins": (time >= 300.0) & (time <= 420.0),
        },
    )
    missing["case_id"] = "M04_missing_measurements"

    disturbed = _monitor_frame(disturbance, seed=105)
    disturbed["case_id"] = "M05_uncommanded_heat"

    return {
        "M01_noise_baseline": baseline,
        "M02_sensor_drift": drifted,
        "M03_sensor_offset": faulted,
        "M04_missing_measurements": missing,
        "M05_uncommanded_heat": disturbed,
    }


def _ordered(
    frame: pd.DataFrame,
    *,
    include_truth: bool,
    include_bias: bool = False,
) -> pd.DataFrame:
    columns = ["time", *SENSORS, *CONTROLS]
    if include_truth:
        columns.extend(f"truth_{sensor}" for sensor in SENSORS)
    if include_bias:
        columns.extend(f"truth_bias_{sensor}" for sensor in SENSORS)
    columns.extend(
        [
            "truth_chip_max",
            "truth_hidden_power",
            "case_id",
            "case_group",
            "fidelity",
            "input_convention",
            "source_model",
        ]
    )
    return frame[columns]


def validate_dataset_frame(frame: pd.DataFrame, role: str, case_id: str) -> dict[str, object]:
    """Apply compact, decision-relevant QA before publishing a CSV."""
    time = frame["time"].to_numpy(dtype=np.float64)
    if len(time) < 2 or not np.isfinite(time).all() or not np.all(np.diff(time) > 0.0):
        raise ValueError(f"{case_id}: invalid time axis")
    truth = (
        frame[[f"truth_{sensor}" for sensor in SENSORS]]
        if role != "train"
        else frame[list(SENSORS)]
    )
    if not np.isfinite(truth.to_numpy()).all():
        raise ValueError(f"{case_id}: non-finite COMSOL truth")
    if truth.to_numpy().min() < -50.0 or truth.to_numpy().max() > 200.0:
        raise ValueError(f"{case_id}: temperature outside the intended engineering range")
    if not np.isfinite(frame[list(CONTROLS)].to_numpy()).all():
        raise ValueError(f"{case_id}: non-finite controls")
    observations = frame[list(SENSORS)].notna().to_numpy()
    if role == "forecast":
        try:
            forecast_origin_index(observations)
        except ValueError as error:
            raise ValueError(f"{case_id}: {error}") from error
    if role == "train" and not observations.all():
        raise ValueError(f"{case_id}: training data must be fully observed")
    if role == "monitor" and not observations.any(axis=1).all():
        raise ValueError(f"{case_id}: monitoring has a timestamp with no usable sensor")
    return {
        "case_id": case_id,
        "role": role,
        "rows": len(frame),
        "time_end_s": float(time[-1]),
        "temperature_min_c": float(truth.to_numpy().min()),
        "temperature_max_c": float(truth.to_numpy().max()),
        "missing_measurements": int((~observations).sum()),
    }


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g")
