"""Convert nonlinear COMSOL tables into self-contained trajectory CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from nonlinear_cases import NonlinearCase

SENSORS = ("chip", "sink_base", "fins")
CONTROLS = ("chip_power", "coolant_temperature", "inlet_air_velocity")
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
    "fins_max_kelvin",
    "outlet_air_kelvin",
    "pressure_drop_pa",
    "radiative_heat_rate_w",
    "heat_balance_w",
    "raw_chip_power",
    "raw_coolant_kelvin",
    "raw_inlet_air_velocity",
    "raw_hidden_power",
    "raw_effective_air_velocity",
    "chip_volume_m3",
    "sink_base_volume_m3",
    "fins_volume_m3",
)


def schedule_frame(case: NonlinearCase) -> pd.DataFrame:
    """Return the exact input schedule embedded in the COMSOL invocation."""
    return pd.DataFrame(
        {
            "time_s": case.time,
            "chip_power_w": case.chip_power,
            "coolant_temperature_c": case.coolant_temperature,
            "inlet_air_velocity_m_per_s": case.inlet_air_velocity,
            "hidden_power_w": case.hidden_power,
            "effective_air_velocity_m_per_s": case.effective_air_velocity,
            "radiation_enabled": np.full(len(case.time), int(case.radiation_enabled)),
        }
    )


def _left_limits(values: np.ndarray) -> np.ndarray:
    sampled = np.asarray(values, dtype=np.float64)
    return np.concatenate([sampled[:1], sampled[:-1]])


def _read_comsol_table(path: Path) -> pd.DataFrame:
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
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _verify_volumes(frame: pd.DataFrame, case: NonlinearCase) -> None:
    for column, expected in EXPECTED_VOLUMES_M3.items():
        if not np.allclose(frame[column], expected, rtol=1e-8, atol=1e-14):
            raise ValueError(
                f"{case.case_id}: {column} changed; verify Application Library selections"
            )


def parse_comsol_table(path: Path, case: NonlinearCase) -> pd.DataFrame:
    """Parse a transient COMSOL text table and verify every applied input."""
    frame = _read_comsol_table(path)
    if len(frame) != len(case.time) or not np.allclose(frame["time"], case.time, atol=1e-9):
        raise ValueError(f"{case.case_id}: COMSOL output times differ from the schedule")

    checks = (
        ("raw_chip_power", case.chip_power, 0.0),
        ("raw_coolant_kelvin", case.coolant_temperature + 273.15, 0.0),
        ("raw_inlet_air_velocity", case.inlet_air_velocity, 1e-12),
        ("raw_hidden_power", case.hidden_power, 0.0),
        ("raw_effective_air_velocity", case.effective_air_velocity, 1e-12),
    )
    for column, expected, tolerance in checks:
        if not np.allclose(frame[column], _left_limits(expected), atol=tolerance):
            raise ValueError(f"{case.case_id}: {column} differs from the applied schedule")

    _verify_volumes(frame, case)
    return frame


def parse_stationary_comsol_table(path: Path, case: NonlinearCase) -> pd.DataFrame:
    """Parse one stationary result used for mesh qualification."""
    frame = _read_comsol_table(path)
    if len(frame) != 1 or not np.isclose(frame["time"].iloc[0], 0.0, atol=1e-12):
        raise ValueError(f"{case.case_id}: expected one stationary row at time zero")
    checks = (
        ("raw_chip_power", case.chip_power[0], 0.0),
        ("raw_coolant_kelvin", case.coolant_temperature[0] + 273.15, 0.0),
        ("raw_inlet_air_velocity", case.inlet_air_velocity[0], 1e-12),
        ("raw_hidden_power", case.hidden_power[0], 0.0),
        ("raw_effective_air_velocity", case.effective_air_velocity[0], 1e-12),
    )
    for column, expected, tolerance in checks:
        if not np.isclose(frame[column].iloc[0], expected, atol=tolerance):
            raise ValueError(f"{case.case_id}: {column} differs from the stationary input")
    _verify_volumes(frame, case)
    return frame


def truth_frame(raw: pd.DataFrame, case: NonlinearCase, *, mesh_profile: str) -> pd.DataFrame:
    """Create common truth, controls, diagnostics, and provenance columns."""
    mesh_size_level = (
        int(mesh_profile.removeprefix("global-")) if mesh_profile.startswith("global-") else np.nan
    )
    return pd.DataFrame(
        {
            "time": raw["time"],
            "truth_chip": raw["chip_kelvin"] - 273.15,
            "truth_sink_base": raw["sink_base_kelvin"] - 273.15,
            "truth_fins": raw["fins_kelvin"] - 273.15,
            "chip_power": case.chip_power[: len(raw)],
            "coolant_temperature": case.coolant_temperature[: len(raw)],
            "inlet_air_velocity": case.inlet_air_velocity[: len(raw)],
            "truth_chip_max": raw["chip_max_kelvin"] - 273.15,
            "truth_fins_max": raw["fins_max_kelvin"] - 273.15,
            "truth_outlet_air_temperature": raw["outlet_air_kelvin"] - 273.15,
            "truth_pressure_drop": raw["pressure_drop_pa"],
            "truth_radiative_heat_rate": raw["radiative_heat_rate_w"],
            "truth_energy_residual": raw["heat_balance_w"],
            "truth_hidden_power": case.hidden_power[: len(raw)],
            "truth_effective_air_velocity": case.effective_air_velocity[: len(raw)],
            "case_id": case.case_id,
            "case_group": case.group,
            "fidelity": case.fidelity,
            "radiation_enabled": case.radiation_enabled,
            "surface_emissivity_sink": 0.90,
            "surface_emissivity_channel": 0.85,
            "mesh_profile": mesh_profile,
            "mesh_size_level": mesh_size_level,
            "input_convention": "left_zero_order_hold",
            "source_model": "COMSOL 6.4 Electronic Chip Cooling",
        }
    )


def _ordered(frame: pd.DataFrame, *, include_truth: bool) -> pd.DataFrame:
    columns = ["time", *SENSORS, *CONTROLS]
    if include_truth:
        columns.extend(f"truth_{sensor}" for sensor in SENSORS)
    columns.extend(
        [
            "truth_chip_max",
            "truth_fins_max",
            "truth_outlet_air_temperature",
            "truth_pressure_drop",
            "truth_radiative_heat_rate",
            "truth_energy_residual",
            "truth_hidden_power",
            "truth_effective_air_velocity",
            "case_id",
            "case_group",
            "fidelity",
            "radiation_enabled",
            "surface_emissivity_sink",
            "surface_emissivity_channel",
            "mesh_profile",
            "mesh_size_level",
            "input_convention",
            "source_model",
        ]
    )
    return frame[columns]


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


def monitor_frame(truth: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Add one reproducible 0.15 K measurement-noise realization."""
    frame = truth.copy()
    rng = np.random.default_rng(seed)
    for sensor in SENSORS:
        frame[sensor] = frame[f"truth_{sensor}"] + rng.normal(0.0, 0.15, len(frame))
    return _ordered(frame, include_truth=True)


def _truth_temperatures(frame: pd.DataFrame) -> pd.DataFrame:
    truth_columns = [f"truth_{sensor}" for sensor in SENSORS]
    return frame[truth_columns] if set(truth_columns) <= set(frame) else frame[list(SENSORS)]


def _validate_temperature_and_inputs(
    frame: pd.DataFrame, truth: pd.DataFrame, case_id: str
) -> None:
    if not np.isfinite(truth.to_numpy()).all():
        raise ValueError(f"{case_id}: non-finite COMSOL temperature")
    if truth.to_numpy().min() < -50.0 or truth.to_numpy().max() > 200.0:
        raise ValueError(f"{case_id}: temperature outside the intended engineering range")
    if not np.isfinite(frame[list(CONTROLS)].to_numpy()).all():
        raise ValueError(f"{case_id}: non-finite public input")
    if (frame["inlet_air_velocity"] <= 0.0).any():
        raise ValueError(f"{case_id}: inlet air velocity must remain positive")


def _validate_diagnostics(frame: pd.DataFrame, case_id: str) -> float:
    diagnostics = frame[
        [
            "truth_chip_max",
            "truth_fins_max",
            "truth_outlet_air_temperature",
            "truth_pressure_drop",
            "truth_radiative_heat_rate",
            "truth_energy_residual",
            "truth_effective_air_velocity",
        ]
    ]
    if not np.isfinite(diagnostics.to_numpy()).all():
        raise ValueError(f"{case_id}: non-finite COMSOL diagnostic")

    chip_average = (frame["truth_chip"] if "truth_chip" in frame else frame["chip"]).to_numpy()
    fins_average = (frame["truth_fins"] if "truth_fins" in frame else frame["fins"]).to_numpy()
    if np.any(frame["truth_chip_max"].to_numpy() + 1e-6 < chip_average):
        raise ValueError(f"{case_id}: chip maximum is below its volume average")
    if np.any(frame["truth_fins_max"].to_numpy() + 1e-6 < fins_average):
        raise ValueError(f"{case_id}: fins maximum is below their volume average")
    if (frame["truth_pressure_drop"] < -1e-8).any():
        raise ValueError(f"{case_id}: inlet-to-outlet pressure drop became negative")

    radiation_enabled = bool(frame["radiation_enabled"].iloc[0])
    radiation_max = float(frame["truth_radiative_heat_rate"].abs().max())
    if radiation_enabled and radiation_max <= 1e-6:
        raise ValueError(f"{case_id}: enabled radiation produced no heat transfer")
    if not radiation_enabled and radiation_max > 1e-12:
        raise ValueError(f"{case_id}: disabled radiation produced a nonzero heat rate")

    return radiation_max


def _validate_provenance(frame: pd.DataFrame, case_id: str) -> None:
    if frame["case_id"].nunique() != 1 or frame["case_id"].iloc[0] != case_id:
        raise ValueError(f"{case_id}: inconsistent case provenance")
    if frame["mesh_profile"].nunique() != 1:
        raise ValueError(f"{case_id}: mixed mesh profiles in one trajectory")
    profile = str(frame["mesh_profile"].iloc[0])
    levels = frame["mesh_size_level"]
    if profile.startswith("global-"):
        expected = int(profile.removeprefix("global-"))
        if levels.nunique() != 1 or float(levels.iloc[0]) != expected:
            raise ValueError(f"{case_id}: global mesh level differs from its profile")
    elif not levels.isna().all():
        raise ValueError(f"{case_id}: local mesh profile must not claim a global level")


def _validate_observations(frame: pd.DataFrame, role: str, case_id: str) -> np.ndarray:
    observations = frame[list(SENSORS)].notna().to_numpy()
    forecast_mask_valid = observations[0].all() and not observations[1:].any()
    if role in {"forecast", "model_gap"} and not forecast_mask_valid:
        raise ValueError(f"{case_id}: forecast data must expose only the initial observation")
    if role not in {"forecast", "model_gap"} and not observations.all():
        raise ValueError(f"{case_id}: {role} data must be fully observed")
    return observations


def validate_dataset_frame(frame: pd.DataFrame, role: str, case_id: str) -> dict[str, object]:
    """Apply compact physical and publication QA to one generated trajectory."""
    time = frame["time"].to_numpy(dtype=np.float64)
    if len(time) < 2 or not np.isfinite(time).all() or not np.all(np.diff(time) > 0.0):
        raise ValueError(f"{case_id}: invalid time axis")

    truth = _truth_temperatures(frame)
    _validate_temperature_and_inputs(frame, truth, case_id)
    radiation_max = _validate_diagnostics(frame, case_id)
    _validate_provenance(frame, case_id)
    observations = _validate_observations(frame, role, case_id)

    return {
        "case_id": case_id,
        "role": role,
        "rows": len(frame),
        "time_end_s": float(time[-1]),
        "temperature_min_c": float(truth.to_numpy().min()),
        "temperature_max_c": float(truth.to_numpy().max()),
        "chip_max_c": float(frame["truth_chip_max"].max()),
        "pressure_drop_max_pa": float(frame["truth_pressure_drop"].max()),
        "radiative_heat_rate_max_abs_w": radiation_max,
        "energy_residual_max_abs_w": float(frame["truth_energy_residual"].abs().max()),
        "energy_residual_p99_abs_w": float(frame["truth_energy_residual"].abs().quantile(0.99)),
        "missing_measurements": int((~observations).sum()),
        "observation_coverage": float(observations.mean()),
    }


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g")
