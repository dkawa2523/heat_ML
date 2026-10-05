"""Pure numerical criteria and comparisons for external COMSOL qualification."""

from __future__ import annotations

import hashlib
import math
from itertools import pairwise

import numpy as np
import pandas as pd

CONTROLS = ("chip_power", "coolant_temperature", "inlet_air_velocity")

SENSORS = ("chip", "sink_base", "fins")
TEMPERATURE_LIMITS_K = {
    "truth_chip": 0.05,
    "truth_sink_base": 0.05,
    "truth_fins": 0.05,
    "truth_chip_max": 0.10,
    "truth_fins_max": 0.10,
    "truth_outlet_air_temperature": 0.05,
}
RELATIVE_LIMITS = {
    "truth_pressure_drop": 0.005,
    "truth_radiative_heat_rate": 0.005,
}
RESULT_COLUMNS = (
    "time",
    "chip_power",
    "coolant_temperature",
    "inlet_air_velocity",
    *TEMPERATURE_LIMITS_K,
    *RELATIVE_LIMITS,
)
PHYSICS_COLUMNS = (
    "truth_hidden_power",
    "truth_effective_air_velocity",
    "radiation_enabled",
    "surface_emissivity_sink",
    "surface_emissivity_channel",
    "mesh_profile",
    "input_convention",
    "source_model",
)


def result_fingerprint(frame: pd.DataFrame) -> str:
    """Hash physical output and settings, independent of variant case names.

    Use the publication precision so a CSV round trip has the same identity.
    Observation masks and mutable qualification flags are deliberately excluded.
    """
    columns = [*RESULT_COLUMNS, *(name for name in PHYSICS_COLUMNS if name in frame)]
    payload = (
        frame[columns]
        .to_csv(index=False, float_format="%.10g", lineterminator="\n")
        .encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


DEFAULT_PROFILES = ("global-8", "local-coarse", "local-medium", "local-fine")
PROFILE_RANK = {profile: index for index, profile in enumerate(DEFAULT_PROFILES)}
QOI_TOLERANCES = {
    "chip_average_c": 0.25,
    "chip_max_c": 0.50,
    "sink_base_average_c": 0.25,
    "fins_average_c": 0.25,
    "fins_max_c": 0.50,
    "outlet_air_average_c": 0.20,
}
BENCHMARK_QOI_TOLERANCES = {
    "chip_average_c": 0.50,
    "chip_max_c": 0.75,
    "sink_base_average_c": 0.50,
    "fins_average_c": 0.50,
    "fins_max_c": 0.75,
    "outlet_air_average_c": 0.20,
}


def _relative_delta(current: float, finer: float) -> float:
    scale = max(abs(finer), 1e-12)
    return abs(current - finer) / scale


def compare_meshes(qoi: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case_id, group in qoi.groupby("case_id", sort=False):
        ordered = group.assign(_rank=group["mesh_profile"].map(PROFILE_RANK)).sort_values("_rank")
        records = ordered.to_dict("records")
        for current, finer in pairwise(records):
            row: dict[str, object] = {
                "case_id": case_id,
                "mesh_profile": current["mesh_profile"],
                "finer_mesh_profile": finer["mesh_profile"],
            }
            accepted = True
            benchmark_accepted = True
            for qoi_name, tolerance in QOI_TOLERANCES.items():
                delta = abs(float(current[qoi_name]) - float(finer[qoi_name]))
                row[f"{qoi_name}_absolute_delta"] = delta
                row[f"{qoi_name}_tolerance"] = tolerance
                row[f"{qoi_name}_benchmark_tolerance"] = BENCHMARK_QOI_TOLERANCES[qoi_name]
                accepted &= delta <= tolerance
                benchmark_accepted &= delta <= BENCHMARK_QOI_TOLERANCES[qoi_name]
            pressure_relative = _relative_delta(
                float(current["pressure_drop_pa"]), float(finer["pressure_drop_pa"])
            )
            row["pressure_drop_relative_delta"] = pressure_relative
            row["pressure_drop_relative_tolerance"] = 0.02
            accepted &= pressure_relative <= 0.02
            benchmark_accepted &= pressure_relative <= 0.02
            if abs(float(finer["radiative_heat_rate_w"])) > 1e-8:
                radiation_relative = _relative_delta(
                    float(current["radiative_heat_rate_w"]),
                    float(finer["radiative_heat_rate_w"]),
                )
                row["radiative_heat_rate_relative_delta"] = radiation_relative
                row["radiative_heat_rate_relative_tolerance"] = 0.02
                accepted &= radiation_relative <= 0.02
                benchmark_accepted &= radiation_relative <= 0.02
            else:
                row["radiative_heat_rate_relative_delta"] = np.nan
                row["radiative_heat_rate_relative_tolerance"] = np.nan
            row["passes_qoi_tolerances"] = accepted
            row["passes_benchmark_tolerances"] = benchmark_accepted
            rows.append(row)
    return pd.DataFrame(rows)


# The transient numerical budget is intentionally smaller than the strict mesh
# budget: 20% of the 0.25/0.50 K mesh limits and 25% of the 2% relative limits.
UNITS = {
    **dict.fromkeys(TEMPERATURE_LIMITS_K, "K"),
    "truth_pressure_drop": "Pa",
    "truth_radiative_heat_rate": "W",
}


def validate_time_step_limits(coarse_max_step_s: float, fine_max_step_s: float) -> None:
    values = (coarse_max_step_s, fine_max_step_s)
    if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("time-step limits must be positive and finite")
    if fine_max_step_s >= coarse_max_step_s:
        raise ValueError("fine maximum time step must be smaller than the coarse value")


def _validate_common_boundary(coarse: pd.DataFrame, fine: pd.DataFrame) -> None:
    required = {
        "time",
        *CONTROLS,
        *TEMPERATURE_LIMITS_K,
        *RELATIVE_LIMITS,
    }
    for label, frame in (("coarse", coarse), ("fine", fine)):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{label} time-step result is missing columns: {sorted(missing)}")
        if not np.isfinite(frame[list(required)].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{label} time-step result contains non-finite values")
        if len(frame) < 2 or not np.all(np.diff(frame["time"].to_numpy(dtype=np.float64)) > 0):
            raise ValueError(f"{label} time-step result needs increasing output times")
    boundary = ["time", *CONTROLS]
    if coarse.shape[0] != fine.shape[0] or not np.allclose(
        coarse[boundary].to_numpy(dtype=np.float64),
        fine[boundary].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("time-step results do not share identical output times and controls")
    for name in PHYSICS_COLUMNS:
        if (name in coarse or name in fine) and (
            name not in coarse or name not in fine or not coarse[name].equals(fine[name])
        ):
            raise ValueError(f"time-step results differ in physical settings: {name}")


def compare_time_steps(
    coarse: pd.DataFrame,
    fine: pd.DataFrame,
    *,
    source_case_id: str,
    mesh_profile: str,
    coarse_max_step_s: float,
    fine_max_step_s: float,
) -> pd.DataFrame:
    """Compare identical output knots without interpolating either transient."""
    validate_time_step_limits(coarse_max_step_s, fine_max_step_s)
    _validate_common_boundary(coarse, fine)
    rows: list[dict[str, object]] = []
    for quantity, limit in TEMPERATURE_LIMITS_K.items():
        difference = np.abs(
            coarse[quantity].to_numpy(dtype=np.float64) - fine[quantity].to_numpy(dtype=np.float64)
        )
        maximum = float(difference.max())
        rows.append(
            {
                "case_id": source_case_id,
                "mesh_profile": mesh_profile,
                "coarse_max_time_step_s": coarse_max_step_s,
                "fine_max_time_step_s": fine_max_step_s,
                "quantity": quantity.removeprefix("truth_"),
                "unit": UNITS[quantity],
                "comparison_metric": "max_abs_difference",
                "max_abs_difference": maximum,
                "fine_reference_peak_abs": float(
                    np.abs(fine[quantity].to_numpy(dtype=np.float64)).max()
                ),
                "observed_value": maximum,
                "acceptance_limit": limit,
                "passed": maximum <= limit,
            }
        )
    for quantity, limit in RELATIVE_LIMITS.items():
        difference = np.abs(
            coarse[quantity].to_numpy(dtype=np.float64) - fine[quantity].to_numpy(dtype=np.float64)
        )
        maximum = float(difference.max())
        reference_peak = float(np.abs(fine[quantity].to_numpy(dtype=np.float64)).max())
        if reference_peak <= 1e-12:
            raise ValueError(f"fine time-step result has no scale for {quantity}")
        relative = maximum / reference_peak
        rows.append(
            {
                "case_id": source_case_id,
                "mesh_profile": mesh_profile,
                "coarse_max_time_step_s": coarse_max_step_s,
                "fine_max_time_step_s": fine_max_step_s,
                "quantity": quantity.removeprefix("truth_"),
                "unit": UNITS[quantity],
                "comparison_metric": "max_abs_difference_over_fine_peak",
                "max_abs_difference": maximum,
                "fine_reference_peak_abs": reference_peak,
                "observed_value": relative,
                "acceptance_limit": limit,
                "passed": relative <= limit,
            }
        )
    evidence = pd.DataFrame(rows)
    evidence["coarse_result_fingerprint"] = result_fingerprint(coarse)
    evidence["fine_result_fingerprint"] = result_fingerprint(fine)
    return evidence
