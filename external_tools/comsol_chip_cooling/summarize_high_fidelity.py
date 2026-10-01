"""Summarize the qualified mesh evidence and compact transient pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = TOOL_ROOT / "data" / "nonlinear_high_fidelity"
CONTROLS = ("time", "chip_power", "coolant_temperature", "inlet_air_velocity")
TRUTH_SENSORS = ("truth_chip", "truth_sink_base", "truth_fins")
SENSORS = ("chip", "sink_base", "fins")
TEMPORAL_QUANTITIES = {
    "chip",
    "sink_base",
    "fins",
    "chip_max",
    "fins_max",
    "outlet_air_temperature",
    "pressure_drop",
    "radiative_heat_rate",
}


def _dynamic_paths(root: Path) -> tuple[Path, Path]:
    dynamic = root / "dynamic" / "eval"
    return (
        dynamic / "forecast" / "HV01_composite_conjugate.csv",
        dynamic / "model_gap" / "HV02_composite_radiation.csv",
    )


def _radiation_delta(base: pd.DataFrame, radiation: pd.DataFrame) -> pd.DataFrame:
    if not np.allclose(
        base[list(CONTROLS)].to_numpy(dtype=np.float64),
        radiation[list(CONTROLS)].to_numpy(dtype=np.float64),
        atol=1e-12,
    ):
        raise ValueError("high-fidelity radiation pair does not share identical inputs")
    result = base[list(CONTROLS)].copy()
    for column in [
        *TRUTH_SENSORS,
        "truth_chip_max",
        "truth_fins_max",
        "truth_outlet_air_temperature",
        "truth_pressure_drop",
    ]:
        result[f"radiation_minus_base_{column.removeprefix('truth_')}"] = (
            radiation[column] - base[column]
        )
    result["radiative_heat_rate_w"] = radiation["truth_radiative_heat_rate"]
    return result


def _case_summary(frame: pd.DataFrame) -> dict[str, object]:
    truth = frame[list(TRUTH_SENSORS)]
    return {
        "case_id": frame["case_id"].iloc[0],
        "rows": len(frame),
        "time_end_s": frame["time"].iloc[-1],
        "temperature_min_c": truth.min().min(),
        "temperature_max_c": truth.max().max(),
        "chip_final_c": frame["truth_chip"].iloc[-1],
        "chip_peak_c": frame["truth_chip_max"].max(),
        "pressure_drop_max_pa": frame["truth_pressure_drop"].max(),
        "radiative_heat_rate_max_abs_w": frame["truth_radiative_heat_rate"].abs().max(),
        "energy_residual_max_abs_w": frame["truth_energy_residual"].abs().max(),
    }


def _temporal_evidence(root: Path) -> dict[str, object] | None:
    path = root / "time_step_convergence.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    required = {
        "coarse_max_time_step_s",
        "fine_max_time_step_s",
        "quantity",
        "max_abs_difference",
        "passed",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path}: missing time-step evidence columns {sorted(missing)}")
    quantities = set(frame["quantity"].astype(str))
    if len(frame) != len(TEMPORAL_QUANTITIES) or quantities != TEMPORAL_QUANTITIES:
        raise ValueError(f"{path}: incomplete time-step quantities {sorted(quantities)}")
    steps = frame[["coarse_max_time_step_s", "fine_max_time_step_s"]].to_numpy(dtype=np.float64)
    differences = frame["max_abs_difference"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(steps).all()
        or (steps <= 0.0).any()
        or not np.isfinite(differences).all()
        or (differences < 0.0).any()
    ):
        raise ValueError(f"{path}: time-step limits must be positive and differences non-negative")
    coarse_steps = frame["coarse_max_time_step_s"].unique()
    fine_steps = frame["fine_max_time_step_s"].unique()
    if len(coarse_steps) != 1 or len(fine_steps) != 1 or fine_steps[0] >= coarse_steps[0]:
        raise ValueError(f"{path}: expected one adjacent coarse/fine time-step pair")
    indexed = frame.set_index("quantity")
    passed = frame["passed"].astype(str).str.lower().eq("true")
    sensor_differences = indexed.loc[list(SENSORS), "max_abs_difference"].to_numpy(dtype=np.float64)
    return {
        "qualified": bool(passed.all()),
        "coarse_max_time_step_s": float(coarse_steps[0]),
        "fine_max_time_step_s": float(fine_steps[0]),
        "max_sensor_temperature_difference_k": float(sensor_differences.max()),
        "sensor_uncertainty": dict(zip(SENSORS, map(float, sensor_differences), strict=True)),
        "basis": (
            f"maximum BDF step {coarse_steps[0]:g} s versus {fine_steps[0]:g} s "
            "at identical output times and controls"
        ),
        "evidence": "time_step_convergence.csv",
    }


def _dynamic_reference(
    radiation: pd.DataFrame,
    steady_reference: pd.DataFrame,
    temporal: dict[str, object] | None,
) -> pd.DataFrame:
    """Expose the physical transient at the same boundary used by experiments."""
    uncertainty = {
        sensor: float(steady_reference[f"mesh_uncertainty_{sensor}"].max()) for sensor in SENSORS
    }
    benchmark_qualified = (
        steady_reference["benchmark_qualified"].astype(str).str.lower().eq("true").all()
    )
    temporal_qualified = bool(temporal and temporal["qualified"])
    temporal_basis = (
        str(temporal["basis"])
        if temporal is not None
        else "adaptive transient solve converged; independent time-step study not performed"
    )
    columns: dict[str, object] = {
        "case_id": radiation["case_id"],
        "time": radiation["time"],
        "chip": radiation["truth_chip"],
        "sink_base": radiation["truth_sink_base"],
        "fins": radiation["truth_fins"],
        "chip_power": radiation["chip_power"],
        "coolant_temperature": radiation["coolant_temperature"],
        "inlet_air_velocity": radiation["inlet_air_velocity"],
        "mesh_uncertainty_chip": uncertainty["chip"],
        "mesh_uncertainty_sink_base": uncertainty["sink_base"],
        "mesh_uncertainty_fins": uncertainty["fins"],
        "mesh_profile": radiation["mesh_profile"],
        "mesh_qualified": False,
        "benchmark_qualified": benchmark_qualified,
        "temporal_qualified": temporal_qualified,
        "qualification_basis": (
            "local-medium benchmark qualification; conservative maximum "
            "local-medium-to-fine difference at MC01/MC02"
        ),
        "temporal_qualification_basis": temporal_basis,
    }
    if temporal is not None:
        sensor_uncertainty = temporal["sensor_uncertainty"]
        if not isinstance(sensor_uncertainty, dict):
            raise TypeError("temporal sensor uncertainty must be a mapping")
        for sensor in SENSORS:
            columns[f"temporal_uncertainty_{sensor}"] = float(sensor_uncertainty[sensor])
    return pd.DataFrame(columns)


def _experiment_template(reference: pd.DataFrame) -> pd.DataFrame:
    """Create boundary-complete rows without inventing measurements."""
    blank = pd.Series("", index=reference.index, dtype="object")
    return pd.DataFrame(
        {
            "case_id": reference["case_id"],
            "time": reference["time"],
            "chip": blank,
            "sink_base": blank,
            "fins": blank,
            "chip_power": reference["chip_power"],
            "coolant_temperature": reference["coolant_temperature"],
            "inlet_air_velocity": reference["inlet_air_velocity"],
            "uncertainty_chip": blank,
            "uncertainty_sink_base": blank,
            "uncertainty_fins": blank,
            "run_id": blank,
            "sample_id": blank,
        }
    )


def _quality_summary(
    root: Path,
    case_summary: pd.DataFrame,
    delta: pd.DataFrame,
    temporal: dict[str, object] | None,
) -> dict[str, object]:
    acceptance = pd.read_csv(root / "mesh_acceptance.csv")
    medium = acceptance.loc[acceptance["mesh_profile"] == "local-medium"].iloc[0]
    reference = pd.read_csv(root / "cae_reference.csv")
    max_mesh_uncertainty = (
        reference[["mesh_uncertainty_chip", "mesh_uncertainty_sink_base", "mesh_uncertainty_fins"]]
        .to_numpy()
        .max()
    )
    max_radiation_delta = (
        delta[
            [
                "radiation_minus_base_chip",
                "radiation_minus_base_sink_base",
                "radiation_minus_base_fins",
            ]
        ]
        .abs()
        .to_numpy()
        .max()
    )
    max_radiative_heat = delta["radiative_heat_rate_w"].abs().max()
    radiation_to_mesh_ratio = max_radiation_delta / max_mesh_uncertainty
    result: dict[str, object] = {
        "schema_version": 1,
        "strict_mesh_qualified": bool(medium["all_cases_pass"]),
        "benchmark_mesh_qualified": bool(medium["all_cases_pass_benchmark"]),
        "transient_time_discretization_qualified": bool(temporal and temporal["qualified"]),
        "experiment_validated": False,
        "max_adjacent_mesh_temperature_difference_k": float(max_mesh_uncertainty),
        "dynamic_cases": case_summary.to_dict(orient="records"),
        "radiation_pair": {
            "max_abs_sensor_temperature_delta_k": float(max_radiation_delta),
            "max_abs_radiative_heat_rate_w": float(max_radiative_heat),
            "temperature_delta_over_adjacent_mesh_difference": float(radiation_to_mesh_ratio),
            "time_and_public_inputs_identical": True,
        },
        "intended_use": "nonlinear model-form screening",
        "excluded_uses": [
            "chip design qualification",
            "hotspot safety decision",
            "final pressure-drop design value",
        ],
    }
    if temporal is not None:
        result["time_step_convergence"] = {
            key: value for key, value in temporal.items() if key != "sensor_uncertainty"
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def summarize(root: Path) -> None:
    """Publish paired effects and experiment-facing views for one data root."""
    root = root.resolve()
    base_path, radiation_path = _dynamic_paths(root)
    base = pd.read_csv(base_path)
    radiation = pd.read_csv(radiation_path)
    delta = _radiation_delta(base, radiation)
    summaries = pd.DataFrame([_case_summary(base), _case_summary(radiation)])
    steady_reference = pd.read_csv(root / "cae_reference.csv")
    temporal = _temporal_evidence(root)
    dynamic_reference = _dynamic_reference(radiation, steady_reference, temporal)
    dynamic_root = root / "dynamic"
    delta.to_csv(dynamic_root / "radiation_delta.csv", index=False, float_format="%.10g")
    summaries.to_csv(dynamic_root / "case_summary.csv", index=False, float_format="%.10g")
    dynamic_reference.to_csv(dynamic_root / "cae_reference.csv", index=False, float_format="%.10g")
    experiment_root = root / "experiment"
    _experiment_template(dynamic_reference).to_csv(
        experiment_root / "experiment_template.csv", index=False, float_format="%.10g"
    )
    _experiment_template(steady_reference).to_csv(
        experiment_root / "steady_experiment_template.csv",
        index=False,
        float_format="%.10g",
    )
    (root / "quality_summary.json").write_text(
        json.dumps(
            _quality_summary(root, summaries, delta, temporal),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    root = args.data_root.resolve()
    summarize(root)
    print(f"High-fidelity quality summary: {root / 'quality_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
