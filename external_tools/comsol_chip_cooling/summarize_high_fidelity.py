"""Summarize the qualified mesh evidence and compact transient pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset_support import write_csv_atomic
from .nonlinear_dataset import validate_dataset_frame
from .qualification_support import (
    build_dynamic_reference,
    mesh_evidence,
    temporal_evidence,
    true_values,
    validate_radiation_pair,
)

TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = TOOL_ROOT / "data" / "nonlinear_high_fidelity"
CONTROLS = ("time", "chip_power", "coolant_temperature", "inlet_air_velocity")
TRUTH_SENSORS = ("truth_chip", "truth_sink_base", "truth_fins")
SENSORS = ("chip", "sink_base", "fins")


def _dynamic_paths(root: Path) -> tuple[Path, Path]:
    dynamic = root / "dynamic" / "eval"
    return (
        dynamic / "forecast" / "HV01_composite_conjugate.csv",
        dynamic / "model_gap" / "HV02_composite_radiation.csv",
    )


def _radiation_delta(base: pd.DataFrame, radiation: pd.DataFrame) -> pd.DataFrame:
    validate_radiation_pair(base, radiation)
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
    case_summary: pd.DataFrame,
    delta: pd.DataFrame,
    temporal: dict[str, object] | None,
    dynamic_reference: pd.DataFrame,
) -> dict[str, object]:
    max_mesh_uncertainty = (
        dynamic_reference[
            ["mesh_uncertainty_chip", "mesh_uncertainty_sink_base", "mesh_uncertainty_fins"]
        ]
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
    mesh_uncertainty_available = bool(np.isfinite(max_mesh_uncertainty))
    radiation_to_mesh_ratio = (
        float(max_radiation_delta / max_mesh_uncertainty)
        if mesh_uncertainty_available and max_mesh_uncertainty > 0
        else None
    )
    result: dict[str, object] = {
        "schema_version": 2,
        "strict_mesh_qualified": bool(true_values(dynamic_reference["mesh_qualified"]).all()),
        "benchmark_mesh_qualified": bool(
            true_values(dynamic_reference["benchmark_qualified"]).all()
        ),
        "transient_time_discretization_qualified": bool(
            true_values(dynamic_reference["temporal_qualified"]).all()
        ),
        "transient_result_fingerprint": dynamic_reference["transient_result_fingerprint"].iloc[0],
        "transient_mesh_profile": dynamic_reference["mesh_profile"].iloc[0],
        "experiment_validated": False,
        "max_adjacent_mesh_temperature_difference_k": (
            float(max_mesh_uncertainty) if mesh_uncertainty_available else None
        ),
        "dynamic_cases": case_summary.to_dict(orient="records"),
        "radiation_pair": {
            "max_abs_sensor_temperature_delta_k": float(max_radiation_delta),
            "max_abs_radiative_heat_rate_w": float(max_radiative_heat),
            "temperature_delta_over_adjacent_mesh_difference": radiation_to_mesh_ratio,
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
    validate_dataset_frame(base, "forecast", "HV01_composite_conjugate")
    validate_dataset_frame(radiation, "model_gap", "HV02_composite_radiation")
    delta = _radiation_delta(base, radiation)
    summaries = pd.DataFrame([_case_summary(base), _case_summary(radiation)])
    steady_reference = pd.read_csv(root / "cae_reference.csv")
    temporal = temporal_evidence(root)
    if radiation["mesh_profile"].nunique(dropna=False) != 1:
        raise ValueError("the dynamic reference must contain one mesh profile")
    mesh = mesh_evidence(root, str(radiation["mesh_profile"].iloc[0]))
    dynamic_reference = build_dynamic_reference(radiation, mesh, temporal)
    quality_json = json.dumps(
        _quality_summary(summaries, delta, temporal, dynamic_reference),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    dynamic_root = root / "dynamic"
    write_csv_atomic(delta, dynamic_root / "radiation_delta.csv")
    write_csv_atomic(summaries, dynamic_root / "case_summary.csv")
    write_csv_atomic(dynamic_reference, dynamic_root / "cae_reference.csv")
    experiment_root = root / "experiment"
    write_csv_atomic(
        _experiment_template(dynamic_reference), experiment_root / "experiment_template.csv"
    )
    write_csv_atomic(
        _experiment_template(steady_reference), experiment_root / "steady_experiment_template.csv"
    )
    temporary = root / ".quality_summary.json.pending"
    try:
        temporary.write_text(quality_json, encoding="utf-8")
        temporary.replace(root / "quality_summary.json")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    root = args.data_root.resolve()
    summarize(root)
    print(f"High-fidelity quality summary: {root / 'quality_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
