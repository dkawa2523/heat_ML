"""Qualify transient COMSOL output against one adjacent BDF step refinement."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .comsol_runtime import ComsolRuntime, select_comsol
from .dataset_support import write_csv_atomic
from .nonlinear_cases import NonlinearCase, high_fidelity_cases
from .nonlinear_dataset import CONTROLS, truth_frame
from .run_high_fidelity import _require_qualified
from .run_nonlinear import JAVA_SOURCE, raw_tables_available, solve_case
from .summarize_high_fidelity import summarize

TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = TOOL_ROOT / "data" / "nonlinear_high_fidelity"
SOURCE_CASE_ID = "HV02_composite_radiation"
DEFAULT_COARSE_MAX_STEP_S = 2.0
DEFAULT_FINE_MAX_STEP_S = 1.0

# The transient numerical budget is intentionally smaller than the strict mesh
# budget: 20% of the 0.25/0.50 K mesh limits and 25% of the 2% relative limits.
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
UNITS = {
    **dict.fromkeys(TEMPERATURE_LIMITS_K, "K"),
    "truth_pressure_drop": "Pa",
    "truth_radiative_heat_rate": "W",
}


def _validate_steps(coarse_max_step_s: float, fine_max_step_s: float) -> None:
    values = (coarse_max_step_s, fine_max_step_s)
    if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("time-step limits must be positive and finite")
    if fine_max_step_s >= coarse_max_step_s:
        raise ValueError("fine maximum time step must be smaller than the coarse value")


def _step_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p") + "s"


def _variant(source: NonlinearCase, maximum_time_step_s: float) -> NonlinearCase:
    tag = _step_tag(maximum_time_step_s)
    return replace(
        source,
        case_id=f"TV_{source.case_id}_max_step_{tag}",
        role="time_step_convergence",
        group="time_step_convergence",
        purpose=(
            f"Qualify {source.case_id} with a maximum BDF time step of {maximum_time_step_s:g} s."
        ),
        reference_case_id=source.case_id,
    )


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
    boundary = ["time", *CONTROLS]
    if coarse.shape[0] != fine.shape[0] or not np.allclose(
        coarse[boundary].to_numpy(dtype=np.float64),
        fine[boundary].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("time-step results do not share identical output times and controls")


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
    _validate_steps(coarse_max_step_s, fine_max_step_s)
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
    return pd.DataFrame(rows)


def _source_case() -> NonlinearCase:
    matches = [case for case in high_fidelity_cases() if case.case_id == SOURCE_CASE_ID]
    if len(matches) != 1:
        raise RuntimeError(f"expected one high-fidelity source case: {SOURCE_CASE_ID}")
    return matches[0]


def _prepare_runtime(
    *,
    comsol_root: str | None,
    variants: list[NonlinearCase],
    mesh_profile: str,
    reuse_raw: bool,
) -> ComsolRuntime | None:
    if reuse_raw and raw_tables_available(variants, mesh_profile):
        print("Reusing existing verified time-step raw tables", flush=True)
        return None
    runtime = select_comsol(comsol_root)
    print(f"Using COMSOL: {runtime.root}", flush=True)
    runtime.compile(JAVA_SOURCE)
    return runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comsol-root")
    parser.add_argument("--mesh-profile", default="local-medium")
    parser.add_argument("--coarse-max-step", type=float, default=DEFAULT_COARSE_MAX_STEP_S)
    parser.add_argument("--fine-max-step", type=float, default=DEFAULT_FINE_MAX_STEP_S)
    parser.add_argument("--reuse-raw", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_steps(args.coarse_max_step, args.fine_max_step)
    data_root = args.data_root.resolve()
    _require_qualified(args.mesh_profile, data_root)
    source = _source_case()
    steps = (args.coarse_max_step, args.fine_max_step)
    variants = [_variant(source, step) for step in steps]
    runtime = _prepare_runtime(
        comsol_root=args.comsol_root,
        variants=variants,
        mesh_profile=args.mesh_profile,
        reuse_raw=args.reuse_raw,
    )

    results: list[pd.DataFrame] = []
    for variant, maximum_step in zip(variants, steps, strict=True):
        print(
            f"{source.case_id}: maximum BDF step {maximum_step:g} s",
            flush=True,
        )
        raw = solve_case(
            variant,
            runtime=runtime,
            mesh_profile=args.mesh_profile,
            reuse_raw=args.reuse_raw,
            maximum_time_step_s=maximum_step,
        )
        result = truth_frame(raw, variant, mesh_profile=args.mesh_profile)
        result["source_case_id"] = source.case_id
        result["maximum_time_step_s"] = maximum_step
        results.append(result)

    evidence = compare_time_steps(
        results[0],
        results[1],
        source_case_id=source.case_id,
        mesh_profile=args.mesh_profile,
        coarse_max_step_s=steps[0],
        fine_max_step_s=steps[1],
    )
    temporal_root = data_root / "temporal"
    for result, maximum_step in zip(results, steps, strict=True):
        write_csv_atomic(result, temporal_root / f"max_step_{_step_tag(maximum_step)}.csv")
    comparison_path = temporal_root / (
        f"comparison_{_step_tag(steps[0])}_to_{_step_tag(steps[1])}.csv"
    )
    write_csv_atomic(evidence, comparison_path)
    write_csv_atomic(evidence, data_root / "time_step_convergence.csv")
    summarize(data_root)

    status = "qualified" if bool(evidence["passed"].all()) else "not qualified"
    print(f"Time-step convergence: {status}; evidence: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
