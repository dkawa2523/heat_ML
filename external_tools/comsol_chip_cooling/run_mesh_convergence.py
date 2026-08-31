"""Run physics-local mesh qualification at two nonlinear operating points."""

from __future__ import annotations

import argparse
import re
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from nonlinear_cases import NonlinearCase, mesh_convergence_cases
from nonlinear_dataset import truth_frame
from run import select_comsol
from run_nonlinear import (
    MESH_PROFILES,
    _case_paths,
    compile_runner,
    inspect_mesh,
    solve_stationary_case,
)

TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = TOOL_ROOT / "data" / "nonlinear_high_fidelity"
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


def _solver_seconds(case: NonlinearCase, profile: str) -> float:
    log_path = _case_paths(case, profile)[2]
    matches = re.findall(r"Class run time:\s*([0-9.]+)\s*s", log_path.read_text(encoding="utf-8"))
    return float(matches[-1]) if matches else np.nan


def _qoi_row(truth: pd.DataFrame, case: NonlinearCase, profile: str) -> dict[str, object]:
    final = truth.iloc[-1]
    return {
        "case_id": case.case_id,
        "mesh_profile": profile,
        "chip_average_c": final["truth_chip"],
        "chip_max_c": final["truth_chip_max"],
        "sink_base_average_c": final["truth_sink_base"],
        "fins_average_c": final["truth_fins"],
        "fins_max_c": final["truth_fins_max"],
        "outlet_air_average_c": final["truth_outlet_air_temperature"],
        "pressure_drop_pa": final["truth_pressure_drop"],
        "radiative_heat_rate_w": final["truth_radiative_heat_rate"],
        "energy_residual_w": final["truth_energy_residual"],
        "solver_seconds": _solver_seconds(case, profile),
    }


def _relative_delta(current: float, finer: float) -> float:
    scale = max(abs(finer), 1e-12)
    return abs(current - finer) / scale


def _comparison_rows(qoi: pd.DataFrame) -> pd.DataFrame:
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


def _acceptance(comparisons: pd.DataFrame, profiles: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for profile in profiles:
        evidence = (
            comparisons[comparisons["mesh_profile"] == profile]
            if "mesh_profile" in comparisons
            else comparisons
        )
        rows.append(
            {
                "mesh_profile": profile,
                "comparison_cases": len(evidence),
                "all_cases_pass": bool(
                    len(evidence) == 2 and evidence["passes_qoi_tolerances"].all()
                ),
                "all_cases_pass_benchmark": bool(
                    len(evidence) == 2 and evidence["passes_benchmark_tolerances"].all()
                ),
                "status": (
                    "strict_and_benchmark_qualified"
                    if len(evidence) == 2 and evidence["passes_qoi_tolerances"].all()
                    else (
                        "benchmark_only"
                        if len(evidence) == 2 and evidence["passes_benchmark_tolerances"].all()
                        else "not_accepted_or_not_yet_compared"
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _cae_reference(
    qoi: pd.DataFrame,
    comparisons: pd.DataFrame,
    cases: list[NonlinearCase],
) -> pd.DataFrame | None:
    """Publish the finest field with adjacent-mesh differences as uncertainty."""
    if qoi.empty or comparisons.empty or len(cases) != 2:
        return None
    available = sorted(
        qoi["mesh_profile"].unique(), key=lambda profile: PROFILE_RANK.get(profile, -1)
    )
    finest = available[-1]
    evidence = comparisons[comparisons["finer_mesh_profile"] == finest]
    if len(evidence) != len(cases):
        return None
    case_by_id = {case.case_id: case for case in cases}
    evidence_by_case = evidence.set_index("case_id")
    rows: list[dict[str, object]] = []
    qualified = bool(evidence["passes_qoi_tolerances"].all())
    benchmark_qualified = bool(evidence["passes_benchmark_tolerances"].all())
    for record in qoi[qoi["mesh_profile"] == finest].to_dict("records"):
        case = case_by_id[str(record["case_id"])]
        delta = evidence_by_case.loc[case.case_id]
        rows.append(
            {
                "case_id": case.case_id,
                "time": 0.0,
                "chip": record["chip_average_c"],
                "sink_base": record["sink_base_average_c"],
                "fins": record["fins_average_c"],
                "chip_power": case.chip_power[0],
                "coolant_temperature": case.coolant_temperature[0],
                "inlet_air_velocity": case.inlet_air_velocity[0],
                "mesh_uncertainty_chip": delta["chip_average_c_absolute_delta"],
                "mesh_uncertainty_sink_base": delta["sink_base_average_c_absolute_delta"],
                "mesh_uncertainty_fins": delta["fins_average_c_absolute_delta"],
                "mesh_profile": finest,
                "mesh_qualified": qualified,
                "benchmark_qualified": benchmark_qualified,
                "qualification_basis": (f"absolute difference from {delta['mesh_profile']}"),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comsol-root")
    parser.add_argument("--profile", action="append", choices=MESH_PROFILES, dest="profiles")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--reuse-raw", action="store_true")
    parser.add_argument("--overwrite-mesh", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = args.profiles or list(DEFAULT_PROFILES)
    cases = mesh_convergence_cases()
    if args.case_ids:
        requested = set(args.case_ids)
        cases = [case for case in cases if case.case_id in requested]
        missing = requested - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"unknown mesh-convergence case IDs: {sorted(missing)}")

    _, batch, compiler, source_model = select_comsol(args.comsol_root)
    compile_runner(compiler)
    data_root = args.data_root.resolve()
    rows: list[dict[str, object]] = []
    for profile in profiles:
        mesh_path = data_root / "mesh" / f"{profile}.csv"
        if args.overwrite_mesh or not mesh_path.is_file():
            inspect_mesh(
                batch=batch,
                source_model=source_model,
                mesh_profile=profile,
                data_root=data_root,
            )
        for case in cases:
            print(f"{profile}: {case.case_id}", flush=True)
            raw = solve_stationary_case(
                case,
                batch=batch,
                source_model=source_model,
                mesh_profile=profile,
                reuse_raw=args.reuse_raw,
            )
            truth = truth_frame(raw, case, mesh_profile=profile)
            rows.append(_qoi_row(truth, case, profile))

    qoi = pd.DataFrame(rows)
    qoi["_rank"] = qoi["mesh_profile"].map(PROFILE_RANK)
    qoi = qoi.sort_values(["case_id", "_rank"]).drop(columns="_rank")
    comparisons = _comparison_rows(qoi)
    acceptance = _acceptance(comparisons, profiles)
    data_root.mkdir(parents=True, exist_ok=True)
    qoi.to_csv(data_root / "mesh_convergence.csv", index=False, float_format="%.10g")
    comparisons.to_csv(data_root / "mesh_convergence_deltas.csv", index=False, float_format="%.10g")
    acceptance.to_csv(data_root / "mesh_acceptance.csv", index=False)
    reference = _cae_reference(qoi, comparisons, cases)
    if reference is not None:
        reference.to_csv(data_root / "cae_reference.csv", index=False, float_format="%.10g")
    print(f"Mesh convergence evidence: {data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
