"""Qualify transient COMSOL output against one adjacent BDF step refinement."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .comsol_runtime import ComsolRuntime, select_comsol
from .dataset_support import write_csv_atomic
from .nonlinear_cases import NonlinearCase, high_fidelity_cases
from .nonlinear_dataset import truth_frame
from .qualification import (
    compare_time_steps as compare_time_steps,
)
from .qualification import (
    validate_time_step_limits,
)
from .qualification_support import file_sha256, require_mesh_qualified
from .run_nonlinear import JAVA_SOURCE, raw_tables_available, solve_case
from .summarize_high_fidelity import summarize

TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = TOOL_ROOT / "data" / "nonlinear_high_fidelity"
SOURCE_CASE_ID = "HV02_composite_radiation"
DEFAULT_COARSE_MAX_STEP_S = 2.0
DEFAULT_FINE_MAX_STEP_S = 1.0


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


def _publish_time_step_results(
    results: list[pd.DataFrame],
    *,
    steps: tuple[float, float],
    data_root: Path,
    source_case_id: str,
    mesh_profile: str,
) -> tuple[pd.DataFrame, Path]:
    """Keep exact result files, file hashes, and comparison in one provenance chain."""
    temporal_root = data_root / "temporal"
    paths = [temporal_root / f"max_step_{_step_tag(step)}.csv" for step in steps]
    for result, path in zip(results, paths, strict=True):
        write_csv_atomic(result, path)
    # Compare publication values so the recorded comparison can be recomputed
    # exactly from the portable files, without access to a COMSOL installation.
    published = [pd.read_csv(path) for path in paths]
    evidence = compare_time_steps(
        published[0],
        published[1],
        source_case_id=source_case_id,
        mesh_profile=mesh_profile,
        coarse_max_step_s=steps[0],
        fine_max_step_s=steps[1],
    )
    for label, path in zip(("coarse", "fine"), paths, strict=True):
        evidence[f"{label}_result_path"] = path.relative_to(data_root).as_posix()
        evidence[f"{label}_result_sha256"] = file_sha256(path)
    comparison_path = temporal_root / (
        f"comparison_{_step_tag(steps[0])}_to_{_step_tag(steps[1])}.csv"
    )
    write_csv_atomic(evidence, comparison_path)
    write_csv_atomic(evidence, data_root / "time_step_convergence.csv")
    return evidence, comparison_path


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
    validate_time_step_limits(args.coarse_max_step, args.fine_max_step)
    data_root = args.data_root.resolve()
    require_mesh_qualified(args.mesh_profile, data_root)
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
        result["solver_settings_verified"] = bool(raw.attrs.get("solver_settings_verified"))
        results.append(result)

    evidence, comparison_path = _publish_time_step_results(
        results,
        steps=steps,
        data_root=data_root,
        source_case_id=source.case_id,
        mesh_profile=args.mesh_profile,
    )
    summarize(data_root)

    verified = all(result["solver_settings_verified"].all() for result in results)
    status = "qualified fine result" if evidence["passed"].all() and verified else "not qualified"
    print(f"Time-step convergence: {status}; evidence: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
