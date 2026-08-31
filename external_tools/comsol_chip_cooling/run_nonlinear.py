"""Generate the nonlinear Electronic Chip Cooling COMSOL dataset."""

from __future__ import annotations

import argparse
import base64
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from comsol_runtime import ComsolRuntime, select_comsol
from nonlinear_cases import NonlinearCase, all_nonlinear_cases
from nonlinear_dataset import (
    forecast_frame,
    monitor_frame,
    parse_comsol_table,
    parse_stationary_comsol_table,
    published_path,
    radiation_pair_summary,
    schedule_frame,
    training_frame,
    truth_frame,
    validate_dataset_frame,
    write_csv,
)

TOOL_ROOT = Path(__file__).resolve().parent
JAVA_SOURCE = TOOL_ROOT / "comsol" / "RunChipCoolingNonlinearCase.java"
JAVA_CLASS = JAVA_SOURCE.with_suffix(".class")
WORK_ROOT = TOOL_ROOT / "work" / "nonlinear"
SAVED_MODEL_CASES = {
    "NT01_power_levels": "electronic_chip_cooling_conjugate.mph",
    "NR01_power_levels": "electronic_chip_cooling_radiation.mph",
}
MESH_PROFILES = tuple(
    [f"global-{level}" for level in range(1, 10)]
    + [
        "local-coarse",
        "local-medium",
        "local-fine",
    ]
)


def _mesh_root(mesh_profile: str) -> Path:
    suffix = mesh_profile.removeprefix("global-").replace("-", "_")
    return WORK_ROOT / f"mesh_{suffix}"


def _case_paths(case: NonlinearCase, mesh_profile: str) -> tuple[Path, Path, Path]:
    mesh_root = _mesh_root(mesh_profile)
    return (
        mesh_root / "schedules" / f"{case.case_id}.csv",
        mesh_root / "raw" / f"{case.case_id}.txt",
        mesh_root / "logs" / f"{case.case_id}.log",
    )


def solve_case(
    case: NonlinearCase,
    *,
    runtime: ComsolRuntime,
    mesh_profile: str,
    reuse_raw: bool,
) -> pd.DataFrame:
    schedule_path, raw_path, log_path = _case_paths(case, mesh_profile)
    for parent in (schedule_path.parent, raw_path.parent, log_path.parent):
        parent.mkdir(parents=True, exist_ok=True)
    schedule = schedule_frame(case)
    schedule.to_csv(schedule_path, index=False, float_format="%.10g")
    if reuse_raw and raw_path.is_file():
        return parse_comsol_table(raw_path, case)
    if raw_path.exists():
        raw_path.unlink()

    encoded = base64.b64encode(schedule.to_csv(index=False).encode("utf-8")).decode("ascii")
    model_output = "-"
    if case.case_id in SAVED_MODEL_CASES:
        model_path = _mesh_root(mesh_profile) / "models" / SAVED_MODEL_CASES[case.case_id]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_output = str(model_path)
    runtime.run_java(
        JAVA_CLASS,
        [runtime.source_model, encoded, raw_path, model_output, mesh_profile],
        log_path=log_path,
        expected_output=raw_path,
        label=f"case {case.case_id}",
    )
    return parse_comsol_table(raw_path, case)


def solve_stationary_case(
    case: NonlinearCase,
    *,
    runtime: ComsolRuntime,
    mesh_profile: str,
    reuse_raw: bool,
) -> pd.DataFrame:
    """Solve and export the stationary field without an artificial transient."""
    schedule_path, raw_path, log_path = _case_paths(case, mesh_profile)
    for parent in (schedule_path.parent, raw_path.parent, log_path.parent):
        parent.mkdir(parents=True, exist_ok=True)
    schedule = schedule_frame(case)
    schedule.to_csv(schedule_path, index=False, float_format="%.10g")
    if reuse_raw and raw_path.is_file():
        return parse_stationary_comsol_table(raw_path, case)
    if raw_path.exists():
        raw_path.unlink()

    encoded = base64.b64encode(schedule.to_csv(index=False).encode("utf-8")).decode("ascii")
    runtime.run_java(
        JAVA_CLASS,
        ["steady-only", runtime.source_model, encoded, raw_path, mesh_profile],
        log_path=log_path,
        expected_output=raw_path,
        label=f"stationary case {case.case_id}",
    )
    return parse_stationary_comsol_table(raw_path, case)


def prune_obsolete_case_files(cases: list[NonlinearCase], *, data_root: Path) -> list[Path]:
    """Remove obsolete published CSVs without deleting reusable COMSOL evidence."""
    removed: list[Path] = []
    for case in cases:
        published_path(data_root, case).parent.mkdir(parents=True, exist_ok=True)
    expected_by_directory: dict[Path, set[str]] = {}
    for case in cases:
        directory = published_path(data_root, case).parent
        expected_by_directory.setdefault(directory, set()).add(case.case_id)
    for directory, expected_ids in expected_by_directory.items():
        for path in directory.glob("*.csv"):
            if path.stem not in expected_ids:
                path.unlink()
                removed.append(path)
    return removed


def _publish_frame(truth: pd.DataFrame, case: NonlinearCase) -> pd.DataFrame:
    if case.role == "train":
        return training_frame(truth)
    if case.role in {"forecast", "model_gap"}:
        return forecast_frame(truth)
    if case.role == "monitor":
        seed = 200 + int(case.case_id[2:4])
        return monitor_frame(truth, seed=seed)
    raise ValueError(f"unsupported nonlinear dataset role: {case.role}")


def build_dataset(
    cases: list[NonlinearCase],
    *,
    runtime: ComsolRuntime,
    data_root: Path,
    mesh_profile: str,
    reuse_raw: bool,
    overwrite: bool,
) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        print(
            f"[{index:02d}/{len(cases):02d}] {case.case_id} ({case.fidelity}): {case.purpose}",
            flush=True,
        )
        raw = solve_case(
            case,
            runtime=runtime,
            mesh_profile=mesh_profile,
            reuse_raw=reuse_raw,
        )
        truth = truth_frame(raw, case, mesh_profile=mesh_profile)
        published = _publish_frame(truth, case)
        target = published_path(data_root, case)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to replace {target}; pass --overwrite")
        summary = validate_dataset_frame(published, case.role, case.case_id)
        write_csv(published, target)
        summaries.append(
            {
                **summary,
                "group": case.group,
                "fidelity": case.fidelity,
                "purpose": case.purpose,
                "path": target.relative_to(TOOL_ROOT).as_posix(),
            }
        )
    return pd.DataFrame(summaries)


def inspect_mesh(*, runtime: ComsolRuntime, mesh_profile: str, data_root: Path) -> Path:
    """Build one mesh and publish the COMSOL-reported statistics as CSV."""
    log_path = _mesh_root(mesh_profile) / "mesh_build.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(runtime.batch),
        "-inputfile",
        str(JAVA_CLASS),
        "mesh-only",
        str(runtime.source_model),
        mesh_profile,
        mesh_profile,
        "-",
        "-nosave",
    ]
    started = time.perf_counter()
    result = subprocess.run(  # noqa: S603 - resolved trusted local executable
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    output = f"{result.stdout}\n{result.stderr}"
    log_path.write_text(output, encoding="utf-8")
    header = next(
        (
            line.removeprefix("MESH_CSV_HEADER,")
            for line in output.splitlines()
            if line.startswith("MESH_CSV_HEADER,")
        ),
        None,
    )
    row = next(
        (
            line.removeprefix("MESH_CSV_ROW,")
            for line in output.splitlines()
            if line.startswith("MESH_CSV_ROW,")
        ),
        None,
    )
    if result.returncode != 0 or "Error running java class" in output or not header or not row:
        tail = "\n".join(output.splitlines()[-100:])
        raise RuntimeError(f"COMSOL mesh {mesh_profile} failed. See {log_path}.\n{tail}")
    frame = pd.read_csv(StringIO(f"{header}\n{row}\n"))
    frame["mesh_build_seconds"] = elapsed
    frame["boundary_layer_stretch"] = np.nan
    frame["sink_first_layer_thickness_mm"] = np.nan
    frame["channel_first_layer_thickness_mm"] = np.nan
    if mesh_profile.startswith("local-"):
        stretch = 1.2
        frame["boundary_layer_stretch"] = stretch
        for region in ("sink", "channel"):
            layers = float(frame[f"{region}_boundary_layers"].iloc[0])
            total = float(frame[f"{region}_boundary_layer_thickness_mm"].iloc[0])
            first = total * (stretch - 1.0) / (stretch**layers - 1.0)
            frame[f"{region}_first_layer_thickness_mm"] = first
    target = data_root / "mesh" / f"{mesh_profile}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, float_format="%.10g")
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comsol-root", help="COMSOL Multiphysics installation root")
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Run only this case ID; repeat to select multiple cases",
    )
    parser.add_argument("--mesh-size", type=int, default=8, choices=range(1, 10))
    parser.add_argument(
        "--mesh-profile",
        choices=MESH_PROFILES,
        help="Explicit global or physics-local mesh profile; overrides --mesh-size",
    )
    parser.add_argument(
        "--mesh-only",
        action="store_true",
        help="Build the selected mesh and publish its statistics without solving",
    )
    parser.add_argument("--reuse-raw", action="store_true", help="Reuse verified raw tables")
    parser.add_argument("--overwrite", action="store_true", help="Replace published CSV files")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=TOOL_ROOT / "data" / "nonlinear",
        help="Published nonlinear dataset directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mesh_profile = args.mesh_profile or f"global-{args.mesh_size}"
    selected = all_nonlinear_cases()
    if args.case_ids:
        requested = set(args.case_ids)
        selected = [case for case in selected if case.case_id in requested]
        missing = requested - {case.case_id for case in selected}
        if missing:
            raise ValueError(f"unknown nonlinear case IDs: {sorted(missing)}")

    runtime = select_comsol(args.comsol_root)
    print(f"Using COMSOL: {runtime.root}", flush=True)
    print(f"Source model: {runtime.source_model}", flush=True)
    runtime.compile(JAVA_SOURCE)
    data_root = args.data_root.resolve()
    if args.mesh_only:
        target = inspect_mesh(
            runtime=runtime,
            mesh_profile=mesh_profile,
            data_root=data_root,
        )
        print(f"Mesh statistics: {target}")
        return 0
    if not args.case_ids and args.overwrite:
        removed = prune_obsolete_case_files(selected, data_root=data_root)
        if removed:
            print(f"Removed {len(removed)} obsolete generated case files", flush=True)
    summary = build_dataset(
        selected,
        runtime=runtime,
        data_root=data_root,
        mesh_profile=mesh_profile,
        reuse_raw=args.reuse_raw,
        overwrite=args.overwrite,
    )
    summary_path = data_root / "qa_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    if not args.case_ids:
        radiation_path = data_root / "radiation_pairs.csv"
        write_csv(radiation_pair_summary(selected, data_root), radiation_path)
        print(f"Radiation pair summary: {radiation_path}")
    print(f"Created {len(summary)} nonlinear datasets under {data_root}")
    print(f"QA summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
