"""Compile the COMSOL runner, solve all cases, and publish trajectory CSVs."""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import pandas as pd
from cases import Case, all_cases
from comsol_runtime import ComsolRuntime, provenance_path, select_comsol
from dataset import (
    forecast_frame,
    monitor_frames,
    parse_comsol_table,
    schedule_frame,
    training_frame,
    truth_frame,
    validate_dataset_frame,
    write_csv,
)

TOOL_ROOT = Path(__file__).resolve().parent
JAVA_SOURCE = TOOL_ROOT / "comsol" / "RunChipCoolingCase.java"
JAVA_CLASS = JAVA_SOURCE.with_suffix(".class")
WORK_ROOT = TOOL_ROOT / "work"
MODEL_CASE = "T03_power_step_8w"


def _case_paths(case: Case) -> tuple[Path, Path, Path]:
    return (
        WORK_ROOT / "schedules" / f"{case.case_id}.csv",
        WORK_ROOT / "raw" / f"{case.case_id}.txt",
        WORK_ROOT / "logs" / f"{case.case_id}.log",
    )


def solve_case(
    case: Case,
    *,
    runtime: ComsolRuntime | None,
    reuse_raw: bool,
) -> pd.DataFrame:
    schedule_path, raw_path, log_path = _case_paths(case)
    for parent in (schedule_path.parent, raw_path.parent, log_path.parent):
        parent.mkdir(parents=True, exist_ok=True)
    schedule = schedule_frame(case)
    schedule.to_csv(schedule_path, index=False, float_format="%.10g")
    if reuse_raw and raw_path.is_file():
        return parse_comsol_table(raw_path, case)
    if runtime is None:
        raise RuntimeError(f"{case.case_id}: reusable raw table is unavailable")

    encoded = base64.b64encode(schedule.to_csv(index=False).encode("utf-8")).decode("ascii")
    model_output = "-"
    if case.case_id == MODEL_CASE:
        model_path = WORK_ROOT / "models" / "electronic_chip_cooling_dataset.mph"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_output = str(model_path)
    pending_path = raw_path.with_suffix(f"{raw_path.suffix}.pending")
    pending_path.unlink(missing_ok=True)
    try:
        runtime.run_java(
            JAVA_CLASS,
            [runtime.source_model, encoded, pending_path, model_output],
            log_path=log_path,
            expected_output=pending_path,
            label=f"case {case.case_id}",
        )
        parsed = parse_comsol_table(pending_path, case)
        pending_path.replace(raw_path)
        return parsed
    finally:
        pending_path.unlink(missing_ok=True)


def _target_path(data_root: Path, role: str, case_id: str) -> Path:
    if role == "train":
        return data_root / "train" / f"{case_id}.csv"
    if role == "forecast":
        return data_root / "eval" / "forecast" / f"{case_id}.csv"
    if role == "monitor":
        return data_root / "eval" / "monitor" / f"{case_id}.csv"
    raise ValueError(f"unsupported published role: {role}")


def _write_checked(
    frame: pd.DataFrame,
    *,
    data_root: Path,
    role: str,
    case_id: str,
    overwrite: bool,
) -> dict[str, object]:
    target = _target_path(data_root, role, case_id)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace {target}; pass --overwrite")
    summary = validate_dataset_frame(frame, role, case_id)
    summary["path"] = provenance_path(target, TOOL_ROOT)
    write_csv(frame, target)
    return summary


def build_dataset(
    cases: list[Case],
    *,
    runtime: ComsolRuntime | None,
    data_root: Path,
    reuse_raw: bool,
    overwrite: bool,
) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    monitor_truth: dict[str, pd.DataFrame] = {}
    for index, case in enumerate(cases, start=1):
        print(f"[{index:02d}/{len(cases):02d}] {case.case_id}: {case.purpose}", flush=True)
        raw = solve_case(case, runtime=runtime, reuse_raw=reuse_raw)
        truth = truth_frame(raw, case)
        if case.role == "train":
            summary = _write_checked(
                training_frame(truth),
                data_root=data_root,
                role="train",
                case_id=case.case_id,
                overwrite=overwrite,
            )
            summaries.append({**summary, "group": case.group, "purpose": case.purpose})
        elif case.role == "forecast":
            summary = _write_checked(
                forecast_frame(truth),
                data_root=data_root,
                role="forecast",
                case_id=case.case_id,
                overwrite=overwrite,
            )
            summaries.append({**summary, "group": case.group, "purpose": case.purpose})
        else:
            monitor_truth[case.case_id] = truth

    required = {"R01_monitor_reference", "R02_uncommanded_heat"}
    if required <= monitor_truth.keys():
        scenarios = monitor_frames(
            monitor_truth["R01_monitor_reference"], monitor_truth["R02_uncommanded_heat"]
        )
        monitor_purpose = {
            "M01_noise_baseline": "Normal innovation width under 0.15 K sensor noise.",
            "M02_sensor_drift": "Absolute drift against the calibrated fins reference sensor.",
            "M03_sensor_offset": "Detection delay for a sudden +3 K chip-sensor fault.",
            "M04_missing_measurements": "State continuity through overlapping sensor gaps.",
            "M05_uncommanded_heat": "Innovation response to a real uncommanded heat load.",
        }
        for case_id, frame in scenarios.items():
            summary = _write_checked(
                frame,
                data_root=data_root,
                role="monitor",
                case_id=case_id,
                overwrite=overwrite,
            )
            summaries.append(
                {
                    **summary,
                    "group": "monitoring",
                    "purpose": monitor_purpose[case_id],
                }
            )
    return pd.DataFrame(summaries)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comsol-root", help="COMSOL Multiphysics installation root")
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Run only this case ID; repeat to select multiple cases",
    )
    parser.add_argument("--reuse-raw", action="store_true", help="Reuse verified raw COMSOL tables")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing published CSVs")
    parser.add_argument(
        "--data-root", type=Path, default=TOOL_ROOT / "data", help="Published dataset directory"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = all_cases()
    if args.case_ids:
        requested = set(args.case_ids)
        selected = [case for case in selected if case.case_id in requested]
        missing = requested - {case.case_id for case in selected}
        if missing:
            raise ValueError(f"unknown case IDs: {sorted(missing)}")
    reusable = args.reuse_raw and all(_case_paths(case)[1].is_file() for case in selected)
    runtime: ComsolRuntime | None = None
    if reusable:
        print("Reusing existing verified raw COMSOL tables", flush=True)
    else:
        runtime = select_comsol(args.comsol_root)
        print(f"Using COMSOL: {runtime.root}", flush=True)
        print(f"Source model: {runtime.source_model}", flush=True)
        runtime.compile(JAVA_SOURCE)
    summary = build_dataset(
        selected,
        runtime=runtime,
        data_root=args.data_root.resolve(),
        reuse_raw=args.reuse_raw,
        overwrite=args.overwrite,
    )
    summary_path = args.data_root.resolve() / "qa_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(summary, summary_path)
    print(f"Created {len(summary)} published datasets under {args.data_root.resolve()}")
    print(f"QA summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
