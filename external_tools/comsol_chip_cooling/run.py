"""Compile the COMSOL runner, solve all cases, and publish trajectory CSVs."""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from cases import Case, all_cases
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
APPLICATION_MODEL = Path(
    "applications/Heat_Transfer_Module/Tutorials,_Forced_and_Natural_Convection/chip_cooling.mph"
)
MODEL_CASE = "T03_power_step_8w"


def _candidate_roots(explicit: str | None) -> list[Path]:
    candidates: list[Path] = []
    for value in (explicit, os.getenv("COMSOL_ROOT")):
        if value:
            candidates.append(Path(value))
    program_files = Path(os.getenv("PROGRAMFILES", "C:/Program Files"))
    install_parent = program_files / "COMSOL" / "COMSOL64"
    candidates.extend([install_parent / "Multiphysics_copy1", install_parent / "Multiphysics"])
    if install_parent.is_dir():
        candidates.extend(sorted(install_parent.glob("Multiphysics*"), reverse=True))
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _executables(root: Path) -> tuple[Path, Path, Path]:
    binary = root / "bin" / "win64"
    return binary / "comsolbatch.exe", binary / "comsolcompile.exe", root / APPLICATION_MODEL


def select_comsol(explicit: str | None) -> tuple[Path, Path, Path, Path]:
    """Select an installation that can check out the required Heat Transfer license."""
    failures: list[str] = []
    for root in _candidate_roots(explicit):
        batch, compiler, source_model = _executables(root)
        if not (batch.is_file() and compiler.is_file() and source_model.is_file()):
            continue
        result = subprocess.run(  # noqa: S603 - resolved trusted local executable
            [str(batch), "-checklicense", str(source_model)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = f"{result.stdout}\n{result.stderr}"
        if result.returncode == 0 and "HEATTRANSFER" in output and "Error" not in output:
            return root, batch, compiler, source_model
        failures.append(f"{root}: {output.strip()[-300:]}")
    detail = "\n".join(failures) if failures else "no complete COMSOL installation found"
    raise RuntimeError(f"No usable COMSOL Heat Transfer license was found.\n{detail}")


def compile_runner(compiler: Path) -> None:
    result = subprocess.run(  # noqa: S603 - resolved trusted local executable
        [str(compiler), str(JAVA_SOURCE)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = f"{result.stdout}\n{result.stderr}"
    failed = (
        result.returncode != 0
        or not JAVA_CLASS.is_file()
        or "Failed to compile" in output
        or "Compilation error" in output
    )
    if failed:
        raise RuntimeError(f"COMSOL Java compilation failed:\n{result.stdout}\n{result.stderr}")


def _case_paths(case: Case) -> tuple[Path, Path, Path]:
    return (
        WORK_ROOT / "schedules" / f"{case.case_id}.csv",
        WORK_ROOT / "raw" / f"{case.case_id}.txt",
        WORK_ROOT / "logs" / f"{case.case_id}.log",
    )


def solve_case(
    case: Case,
    *,
    batch: Path,
    source_model: Path,
    reuse_raw: bool,
) -> pd.DataFrame:
    schedule_path, raw_path, log_path = _case_paths(case)
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
    if case.case_id == MODEL_CASE:
        model_path = WORK_ROOT / "models" / "electronic_chip_cooling_dataset.mph"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_output = str(model_path)
    command = [
        str(batch),
        "-inputfile",
        str(JAVA_CLASS),
        str(source_model),
        encoded,
        str(raw_path),
        model_output,
        "-nosave",
    ]
    result = subprocess.run(  # noqa: S603 - resolved trusted local executable
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = f"{result.stdout}\n{result.stderr}"
    log_path.write_text(output, encoding="utf-8")
    failed = (
        result.returncode != 0 or "Error running java class" in output or not raw_path.is_file()
    )
    if failed:
        tail = "\n".join(output.splitlines()[-80:])
        raise RuntimeError(f"COMSOL case {case.case_id} failed. See {log_path}.\n{tail}")
    return parse_comsol_table(raw_path, case)


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
    write_csv(frame, target)
    summary["path"] = target.relative_to(TOOL_ROOT).as_posix()
    return summary


def build_dataset(
    cases: list[Case],
    *,
    batch: Path,
    source_model: Path,
    data_root: Path,
    reuse_raw: bool,
    overwrite: bool,
) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    monitor_truth: dict[str, pd.DataFrame] = {}
    for index, case in enumerate(cases, start=1):
        print(f"[{index:02d}/{len(cases):02d}] {case.case_id}: {case.purpose}", flush=True)
        raw = solve_case(case, batch=batch, source_model=source_model, reuse_raw=reuse_raw)
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
    root, batch, compiler, source_model = select_comsol(args.comsol_root)
    print(f"Using COMSOL: {root}", flush=True)
    print(f"Source model: {source_model}", flush=True)
    compile_runner(compiler)
    summary = build_dataset(
        selected,
        batch=batch,
        source_model=source_model,
        data_root=args.data_root.resolve(),
        reuse_raw=args.reuse_raw,
        overwrite=args.overwrite,
    )
    summary_path = args.data_root.resolve() / "qa_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Created {len(summary)} published datasets under {args.data_root.resolve()}")
    print(f"QA summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
