"""Compare CAE and measurements at an identical case/time evaluation boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from celltemp.workflows.common import staged_output_directory

SENSORS = ("chip", "sink_base", "fins")
CONTROLS = ("chip_power", "coolant_temperature", "inlet_air_velocity")
KEY = ("case_id", "time")
CONTROL_RTOL = 1e-8


def _temporal_uncertainty_columns(frame: pd.DataFrame, path: Path, kind: str) -> list[str]:
    if kind != "cae":
        return []
    candidates = [f"temporal_uncertainty_{sensor}" for sensor in SENSORS]
    present = [column in frame for column in candidates]
    if any(present) and not all(present):
        raise ValueError(f"{path}: temporal uncertainty columns must be complete")
    return candidates if all(present) else []


def _read(path: Path, *, kind: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    uncertainty_prefix = {"cae": "mesh_uncertainty", "experiment": "uncertainty"}.get(kind)
    if uncertainty_prefix is None:
        raise ValueError(f"unsupported comparison data kind: {kind}")
    uncertainty_columns = [f"{uncertainty_prefix}_{sensor}" for sensor in SENSORS]
    temporal_columns = _temporal_uncertainty_columns(frame, path, kind)
    required = {*KEY, *SENSORS, *CONTROLS, *uncertainty_columns, *temporal_columns}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path}: missing {kind} columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: {kind} data contains no rows")
    if frame[list(KEY)].isna().any().any() or frame.duplicated(list(KEY)).any():
        raise ValueError(f"{path}: {KEY} must be non-null and unique")
    numeric = ["time", *SENSORS, *CONTROLS, *uncertainty_columns, *temporal_columns]
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path}: non-finite {kind} values")
    for _, trajectory in frame.groupby("case_id", sort=False):
        if len(trajectory) > 1 and not np.all(np.diff(trajectory["time"]) > 0.0):
            raise ValueError(f"{path}: time must increase within each case")
    if (frame[uncertainty_columns] <= 0.0).any().any():
        raise ValueError(f"{path}: standard uncertainties must be positive")
    if temporal_columns and (frame[temporal_columns] < 0.0).any().any():
        raise ValueError(f"{path}: temporal uncertainties must be non-negative")
    return frame


def compare(cae: pd.DataFrame, experiment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = cae.merge(
        experiment,
        on=list(KEY),
        how="outer",
        suffixes=("_cae", "_experiment"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = joined[joined["_merge"] != "both"]
    if not unmatched.empty:
        counts = unmatched["_merge"].value_counts().to_dict()
        raise ValueError(f"CAE/experiment keys do not define the same boundary: {counts}")
    for control in CONTROLS:
        delta = (joined[f"{control}_cae"] - joined[f"{control}_experiment"]).abs()
        scale = np.maximum(joined[f"{control}_cae"].abs(), 1.0)
        if not np.all(delta <= CONTROL_RTOL * scale):
            raise ValueError(f"control {control} differs between CAE and experiment")

    residual_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for sensor in SENSORS:
        residual = joined[f"{sensor}_experiment"] - joined[f"{sensor}_cae"]
        temporal_column = f"temporal_uncertainty_{sensor}"
        temporal_uncertainty = (
            joined[temporal_column]
            if temporal_column in joined
            else pd.Series(0.0, index=joined.index)
        )
        combined_uncertainty = np.sqrt(
            joined[f"uncertainty_{sensor}"] ** 2
            + joined[f"mesh_uncertainty_{sensor}"] ** 2
            + temporal_uncertainty**2
        )
        normalized = residual / combined_uncertainty
        residual_rows.append(
            pd.DataFrame(
                {
                    "case_id": joined["case_id"],
                    "time": joined["time"],
                    "sensor": sensor,
                    "cae_c": joined[f"{sensor}_cae"],
                    "experiment_c": joined[f"{sensor}_experiment"],
                    "residual_experiment_minus_cae_c": residual,
                    "experiment_standard_uncertainty_c": joined[f"uncertainty_{sensor}"],
                    "mesh_standard_uncertainty_c": joined[f"mesh_uncertainty_{sensor}"],
                    "temporal_standard_uncertainty_c": temporal_uncertainty,
                    "combined_standard_uncertainty_c": combined_uncertainty,
                    "normalized_residual": normalized,
                }
            )
        )
        metric_rows.append(
            {
                "sensor": sensor,
                "rows": len(joined),
                "bias_c": residual.mean(),
                "mae_c": residual.abs().mean(),
                "rmse_c": float(np.sqrt(np.mean(residual.to_numpy() ** 2))),
                "max_abs_error_c": residual.abs().max(),
                "normalized_rmse": float(np.sqrt(np.mean(normalized.to_numpy() ** 2))),
                "fraction_within_95_percent_uncertainty": (normalized.abs() <= 1.96).mean(),
            }
        )
    return pd.concat(residual_rows, ignore_index=True), pd.DataFrame(metric_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cae", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-unqualified-cae",
        action="store_true",
        help="Allow diagnostic comparison when benchmark_qualified is false",
    )
    return parser.parse_args()


def _write_outputs(
    target: Path,
    residuals: pd.DataFrame,
    metrics: pd.DataFrame,
    status: dict[str, object],
) -> None:
    """Serialize one complete comparison result into an empty staging directory."""
    residuals.to_csv(target / "aligned_residuals.csv", index=False, float_format="%.10g")
    metrics.to_csv(target / "validation_metrics.csv", index=False, float_format="%.10g")
    (target / "validation_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def run_validation(
    cae_path: Path,
    experiment_path: Path,
    output: Path,
    *,
    allow_unqualified_cae: bool = False,
) -> None:
    """Validate, compare, and atomically replace one experiment result directory."""
    cae = _read(cae_path, kind="cae")
    if (
        not allow_unqualified_cae
        and "benchmark_qualified" in cae
        and not cae["benchmark_qualified"].astype(str).str.lower().eq("true").all()
    ):
        raise ValueError("CAE reference is not benchmark-qualified; refine the mesh first")
    experiment = _read(experiment_path, kind="experiment")
    residuals, metrics = compare(cae, experiment)
    status: dict[str, object] = {
        "status": "evaluated",
        "acceptance_passed": None,
        "boundary_key": list(KEY),
        "rows": len(cae),
        "cases": len(cae["case_id"].unique()),
        "interpolation_used": False,
        "acceptance_limit": "not imposed; set from the intended engineering decision",
    }
    with staged_output_directory(output.resolve(), overwrite=True) as staging:
        _write_outputs(staging, residuals, metrics, status)


def main() -> int:
    args = parse_args()
    run_validation(
        args.cae,
        args.experiment,
        args.output,
        allow_unqualified_cae=args.allow_unqualified_cae,
    )
    print(f"Experiment validation: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
