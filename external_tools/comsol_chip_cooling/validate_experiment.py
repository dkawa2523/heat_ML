"""Compare CAE and measurements at an identical case/time evaluation boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = ("chip", "sink_base", "fins")
CONTROLS = ("chip_power", "coolant_temperature", "inlet_air_velocity")
KEY = ("case_id", "time")


def _read(path: Path, *, kind: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {*KEY, *SENSORS, *CONTROLS}
    if kind == "experiment":
        required.update(f"uncertainty_{sensor}" for sensor in SENSORS)
    else:
        required.update(f"mesh_uncertainty_{sensor}" for sensor in SENSORS)
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path}: missing {kind} columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: {kind} data contains no rows")
    if frame[list(KEY)].isna().any().any() or frame.duplicated(list(KEY)).any():
        raise ValueError(f"{path}: {KEY} must be non-null and unique")
    numeric = ["time", *SENSORS, *CONTROLS]
    if kind == "experiment":
        numeric.extend(f"uncertainty_{sensor}" for sensor in SENSORS)
    else:
        numeric.extend(f"mesh_uncertainty_{sensor}" for sensor in SENSORS)
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path}: non-finite {kind} values")
    for _, trajectory in frame.groupby("case_id", sort=False):
        if len(trajectory) > 1 and not np.all(np.diff(trajectory["time"]) > 0.0):
            raise ValueError(f"{path}: time must increase within each case")
    uncertainty_columns = [
        f"uncertainty_{sensor}" if kind == "experiment" else f"mesh_uncertainty_{sensor}"
        for sensor in SENSORS
    ]
    if (frame[uncertainty_columns] <= 0.0).any().any():
        raise ValueError(f"{path}: standard uncertainties must be positive")
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
        if not np.all(delta <= 1e-8 * scale):
            raise ValueError(f"control {control} differs between CAE and experiment")

    residual_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for sensor in SENSORS:
        residual = joined[f"{sensor}_experiment"] - joined[f"{sensor}_cae"]
        combined_uncertainty = np.sqrt(
            joined[f"uncertainty_{sensor}"] ** 2 + joined[f"mesh_uncertainty_{sensor}"] ** 2
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


def main() -> int:
    args = parse_args()
    cae = _read(args.cae, kind="cae")
    if (
        not args.allow_unqualified_cae
        and "benchmark_qualified" in cae
        and not cae["benchmark_qualified"].astype(str).str.lower().eq("true").all()
    ):
        raise ValueError("CAE reference is not benchmark-qualified; refine the mesh first")
    experiment = _read(args.experiment, kind="experiment")
    residuals, metrics = compare(cae, experiment)
    args.output.mkdir(parents=True, exist_ok=True)
    residuals.to_csv(args.output / "aligned_residuals.csv", index=False, float_format="%.10g")
    metrics.to_csv(args.output / "validation_metrics.csv", index=False, float_format="%.10g")
    status = {
        "status": "evaluated",
        "acceptance_passed": None,
        "boundary_key": list(KEY),
        "rows": len(cae),
        "cases": int(cae["case_id"].nunique()),
        "interpolation_used": False,
        "acceptance_limit": "not imposed; set from the intended engineering decision",
    }
    (args.output / "validation_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(f"Experiment validation: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
