"""Build compact mesh and radiation evidence from generated nonlinear data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nonlinear_cases import NonlinearCase, all_nonlinear_cases
from nonlinear_dataset import parse_comsol_table, truth_frame

TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = TOOL_ROOT / "data" / "nonlinear"
DEFAULT_WORK_ROOT = TOOL_ROOT / "work" / "nonlinear"
MESH_LEVELS = (9, 8, 7)
MESH_CASE_ID = "NT01_power_levels"
RADIATION_PAIRS = {
    "NR01_power_levels": "NT01_power_levels",
    "NR02_power_ramp": "NT05_power_ramp",
    "NR03_combined_levels": "NT06_combined_levels",
    "NR04_hot_low_flow_corner": "NF04_hot_low_flow_corner",
    "NR05_stationary_hot_start": "NF09_stationary_hot_start",
}


def _case_path(data_root: Path, case: NonlinearCase) -> Path:
    directories = {
        "train": data_root / "train",
        "forecast": data_root / "eval" / "forecast",
        "monitor": data_root / "eval" / "monitor",
        "model_gap": data_root / "eval" / "model_gap",
    }
    return directories[case.role] / f"{case.case_id}.csv"


def _truth(frame: pd.DataFrame, sensor: str) -> pd.Series:
    column = f"truth_{sensor}"
    return frame[column] if column in frame else frame[sensor]


def mesh_sensitivity(*, cases: dict[str, NonlinearCase], work_root: Path) -> pd.DataFrame:
    case = cases[MESH_CASE_ID]
    frames: dict[int, pd.DataFrame] = {}
    for level in MESH_LEVELS:
        raw_path = work_root / f"mesh_{level}" / "raw" / f"{case.case_id}.txt"
        raw = parse_comsol_table(raw_path, case)
        frames[level] = truth_frame(raw, case, mesh_profile=f"global-{level}")

    rows: list[dict[str, float | int | str]] = []
    for index, level in enumerate(MESH_LEVELS):
        frame = frames[level]
        row: dict[str, float | int | str] = {
            "case_id": case.case_id,
            "mesh_size_level": level,
            "rows": len(frame),
            "chip_final_c": float(frame["truth_chip"].iloc[-1]),
            "chip_max_final_c": float(frame["truth_chip_max"].iloc[-1]),
            "outlet_air_final_c": float(frame["truth_outlet_air_temperature"].iloc[-1]),
            "pressure_drop_final_pa": float(frame["truth_pressure_drop"].iloc[-1]),
            "energy_residual_max_abs_w": float(frame["truth_energy_residual"].abs().max()),
        }
        if index + 1 < len(MESH_LEVELS):
            finer_level = MESH_LEVELS[index + 1]
            finer = frames[finer_level]
            chip_delta = finer["truth_chip"] - frame["truth_chip"]
            pressure_delta = finer["truth_pressure_drop"] - frame["truth_pressure_drop"]
            row.update(
                {
                    "finer_mesh_size_level": finer_level,
                    "chip_rmse_to_finer_c": float(np.sqrt(np.mean(chip_delta.to_numpy() ** 2))),
                    "chip_max_abs_to_finer_c": float(chip_delta.abs().max()),
                    "chip_final_delta_finer_minus_current_c": float(chip_delta.iloc[-1]),
                    "pressure_drop_rmse_to_finer_pa": float(
                        np.sqrt(np.mean(pressure_delta.to_numpy() ** 2))
                    ),
                    "pressure_drop_final_delta_finer_minus_current_pa": float(
                        pressure_delta.iloc[-1]
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def radiation_pairs(*, cases: dict[str, NonlinearCase], data_root: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    controls = ["time", "chip_power", "coolant_temperature", "inlet_air_velocity"]
    for radiation_id, base_id in RADIATION_PAIRS.items():
        radiation_case = cases[radiation_id]
        base_case = cases[base_id]
        radiation = pd.read_csv(_case_path(data_root, radiation_case))
        base = pd.read_csv(_case_path(data_root, base_case))
        if len(radiation) != len(base) or not np.allclose(
            radiation[controls].to_numpy(dtype=np.float64),
            base[controls].to_numpy(dtype=np.float64),
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"{radiation_id}: radiation/base controls do not match")

        chip_delta = _truth(radiation, "chip") - _truth(base, "chip")
        peak_delta = radiation["truth_chip_max"] - base["truth_chip_max"]
        rows.append(
            {
                "radiation_case_id": radiation_id,
                "base_case_id": base_id,
                "rows": len(radiation),
                "chip_delta_final_c": float(chip_delta.iloc[-1]),
                "chip_delta_min_c": float(chip_delta.min()),
                "chip_delta_max_c": float(chip_delta.max()),
                "chip_peak_delta_final_c": float(peak_delta.iloc[-1]),
                "radiative_heat_rate_max_abs_w": float(
                    radiation["truth_radiative_heat_rate"].abs().max()
                ),
                "radiative_heat_rate_final_w": float(
                    radiation["truth_radiative_heat_rate"].iloc[-1]
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    work_root = args.work_root.resolve()
    cases = {case.case_id: case for case in all_nonlinear_cases()}

    mesh = mesh_sensitivity(cases=cases, work_root=work_root)
    radiation = radiation_pairs(cases=cases, data_root=data_root)
    mesh_path = data_root / "mesh_sensitivity.csv"
    radiation_path = data_root / "radiation_pairs.csv"
    mesh.to_csv(mesh_path, index=False, float_format="%.10g")
    radiation.to_csv(radiation_path, index=False, float_format="%.10g")
    print(f"Mesh evidence: {mesh_path}")
    print(f"Radiation pairs: {radiation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
