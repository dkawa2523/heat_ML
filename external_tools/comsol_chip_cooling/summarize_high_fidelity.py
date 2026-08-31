"""Summarize the qualified mesh evidence and compact transient pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
    if not np.allclose(
        base[list(CONTROLS)].to_numpy(dtype=np.float64),
        radiation[list(CONTROLS)].to_numpy(dtype=np.float64),
        atol=1e-12,
    ):
        raise ValueError("high-fidelity radiation pair does not share identical inputs")
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


def _dynamic_reference(radiation: pd.DataFrame, steady_reference: pd.DataFrame) -> pd.DataFrame:
    """Expose the physical transient at the same boundary used by experiments."""
    uncertainty = {
        sensor: float(steady_reference[f"mesh_uncertainty_{sensor}"].max()) for sensor in SENSORS
    }
    benchmark_qualified = (
        steady_reference["benchmark_qualified"].astype(str).str.lower().eq("true").all()
    )
    reference = pd.DataFrame(
        {
            "case_id": radiation["case_id"],
            "time": radiation["time"],
            "chip": radiation["truth_chip"],
            "sink_base": radiation["truth_sink_base"],
            "fins": radiation["truth_fins"],
            "chip_power": radiation["chip_power"],
            "coolant_temperature": radiation["coolant_temperature"],
            "inlet_air_velocity": radiation["inlet_air_velocity"],
            "mesh_uncertainty_chip": uncertainty["chip"],
            "mesh_uncertainty_sink_base": uncertainty["sink_base"],
            "mesh_uncertainty_fins": uncertainty["fins"],
            "mesh_profile": radiation["mesh_profile"],
            "mesh_qualified": False,
            "benchmark_qualified": benchmark_qualified,
            "temporal_qualified": False,
            "qualification_basis": (
                "local-medium benchmark qualification; conservative maximum "
                "local-medium-to-fine difference at MC01/MC02"
            ),
            "temporal_qualification_basis": (
                "adaptive transient solve converged; independent time-step study not performed"
            ),
        }
    )
    return reference


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


def _quality_report(
    root: Path,
    case_summary: pd.DataFrame,
    delta: pd.DataFrame,
) -> str:
    acceptance = pd.read_csv(root / "mesh_acceptance.csv")
    medium = acceptance.loc[acceptance["mesh_profile"] == "local-medium"].iloc[0]
    reference = pd.read_csv(root / "cae_reference.csv")
    max_mesh_uncertainty = (
        reference[["mesh_uncertainty_chip", "mesh_uncertainty_sink_base", "mesh_uncertainty_fins"]]
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
    radiation_to_mesh_ratio = max_radiation_delta / max_mesh_uncertainty
    lines = [
        "# High-fidelity dataset quality report",
        "",
        "## 判定",
        "",
        f"- local-medium strict mesh qualification: `{bool(medium['all_cases_pass'])}`",
        f"- local-medium benchmark qualification: `{bool(medium['all_cases_pass_benchmark'])}`",
        f"- adjacent-mesh temperature uncertainty maximum: `{max_mesh_uncertainty:.4f} degC`",
        "- experiment validation: `not performed (no matching measurements supplied)`",
        "- transient time-discretization qualification: `not performed`",
        "",
        "## 動的CAE",
        "",
        "| case | rows | end [s] | T min/max [degC] | chip final/peak [degC] | "
        "max dp [Pa] | max |energy residual| [W] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in case_summary.to_dict("records"):
        lines.append(
            f"| {row['case_id']} | {row['rows']} | {row['time_end_s']:.0f} | "
            f"{row['temperature_min_c']:.3f}/{row['temperature_max_c']:.3f} | "
            f"{row['chip_final_c']:.3f}/{row['chip_peak_c']:.3f} | "
            f"{row['pressure_drop_max_pa']:.6f} | "
            f"{row['energy_residual_max_abs_w']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 放射pair",
            "",
            f"- maximum absolute sensor-temperature delta: `{max_radiation_delta:.4f} degC`",
            f"- maximum absolute radiative heat rate: `{max_radiative_heat:.4f} W`",
            "- temperature delta / conservative adjacent-mesh uncertainty: "
            f"`{radiation_to_mesh_ratio:.3f}`",
            "- time and all three public inputs: identical",
            "- quantitative resolution: `screening only; "
            "paired transient mesh refinement not performed`",
            "",
            "## 用途境界",
            "",
            "本データは本コードの非線形model-form error評価用です。strict mesh基準は未達のため、",
            "chip設計保証、hotspot安全判定、圧力損失の最終設計値には使いません。実験templateの",
            "時刻と入力はCAE境界から生成済みですが、温度・不確かさ欄は空であり実測値として数えません。",
            "",
        ]
    )
    return "\n".join(lines)


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
    delta = _radiation_delta(base, radiation)
    summaries = pd.DataFrame([_case_summary(base), _case_summary(radiation)])
    steady_reference = pd.read_csv(root / "cae_reference.csv")
    dynamic_reference = _dynamic_reference(radiation, steady_reference)
    dynamic_root = root / "dynamic"
    delta.to_csv(dynamic_root / "radiation_delta.csv", index=False, float_format="%.10g")
    summaries.to_csv(dynamic_root / "case_summary.csv", index=False, float_format="%.10g")
    dynamic_reference.to_csv(dynamic_root / "cae_reference.csv", index=False, float_format="%.10g")
    experiment_root = root / "experiment"
    _experiment_template(dynamic_reference).to_csv(
        experiment_root / "experiment_template.csv", index=False, float_format="%.10g"
    )
    _experiment_template(steady_reference).to_csv(
        experiment_root / "steady_experiment_template.csv",
        index=False,
        float_format="%.10g",
    )
    (root / "quality_report.md").write_text(
        _quality_report(root, summaries, delta), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    root = args.data_root.resolve()
    summarize(root)
    print(f"High-fidelity quality report: {root / 'quality_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
