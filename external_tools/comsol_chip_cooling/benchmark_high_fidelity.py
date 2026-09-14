"""Evaluate the shared thermal network against the local-mesh nonlinear CAE pair."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from celltemp.artifact import fitted_parameters, load_artifact
from celltemp.config import as_path, load_config
from celltemp.engine import ThermalRCModel
from celltemp.inference import forecast
from celltemp.io import load_system_spec, trajectory_from_frame
from celltemp.workflows import run_forecast, run_train
from celltemp.workflows.common import staged_output_directory

SENSORS = ("chip", "sink_base", "fins")
CONTROLS = ("chip_power", "coolant_temperature", "inlet_air_velocity")
CASE_LOCATIONS = {
    "HV01_composite_conjugate": Path(
        "data/nonlinear_high_fidelity/dynamic/eval/forecast/HV01_composite_conjugate.csv"
    ),
    "HV02_composite_radiation": Path(
        "data/nonlinear_high_fidelity/dynamic/eval/model_gap/HV02_composite_radiation.csv"
    ),
}
RESPONSE_PHASES = (
    ("initialization", 0.0, 60.0),
    ("power_excitation", 60.0, 120.0),
    ("airflow_excitation", 120.0, 160.0),
    ("coupled_hot_low_flow", 160.0, 220.0),
)


def _rmse(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array**2)))


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(values, dtype=np.float64))))


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values, dtype=np.float64))))


def _percent_improvement(prior: float, fitted: float) -> float:
    return float(100.0 * (prior - fitted) / prior) if prior > 0.0 else 0.0


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _request(case_id: str, frame: pd.DataFrame):
    return trajectory_from_frame(
        case_id=case_id,
        frame=frame,
        time_col="time",
        sensor_cols=SENSORS,
        control_cols=CONTROLS,
        control_convention="left",
    )


def _predict(model: ThermalRCModel, case_id: str, frame: pd.DataFrame) -> np.ndarray:
    return forecast(model, _request(case_id, frame)).sensor_temperature


def _load_cases(root: Path) -> dict[str, pd.DataFrame]:
    frames = {case_id: pd.read_csv(root / relative) for case_id, relative in CASE_LOCATIONS.items()}
    required = {
        "time",
        *SENSORS,
        *CONTROLS,
        *(f"truth_{sensor}" for sensor in SENSORS),
        "truth_chip_max",
    }
    for case_id, frame in frames.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{case_id}: missing benchmark columns {missing}")
        if not frame[list(SENSORS)].iloc[0].notna().all():
            raise ValueError(f"{case_id}: initial observations must be complete")
        if frame[list(SENSORS)].iloc[1:].notna().any().any():
            raise ValueError(f"{case_id}: future observations would leak CAE truth")
        if not np.isfinite(frame[[*(f"truth_{name}" for name in SENSORS), *CONTROLS]]).all().all():
            raise ValueError(f"{case_id}: truth and controls must be finite")
    return frames


def _validate_pair(frames: dict[str, pd.DataFrame]) -> bool:
    base = frames["HV01_composite_conjugate"]
    radiation = frames["HV02_composite_radiation"]
    columns = ["time", *CONTROLS]
    return bool(
        np.allclose(
            base[columns].to_numpy(dtype=np.float64),
            radiation[columns].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-10,
        )
    )


def _experiment_status(root: Path) -> tuple[bool, bool]:
    path = root / "data/nonlinear_high_fidelity/experiment/result/validation_status.json"
    if not path.exists():
        return False, False
    status = json.loads(path.read_text(encoding="utf-8"))
    compared = status.get("status") == "evaluated"
    accepted = compared and status.get("acceptance_passed") is True
    return compared, accepted


def _reference_quality(root: Path, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    path = root / "data/nonlinear_high_fidelity/dynamic/cae_reference.csv"
    reference = pd.read_csv(path)
    radiation = frames["HV02_composite_radiation"]
    columns = ["time", *SENSORS, *CONTROLS]
    expected = radiation[["time", *(f"truth_{name}" for name in SENSORS), *CONTROLS]].copy()
    expected.columns = columns
    reference_matches = bool(
        np.allclose(
            reference[columns].to_numpy(dtype=np.float64),
            expected.to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-9,
        )
    )
    uncertainty = {
        sensor: float(reference[f"mesh_uncertainty_{sensor}"].max()) for sensor in SENSORS
    }
    experiment_compared, experiment_validated = _experiment_status(root)
    return {
        "dynamic_reference_matches_hv02": reference_matches,
        "mesh_profile": str(reference["mesh_profile"].iat[0]),
        "mesh_qualified": bool(reference["mesh_qualified"].all()),
        "benchmark_qualified": bool(reference["benchmark_qualified"].all()),
        "temporal_qualified": bool(reference["temporal_qualified"].all()),
        "experiment_compared": experiment_compared,
        "experiment_validated": experiment_validated,
        "mesh_difference_k": uncertainty,
    }


def _forecast_directories(cfg: dict[str, Any], root: Path) -> dict[str, Path]:
    conjugate = as_path(str(cfg["forecast"]["output_dir"]), root)
    radiation = root / "work/high_fidelity_benchmark/forecast/radiation"
    return {
        "HV01_composite_conjugate": conjugate,
        "HV02_composite_radiation": radiation,
    }


def _run_workflows(cfg: dict[str, Any], config_path: Path) -> tuple[Path, dict[str, Path]]:
    root = config_path.parent
    run_dir = run_train(cfg, config_path)
    output_dirs = _forecast_directories(cfg, root)
    run_forecast(cfg, config_path)
    radiation_cfg = copy.deepcopy(cfg)
    radiation_cfg["forecast"]["input_dir"] = str(CASE_LOCATIONS["HV02_composite_radiation"].parent)
    radiation_cfg["forecast"]["output_dir"] = str(
        output_dirs["HV02_composite_radiation"].relative_to(root)
    )
    run_forecast(radiation_cfg, config_path)
    return run_dir, output_dirs


def _saved_prediction(
    case_id: str,
    frame: pd.DataFrame,
    directory: Path,
    model: ThermalRCModel,
) -> np.ndarray:
    result = pd.read_csv(directory / f"{case_id}.csv")
    if not np.allclose(result["time"], frame["time"], rtol=0.0, atol=1e-10):
        raise ValueError(f"{case_id}: forecast output is not time-aligned")
    columns = [f"temperature_{sensor}" for sensor in SENSORS]
    saved = result[columns].to_numpy(dtype=np.float64)
    direct = _predict(model, case_id, frame)
    if not np.allclose(saved, direct, rtol=1e-10, atol=1e-10):
        raise ValueError(f"{case_id}: public forecast differs from artifact replay")
    if not np.isfinite(saved).all():
        raise ValueError(f"{case_id}: forecast contains non-finite values")
    return saved


def _prediction_frame(
    source: pd.DataFrame,
    predicted: np.ndarray,
    prior: np.ndarray,
    uncertainty: dict[str, float],
) -> pd.DataFrame:
    result = source[["time", *CONTROLS]].copy()
    for index, sensor in enumerate(SENSORS):
        truth = source[f"truth_{sensor}"].to_numpy(dtype=np.float64)
        result[f"truth_{sensor}"] = truth
        result[f"predicted_{sensor}"] = predicted[:, index]
        result[f"error_{sensor}"] = predicted[:, index] - truth
        result[f"prior_{sensor}"] = prior[:, index]
        result[f"mesh_difference_{sensor}"] = uncertainty[sensor]
    result["truth_chip_max"] = source["truth_chip_max"]
    result["hotspot_underprediction"] = source["truth_chip_max"] - predicted[:, 0]
    return result


def _sensor_rows(
    case_id: str,
    truth: np.ndarray,
    predicted: np.ndarray,
    prior: np.ndarray,
    uncertainty: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sensor in enumerate(SENSORS):
        error = predicted[1:, index] - truth[1:, index]
        prior_error = prior[1:, index] - truth[1:, index]
        fitted_rmse = _rmse(error)
        prior_rmse = _rmse(prior_error)
        mesh_difference = uncertainty[sensor]
        rows.append(
            {
                "case_id": case_id,
                "sensor": sensor,
                "n_points": len(error),
                "rmse_k": fitted_rmse,
                "mae_k": _mae(error),
                "bias_k": float(np.mean(error)),
                "max_abs_error_k": _max_abs(error),
                "terminal_error_k": float(error[-1]),
                "prior_rmse_k": prior_rmse,
                "rmse_improvement_percent": _percent_improvement(prior_rmse, fitted_rmse),
                "mesh_difference_k": mesh_difference,
                "rmse_over_mesh_difference": fitted_rmse / mesh_difference,
                "max_abs_error_over_mesh_difference": _max_abs(error) / mesh_difference,
                "fraction_abs_error_within_mesh_difference": float(
                    np.mean(np.abs(error) <= mesh_difference)
                ),
                "model_error_resolved_above_mesh_difference": bool(fitted_rmse > mesh_difference),
                "pointwise_error_resolved_above_mesh_difference": bool(
                    _max_abs(error) > mesh_difference
                ),
            }
        )
    return rows


def _case_row(
    case_id: str,
    source: pd.DataFrame,
    truth: np.ndarray,
    predicted: np.ndarray,
    prior: np.ndarray,
    uncertainty: dict[str, float],
) -> dict[str, Any]:
    error = predicted[1:] - truth[1:]
    prior_error = prior[1:] - truth[1:]
    fitted_rmse = _rmse(error)
    prior_rmse = _rmse(prior_error)
    mesh = np.asarray([uncertainty[name] for name in SENSORS])
    hotspot_error = source["truth_chip_max"].to_numpy(dtype=np.float64)[1:] - predicted[1:, 0]
    return {
        "case_id": case_id,
        "radiation_enabled": bool(source["radiation_enabled"].iat[0]),
        "n_forecast_timepoints": len(error),
        "n_scalar_predictions": int(error.size),
        "rmse_k": fitted_rmse,
        "mae_k": _mae(error),
        "max_abs_error_k": _max_abs(error),
        "prior_rmse_k": prior_rmse,
        "rmse_improvement_percent": _percent_improvement(prior_rmse, fitted_rmse),
        "rms_error_over_mesh_difference": _rmse(error / mesh),
        "max_abs_error_over_mesh_difference": float(np.max(np.abs(error) / mesh)),
        "fraction_abs_error_within_mesh_difference": float(np.mean(np.abs(error) <= mesh)),
        "chip_peak_error_k": float(np.max(predicted[:, 0]) - np.max(truth[:, 0])),
        "max_hotspot_underprediction_k": float(max(0.0, np.max(hotspot_error))),
    }


def _phase_rows(
    case_id: str,
    source: pd.DataFrame,
    truth: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, Any]]:
    time = source["time"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for phase, start, end in RESPONSE_PHASES:
        mask = (time > start) & (time <= end)
        error = predicted[mask] - truth[mask]
        rows.append(
            {
                "case_id": case_id,
                "phase": phase,
                "response_time_start_exclusive_s": start,
                "response_time_end_inclusive_s": end,
                "n_timepoints": int(mask.sum()),
                "rmse_k": _rmse(error),
                "mae_k": _mae(error),
                "max_abs_error_k": _max_abs(error),
            }
        )
    return rows


def _pair_rows(
    frames: dict[str, pd.DataFrame],
    predictions: dict[str, np.ndarray],
    uncertainty: dict[str, float],
) -> pd.DataFrame:
    base_id = "HV01_composite_conjugate"
    radiation_id = "HV02_composite_radiation"
    base_truth = frames[base_id][[f"truth_{name}" for name in SENSORS]].to_numpy(dtype=np.float64)
    radiation_truth = frames[radiation_id][[f"truth_{name}" for name in SENSORS]].to_numpy(
        dtype=np.float64
    )
    truth_delta = radiation_truth - base_truth
    predicted_delta = predictions[radiation_id] - predictions[base_id]
    rows: list[dict[str, Any]] = []
    for index, sensor in enumerate(SENSORS):
        delta_error = predicted_delta[1:, index] - truth_delta[1:, index]
        max_effect = _max_abs(truth_delta[:, index])
        rows.append(
            {
                "sensor": sensor,
                "truth_terminal_radiation_delta_k": float(truth_delta[-1, index]),
                "predicted_terminal_pair_delta_k": float(predicted_delta[-1, index]),
                "max_abs_truth_radiation_delta_k": max_effect,
                "max_abs_predicted_pair_delta_k": _max_abs(predicted_delta[:, index]),
                "pair_delta_rmse_k": _rmse(delta_error),
                "pair_delta_max_abs_error_k": _max_abs(delta_error),
                "mesh_difference_k": uncertainty[sensor],
                "radiation_effect_over_mesh_difference": max_effect / uncertainty[sensor],
                "terminal_predicted_to_truth_delta_ratio_abs": float(
                    abs(predicted_delta[-1, index] / truth_delta[-1, index])
                ),
                "terminal_direction_matches": bool(
                    np.sign(predicted_delta[-1, index]) == np.sign(truth_delta[-1, index])
                ),
            }
        )
    return pd.DataFrame(rows)


def _truth_leakage_check(frames: dict[str, pd.DataFrame], model: ThermalRCModel) -> bool:
    for case_id, source in frames.items():
        baseline = _predict(model, case_id, source)
        altered = source.copy()
        truth_columns = [column for column in altered if column.startswith("truth_")]
        altered[truth_columns] = altered[truth_columns] + 10_000.0
        if not np.array_equal(baseline, _predict(model, case_id, altered)):
            return False
    return True


def _input_coverage(
    frames: dict[str, pd.DataFrame], metadata: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    train_ranges = metadata["train_control_ranges"]
    result: dict[str, dict[str, Any]] = {}
    for control in CONTROLS:
        values = np.concatenate(
            [frame[control].to_numpy(dtype=np.float64) for frame in frames.values()]
        )
        evaluation_range = [float(values.min()), float(values.max())]
        training_range = [float(value) for value in train_ranges[control]]
        result[control] = {
            "training_range": training_range,
            "evaluation_range": evaluation_range,
            "within_training_range": bool(
                evaluation_range[0] >= training_range[0]
                and evaluation_range[1] <= training_range[1]
            ),
        }
    return result


def _is_control_coupled(model: ThermalRCModel, control: str) -> bool:
    edge_coupled = any(
        getattr(item.conductance, "control", None) == control for item in model.spec.edges
    )
    source_coupled = any(
        getattr(item.heat_rate, "control", None) == control for item in model.spec.sources
    )
    boundary_coupled = any(
        item.reservoir_temperature.control == control
        or getattr(item.conductance, "control", None) == control
        for item in model.spec.boundaries
    )
    return edge_coupled or source_coupled or boundary_coupled


def _summary(
    run_dir: Path,
    artifact,
    frames: dict[str, pd.DataFrame],
    cases: pd.DataFrame,
    sensors: pd.DataFrame,
    phases: pd.DataFrame,
    pair: pd.DataFrame,
    quality: dict[str, Any],
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    training_metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    split = pd.read_csv(run_dir / "split.csv")
    leakage_free = _truth_leakage_check(frames, artifact.model)
    controls_complete = artifact.control_names == CONTROLS
    pair_aligned = _validate_pair(frames)
    workflow_pass = bool(
        controls_complete
        and leakage_free
        and pair_aligned
        and quality["dynamic_reference_matches_hv02"]
    )
    rms_model_error = bool(sensors["model_error_resolved_above_mesh_difference"].any())
    pointwise_model_error = bool(sensors["pointwise_error_resolved_above_mesh_difference"].any())
    if rms_model_error:
        adequacy = "sensor_rmse_resolved_above_mesh_difference"
    elif pointwise_model_error:
        adequacy = "pointwise_error_resolved_but_sensor_rmse_within_mesh_difference"
    else:
        adequacy = "not_resolved_beyond_mesh_difference"
    strict_qualified = bool(
        quality["mesh_qualified"]
        and quality["temporal_qualified"]
        and quality["experiment_validated"]
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "External open-loop evaluation of the flow-dependent three-node thermal network "
            "against local-mesh conjugate-flow CAE, with radiation as a paired model gap."
        ),
        "data_quality": {
            **quality,
            "pair_time_and_controls_identical": pair_aligned,
            "strict_validation_status": "qualified" if strict_qualified else "not_qualified",
            "permitted_use": (
                "validated engineering evaluation" if strict_qualified else "model-form screening"
            ),
        },
        "workflow": {
            "status": "pass" if workflow_pass else "fail",
            "artifact_controls": list(artifact.control_names),
            "complete_external_control_boundary": controls_complete,
            "forecast_truth_is_not_an_input": leakage_free,
            "all_predictions_finite": bool(
                all(np.isfinite(value).all() for value in predictions.values())
            ),
        },
        "model": {
            "type": "three-node thermal network using shared scalar response laws",
            "inlet_air_velocity_coupled_to_physics": _is_control_coupled(
                artifact.model, "inlet_air_velocity"
            ),
            "fitted_parameters": fitted_parameters(artifact.model),
            "input_coverage": _input_coverage(frames, artifact.metadata),
        },
        "training": {
            "data": "data/nonlinear/train/*.csv",
            "split_assignment": _records(split[["case_id", "split"]]),
            "best_epoch": artifact.metadata["training"]["best_epoch"],
            "best_causal_validation_rmse_k": artifact.metadata["training"][
                "best_causal_validation_rmse"
            ],
            "metrics": training_metrics,
        },
        "evaluation": {
            "cases": _records(cases),
            "phases": _records(phases),
            "mean_case_rmse_k": float(cases["rmse_k"].mean()),
            "worst_case": str(cases.loc[cases["rmse_k"].idxmax(), "case_id"]),
            "worst_case_rmse_k": float(cases["rmse_k"].max()),
            "mean_prior_rmse_k": float(cases["prior_rmse_k"].mean()),
            "model_adequacy_status": adequacy,
        },
        "radiation_pair": {
            "sensors": _records(pair),
            "max_abs_truth_effect_k": float(pair["max_abs_truth_radiation_delta_k"].max()),
            "max_effect_over_mesh_difference": float(
                pair["radiation_effect_over_mesh_difference"].max()
            ),
            "interpretation": "directional screening only",
        },
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], decimals: int = 4) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for record in frame[columns].to_dict(orient="records"):
        values = [
            f"{value:.{decimals}f}" if isinstance(value, float) else str(value)
            for value in record.values()
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def _report(
    summary: dict[str, Any],
    cases: pd.DataFrame,
    sensors: pd.DataFrame,
    phases: pd.DataFrame,
    pair: pd.DataFrame,
) -> str:
    case_table = _markdown_table(
        cases,
        [
            "case_id",
            "rmse_k",
            "prior_rmse_k",
            "rms_error_over_mesh_difference",
            "max_abs_error_k",
            "max_abs_error_over_mesh_difference",
            "max_hotspot_underprediction_k",
        ],
    )
    sensor_table = _markdown_table(
        sensors,
        [
            "case_id",
            "sensor",
            "rmse_k",
            "bias_k",
            "mesh_difference_k",
            "rmse_over_mesh_difference",
            "max_abs_error_over_mesh_difference",
        ],
    )
    phase_table = _markdown_table(
        phases,
        ["case_id", "phase", "n_timepoints", "rmse_k", "max_abs_error_k"],
    )
    pair_table = _markdown_table(
        pair,
        [
            "sensor",
            "truth_terminal_radiation_delta_k",
            "predicted_terminal_pair_delta_k",
            "pair_delta_rmse_k",
            "radiation_effect_over_mesh_difference",
            "terminal_predicted_to_truth_delta_ratio_abs",
        ],
    )
    evaluation = summary["evaluation"]
    workflow = summary["workflow"]
    quality = summary["data_quality"]
    velocity_coupled = summary["model"]["inlet_air_velocity_coupled_to_physics"]
    radiation_effect = summary["radiation_pair"]["max_abs_truth_effect_k"]
    radiation_ratio = summary["radiation_pair"]["max_effect_over_mesh_difference"]
    radiation_capture = float(pair["terminal_predicted_to_truth_delta_ratio_abs"].max())
    training = summary["training"]["metrics"]
    training_table = _markdown_table(
        pd.DataFrame(
            [
                {"split": split, **metrics}
                for split, metrics in summary["training"]["metrics"].items()
            ]
        ),
        ["split", "n_cases", "mean_case_causal_rmse", "worst_case_causal_rmse"],
    )
    heldout_case = next(
        row["case_id"] for row in summary["training"]["split_assignment"] if row["split"] == "test"
    )
    worst_point_error = float(cases["max_abs_error_k"].max())
    worst_point_ratio = float(cases["max_abs_error_over_mesh_difference"].max())
    resolution_statement = (
        "一部時刻のモデル差はmesh差を超えます。"
        if worst_point_ratio > 1.0
        else "最大点誤差も保守的なmesh差を下回り、現データではモデル差を数値差から分離できません。"
    )
    return f"""# 高忠実度非線形CAEベンチマーク結果

## 結論

公開の学習・forecast workflowは正常完走し、予測値はfinite、時刻・入力はCAEと一致し、
`truth_*`を変更しても予測が変わらないことを確認しました。workflow判定は
**{workflow["status"]}**です。

一方、局所mesh pairに対する流速依存3-node thermal networkのcase平均RMSEは
**{evaluation["mean_case_rmse_k"]:.4f} K**、worstは
**{evaluation["worst_case_rmse_k"]:.4f} K**でした。判定は
**{evaluation["model_adequacy_status"]}**です。入口流速は評価境界に保持されていますが、
共通power-law scalar lawを持つboundary conductanceへ結合されています
(`{velocity_coupled}`)。残差は、固定係数ではなく
この最小流速依存モデルで共役流れをどこまで集約できるかを示します。

## 同定データ内のscreening

{training_table}

holdout `{heldout_case}` の因果RMSEは
**{training["test"]["mean_case_causal_rmse"]:.4f} K**でした。
これはcold/high-flowの組合せ汎化に対する最小thermal networkのscreening evidenceです。
ただし、この10本はglobal-8 meshであり設計精度の絶対誤差判定には使いません。

## ケース別

{case_table}

case RMSEは保守的な隣接mesh差内ですが、最大点誤差は **{worst_point_error:.4f} K**、
mesh差に対する最大比は **{worst_point_ratio:.3f}**でした。したがって平均精度はmesh不確かさから
分離できず、{resolution_statement}これは製品合否閾値ではありません。

## センサ別

{sensor_table}

## 応答区間別

{phase_table}

区間はleft-ZOHに合わせ、行のcommandが次の区間を駆動した後の応答時刻で集計しています。初期化から
power、airflow、hot/low-flow複合条件へ進むにつれて、モデル誤差が増える箇所を分離しています。

## 放射ペア

{pair_table}

放射による最大温度差は **{radiation_effect:.4f} K**、隣接mesh差に対する最大比は
**{radiation_ratio:.3f}**です。放射項を持たない現行thermal networkで終端まで残った
初期差由来の予測pair差は、
真の終端差の最大 **{radiation_capture:.3%}**に留まりました。CAE pairから放射の方向性は
確認できますが、mesh収束した
効果量の確定や現行モデルによる放射応答の再現はできません。

## 利用境界

- local-medium meshはbenchmark用途には合格: `{quality["benchmark_qualified"]}`
- 厳格mesh収束: `{quality["mesh_qualified"]}`
- 独立時間刻み収束: `{quality["temporal_qualified"]}`
- 同一境界の実験比較実施: `{quality["experiment_compared"]}`
- 同一境界の実験妥当化: `{quality["experiment_validated"]}`

本結果はmodel-form screeningには利用できますが、絶対温度保証、hotspot安全判定、製品設計認証には
利用できません。本結果だけを根拠に空気状態や放射項をcoreへ追加しません。次の基盤評価は、目的対象である
半導体製造装置のwafer/chuck・stage・coolant・process入力を同じ外部評価境界で扱うCAE/実験datasetで
行います。本ケース側では過渡時間刻み収束と実測値による妥当化が未完です。
"""


def run_benchmark(config_path: Path, output: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parent
    cfg = load_config(config_path)
    frames = _load_cases(root)
    quality = _reference_quality(root, frames)
    run_dir, forecast_dirs = _run_workflows(cfg, config_path)
    artifact = load_artifact(run_dir / "artifact")
    prior = ThermalRCModel(
        load_system_spec(as_path(str(cfg["system"]), root)),
        integrator=str(cfg.get("engine", {}).get("integrator", "exact")),
    )

    predictions: dict[str, np.ndarray] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    case_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    uncertainty = quality["mesh_difference_k"]
    for case_id, source in frames.items():
        predicted = _saved_prediction(case_id, source, forecast_dirs[case_id], artifact.model)
        prior_prediction = _predict(prior, case_id, source)
        truth = source[[f"truth_{name}" for name in SENSORS]].to_numpy(dtype=np.float64)
        predictions[case_id] = predicted
        prediction_frames[case_id] = _prediction_frame(
            source, predicted, prior_prediction, uncertainty
        )
        case_rows.append(
            _case_row(case_id, source, truth, predicted, prior_prediction, uncertainty)
        )
        sensor_rows.extend(_sensor_rows(case_id, truth, predicted, prior_prediction, uncertainty))
        phase_rows.extend(_phase_rows(case_id, source, truth, predicted))

    cases = pd.DataFrame(case_rows)
    sensors = pd.DataFrame(sensor_rows)
    phases = pd.DataFrame(phase_rows)
    pair = _pair_rows(frames, predictions, uncertainty)
    summary = _summary(
        run_dir,
        artifact,
        frames,
        cases,
        sensors,
        phases,
        pair,
        quality,
        predictions,
    )
    with staged_output_directory(output.resolve(), overwrite=True) as target:
        cases.to_csv(
            target / "case_metrics.csv", index=False, float_format="%.10g", lineterminator="\n"
        )
        sensors.to_csv(
            target / "sensor_metrics.csv", index=False, float_format="%.10g", lineterminator="\n"
        )
        phases.to_csv(
            target / "phase_metrics.csv", index=False, float_format="%.10g", lineterminator="\n"
        )
        pair.to_csv(
            target / "radiation_pair_metrics.csv",
            index=False,
            float_format="%.10g",
            lineterminator="\n",
        )
        prediction_dir = target / "predictions"
        prediction_dir.mkdir()
        for case_id, frame in prediction_frames.items():
            frame.to_csv(
                prediction_dir / f"{case_id}.csv",
                index=False,
                float_format="%.10g",
                lineterminator="\n",
            )
        (target / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
            newline="\n",
        )
        (target / "report.md").write_text(
            _report(summary, cases, sensors, phases, pair), encoding="utf-8", newline="\n"
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_root / "high_fidelity_benchmark.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "data/nonlinear_high_fidelity/benchmark",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_benchmark(args.config, args.output)
    print(json.dumps(summary["workflow"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["evaluation"], indent=2, ensure_ascii=False))
    print(f"saved benchmark: {args.output.resolve()}")
    return 0 if summary["workflow"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
