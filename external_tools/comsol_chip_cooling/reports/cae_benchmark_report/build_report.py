# ruff: noqa: E501, RUF001, RUF007, S608
"""Build the self-contained technical report for all COMSOL CAE benchmark cases.

The script does not rerun COMSOL or the celltemp workflows.  It normalizes the
already generated CAE datasets and the latest benchmark evaluation outputs into
bounded, reviewable tables, then writes the canonical Data Analytics artifact
consumed by the portable report builder.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = REPORT_DIR / "source_data"
REPO_ROOT = Path(__file__).resolve().parents[4]
TOOL_ROOT = REPO_ROOT / "external_tools" / "comsol_chip_cooling"

LINEAR_QA = TOOL_ROOT / "data" / "qa_summary.csv"
LINEAR_EVAL = TOOL_ROOT / "work" / "evaluation"
LINEAR_TRAIN_RUN = TOOL_ROOT / "work" / "outputs" / "runs" / "comsol_chip_cooling"
NONLINEAR_QA = TOOL_ROOT / "data" / "nonlinear" / "qa_summary.csv"
NONLINEAR_RADIATION = TOOL_ROOT / "data" / "nonlinear" / "radiation_pairs.csv"
NONLINEAR_RUN = (
    TOOL_ROOT / "work" / "high_fidelity_benchmark" / "runs" / "nonlinear_high_fidelity_network"
)
HF_ROOT = TOOL_ROOT / "data" / "nonlinear_high_fidelity"
HF_BENCHMARK = HF_ROOT / "benchmark"

GENERATED_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
TITLE = "CAEベンチマークケース技術レポート"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"required report input is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty report dataset: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    return float(value)


def as_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    return int(float(value))


def numeric_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        output: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or value == "":
                output[key] = None
                continue
            text = str(value).strip()
            lower = text.lower()
            if lower == "true":
                output[key] = True
            elif lower == "false":
                output[key] = False
            else:
                try:
                    number = float(text)
                    output[key] = int(number) if number.is_integer() else number
                except ValueError:
                    output[key] = text
        converted.append(output)
    return converted


def compact_number(value: float, digits: int = 3) -> str:
    rounded = round(float(value), digits)
    return f"{rounded:.{digits}f}".rstrip("0").rstrip(".")


def field_range(rows: list[dict[str, str]], field: str, digits: int) -> str:
    values = [float(row[field]) for row in rows if row.get(field) not in {None, ""}]
    low, high = min(values), max(values)
    if abs(high - low) < 10 ** (-(digits + 1)):
        return compact_number(low, digits)
    return f"{compact_number(low, digits)}–{compact_number(high, digits)}"


def time_step_summary(rows: list[dict[str, str]]) -> str:
    times = [float(row["time"]) for row in rows]
    steps = sorted({round(right - left, 8) for left, right in zip(times, times[1:])})
    if len(steps) == 1:
        step_text = compact_number(steps[0], 3)
    else:
        step_text = "/".join(compact_number(value, 3) for value in steps)
    return f"{compact_number(times[-1], 1)} s, Δt {step_text} s"


def input_summary(data_path: Path, *, nonlinear: bool, initialization: bool = False) -> str:
    rows = read_csv(data_path)
    parts = [
        f"P {field_range(rows, 'chip_power', 2)} W",
        f"Tin {field_range(rows, 'coolant_temperature', 2)} °C",
    ]
    if nonlinear:
        parts.append(f"v {field_range(rows, 'inlet_air_velocity', 3)} m/s")
    if initialization:
        initial = rows[0].get("truth_chip") or rows[0].get("chip")
        if initial not in {None, ""}:
            parts.append(f"Tchip,0 {compact_number(float(initial), 2)} °C")
    parts.append(time_step_summary(rows))
    return "; ".join(parts)


ROLE_LABELS = {
    "train": "同定",
    "forecast": "外部予測",
    "monitor": "監視",
    "model_gap": "モデル差",
    "mesh_convergence": "メッシュ収束",
}

SPLIT_LABELS = {"train": "学習", "val": "検証", "test": "内部テスト"}

LINEAR_GROUP_TARGETS = {
    "power_steps": "入熱利得、chip熱容量、内部時定数のレベル一貫性",
    "coolant_steps": "発熱経路と独立した境界温度・対流経路の分離",
    "ramps": "step波形だけに依存しない連続入力追従",
    "multilevel": "複数運転レベル・複数時間スケールでの係数一貫性",
    "combined": "発熱と境界温度の重ね合わせ、および係数分離",
    "initialization": "単一初期温度への依存回避と初期状態からの回復",
    "interpolation": "学習範囲内の未学習レベル・同時入力の内挿",
    "extrapolation": "学習範囲外での符号、安定性、誤差増幅",
    "dynamics": "短周期モードまたは未知recipeへの時系列汎化",
    "sampling": "再サンプリングなしの可変時間刻み積分",
}

LINEAR_CASE_TARGETS = {
    "F01_power_interpolation": "未学習7 Wでの発熱内挿",
    "F02_joint_interpolation": "発熱・境界温度の同時内挿と重ね合わせ",
    "F03_power_extrapolation": "学習上限12 Wを超える16 W hot-side外挿",
    "F04_coolant_extrapolation": "学習下限15 °Cを下回る10 °C cold-side外挿",
    "F05_short_pulses": "反復短パルスで3-node縮約が落とす高速モード",
    "F06_unseen_recipe": "未知の入力順序・重なりに対するrecipe汎化",
    "F07_hot_initial_state": "未学習50 °C一様初期状態からの冷却・再加熱",
    "F08_variable_sampling": "1/2/3秒の非均一時刻を直接積分できるか",
    "M01_noise_baseline": "0.15 K測定noise下の正常innovationとfalse alert",
    "M02_sensor_drift": "slow sensor driftと物理温度の分離",
    "M03_sensor_offset": "+3 K chip sensor stepの検出とbias帰属",
    "M04_missing_measurements": "base/fins欠測窓でも状態推定を継続できるか",
    "M05_uncommanded_heat": "commandにない3 W chip発熱の検出と熱外乱帰属",
}

NONLINEAR_CASE_TARGETS = {
    "NT01_power_levels": "発熱レベル依存の熱抵抗・時定数を同定",
    "NT02_power_reverse": "高温後の冷却と再加熱の非対称性を分離",
    "NT03_inlet_temperature_levels": "入口温度レベルに対する空気側応答を同定",
    "NT04_air_velocity_levels": "入口流速で変化する冷却conductanceを同定",
    "NT05_power_ramp": "step形状に依存しない連続非線形応答を同定",
    "NT06_combined_levels": "3入力の非同期変化から効果を分離",
    "NT07_combined_ramps": "運転域内部を連続3入力で被覆",
    "NT08_hot_low_flow": "高温・低流量の学習端を被覆",
    "NT09_cold_high_flow": "低温・高流量の学習端を被覆しheld-out評価",
    "NT10_stationary_hot_start": "非一様な発熱定常場からの初期化を評価",
    "NF01_fixed_flow_interpolation": "固定流速で発熱・入口温度を内挿",
    "NF02_unseen_three_input_recipe": "未知の3入力順序・重なりへのrecipe汎化",
    "NF03_power_extrapolation": "学習上限12 Wを超える16 W外挿",
    "NF04_hot_low_flow_corner": "40 °C・0.05 m/s・12 Wの複合外挿",
    "NF05_cold_inlet_extrapolation": "学習下限を下回る入口10 °C外挿",
    "NF06_air_velocity_interpolation": "未学習0.075/0.15 m/sの流速内挿",
    "NF07_air_velocity_extrapolation": "0.30 m/sへの流速外挿",
    "NF08_short_power_pulses": "20秒・15 W反復pulseで欠落fast modeを露出",
    "NF09_stationary_hot_start": "未学習の非一様12 W定常場から予測",
    "NM01_nominal_nonlinear": "非線形通常運転でのfalse alertを評価",
    "NM02_uncommanded_heat": "420–520秒の未指令3 W発熱を検出",
    "NM03_uncommanded_cooling_loss": "commandにない一時流量低下を検出",
    "NR01_power_levels": "発熱レベルごとの放射省略差を同一入力pairで測定",
    "NR02_power_ramp": "連続温度sweep中の放射省略差を測定",
    "NR03_combined_levels": "3入力運転中の放射省略差を測定",
    "NR04_hot_low_flow_corner": "弱対流・高温端での放射省略差を測定",
    "NR05_stationary_hot_start": "非一様hot-start冷却中の放射省略差を測定",
}


def build_linear_catalog() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    qa = numeric_rows(read_csv(LINEAR_QA))
    forecast = numeric_rows(read_csv(LINEAR_EVAL / "forecast_cases.csv"))
    monitor = numeric_rows(read_csv(LINEAR_EVAL / "monitor_cases.csv"))
    events = numeric_rows(read_csv(LINEAR_EVAL / "monitor_events.csv"))
    training = numeric_rows(read_csv(LINEAR_TRAIN_RUN / "metrics_by_case.csv"))

    forecast_by_id = {row["case_id"]: row for row in forecast}
    monitor_by_id = {row["case_id"]: row for row in monitor}
    event_by_id = {row["case_id"]: row for row in events}
    training_by_id = {row["case_id"]: row for row in training}

    monitor_output: list[dict[str, Any]] = []
    for row in monitor:
        event = event_by_id.get(row["case_id"], {})
        monitor_output.append(
            {
                **row,
                "detected": event.get("detected"),
                "detection_delay_s": event.get("detection_delay_s"),
                "false_alerts_before_event": event.get("false_alerts_before_event"),
                "alert_rate_percent": 100.0 * float(row["alert_rate"]),
            }
        )

    catalog: list[dict[str, Any]] = []
    for row in qa:
        case_id = str(row["case_id"])
        role = str(row["role"])
        group = str(row["group"])
        data_path = TOOL_ROOT / str(row["path"])
        target = LINEAR_CASE_TARGETS.get(
            case_id, LINEAR_GROUP_TARGETS.get(group, str(row["purpose"]))
        )
        if role == "train":
            metric = training_by_id[case_id]
            result = (
                f"{SPLIT_LABELS[str(metric['split'])]} RMSE {float(metric['rmse']):.3f} K; "
                f"CAE Tmax {float(row['temperature_max_c']):.1f} °C"
            )
        elif role == "forecast":
            metric = forecast_by_id[case_id]
            result = (
                f"外部RMSE {float(metric['rmse_k']):.3f} K; "
                f"最大誤差 {float(metric['max_abs_error_k']):.3f} K; "
                f"prior比 {float(metric['rmse_improvement_percent']):.1f}%改善"
            )
        else:
            metric = monitor_by_id[case_id]
            if case_id == "M01_noise_baseline":
                result = (
                    f"alert率 {100.0 * float(metric['alert_rate']):.1f}%; "
                    f"posterior物理RMSE {float(metric['posterior_physical_rmse_k']):.3f} K"
                )
            elif case_id == "M02_sensor_drift":
                result = (
                    f"alert率 {100.0 * float(metric['alert_rate']):.1f}%; "
                    f"最終bias誤差 {float(metric['max_final_sensor_bias_error_k']):.3f} K"
                )
            elif case_id == "M03_sensor_offset":
                event = event_by_id[case_id]
                result = (
                    f"検出遅延 {float(event['detection_delay_s']):.0f} s; "
                    f"alert率 {100.0 * float(metric['alert_rate']):.1f}%; "
                    f"最終bias誤差 {float(metric['max_final_sensor_bias_error_k']):.3f} K"
                )
            elif case_id == "M04_missing_measurements":
                result = (
                    f"欠測 {int(metric['missing_measurements'])}点; "
                    f"欠測窓RMSE {float(metric['missing_posterior_rmse_k']):.3f} K; 推定finite"
                )
            else:
                event = event_by_id[case_id]
                result = (
                    f"検出遅延 {float(event['detection_delay_s']):.0f} s; "
                    f"peak外乱 {float(metric['peak_event_chip_disturbance_w']):.3f} W; "
                    f"alert率 {100.0 * float(metric['alert_rate']):.1f}%"
                )
        catalog.append(
            {
                "layer": "線形CAE",
                "case_id": case_id,
                "role": ROLE_LABELS[role],
                "problem": input_summary(
                    data_path,
                    nonlinear=False,
                    initialization=group == "initialization",
                ),
                "evaluation_target": target,
                "result": result,
                "source_path": str(row["path"]),
            }
        )
    return catalog, forecast, monitor_output


def build_nonlinear_catalog() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    qa = numeric_rows(read_csv(NONLINEAR_QA))
    radiation = numeric_rows(read_csv(NONLINEAR_RADIATION))
    model_metrics = numeric_rows(read_csv(NONLINEAR_RUN / "metrics_by_case.csv"))
    radiation_by_id = {row["radiation_case_id"]: row for row in radiation}
    metrics_by_id = {row["case_id"]: row for row in model_metrics}

    catalog: list[dict[str, Any]] = []
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qa:
        role = str(row["role"])
        by_role[role].append(row)
        case_id = str(row["case_id"])
        if role == "train":
            metric = metrics_by_id[case_id]
            result = (
                f"{SPLIT_LABELS[str(metric['split'])]} RMSE {float(metric['rmse']):.3f} K; "
                f"Tchip,max {float(row['chip_max_c']):.1f} °C; "
                f"dp,max {float(row['pressure_drop_max_pa']):.4f} Pa"
            )
        elif role == "forecast":
            result = (
                f"CAE truth生成済み; Tchip,max {float(row['chip_max_c']):.1f} °C; "
                "現行成果物では独立open-loop誤差を未集計"
            )
        elif role == "monitor":
            disturbance = {
                "NM01_nominal_nonlinear": "正常noise系列",
                "NM02_uncommanded_heat": "未指令3 W発熱を注入",
                "NM03_uncommanded_cooling_loss": "未指令流量低下を注入",
            }[case_id]
            result = f"CAE/0.15 K noise生成済み; {disturbance}; 監視器の検出結果は未集計"
        else:
            pair = radiation_by_id[case_id]
            result = (
                f"base比 terminal ΔTchip {float(pair['chip_delta_final_c']):.2f} K; "
                f"max|Qrad| {float(pair['radiative_heat_rate_max_abs_w']):.3f} W"
            )

        catalog.append(
            {
                "layer": "非線形CAE",
                "case_id": case_id,
                "role": ROLE_LABELS[role],
                "problem": input_summary(
                    TOOL_ROOT / str(row["path"]),
                    nonlinear=True,
                    initialization="stationary_hot_start" in case_id,
                ),
                "evaluation_target": NONLINEAR_CASE_TARGETS[case_id],
                "result": result,
                "source_path": str(row["path"]),
            }
        )

    envelope: list[dict[str, Any]] = []
    for role in ("train", "forecast", "monitor", "model_gap"):
        rows = by_role[role]
        worst = max(rows, key=lambda item: float(item["chip_max_c"]))
        envelope.append(
            {
                "role": ROLE_LABELS[role],
                "cases": len(rows),
                "max_chip_c": float(worst["chip_max_c"]),
                "max_chip_case": worst["case_id"],
                "max_pressure_drop_pa": max(float(item["pressure_drop_max_pa"]) for item in rows),
                "max_energy_residual_w": max(
                    float(item["energy_residual_max_abs_w"]) for item in rows
                ),
            }
        )
    radiation_chart = [
        {
            **row,
            "pair": f"{row['radiation_case_id']} / {row['base_case_id']}",
            "abs_terminal_chip_delta_k": abs(float(row["chip_delta_final_c"])),
        }
        for row in radiation
    ]
    return catalog, envelope, radiation_chart, model_metrics


def build_mesh_catalog() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    convergence = numeric_rows(read_csv(HF_ROOT / "mesh_convergence.csv"))
    deltas = numeric_rows(read_csv(HF_ROOT / "mesh_convergence_deltas.csv"))
    adjacent = {
        row["case_id"]: row
        for row in deltas
        if row["mesh_profile"] == "local-medium" and row["finer_mesh_profile"] == "local-fine"
    }
    settings = {
        "MC01_nominal": (
            "12 W; Tin 25 °C; v 0.10 m/s; 放射なし; 4 mesh profile",
            "定格熱抵抗、fin間流路、圧力損失の離散化感度",
        ),
        "MC02_hot_low_flow_radiation": (
            "12 W; Tin 40 °C; v 0.05 m/s; 放射あり; 4 mesh profile",
            "高温・低流量・弱対流・放射条件での離散化感度",
        ),
    }
    catalog: list[dict[str, Any]] = []
    for case_id, (problem, target) in settings.items():
        row = adjacent[case_id]
        strict = bool(row["passes_qoi_tolerances"])
        benchmark = bool(row["passes_benchmark_tolerances"])
        catalog.append(
            {
                "layer": "局所メッシュ",
                "case_id": case_id,
                "role": "メッシュ収束",
                "problem": problem,
                "evaluation_target": target,
                "result": (
                    f"medium→fine ΔTchip,avg {float(row['chip_average_c_absolute_delta']):.3f} K; "
                    f"Δdp {100.0 * float(row['pressure_drop_relative_delta']):.2f}%; "
                    f"strict {'合格' if strict else '未達'} / benchmark {'合格' if benchmark else '未達'}"
                ),
                "source_path": "data/nonlinear_high_fidelity/mesh_convergence_deltas.csv",
            }
        )
    profile_order = {"global-8": 1, "local-coarse": 2, "local-medium": 3, "local-fine": 4}
    chart_rows = sorted(
        (
            {
                **row,
                "mesh_label": {
                    "global-8": "global-8",
                    "local-coarse": "local coarse",
                    "local-medium": "local medium",
                    "local-fine": "local fine",
                }[str(row["mesh_profile"])],
                "case_label": "MC01 nominal"
                if row["case_id"] == "MC01_nominal"
                else "MC02 hot/low-flow + radiation",
                "profile_order": profile_order[str(row["mesh_profile"])],
            }
            for row in convergence
        ),
        key=lambda item: (item["profile_order"], item["case_id"]),
    )
    return catalog, chart_rows


def build_high_fidelity_catalog() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    qa = numeric_rows(read_csv(HF_ROOT / "dynamic" / "qa_summary.csv"))
    metrics = numeric_rows(read_csv(HF_BENCHMARK / "case_metrics.csv"))
    metrics_by_id = {row["case_id"]: row for row in metrics}

    catalog: list[dict[str, Any]] = []
    for row in qa:
        case_id = str(row["case_id"])
        metric = metrics_by_id[case_id]
        target = (
            "局所mesh上で発熱→流速低下→入口昇温の複合非線形予測"
            if case_id == "HV01_composite_conjugate"
            else "HV01と同一入力で放射だけを追加しmodel-form差を分離"
        )
        catalog.append(
            {
                "layer": "高忠実度過渡",
                "case_id": case_id,
                "role": ROLE_LABELS[str(row["role"])],
                "problem": input_summary(TOOL_ROOT / str(row["path"]), nonlinear=True),
                "evaluation_target": target,
                "result": (
                    f"RMSE {float(metric['rmse_k']):.3f} K; "
                    f"最大誤差 {float(metric['max_abs_error_k']):.3f} K; "
                    f"mesh差内 {100.0 * float(metric['fraction_abs_error_within_mesh_difference']):.0f}%"
                ),
                "source_path": str(row["path"]),
            }
        )

    error_comparison: list[dict[str, Any]] = []
    for row in metrics:
        mesh_difference = float(row["rmse_k"]) / float(row["rms_error_over_mesh_difference"])
        label = "HV01 共役熱流動" if row["case_id"] == "HV01_composite_conjugate" else "HV02 + 放射"
        for metric_label, value in (
            ("RMSE", float(row["rmse_k"])),
            ("最大絶対誤差", float(row["max_abs_error_k"])),
            ("保守的隣接mesh差", mesh_difference),
        ):
            error_comparison.append(
                {
                    "case_id": row["case_id"],
                    "case_label": label,
                    "metric": metric_label,
                    "value_k": value,
                    "prior_rmse_k": float(row["prior_rmse_k"]),
                    "rmse_improvement_percent": float(row["rmse_improvement_percent"]),
                }
            )

    predictions: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = {}
    for case_id in ("HV01_composite_conjugate", "HV02_composite_radiation"):
        rows = numeric_rows(read_csv(HF_BENCHMARK / "predictions" / f"{case_id}.csv"))
        enriched = [{"case_id": case_id, **row} for row in rows]
        predictions.extend(enriched)
        by_case[case_id] = enriched

    radiation_trace: list[dict[str, Any]] = []
    base = {float(row["time"]): row for row in by_case["HV01_composite_conjugate"]}
    for row in by_case["HV02_composite_radiation"]:
        reference = base[float(row["time"])]
        radiation_trace.append(
            {
                "time": float(row["time"]),
                "truth_delta_chip_k": float(row["truth_chip"]) - float(reference["truth_chip"]),
                "predicted_delta_chip_k": float(row["predicted_chip"])
                - float(reference["predicted_chip"]),
                "mesh_difference_chip_k": float(row["mesh_difference_chip"]),
            }
        )
    return catalog, metrics, error_comparison, predictions, radiation_trace


def source(
    source_id: str,
    label: str,
    filename: str,
    description: str,
    *,
    sql: str | None = None,
    filters: list[str] | None = None,
    definitions: list[str] | None = None,
) -> dict[str, Any]:
    relative = (
        f"external_tools/comsol_chip_cooling/reports/cae_benchmark_report/source_data/{filename}"
    )
    query_sql = sql or f"SELECT * FROM read_csv_auto('{relative}', header = true)"
    return {
        "id": source_id,
        "label": label,
        "path": relative,
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "description": description,
            "sql": query_sql,
            "tables_used": [relative],
            "filters": filters or ["レポート生成時点で確認済みの全行"],
            "metric_definitions": definitions or [],
            "executed_at": GENERATED_AT,
        },
    }


def document_source(source_id: str, label: str, path: str) -> dict[str, Any]:
    return {"id": source_id, "label": label, "path": path}


def chart(
    chart_id: str,
    title: str,
    subtitle: str,
    chart_type: str,
    dataset: str,
    source_id: str,
    x: dict[str, Any],
    y: dict[str, Any],
    *,
    color: dict[str, Any] | None = None,
    intent: str = "comparison",
    y_axis_title: str | None = None,
    unit: str | None = None,
    reference_lines: list[dict[str, Any]] | None = None,
    combination_rationale: str | None = None,
    palette_kind: str = "sequential",
) -> dict[str, Any]:
    encodings: dict[str, Any] = {"x": x, "y": y}
    if color:
        encodings["color"] = color
    output: dict[str, Any] = {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "showDescription": True,
        "intent": intent,
        "type": chart_type,
        "dataset": dataset,
        "sourceId": source_id,
        "encodings": encodings,
        "layout": "full",
        "valueFormat": "number",
        "palette": {"kind": palette_kind},
        "surface": {"surface": "card", "viewMode": "both"},
    }
    if y_axis_title:
        output["yAxisTitle"] = y_axis_title
    if unit:
        output["unit"] = unit
    if reference_lines:
        output["referenceLines"] = reference_lines
    if combination_rationale:
        output["combinationRationale"] = combination_rationale
    return output


def table(
    table_id: str,
    title: str,
    subtitle: str,
    dataset: str,
    source_id: str,
    columns: list[tuple[str, str, str]],
    *,
    sort_field: str = "case_id",
) -> dict[str, Any]:
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "showDescription": True,
        "dataset": dataset,
        "sourceId": source_id,
        "density": "compact",
        "layout": "full",
        "defaultSort": {"field": sort_field, "direction": "asc"},
        "columns": [
            {"field": field, "label": label, "type": data_type}
            for field, label, data_type in columns
        ],
    }


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    linear_catalog, linear_forecast, linear_monitor = build_linear_catalog()
    nonlinear_catalog, nonlinear_envelope, radiation_pairs, nonlinear_metrics = (
        build_nonlinear_catalog()
    )
    mesh_catalog, mesh_chart = build_mesh_catalog()
    hf_catalog, hf_metrics, hf_error, hf_predictions, hf_radiation_trace = (
        build_high_fidelity_catalog()
    )

    all_cases = [*linear_catalog, *nonlinear_catalog, *mesh_catalog, *hf_catalog]
    if (
        len(linear_catalog) != 29
        or len(nonlinear_catalog) != 27
        or len(mesh_catalog) != 2
        or len(hf_catalog) != 2
    ):
        raise ValueError(
            "unexpected case inventory: "
            f"linear={len(linear_catalog)}, nonlinear={len(nonlinear_catalog)}, "
            f"mesh={len(mesh_catalog)}, high_fidelity={len(hf_catalog)}"
        )
    if len(all_cases) != 60:
        raise ValueError(f"expected 60 reader-facing case IDs, found {len(all_cases)}")

    coverage_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in all_cases:
        coverage_counts[(str(row["layer"]), str(row["role"]))] += 1
    coverage = [
        {"layer": layer, "role": role, "cases": count}
        for (layer, role), count in coverage_counts.items()
    ]

    metric_definitions = [
        {
            "metric": "温度RMSE",
            "definition": "初期行を除く評価時刻×3領域平均温度の誤差二乗平均平方根",
            "interpretation": "open-loop温度軌跡の代表誤差。hotspot誤差とは別。",
        },
        {
            "metric": "最大絶対誤差",
            "definition": "評価時刻・対象sensor全体での |予測−CAE| 最大値",
            "interpretation": "局所的な最悪予測ずれ。ただしCAE離散化誤差を含む。",
        },
        {
            "metric": "prior RMSE",
            "definition": "同定前の初期parameterモデルを同じ軌跡へ適用したRMSE",
            "interpretation": "学習による改善を示す比較基準で、ゼロ予測との比較ではない。",
        },
        {
            "metric": "NIS / 検出遅延",
            "definition": "innovation共分散で正規化した二乗残差と、注入開始から最初のalertまでの時間",
            "interpretation": "監視器の異常検出性能。物理外乱とsensor biasの帰属も別途確認。",
        },
        {
            "metric": "隣接mesh差",
            "definition": "同一問題のlocal-mediumとlocal-fine間の評価量絶対差",
            "interpretation": "統計的信頼区間ではなく、離散化不確かさの保守的proxy。",
        },
        {
            "metric": "放射pair差",
            "definition": "同一時刻・同一3入力で radiation − no-radiation を取った温度差",
            "interpretation": "放射省略の方向と大きさをscreening。pair差自体のmesh収束が必要。",
        },
    ]

    hf01 = [row for row in hf_predictions if row["case_id"] == "HV01_composite_conjugate"]
    hf02 = [row for row in hf_predictions if row["case_id"] == "HV02_composite_radiation"]

    output_files = {
        "case_catalog.csv": all_cases,
        "linear_case_catalog.csv": linear_catalog,
        "linear_forecast_metrics.csv": linear_forecast,
        "linear_monitor_metrics.csv": linear_monitor,
        "nonlinear_case_catalog.csv": nonlinear_catalog,
        "nonlinear_role_envelope.csv": nonlinear_envelope,
        "nonlinear_model_metrics.csv": nonlinear_metrics,
        "radiation_pairs.csv": radiation_pairs,
        "mesh_case_catalog.csv": mesh_catalog,
        "mesh_convergence.csv": mesh_chart,
        "high_fidelity_case_catalog.csv": hf_catalog,
        "high_fidelity_case_metrics.csv": hf_metrics,
        "high_fidelity_error_comparison.csv": hf_error,
        "high_fidelity_predictions.csv": hf_predictions,
        "high_fidelity_radiation_trace.csv": hf_radiation_trace,
        "coverage.csv": coverage,
        "metric_definitions.csv": metric_definitions,
    }
    for filename, rows in output_files.items():
        write_csv(SOURCE_DIR / filename, rows)

    linear_summary = {
        "mean_rmse_k": sum(float(row["rmse_k"]) for row in linear_forecast) / len(linear_forecast),
        "worst_rmse_k": max(float(row["rmse_k"]) for row in linear_forecast),
        "worst_case": max(linear_forecast, key=lambda row: float(row["rmse_k"]))["case_id"],
    }
    nonlinear_test = next(row for row in nonlinear_metrics if row["split"] == "test")
    nonlinear_val = next(row for row in nonlinear_metrics if row["split"] == "val")
    hf_summary = {
        "mean_rmse_k": sum(float(row["rmse_k"]) for row in hf_metrics) / len(hf_metrics),
        "worst_rmse_k": max(float(row["rmse_k"]) for row in hf_metrics),
    }
    headline = [
        {"case_ids": 60, "underlying_comsol_solves": 63},
    ]
    linear_card = [{**linear_summary, "mean_screening_limit_k": 0.25}]
    nonlinear_card = [
        {
            "test_rmse_k": float(nonlinear_test["rmse"]),
            "validation_rmse_k": float(nonlinear_val["rmse"]),
        }
    ]
    hf_card = [{**hf_summary, "all_points_within_mesh_proxy": 1.0}]
    mesh_card = [{"max_adjacent_mesh_delta_k": 0.4962085957, "benchmark_limit_k": 0.5}]
    for filename, rows in {
        "headline_summary.csv": headline,
        "linear_summary.csv": linear_card,
        "nonlinear_summary.csv": nonlinear_card,
        "high_fidelity_summary.csv": hf_card,
        "mesh_summary.csv": mesh_card,
    }.items():
        write_csv(SOURCE_DIR / filename, rows)

    sources = [
        source(
            "case_inventory",
            "CAE case inventory",
            "case_catalog.csv",
            "Reader-facing case IDs grouped by benchmark layer and role.",
            definitions=[
                "case_ids = published/evaluation scenario IDs: linear 29 + nonlinear 27 + mesh operating points 2 + high-fidelity transient 2",
                "underlying_comsol_solves = linear physical solves 26 + nonlinear 27 + mesh variants 8 + high-fidelity transient 2",
            ],
        ),
        source(
            "coverage_source",
            "CAE case coverage",
            "case_catalog.csv",
            "Count of reader-facing case IDs by benchmark layer and role.",
            sql=(
                "SELECT layer, role, COUNT(*) AS cases "
                "FROM read_csv_auto('external_tools/comsol_chip_cooling/reports/cae_benchmark_report/source_data/case_catalog.csv', header = true) "
                "GROUP BY layer, role ORDER BY layer, role"
            ),
        ),
        source(
            "linear_catalog_source",
            "Linear CAE case catalog",
            "linear_case_catalog.csv",
            "All 29 published linear identification, forecast, and monitoring scenarios.",
        ),
        source(
            "linear_forecast_source",
            "Linear external forecast metrics",
            "linear_forecast_metrics.csv",
            "Case-level open-loop forecast metrics generated by evaluate.py.",
            filters=["初期行を評価から除外", "3領域平均温度を集約", "外部forecast 8ケース"],
            definitions=[
                "rmse_k = sqrt(mean((prediction - CAE truth)^2)) over forecast times and 3 sensors",
                "rmse_improvement_percent = 100 * (prior_rmse_k - rmse_k) / prior_rmse_k",
            ],
        ),
        source(
            "linear_monitor_source",
            "Linear causal-monitor metrics",
            "linear_monitor_metrics.csv",
            "Case-level observer, alert, bias, disturbance, and missing-data metrics.",
            filters=["NIS confidence 0.9999", "fins sensor fixed as the reference bias gauge"],
            definitions=[
                "alert_rate_percent = 100 * alert_timepoints / evaluable timepoints",
                "detection_delay_s = first detection time - injected event start time",
            ],
        ),
        source(
            "nonlinear_catalog_source",
            "Nonlinear CAE case catalog",
            "nonlinear_case_catalog.csv",
            "All 27 conjugate-flow, monitoring, and radiation-pair scenarios.",
        ),
        source(
            "nonlinear_envelope_source",
            "Nonlinear CAE operating envelope",
            "nonlinear_role_envelope.csv",
            "Worst chip temperature and physical diagnostic ranges summarized by role.",
            definitions=[
                "max_chip_c = maximum truth chip maximum temperature across cases in each role"
            ],
        ),
        source(
            "nonlinear_model_source",
            "Nonlinear identification split metrics",
            "nonlinear_model_metrics.csv",
            "Current generic scalar-law model metrics on the 10 nonlinear identification trajectories.",
            filters=["8 train, 1 validation, 1 held-out internal test case"],
            definitions=[
                "rmse = all observed timepoints and 3 region-average sensors for each case"
            ],
        ),
        source(
            "radiation_pairs_source",
            "Medium-fidelity radiation pairs",
            "radiation_pairs.csv",
            "Radiation-minus-no-radiation differences for five identical-input CAE pairs.",
            definitions=[
                "abs_terminal_chip_delta_k = absolute terminal chip-average temperature pair difference"
            ],
        ),
        source(
            "mesh_catalog_source",
            "Mesh qualification case catalog",
            "mesh_case_catalog.csv",
            "The two steady operating points used for local-mesh qualification.",
        ),
        source(
            "mesh_convergence_source",
            "Mesh convergence results",
            "mesh_convergence.csv",
            "Steady chip-average temperatures across global and three local mesh profiles.",
            filters=["MC01 nominal and MC02 hot/low-flow + radiation", "steady-state solutions"],
            definitions=[
                "benchmark qualification uses 0.50 K region-average temperature tolerance"
            ],
        ),
        source(
            "hf_catalog_source",
            "High-fidelity transient case catalog",
            "high_fidelity_case_catalog.csv",
            "The two information-dense local-medium transient evaluation cases.",
        ),
        source(
            "hf_metrics_source",
            "High-fidelity benchmark case metrics",
            "high_fidelity_case_metrics.csv",
            "External forecast error, prior comparison, and mesh-difference ratios for HV01/HV02.",
            filters=[
                "initial row excluded",
                "11 forecast times × 3 region-average sensors per case",
            ],
            definitions=[
                "rms_error_over_mesh_difference = RMSE divided by the conservative adjacent-mesh temperature difference",
                "fraction_abs_error_within_mesh_difference = fraction of scalar errors below the sensor-specific mesh proxy",
            ],
        ),
        source(
            "hf_error_source",
            "High-fidelity error and mesh comparison",
            "high_fidelity_error_comparison.csv",
            "Tidy comparison of RMSE, maximum error, and conservative adjacent-mesh difference.",
        ),
        source(
            "hf_predictions_source",
            "High-fidelity temperature trajectories",
            "high_fidelity_predictions.csv",
            "CAE truth and generic-model chip temperature trajectories for HV01 and HV02.",
            filters=["12 time rows per case", "same left zero-order-hold input convention"],
        ),
        source(
            "hf_radiation_trace_source",
            "High-fidelity radiation pair trajectory",
            "high_fidelity_radiation_trace.csv",
            "Truth and predicted HV02-minus-HV01 chip-average temperature differences.",
            definitions=[
                "truth_delta_chip_k = truth_chip(HV02 radiation) - truth_chip(HV01 conjugate)",
                "predicted_delta_chip_k = predicted_chip(HV02 radiation) - predicted_chip(HV01 conjugate)",
            ],
        ),
        source(
            "metric_definition_source",
            "Benchmark metric definitions",
            "metric_definitions.csv",
            "Reader-facing definitions and interpretation boundaries for all reported metrics.",
        ),
        source(
            "headline_source",
            "Report headline case counts",
            "headline_summary.csv",
            "Headline inventory of reader-facing case IDs and underlying COMSOL solve executions.",
        ),
        source(
            "linear_summary_source",
            "Linear forecast summary",
            "linear_summary.csv",
            "Mean and worst external linear forecast RMSE with the screening limit.",
        ),
        source(
            "nonlinear_summary_source",
            "Nonlinear held-out summary",
            "nonlinear_summary.csv",
            "Validation and held-out internal-test RMSE for the nonlinear identification set.",
        ),
        source(
            "hf_summary_source",
            "High-fidelity forecast summary",
            "high_fidelity_summary.csv",
            "Mean/worst HV RMSE and fraction of errors within the mesh proxy.",
        ),
        source(
            "mesh_summary_source",
            "Mesh uncertainty summary",
            "mesh_summary.csv",
            "Largest local-medium to local-fine region-average temperature difference.",
        ),
        document_source(
            "linear_problem_doc",
            "Linear Electronic Chip Cooling problem definition",
            "external_tools/comsol_chip_cooling/docs/problem_definition.md",
        ),
        document_source(
            "nonlinear_problem_doc",
            "Nonlinear Electronic Chip Cooling problem definition",
            "external_tools/comsol_chip_cooling/docs/nonlinear_problem_definition.md",
        ),
        document_source(
            "hf_validation_doc",
            "Local-mesh and same-boundary validation definition",
            "external_tools/comsol_chip_cooling/docs/high_fidelity_validation.md",
        ),
    ]

    cards = [
        {
            "id": "case_count_card",
            "description": "第三者が追跡できるreader-facing scenario ID数。monitor派生系列とmesh profile差を区別。",
            "dataset": "headline_summary",
            "sourceId": "headline_source",
            "metrics": [
                {"label": "評価scenario ID", "field": "case_ids", "format": "number"},
                {"label": "COMSOL solve", "field": "underlying_comsol_solves", "format": "number"},
            ],
        },
        {
            "id": "linear_rmse_card",
            "description": "physics-matched線形外部forecast 8ケースの平均温度RMSE。",
            "dataset": "linear_summary",
            "sourceId": "linear_summary_source",
            "metrics": [
                {"label": "線形外部平均RMSE [K]", "field": "mean_rmse_k", "format": "number"},
                {
                    "label": "screening上限 [K]",
                    "field": "mean_screening_limit_k",
                    "format": "number",
                },
            ],
        },
        {
            "id": "nonlinear_test_card",
            "description": "非線形同定10ケースのうち、held-out cold/high-flowケースの内部test結果。",
            "dataset": "nonlinear_summary",
            "sourceId": "nonlinear_summary_source",
            "metrics": [
                {"label": "非線形held-out RMSE [K]", "field": "test_rmse_k", "format": "number"},
                {"label": "validation RMSE [K]", "field": "validation_rmse_k", "format": "number"},
            ],
        },
        {
            "id": "hf_rmse_card",
            "description": "局所medium mesh上のHV01/HV02外部予測2ケース。",
            "dataset": "high_fidelity_summary",
            "sourceId": "hf_summary_source",
            "metrics": [
                {"label": "高忠実度平均RMSE [K]", "field": "mean_rmse_k", "format": "number"},
                {"label": "worst RMSE [K]", "field": "worst_rmse_k", "format": "number"},
            ],
        },
        {
            "id": "mesh_delta_card",
            "description": "local-medium採用時に付与する保守的な隣接mesh温度差。",
            "dataset": "mesh_summary",
            "sourceId": "mesh_summary_source",
            "metrics": [
                {
                    "label": "最大隣接mesh差 [K]",
                    "field": "max_adjacent_mesh_delta_k",
                    "format": "number",
                },
                {"label": "benchmark基準 [K]", "field": "benchmark_limit_k", "format": "number"},
            ],
        },
    ]

    charts = [
        chart(
            "coverage_chart",
            "CAEケース構成",
            "reader-facing scenario ID 60件。色は同定・外部予測・監視・モデル差・mesh収束の役割。",
            "stackedBar",
            "coverage",
            "coverage_source",
            {"field": "layer", "type": "ordinal", "label": "評価層"},
            {"field": "cases", "type": "quantitative", "label": "ケース数", "unit": "件"},
            color={"field": "role", "type": "nominal", "label": "役割"},
            y_axis_title="ケース数",
            unit="件",
            palette_kind="categorical",
        ),
        chart(
            "linear_forecast_chart",
            "線形CAE外部forecastのケース別RMSE",
            "8ケース、初期行を除く3領域平均温度。全ケースがworst screening基準0.75 Kを下回る。",
            "horizontalBar",
            "linear_forecast",
            "linear_forecast_source",
            {"field": "case_id", "type": "ordinal", "label": "ケース"},
            {"field": "rmse_k", "type": "quantitative", "label": "RMSE", "unit": "K"},
            y_axis_title="RMSE [K]",
            unit="K",
            reference_lines=[
                {
                    "axis": "y",
                    "value": 0.75,
                    "label": "worst screening 0.75 K",
                    "color": "neutral",
                    "lineStyle": "dashed",
                }
            ],
        ),
        chart(
            "linear_monitor_chart",
            "線形CAE監視ケースのalert率",
            "M01/M02/M04は0%、sensor offset M03は27.3%、未指令発熱M05は4.3%。",
            "horizontalBar",
            "linear_monitor",
            "linear_monitor_source",
            {"field": "case_id", "type": "ordinal", "label": "ケース"},
            {
                "field": "alert_rate_percent",
                "type": "quantitative",
                "label": "alert率",
                "unit": "%",
            },
            y_axis_title="alert率 [%]",
            unit="%",
        ),
        chart(
            "nonlinear_envelope_chart",
            "非線形CAEの役割別最大chip温度",
            "各役割内の最大truth hotspot。forecast群は最大143.2 °Cまで外挿端を含む。",
            "horizontalBar",
            "nonlinear_envelope",
            "nonlinear_envelope_source",
            {"field": "role", "type": "ordinal", "label": "役割"},
            {"field": "max_chip_c", "type": "quantitative", "label": "最大chip温度", "unit": "°C"},
            y_axis_title="最大chip温度 [°C]",
            unit="°C",
        ),
        chart(
            "radiation_pairs_chart",
            "中忠実度CAEの放射pair終端温度差",
            "同一入力で放射あり−なしを比較したchip平均温度差の絶対値。mesh未収束のためscreening値。",
            "horizontalBar",
            "radiation_pairs",
            "radiation_pairs_source",
            {"field": "pair", "type": "ordinal", "label": "radiation / base pair"},
            {
                "field": "abs_terminal_chip_delta_k",
                "type": "quantitative",
                "label": "|終端ΔTchip|",
                "unit": "K",
            },
            y_axis_title="|終端ΔTchip| [K]",
            unit="K",
        ),
        chart(
            "mesh_convergence_chart",
            "mesh profile別の定常chip平均温度",
            "global meshと3段階の局所meshを同一条件で比較。局所medium→fine差はMC01で0.496 K。",
            "bar",
            "mesh_convergence",
            "mesh_convergence_source",
            {"field": "mesh_label", "type": "ordinal", "label": "mesh profile"},
            {
                "field": "chip_average_c",
                "type": "quantitative",
                "label": "chip平均温度",
                "unit": "°C",
            },
            color={"field": "case_label", "type": "nominal", "label": "定常ケース"},
            y_axis_title="chip平均温度 [°C]",
            unit="°C",
            palette_kind="categorical",
        ),
        chart(
            "hf_error_chart",
            "高忠実度過渡の誤差と隣接mesh差",
            "各ケースのRMSE・最大絶対誤差と、約0.49 Kの保守的mesh差proxyを同じ温度単位で比較。",
            "bar",
            "hf_error_comparison",
            "hf_error_source",
            {"field": "case_label", "type": "ordinal", "label": "ケース"},
            {"field": "value_k", "type": "quantitative", "label": "温度差", "unit": "K"},
            color={"field": "metric", "type": "nominal", "label": "評価量"},
            y_axis_title="温度差 [K]",
            unit="K",
            palette_kind="categorical",
        ),
        chart(
            "hv01_trace_chart",
            "HV01 chip平均温度のCAE・予測軌跡",
            "0–220 s、12点。発熱・流速・入口温度を順次励起した共役熱流動ケース。",
            "line",
            "hv01_trace",
            "hf_predictions_source",
            {"field": "time", "type": "quantitative", "label": "時間", "unit": "s"},
            {
                "fields": ["truth_chip", "predicted_chip"],
                "type": "quantitative",
                "label": "chip平均温度",
                "unit": "°C",
            },
            intent="trend",
            y_axis_title="chip平均温度 [°C]",
            unit="°C",
            combination_rationale="同じ領域平均温度・同じ時刻・同じ単位のCAE truthとopen-loop予測を直接比較する。",
            palette_kind="categorical",
        ),
        chart(
            "hv02_trace_chart",
            "HV02 chip平均温度のCAE・予測軌跡",
            "0–220 s、12点。HV01と同じ入力に表面間放射だけを追加。",
            "line",
            "hv02_trace",
            "hf_predictions_source",
            {"field": "time", "type": "quantitative", "label": "時間", "unit": "s"},
            {
                "fields": ["truth_chip", "predicted_chip"],
                "type": "quantitative",
                "label": "chip平均温度",
                "unit": "°C",
            },
            intent="trend",
            y_axis_title="chip平均温度 [°C]",
            unit="°C",
            combination_rationale="同じ領域平均温度・同じ時刻・同じ単位のCAE truthとopen-loop予測を直接比較する。",
            palette_kind="categorical",
        ),
        chart(
            "hf_radiation_trace_chart",
            "高忠実度pairのchip放射差",
            "HV02−HV01。truthは終端−0.381 K、予測pair差は−0.0012 Kで、放射方向のみ一致。",
            "line",
            "hf_radiation_trace",
            "hf_radiation_trace_source",
            {"field": "time", "type": "quantitative", "label": "時間", "unit": "s"},
            {
                "fields": ["truth_delta_chip_k", "predicted_delta_chip_k"],
                "type": "quantitative",
                "label": "HV02−HV01 chip温度差",
                "unit": "K",
            },
            intent="trend",
            y_axis_title="HV02−HV01 ΔTchip [K]",
            unit="K",
            combination_rationale="同一入力pairについて、CAEが示す放射差と現在の汎用モデルが生成するpair差を同じ単位で比較する。",
            palette_kind="categorical",
        ),
    ]

    case_columns = [
        ("case_id", "Case ID", "text"),
        ("role", "役割", "text"),
        ("problem", "問題設定", "text"),
        ("evaluation_target", "評価対象", "text"),
        ("result", "結果・判定", "text"),
    ]
    tables = [
        table(
            "linear_catalog_table",
            "線形CAE 全29ケース",
            "問題設定、評価対象、最新の同定・予測・監視結果をcase ID単位で対応付け。",
            "linear_catalog",
            "linear_catalog_source",
            case_columns,
        ),
        table(
            "nonlinear_catalog_table",
            "非線形CAE 全27ケース",
            "forecast/monitorはデータ生成状態とモデル評価状態を分離して明記。",
            "nonlinear_catalog",
            "nonlinear_catalog_source",
            case_columns,
        ),
        table(
            "mesh_catalog_table",
            "局所mesh収束 2 operating points",
            "local-medium→fineの差とstrict/benchmark両基準の判定。",
            "mesh_catalog",
            "mesh_catalog_source",
            case_columns,
        ),
        table(
            "hf_catalog_table",
            "高忠実度過渡 2ケース",
            "同一入力pairに対する外部予測誤差とmesh proxy内率。",
            "hf_catalog",
            "hf_catalog_source",
            case_columns,
        ),
        table(
            "metric_definition_table",
            "評価指標の定義",
            "温度平均、hotspot、監視、mesh差、放射pairを同じ『誤差』として混同しないための定義。",
            "metric_definitions",
            "metric_definition_source",
            [
                ("metric", "指標", "text"),
                ("definition", "計算・母集団", "text"),
                ("interpretation", "解釈境界", "text"),
            ],
            sort_field="metric",
        ),
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": """## 技術サマリー

**現状のCAE群は、汎用熱RC基盤の機能検証には有用ですが、半導体製造装置の設計妥当化を完了したものではありません。** 線形physics-matched層は8外部予測で平均RMSE **0.036 K**、worst **0.083 K**、監視を含む10/10 screening checkを満たしました。これは同定・可変dt・open-loop・observer workflowが正しくつながることを強く支持します。

**非線形層は広い運転域と省略物理を露出しますが、評価の完成度は不均一です。** 10同定ケースのheld-out内部testは **0.405 K RMSE**です。一方、NF01–NF09の独立open-loop集計とNM01–NM03の監視器集計は現行成果物に無く、CAEデータ生成完了をモデル性能合格とは扱えません。

**局所mesh上のHV01/HV02は平均RMSE 0.096 K、worst 0.144 Kで、全誤差が保守的隣接mesh差内でした。** ただし最大隣接mesh差は **0.496 K**、strict収束・時間刻み収束・実験妥当化は未完です。したがって「モデル誤差が小さい」ことは示せても、「CAE数値不確かさを超えてモデル構造が正しい」とはまだ判定できません。

**放射の固有効果を現在の汎用モデルは再現していませんが、この一例だけで本体へ放射専用物理を追加すべきではありません。** HV pairのtruth終端差はchipで −0.381 K、予測pair差は −0.0012 Kですが、truth差自体が0.496 K mesh proxy未満です。次の優先事項は、装置固有実験を同じ評価境界へ投入し、複数装置ケースで再現する誤差構造だけを汎用lawとして採用することです。""",
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "case_count_card",
                "linear_rmse_card",
                "nonlinear_test_card",
                "hf_rmse_card",
                "mesh_delta_card",
            ],
        },
        {
            "id": "problem_boundary",
            "type": "markdown",
            "body": """## 問題設定と評価境界

出発点はCOMSOL 6.4 Application Libraryの **Electronic Chip Cooling** です。観測はmesh番号に依存するpoint probeではなく、`chip`、`sink_base`、`fins`の体積平均温度に統一しています。`truth_chip_max`はhotspot診断であり、RC node温度と同一視しません。

```text
chip_power → [silicon chip] → [50 µm grease/contact] → [Al sink base] → [4 fins]
                 chip              sink_base              fins
                                      ↓                      ↓
線形層:                          h=10 W/(m²K), coolant_temperature
非線形層:                temperature-dependent air + laminar conjugate flow
                         controls = power, inlet temperature, inlet velocity
高忠実度層:              fin流路・境界層・chip/contactを局所mesh化
```

すべての時系列入力はrow `k`を区間 `[t[k], t[k+1])`へ適用するleft zero-order holdです。forecast/model-gap CSVは初期観測だけを公開し、全時刻の`truth_*`は評価専用なので、CAE truthを予測入力へ漏洩させません。""",
        },
        {
            "id": "coverage_reading",
            "type": "markdown",
            "body": """### ケース構成の読み方

ケースは一つの巨大な合否表ではなく、**線形workflow確認 → 非線形adequacy screening → mesh資格確認 → 高忠実度外部予測**の順に役割を分けています。60 scenario IDは、線形monitor 5系列が2本のCAE truthから派生する一方、mesh 2条件を4 profileずつ解くため、基礎となるCOMSOL solveは63件です。""",
        },
        {
            "id": "coverage_chart_block",
            "type": "chart",
            "chartId": "coverage_chart",
            "layout": "full",
        },
        {
            "id": "linear_results",
            "type": "markdown",
            "sourceId": "linear_problem_doc",
            "body": """## 線形CAEはworkflowの整合性を高精度で確認した

線形層は流体・放射を解かず、`storage + conduction/contact + constant convection`を3-node RCと対応させたphysics-matched問題です。16同定ケースがpower/coolantのstep・ramp・multilevel・combined・warm-startを分離し、8外部予測が内挿、外挿、short pulse、未知recipe、hot-start、可変dtを確認します。monitor 5系列はnoise、drift、offset、欠測、実在する未指令発熱を分けます。

この層の小さな誤差は、本体のデータ契約・積分・同定・observerが一貫して働く証拠です。一方、COMSOL側のphysicsを本体構造へ合わせた試験なので、共役流・放射・装置実測へそのまま外挿する根拠にはしません。""",
        },
        {
            "id": "linear_forecast_reading",
            "type": "markdown",
            "body": """### 外部予測ではshort pulseが最も難しい

8ケース平均RMSEは0.036 Kで、全ケースがworst screening基準0.75 Kを十分下回りました。最大はF05 short pulsesの0.083 K、次点はF06 unseen recipeの0.059 Kです。空間hotspot underpredictionは最大0.328 Kなので、平均温度の高精度とhotspot保証は分けて扱う必要があります。""",
        },
        {
            "id": "linear_forecast_chart_block",
            "type": "chart",
            "chartId": "linear_forecast_chart",
            "layout": "full",
        },
        {
            "id": "linear_monitor_reading",
            "type": "markdown",
            "body": """### 正常・drift・欠測ではfalse alertを抑え、注入事象を検出した

M01 noise、M02 drift、M04 missingではalert率0%でした。M03の+3 K sensor offsetは事象開始時に検出し、M05の3 W未指令発熱は10秒で検出、peak外乱推定3.182 Wでした。M02/M03の絶対bias推定は`fins`を校正済みreferenceとして0へ固定した条件に依存します。""",
        },
        {
            "id": "linear_monitor_chart_block",
            "type": "chart",
            "chartId": "linear_monitor_chart",
            "layout": "full",
        },
        {
            "id": "linear_catalog_block",
            "type": "table",
            "tableId": "linear_catalog_table",
            "layout": "full",
        },
        {
            "id": "nonlinear_results",
            "type": "markdown",
            "sourceId": "nonlinear_problem_doc",
            "body": """## 非線形CAEは運転域と省略物理を露出したが、全workflow評価は未完

非線形層は3D laminar conjugate heat transfer、温度依存air、3入力（power・入口温度・入口流速）を解き、必要なpairだけ表面間放射を追加します。通常ケースは900秒・10秒刻み、NF08だけ600秒・1秒刻みです。入力域は0–16 W、10–40 °C、0.05–0.30 m/sで、外部forecastには143.2 °Cのhot-start端まで含みます。

10同定ケースでは8 train / 1 validation / 1 held-out testを実行し、testのNT09 cold/high-flowは0.405 K RMSEでした。しかしNF forecast 9件とNM monitor 3件について、最新成果物にはモデル側の集計結果がありません。したがってこの表では、**CAE truth生成**と**モデル性能評価**を明示的に分離しています。""",
        },
        {
            "id": "nonlinear_envelope_reading",
            "type": "markdown",
            "body": """### forecast群は学習域より厳しい温度端を意図的に含む

役割別の最大hotspotは同定87.8 °Cに対しforecast 143.2 °Cです。これはF03/F04/F09が高発熱・高入口温度・低流量・hot-startを組み合わせ、モデルの安定性と誤差増幅を露出するためです。設計上限を意味する値ではなく、Application Library形状上のstress testです。""",
        },
        {
            "id": "nonlinear_envelope_chart_block",
            "type": "chart",
            "chartId": "nonlinear_envelope_chart",
            "layout": "full",
        },
        {
            "id": "radiation_pair_reading",
            "type": "markdown",
            "body": """### 中忠実度pairでは高温・低流量ほど放射省略差が拡大した

5 pairの終端chip差は4.05–15.15 Kで、最大は非一様hot-startのNR05です。ただしこの27ケース群のglobal level-8 meshは収束しておらず、hot-start pairはphysicsごとに作った初期定常場の差も含みます。ここから読めるのは放射影響の方向・運転依存性であり、絶対温度差の設計確定値ではありません。""",
        },
        {
            "id": "radiation_pairs_chart_block",
            "type": "chart",
            "chartId": "radiation_pairs_chart",
            "layout": "full",
        },
        {
            "id": "nonlinear_catalog_block",
            "type": "table",
            "tableId": "nonlinear_catalog_table",
            "layout": "full",
        },
        {
            "id": "mesh_results",
            "type": "markdown",
            "sourceId": "hf_validation_doc",
            "body": """## 局所meshは用途限定基準を満たしたがstrict収束していない

fin間流路、heat-sink wall、channel wall、chip/contact、固体を個別に細分化し、2定常点をglobal-8、local-coarse、local-medium、local-fineで独立に解きました。global-8→local-coarseではMC01 chip平均が15.913 K変わり、元のglobal meshを絶対温度設計値に使えないことが明確です。

local-medium→fine差はMC01でchip平均0.496 K、MC02で0.154 Kでした。両点とも用途限定0.50 K benchmark基準は満たしますが、MC01はstrict 0.25 K基準を満たしません。そのためlocal-mediumはモデル比較用に限定し、0.496 Kを保守的mesh proxyとして付与します。""",
        },
        {
            "id": "mesh_chart_reading",
            "type": "markdown",
            "body": """### 局所化による絶対温度の変化は、単なる全体細分化では無視できない

MC01では局所mesh化でchip平均温度が約94 °Cから約111 °Cへ移動し、その後の隣接差は縮小しました。MC02は局所coarse以降の変化が小さい一方、2条件で収束速度が異なるため、単一nominal点だけでmeshを採用してはいけません。""",
        },
        {
            "id": "mesh_convergence_chart_block",
            "type": "chart",
            "chartId": "mesh_convergence_chart",
            "layout": "full",
        },
        {
            "id": "mesh_catalog_block",
            "type": "table",
            "tableId": "mesh_catalog_table",
            "layout": "full",
        },
        {
            "id": "hf_results",
            "type": "markdown",
            "body": """## 高忠実度過渡はpriorを大幅改善したが、model adequacyはmesh差を超えて未解決

HV01/HV02は0–220秒・20秒刻みの同一3入力pairです。40–100秒で0→12 W、100–160秒で0.10→0.05 m/s、140–200秒で25→35 °Cを励起します。HV01は共役熱流動、HV02は表面間放射だけを追加します。

HV01 RMSEは0.048 K（prior 0.977 Kから95.1%改善）、HV02は0.144 K（prior 0.801 Kから82.1%改善）でした。最大誤差0.344 Kを含む全33 scalar error/caseがsensor別mesh proxy内です。これは運転域内の予測が数値不確かさより小さいことを示しますが、逆にmodel-form差をmesh差より細かく識別できていません。""",
        },
        {
            "id": "hf_error_reading",
            "type": "markdown",
            "body": """### 外部予測誤差は保守的mesh差より小さい

RMSEと最大絶対誤差を約0.49 Kの隣接mesh proxyと同じ軸で比較すると、HV02のworst errorでもproxyの約70%です。統計的な信頼区間ではないため「95%保証」には使えませんが、現段階ではモデル調整よりCAE不確かさ低減が先です。""",
        },
        {
            "id": "hf_error_chart_block",
            "type": "chart",
            "chartId": "hf_error_chart",
            "layout": "full",
        },
        {
            "id": "hv01_reading",
            "type": "markdown",
            "body": """### HV01は複合入力の温度軌跡を追従した

共役熱流動だけのHV01では、発熱ramp後から高温・低流量端までtruthと予測がほぼ重なり、最大ずれは0.112 Kでした。22分未満の短い過渡であるため、さらに遅い装置熱容量の妥当性はこのケースからは分かりません。""",
        },
        {
            "id": "hv01_trace_chart_block",
            "type": "chart",
            "chartId": "hv01_trace_chart",
            "layout": "full",
        },
        {
            "id": "hv02_reading",
            "type": "markdown",
            "body": """### HV02は放射を含むtruthにも低RMSEだが、後半に系統差が増える

HV02ではairflow励起以降に差が拡大し、160–220秒のcoupled hot/low-flow phase RMSEが0.253 K、最大誤差0.344 Kでした。全体RMSEだけを見ると見落とすため、phase別に誤差増幅を監視する必要があります。""",
        },
        {
            "id": "hv02_trace_chart_block",
            "type": "chart",
            "chartId": "hv02_trace_chart",
            "layout": "full",
        },
        {
            "id": "hf_pair_reading",
            "type": "markdown",
            "body": """### 現在の汎用モデルは放射pair差の方向だけを再現した

CAEのHV02−HV01 chip差は終端−0.381 Kですが、モデルpair差は−0.0012 Kで振幅の約0.32%です。モデルが放射専用状態を持たないため自然な結果です。ただしtruth差は保守的mesh proxy 0.496 Kより小さく、paired transient mesh refinementも未実施です。この一例だけを根拠に本体へ放射専用物理を足すと、Electronic Chip Cooling固有の実装になり得ます。""",
        },
        {
            "id": "hf_radiation_trace_chart_block",
            "type": "chart",
            "chartId": "hf_radiation_trace_chart",
            "layout": "full",
        },
        {
            "id": "hf_catalog_block",
            "type": "table",
            "tableId": "hf_catalog_table",
            "layout": "full",
        },
        {
            "id": "metrics_method",
            "type": "markdown",
            "body": """## 指標と評価方法

**同定**はcase単位splitを用い、同じtrajectoryの時間行をtrain/testへ分割しません。**forecast**は初期観測から先をopen-loopで進め、初期行を誤差集計から除外します。**monitor**はCAE truthへ決定論的noise・fault・missingを付与し、NIS、検出遅延、bias、unknown heat、欠測時posteriorを別々に評価します。

**mesh比較**は同じ定常問題の隣接profile差で、統計的uncertaintyではありません。**radiation pair**は時刻・公開入力を一致させた差分で物理追加効果を分離します。**実験比較**はtemplateとvalidation codeだけが存在し、測定温度が空なので評価件数0です。""",
        },
        {
            "id": "metric_definition_block",
            "type": "table",
            "tableId": "metric_definition_table",
            "layout": "full",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": """## 限界・不確かさ・頑健性

- **半導体製造装置の代表性:** Electronic Chip Coolingはchip–grease–sink–airの単一形状です。wafer/chuck、複数heater、真空、冷却channel、接触・組付けばらつき、装置筐体の長時間熱容量を代表しません。
- **観測境界:** 3領域の体積平均であり、単一点thermocoupleや安全上のhotspotとは異なります。最大hotspot underpredictionを別指標として残しています。
- **数値収束:** local-mediumはbenchmark-only qualifiedです。strict mesh収束、過渡時間刻み収束、solver tolerance感度は未完です。
- **外部評価の欠落:** NF01–NF09のcurrent-model open-loop集計、NM01–NM03のobserver集計、実験同一境界比較が未実施です。
- **放射の識別性:** medium-fidelityの大きなpair差はmesh未収束、high-fidelityの小さなpair差はmesh proxy未満です。両者は同じ確度の数値ではありません。
- **因果主張:** 予測誤差とNIS検出は記述・予測的評価です。実装変更が実機温度を改善するという因果効果は示しません。""",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": """## 推奨する次の評価

1. **既存の汎用モデルを変えず、NF01–NF09とNM01–NM03を一括評価する。** 内挿、外挿、fast mode、未指令発熱、冷却低下を同じcase-level表へ集計し、どの誤差が複数ケースで再現するか確認します。
2. **HV pairの数値識別力を上げる。** paired transientをlocal-fineまたはQoI適応meshで再計算し、時間刻み系列も追加して、放射差と数値差を分離します。
3. **実験templateへ実測値と標準不確かさを投入する。** CAEと同じ`case_id, time, sensor, controls`を維持し、補間せず、領域平均近似の空間集約誤差を含めます。
4. **半導体製造装置を代表する複数外部ケースへ拡張する。** heater plate/wafer stage、liquid-cooled chuck、真空＋放射、接触抵抗変動、複数sensor・複数heaterを別々のbenchmark layerとして追加します。
5. **共通して残る誤差だけを本体の汎用lawへ昇格する。** 単一COMSOLケース固有の放射、流路、geometryをcoreへ埋め込まず、input-dependent edge/source/boundary lawで表せる再現性の高い依存性だけを採用します。""",
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": """## 追加で答えるべき問い

- 実務上の一次判定量は領域平均温度、hotspot、温度均一性、整定時間のどれか。
- 対象装置で公開可能なcontrol、実測sensor、校正reference、不確かさは何か。
- 予測・監視に必要な時間範囲は秒、分、時間のどこまでか。
- CAE差・実験差・model差のどれを、どの許容値で受入れるか。
- 複数装置で共通する非線形性は流速依存conductance、温度依存source、接触依存edgeのどれか。""",
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "Electronic Chip Coolingを用いた全CAEケースの問題設定・評価対象・結果・解釈を監査可能に整理した技術レポート。",
            "generatedAt": GENERATED_AT,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "headline_summary": headline,
                "linear_summary": linear_card,
                "nonlinear_summary": nonlinear_card,
                "high_fidelity_summary": hf_card,
                "mesh_summary": mesh_card,
                "coverage": coverage,
                "linear_forecast": linear_forecast,
                "linear_monitor": linear_monitor,
                "linear_catalog": linear_catalog,
                "nonlinear_catalog": nonlinear_catalog,
                "nonlinear_envelope": nonlinear_envelope,
                "radiation_pairs": radiation_pairs,
                "mesh_catalog": mesh_catalog,
                "mesh_convergence": mesh_chart,
                "hf_catalog": hf_catalog,
                "hf_error_comparison": hf_error,
                "hv01_trace": hf01,
                "hv02_trace": hf02,
                "hf_radiation_trace": hf_radiation_trace,
                "metric_definitions": metric_definitions,
            },
        },
        "sources": sources,
        "package_info": {
            "report_generator": "external_tools/comsol_chip_cooling/reports/cae_benchmark_report/build_report.py",
            "snapshot_scope": "COMSOL Electronic Chip Cooling CAE benchmark evidence available on 2026-08-31",
        },
    }
    (REPORT_DIR / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    chart_map = f"""# Chart map and report QA notes

Generated: {GENERATED_AT}

Audience: technical. Delivery mode: portable HTML from canonical artifact.json.

## Required-structure mapping

1. Title → `title`
2. Technical summary → `technical_summary` + headline metric strip
3. Key findings with visual evidence → linear, nonlinear, mesh, and high-fidelity result sections
4. Scope/data/definitions → `problem_boundary`, case catalogs, metric definition table
5. Methodology → `metrics_method`
6. Limitations/uncertainty/robustness → `limitations`
7. Recommended next steps → `next_steps`
8. Further questions → `further_questions`

## Visual contracts

| Visual | Question | Family/type | Fields | Data sufficiency | Palette policy | Supported claim |
|---|---|---|---|---|---|---|
| coverage_chart | How are case IDs distributed by layer and role? | composition / stacked bar | layer, role, cases | 9 nonzero groups | relaxed categorical | The suite is layered rather than one undifferentiated benchmark. |
| linear_forecast_chart | Which linear forecast is hardest? | comparison / horizontal bar | case_id, rmse_k | 8 cases | single-root | Short pulse is worst but all cases are far below the criterion. |
| linear_monitor_chart | Which monitoring cases trigger alerts? | comparison / horizontal bar | case_id, alert_rate_percent | 5 cases | single-root | Alerts are confined to injected abrupt offset/heat cases. |
| nonlinear_envelope_chart | How far does each nonlinear role extend thermally? | comparison / horizontal bar | role, max_chip_c | 4 roles | single-root | Forecast cases intentionally extend beyond the training thermal envelope. |
| radiation_pairs_chart | How does the medium-fidelity radiation gap vary by condition? | comparison / horizontal bar | pair, abs_terminal_chip_delta_k | 5 matched pairs | single-root | Hot/low-flow and hot-start show the largest screening gaps. |
| mesh_convergence_chart | How does chip temperature move with mesh profile? | comparison / grouped bar | mesh_label, case_label, chip_average_c | 4 profiles × 2 cases | hard two-root | Global mesh is inadequate and convergence differs by operating point. |
| hf_error_chart | Are high-fidelity model errors smaller than the mesh proxy? | benchmark / grouped bar | case_label, metric, value_k | 2 cases × 3 metrics | categorical (three semantic metrics) | All model errors are below the conservative adjacent-mesh difference. |
| hv01_trace_chart | Does the model track the non-radiating composite transient? | trend / line | time, truth_chip, predicted_chip | 12 ordered points | hard two-root | HV01 temperature trajectory is closely tracked. |
| hv02_trace_chart | Does error grow when radiation is present? | trend / line | time, truth_chip, predicted_chip | 12 ordered points | hard two-root | Error grows in the coupled hot/low-flow phase. |
| hf_radiation_trace_chart | Does the model reproduce the paired radiation-only delta? | trend / line | time, truth_delta_chip_k, predicted_delta_chip_k | 12 paired points | hard two-root | Direction matches but amplitude is almost absent in the model. |

The three line charts are retained because they answer distinct temporal questions on the same 12-point experimental design. Bar-family repetition is intentional: the other questions are categorical comparisons, not continuous trends.

## Evidence gaps kept visible

- NF01–NF09 open-loop model metrics are not present in the current benchmark outputs.
- NM01–NM03 observer metrics are not present in the current benchmark outputs.
- No measured experiment values have been supplied; templates are not counted as validation.
- High-fidelity strict mesh convergence and transient time-step convergence are not complete.
"""
    (REPORT_DIR / "chart_map.md").write_text(chart_map, encoding="utf-8")

    readme = """# CAE benchmark report artifact

Primary deliverable: `report.html`.

- `artifact.json`: canonical bounded report input.
- `source_data/`: normalized snapshot tables used by the report.
- `chart_map.md`: chart contracts, structure mapping, and explicit evidence gaps.
- `build_report.py`: deterministic normalization and artifact authoring.

Rebuild the artifact from repository root:

```powershell
.venv\\Scripts\\python.exe external_tools/comsol_chip_cooling/reports/cae_benchmark_report/build_report.py
```

Then package and verify `artifact.json` with the Data Analytics portable report builder. The script consumes the published CAE datasets plus the latest ignored `work/evaluation` and model-run outputs; rerun the corresponding benchmark workflows first if those work products are absent.
"""
    (REPORT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact": str(REPORT_DIR / "artifact.json"),
                "case_ids": len(all_cases),
                "charts": len(charts),
                "tables": len(tables),
                "datasets": len(artifact["snapshot"]["datasets"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
