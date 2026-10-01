"""CSV summaries and Japanese scientific figures for model comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg", force=True)


MODEL_ORDER = ("physical_rc", "mlp", "cnn1d", "tcn", "gru", "lstm")
MODEL_LABELS = {
    "physical_rc": "物理RC",
    "mlp": "MLP",
    "cnn1d": "1D-CNN",
    "tcn": "TCN",
    "gru": "GRU",
    "lstm": "LSTM",
}
MODEL_COLORS = {
    "physical_rc": "#007C83",
    "mlp": "#E76F51",
    "cnn1d": "#F4A261",
    "tcn": "#5B8FF9",
    "gru": "#7A5195",
    "lstm": "#2A9D8F",
}
EVALUATION_LABELS = {
    "topcell": "TopCell",
    "linear_comsol": "線形COMSOL",
    "nonlinear_comsol": "非線形COMSOL",
    "high_fidelity_comsol": "高忠実度COMSOL",
}
INTERNAL_DATASETS = ("topcell", "linear_comsol", "nonlinear_comsol")
SPLIT_LABELS = {
    "train": "train (学習内)",
    "validation": "validation (選択用)",
    "test": "test (保留)",
}
SPLIT_COLORS = {"train": "#B8D8D8", "validation": "#F4D35E", "test": "#EE964B"}


def configure_japanese_plotting() -> None:
    """Choose an installed Japanese font without adding a plotting dependency."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"):
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def pooled_r2(frame: pd.DataFrame) -> float:
    truth = frame["truth"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    paired = np.isfinite(truth) & np.isfinite(predicted)
    truth = truth[paired]
    predicted = predicted[paired]
    if len(truth) < 2:
        return float("nan")
    total = float(np.sum((truth - np.mean(truth)) ** 2))
    return float("nan") if total == 0.0 else 1.0 - float(np.sum((predicted - truth) ** 2)) / total


def summarize_predictions(
    case_metrics: pd.DataFrame,
    prediction_points: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize external cases without mixing different benchmark categories."""
    rows: list[dict[str, object]] = []
    keys = ["evaluation", "category", "model"]
    for key, cases in case_metrics.groupby(keys, sort=False):
        evaluation, category, model = key
        points = prediction_points[
            (prediction_points["evaluation"] == evaluation)
            & (prediction_points["category"] == category)
            & (prediction_points["model"] == model)
        ]
        rows.append(
            {
                "evaluation": evaluation,
                "category": category,
                "model": model,
                "n_cases": int(cases["case_id"].nunique()),
                "mean_case_rmse_k": float(cases["rmse_k"].mean()),
                "worst_case_rmse_k": float(cases["rmse_k"].max()),
                "mean_case_mae_k": float(cases["mae_k"].mean()),
                "max_abs_error_k": float(cases["max_abs_error_k"].max()),
                "pooled_r2": pooled_r2(points),
                "n_points": len(points),
            }
        )
    return pd.DataFrame(rows)


def summarize_internal_predictions(
    case_metrics: pd.DataFrame,
    test_points: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize internal splits; pooled R² is retained for the held-out test only."""
    rows: list[dict[str, object]] = []
    for key, cases in case_metrics.groupby(["dataset", "split", "model"], sort=False):
        dataset, split, model = key
        points = test_points[
            (test_points["dataset"] == dataset)
            & (test_points["split"] == split)
            & (test_points["model"] == model)
        ]
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "model": model,
                "n_cases": int(cases["case_id"].nunique()),
                "mean_case_rmse_k": float(cases["rmse_k"].mean()),
                "worst_case_rmse_k": float(cases["rmse_k"].max()),
                "mean_case_mae_k": float(cases["mae_k"].mean()),
                "max_abs_error_k": float(cases["max_abs_error_k"].max()),
                "pooled_r2": pooled_r2(points) if not points.empty else float("nan"),
                "n_points": len(points),
            }
        )
    return pd.DataFrame(rows)


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    order = {name: index for index, name in enumerate(MODEL_ORDER)}
    return frame.assign(_order=frame["model"].map(order)).sort_values("_order")


def plot_overview(summary: pd.DataFrame, output: Path) -> None:
    core = summary[summary["category"] == "core"]
    evaluations = [name for name in EVALUATION_LABELS if name in set(core["evaluation"])]
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for axis, evaluation in zip(axes.flat, evaluations):
        table = _ordered(core[core["evaluation"] == evaluation])
        x = np.arange(len(table))
        colors = [MODEL_COLORS[str(name)] for name in table["model"]]
        axis.bar(x, table["mean_case_rmse_k"], color=colors, alpha=0.85, label="平均")
        axis.scatter(
            x,
            table["worst_case_rmse_k"],
            marker="x",
            s=55,
            color="#1D3557",
            label="最大ケース",
            zorder=3,
        )
        axis.set_xticks(x, [MODEL_LABELS[str(name)] for name in table["model"]], rotation=25)
        axis.set_ylabel("RMSE [K]")
        axis.set_title(EVALUATION_LABELS[evaluation])
        axis.grid(axis="y", alpha=0.25)
        if float(table["worst_case_rmse_k"].max()) / float(table["mean_case_rmse_k"].min()) > 50.0:
            axis.set_yscale("log")
        axis.legend(frameon=False, fontsize=9)
    for axis in axes.flat[len(evaluations) :]:
        axis.set_visible(False)
    figure.suptitle("外部ケースにおける物理RCとニューラル時系列モデルの比較", fontsize=17)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _selected_case_and_sensor(
    case_metrics: pd.DataFrame,
    points: pd.DataFrame,
    evaluation: str,
) -> tuple[str, str]:
    rc_cases = case_metrics[
        (case_metrics["evaluation"] == evaluation)
        & (case_metrics["category"] == "core")
        & (case_metrics["model"] == "physical_rc")
    ]
    case_id = str(rc_cases.loc[rc_cases["rmse_k"].idxmax(), "case_id"])
    rc_points = points[
        (points["evaluation"] == evaluation)
        & (points["category"] == "core")
        & (points["case_id"] == case_id)
        & (points["model"] == "physical_rc")
    ].copy()
    rc_points["square_error"] = (rc_points["predicted"] - rc_points["truth"]) ** 2
    sensor_rmse = rc_points.groupby("sensor")["square_error"].mean().pow(0.5)
    return case_id, str(sensor_rmse.idxmax())


def plot_timeseries_panels(
    case_metrics: pd.DataFrame,
    points: pd.DataFrame,
    evaluation: str,
    output: Path,
) -> None:
    case_id, sensor = _selected_case_and_sensor(case_metrics, points, evaluation)
    selected = points[
        (points["evaluation"] == evaluation)
        & (points["category"] == "core")
        & (points["case_id"] == case_id)
        & (points["sensor"] == sensor)
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for axis, model in zip(axes.flat, MODEL_ORDER):
        model_rows = selected[selected["model"] == model].sort_values("time")
        axis.plot(model_rows["time"], model_rows["truth"], color="#111111", label="真値")
        axis.plot(
            model_rows["time"],
            model_rows["predicted"],
            color=MODEL_COLORS[model],
            linestyle="--",
            label="予測",
        )
        error = model_rows["predicted"].to_numpy() - model_rows["truth"].to_numpy()
        rmse = float(np.sqrt(np.mean(error**2)))
        axis.set_title(f"{MODEL_LABELS[model]}  RMSE={rmse:.3f} K")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    for axis in axes[1, :]:
        axis.set_xlabel("時間 [s]")
    for axis in axes[:, 0]:
        axis.set_ylabel("温度 [°C]")
    title = EVALUATION_LABELS[evaluation]
    figure.suptitle(
        f"{title}: 同一外部ケース・同一センサの時系列比較\n"
        f"ケース={case_id}, センサ={sensor} (物理RCの最大誤差ケースから選択)",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_parity_panels(points: pd.DataFrame, evaluation: str, output: Path) -> None:
    selected = points[(points["evaluation"] == evaluation) & (points["category"] == "core")]
    finite = selected[np.isfinite(selected["truth"]) & np.isfinite(selected["predicted"])]
    lower = float(min(finite["truth"].min(), finite["predicted"].min()))
    upper = float(max(finite["truth"].max(), finite["predicted"].max()))
    padding = max(0.05 * (upper - lower), 0.1)
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
    for axis, model in zip(axes.flat, MODEL_ORDER):
        model_rows = finite[finite["model"] == model]
        axis.scatter(
            model_rows["truth"],
            model_rows["predicted"],
            s=9,
            alpha=0.35,
            color=MODEL_COLORS[model],
            edgecolors="none",
        )
        axis.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            color="#333333",
            linewidth=1,
        )
        axis.set_title(f"{MODEL_LABELS[model]}  R²={pooled_r2(model_rows):.4f}")
        axis.grid(alpha=0.2)
    for axis in axes[1, :]:
        axis.set_xlabel("真値 [°C]")
    for axis in axes[:, 0]:
        axis.set_ylabel("予測値 [°C]")
    figure.suptitle(
        f"{EVALUATION_LABELS[evaluation]}: 外部全ケース・全センサの真値-予測比較",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_training_history(history: pd.DataFrame, output: Path) -> None:
    datasets = [
        name
        for name in ("topcell", "linear_comsol", "nonlinear_comsol")
        if name in set(history["dataset"])
    ]
    figure, axes = plt.subplots(1, len(datasets), figsize=(16, 4.8), squeeze=False)
    for axis, dataset in zip(axes.flat, datasets):
        subset = history[
            (history["dataset"] == dataset) & np.isfinite(history["validation_open_loop_rmse"])
        ]
        for model in MODEL_ORDER[1:]:
            rows = subset[subset["model"] == model]
            axis.plot(
                rows["epoch"],
                rows["validation_open_loop_rmse"],
                marker="o",
                markersize=3,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        axis.set_title(EVALUATION_LABELS[dataset])
        axis.set_xlabel("epoch")
        axis.set_ylabel("検証open-loop RMSE [K]")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=9)
    figure.suptitle("学習中の因果open-loop検証誤差", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_model_gap(summary: pd.DataFrame, output: Path) -> None:
    gap = summary[summary["category"] == "model_gap"]
    evaluations = [name for name in EVALUATION_LABELS if name in set(gap["evaluation"])]
    figure, axes = plt.subplots(1, len(evaluations), figsize=(16, 4.8), squeeze=False)
    for axis, evaluation in zip(axes.flat, evaluations):
        table = _ordered(gap[gap["evaluation"] == evaluation])
        x = np.arange(len(table))
        axis.bar(
            x,
            table["mean_case_rmse_k"],
            color=[MODEL_COLORS[str(model)] for model in table["model"]],
        )
        axis.set_xticks(x, [MODEL_LABELS[str(model)] for model in table["model"]], rotation=25)
        axis.set_ylabel("平均RMSE [K]")
        axis.set_title(EVALUATION_LABELS[evaluation])
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("既知のmodel-gapケース (通常性能とは分離)", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _diagram_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    title: str,
    body: str,
    color: str,
) -> None:
    x, y = xy
    width, height = size
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=color,
            edgecolor="#334155",
            linewidth=1.4,
        )
    )
    axis.text(x + width / 2, y + height * 0.68, title, ha="center", va="center", weight="bold")
    axis.text(x + width / 2, y + height * 0.31, body, ha="center", va="center", fontsize=10)


def _boundary_count_lines(boundaries: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for dataset in INTERNAL_DATASETS:
        internal = boundaries[
            (boundaries["training_dataset"] == dataset) & (boundaries["scope"] == "internal")
        ]
        counts = {str(row["boundary"]): int(row["n_cases"]) for _, row in internal.iterrows()}
        external = boundaries[
            (boundaries["training_dataset"] == dataset) & (boundaries["scope"] == "external")
        ]
        external_text = ", ".join(
            f"{EVALUATION_LABELS.get(str(row['evaluation_dataset']), row['evaluation_dataset'])} "
            f"{row['category']}={int(row['n_cases'])}"
            for _, row in external.iterrows()
        )
        lines.append(
            f"{EVALUATION_LABELS[dataset]}: 内部 train={counts.get('train', 0)}, "
            f"validation={counts.get('validation', 0)}, test={counts.get('test', 0)}"
            f" | 外部 {external_text}"
        )
    return lines


def plot_evaluation_boundary(boundaries: pd.DataFrame, output: Path) -> None:
    """Show the data-flow definition that separates internal and external forecasts."""
    figure, axis = plt.subplots(figsize=(16, 9))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    axis.text(
        0.5, 0.96, "内部予測と外部予測を分けるデータ境界", ha="center", fontsize=20, weight="bold"
    )
    axis.text(
        0.5,
        0.905,
        "真値の遮断: 内部は時刻0だけ、外部は各dataset所定の観測prefixまで。"
        "以後は入力波形とdtだけで予測",
        ha="center",
        fontsize=12,
    )

    axis.add_patch(
        FancyBboxPatch(
            (0.025, 0.32),
            0.59,
            0.52,
            boxstyle="round,pad=0.015",
            facecolor="#F8FAFC",
            edgecolor="#0284C7",
            linewidth=2.2,
        )
    )
    axis.text(
        0.32,
        0.80,
        "内部: 設定 data.directory の全ケースを、case単位で一度だけ分割",
        ha="center",
        fontsize=13,
        weight="bold",
        color="#0369A1",
    )
    _diagram_box(
        axis, (0.055, 0.56), (0.16, 0.16), "train", "係数・重みを更新\n誤差はin-sample", "#D8F3DC"
    )
    _diagram_box(
        axis,
        (0.24, 0.56),
        (0.16, 0.16),
        "validation",
        "更新には不使用\nbest epochを選択",
        "#FFF3B0",
    )
    _diagram_box(
        axis,
        (0.425, 0.56),
        (0.16, 0.16),
        "internal test",
        "更新・選択に不使用\n内部予測の主評価",
        "#FFD6A5",
    )
    _diagram_box(
        axis, (0.235, 0.365), (0.17, 0.105), "固定済みモデル", "RC係数 / NN重み", "#DBEAFE"
    )
    for start in ((0.135, 0.55), (0.32, 0.55)):
        axis.annotate("", xy=(0.32, 0.47), xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8})
    axis.annotate(
        "", xy=(0.505, 0.55), xytext=(0.405, 0.42), arrowprops={"arrowstyle": "->", "lw": 1.8}
    )

    axis.add_patch(
        FancyBboxPatch(
            (0.65, 0.32),
            0.325,
            0.52,
            boxstyle="round,pad=0.015",
            facecolor="#FFF7ED",
            edgecolor="#EA580C",
            linewidth=2.2,
        )
    )
    axis.text(
        0.812,
        0.785,
        "外部: 別ディレクトリのケース\nsplitter・学習・epoch選択へ一度も投入しない",
        ha="center",
        fontsize=13,
        weight="bold",
        color="#C2410C",
    )
    _diagram_box(
        axis, (0.685, 0.58), (0.255, 0.13), "external core", "通常の外部予測性能", "#FFEDD5"
    )
    _diagram_box(
        axis,
        (0.685, 0.39),
        (0.255, 0.13),
        "model-gap / 高忠実度transfer",
        "責務外差や別忠実度を分離評価",
        "#FDE2E4",
    )
    axis.annotate(
        "",
        xy=(0.68, 0.65),
        xytext=(0.405, 0.42),
        arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#475569"},
    )
    axis.annotate(
        "",
        xy=(0.68, 0.455),
        xytext=(0.405, 0.42),
        arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#475569"},
    )
    axis.text(0.55, 0.49, "固定後に適用", ha="center", fontsize=10.5, color="#475569")
    axis.text(
        0.5,
        0.275,
        "外部benchmarkは『学習データから独立』を意味する。実機妥当化済みという意味ではない。",
        ha="center",
        fontsize=12,
        weight="bold",
        color="#9A3412",
    )
    for index, line in enumerate(_boundary_count_lines(boundaries)):
        axis.text(0.04, 0.205 - index * 0.052, line, ha="left", fontsize=10.5)
    axis.text(
        0.04,
        0.035,
        "予測評価では初期行を誤差集計から除外。train/validation/testは同じ分割をRC・全NNで共有。",
        ha="left",
        fontsize=10.5,
        color="#475569",
    )
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_internal_split_overview(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, len(INTERNAL_DATASETS), figsize=(17, 5.4), squeeze=False)
    for axis, dataset in zip(axes.flat, INTERNAL_DATASETS):
        selected = summary[summary["dataset"] == dataset]
        x = np.arange(len(MODEL_ORDER))
        width = 0.24
        for offset, split in enumerate(("train", "validation", "test")):
            table = _ordered(selected[selected["split"] == split]).set_index("model")
            values = table["mean_case_rmse_k"].reindex(MODEL_ORDER).to_numpy(dtype=float)
            axis.bar(
                x + (offset - 1) * width,
                values,
                width=width,
                color=SPLIT_COLORS[split],
                label=SPLIT_LABELS[split],
            )
        axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=25)
        axis.set_ylabel("case平均RMSE [K]")
        axis.set_title(EVALUATION_LABELS[dataset])
        axis.grid(axis="y", alpha=0.25)
        positive = selected.loc[selected["mean_case_rmse_k"] > 0, "mean_case_rmse_k"]
        if not positive.empty and float(positive.max() / positive.min()) > 50.0:
            axis.set_yscale("log")
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "同一データ集合内のopen-loop予測: train / validation / held-out test", fontsize=16
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _selected_internal_case_and_sensor(
    case_metrics: pd.DataFrame,
    points: pd.DataFrame,
    dataset: str,
) -> tuple[str, str]:
    rc_cases = case_metrics[
        (case_metrics["dataset"] == dataset)
        & (case_metrics["split"] == "test")
        & (case_metrics["model"] == "physical_rc")
    ]
    case_id = str(rc_cases.loc[rc_cases["rmse_k"].idxmax(), "case_id"])
    rc_points = points[
        (points["dataset"] == dataset)
        & (points["case_id"] == case_id)
        & (points["model"] == "physical_rc")
    ].copy()
    rc_points["square_error"] = (rc_points["predicted"] - rc_points["truth"]) ** 2
    sensor_rmse = rc_points.groupby("sensor")["square_error"].mean().pow(0.5)
    return case_id, str(sensor_rmse.idxmax())


def plot_internal_timeseries(
    case_metrics: pd.DataFrame,
    points: pd.DataFrame,
    dataset: str,
    output: Path,
) -> None:
    case_id, sensor = _selected_internal_case_and_sensor(case_metrics, points, dataset)
    selected = points[
        (points["dataset"] == dataset)
        & (points["case_id"] == case_id)
        & (points["sensor"] == sensor)
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for axis, model in zip(axes.flat, MODEL_ORDER):
        rows = selected[selected["model"] == model].sort_values("time")
        axis.plot(rows["time"], rows["truth"], color="#111111", label="真値")
        axis.plot(
            rows["time"], rows["predicted"], color=MODEL_COLORS[model], linestyle="--", label="予測"
        )
        error = rows["predicted"].to_numpy() - rows["truth"].to_numpy()
        axis.set_title(f"{MODEL_LABELS[model]}  RMSE={np.sqrt(np.mean(error**2)):.3f} K")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    for axis in axes[1, :]:
        axis.set_xlabel("時間 [s]")
    for axis in axes[:, 0]:
        axis.set_ylabel("温度 [°C]")
    figure.suptitle(
        f"{EVALUATION_LABELS[dataset]} 内部test: 初期温度だけからの時系列予測\n"
        f"ケース={case_id}, センサ={sensor} (物理RCの最大誤差case)",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_internal_parity(points: pd.DataFrame, dataset: str, output: Path) -> None:
    selected = points[points["dataset"] == dataset]
    finite = selected[np.isfinite(selected["truth"]) & np.isfinite(selected["predicted"])]
    lower = float(min(finite["truth"].min(), finite["predicted"].min()))
    upper = float(max(finite["truth"].max(), finite["predicted"].max()))
    padding = max(0.05 * (upper - lower), 0.1)
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
    for axis, model in zip(axes.flat, MODEL_ORDER):
        rows = finite[finite["model"] == model]
        axis.scatter(
            rows["truth"],
            rows["predicted"],
            s=9,
            alpha=0.35,
            color=MODEL_COLORS[model],
            edgecolors="none",
        )
        axis.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            color="#333333",
            linewidth=1,
        )
        axis.set_title(f"{MODEL_LABELS[model]}  R²={pooled_r2(rows):.4f}")
        axis.grid(alpha=0.2)
    for axis in axes[1, :]:
        axis.set_xlabel("真値 [°C]")
    for axis in axes[:, 0]:
        axis.set_ylabel("予測値 [°C]")
    figure.suptitle(
        f"{EVALUATION_LABELS[dataset]} 内部test: 全保留ケース・全センサの真値-予測比較",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_internal_vs_external(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, len(INTERNAL_DATASETS), figsize=(17, 5.4), squeeze=False)
    for axis, dataset in zip(axes.flat, INTERNAL_DATASETS):
        internal = _ordered(
            internal_summary[
                (internal_summary["dataset"] == dataset) & (internal_summary["split"] == "test")
            ]
        ).set_index("model")
        external = _ordered(
            external_summary[
                (external_summary["evaluation"] == dataset)
                & (external_summary["category"] == "core")
            ]
        ).set_index("model")
        x = np.arange(len(MODEL_ORDER))
        internal_values = internal["mean_case_rmse_k"].reindex(MODEL_ORDER).to_numpy(dtype=float)
        external_values = external["mean_case_rmse_k"].reindex(MODEL_ORDER).to_numpy(dtype=float)
        axis.bar(x - 0.19, internal_values, width=0.38, color="#EE964B", label="内部test")
        axis.bar(x + 0.19, external_values, width=0.38, color="#577590", label="外部core")
        axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=25)
        axis.set_ylabel("case平均RMSE [K]")
        axis.set_title(EVALUATION_LABELS[dataset])
        axis.grid(axis="y", alpha=0.25)
        values = np.concatenate([internal_values, external_values])
        positive = values[values > 0.0]
        if len(positive) and float(positive.max() / positive.min()) > 50.0:
            axis.set_yscale("log")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("内部testと外部coreの境界別比較 (評価ケース集合は別)", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_observation_noise_sensitivity(summary: pd.DataFrame, output: Path) -> None:
    """Show inference-time observation-noise sensitivity at each data boundary."""
    panels = [
        *((dataset, "internal_test") for dataset in INTERNAL_DATASETS),
        *((evaluation, "external_core") for evaluation in EVALUATION_LABELS),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
    for axis, (evaluation, boundary) in zip(axes.flat, panels):
        table = summary[(summary["evaluation"] == evaluation) & (summary["boundary"] == boundary)]
        for model in MODEL_ORDER:
            model_rows = table[table["model"] == model].sort_values("noise_std_k")
            if model_rows.empty:
                continue
            axis.plot(
                model_rows["noise_std_k"],
                model_rows["mean_case_rmse_k"],
                marker="o",
                linewidth=2,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        boundary_label = "内部test" if boundary == "internal_test" else "外部core"
        axis.set_title(f"{EVALUATION_LABELS[evaluation]} / {boundary_label}")
        axis.set_xticks([0.0, 0.15, 0.50], ["clean", "0.15", "0.50"])
        axis.set_xlabel("測定誤差の標準偏差 [K]")
        axis.set_ylabel("case平均RMSE [K]")
        axis.grid(alpha=0.25)
        positive = table["mean_case_rmse_k"].to_numpy(dtype=float)
        positive = positive[positive > 0.0]
        if len(positive) and float(positive.max() / positive.min()) > 100.0:
            axis.set_yscale("log")
    note_axis = axes.flat[-1]
    note_axis.axis("off")
    note_axis.text(
        0.02,
        0.78,
        "評価範囲",
        fontsize=14,
        weight="bold",
        transform=note_axis.transAxes,
    )
    note_axis.text(
        0.02,
        0.67,
        "・cleanで学習済みのmodelを固定\n"
        "・予測開始までに観測済みの温度だけを摂動\n"
        "・将来のclean truthに対して採点\n"
        "・0.15 Kはmonitor設定と同じ通常の測定ばらつき\n"
        "・未知物理、入力誤差、学習dataの測定誤差は含めない",
        fontsize=11,
        linespacing=1.55,
        va="top",
        transform=note_axis.transAxes,
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(MODEL_ORDER), frameon=False)
    figure.suptitle(
        "予測開始までの測定ばらつきに対するopen-loop予測感度 (clean truth基準)",
        fontsize=17,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_rc_model_gap_risk(summary: pd.DataFrame, output: Path) -> None:
    """Contrast matched-condition RC accuracy with known omitted-physics cases."""
    table = summary.set_index("evaluation").reindex(
        ["topcell", "nonlinear_comsol", "high_fidelity_comsol"]
    )
    table = table.dropna(subset=["core_mean_case_rmse_k"])
    x = np.arange(len(table))
    figure, axis = plt.subplots(figsize=(12, 6.5))
    axis.bar(
        x - 0.2,
        table["core_mean_case_rmse_k"],
        width=0.4,
        color="#4C956C",
        label="通常core (想定内)",
    )
    gap_bars = axis.bar(
        x + 0.2,
        table["model_gap_mean_case_rmse_k"],
        width=0.4,
        color="#D1495B",
        label="既知model-gap (未知項を追加)",
    )
    axis.set_yscale("log")
    axis.set_ylabel("物理RC case平均RMSE [K] (対数軸)")
    axis.set_xticks(x, [EVALUATION_LABELS[name] for name in table.index])
    axis.grid(axis="y", which="both", alpha=0.25)
    axis.legend(frameon=False)
    for bar, ratio in zip(gap_bars, table["model_gap_to_core_ratio"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.12,
            f"{float(ratio):.1f}倍",
            ha="center",
            va="bottom",
            weight="bold",
            color="#9B2226",
        )
    axis.set_title("通常caseの低誤差だけでは、未知物理への妥当性を示せない", fontsize=16)
    figure.text(
        0.5,
        0.01,
        "TopCell: 温度依存熱損失 / COMSOL: 表面間放射。model-gapは通常性能へ平均しない。",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def write_figures(
    summary: pd.DataFrame,
    case_metrics: pd.DataFrame,
    prediction_points: pd.DataFrame,
    training_history: pd.DataFrame,
    internal_summary: pd.DataFrame,
    internal_case_metrics: pd.DataFrame,
    internal_test_points: pd.DataFrame,
    evaluation_boundaries: pd.DataFrame,
    noise_summary: pd.DataFrame,
    rc_model_gap_summary: pd.DataFrame,
    output_directory: Path,
) -> None:
    configure_japanese_plotting()
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_evaluation_boundary(
        evaluation_boundaries, output_directory / "00_internal_external_definition.png"
    )
    plot_overview(summary, output_directory / "01_external_rmse_overview.png")
    plot_training_history(training_history, output_directory / "02_training_validation_history.png")
    for index, evaluation in enumerate(EVALUATION_LABELS, start=3):
        plot_timeseries_panels(
            case_metrics,
            prediction_points,
            evaluation,
            output_directory / f"{index:02d}_{evaluation}_timeseries.png",
        )
        plot_parity_panels(
            prediction_points,
            evaluation,
            output_directory / f"{index:02d}_{evaluation}_parity.png",
        )
    plot_model_gap(summary, output_directory / "07_model_gap_rmse.png")
    plot_internal_split_overview(
        internal_summary, output_directory / "08_internal_rmse_by_split.png"
    )
    for index, dataset in enumerate(INTERNAL_DATASETS, start=9):
        plot_internal_timeseries(
            internal_case_metrics,
            internal_test_points,
            dataset,
            output_directory / f"{index:02d}_{dataset}_internal_test_timeseries.png",
        )
        plot_internal_parity(
            internal_test_points,
            dataset,
            output_directory / f"{index:02d}_{dataset}_internal_test_parity.png",
        )
    plot_internal_vs_external(
        internal_summary,
        summary,
        output_directory / "12_internal_test_vs_external_core.png",
    )
    plot_observation_noise_sensitivity(
        noise_summary,
        output_directory / "13_observation_noise_sensitivity.png",
    )
    plot_rc_model_gap_risk(
        rc_model_gap_summary,
        output_directory / "14_rc_unknown_physics_risk.png",
    )
