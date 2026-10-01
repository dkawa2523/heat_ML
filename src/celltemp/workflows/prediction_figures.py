"""Publication-style held-out prediction figures shared by evaluation workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_SENSOR_COLORS = ("#1f5a94", "#d17c0f", "#8a6d1d", "#6b7280", "#a44a7a")


@dataclass(frozen=True)
class PredictionCase:
    """One held-out case at a common time and sensor boundary."""

    case_id: str
    time: np.ndarray
    truth: np.ndarray
    predicted: np.ndarray
    evaluation_start_index: int = 1

    def __post_init__(self) -> None:
        if self.truth.ndim != 2 or self.predicted.shape != self.truth.shape:
            raise ValueError("truth and predicted must be equally shaped two-dimensional arrays")
        if self.time.ndim != 1 or len(self.time) != len(self.truth):
            raise ValueError("time must have one value for every prediction row")
        if not 0 <= self.evaluation_start_index < len(self.time):
            raise ValueError("evaluation_start_index is outside the prediction rows")


@dataclass(frozen=True)
class PredictionFigureResult:
    """Values displayed by a pair of prediction-validation figures."""

    representative_case: str
    r_squared: float
    n_points: int
    timeseries_path: Path
    parity_path: Path


def coefficient_of_determination(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Return ordinary R-squared after excluding non-finite pairs."""
    valid = np.isfinite(truth) & np.isfinite(predicted)
    observed = truth[valid]
    estimate = predicted[valid]
    if not observed.size:
        raise ValueError("no finite truth/prediction pairs")
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return float(1.0 - np.sum((estimate - observed) ** 2) / denominator)


def _evaluated_pairs(case: PredictionCase) -> tuple[np.ndarray, np.ndarray]:
    start = case.evaluation_start_index
    truth = case.truth[start:]
    predicted = case.predicted[start:]
    valid = np.isfinite(truth) & np.isfinite(predicted)
    return truth[valid], predicted[valid]


def _representative_case(cases: Sequence[PredictionCase]) -> PredictionCase:
    def rmse(case: PredictionCase) -> float:
        truth, predicted = _evaluated_pairs(case)
        return float(np.sqrt(np.mean((predicted - truth) ** 2))) if truth.size else -np.inf

    return max(cases, key=rmse)


def _axis_bounds(truth: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    values = np.concatenate((truth, predicted))
    lower = float(values.min())
    upper = float(values.max())
    margin = max(0.03 * (upper - lower), 0.25)
    return lower - margin, upper + margin


def _plot_timeseries(
    case: PredictionCase,
    sensor_names: Sequence[str],
    target: Path,
    *,
    title: str,
    temperature_unit: str,
) -> None:
    figure, axes = plt.subplots(
        len(sensor_names),
        1,
        figsize=(8.2, 2.0 * len(sensor_names) + 0.8),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for index, sensor in enumerate(sensor_names):
        axis = axes[index, 0]
        axis.plot(case.time, case.truth[:, index], color="#202124", linewidth=1.8, label="True")
        axis.plot(
            case.time,
            case.predicted[:, index],
            color="#1f5a94",
            linestyle="--",
            linewidth=1.6,
            label="Fitted RC",
        )
        axis.set_ylabel(f"{sensor}\n[{temperature_unit}]")
        axis.grid(color="#d9dde3", linewidth=0.7, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        if index == 0:
            axis.legend(frameon=False, ncols=2, loc="best")
    axes[-1, 0].set_xlabel("Time [s]")
    figure.suptitle(f"{title}\nWorst held-out case: {case.case_id}", fontsize=12)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_parity(
    cases: Sequence[PredictionCase],
    sensor_names: Sequence[str],
    target: Path,
    *,
    title: str,
    temperature_unit: str,
) -> tuple[float, int]:
    truth_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    figure, axis = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    for index, sensor in enumerate(sensor_names):
        sensor_truth = np.concatenate(
            [case.truth[case.evaluation_start_index :, index] for case in cases]
        )
        sensor_predicted = np.concatenate(
            [case.predicted[case.evaluation_start_index :, index] for case in cases]
        )
        valid = np.isfinite(sensor_truth) & np.isfinite(sensor_predicted)
        sensor_truth = sensor_truth[valid]
        sensor_predicted = sensor_predicted[valid]
        truth_parts.append(sensor_truth)
        predicted_parts.append(sensor_predicted)
        axis.scatter(
            sensor_truth,
            sensor_predicted,
            s=17,
            alpha=0.48,
            color=_SENSOR_COLORS[index % len(_SENSOR_COLORS)],
            edgecolors="none",
            label=sensor,
            rasterized=True,
        )

    truth = np.concatenate(truth_parts)
    predicted = np.concatenate(predicted_parts)
    lower, upper = _axis_bounds(truth, predicted)
    r_squared = coefficient_of_determination(truth, predicted)
    axis.plot([lower, upper], [lower, upper], color="#202124", linestyle="--", linewidth=1.2)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(f"True temperature [{temperature_unit}]")
    axis.set_ylabel(f"Predicted temperature [{temperature_unit}]")
    axis.set_title(f"{title}\nAll held-out predictions")
    axis.grid(color="#d9dde3", linewidth=0.7, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    r_squared_text = "undefined" if np.isnan(r_squared) else f"{r_squared:.5f}"
    axis.text(
        0.04,
        0.96,
        f"R² = {r_squared_text}\nn = {truth.size:,}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#9aa0a6"},
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return r_squared, int(truth.size)


def write_prediction_figures(
    cases: Sequence[PredictionCase],
    target: Path,
    *,
    sensor_names: Sequence[str],
    title: str,
    file_prefix: str = "prediction",
    temperature_unit: str = "°C",
) -> PredictionFigureResult:
    """Write one conservative waveform and one pooled held-out parity plot."""
    if not cases:
        raise ValueError("at least one prediction case is required")
    sensor_names = tuple(sensor_names)
    if not sensor_names or any(case.truth.shape[1] != len(sensor_names) for case in cases):
        raise ValueError("sensor_names must match every prediction column")

    representative = _representative_case(cases)
    timeseries_path = target / f"{file_prefix}_timeseries.png"
    parity_path = target / f"{file_prefix}_parity.png"
    _plot_timeseries(
        representative,
        sensor_names,
        timeseries_path,
        title=title,
        temperature_unit=temperature_unit,
    )
    r_squared, n_points = _plot_parity(
        cases,
        sensor_names,
        parity_path,
        title=title,
        temperature_unit=temperature_unit,
    )
    return PredictionFigureResult(
        representative_case=representative.case_id,
        r_squared=r_squared,
        n_points=n_points,
        timeseries_path=timeseries_path,
        parity_path=parity_path,
    )
