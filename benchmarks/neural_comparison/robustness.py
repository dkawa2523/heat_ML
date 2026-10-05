"""Observation-noise screening for fixed, clean-trained forecast models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from celltemp.analysis import prediction_error_metrics
from celltemp.domain import Trajectory
from celltemp.engine import ThermalRCModel
from celltemp.inference import forecast

from .evaluation import validate_case_metrics
from .internal import initial_condition_request
from .training import NeuralTrainingResult, SequencePreprocessor, forecast_sequence

NOISE_STD_LEVELS_K = (0.0, 0.15, 0.50)
NOISE_REPEATS = 5


@dataclass(frozen=True)
class NoiseEvaluationCase:
    """One causal request and its clean truth for a single evaluation boundary."""

    evaluation: str
    boundary: str
    category: str
    request: Trajectory
    truth_from_origin: np.ndarray
    metric_start: int


def internal_test_noise_cases(
    dataset: str,
    trajectories: Sequence[Trajectory],
) -> list[NoiseEvaluationCase]:
    """Build held-out requests that expose only the initial temperature row."""
    return [
        NoiseEvaluationCase(
            evaluation=dataset,
            boundary="internal_test",
            category="configured_data_pool",
            request=initial_condition_request(trajectory),
            truth_from_origin=trajectory.temperature,
            metric_start=1,
        )
        for trajectory in trajectories
    ]


def add_observation_noise(
    trajectory: Trajectory,
    *,
    noise_std_k: float,
    rng: np.random.Generator,
) -> tuple[Trajectory, float]:
    """Perturb observed temperatures only and return the realized noise RMS."""
    if not np.isfinite(noise_std_k) or noise_std_k < 0.0:
        raise ValueError("noise_std_k must be finite and non-negative")
    if noise_std_k == 0.0:
        return trajectory, 0.0
    temperature = np.array(trajectory.temperature, copy=True)
    sampled_noise = rng.normal(0.0, noise_std_k, size=int(trajectory.mask.sum()))
    temperature[trajectory.mask] += sampled_noise
    noisy = Trajectory(
        case_id=trajectory.case_id,
        time=trajectory.time,
        temperature=temperature,
        commands=trajectory.commands,
        sensor_names=trajectory.sensor_names,
        control_names=trajectory.control_names,
        observation_mask=trajectory.mask,
        metadata=trajectory.metadata,
        initial_actuator=trajectory.initial_actuator,
    )
    return noisy, float(np.sqrt(np.mean(sampled_noise**2)))


def evaluate_observation_noise(
    cases: Sequence[NoiseEvaluationCase],
    rc_model: ThermalRCModel,
    trained: Mapping[str, NeuralTrainingResult],
    preprocessor: SequencePreprocessor,
    history_steps: int,
    *,
    seed: int,
    noise_std_levels_k: Sequence[float] = NOISE_STD_LEVELS_K,
    repeats: int = NOISE_REPEATS,
) -> list[dict[str, object]]:
    """Score fixed models after perturbing only their causal temperature observations."""
    if not cases:
        return []
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    levels = tuple(float(level) for level in noise_std_levels_k)
    if not levels or levels[0] != 0.0 or any(level < 0.0 for level in levels):
        raise ValueError("noise levels must start at zero and be non-negative")

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for noise_std_k in levels:
        level_repeats = 1 if noise_std_k == 0.0 else repeats
        for repeat in range(level_repeats):
            for case in cases:
                request, realized_noise_rmse_k = add_observation_noise(
                    case.request,
                    noise_std_k=noise_std_k,
                    rng=rng,
                )
                rc_result = forecast(rc_model, request)
                origin = rc_result.forecast_origin_index
                if origin + len(case.truth_from_origin) > len(request.time):
                    raise ValueError(f"truth for {request.case_id} is not aligned to its origin")
                predictions = {"physical_rc": rc_result.sensor_temperature}
                predictions.update(
                    {
                        name: forecast_sequence(
                            result.model,
                            request,
                            preprocessor,
                            history_steps=history_steps,
                            origin=origin,
                        )
                        for name, result in trained.items()
                    }
                )
                truth = case.truth_from_origin[case.metric_start :]
                for model_name, full_prediction in predictions.items():
                    predicted = full_prediction[case.metric_start :]
                    metrics = prediction_error_metrics(truth, predicted)
                    rows.append(
                        {
                            "evaluation": case.evaluation,
                            "boundary": case.boundary,
                            "category": case.category,
                            "case_id": request.case_id,
                            "model": model_name,
                            "noise_std_k": noise_std_k,
                            "repeat": repeat,
                            "observed_temperature_points": int(request.mask.sum()),
                            "realized_noise_rmse_k": realized_noise_rmse_k,
                            "forecast_origin_index": origin,
                            **metrics,
                        }
                    )
    return rows


def summarize_observation_noise(case_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate case errors without mixing internal and external boundaries."""
    validate_case_metrics(case_metrics)
    keys = ["evaluation", "boundary", "category", "model", "noise_std_k"]
    repeat_keys = [*keys, "repeat"]
    repeat_summary = (
        case_metrics.groupby(repeat_keys, sort=False)
        .agg(
            mean_case_rmse_k=("rmse_k", "mean"),
            worst_case_rmse_k=("rmse_k", "max"),
            mean_realized_noise_rmse_k=("realized_noise_rmse_k", "mean"),
            n_cases=("case_id", "nunique"),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for key, repeats_frame in repeat_summary.groupby(keys, sort=False):
        evaluation, boundary, category, model, noise_std_k = key
        baseline = repeat_summary[
            (repeat_summary["evaluation"] == evaluation)
            & (repeat_summary["boundary"] == boundary)
            & (repeat_summary["category"] == category)
            & (repeat_summary["model"] == model)
            & (repeat_summary["noise_std_k"] == 0.0)
        ]
        if len(baseline) != 1:
            raise ValueError(f"expected one clean baseline for {evaluation}/{boundary}/{model}")
        clean_rmse = float(baseline["mean_case_rmse_k"].iat[0])
        mean_rmse = float(repeats_frame["mean_case_rmse_k"].mean())
        rows.append(
            {
                "evaluation": evaluation,
                "boundary": boundary,
                "category": category,
                "model": model,
                "noise_std_k": float(noise_std_k),
                "n_cases": int(repeats_frame["n_cases"].iat[0]),
                "n_repeats": len(repeats_frame),
                "mean_case_rmse_k": mean_rmse,
                "repeat_mean_rmse_std_k": float(repeats_frame["mean_case_rmse_k"].std(ddof=0)),
                "worst_case_rmse_k": float(repeats_frame["worst_case_rmse_k"].max()),
                "mean_realized_noise_rmse_k": float(
                    repeats_frame["mean_realized_noise_rmse_k"].mean()
                ),
                "delta_vs_clean_k": mean_rmse - clean_rmse,
                "ratio_to_clean": mean_rmse / clean_rmse if clean_rmse > 0.0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_rc_model_gap(summary: pd.DataFrame) -> pd.DataFrame:
    """Pair RC core and model-gap evidence so omitted physics stays visible."""
    effect_by_evaluation = {
        "topcell": "temperature-dependent heat loss",
        "nonlinear_comsol": "surface-to-surface radiation",
        "high_fidelity_comsol": "surface-to-surface radiation",
    }
    rc = summary[summary["model"] == "physical_rc"]
    rows: list[dict[str, object]] = []
    for evaluation, omitted_effect in effect_by_evaluation.items():
        core = rc[(rc["evaluation"] == evaluation) & (rc["category"] == "core")]
        gap = rc[(rc["evaluation"] == evaluation) & (rc["category"] == "model_gap")]
        if len(core) != 1 or len(gap) != 1:
            continue
        core_rmse = float(core["mean_case_rmse_k"].iat[0])
        gap_rmse = float(gap["mean_case_rmse_k"].iat[0])
        rows.append(
            {
                "evaluation": evaluation,
                "omitted_effect": omitted_effect,
                "core_cases": int(core["n_cases"].iat[0]),
                "model_gap_cases": int(gap["n_cases"].iat[0]),
                "core_mean_case_rmse_k": core_rmse,
                "model_gap_mean_case_rmse_k": gap_rmse,
                "model_gap_to_core_ratio": gap_rmse / core_rmse,
            }
        )
    return pd.DataFrame(rows)
