"""Train historical neural baselines on current data and compare them with physical RC."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from benchmarks.validation_figures import VALIDATION_FIGURE_DIRECTORY
from celltemp.analysis import prediction_error_metrics
from celltemp.artifact import load_artifact
from celltemp.config import as_path, load_config
from celltemp.domain import Trajectory
from celltemp.engine import ThermalRCModel
from celltemp.inference import forecast
from celltemp.io import load_system_spec, load_trajectories
from celltemp.learning import split_trajectories

from .internal import evaluate_internal_splits
from .models import MODEL_TYPES, SequenceModelSpec
from .reporting import summarize_internal_predictions, summarize_predictions, write_figures
from .robustness import (
    NOISE_REPEATS,
    NOISE_STD_LEVELS_K,
    NoiseEvaluationCase,
    evaluate_observation_noise,
    internal_test_noise_cases,
    summarize_observation_noise,
    summarize_rc_model_gap,
)
from .training import (
    NeuralTrainingConfig,
    NeuralTrainingResult,
    SequencePreprocessor,
    fit_model,
    forecast_sequence,
    prepare_training_data,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "docs" / "neural_model_comparison"
MODEL_NAMES = tuple(MODEL_TYPES)


@dataclass(frozen=True)
class EvaluationSource:
    evaluation: str
    directory: Path
    category: str
    exclude_origin: bool


@dataclass(frozen=True)
class BenchmarkSpec:
    dataset: str
    config_path: Path
    artifact_path: Path
    evaluations: tuple[EvaluationSource, ...]


@dataclass(frozen=True)
class TrainingProblem:
    config: dict[str, Any]
    sensor_names: tuple[str, ...]
    control_names: tuple[str, ...]
    splits: dict[str, list[Trajectory]]
    source_directory: Path


def benchmark_specs() -> tuple[BenchmarkSpec, ...]:
    topcell = REPOSITORY_ROOT / "benchmarks" / "topcell"
    comsol = REPOSITORY_ROOT / "external_tools" / "comsol_chip_cooling"
    return (
        BenchmarkSpec(
            dataset="topcell",
            config_path=topcell / "config.yaml",
            artifact_path=topcell / "work" / "outputs" / "runs" / "topcell" / "artifact",
            evaluations=(
                EvaluationSource(
                    "topcell",
                    topcell / "work" / "data" / "eval" / "forecast",
                    "auto",
                    False,
                ),
            ),
        ),
        BenchmarkSpec(
            dataset="linear_comsol",
            config_path=comsol / "config.yaml",
            artifact_path=(
                comsol / "work" / "outputs" / "runs" / "comsol_chip_cooling" / "artifact"
            ),
            evaluations=(
                EvaluationSource(
                    "linear_comsol",
                    comsol / "data" / "eval" / "forecast",
                    "core",
                    True,
                ),
            ),
        ),
        BenchmarkSpec(
            dataset="nonlinear_comsol",
            config_path=comsol / "high_fidelity_benchmark.yaml",
            artifact_path=(
                comsol
                / "work"
                / "high_fidelity_benchmark"
                / "runs"
                / "nonlinear_high_fidelity_network"
                / "artifact"
            ),
            evaluations=(
                EvaluationSource(
                    "nonlinear_comsol",
                    comsol / "data" / "nonlinear" / "eval" / "forecast",
                    "core",
                    True,
                ),
                EvaluationSource(
                    "nonlinear_comsol",
                    comsol / "data" / "nonlinear" / "eval" / "model_gap",
                    "model_gap",
                    True,
                ),
                EvaluationSource(
                    "high_fidelity_comsol",
                    comsol / "data" / "nonlinear_high_fidelity" / "dynamic" / "eval" / "forecast",
                    "core",
                    True,
                ),
                EvaluationSource(
                    "high_fidelity_comsol",
                    comsol / "data" / "nonlinear_high_fidelity" / "dynamic" / "eval" / "model_gap",
                    "model_gap",
                    True,
                ),
            ),
        ),
    )


def _trajectory_config(
    base: dict[str, Any],
    sensor_names: tuple[str, ...],
    control_names: tuple[str, ...],
    *,
    directory: Path | None = None,
    allow_missing: bool = False,
) -> dict[str, Any]:
    values = dict(base)
    values["sensor_cols"] = sensor_names
    values["control_cols"] = control_names
    values["allow_missing_temperatures"] = allow_missing
    if directory is not None:
        values["directory"] = str(directory)
        values["pattern"] = "*.csv"
        values["dt"] = None
    return values


def _load_training_problem(spec: BenchmarkSpec) -> TrainingProblem:
    config = load_config(spec.config_path)
    config_root = spec.config_path.parent
    system = load_system_spec(as_path(str(config["system"]), config_root))
    data_config = _trajectory_config(
        config["data"],
        system.sensor_names,
        system.control_names,
    )
    trajectories = load_trajectories(data_config, config_root)
    splits = split_trajectories(trajectories, config.get("split"), seed=int(config["seed"]))
    return TrainingProblem(
        config=config,
        sensor_names=system.sensor_names,
        control_names=system.control_names,
        splits=splits,
        source_directory=as_path(str(config["data"]["directory"]), config_root).resolve(),
    )


def _training_history_rows(
    dataset: str,
    model: str,
    result: NeuralTrainingResult,
) -> list[dict[str, object]]:
    return [{"dataset": dataset, "model": model, **row} for row in result.history]


def _save_training_artifact(
    directory: Path,
    model_name: str,
    result: NeuralTrainingResult,
    preprocessor: SequencePreprocessor,
    model_spec: SequenceModelSpec,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), directory / f"{model_name}.pt")
    metadata = {
        "model": model_name,
        "history_steps": model_spec.history_steps,
        "hidden_dim": model_spec.hidden_dim,
        "dropout": model_spec.dropout,
        "best_epoch": result.best_epoch,
        "best_validation_open_loop_rmse_k": result.best_validation_rmse,
        "parameter_count": result.parameter_count,
        "elapsed_seconds": result.elapsed_seconds,
        "preprocessor": preprocessor.as_dict(),
    }
    (directory / f"{model_name}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _category(source: EvaluationSource, frame: pd.DataFrame) -> str:
    if source.category != "auto":
        return source.category
    column = "benchmark_group" if "benchmark_group" in frame else "case_group"
    return "model_gap" if str(frame[column].iat[0]) == "model_gap" else "core"


def _load_requests(
    source: EvaluationSource,
    sensor_names: tuple[str, ...],
    control_names: tuple[str, ...],
) -> list[Any]:
    values = {
        "directory": str(source.directory),
        "pattern": "*.csv",
        "time_col": "time",
        "sensor_cols": sensor_names,
        "control_cols": control_names,
        "control_convention": "left",
        "sep": ",",
        "allow_missing_temperatures": True,
        "dt": None,
    }
    return load_trajectories(values, REPOSITORY_ROOT)


def _prediction_rows(
    evaluation: str,
    category: str,
    case_id: str,
    model_name: str,
    time: np.ndarray,
    sensor_names: tuple[str, ...],
    truth: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for time_index, timestamp in enumerate(time):
        for sensor_index, sensor in enumerate(sensor_names):
            rows.append(
                {
                    "evaluation": evaluation,
                    "category": category,
                    "case_id": case_id,
                    "model": model_name,
                    "time": float(timestamp),
                    "sensor": sensor,
                    "truth": float(truth[time_index, sensor_index]),
                    "predicted": float(predicted[time_index, sensor_index]),
                }
            )
    return rows


def _evaluate_source(
    source: EvaluationSource,
    rc_model: ThermalRCModel,
    sensor_names: tuple[str, ...],
    control_names: tuple[str, ...],
    trained: dict[str, NeuralTrainingResult],
    preprocessor: SequencePreprocessor,
    history_steps: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[NoiseEvaluationCase],
]:
    requests = _load_requests(source, sensor_names, control_names)
    case_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    noise_cases: list[NoiseEvaluationCase] = []
    for request in requests:
        frame = pd.read_csv(str(request.metadata["path"]))
        category = _category(source, frame)
        rc_result = forecast(rc_model, request)
        origin = rc_result.forecast_origin_index
        truth_columns = [f"truth_{sensor}" for sensor in sensor_names]
        truth = frame[truth_columns].to_numpy(dtype=float)[origin:]
        start = 1 if source.exclude_origin else 0
        if category == "core":
            noise_cases.append(
                NoiseEvaluationCase(
                    evaluation=source.evaluation,
                    boundary="external_core",
                    category=category,
                    request=request,
                    truth_from_origin=truth,
                    metric_start=start,
                )
            )
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
        for model_name, predicted in predictions.items():
            evaluated_truth = truth[start:]
            evaluated_prediction = predicted[start:]
            metrics = prediction_error_metrics(evaluated_truth, evaluated_prediction)
            case_rows.append(
                {
                    "evaluation": source.evaluation,
                    "category": category,
                    "case_id": request.case_id,
                    "model": model_name,
                    "forecast_origin_index": origin,
                    **metrics,
                }
            )
            point_rows.extend(
                _prediction_rows(
                    source.evaluation,
                    category,
                    request.case_id,
                    model_name,
                    request.time[origin + start :],
                    sensor_names,
                    evaluated_truth,
                    evaluated_prediction,
                )
            )
    return case_rows, point_rows, noise_cases


def _train_benchmark(
    spec: BenchmarkSpec,
    problem: TrainingProblem,
    training_config: NeuralTrainingConfig,
    work_directory: Path,
) -> tuple[
    dict[str, NeuralTrainingResult],
    SequencePreprocessor,
    SequenceModelSpec,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    train_cases = problem.splits["train"]
    validation_cases = problem.splits["val"]
    preprocessor = SequencePreprocessor.fit(train_cases)
    model_spec = SequenceModelSpec(
        n_sensors=len(problem.sensor_names),
        n_controls=len(problem.control_names),
        history_steps=8,
    )
    training_data = prepare_training_data(
        train_cases,
        preprocessor,
        history_steps=model_spec.history_steps,
    )
    trained: dict[str, NeuralTrainingResult] = {}
    history_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        print(f"[{spec.dataset}] training {model_name}", flush=True)
        result = fit_model(
            model_name,
            model_spec,
            training_data,
            validation_cases,
            preprocessor,
            config=training_config,
            seed=int(problem.config["seed"]),
        )
        trained[model_name] = result
        history_rows.extend(_training_history_rows(spec.dataset, model_name, result))
        summary_rows.append(
            {
                "dataset": spec.dataset,
                "model": model_name,
                "best_epoch": result.best_epoch,
                "best_validation_open_loop_rmse_k": result.best_validation_rmse,
                "parameter_count": result.parameter_count,
                "training_seconds": result.elapsed_seconds,
                "train_cases": len(train_cases),
                "validation_cases": len(validation_cases),
                "test_cases": len(problem.splits["test"]),
            }
        )
        _save_training_artifact(
            work_directory / spec.dataset,
            model_name,
            result,
            preprocessor,
            model_spec,
        )
        print(
            f"[{spec.dataset}] {model_name}: epoch={result.best_epoch} "
            f"val_rmse={result.best_validation_rmse:.6f} K",
            flush=True,
        )
    return trained, preprocessor, model_spec, history_rows, summary_rows


def _repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _internal_boundary_rows(
    spec: BenchmarkSpec,
    problem: TrainingProblem,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    definitions = {
        "train": "同一設定データ集合。係数・重みの更新に使用するin-sample予測。",
        "validation": "同一設定データ集合。更新には使わずbest epochの選択に使用。",
        "test": "同一設定データ集合。更新にも選択にも使わない内部保留case。",
    }
    for raw_split, boundary in (("train", "train"), ("val", "validation"), ("test", "test")):
        cases = problem.splits[raw_split]
        rows.append(
            {
                "training_dataset": spec.dataset,
                "evaluation_dataset": spec.dataset,
                "scope": "internal",
                "boundary": boundary,
                "category": "configured_data_pool",
                "source_directory": _repository_path(problem.source_directory),
                "directory_passed_to_splitter": True,
                "used_for_parameter_fit": boundary == "train",
                "used_for_model_selection": boundary == "validation",
                "primary_evaluation": boundary == "test",
                "forecast_observations": "time index 0 only; future truth absent",
                "metric_origin": "time index 0 excluded",
                "n_cases": len(cases),
                "case_ids": ";".join(sorted(case.case_id for case in cases)),
                "definition_ja": definitions[boundary],
            }
        )
    return rows


def _external_boundary_rows(
    spec: BenchmarkSpec,
    source: EvaluationSource,
    case_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    frame = pd.DataFrame(case_rows)
    rows: list[dict[str, object]] = []
    for category, category_rows in frame.groupby("category", sort=False):
        case_ids = sorted(category_rows["case_id"].astype(str).unique())
        rows.append(
            {
                "training_dataset": spec.dataset,
                "evaluation_dataset": source.evaluation,
                "scope": "external",
                "boundary": "external",
                "category": category,
                "source_directory": _repository_path(source.directory),
                "directory_passed_to_splitter": False,
                "used_for_parameter_fit": False,
                "used_for_model_selection": False,
                "primary_evaluation": category == "core",
                "forecast_observations": "dataset-defined observed prefix; origin from mask",
                "metric_origin": (
                    "forecast origin excluded"
                    if source.exclude_origin
                    else "forecast origin included"
                ),
                "n_cases": len(case_ids),
                "case_ids": ";".join(case_ids),
                "definition_ja": (
                    "別ディレクトリ。splitter・係数学習・epoch選択へ未投入。"
                    "実機妥当化済みを意味しない。"
                ),
            }
        )
    return rows


def run(output_directory: Path, *, epochs: int) -> Path:
    """Execute three training problems plus internal and external forecasts."""
    torch.set_num_threads(min(8, torch.get_num_threads()))
    training_config = NeuralTrainingConfig(epochs=epochs)
    work_directory = REPOSITORY_ROOT / "benchmarks" / "neural_comparison" / "work"
    all_case_rows: list[dict[str, object]] = []
    all_point_rows: list[dict[str, object]] = []
    all_history_rows: list[dict[str, object]] = []
    all_training_rows: list[dict[str, object]] = []
    all_internal_case_rows: list[dict[str, object]] = []
    all_internal_test_points: list[dict[str, object]] = []
    all_boundary_rows: list[dict[str, object]] = []
    all_noise_case_rows: list[dict[str, object]] = []

    for spec_index, spec in enumerate(benchmark_specs()):
        problem = _load_training_problem(spec)
        trained, preprocessor, model_spec, history_rows, training_rows = _train_benchmark(
            spec,
            problem,
            training_config,
            work_directory,
        )
        all_history_rows.extend(history_rows)
        all_training_rows.extend(training_rows)
        artifact = load_artifact(spec.artifact_path)
        internal_case_rows, internal_test_points = evaluate_internal_splits(
            spec.dataset,
            artifact.model,
            problem.splits,
            trained,
            preprocessor,
            model_spec.history_steps,
        )
        all_internal_case_rows.extend(internal_case_rows)
        all_internal_test_points.extend(internal_test_points)
        all_boundary_rows.extend(_internal_boundary_rows(spec, problem))
        all_noise_case_rows.extend(
            evaluate_observation_noise(
                internal_test_noise_cases(spec.dataset, problem.splits["test"]),
                artifact.model,
                trained,
                preprocessor,
                model_spec.history_steps,
                seed=int(problem.config["seed"]) + spec_index * 10_000,
            )
        )
        for source_index, source in enumerate(spec.evaluations):
            print(f"[{source.evaluation}] evaluating {source.directory.name}", flush=True)
            case_rows, point_rows, noise_cases = _evaluate_source(
                source,
                artifact.model,
                problem.sensor_names,
                problem.control_names,
                trained,
                preprocessor,
                model_spec.history_steps,
            )
            all_case_rows.extend(case_rows)
            all_point_rows.extend(point_rows)
            all_boundary_rows.extend(_external_boundary_rows(spec, source, case_rows))
            all_noise_case_rows.extend(
                evaluate_observation_noise(
                    noise_cases,
                    artifact.model,
                    trained,
                    preprocessor,
                    model_spec.history_steps,
                    seed=(
                        int(problem.config["seed"])
                        + spec_index * 10_000
                        + (source_index + 1) * 1_000
                    ),
                )
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    case_metrics = pd.DataFrame(all_case_rows)
    prediction_points = pd.DataFrame(all_point_rows)
    training_history = pd.DataFrame(all_history_rows)
    training_summary = pd.DataFrame(all_training_rows)
    internal_case_metrics = pd.DataFrame(all_internal_case_rows)
    internal_test_points = pd.DataFrame(all_internal_test_points)
    evaluation_boundaries = pd.DataFrame(all_boundary_rows)
    noise_case_metrics = pd.DataFrame(all_noise_case_rows)
    summary = summarize_predictions(case_metrics, prediction_points)
    internal_summary = summarize_internal_predictions(internal_case_metrics, internal_test_points)
    noise_summary = summarize_observation_noise(noise_case_metrics)
    rc_model_gap_summary = summarize_rc_model_gap(summary)
    case_metrics.to_csv(output_directory / "case_metrics.csv", index=False)
    prediction_points.to_csv(output_directory / "prediction_points.csv", index=False)
    training_history.to_csv(output_directory / "training_history.csv", index=False)
    training_summary.to_csv(output_directory / "training_summary.csv", index=False)
    summary.to_csv(output_directory / "summary.csv", index=False)
    internal_case_metrics.to_csv(output_directory / "internal_case_metrics.csv", index=False)
    internal_test_points.to_csv(
        output_directory / "internal_test_prediction_points.csv",
        index=False,
    )
    internal_summary.to_csv(output_directory / "internal_summary.csv", index=False)
    evaluation_boundaries.to_csv(output_directory / "evaluation_boundaries.csv", index=False)
    noise_case_metrics.to_csv(output_directory / "noise_case_metrics.csv", index=False)
    noise_summary.to_csv(output_directory / "noise_summary.csv", index=False)
    rc_model_gap_summary.to_csv(output_directory / "rc_model_gap_summary.csv", index=False)
    protocol = {
        "seed": 42,
        "models": list(MODEL_NAMES),
        "history_steps": 8,
        "target": "standardized temperature rate dT/dt",
        "training_loss": "case-balanced one-step Huber",
        "model_selection": "mean complete causal open-loop validation RMSE",
        "internal_definition": (
            "same configured data directory, whole-case train/validation/test split"
        ),
        "internal_forecast": (
            "time-index 0 temperature only, then commands and dt; future truth absent; "
            "index 0 excluded from metrics"
        ),
        "primary_internal_result": "held-out test; unused by fitting and model selection",
        "external_definition": (
            "separate evaluation directories never passed to splitting, fitting, or model selection"
        ),
        "external_forecast": "observed prefix then commands and dt only",
        "observation_noise_screening": {
            "scope": (
                "fixed clean-trained models; Gaussian noise added only to causally observed "
                "temperature inputs; scored against clean truth"
            ),
            "boundaries": ["internal_test", "external_core"],
            "noise_std_k": list(NOISE_STD_LEVELS_K),
            "positive_noise_repeats": NOISE_REPEATS,
            "excluded_uncertainty": [
                "training-label noise",
                "command uncertainty",
                "reference-truth noise",
                "parameter uncertainty",
                "model-form uncertainty",
            ],
        },
        "metric_origin": (
            "TopCell includes its estimated origin; COMSOL excludes initialization row"
        ),
        "epochs_max": epochs,
        "device": "cpu",
        "note": "TPU is not a model in repository history; TCN is evaluated.",
    }
    (output_directory / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_figures(
        summary,
        case_metrics,
        prediction_points,
        training_history,
        internal_summary,
        internal_case_metrics,
        internal_test_points,
        evaluation_boundaries,
        noise_summary,
        rc_model_gap_summary,
        VALIDATION_FIGURE_DIRECTORY,
    )
    return output_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=300)
    arguments = parser.parse_args()
    if arguments.epochs <= 0:
        parser.error("--epochs must be positive")
    output = run(arguments.output_dir.resolve(), epochs=arguments.epochs)
    print(f"wrote neural comparison to {output}")


if __name__ == "__main__":
    main()
