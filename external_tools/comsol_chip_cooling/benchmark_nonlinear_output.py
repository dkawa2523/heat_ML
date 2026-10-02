"""Publish nonlinear benchmark tables, predictions, and validation figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.validation_figures import VALIDATION_FIGURE_DIRECTORY
from celltemp.workflows.common import staged_output_directory
from celltemp.workflows.prediction_figures import PredictionCase, write_prediction_figures


def _write_csv(frame: pd.DataFrame, target: Path) -> None:
    frame.to_csv(target, index=False, float_format="%.10g", lineterminator="\n")


def _prediction_cases(
    frames: dict[str, pd.DataFrame], sensor_names: tuple[str, ...]
) -> list[PredictionCase]:
    return [
        PredictionCase(
            case_id=case_id,
            time=frame["time"].to_numpy(dtype=float),
            truth=frame[[f"truth_{sensor}" for sensor in sensor_names]].to_numpy(dtype=float),
            predicted=frame[[f"predicted_{sensor}" for sensor in sensor_names]].to_numpy(
                dtype=float
            ),
        )
        for case_id, frame in frames.items()
    ]


def publish_benchmark(
    output: Path,
    *,
    case_metrics: pd.DataFrame,
    group_metrics: pd.DataFrame,
    sensor_metrics: pd.DataFrame,
    model_comparison: pd.DataFrame,
    residual_dependence: pd.DataFrame,
    radiation_pair_metrics: pd.DataFrame,
    prediction_frames: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    sensor_names: tuple[str, ...],
) -> None:
    """Replace the benchmark directory after every output has been written."""
    tables = (
        ("case_metrics.csv", case_metrics),
        ("group_metrics.csv", group_metrics),
        ("sensor_metrics.csv", sensor_metrics),
        ("model_comparison.csv", model_comparison),
        ("residual_dependence.csv", residual_dependence),
        ("radiation_pair_metrics.csv", radiation_pair_metrics),
    )
    with staged_output_directory(output.resolve(), overwrite=True) as target:
        for name, frame in tables:
            _write_csv(frame, target / name)

        prediction_dir = target / "predictions"
        prediction_dir.mkdir()
        for case_id, frame in prediction_frames.items():
            _write_csv(frame, prediction_dir / f"{case_id}.csv")

        cases = _prediction_cases(prediction_frames, sensor_names)
        groups = case_metrics.set_index("case_id")["case_group"].astype(str).to_dict()
        non_radiation = [case for case in cases if groups[case.case_id] != "radiation"]
        radiation = [case for case in cases if groups[case.case_id] == "radiation"]
        write_prediction_figures(
            non_radiation,
            VALIDATION_FIGURE_DIRECTORY,
            sensor_names=sensor_names,
            title="Nonlinear COMSOL forecast — non-radiation cases",
            file_prefix="nonlinear_comsol_core_prediction",
        )
        write_prediction_figures(
            radiation,
            VALIDATION_FIGURE_DIRECTORY,
            sensor_names=sensor_names,
            title="Nonlinear COMSOL forecast — radiation model gap",
            file_prefix="nonlinear_comsol_model_gap_prediction",
        )
        (target / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
            newline="\n",
        )
