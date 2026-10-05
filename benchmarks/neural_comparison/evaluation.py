"""Fail-closed validation of model-comparison evidence tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def validate_matching_splits(expected: Mapping[str, Sequence[str]], saved: pd.DataFrame) -> None:
    """Prevent an existing RC artifact from being scored on cases it fitted."""
    if not {"case_id", "split"} <= set(saved) or saved["case_id"].duplicated().any():
        raise ValueError("RC training evidence needs unique case_id and split assignments")
    actual = dict(zip(saved["case_id"].astype(str), saved["split"].astype(str), strict=True))
    requested = {case_id: split for split, cases in expected.items() for case_id in cases}
    if actual != requested:
        raise ValueError(
            "RC artifact split differs from the neural training split; retrain RC first"
        )


def validate_training_inputs(expected: Mapping[str, tuple[str, str]], saved: object) -> None:
    """Bind a reused model to the exact input bytes and partition it fitted."""
    if not isinstance(saved, list) or not saved:
        raise ValueError("RC artifact lacks hashed training inputs; retrain RC first")
    actual: dict[str, tuple[str, str]] = {}
    for item in saved:
        if not isinstance(item, dict):
            raise ValueError(
                "RC artifact training inputs must contain case, split, and SHA256 records"
            )
        case_id = str(item.get("case_id", ""))
        if not case_id or case_id in actual:
            raise ValueError("RC artifact training inputs must contain unique case identifiers")
        actual[case_id] = (str(item.get("split", "")), str(item.get("sha256", "")))
    if actual != dict(expected):
        raise ValueError("RC artifact training input bytes or splits changed; retrain RC first")


def validate_case_metrics(frame: pd.DataFrame) -> None:
    """Require each declared case to contribute finite error and observed truth."""
    required = {"rmse_k", "n_points"}
    if frame.empty or not required <= set(frame):
        raise ValueError("case metrics need non-empty RMSE and evaluation point counts")
    errors = [name for name in ("rmse_k", "mae_k", "max_abs_error_k") if name in frame]
    if not np.isfinite(frame[errors].to_numpy(dtype=float)).all():
        raise ValueError("every case error metric must be finite; failed cases cannot be omitted")
    counts = frame["n_points"].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts <= 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("every case must contain a positive integer evaluation point count")
    if "n_truth_points" in frame and not frame["n_points"].eq(frame["n_truth_points"]).all():
        raise ValueError("every observed truth value must have a scored prediction")
    if (
        "prediction_coverage_fraction" in frame
        and not frame["prediction_coverage_fraction"].eq(1.0).all()
    ):
        raise ValueError("prediction coverage must be complete for every case")


def validate_prediction_points(frame: pd.DataFrame) -> None:
    """Permit missing truth, while rejecting every non-finite prediction."""
    if not np.isfinite(frame["predicted"].to_numpy(dtype=float)).all():
        raise ValueError("prediction points must all be finite")
    if np.isinf(frame["truth"].to_numpy(dtype=float)).any():
        raise ValueError("truth points may be missing, but cannot contain infinity")


def validate_evaluation_points(
    cases: pd.DataFrame, points: pd.DataFrame, keys: Sequence[str]
) -> None:
    """Ensure retained waveforms cover exactly the points declared by each case score."""
    validate_prediction_points(points)
    if points.duplicated([*keys, "time", "sensor"]).any():
        raise ValueError(
            "prediction points must be unique within each case, model, time, and sensor"
        )
    expected = cases.set_index(list(keys))["n_points"]
    if expected.index.has_duplicates:
        raise ValueError("case metrics must contain one score per evaluation case and model")
    actual = (
        points.assign(_truth_observed=np.isfinite(points["truth"].to_numpy(dtype=float)))
        .groupby(list(keys), sort=False, dropna=False)["_truth_observed"]
        .sum()
    )
    if set(expected.index) != set(actual.index) or not np.array_equal(
        expected.to_numpy(), actual.reindex(expected.index).to_numpy()
    ):
        raise ValueError(
            "prediction point coverage differs from the declared case evaluation counts"
        )
