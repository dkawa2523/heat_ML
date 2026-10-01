"""Small table helpers shared by the COMSOL evaluation scripts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def assert_aligned(source: pd.DataFrame, result: pd.DataFrame, case_id: str) -> None:
    """Require an evaluation result to preserve its source rows and time axis."""
    if len(source) != len(result):
        raise ValueError(f"{case_id}: source and result row counts differ")
    if not np.allclose(source["time"], result["time"], rtol=0.0, atol=1e-9):
        raise ValueError(f"{case_id}: source and result times differ")


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a table to records accepted by ``json.dumps(allow_nan=False)``."""
    columns = list(frame.columns)
    return [
        {column: _json_value(value) for column, value in zip(columns, row, strict=True)}
        for row in frame.itertuples(index=False, name=None)
    ]


def percent_improvement(baseline: float, candidate: float) -> float:
    """Return the relative error reduction; zero baselines have no reduction scale."""
    return float(100.0 * (baseline - candidate) / baseline) if baseline > 0.0 else 0.0
