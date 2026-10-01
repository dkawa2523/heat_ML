"""Mechanical helpers shared by the linear and nonlinear dataset adapters."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


def left_limits(values: np.ndarray) -> np.ndarray:
    """Sample the completed interval at each output knot."""
    sampled = np.asarray(values, dtype=np.float64)
    return np.concatenate([sampled[:1], sampled[:-1]])


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Replace one CSV only after its complete temporary file is closed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            frame.to_csv(stream, index=False, float_format="%.10g")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
