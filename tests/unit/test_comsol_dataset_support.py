from pathlib import Path

import numpy as np
import pandas as pd

from external_tools.comsol_chip_cooling.dataset_support import (
    left_limits,
    write_csv_atomic,
)


def test_left_limits_uses_the_completed_interval_value() -> None:
    result = left_limits(np.array([2, 4, 7]))

    np.testing.assert_array_equal(result, np.array([2.0, 2.0, 4.0]))


def test_write_csv_atomic_replaces_the_complete_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "trajectory.csv"
    write_csv_atomic(pd.DataFrame({"time": [0.0], "temperature": [25.0]}), target)
    replacement = pd.DataFrame({"time": [0.0, 1.0], "temperature": [26.0, 27.5]})

    write_csv_atomic(replacement, target)

    pd.testing.assert_frame_equal(pd.read_csv(target), replacement, check_dtype=False)
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))
