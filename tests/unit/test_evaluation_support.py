import json

import numpy as np
import pandas as pd

from external_tools.comsol_chip_cooling.evaluation_support import (
    json_records,
    percent_improvement,
)


def test_json_records_returns_standard_json_scalars() -> None:
    frame = pd.DataFrame(
        [
            {
                "count": np.int64(2),
                "enabled": np.bool_(True),
                "score": np.float64(1.5),
                "missing": np.nan,
                "infinite": np.inf,
                "nullable": pd.NA,
                "not_a_time": pd.NaT,
            }
        ]
    )

    records = json_records(frame)

    assert records == [
        {
            "count": 2,
            "enabled": True,
            "score": 1.5,
            "missing": None,
            "infinite": None,
            "nullable": None,
            "not_a_time": None,
        }
    ]
    json.dumps(records, allow_nan=False)


def test_percent_improvement_preserves_existing_zero_baseline_semantics() -> None:
    assert percent_improvement(10.0, 2.0) == 80.0
    assert percent_improvement(0.0, 2.0) == 0.0
