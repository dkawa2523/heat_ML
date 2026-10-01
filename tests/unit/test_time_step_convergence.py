from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from external_tools.comsol_chip_cooling import run_time_step_convergence
from external_tools.comsol_chip_cooling.summarize_high_fidelity import (
    _dynamic_reference,
    _temporal_evidence,
)


def _time_step_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    fine = pd.DataFrame(
        {
            "time": [0.0, 20.0],
            "chip_power": [0.0, 12.0],
            "coolant_temperature": [25.0, 25.0],
            "inlet_air_velocity": [0.1, 0.1],
            "truth_chip": [25.0, 35.0],
            "truth_sink_base": [25.0, 34.0],
            "truth_fins": [25.0, 33.0],
            "truth_chip_max": [25.0, 35.5],
            "truth_fins_max": [25.0, 33.5],
            "truth_outlet_air_temperature": [25.0, 27.0],
            "truth_pressure_drop": [1.0, 2.0],
            "truth_radiative_heat_rate": [-0.2, -0.4],
        }
    )
    coarse = fine.copy()
    for quantity in ("truth_chip", "truth_sink_base", "truth_fins"):
        coarse[quantity] += [0.01, 0.04]
    for quantity in ("truth_chip_max", "truth_fins_max"):
        coarse[quantity] += [0.02, 0.08]
    coarse["truth_outlet_air_temperature"] += [0.01, 0.04]
    coarse["truth_pressure_drop"] += [0.001, 0.005]
    coarse["truth_radiative_heat_rate"] += [0.0005, 0.001]
    return coarse, fine


def test_time_step_comparison_uses_identical_knots_and_predeclared_limits(
    tmp_path: Path,
) -> None:
    coarse, fine = _time_step_frames()

    evidence = run_time_step_convergence.compare_time_steps(
        coarse,
        fine,
        source_case_id="case_a",
        mesh_profile="local-medium",
        coarse_max_step_s=2.0,
        fine_max_step_s=1.0,
    )
    evidence.to_csv(tmp_path / "time_step_convergence.csv", index=False)
    summary = _temporal_evidence(tmp_path)

    assert len(evidence) == 8
    assert evidence["passed"].all()
    chip = evidence.set_index("quantity").loc["chip"]
    assert chip["comparison_metric"] == "max_abs_difference"
    assert chip["acceptance_limit"] == 0.05
    assert chip["observed_value"] == pytest.approx(0.04)
    assert summary is not None
    assert summary["qualified"] is True
    assert summary["sensor_uncertainty"] == pytest.approx(
        {"chip": 0.04, "sink_base": 0.04, "fins": 0.04}
    )

    radiation = fine.rename(
        columns={
            "truth_chip": "truth_chip",
            "truth_sink_base": "truth_sink_base",
            "truth_fins": "truth_fins",
        }
    ).assign(case_id="case_a", mesh_profile="local-medium")
    steady = pd.DataFrame(
        {
            "mesh_uncertainty_chip": [0.4],
            "mesh_uncertainty_sink_base": [0.4],
            "mesh_uncertainty_fins": [0.4],
            "benchmark_qualified": [True],
        }
    )
    reference = _dynamic_reference(radiation, steady, summary)
    assert reference["temporal_qualified"].all()
    for sensor in ("chip", "sink_base", "fins"):
        np.testing.assert_allclose(reference[f"temporal_uncertainty_{sensor}"], 0.04)


def test_time_step_comparison_rejects_changed_control_or_non_refinement() -> None:
    coarse, fine = _time_step_frames()
    fine.loc[1, "chip_power"] = 11.0

    with pytest.raises(ValueError, match="identical output times and controls"):
        run_time_step_convergence.compare_time_steps(
            coarse,
            fine,
            source_case_id="case_a",
            mesh_profile="local-medium",
            coarse_max_step_s=2.0,
            fine_max_step_s=1.0,
        )

    with pytest.raises(ValueError, match="fine maximum time step"):
        run_time_step_convergence.compare_time_steps(
            coarse,
            coarse,
            source_case_id="case_a",
            mesh_profile="local-medium",
            coarse_max_step_s=1.0,
            fine_max_step_s=2.0,
        )


def test_temporal_evidence_rejects_incomplete_quantities(tmp_path: Path) -> None:
    coarse, fine = _time_step_frames()
    evidence = run_time_step_convergence.compare_time_steps(
        coarse,
        fine,
        source_case_id="case_a",
        mesh_profile="local-medium",
        coarse_max_step_s=2.0,
        fine_max_step_s=1.0,
    )
    evidence.iloc[:-1].to_csv(tmp_path / "time_step_convergence.csv", index=False)

    with pytest.raises(ValueError, match="incomplete time-step quantities"):
        _temporal_evidence(tmp_path)
