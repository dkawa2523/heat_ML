from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from external_tools.comsol_chip_cooling import run_time_step_convergence
from external_tools.comsol_chip_cooling.qualification_support import (
    build_dynamic_reference,
    load_verified_dynamic_reference,
    mesh_evidence,
    temporal_evidence,
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
            "truth_hidden_power": [0.0, 0.0],
            "truth_effective_air_velocity": [0.1, 0.1],
            "radiation_enabled": [True, True],
            "surface_emissivity_sink": [0.9, 0.9],
            "surface_emissivity_channel": [0.85, 0.85],
            "mesh_profile": ["local-medium", "local-medium"],
            "input_convention": ["left_zero_order_hold", "left_zero_order_hold"],
            "source_model": ["COMSOL 6.4 Electronic Chip Cooling"] * 2,
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


def _publish_comparison(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    coarse, fine = _time_step_frames()
    for result, step in zip((coarse, fine), (2.0, 1.0), strict=True):
        result["source_case_id"] = "case_a"
        result["case_id"] = f"variant_{step:g}"
        result["mesh_profile"] = "local-medium"
        result["maximum_time_step_s"] = step
        result["solver_settings_verified"] = True
    evidence, _ = run_time_step_convergence._publish_time_step_results(
        [coarse, fine],
        steps=(2.0, 1.0),
        data_root=tmp_path,
        source_case_id="case_a",
        mesh_profile="local-medium",
    )
    return evidence, fine.assign(case_id="case_a")


def _mesh_evidence() -> dict[str, object]:
    return {
        "mesh_profile": "local-medium",
        "benchmark_qualified": True,
        "sensor_uncertainty": dict.fromkeys(("chip", "sink_base", "fins"), 0.4),
        "basis": "local-medium versus local-fine at MC01/MC02",
    }


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
    _, radiation = _publish_comparison(tmp_path)
    summary = temporal_evidence(tmp_path)

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

    reference = build_dynamic_reference(radiation, _mesh_evidence(), summary)
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
        temporal_evidence(tmp_path)


def test_legacy_time_step_numbers_cannot_qualify_another_solve(tmp_path: Path) -> None:
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
    summary = temporal_evidence(tmp_path)
    assert summary is not None
    assert summary["comparison_passed"] is True
    assert summary["qualified"] is False
    assert summary["result_provenance_verified"] is False


@pytest.mark.parametrize("changed", ["adaptive", "truth", "case", "mesh", "step", "control"])
def test_temporal_qualification_belongs_to_the_actual_fine_result(
    tmp_path: Path,
    changed: str,
) -> None:
    _, radiation = _publish_comparison(tmp_path)
    if changed == "adaptive":
        radiation = radiation.drop(columns=["maximum_time_step_s", "solver_settings_verified"])
    elif changed == "truth":
        radiation.loc[1, "truth_chip"] = (
            radiation["truth_chip"].to_numpy(dtype=np.float64)[1] + 0.001
        )
    elif changed == "case":
        radiation["case_id"] = "case_b"
    elif changed == "mesh":
        radiation["mesh_profile"] = "local-coarse"
    elif changed == "step":
        radiation["maximum_time_step_s"] = 2.0
    elif changed == "control":
        radiation.loc[1, "chip_power"] = 11.0
    reference = build_dynamic_reference(radiation, _mesh_evidence(), temporal_evidence(tmp_path))
    assert not reference["temporal_qualified"].any()
    assert not any(str(name).startswith("temporal_uncertainty_") for name in reference)


def test_temporal_result_file_tampering_is_rejected(tmp_path: Path) -> None:
    _publish_comparison(tmp_path)
    path = tmp_path / "temporal" / "max_step_1s.csv"
    result = pd.read_csv(path)
    result.loc[1, "truth_chip"] = result["truth_chip"].to_numpy(dtype=np.float64)[1] + 1.0
    result.to_csv(path, index=False)
    with pytest.raises(ValueError, match="file hash does not match"):
        temporal_evidence(tmp_path)


def test_temporal_flags_must_agree_with_the_fixed_budget(tmp_path: Path) -> None:
    evidence, _ = _publish_comparison(tmp_path)
    evidence["passed"] = False
    evidence.to_csv(tmp_path / "time_step_convergence.csv", index=False)
    with pytest.raises(ValueError, match="declared acceptance budget"):
        temporal_evidence(tmp_path)


def test_temporal_budget_and_numbers_cannot_be_relabelled_as_passed(tmp_path: Path) -> None:
    evidence, _ = _publish_comparison(tmp_path)
    evidence.loc[evidence["quantity"] == "chip", "max_abs_difference"] = 1.0
    evidence["passed"] = True
    evidence.to_csv(tmp_path / "time_step_convergence.csv", index=False)
    with pytest.raises(ValueError, match="comparison differs"):
        temporal_evidence(tmp_path)
    evidence.loc[evidence["quantity"] == "chip", "acceptance_limit"] = 2.0
    evidence.to_csv(tmp_path / "time_step_convergence.csv", index=False)
    with pytest.raises(ValueError, match="declared budget"):
        temporal_evidence(tmp_path)


def test_time_step_comparison_rejects_different_physics_and_duplicate_times() -> None:
    coarse, fine = _time_step_frames()
    coarse["surface_emissivity_sink"] = 0.9
    fine["surface_emissivity_sink"] = 0.8
    kwargs = {
        "source_case_id": "case_a",
        "mesh_profile": "local-medium",
        "coarse_max_step_s": 2.0,
        "fine_max_step_s": 1.0,
    }
    with pytest.raises(ValueError, match="physical settings"):
        run_time_step_convergence.compare_time_steps(coarse, fine, **kwargs)
    fine["time"] = 0.0
    with pytest.raises(ValueError, match="increasing output times"):
        run_time_step_convergence.compare_time_steps(coarse, fine, **kwargs)


def test_reference_consumer_accepts_the_verified_fine_result_and_its_uncertainty(
    tmp_path: Path,
) -> None:
    _, radiation = _publish_comparison(tmp_path)
    source_root = run_time_step_convergence.DEFAULT_DATA_ROOT
    for name in ("mesh_convergence.csv", "mesh_convergence_deltas.csv", "mesh_acceptance.csv"):
        (tmp_path / name).write_bytes((source_root / name).read_bytes())
    reference = build_dynamic_reference(
        radiation, mesh_evidence(tmp_path, "local-medium"), temporal_evidence(tmp_path)
    )
    path = tmp_path / "dynamic/cae_reference.csv"
    path.parent.mkdir()
    reference.to_csv(path, index=False)
    loaded = load_verified_dynamic_reference(tmp_path, radiation, case_id="case_a")
    assert loaded["temporal_qualified"].all()
    np.testing.assert_allclose(loaded["temporal_uncertainty_chip"], 0.04)
