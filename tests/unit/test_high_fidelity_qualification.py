import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from external_tools.comsol_chip_cooling.qualification_support import (
    build_dynamic_reference,
    load_verified_dynamic_reference,
    mesh_evidence,
    raw_provenance_verified,
    require_mesh_qualified,
    verify_transient_solver_settings,
    write_raw_provenance,
)
from external_tools.comsol_chip_cooling.run_high_fidelity import DEFAULT_ROOT
from external_tools.comsol_chip_cooling.summarize_high_fidelity import (
    _radiation_delta,
    summarize,
)


def _radiation() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_ROOT / "dynamic/eval/model_gap/HV02_composite_radiation.csv")


def _copy_mesh_evidence(tmp_path: Path) -> None:
    for name in ("mesh_convergence", "mesh_convergence_deltas", "mesh_acceptance"):
        path = DEFAULT_ROOT / f"{name}.csv"
        (tmp_path / path.name).write_bytes(path.read_bytes())


def _copy_dynamic_reference(tmp_path: Path) -> Path:
    _copy_mesh_evidence(tmp_path)
    reference_path = tmp_path / "dynamic" / "cae_reference.csv"
    reference_path.parent.mkdir()
    reference_path.write_bytes((DEFAULT_ROOT / "dynamic" / "cae_reference.csv").read_bytes())
    (tmp_path / "time_step_convergence.csv").write_bytes(
        (DEFAULT_ROOT / "time_step_convergence.csv").read_bytes()
    )
    return reference_path


def test_reference_consumer_verifies_public_screening_without_promoting_legacy_time_evidence() -> (
    None
):
    reference = load_verified_dynamic_reference(DEFAULT_ROOT, _radiation())
    assert reference["benchmark_qualified"].all()
    assert not reference["mesh_qualified"].any()
    assert not reference["temporal_qualified"].any()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("mesh_profile", "local-coarse"),
        ("case_id", "other_case"),
        ("source_model", "other_solver"),
        ("truth_hidden_power", 1.0),
        ("surface_emissivity_sink", 0.80),
        ("truth_chip_max", 100.0),
    ],
)
def test_reference_consumer_rejects_changed_source_with_identical_sensor_temperatures(
    column: str, value: str | float
) -> None:
    source = _radiation().assign(**{column: value})
    with pytest.raises(ValueError, match="reference"):
        load_verified_dynamic_reference(DEFAULT_ROOT, source)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("benchmark_qualified", False),
        ("temporal_qualified", True),
        ("mesh_uncertainty_chip", 0.01),
        ("transient_result_fingerprint", "unverified"),
        ("mesh_evidence_sha256", "{}"),
    ],
)
def test_reference_consumer_rejects_relabelled_publication(
    tmp_path: Path, column: str, value: str | float | bool
) -> None:
    path = _copy_dynamic_reference(tmp_path)
    pd.read_csv(path).assign(**{column: value}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=f"reference {column}"):
        load_verified_dynamic_reference(tmp_path, _radiation())


def test_reference_consumer_rejects_changed_mesh_evidence(tmp_path: Path) -> None:
    _copy_dynamic_reference(tmp_path)
    path = tmp_path / "mesh_convergence.csv"
    fields = pd.read_csv(path)
    fields.loc[fields["mesh_profile"] == "local-fine", "chip_average_c"] = 100.0
    fields.to_csv(path, index=False)
    with pytest.raises(ValueError, match="source/evidence"):
        load_verified_dynamic_reference(tmp_path, _radiation())


def test_reference_consumer_requires_publication_provenance(tmp_path: Path) -> None:
    path = _copy_dynamic_reference(tmp_path)
    pd.read_csv(path).drop(columns="transient_result_fingerprint").to_csv(path, index=False)
    with pytest.raises(ValueError, match="reference schema"):
        load_verified_dynamic_reference(tmp_path, _radiation())


def test_reference_consumer_rejects_unmatched_temporal_uncertainty(tmp_path: Path) -> None:
    path = _copy_dynamic_reference(tmp_path)
    pd.read_csv(path).assign(temporal_uncertainty_chip=0.01).to_csv(path, index=False)
    with pytest.raises(ValueError, match="reference schema"):
        load_verified_dynamic_reference(tmp_path, _radiation())


def test_unqualified_current_reference_remains_available_for_screening(tmp_path: Path) -> None:
    _copy_dynamic_reference(tmp_path)
    source = _radiation().assign(mesh_profile="local-coarse")
    expected = build_dynamic_reference(source, mesh_evidence(tmp_path, "local-coarse"), None)
    expected.to_csv(tmp_path / "dynamic/cae_reference.csv", index=False)
    reference = load_verified_dynamic_reference(tmp_path, source)
    assert not reference["benchmark_qualified"].any()
    assert reference["mesh_profile"].eq("local-coarse").all()


def test_qualification_libraries_do_not_import_solver_runners_or_product_core() -> None:
    tool_root = DEFAULT_ROOT.parent.parent
    for name in ("qualification.py", "qualification_support.py"):
        tree = ast.parse((tool_root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith(("celltemp", "run_", "summarize_"))
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith(("celltemp", "torch")) for alias in node.names)


@pytest.mark.parametrize(
    ("profile", "qualified"),
    [("local-coarse", False), ("local-medium", True), ("local-fine", False), ("unknown", False)],
)
def test_mesh_qualification_uses_actual_profile(profile: str, qualified: bool) -> None:
    mesh = mesh_evidence(DEFAULT_ROOT, profile)
    radiation = _radiation().assign(mesh_profile=profile)
    reference = build_dynamic_reference(radiation, mesh, None)
    assert reference["benchmark_qualified"].all() == qualified
    assert not reference["mesh_qualified"].any()


def test_screening_qualification_does_not_extend_outside_study_boundary() -> None:
    radiation = _radiation()
    radiation.loc[1, "chip_power"] = 13.0
    reference = build_dynamic_reference(
        radiation, mesh_evidence(DEFAULT_ROOT, "local-medium"), None
    )
    assert not reference["benchmark_qualified"].any()


def test_medium_evidence_cannot_qualify_coarse_data() -> None:
    radiation = _radiation().assign(mesh_profile="local-coarse")
    reference = build_dynamic_reference(
        radiation, mesh_evidence(DEFAULT_ROOT, "local-medium"), None
    )
    assert not reference["benchmark_qualified"].any()
    assert reference["mesh_uncertainty_chip"].isna().all()


def test_mesh_flags_must_agree_with_numerical_fields(tmp_path: Path) -> None:
    _copy_mesh_evidence(tmp_path)
    path = tmp_path / "mesh_acceptance.csv"
    acceptance = pd.read_csv(path)
    acceptance.loc[acceptance["mesh_profile"] == "local-coarse", "all_cases_pass_benchmark"] = True
    acceptance.to_csv(path, index=False)
    assert mesh_evidence(tmp_path, "local-coarse")["benchmark_qualified"] is False
    with pytest.raises(ValueError, match="not benchmark-qualified"):
        require_mesh_qualified("local-coarse", tmp_path)


def test_mesh_differences_must_match_the_field_values(tmp_path: Path) -> None:
    _copy_mesh_evidence(tmp_path)
    path = tmp_path / "mesh_convergence_deltas.csv"
    differences = pd.read_csv(path)
    medium = differences["mesh_profile"] == "local-medium"
    differences.loc[medium, "chip_average_c_absolute_delta"] = 0
    differences.to_csv(path, index=False)
    assert mesh_evidence(tmp_path, "local-medium")["benchmark_qualified"] is False


def test_partial_or_duplicate_mesh_cases_cannot_qualify(tmp_path: Path) -> None:
    _copy_mesh_evidence(tmp_path)
    path = tmp_path / "mesh_convergence.csv"
    fields = pd.read_csv(path)
    pd.concat([fields, fields.iloc[[0]]]).to_csv(path, index=False)
    assert mesh_evidence(tmp_path, "local-medium")["benchmark_qualified"] is False


def test_radiation_pair_requires_identical_numerical_settings() -> None:
    radiation = _radiation()
    base = radiation.assign(mesh_profile="local-coarse")
    with pytest.raises(ValueError, match="settings: mesh_profile"):
        _radiation_delta(base, radiation)


@pytest.mark.parametrize(
    "declaration",
    [
        "mesh profile local-medium, maximum BDF step 2.0 s",
        "mesh profile local-coarse, maximum BDF step 1.0 s",
        "mesh profile local-medium, adaptive maximum time step",
    ],
)
def test_raw_reuse_requires_the_requested_solver_settings(
    tmp_path: Path,
    declaration: str,
) -> None:
    path = tmp_path / "solve.log"
    path.write_text(
        f"Solving radiation transient with 12 output times, {declaration} (sol4)", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="BDF step"):
        verify_transient_solver_settings(path, mesh_profile="local-medium", maximum_time_step_s=1.0)


def test_solver_log_verifies_fixed_step_and_requires_evidence(tmp_path: Path) -> None:
    path = tmp_path / "solve.log"
    with pytest.raises(ValueError, match="missing solver-settings evidence"):
        verify_transient_solver_settings(path, mesh_profile="local-medium", maximum_time_step_s=1.0)
    path.write_text(
        "Solving radiation transient with 12 output times, mesh profile local-medium, "
        "maximum BDF step 1.0 s (sol4)",
        encoding="utf-8",
    )
    verify_transient_solver_settings(path, mesh_profile="local-medium", maximum_time_step_s=1.0)


@pytest.mark.parametrize("changed", ["raw", "log", "schedule", "java", "settings"])
def test_raw_provenance_binds_settings_to_the_executed_result(
    tmp_path: Path,
    changed: str,
) -> None:
    paths = [tmp_path / name for name in ("raw.txt", "solve.log", "schedule.csv", "runner.java")]
    for path in paths:
        path.write_text("original", encoding="utf-8")
    kwargs = {"mesh_profile": "local-medium", "maximum_time_step_s": 1.0}
    assert not raw_provenance_verified(*paths, **kwargs)
    write_raw_provenance(*paths, **kwargs)
    assert raw_provenance_verified(*paths, **kwargs)
    if changed == "settings":
        kwargs["maximum_time_step_s"] = 2.0
    else:
        index = ("raw", "log", "schedule", "java").index(changed)
        paths[index].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="raw solver provenance differs"):
        raw_provenance_verified(*paths, **kwargs)


def test_screening_qualification_requires_known_physics() -> None:
    radiation = _radiation().assign(truth_hidden_power=1.0)
    reference = build_dynamic_reference(
        radiation, mesh_evidence(DEFAULT_ROOT, "local-medium"), None
    )
    assert not reference["benchmark_qualified"].any()


@pytest.mark.parametrize(
    ("profile", "qualified"), [("local-medium", True), ("local-coarse", False), ("unknown", False)]
)
def test_summary_retains_public_screening_and_uses_actual_dataset_qualification(
    tmp_path: Path,
    profile: str,
    qualified: bool,
) -> None:
    _copy_mesh_evidence(tmp_path)
    for name in ("cae_reference.csv", "time_step_convergence.csv"):
        (tmp_path / name).write_bytes((DEFAULT_ROOT / name).read_bytes())
    for name in (
        "dynamic/eval/forecast/HV01_composite_conjugate.csv",
        "dynamic/eval/model_gap/HV02_composite_radiation.csv",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(DEFAULT_ROOT / name).assign(mesh_profile=profile).to_csv(target, index=False)
    (tmp_path / "experiment").mkdir()
    summarize(tmp_path)
    reference = pd.read_csv(tmp_path / "dynamic/cae_reference.csv")
    quality = json.loads((tmp_path / "quality_summary.json").read_text(encoding="utf-8"))
    assert reference["benchmark_qualified"].all() == qualified
    assert quality["benchmark_mesh_qualified"] == qualified
    assert not quality["transient_time_discretization_qualified"]
    assert not any(str(name).startswith("temporal_uncertainty_") for name in reference)
    source = pd.read_csv(DEFAULT_ROOT / "dynamic/eval/model_gap/HV02_composite_radiation.csv")
    for sensor in ("chip", "sink_base", "fins"):
        pd.testing.assert_series_equal(
            reference[sensor], source[f"truth_{sensor}"], check_names=False
        )
    if profile == "unknown":
        assert quality["max_adjacent_mesh_temperature_difference_k"] is None
