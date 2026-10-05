"""Bind screening qualifications to the mesh, boundary, and numerical result."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from .nonlinear_cases import mesh_convergence_cases
from .qualification import (
    PHYSICS_COLUMNS,
    PROFILE_RANK,
    QOI_TOLERANCES,
    RELATIVE_LIMITS,
    RESULT_COLUMNS,
    SENSORS,
    TEMPERATURE_LIMITS_K,
    compare_meshes,
    compare_time_steps,
)
from .qualification import (
    result_fingerprint as result_fingerprint,
)


def true_values(values: pd.Series) -> pd.Series:
    """Accept explicit booleans only; the string 'False' is never truthy."""
    return values.astype(str).str.lower().eq("true")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_transient_solver_settings(
    log_path: Path, *, mesh_profile: str, maximum_time_step_s: float | None
) -> None:
    """Require the runner's explicit mesh/BDF declaration for raw reuse."""
    if not log_path.is_file():
        raise ValueError(f"missing solver-settings evidence: {log_path}")
    declarations = re.findall(
        r"transient with \d+ output times, mesh profile ([\w-]+), "
        r"(maximum BDF step [0-9.eE+-]+ s|adaptive maximum time step)",
        log_path.read_text(encoding="utf-8"),
    )
    if len(declarations) != 1:
        raise ValueError(f"unverified maximum BDF step in {log_path}")
    profile, declaration = declarations[0]
    actual_step = None if declaration.startswith("adaptive") else float(declaration.split()[3])
    if profile != mesh_profile or actual_step != maximum_time_step_s:
        raise ValueError(
            f"solver mesh/maximum BDF step differs from requested settings: {log_path}"
        )


def _raw_provenance(
    raw_path: Path,
    log_path: Path,
    schedule_path: Path,
    java_source: Path,
    *,
    mesh_profile: str,
    maximum_time_step_s: float | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "raw_result_sha256": file_sha256(raw_path),
        "solver_log_sha256": file_sha256(log_path),
        "schedule_sha256": file_sha256(schedule_path),
        "java_source_sha256": file_sha256(java_source),
        "mesh_profile": mesh_profile,
        "maximum_time_step_s": maximum_time_step_s,
    }


def write_raw_provenance(
    raw_path: Path,
    log_path: Path,
    schedule_path: Path,
    java_source: Path,
    *,
    mesh_profile: str,
    maximum_time_step_s: float | None,
) -> None:
    """Record the complete result only after this invocation has run successfully."""
    provenance = _raw_provenance(
        raw_path,
        log_path,
        schedule_path,
        java_source,
        mesh_profile=mesh_profile,
        maximum_time_step_s=maximum_time_step_s,
    )
    target = raw_path.with_suffix(raw_path.suffix + ".provenance.json")
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(provenance, stream, sort_keys=True, allow_nan=False)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def raw_provenance_verified(
    raw_path: Path,
    log_path: Path,
    schedule_path: Path,
    java_source: Path,
    *,
    mesh_profile: str,
    maximum_time_step_s: float | None,
) -> bool:
    """Unrecorded legacy raw tables remain usable for unqualified screening."""
    path = raw_path.with_suffix(raw_path.suffix + ".provenance.json")
    if not path.is_file():
        return False
    actual = _raw_provenance(
        raw_path,
        log_path,
        schedule_path,
        java_source,
        mesh_profile=mesh_profile,
        maximum_time_step_s=maximum_time_step_s,
    )
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded != actual:
        raise ValueError(f"raw solver provenance differs from the current files/settings: {path}")
    return True


def _unknown_mesh(profile: str, reason: str) -> dict[str, object]:
    return {
        "mesh_profile": profile,
        "strict_qualified": False,
        "benchmark_qualified": False,
        "sensor_uncertainty": None,
        "basis": reason,
    }


def mesh_evidence(root: Path, profile: str) -> dict[str, object]:
    """Recompute qualification for the actual profile from both stationary fields."""
    paths = {
        name: root / f"{name}.csv"
        for name in ("mesh_convergence", "mesh_convergence_deltas", "mesh_acceptance")
    }
    if any(not path.is_file() for path in paths.values()):
        return _unknown_mesh(profile, "complete mesh evidence is unavailable")
    qoi = pd.read_csv(paths["mesh_convergence"])
    supplied = pd.read_csv(paths["mesh_convergence_deltas"])
    acceptance = pd.read_csv(paths["mesh_acceptance"])
    numeric_qoi = [*QOI_TOLERANCES, "pressure_drop_pa", "radiative_heat_rate_w"]
    if not {"case_id", "mesh_profile", *numeric_qoi} <= set(qoi):
        return _unknown_mesh(profile, "stationary fields have an incomplete schema")
    if not np.isfinite(qoi[numeric_qoi].to_numpy(dtype=np.float64)).all():
        return _unknown_mesh(profile, "stationary fields contain non-finite quantities")
    expected_cases = {case.case_id for case in mesh_convergence_cases()}
    selected = supplied.loc[supplied["mesh_profile"] == profile]
    status = acceptance.loc[acceptance["mesh_profile"] == profile]
    if (
        len(selected) != len(expected_cases)
        or set(selected["case_id"]) != expected_cases
        or selected["finer_mesh_profile"].nunique() != 1
        or len(status) != 1
        or qoi.duplicated(["case_id", "mesh_profile"]).any()
    ):
        return _unknown_mesh(profile, "the actual mesh has no complete adjacent comparison")
    finer = str(selected["finer_mesh_profile"].iloc[0])
    if profile not in PROFILE_RANK or PROFILE_RANK.get(finer) != PROFILE_RANK[profile] + 1:
        return _unknown_mesh(profile, "mesh evidence is not an adjacent known refinement")
    recomputed = compare_meshes(qoi)
    actual = recomputed.loc[recomputed["mesh_profile"] == profile].set_index("case_id")
    selected = selected.set_index("case_id").sort_index()
    actual = actual.sort_index()
    if actual.index.tolist() != selected.index.tolist() or not actual["finer_mesh_profile"].equals(
        selected["finer_mesh_profile"]
    ):
        return _unknown_mesh(profile, "mesh comparison provenance does not match the fields")
    numeric = [name for name in actual if str(name).endswith(("_delta", "_tolerance"))]
    if not np.allclose(
        actual[numeric].to_numpy(dtype=np.float64),
        selected[numeric].to_numpy(dtype=np.float64),
        rtol=1e-6,
        atol=1e-7,
        equal_nan=True,
    ):
        return _unknown_mesh(profile, "mesh comparison differs from the stationary fields")
    strict = bool(actual["passes_qoi_tolerances"].all())
    benchmark = bool(actual["passes_benchmark_tolerances"].all())
    if (
        strict != bool(true_values(status["all_cases_pass"]).iloc[0])
        or benchmark != bool(true_values(status["all_cases_pass_benchmark"]).iloc[0])
        or int(status["comparison_cases"].iloc[0]) != len(expected_cases)
    ):
        return _unknown_mesh(profile, "mesh acceptance differs from the stationary fields")
    uncertainty = {
        sensor: float(selected[f"{quantity}_absolute_delta"].max())
        for sensor, quantity in zip(
            SENSORS, ("chip_average_c", "sink_base_average_c", "fins_average_c"), strict=True
        )
    }
    return {
        "mesh_profile": profile,
        "strict_qualified": strict,
        "benchmark_qualified": benchmark,
        "sensor_uncertainty": uncertainty,
        "basis": (
            f"{profile} versus {selected['finer_mesh_profile'].iloc[0]} at MC01/MC02; "
            "steady-field evidence for nonlinear model-form screening"
        ),
        "evidence_sha256": {name: file_sha256(path) for name, path in paths.items()},
    }


def within_screening_boundary(frame: pd.DataFrame) -> bool:
    """The steady study covers this stated transient screening envelope only."""
    if not set(PHYSICS_COLUMNS) <= set(frame):
        return False
    fixed = {
        "source_model": "COMSOL 6.4 Electronic Chip Cooling",
        "input_convention": "left_zero_order_hold",
        "surface_emissivity_sink": 0.90,
        "surface_emissivity_channel": 0.85,
        "truth_hidden_power": 0.0,
    }
    if any(not frame[name].eq(value).all() for name, value in fixed.items()):
        return False
    if not frame["truth_effective_air_velocity"].equals(frame["inlet_air_velocity"]):
        return False
    cases = mesh_convergence_cases()
    for column in ("chip_power", "coolant_temperature", "inlet_air_velocity"):
        values = frame[column].to_numpy(dtype=np.float64)
        bounds = np.concatenate([getattr(case, column) for case in cases])
        minimum = 0.0 if column == "chip_power" else float(bounds.min())
        if (
            not np.isfinite(values).all()
            or (values < minimum).any()
            or (values > bounds.max()).any()
        ):
            return False
    return True


def require_mesh_qualified(profile: str, evidence_root: Path) -> None:
    """Require verified stationary screening evidence before running a transient."""
    path = evidence_root / "mesh_acceptance.csv"
    if not path.is_file():
        raise ValueError(f"missing mesh qualification evidence: {path}")
    if not mesh_evidence(evidence_root, profile)["benchmark_qualified"]:
        raise ValueError(
            f"mesh profile {profile} is not benchmark-qualified against its next refinement "
            f"in {path}"
        )


def validate_radiation_pair(base: pd.DataFrame, radiation: pd.DataFrame) -> None:
    """Require the radiation toggle to be the only changed boundary setting."""
    controls = ["time", "chip_power", "coolant_temperature", "inlet_air_velocity"]
    if (
        not set(controls) <= set(base)
        or not set(controls) <= set(radiation)
        or len(base) != len(radiation)
        or not np.isfinite(base[controls].to_numpy(dtype=np.float64)).all()
        or not np.isfinite(radiation[controls].to_numpy(dtype=np.float64)).all()
        or not np.allclose(
            base[controls].to_numpy(dtype=np.float64),
            radiation[controls].to_numpy(dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        )
    ):
        raise ValueError("high-fidelity radiation pair does not share identical inputs")
    for name in (
        "mesh_profile",
        "maximum_time_step_s",
        "solver_settings_verified",
        "source_model",
        "input_convention",
        "surface_emissivity_sink",
        "surface_emissivity_channel",
        "truth_hidden_power",
        "truth_effective_air_velocity",
    ):
        if (name in base or name in radiation) and (
            name not in base or name not in radiation or not base[name].equals(radiation[name])
        ):
            raise ValueError(f"high-fidelity radiation pair differs in settings: {name}")
    if (
        "radiation_enabled" not in base
        or "radiation_enabled" not in radiation
        or not base["radiation_enabled"].astype(str).str.lower().eq("false").all()
        or not true_values(radiation["radiation_enabled"]).all()
    ):
        raise ValueError("high-fidelity radiation pair must contain disabled/enabled radiation")


TEMPORAL_QUANTITIES = {
    "chip",
    "sink_base",
    "fins",
    "chip_max",
    "fins_max",
    "outlet_air_temperature",
    "pressure_drop",
    "radiative_heat_rate",
}


def temporal_evidence(root: Path) -> dict[str, object] | None:
    path = root / "time_step_convergence.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    required = {
        "coarse_max_time_step_s",
        "fine_max_time_step_s",
        "quantity",
        "max_abs_difference",
        "fine_reference_peak_abs",
        "acceptance_limit",
        "passed",
        "case_id",
        "mesh_profile",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path}: missing time-step evidence columns {sorted(missing)}")
    quantities = set(frame["quantity"].astype(str))
    if len(frame) != len(TEMPORAL_QUANTITIES) or quantities != TEMPORAL_QUANTITIES:
        raise ValueError(f"{path}: incomplete time-step quantities {sorted(quantities)}")
    steps = frame[["coarse_max_time_step_s", "fine_max_time_step_s"]].to_numpy(dtype=np.float64)
    differences = frame["max_abs_difference"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(steps).all()
        or (steps <= 0.0).any()
        or not np.isfinite(differences).all()
        or (differences < 0.0).any()
    ):
        raise ValueError(f"{path}: time-step limits must be positive and differences non-negative")
    coarse_steps = frame["coarse_max_time_step_s"].unique()
    fine_steps = frame["fine_max_time_step_s"].unique()
    if len(coarse_steps) != 1 or len(fine_steps) != 1 or fine_steps[0] >= coarse_steps[0]:
        raise ValueError(f"{path}: expected one adjacent coarse/fine time-step pair")
    indexed = frame.set_index("quantity")
    limits = {
        name.removeprefix("truth_"): value
        for name, value in {**TEMPERATURE_LIMITS_K, **RELATIVE_LIMITS}.items()
    }
    expected_limits = frame["quantity"].map(limits).to_numpy(dtype=np.float64)
    if not np.array_equal(frame["acceptance_limit"].to_numpy(dtype=np.float64), expected_limits):
        raise ValueError(f"{path}: time-step acceptance limits differ from the declared budget")
    observed = differences.copy()
    relative = frame["quantity"].isin(name.removeprefix("truth_") for name in RELATIVE_LIMITS)
    peaks = frame.loc[relative, "fine_reference_peak_abs"].to_numpy(dtype=np.float64)
    if not np.isfinite(peaks).all() or (peaks <= 1e-12).any():
        raise ValueError(f"{path}: relative quantities need a finite positive fine peak")
    observed[relative.to_numpy()] /= peaks
    passed = observed <= expected_limits
    verified = _verify_temporal_results(root, frame)
    published_flags = frame["passed"].astype(str).str.lower()
    if not published_flags.isin(["true", "false"]).all() or not np.array_equal(
        true_values(frame["passed"]).to_numpy(), passed
    ):
        raise ValueError("time-step comparison differs from the declared acceptance budget")
    sensor_differences = indexed.loc[list(SENSORS), "max_abs_difference"].to_numpy(dtype=np.float64)
    return {
        "qualified": bool(passed.all() and verified),
        "comparison_passed": bool(passed.all()),
        "result_provenance_verified": verified,
        "case_id": str(frame["case_id"].iloc[0]),
        "mesh_profile": str(frame["mesh_profile"].iloc[0]),
        "fine_result_fingerprint": (
            str(frame["fine_result_fingerprint"].iloc[0]) if verified else None
        ),
        "coarse_max_time_step_s": float(coarse_steps[0]),
        "fine_max_time_step_s": float(fine_steps[0]),
        "max_sensor_temperature_difference_k": float(sensor_differences.max()),
        "sensor_uncertainty": dict(zip(SENSORS, map(float, sensor_differences), strict=True)),
        "basis": (
            f"maximum BDF step {coarse_steps[0]:g} s versus {fine_steps[0]:g} s "
            "at identical output times and controls"
        ),
        "evidence": "time_step_convergence.csv",
    }


def _verify_temporal_results(root: Path, evidence: pd.DataFrame) -> bool:
    """Legacy numeric differences are useful, but cannot qualify another solve."""
    provenance = {
        f"{label}_result_{suffix}"
        for label in ("coarse", "fine")
        for suffix in ("path", "sha256", "fingerprint")
    }
    if not provenance <= set(evidence):
        return False
    metadata = ["case_id", "mesh_profile", *sorted(provenance)]
    if any(evidence[name].nunique(dropna=False) != 1 for name in metadata):
        raise ValueError("time-step evidence mixes result provenance")
    results = []
    for label in ("coarse", "fine"):
        path = (root / str(evidence[f"{label}_result_path"].iloc[0])).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("time-step result path must stay within the data root")
        if not path.is_file() or file_sha256(path) != evidence[f"{label}_result_sha256"].iloc[0]:
            raise ValueError(f"{label} time-step result file hash does not match evidence")
        result = pd.read_csv(path)
        expected_step = float(evidence[f"{label}_max_time_step_s"].iloc[0])
        required = {
            "source_case_id",
            "mesh_profile",
            "maximum_time_step_s",
            "solver_settings_verified",
            *PHYSICS_COLUMNS,
        }
        if not required <= set(result):
            return False
        if not true_values(result["solver_settings_verified"]).all():
            return False
        matching = (
            result["source_case_id"].eq(evidence["case_id"].iloc[0]).all()
            and result["mesh_profile"].eq(evidence["mesh_profile"].iloc[0]).all()
            and result["maximum_time_step_s"].eq(expected_step).all()
            and result_fingerprint(result) == evidence[f"{label}_result_fingerprint"].iloc[0]
        )
        if not matching:
            raise ValueError(f"{label} time-step result settings/fingerprint differ from evidence")
        results.append(result)
    recomputed = compare_time_steps(
        results[0],
        results[1],
        source_case_id=str(evidence["case_id"].iloc[0]),
        mesh_profile=str(evidence["mesh_profile"].iloc[0]),
        coarse_max_step_s=float(evidence["coarse_max_time_step_s"].iloc[0]),
        fine_max_step_s=float(evidence["fine_max_time_step_s"].iloc[0]),
    ).set_index("quantity")
    supplied = evidence.set_index("quantity").loc[recomputed.index]
    numeric = [
        "max_abs_difference",
        "fine_reference_peak_abs",
        "observed_value",
        "acceptance_limit",
    ]
    if not np.allclose(
        recomputed[numeric].to_numpy(dtype=np.float64),
        supplied[numeric].to_numpy(dtype=np.float64),
        rtol=1e-8,
        atol=1e-12,
    ):
        raise ValueError("time-step comparison differs from the recorded numerical results")
    return True


def _temporal_matches(radiation: pd.DataFrame, temporal: dict[str, object] | None) -> bool:
    if not temporal or not temporal["result_provenance_verified"]:
        return False
    if not {"maximum_time_step_s", "solver_settings_verified"} <= set(radiation):
        return False
    expected_step = temporal["fine_max_time_step_s"]
    if not isinstance(expected_step, (int, float)):
        return False
    return bool(
        radiation["case_id"].eq(str(temporal["case_id"])).all()
        and radiation["mesh_profile"].eq(str(temporal["mesh_profile"])).all()
        and radiation["maximum_time_step_s"].eq(expected_step).all()
        and true_values(radiation["solver_settings_verified"]).all()
        and result_fingerprint(radiation) == temporal["fine_result_fingerprint"]
    )


def build_dynamic_reference(
    radiation: pd.DataFrame,
    mesh: dict[str, object],
    temporal: dict[str, object] | None,
) -> pd.DataFrame:
    """Expose the physical transient at the same boundary used by experiments."""
    if radiation["mesh_profile"].nunique(dropna=False) != 1:
        raise ValueError("the dynamic reference must contain one mesh profile")
    matching_mesh = radiation["mesh_profile"].eq(str(mesh["mesh_profile"])).all()
    uncertainty = mesh["sensor_uncertainty"] if matching_mesh else None
    if not isinstance(uncertainty, dict):
        uncertainty = dict.fromkeys(SENSORS, np.nan)
    benchmark_qualified = bool(
        matching_mesh and mesh["benchmark_qualified"] and within_screening_boundary(radiation)
    )
    temporal_matches = _temporal_matches(radiation, temporal)
    temporal_qualified = bool(temporal_matches and temporal and temporal["qualified"])
    temporal_basis = (
        str(temporal["basis"])
        if temporal_matches and temporal is not None
        else "the actual transient is not the verified fine result of the time-step comparison"
    )
    columns: dict[str, object] = {
        "case_id": radiation["case_id"],
        "time": radiation["time"],
        "chip": radiation["truth_chip"],
        "sink_base": radiation["truth_sink_base"],
        "fins": radiation["truth_fins"],
        "chip_power": radiation["chip_power"],
        "coolant_temperature": radiation["coolant_temperature"],
        "inlet_air_velocity": radiation["inlet_air_velocity"],
        "mesh_uncertainty_chip": uncertainty["chip"],
        "mesh_uncertainty_sink_base": uncertainty["sink_base"],
        "mesh_uncertainty_fins": uncertainty["fins"],
        "mesh_profile": radiation["mesh_profile"],
        "mesh_qualified": False,
        "benchmark_qualified": benchmark_qualified,
        "temporal_qualified": temporal_qualified,
        "qualification_basis": (
            str(mesh["basis"])
            if benchmark_qualified
            else "actual mesh/boundary is not benchmark-qualified: " + str(mesh["basis"])
        ),
        "mesh_evidence_sha256": json.dumps(mesh.get("evidence_sha256", {}), sort_keys=True),
        "transient_result_fingerprint": result_fingerprint(radiation),
        "temporal_qualification_basis": temporal_basis,
    }
    if temporal_matches and temporal is not None:
        sensor_uncertainty = temporal["sensor_uncertainty"]
        if not isinstance(sensor_uncertainty, dict):
            raise TypeError("temporal sensor uncertainty must be a mapping")
        for sensor in SENSORS:
            columns[f"temporal_uncertainty_{sensor}"] = float(sensor_uncertainty[sensor])
    return pd.DataFrame(columns)


def load_verified_dynamic_reference(
    root: Path,
    radiation: pd.DataFrame | None = None,
    *,
    case_id: str = "HV02_composite_radiation",
) -> pd.DataFrame:
    """Validate a published reference once at the consuming workflow boundary.

    Publication flags alone cannot establish qualification. Rebuild the expected
    view from the actual transient and current mesh/time evidence, then verify
    the stored identity, flags, and uncertainties before exposing that view.
    Missing legacy provenance requires a fresh summary, not a new COMSOL solve.
    """
    if radiation is None:
        radiation = pd.read_csv(root / "dynamic/eval/model_gap/HV02_composite_radiation.csv")
    required = {"case_id", *RESULT_COLUMNS, *PHYSICS_COLUMNS}
    missing = required - set(radiation)
    if missing:
        raise ValueError(f"{case_id}: missing reference source columns {sorted(missing)}")
    if (
        radiation.empty
        or not radiation["case_id"].eq(case_id).all()
        or radiation["mesh_profile"].isna().any()
        or radiation["mesh_profile"].nunique(dropna=False) != 1
        or not np.isfinite(radiation[list(RESULT_COLUMNS)].to_numpy(dtype=np.float64)).all()
        or not np.all(np.diff(radiation["time"].to_numpy(dtype=np.float64)) > 0.0)
    ):
        raise ValueError(f"{case_id}: invalid reference source case, mesh, or physical results")
    mesh = mesh_evidence(root, str(radiation["mesh_profile"].iloc[0]))
    expected = build_dynamic_reference(radiation, mesh, temporal_evidence(root))
    path = root / "dynamic" / "cae_reference.csv"
    reference = pd.read_csv(path)
    missing = set(expected) - set(reference)
    extra_temporal = {
        name for name in reference if str(name).startswith("temporal_uncertainty_")
    } - set(expected)
    if missing or extra_temporal or len(reference) != len(expected):
        raise ValueError(
            f"{path}: reference schema/rows do not match the verified source; "
            "regenerate the high-fidelity summary"
        )
    flags = {"mesh_qualified", "benchmark_qualified", "temporal_qualified"}
    for name in expected:
        actual = reference[name]
        wanted = expected[name]
        if name in flags:
            values = actual.astype(str).str.lower()
            matches = values.isin(["true", "false"]).all() and true_values(actual).equals(
                true_values(wanted)
            )
        elif pd.api.types.is_numeric_dtype(wanted):
            matches = np.allclose(
                actual.to_numpy(dtype=np.float64),
                wanted.to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=1e-9,
                equal_nan=True,
            )
        else:
            matches = actual.astype(str).equals(wanted.astype(str))
        if not matches:
            raise ValueError(f"{path}: reference {name} does not match the source/evidence")
    return reference
