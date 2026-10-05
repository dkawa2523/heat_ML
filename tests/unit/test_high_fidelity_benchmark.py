from pathlib import Path

import pandas as pd
import pytest

from external_tools.comsol_chip_cooling import benchmark_high_fidelity

TOOL_ROOT = Path(__file__).parents[2] / "external_tools/comsol_chip_cooling"


def test_benchmark_consumer_keeps_public_screening_qualification() -> None:
    frames = benchmark_high_fidelity._load_cases(TOOL_ROOT)
    quality = benchmark_high_fidelity._reference_quality(TOOL_ROOT, frames)
    assert quality["dynamic_reference_matches_hv02"] is True
    assert quality["mesh_profile"] == "local-medium"
    assert quality["benchmark_qualified"] is True
    assert quality["temporal_qualified"] is False


@pytest.mark.parametrize(
    ("column", "value"),
    [("mesh_profile", "local-coarse"), ("truth_hidden_power", 1.0)],
)
def test_benchmark_rejects_source_identity_changes_before_starting_workflows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, column: str, value: str | float
) -> None:
    frames = {
        case_id: frame.assign(**{column: value})
        for case_id, frame in benchmark_high_fidelity._load_cases(TOOL_ROOT).items()
    }
    monkeypatch.setattr(benchmark_high_fidelity, "_load_cases", lambda _: frames)

    def forbidden_workflow_start(*args: object) -> None:
        raise AssertionError("invalid CAE reference reached training/forecasting")

    monkeypatch.setattr(benchmark_high_fidelity, "_run_workflows", forbidden_workflow_start)
    with pytest.raises(ValueError, match="source/evidence"):
        benchmark_high_fidelity.run_benchmark(TOOL_ROOT / "high_fidelity_benchmark.yaml", tmp_path)
    assert not list(tmp_path.iterdir())


def test_benchmark_radiation_pair_rejects_a_different_numerical_configuration() -> None:
    frames: dict[str, pd.DataFrame] = benchmark_high_fidelity._load_cases(TOOL_ROOT)
    frames["HV02_composite_radiation"] = frames["HV02_composite_radiation"].assign(
        mesh_profile="local-coarse"
    )
    assert not benchmark_high_fidelity._validate_pair(frames)
    with pytest.raises(ValueError, match="settings: mesh_profile"):
        benchmark_high_fidelity._reference_quality(TOOL_ROOT, frames)
