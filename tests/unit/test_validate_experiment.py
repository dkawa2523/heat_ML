import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from external_tools.comsol_chip_cooling import validate_experiment


def _comparison_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    common = {
        "case_id": ["case_a", "case_a"],
        "time": [0.0, 1.0],
        "chip_power": [8.0, 8.0],
        "coolant_temperature": [20.0, 20.0],
        "inlet_air_velocity": [1.5, 1.5],
    }
    cae = pd.DataFrame(
        {
            **common,
            "chip": [40.0, 41.0],
            "sink_base": [35.0, 36.0],
            "fins": [30.0, 31.0],
            "mesh_uncertainty_chip": [4.0, 4.0],
            "mesh_uncertainty_sink_base": [4.0, 4.0],
            "mesh_uncertainty_fins": [4.0, 4.0],
            "benchmark_qualified": [True, True],
        }
    )
    experiment = pd.DataFrame(
        {
            **common,
            "chip": [43.0, 44.0],
            "sink_base": [38.0, 39.0],
            "fins": [33.0, 34.0],
            "uncertainty_chip": [3.0, 3.0],
            "uncertainty_sink_base": [3.0, 3.0],
            "uncertainty_fins": [3.0, 3.0],
        }
    )
    return cae, experiment


def test_compare_combines_standard_uncertainties_on_the_exact_boundary() -> None:
    cae, experiment = _comparison_frames()

    residuals, metrics = validate_experiment.compare(cae, experiment)

    assert len(residuals) == len(cae) * len(validate_experiment.SENSORS)
    np.testing.assert_allclose(residuals["combined_standard_uncertainty_c"], 5.0)
    np.testing.assert_allclose(residuals["normalized_residual"], 0.6)
    np.testing.assert_allclose(metrics["bias_c"], 3.0)
    np.testing.assert_allclose(metrics["normalized_rmse"], 0.6)
    np.testing.assert_allclose(metrics["fraction_within_95_percent_uncertainty"], 1.0)


def test_compare_keeps_mesh_temporal_and_measurement_uncertainty_separate() -> None:
    cae, experiment = _comparison_frames()
    for sensor in validate_experiment.SENSORS:
        cae[f"temporal_uncertainty_{sensor}"] = 1.0

    residuals, metrics = validate_experiment.compare(cae, experiment)

    np.testing.assert_allclose(residuals["experiment_standard_uncertainty_c"], 3.0)
    np.testing.assert_allclose(residuals["mesh_standard_uncertainty_c"], 4.0)
    np.testing.assert_allclose(residuals["temporal_standard_uncertainty_c"], 1.0)
    np.testing.assert_allclose(residuals["combined_standard_uncertainty_c"], np.sqrt(26.0))
    np.testing.assert_allclose(metrics["normalized_rmse"], 3.0 / np.sqrt(26.0))


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("time", 2.0, "same boundary"),
        ("chip_power", 8.5, "control chip_power differs"),
    ],
)
def test_compare_rejects_misaligned_keys_or_controls(
    column: str,
    replacement: float,
    message: str,
) -> None:
    cae, experiment = _comparison_frames()
    experiment.loc[1, column] = replacement

    with pytest.raises(ValueError, match=message):
        validate_experiment.compare(cae, experiment)


def test_empty_experiment_template_is_not_measurement_data() -> None:
    template = (
        Path(__file__).parents[2]
        / "external_tools/comsol_chip_cooling/data/nonlinear_high_fidelity/experiment"
        / "experiment_template.csv"
    )

    with pytest.raises(ValueError, match="non-finite experiment values"):
        validate_experiment._read(template, kind="experiment")


def test_run_validation_preserves_the_complete_result_on_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cae, experiment = _comparison_frames()
    cae_path = tmp_path / "cae.csv"
    experiment_path = tmp_path / "experiment.csv"
    output = tmp_path / "result"
    cae.to_csv(cae_path, index=False)
    experiment.to_csv(experiment_path, index=False)

    validate_experiment.run_validation(cae_path, experiment_path, output)
    previous_status = (output / "validation_status.json").read_bytes()
    status = json.loads(previous_status)
    assert status["acceptance_passed"] is None
    assert {path.name for path in output.iterdir()} == {
        "aligned_residuals.csv",
        "validation_metrics.csv",
        "validation_status.json",
    }

    def fail_after_partial_write(
        target: Path,
        residuals: pd.DataFrame,
        metrics: pd.DataFrame,
        status: dict[str, object],
    ) -> None:
        del residuals, metrics, status
        (target / "partial.csv").write_text("incomplete", encoding="utf-8")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(validate_experiment, "_write_outputs", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        validate_experiment.run_validation(cae_path, experiment_path, output)

    assert (output / "validation_status.json").read_bytes() == previous_status
    assert not (output / "partial.csv").exists()
    assert not list(tmp_path.glob(".result-*"))
