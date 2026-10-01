from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from benchmarks.neural_comparison.internal import initial_condition_request
from benchmarks.neural_comparison.models import MODEL_TYPES, SequenceModelSpec, build_model
from benchmarks.neural_comparison.reporting import (
    INTERNAL_DATASETS,
    MODEL_ORDER,
    configure_japanese_plotting,
    plot_internal_split_overview,
    plot_internal_vs_external,
)
from benchmarks.neural_comparison.robustness import (
    add_observation_noise,
    summarize_observation_noise,
)
from benchmarks.neural_comparison.training import SequencePreprocessor, forecast_sequence
from celltemp.domain import Trajectory


@pytest.mark.parametrize("model_name", tuple(MODEL_TYPES))
def test_neural_comparison_models_return_one_rate_per_sensor(model_name: str) -> None:
    spec = SequenceModelSpec(n_sensors=3, n_controls=2, history_steps=8)
    model = build_model(model_name, spec)

    predicted = model(
        torch.zeros((4, spec.history_steps, spec.history_features)),
        torch.zeros((4, spec.query_features)),
    )

    assert predicted.shape == (4, spec.n_sensors)
    assert torch.isfinite(predicted).all()


def _trajectory(
    case_id: str,
    temperatures: np.ndarray,
    *,
    observation_mask: np.ndarray | None = None,
) -> Trajectory:
    steps = len(temperatures) - 1
    return Trajectory(
        case_id=case_id,
        time=np.arange(steps + 1, dtype=float),
        temperature=np.asarray(temperatures, dtype=float),
        commands=np.linspace(0.0, 1.0, steps)[:, None],
        sensor_names=("sensor",),
        control_names=("power",),
        observation_mask=observation_mask,
    )


def test_open_loop_neural_forecast_does_not_read_future_temperature_truth() -> None:
    training = _trajectory("training", np.linspace(20.0, 25.0, 9)[:, None])
    preprocessor = SequencePreprocessor.fit([training])
    spec = SequenceModelSpec(n_sensors=1, n_controls=1, history_steps=4)
    model = build_model("mlp", spec)
    mask = np.array([[True], [True], [False], [False], [False]])
    request_a = _trajectory(
        "request-a",
        np.array([[20.0], [21.0], [22.0], [23.0], [24.0]]),
        observation_mask=mask,
    )
    request_b = _trajectory(
        "request-b",
        np.array([[20.0], [21.0], [200.0], [-50.0], [999.0]]),
        observation_mask=mask,
    )

    forecast_a = forecast_sequence(
        model,
        request_a,
        preprocessor,
        history_steps=spec.history_steps,
        origin=1,
    )
    forecast_b = forecast_sequence(
        model,
        request_b,
        preprocessor,
        history_steps=spec.history_steps,
        origin=1,
    )

    np.testing.assert_allclose(forecast_a, forecast_b)


def test_internal_request_keeps_only_initial_temperature_observation() -> None:
    trajectory = _trajectory("internal", np.array([[20.0], [21.0], [22.0], [23.0]]))

    request = initial_condition_request(trajectory)

    np.testing.assert_allclose(request.temperature[0], trajectory.temperature[0])
    assert request.mask[0].all()
    assert not request.mask[1:].any()
    assert np.isnan(request.temperature[1:]).all()
    np.testing.assert_allclose(request.commands, trajectory.commands)


def test_observation_noise_changes_only_causally_observed_temperatures() -> None:
    mask = np.array([[True], [True], [False], [False]])
    trajectory = _trajectory(
        "noise",
        np.array([[20.0], [21.0], [np.nan], [np.nan]]),
        observation_mask=mask,
    )

    noisy, realized_rmse = add_observation_noise(
        trajectory,
        noise_std_k=0.15,
        rng=np.random.default_rng(7),
    )

    assert np.isfinite(noisy.temperature[mask]).all()
    assert np.isnan(noisy.temperature[~mask]).all()
    np.testing.assert_allclose(noisy.commands, trajectory.commands)
    assert realized_rmse > 0.0


def test_noise_summary_keeps_clean_baseline_and_boundary_separate() -> None:
    rows = []
    for boundary, clean, noisy in (
        ("internal_test", 0.1, 0.2),
        ("external_core", 0.3, 0.6),
    ):
        rows.append(
            {
                "evaluation": "example",
                "boundary": boundary,
                "category": "core",
                "model": "physical_rc",
                "noise_std_k": 0.0,
                "repeat": 0,
                "case_id": "clean",
                "rmse_k": clean,
                "realized_noise_rmse_k": 0.0,
            }
        )
        for repeat in range(2):
            rows.append(
                {
                    "evaluation": "example",
                    "boundary": boundary,
                    "category": "core",
                    "model": "physical_rc",
                    "noise_std_k": 0.15,
                    "repeat": repeat,
                    "case_id": "noisy",
                    "rmse_k": noisy,
                    "realized_noise_rmse_k": 0.15,
                }
            )

    summary = summarize_observation_noise(pd.DataFrame(rows))

    noisy_rows = summary[summary["noise_std_k"] == 0.15].set_index("boundary")
    assert noisy_rows.loc["internal_test", "ratio_to_clean"] == pytest.approx(2.0)
    assert noisy_rows.loc["external_core", "ratio_to_clean"] == pytest.approx(2.0)


def test_internal_summary_figures_render_all_data_boundaries(tmp_path: Path) -> None:
    configure_japanese_plotting()
    internal = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "split": split,
                "model": model,
                "mean_case_rmse_k": float(index + 1),
            }
            for dataset in INTERNAL_DATASETS
            for split in ("train", "validation", "test")
            for index, model in enumerate(MODEL_ORDER)
        ]
    )
    external = pd.DataFrame(
        [
            {
                "evaluation": dataset,
                "category": "core",
                "model": model,
                "mean_case_rmse_k": float(index + 2),
            }
            for dataset in INTERNAL_DATASETS
            for index, model in enumerate(MODEL_ORDER)
        ]
    )

    plot_internal_split_overview(internal, tmp_path / "splits.png")
    plot_internal_vs_external(internal, external, tmp_path / "boundaries.png")

    assert (tmp_path / "splits.png").stat().st_size > 0
    assert (tmp_path / "boundaries.png").stat().st_size > 0
