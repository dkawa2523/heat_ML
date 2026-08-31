import numpy as np
import pytest
import torch

from celltemp.domain import (
    ActuatorSpec,
    PositivePartLawSpec,
    SourceSpec,
    ThermalSystemSpec,
    Trajectory,
)
from celltemp.engine import ThermalRCModel
from celltemp.learning import TrainingConfig, fit_thermal_model, trajectory_loss, trajectory_rmse
from celltemp.learning.trainer import _valid_starts


def _source_system(gain: float) -> ThermalSystemSpec:
    return ThermalSystemSpec(
        node_names=("body",),
        heat_capacity=(2.0,),
        edges=(),
        actuators=(ActuatorSpec("power", tau=0.0, learnable=False),),
        sources=(SourceSpec("heater", (1.0,), PositivePartLawSpec("power", gain)),),
    )


def _synthetic_trajectory() -> Trajectory:
    truth = ThermalRCModel(_source_system(0.2))
    commands = torch.tensor([[0.0], [2.0], [4.0], [1.0]] * 5, dtype=torch.float64)
    dt = torch.ones(len(commands), dtype=torch.float64)
    states, _ = truth.forward_trajectory(torch.tensor([20.0]), commands, dt)
    return Trajectory(
        case_id="synthetic",
        time=np.arange(len(commands) + 1, dtype=float),
        temperature=states.detach().numpy(),
        commands=commands.numpy(),
        sensor_names=("body",),
        control_names=("power",),
    )


def test_trajectory_objective_is_zero_for_generating_model() -> None:
    trajectory = _synthetic_trajectory()
    model = ThermalRCModel(_source_system(0.2))
    assert float(trajectory_loss(model, trajectory).detach()) < 1e-12
    assert trajectory_rmse(model, trajectory) < 1e-12


def test_fit_reduces_full_rollout_error() -> None:
    trajectory = _synthetic_trajectory()
    model = ThermalRCModel(_source_system(0.08))
    extension_parameter = torch.nn.Parameter(torch.tensor(10.0, dtype=torch.float64))
    model.register_parameter("extension_parameter", extension_parameter)
    before = trajectory_rmse(model, trajectory)
    result = fit_thermal_model(
        model,
        [trajectory],
        [trajectory],
        config=TrainingConfig(
            epochs=30,
            batch_size=2,
            horizon=None,
            learning_rate=0.08,
            validation_every=2,
            patience=20,
            seed=1,
        ),
    )
    after = trajectory_rmse(model, trajectory)
    assert result.best_epoch > 0
    assert after < before * 0.1
    assert extension_parameter.item() == 10.0


def test_full_trajectory_training_batches_different_lengths() -> None:
    trajectory = _synthetic_trajectory()
    shorter = Trajectory(
        case_id="shorter",
        time=trajectory.time[:9],
        temperature=trajectory.temperature[:9],
        commands=trajectory.commands[:8],
        sensor_names=trajectory.sensor_names,
        control_names=trajectory.control_names,
    )
    model = ThermalRCModel(_source_system(0.12))
    before = np.mean([trajectory_rmse(model, item) for item in (trajectory, shorter)])
    fit_thermal_model(
        model,
        [trajectory, shorter],
        config=TrainingConfig(
            epochs=20,
            batch_size=2,
            horizon=None,
            learning_rate=0.08,
            validation_every=2,
            patience=20,
            seed=2,
        ),
    )
    after = np.mean([trajectory_rmse(model, item) for item in (trajectory, shorter)])
    assert after < before * 0.2


def test_shooting_starts_preserve_the_requested_horizon() -> None:
    trajectory = _synthetic_trajectory()
    assert np.array_equal(_valid_starts(trajectory, 12), np.arange(9))
    assert np.array_equal(_valid_starts(trajectory, 100), np.array([0]))

    sparse_temperature = np.full_like(trajectory.temperature, np.nan)
    sparse_temperature[[0, 3, 6]] = trajectory.temperature[[0, 3, 6]]
    sparse = Trajectory(
        case_id="sparse",
        time=trajectory.time,
        temperature=sparse_temperature,
        commands=trajectory.commands,
        sensor_names=trajectory.sensor_names,
        control_names=trajectory.control_names,
    )
    assert np.array_equal(_valid_starts(sparse, 4), np.array([0, 3]))


@pytest.mark.parametrize(
    "options",
    [
        {"epochs": 0},
        {"horizon": 0},
        {"learning_rate": 0.0},
        {"prior_weight": -1.0},
        {"gradient_clip": -1.0},
    ],
)
def test_training_config_rejects_nonpositive_values(options: dict) -> None:
    with pytest.raises(ValueError, match=r"positive|non-negative"):
        TrainingConfig(**options)


def test_learning_rejects_missing_cases_and_wrong_control_names() -> None:
    model = ThermalRCModel(_source_system(0.2))
    with pytest.raises(ValueError, match="at least one"):
        fit_thermal_model(model, [])

    trajectory = _synthetic_trajectory()
    wrong = Trajectory(
        case_id="wrong",
        time=trajectory.time,
        temperature=trajectory.temperature,
        commands=trajectory.commands,
        sensor_names=trajectory.sensor_names,
        control_names=("wrong",),
    )
    with pytest.raises(ValueError, match="controls"):
        trajectory_loss(model, wrong)

    missing_initial = trajectory.temperature.copy()
    missing_initial[0] = np.nan
    delayed = Trajectory(
        case_id="missing-initial",
        time=trajectory.time,
        temperature=missing_initial,
        commands=trajectory.commands,
        sensor_names=trajectory.sensor_names,
        control_names=trajectory.control_names,
    )
    with pytest.raises(ValueError, match="initial observation"):
        fit_thermal_model(model, [delayed])

    only_initial = trajectory.temperature.copy()
    only_initial[1:] = np.nan
    no_targets = Trajectory(
        case_id="no-targets",
        time=trajectory.time,
        temperature=only_initial,
        commands=trajectory.commands,
        sensor_names=trajectory.sensor_names,
        control_names=trajectory.control_names,
    )
    with pytest.raises(ValueError, match="observation after the initial row"):
        fit_thermal_model(model, [no_targets])


def test_trajectory_loss_rejects_empty_interval_selection() -> None:
    model = ThermalRCModel(_source_system(0.2))
    trajectory = _synthetic_trajectory()
    with pytest.raises(ValueError, match="start/stop"):
        trajectory_loss(model, trajectory, start=2, stop=2)
