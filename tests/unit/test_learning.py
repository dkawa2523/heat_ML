import numpy as np
import pytest
import torch

from celltemp.domain import (
    ActuatorSpec,
    ConstantLawSpec,
    EdgeSpec,
    PositivePartLawSpec,
    SourceSpec,
    ThermalSystemSpec,
    Trajectory,
)
from celltemp.engine import ThermalRCModel
from celltemp.learning import (
    TrainingConfig,
    fit_thermal_model,
    initial_observation_rmse,
    trajectory_loss,
    trajectory_rmse,
)
from celltemp.learning.objective import profiled_initial_temperature, trajectory_tensors
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


def test_trajectory_objective_profiles_hidden_initial_temperature() -> None:
    spec = ThermalSystemSpec(
        node_names=("surface", "core"),
        heat_capacity=(1.0, 2.0),
        edges=(EdgeSpec("surface", "core", ConstantLawSpec(0.25, learnable=False)),),
        actuators=(),
        sensor_names=("surface_tc",),
        sensor_nodes=("surface",),
    )
    model = ThermalRCModel(spec)
    steps = 20
    states, _ = model.forward_trajectory(
        torch.tensor([80.0, 20.0], dtype=torch.float64),
        torch.empty((steps, 0), dtype=torch.float64),
        torch.ones(steps, dtype=torch.float64),
    )
    trajectory = Trajectory(
        case_id="hidden-initial-state",
        time=np.arange(steps + 1, dtype=float),
        temperature=model.observe(states).detach().numpy(),
        commands=np.empty((steps, 0)),
        sensor_names=("surface_tc",),
        control_names=(),
    )

    assert trajectory_rmse(model, trajectory) < 3e-3
    assert initial_observation_rmse(model, trajectory) > 1.0


def test_hidden_initial_profile_uses_weighted_observation_null_space() -> None:
    from celltemp.domain import BoundarySpec, ReservoirTemperatureSpec

    spec = ThermalSystemSpec(
        node_names=("core", "surface"),
        heat_capacity=(3.0, 1.0),
        edges=(EdgeSpec("core", "surface", ConstantLawSpec(0.2, learnable=False)),),
        actuators=(),
        boundaries=(
            BoundarySpec(
                "ambient",
                (0.0, 1.0),
                ReservoirTemperatureSpec(20.0),
                ConstantLawSpec(0.1, learnable=False),
            ),
        ),
        sensor_names=("weighted_tc",),
        sensor_weights=((0.25, 0.75),),
    )
    model = ThermalRCModel(spec)
    steps = 30
    states, _ = model.forward_trajectory(
        torch.tensor([80.0, 20.0], dtype=torch.float64),
        torch.empty((steps, 0), dtype=torch.float64),
        torch.ones(steps, dtype=torch.float64),
    )
    trajectory = Trajectory(
        case_id="weighted-observation",
        time=np.arange(steps + 1, dtype=float),
        temperature=model.observe(states).numpy(),
        commands=np.empty((steps, 0)),
        sensor_names=spec.sensor_names,
        control_names=(),
    )

    assert trajectory_rmse(model, trajectory) < 2e-3
    assert trajectory_rmse(model, trajectory) < initial_observation_rmse(model, trajectory) / 100.0


def test_hidden_initial_profile_has_a_physical_scale_when_observability_is_weak() -> None:
    spec = ThermalSystemSpec(
        node_names=("measured", "hidden"),
        heat_capacity=(1.0, 1.0),
        edges=(EdgeSpec("measured", "hidden", ConstantLawSpec(1e-4, learnable=False)),),
        actuators=(),
        sensor_names=("sensor",),
        sensor_nodes=("measured",),
    )
    model = ThermalRCModel(spec)
    temperature = np.full((6, 1), 20.0)
    temperature[-1, 0] = 21.0
    trajectory = Trajectory(
        case_id="weakly-observable",
        time=np.arange(6, dtype=float),
        temperature=temperature,
        commands=np.empty((5, 0)),
        sensor_names=("sensor",),
        control_names=(),
    )
    observed, commands, dt, mask = trajectory_tensors(model, trajectory)

    initial = profiled_initial_temperature(
        model,
        observed,
        commands,
        dt,
        mask,
        start=0,
        stop=5,
    )

    assert 20.0 <= initial[1] <= 25.0


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
        {"initial_temperature_prior_std": 0.0},
        {"prior_weight": -1.0},
        {"gradient_clip": -1.0},
        {"prior_weight": float("nan")},
        {"gradient_clip": float("inf")},
    ],
)
def test_training_config_rejects_invalid_numeric_values(options: dict) -> None:
    with pytest.raises(ValueError, match=r"positive|non-negative"):
        TrainingConfig(**options)


def test_fit_rejects_nonfinite_loss() -> None:
    model = ThermalRCModel(_source_system(0.2))
    model.source_laws.log_scale_multiplier.data.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        fit_thermal_model(
            model,
            [_synthetic_trajectory()],
            config=TrainingConfig(epochs=1),
        )


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
