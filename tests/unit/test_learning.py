from typing import Any, cast

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
    predict_trajectory,
    trajectory_loss,
    trajectory_rmse,
)
from celltemp.learning.objective import (
    actuator_before_interval,
    masked_huber_loss,
    profiled_initial_temperature,
    trajectory_initial_actuator,
    trajectory_tensors,
)
from celltemp.learning.trainer import _rollout_case_losses, _valid_starts


def _source_system(gain: float) -> ThermalSystemSpec:
    return ThermalSystemSpec(
        node_names=("body",),
        heat_capacity=(2.0,),
        edges=(),
        actuators=(ActuatorSpec("power", tau=0.0, learnable=False),),
        sources=(SourceSpec("heater", (1.0,), PositivePartLawSpec("power", gain)),),
    )


@pytest.mark.parametrize(
    "option", ["epochs", "batch_size", "validation_every", "patience", "horizon"]
)
@pytest.mark.parametrize("value", [True, 2.5, "2", 0, -1])
def test_training_counts_require_positive_integers(option: str, value: object) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        TrainingConfig(**{option: cast(Any, value)})


@pytest.mark.parametrize("value", [True, 1.5, "42", None, -1])
def test_training_seed_requires_a_non_negative_integer(value: object) -> None:
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        TrainingConfig(seed=cast(Any, value))


@pytest.mark.parametrize(
    "option",
    [
        "learning_rate",
        "huber_delta",
        "initial_temperature_prior_std",
        "prior_weight",
        "gradient_clip",
    ],
)
@pytest.mark.parametrize("value", [True, "0.5", None, float("inf"), float("nan")])
def test_training_real_values_reject_booleans_wrong_types_and_non_finite(
    option: str, value: object
) -> None:
    with pytest.raises(ValueError, match="finite"):
        TrainingConfig(**{option: cast(Any, value)})


def test_training_defaults_and_boundary_values_remain_valid() -> None:
    assert TrainingConfig().epochs == 80
    assert TrainingConfig(seed=0, horizon=1, prior_weight=0, gradient_clip=0).seed == 0


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


def test_fit_rejects_nonfinite_model_before_integration() -> None:
    model = ThermalRCModel(_source_system(0.2))
    model.source_laws.log_scale_multiplier.data.fill_(float("nan"))
    with pytest.raises(ValueError, match="must be finite"):
        fit_thermal_model(model, [_synthetic_trajectory()], config=TrainingConfig(epochs=1))


def test_fit_rejects_nonfinite_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ThermalRCModel(_source_system(0.2))
    monkeypatch.setattr(
        "celltemp.learning.trainer._rollout_case_losses",
        lambda *args, **kwargs: torch.tensor([float("nan")], dtype=model.capacity.dtype),
    )
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


def _hidden_actuator_case(integrator: str) -> tuple[ThermalRCModel, Trajectory]:
    spec = ThermalSystemSpec(
        node_names=("surface", "core"),
        heat_capacity=(1.0, 2.0),
        edges=(EdgeSpec("surface", "core", ConstantLawSpec(0.2)),),
        actuators=(ActuatorSpec("power", tau=1.7),),
        sources=(SourceSpec("heater", (1.0, 0.0), PositivePartLawSpec("power", 0.13, 0.5)),),
        sensor_names=("weighted_tc",),
        sensor_weights=((0.25, 0.75),),
    )
    model = ThermalRCModel(spec, integrator=integrator)
    commands = torch.tensor([[2.0], [0.2], [3.0], [0.0], [1.4], [0.0]], dtype=torch.float64)
    dt = torch.tensor([0.3, 0.7, 0.2, 0.8, 1.1, 0.4], dtype=torch.float64)
    initial_actuator = torch.tensor([0.6], dtype=torch.float64)
    states, _ = model.forward_trajectory(
        torch.tensor([70.0, 25.0], dtype=torch.float64), commands, dt, initial_actuator
    )
    temperature = model.observe(states).detach().numpy()
    temperature[:, 0] += np.array([0.0, 0.1, -0.2, 0.05, -0.1, 0.1, 0.05])
    temperature[4] = np.nan
    trajectory = Trajectory(
        case_id="hidden-actuator",
        time=np.r_[0.0, np.cumsum(dt.numpy())],
        temperature=temperature,
        commands=commands.numpy(),
        sensor_names=spec.sensor_names,
        control_names=spec.control_names,
        initial_actuator=initial_actuator.numpy(),
    )
    model.edge_laws.log_offset_multiplier.data.fill_(0.15)
    model.source_laws.log_scale_multiplier.data.fill_(-0.1)
    model.log_tau_multiplier.data.fill_(0.08)
    return model, trajectory


@pytest.mark.parametrize("integrator", ["exact", "implicit"])
@pytest.mark.parametrize("start", [0, 2])
def test_profiled_rollout_reuse_preserves_prediction_and_gradients(
    integrator: str, start: int
) -> None:
    model, trajectory = _hidden_actuator_case(integrator)
    observed, commands, dt, mask = trajectory_tensors(model, trajectory)
    starting_actuator = trajectory_initial_actuator(model, trajectory, commands)
    stop = len(dt)
    initial = profiled_initial_temperature(
        model,
        observed,
        commands,
        dt,
        mask,
        start=start,
        stop=stop,
        initial_actuator=starting_actuator,
    )
    shooting_actuator = actuator_before_interval(model, commands, dt, start, starting_actuator)
    repeated_states, _ = model.forward_trajectory(
        initial, commands[start:stop], dt[start:stop], shooting_actuator
    )
    expected = model.observe(repeated_states)
    actual = predict_trajectory(model, trajectory, start=start, stop=stop)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    target, target_mask = observed[start + 1 : stop + 1], mask[start + 1 : stop + 1]
    parameters = tuple(model.parameters())
    actual_gradients = torch.autograd.grad(
        masked_huber_loss(actual[1:], target, target_mask, delta=1.0),
        parameters,
        allow_unused=True,
    )
    expected_gradients = torch.autograd.grad(
        masked_huber_loss(expected[1:], target, target_mask, delta=1.0),
        parameters,
        allow_unused=True,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        if expected_gradient is None:
            assert actual_gradient is None
        else:
            assert actual_gradient is not None
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-8, atol=1e-10)


def test_hidden_profile_reuses_rollout_and_replays_actuator_prefix_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, trajectory = _hidden_actuator_case("exact")
    rollout_calls: list[int] = []
    actuator_calls: list[int] = []
    original_rollout = model.forward_batch
    original_actuator = model.actuator_step

    def counted_rollout(*args, **kwargs):
        rollout_calls.append(1)
        return original_rollout(*args, **kwargs)

    def counted_actuator(*args, **kwargs):
        actuator_calls.append(1)
        return original_actuator(*args, **kwargs)

    monkeypatch.setattr(model, "forward_batch", counted_rollout)
    monkeypatch.setattr(model, "actuator_step", counted_actuator)
    losses = _rollout_case_losses(
        model, [(trajectory, 2, 6)], huber_delta=1.0, initial_temperature_prior_std=50.0
    )

    assert bool(torch.isfinite(losses).all())
    assert len(rollout_calls) == 1
    assert len(actuator_calls) == 2


def test_mixed_observed_and_hidden_cases_preserve_case_losses() -> None:
    spec = ThermalSystemSpec(
        node_names=("surface", "core"),
        heat_capacity=(1.0, 2.0),
        edges=(EdgeSpec("surface", "core", ConstantLawSpec(0.2)),),
        actuators=(),
    )
    model = ThermalRCModel(spec)
    commands = np.empty((3, 0))
    states, _ = model.forward_trajectory(
        torch.tensor([70.0, 25.0], dtype=torch.float64),
        torch.tensor(commands),
        torch.ones(3, dtype=torch.float64),
    )
    observed = states.detach().numpy()
    cases = []
    for index, missing_core in enumerate((True, False, True)):
        temperature = observed.copy()
        temperature[1:, 0] += 0.1 * (index + 1)
        if missing_core:
            temperature[0, 1] = np.nan
        cases.append(
            Trajectory(
                case_id=f"case-{index}",
                time=np.arange(4.0),
                temperature=temperature,
                commands=commands,
                sensor_names=spec.sensor_names,
                control_names=(),
            )
        )
    actual = _rollout_case_losses(
        model,
        [(case, 0, 3) for case in cases],
        huber_delta=1.0,
        initial_temperature_prior_std=50.0,
    )
    expected = torch.stack([trajectory_loss(model, case) for case in cases])
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("metric", [trajectory_loss, trajectory_rmse, initial_observation_rmse])
def test_objective_prepares_trajectory_tensors_once(
    monkeypatch: pytest.MonkeyPatch,
    metric,
) -> None:
    from celltemp.learning import objective

    model = ThermalRCModel(_source_system(0.2))
    trajectory = _synthetic_trajectory()
    prepared_cases: list[str] = []
    original = objective.trajectory_tensors

    def counted_prepare(model, trajectory):
        prepared_cases.append(trajectory.case_id)
        return original(model, trajectory)

    monkeypatch.setattr(objective, "trajectory_tensors", counted_prepare)
    metric(model, trajectory)
    assert prepared_cases == [trajectory.case_id]


def test_fit_reuses_prepared_inputs_for_training_and_causal_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from celltemp.learning import objective

    model = ThermalRCModel(_source_system(0.12))
    training_case = _synthetic_trajectory()
    validation_case = Trajectory(
        case_id="validation",
        time=training_case.time,
        temperature=training_case.temperature,
        commands=training_case.commands,
        sensor_names=training_case.sensor_names,
        control_names=training_case.control_names,
    )
    prepared_cases: list[str] = []
    original = objective.trajectory_tensors

    def counted_prepare(model, trajectory):
        prepared_cases.append(trajectory.case_id)
        return original(model, trajectory)

    monkeypatch.setattr(objective, "trajectory_tensors", counted_prepare)
    result = fit_thermal_model(
        model,
        [training_case],
        [training_case, validation_case],
        config=TrainingConfig(epochs=3, validation_every=1, patience=3),
    )
    assert result.best_epoch > 0
    assert prepared_cases == [training_case.case_id, validation_case.case_id]
    assert result.best_causal_validation_rmse == pytest.approx(
        np.mean(
            [initial_observation_rmse(model, case) for case in (training_case, validation_case)]
        )
    )
