"""Physical and numerical invariants of the shared thermal engine."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from celltemp.domain import (
    ActuatorSpec,
    BoundarySpec,
    ConstantLawSpec,
    EdgeSpec,
    PositivePartLawSpec,
    PowerLawSpec,
    ReservoirTemperatureSpec,
    SourceSpec,
    ThermalSystemSpec,
)
from celltemp.engine import KalmanObserver, ThermalRCModel, ThermalState
from celltemp.engine.integrator import exact_affine_step, implicit_euler_step

DTYPE = torch.float64


def conduction_model() -> ThermalRCModel:
    spec = ThermalSystemSpec(
        node_names=("a", "b"),
        heat_capacity=(2.0, 1.0),
        edges=(EdgeSpec("a", "b", ConstantLawSpec(0.5)),),
        actuators=(),
    )
    return ThermalRCModel(spec)


def test_internal_conduction_conserves_capacity_weighted_energy() -> None:
    model = conduction_model()
    initial = torch.tensor([100.0, 20.0], dtype=DTYPE)
    state = ThermalState(initial, torch.empty(0, dtype=DTYPE))
    after = model.step(state, torch.empty(0, dtype=DTYPE), 17.0)
    torch.testing.assert_close(
        model.stored_energy(after.temperature),
        model.stored_energy(initial),
        rtol=1e-12,
        atol=1e-12,
    )


def test_passive_conduction_stays_inside_initial_temperature_bounds() -> None:
    model = conduction_model()
    state = ThermalState(torch.tensor([100.0, 20.0], dtype=DTYPE), torch.empty(0, dtype=DTYPE))
    after = model.step(state, torch.empty(0, dtype=DTYPE), 100.0)
    assert torch.all(after.temperature >= 20.0)
    assert torch.all(after.temperature <= 100.0)


def test_exact_integrator_is_consistent_across_timestep_subdivision() -> None:
    model = conduction_model()
    initial = ThermalState(torch.tensor([80.0, 20.0], dtype=DTYPE), torch.empty(0, dtype=DTYPE))
    whole = model.step(initial, torch.empty(0, dtype=DTYPE), 1.0)
    half = model.step(initial, torch.empty(0, dtype=DTYPE), 0.5)
    halves = model.step(half, torch.empty(0, dtype=DTYPE), 0.5)
    torch.testing.assert_close(whole.temperature, halves.temperature, rtol=1e-12, atol=1e-12)


def test_actuator_step_matches_closed_form() -> None:
    spec = ThermalSystemSpec(
        node_names=("node",),
        heat_capacity=(1.0,),
        edges=(),
        actuators=(ActuatorSpec("heater", tau=5.0),),
    )
    model = ThermalRCModel(spec)
    out = model.actuator_step(
        torch.tensor([0.0], dtype=DTYPE), torch.tensor([10.0], dtype=DTYPE), 5.0
    )
    assert out.item() == torch.tensor(10.0 * (1.0 - math.exp(-1.0)), dtype=DTYPE).item()


def test_joint_exact_transition_matches_lagged_heat_analytic_solution() -> None:
    spec = ThermalSystemSpec(
        node_names=("node",),
        heat_capacity=(1.0,),
        edges=(),
        actuators=(ActuatorSpec("heater", tau=1.0),),
        sources=(SourceSpec("heat", (1.0,), PositivePartLawSpec("heater", 1.0)),),
    )
    model = ThermalRCModel(spec)
    initial = ThermalState(
        torch.tensor([0.0], dtype=DTYPE),
        torch.tensor([0.0], dtype=DTYPE),
    )
    whole = model.step(initial, torch.tensor([1.0], dtype=DTYPE), 1.0)
    half = model.step(initial, torch.tensor([1.0], dtype=DTYPE), 0.5)
    halves = model.step(half, torch.tensor([1.0], dtype=DTYPE), 0.5)
    expected = torch.tensor([math.exp(-1.0)], dtype=DTYPE)
    torch.testing.assert_close(whole.temperature, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(whole.temperature, halves.temperature, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(whole.actuator, halves.actuator, rtol=1e-12, atol=1e-12)


def test_joint_exact_transition_splits_at_source_threshold_crossing() -> None:
    spec = ThermalSystemSpec(
        node_names=("node",),
        heat_capacity=(1.0,),
        edges=(),
        actuators=(ActuatorSpec("heater", tau=1.0),),
        sources=(SourceSpec("heat", (1.0,), PositivePartLawSpec("heater", 1.0, threshold=0.5)),),
    )
    model = ThermalRCModel(spec)
    initial = ThermalState(
        torch.tensor([0.0], dtype=DTYPE),
        torch.tensor([0.0], dtype=DTYPE),
    )
    result = model.step(initial, torch.tensor([1.0], dtype=DTYPE), 2.0)
    crossing = math.log(2.0)
    expected = 0.5 * (2.0 - crossing) + math.exp(-2.0) - 0.5
    torch.testing.assert_close(
        result.temperature,
        torch.tensor([expected], dtype=DTYPE),
        rtol=1e-12,
        atol=1e-12,
    )
    result.temperature.sum().backward()
    tau_gradient = model.log_tau_multiplier.grad
    source_gradient = model.source_laws.log_scale_multiplier.grad
    assert tau_gradient is not None
    assert torch.isfinite(tau_gradient).all()
    assert source_gradient is not None
    assert torch.isfinite(source_gradient).all()


def test_joint_exact_batch_matches_individual_threshold_regimes() -> None:
    spec = ThermalSystemSpec(
        node_names=("node",),
        heat_capacity=(1.0,),
        edges=(),
        actuators=(ActuatorSpec("heater", tau=1.0),),
        sources=(SourceSpec("heat", (1.0,), PositivePartLawSpec("heater", 1.0, threshold=0.5)),),
    )
    model = ThermalRCModel(spec)
    initial_temperature = torch.tensor([[0.0], [2.0]], dtype=DTYPE)
    initial_actuator = torch.zeros((2, 1), dtype=DTYPE)
    commands = torch.tensor([[[1.0]], [[0.4]]], dtype=DTYPE)
    dt = torch.tensor([[2.0], [1.0]], dtype=DTYPE)
    batched_temperature, batched_actuator = model.forward_batch(
        initial_temperature,
        commands,
        dt,
        initial_actuator,
    )
    for index in range(2):
        individual = model.step(
            ThermalState(initial_temperature[index], initial_actuator[index]),
            commands[index, 0],
            dt[index, 0],
        )
        torch.testing.assert_close(batched_temperature[index, -1], individual.temperature)
        torch.testing.assert_close(batched_actuator[index, -1], individual.actuator)


def test_threshold_source_operates_in_physical_control_units() -> None:
    spec = ThermalSystemSpec(
        node_names=("node",),
        heat_capacity=(1.0,),
        edges=(),
        actuators=(ActuatorSpec("heater", tau=0.0),),
        sources=(
            SourceSpec(
                "heater_source",
                (1.0,),
                PositivePartLawSpec("heater", 2.0, threshold=10.0),
            ),
        ),
    )
    model = ThermalRCModel(spec)
    state = ThermalState(torch.tensor([20.0], dtype=DTYPE), torch.tensor([0.0], dtype=DTYPE))
    after = model.step(state, torch.tensor([12.0], dtype=DTYPE), 1.0)
    torch.testing.assert_close(after.temperature, torch.tensor([24.0], dtype=DTYPE))
    after.temperature.sum().backward()
    assert model.log_tau_multiplier.grad is None
    source_gradient = model.source_laws.log_scale_multiplier.grad
    assert source_gradient is not None
    assert torch.isfinite(source_gradient).all()


def test_boundary_relaxes_exactly_toward_its_temperature() -> None:
    spec = ThermalSystemSpec(
        node_names=("node",),
        heat_capacity=(1.0,),
        edges=(),
        actuators=(),
        boundaries=(
            BoundarySpec(
                "ambient",
                (1.0,),
                ReservoirTemperatureSpec(10.0),
                ConstantLawSpec(1.0),
            ),
        ),
    )
    model = ThermalRCModel(spec)
    state = ThermalState(torch.tensor([20.0], dtype=DTYPE), torch.empty(0, dtype=DTYPE))
    after = model.step(state, torch.empty(0, dtype=DTYPE), 1.0)
    expected = 10.0 + 10.0 * math.exp(-1.0)
    torch.testing.assert_close(after.temperature, torch.tensor([expected], dtype=DTYPE))


def test_power_law_boundary_uses_physical_flow_and_preserves_exact_step() -> None:
    spec = ThermalSystemSpec(
        node_names=("node",),
        heat_capacity=(2.0,),
        edges=(),
        actuators=(
            ActuatorSpec("coolant", tau=0.0, learnable=False),
            ActuatorSpec("flow", tau=0.0, learnable=False),
        ),
        boundaries=(
            BoundarySpec(
                "cooling",
                (1.0,),
                ReservoirTemperatureSpec(0.0, control="coolant", slope=1.0),
                PowerLawSpec(
                    "flow",
                    reference=1.0,
                    offset=0.2,
                    scale=0.8,
                    exponent=1.0,
                    exponent_learnable=True,
                ),
            ),
        ),
    )
    model = ThermalRCModel(spec)
    state = ThermalState(
        torch.tensor([30.0], dtype=DTYPE),
        torch.zeros(2, dtype=DTYPE),
    )
    command = torch.tensor([10.0, 1.0], dtype=DTYPE)
    after = model.step(state, command, 2.0)
    expected = 10.0 + 20.0 * math.exp(-1.0)
    torch.testing.assert_close(after.temperature, torch.tensor([expected], dtype=DTYPE))
    conductance = model.boundary_conductance(
        torch.tensor([[10.0, 0.0], [10.0, 1.0], [10.0, 2.0]], dtype=DTYPE)
    )
    assert torch.all(conductance[1:] > conductance[:-1])

    observer = KalmanObserver(model)
    observer_state = observer.initialize(
        torch.tensor([30.0], dtype=DTYPE),
        actuator=torch.zeros(2, dtype=DTYPE),
    )
    low_flow = observer.predict(observer_state, torch.tensor([10.0, 0.0], dtype=DTYPE), 1.0)
    high_flow = observer.predict(observer_state, torch.tensor([10.0, 2.0], dtype=DTYPE), 1.0)
    assert high_flow.temperature[0] < low_flow.temperature[0]

    after.temperature.sum().backward()
    forced_gradient = model.boundary_laws.log_scale_multiplier.grad
    exponent_gradient = model.boundary_laws.log_exponent_multiplier.grad
    assert forced_gradient is not None
    assert exponent_gradient is not None
    assert torch.isfinite(forced_gradient).all()
    assert torch.isfinite(exponent_gradient).all()


def test_input_dependent_internal_path_remains_reciprocal_and_conservative() -> None:
    spec = ThermalSystemSpec(
        node_names=("body", "stage"),
        heat_capacity=(2.0, 3.0),
        edges=(
            EdgeSpec(
                "body",
                "stage",
                PowerLawSpec(
                    "contact_input",
                    reference=1.0,
                    offset=0.2,
                    scale=0.8,
                    exponent=1.0,
                ),
            ),
        ),
        actuators=(ActuatorSpec("contact_input", tau=0.0, learnable=False),),
    )
    model = ThermalRCModel(spec)
    initial_temperature = torch.tensor([100.0, 20.0], dtype=DTYPE)
    initial = ThermalState(initial_temperature, torch.zeros(1, dtype=DTYPE))
    low = model.step(initial, torch.tensor([0.0], dtype=DTYPE), 1.0)
    high = model.step(initial, torch.tensor([2.0], dtype=DTYPE), 1.0)

    torch.testing.assert_close(
        model.stored_energy(low.temperature),
        model.stored_energy(initial_temperature),
    )
    torch.testing.assert_close(
        model.stored_energy(high.temperature),
        model.stored_energy(initial_temperature),
    )
    assert torch.diff(low.temperature).abs() > torch.diff(high.temperature).abs()
    torch.testing.assert_close(
        model.conductance(torch.tensor([2.0], dtype=DTYPE)), torch.tensor([1.8], dtype=DTYPE)
    )
    batched, _ = model.forward_batch(
        initial_temperature.repeat(2, 1),
        torch.tensor([[[0.0]], [[2.0]]], dtype=DTYPE),
        torch.tensor([1.0], dtype=DTYPE),
    )
    torch.testing.assert_close(batched[0, -1], low.temperature)
    torch.testing.assert_close(batched[1, -1], high.temperature)


def test_engine_owns_lagged_coefficient_integrator_compatibility() -> None:
    response = PowerLawSpec(
        "drive",
        reference=1.0,
        offset=0.1,
        scale=0.3,
        exponent=1.0,
    )
    spec = ThermalSystemSpec(
        node_names=("body", "stage"),
        heat_capacity=(2.0, 3.0),
        edges=(EdgeSpec("body", "stage", response),),
        actuators=(ActuatorSpec("drive", tau=2.0, learnable=False),),
        sources=(SourceSpec("process_heat", (1.0, 0.0), response),),
        boundaries=(
            BoundarySpec(
                "reservoir",
                (0.0, 1.0),
                ReservoirTemperatureSpec(20.0),
                response,
            ),
        ),
    )
    with pytest.raises(ValueError, match="exact integrator requires zero-tau"):
        ThermalRCModel(spec, integrator="exact")

    model = ThermalRCModel(spec, integrator="implicit")
    initial = ThermalState(
        torch.tensor([40.0, 20.0], dtype=DTYPE),
        torch.zeros(1, dtype=DTYPE),
    )
    result = model.step(initial, torch.tensor([2.0], dtype=DTYPE), 1.0)
    assert torch.isfinite(result.temperature).all()
    assert result.actuator[0] > 0.0

    observer = KalmanObserver(model)
    observer_state = observer.initialize(initial.temperature, actuator=initial.actuator)
    predicted = observer.predict(observer_state, torch.tensor([2.0], dtype=DTYPE), 1.0)
    assert torch.isfinite(predicted.temperature).all()
    assert torch.isfinite(predicted.covariance).all()


def test_power_law_can_drive_source_heat_rate_without_boundary_semantics() -> None:
    spec = ThermalSystemSpec(
        node_names=("body",),
        heat_capacity=(2.0,),
        edges=(),
        actuators=(ActuatorSpec("process_input", tau=0.0, learnable=False),),
        sources=(
            SourceSpec(
                "process_heat",
                (1.0,),
                PowerLawSpec(
                    "process_input",
                    reference=1.0,
                    offset=0.2,
                    scale=0.8,
                    exponent=1.0,
                ),
            ),
        ),
    )
    model = ThermalRCModel(spec)
    initial = ThermalState(
        torch.tensor([20.0], dtype=DTYPE),
        torch.zeros(1, dtype=DTYPE),
    )
    result = model.step(initial, torch.tensor([2.0], dtype=DTYPE), 3.0)
    torch.testing.assert_close(result.temperature, torch.tensor([22.7], dtype=DTYPE))
    torch.testing.assert_close(
        model.source_heat_rate(torch.tensor([2.0], dtype=DTYPE)),
        torch.tensor([1.8], dtype=DTYPE),
    )


def test_exact_batch_keeps_distinct_flow_dependent_operators() -> None:
    conductance_law = PowerLawSpec(
        "flow",
        reference=1.0,
        offset=0.2,
        scale=0.8,
        exponent=1.0,
    )
    spec = ThermalSystemSpec(
        node_names=("node",),
        heat_capacity=(1.0,),
        edges=(),
        actuators=(ActuatorSpec("flow", tau=0.0, learnable=False),),
        boundaries=(
            BoundarySpec(
                "ambient",
                (1.0,),
                ReservoirTemperatureSpec(10.0),
                conductance_law,
            ),
        ),
    )
    model = ThermalRCModel(spec)
    initial = torch.tensor([[30.0], [30.0]], dtype=DTYPE)
    commands = torch.tensor([[[0.0]], [[2.0]]], dtype=DTYPE)
    batched, _ = model.forward_batch(initial, commands, torch.tensor([1.0], dtype=DTYPE))
    for index in range(2):
        individual, _ = model.forward_trajectory(initial[index], commands[index], torch.ones(1))
        torch.testing.assert_close(batched[index], individual)
    assert batched[1, -1, 0] < batched[0, -1, 0]


def test_trajectory_loss_can_differentiate_through_the_integrator() -> None:
    model = conduction_model()
    temperatures, _ = model.forward_trajectory(
        torch.tensor([80.0, 20.0], dtype=DTYPE),
        torch.empty((3, 0), dtype=DTYPE),
        torch.ones(3, dtype=DTYPE),
    )
    temperatures[-1, 0].backward()
    edge_gradient = model.edge_laws.log_offset_multiplier.grad
    assert edge_gradient is not None
    assert torch.isfinite(edge_gradient).all()


def test_observer_updates_available_sensor_and_keeps_hidden_node() -> None:
    spec = ThermalSystemSpec(
        node_names=("hidden", "observed"),
        heat_capacity=(1.0, 1.0),
        edges=(EdgeSpec("hidden", "observed", ConstantLawSpec(0.2)),),
        actuators=(),
        sensor_names=("surface",),
        sensor_nodes=("observed",),
    )
    model = ThermalRCModel(spec)
    observer = KalmanObserver(model)
    state = observer.initialize(torch.tensor([50.0], dtype=DTYPE))
    updated, innovation = observer.step(
        state,
        torch.empty(0, dtype=DTYPE),
        torch.tensor([52.0], dtype=DTYPE),
        1.0,
    )
    assert updated.temperature.shape == (2,)
    assert innovation.shape == (1,)
    assert torch.isfinite(updated.covariance).all()


def test_observer_can_skip_a_completely_missing_measurement() -> None:
    model = conduction_model()
    observer = KalmanObserver(model)
    state = observer.initialize(torch.tensor([40.0, 50.0], dtype=DTYPE))
    predicted = observer.predict(state, torch.empty(0, dtype=DTYPE), 1.0)
    updated, innovation = observer.update(
        predicted,
        torch.tensor([float("nan"), float("nan")], dtype=DTYPE),
        torch.tensor([False, False]),
    )
    torch.testing.assert_close(updated.temperature, predicted.temperature)
    assert innovation.numel() == 0


def test_observer_limits_gross_sensor_outlier_state_correction() -> None:
    model = conduction_model()
    robust = KalmanObserver(model, sensor_std=0.1, innovation_gate_sigma=3.0)
    conventional = KalmanObserver(model, sensor_std=0.1, innovation_gate_sigma=1e9)
    state = robust.initialize(torch.tensor([20.0, 20.0], dtype=DTYPE))
    for _ in range(10):
        predicted = robust.predict(state, torch.empty(0, dtype=DTYPE), 1.0)
        state, _ = robust.update(predicted, torch.tensor([20.0, 20.0], dtype=DTYPE))

    predicted = robust.predict(state, torch.empty(0, dtype=DTYPE), 1.0)
    observation = torch.tensor([30.0, 20.0], dtype=DTYPE)
    robust_state, robust_innovation = robust.update(predicted, observation)
    conventional_state, conventional_innovation = conventional.update(predicted, observation)
    robust_shift = torch.linalg.vector_norm(robust_state.temperature - predicted.temperature)
    conventional_shift = torch.linalg.vector_norm(
        conventional_state.temperature - predicted.temperature
    )
    torch.testing.assert_close(robust_innovation, conventional_innovation)
    assert robust_shift < conventional_shift * 0.5


def test_observer_unknown_heat_uses_declared_source_path_and_physical_units() -> None:
    spec = ThermalSystemSpec(
        node_names=("heated", "passive"),
        heat_capacity=(2.0, 3.0),
        edges=(EdgeSpec("heated", "passive", ConstantLawSpec(0.4)),),
        actuators=(ActuatorSpec("heater", tau=0.0),),
        sources=(
            SourceSpec(
                "heater_source",
                (1.0, 0.0),
                PositivePartLawSpec("heater", 1.0),
            ),
        ),
    )
    model = ThermalRCModel(spec)
    observer = KalmanObserver(model, disturbance_process_std=0.1)
    state = observer.initialize(
        torch.tensor([20.0, 20.0], dtype=DTYPE),
        actuator=torch.tensor([0.0], dtype=DTYPE),
    )
    assert state.heat_disturbance.shape == (1,)
    disturbed = replace(state, heat_disturbance=torch.tensor([1.0], dtype=DTYPE))
    predicted = observer.predict(disturbed, torch.tensor([0.0], dtype=DTYPE), 1.0)
    node_heat = observer.node_heat_disturbance(predicted)
    torch.testing.assert_close(node_heat, torch.tensor([1.0, 0.0], dtype=DTYPE))
    assert predicted.temperature[0] > state.temperature[0]
    assert predicted.temperature[1] > state.temperature[1]
    assert torch.abs(predicted.covariance[0, model.n_nodes]) > 0.0
    assert torch.linalg.eigvalsh(predicted.covariance).min() >= -1e-12


def test_observer_process_covariance_matches_integrated_heat_random_walk() -> None:
    spec = ThermalSystemSpec(
        node_names=("node",),
        heat_capacity=(2.0,),
        edges=(),
        actuators=(ActuatorSpec("heater", tau=0.0),),
        sources=(SourceSpec("heater_source", (1.0,), PositivePartLawSpec("heater", 1.0)),),
    )
    observer = KalmanObserver(
        ThermalRCModel(spec),
        disturbance_process_std=0.1,
        bias_process_std=0.0,
        initial_temperature_std=0.0,
        initial_disturbance_std=0.0,
        initial_bias_std=0.0,
    )
    state = observer.initialize(
        torch.tensor([20.0], dtype=DTYPE),
        actuator=torch.tensor([0.0], dtype=DTYPE),
    )
    predicted = observer.predict(state, torch.tensor([0.0], dtype=DTYPE), 1.0)
    expected = torch.tensor(
        [[0.01 * 0.5**2 / 3.0, 0.01 * 0.5 / 2.0], [0.01 * 0.5 / 2.0, 0.01]],
        dtype=DTYPE,
    )
    torch.testing.assert_close(predicted.covariance, expected, rtol=1e-12, atol=1e-12)


def test_affine_integrators_support_batch_specific_timesteps() -> None:
    state = torch.tensor([[20.0], [20.0]], dtype=DTYPE)
    matrix = torch.tensor([[-0.5]], dtype=DTYPE)
    forcing = torch.tensor([[5.0], [5.0]], dtype=DTYPE)
    dt = torch.tensor([1.0, 2.0], dtype=DTYPE)
    exact = exact_affine_step(state, matrix, forcing, dt)
    expected = 10.0 + 10.0 * torch.exp(-0.5 * dt)
    torch.testing.assert_close(exact[:, 0], expected)

    implicit = implicit_euler_step(state, matrix, forcing, dt)
    torch.testing.assert_close(implicit[:, 0], (20.0 + 5.0 * dt) / (1.0 + 0.5 * dt))


def test_integrators_reject_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        exact_affine_step(torch.zeros(2), torch.eye(2), torch.zeros(3), 1.0)
    with pytest.raises(ValueError, match="incompatible"):
        implicit_euler_step(torch.zeros(2), torch.eye(3), torch.zeros(2), 1.0)


def test_forward_batch_variable_dt_matches_individual_rollouts() -> None:
    model = conduction_model()
    initial = torch.tensor([[80.0, 20.0], [50.0, 30.0]], dtype=DTYPE)
    commands = torch.empty((2, 2, 0), dtype=DTYPE)
    dt = torch.tensor([[1.0, 2.0], [0.5, 1.5]], dtype=DTYPE)
    batched, _ = model.forward_batch(initial, commands, dt)
    for index in range(2):
        individual, _ = model.forward_trajectory(initial[index], commands[index], dt[index])
        torch.testing.assert_close(batched[index], individual)


def test_implicit_model_uses_a_stable_batch_step() -> None:
    model = ThermalRCModel(conduction_model().spec, integrator="implicit")
    states, _ = model.forward_batch(
        torch.tensor([[100.0, 20.0]], dtype=DTYPE),
        torch.empty((1, 2, 0), dtype=DTYPE),
        torch.tensor([10.0, 10.0], dtype=DTYPE),
    )
    assert torch.isfinite(states).all()
    assert states.min() >= 20.0
    assert states.max() <= 100.0


def test_implicit_transition_matrix_matches_implicit_state_sensitivity() -> None:
    model = ThermalRCModel(conduction_model().spec, integrator="implicit")
    step = torch.tensor(2.0, dtype=torch.float64)
    transition = model.temperature_transition_matrix(step)
    matrix = model.system_matrix()
    identity = torch.eye(model.n_nodes, dtype=torch.float64)
    expected = torch.linalg.solve(identity - step * matrix, identity)
    torch.testing.assert_close(transition, expected)


def test_state_rejects_nonfinite_and_mismatched_batches() -> None:
    with pytest.raises(ValueError, match="batch shapes"):
        ThermalState(torch.zeros((2, 1)), torch.zeros((3, 1)))
    with pytest.raises(ValueError, match="finite"):
        ThermalState(torch.tensor([float("nan")]), torch.zeros(1))
