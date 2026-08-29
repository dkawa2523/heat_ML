"""Physical and numerical invariants of the shared thermal engine."""

from __future__ import annotations

import math

import pytest
import torch

from celltemp.domain import (
    ActuatorSpec,
    BoundarySpec,
    EdgeSpec,
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
        edges=(EdgeSpec("a", "b", 0.5),),
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


def test_threshold_source_operates_in_physical_control_units() -> None:
    spec = ThermalSystemSpec(
        node_names=("node",),
        heat_capacity=(1.0,),
        edges=(),
        actuators=(ActuatorSpec("heater", tau=0.0),),
        sources=(SourceSpec("heater_source", "heater", (1.0,), gain=2.0, threshold=10.0),),
    )
    model = ThermalRCModel(spec)
    state = ThermalState(torch.tensor([20.0], dtype=DTYPE), torch.tensor([0.0], dtype=DTYPE))
    after = model.step(state, torch.tensor([12.0], dtype=DTYPE), 1.0)
    torch.testing.assert_close(after.temperature, torch.tensor([24.0], dtype=DTYPE))


def test_boundary_relaxes_exactly_toward_its_temperature() -> None:
    spec = ThermalSystemSpec(
        node_names=("node",),
        heat_capacity=(1.0,),
        edges=(),
        actuators=(),
        boundaries=(BoundarySpec("ambient", (1.0,), conductance=1.0, temperature_intercept=10.0),),
    )
    model = ThermalRCModel(spec)
    state = ThermalState(torch.tensor([20.0], dtype=DTYPE), torch.empty(0, dtype=DTYPE))
    after = model.step(state, torch.empty(0, dtype=DTYPE), 1.0)
    expected = 10.0 + 10.0 * math.exp(-1.0)
    torch.testing.assert_close(after.temperature, torch.tensor([expected], dtype=DTYPE))


def test_trajectory_loss_can_differentiate_through_the_integrator() -> None:
    model = conduction_model()
    temperatures, _ = model.forward_trajectory(
        torch.tensor([80.0, 20.0], dtype=DTYPE),
        torch.empty((3, 0), dtype=DTYPE),
        torch.ones(3, dtype=DTYPE),
    )
    temperatures[-1, 0].backward()
    assert model.log_edge_multiplier.grad is not None
    assert torch.isfinite(model.log_edge_multiplier.grad).all()


def test_observer_updates_available_sensor_and_keeps_hidden_node() -> None:
    spec = ThermalSystemSpec(
        node_names=("hidden", "observed"),
        heat_capacity=(1.0, 1.0),
        edges=(EdgeSpec("hidden", "observed", 0.2),),
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
