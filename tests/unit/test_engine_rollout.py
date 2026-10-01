"""Physical and numerical invariants of the shared thermal engine."""

from __future__ import annotations

import pytest
import torch

from celltemp.engine import ThermalRCModel, ThermalState
from celltemp.engine.integrator import exact_affine_step, implicit_euler_step
from tests.unit._engine_cases import DTYPE, conduction_model


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
