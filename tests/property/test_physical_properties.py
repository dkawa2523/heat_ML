"""Generated checks for the invariants the engine must preserve for any timestep."""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from celltemp.domain import ActuatorSpec, ConstantLawSpec, EdgeSpec, ThermalSystemSpec
from celltemp.engine import ThermalRCModel, ThermalState
from celltemp.engine.integrator import exact_process_operators


def _closed_pair() -> ThermalRCModel:
    return ThermalRCModel(
        ThermalSystemSpec(
            node_names=("a", "b"),
            heat_capacity=(1.5, 3.0),
            edges=(EdgeSpec("a", "b", ConstantLawSpec(0.4, learnable=False)),),
            actuators=(ActuatorSpec("unused", tau=0.0),),
        )
    )


@settings(max_examples=30, deadline=None)
@given(
    first=st.floats(-100.0, 300.0, allow_nan=False, allow_infinity=False),
    second=st.floats(-100.0, 300.0, allow_nan=False, allow_infinity=False),
    dt=st.floats(1e-4, 100.0, allow_nan=False, allow_infinity=False),
)
def test_closed_conduction_conserves_energy_and_stays_bounded(
    first: float, second: float, dt: float
) -> None:
    model = _closed_pair()
    temperature = torch.tensor([first, second], dtype=torch.float64)
    state = ThermalState(temperature, torch.zeros(1, dtype=torch.float64))
    result = model.step(state, torch.zeros(1, dtype=torch.float64), dt)
    torch.testing.assert_close(
        model.stored_energy(result.temperature), model.stored_energy(temperature)
    )
    assert result.temperature.min() >= temperature.min() - 1e-9
    assert result.temperature.max() <= temperature.max() + 1e-9


@settings(max_examples=30, deadline=None)
@given(
    initial=st.floats(-200.0, 200.0, allow_nan=False, allow_infinity=False),
    command=st.floats(-200.0, 200.0, allow_nan=False, allow_infinity=False),
    tau=st.floats(1e-3, 100.0, allow_nan=False, allow_infinity=False),
    dt=st.floats(1e-4, 100.0, allow_nan=False, allow_infinity=False),
)
def test_first_order_actuator_never_overshoots(
    initial: float, command: float, tau: float, dt: float
) -> None:
    model = ThermalRCModel(
        ThermalSystemSpec(
            node_names=("body",),
            heat_capacity=(1.0,),
            edges=(),
            actuators=(ActuatorSpec("power", tau=tau, learnable=False),),
        )
    )
    result = model.actuator_step(
        torch.tensor([initial], dtype=torch.float64),
        torch.tensor([command], dtype=torch.float64),
        dt,
    )
    assert min(initial, command) - 1e-10 <= result.item() <= max(initial, command) + 1e-10


@settings(max_examples=20, deadline=None)
@given(
    conductance=st.floats(0.01, 100.0, allow_nan=False, allow_infinity=False),
    capacity=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False),
    dt=st.floats(0.01, 10000.0, allow_nan=False, allow_infinity=False),
)
def test_unknown_heat_covariance_is_finite_and_positive_for_stiff_intervals(
    conductance: float,
    capacity: float,
    dt: float,
) -> None:
    matrix = torch.tensor(
        [[-conductance / capacity, 1.0 / capacity], [0.0, 0.0]], dtype=torch.float64
    )
    density = torch.diag(torch.tensor([0.0, 0.01], dtype=torch.float64))
    transition, covariance = exact_process_operators(matrix, density, dt)
    assert torch.isfinite(transition).all()
    assert torch.isfinite(covariance).all()
    torch.testing.assert_close(covariance, covariance.T, rtol=0.0, atol=1e-12)
    scale = max(1.0, float(torch.diag(covariance).max()))
    assert torch.linalg.eigvalsh(covariance).min() >= -1e-12 * scale
    torch.testing.assert_close(covariance[1, 1], torch.tensor(0.01 * dt, dtype=torch.float64))
