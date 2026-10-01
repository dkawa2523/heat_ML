"""Physical and numerical invariants of the shared thermal engine."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from celltemp.domain import (
    ActuatorSpec,
    BoundarySpec,
    ConstantLawSpec,
    EdgeSpec,
    PositivePartLawSpec,
    ReservoirTemperatureSpec,
    SourceSpec,
    ThermalSystemSpec,
)
from celltemp.engine import KalmanObserver, ThermalRCModel
from tests.unit._engine_cases import DTYPE, conduction_model


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


def test_observer_initial_posterior_assimilates_first_measurement_covariance() -> None:
    model = conduction_model()
    observer = KalmanObserver(model, initial_temperature_std=10.0, sensor_std=0.1)
    observation = torch.tensor([40.0, 50.0], dtype=DTYPE)
    prior = observer.initialize(observation)
    posterior = observer.initialize_posterior(observation)

    torch.testing.assert_close(posterior.temperature, prior.temperature)
    assert torch.all(
        torch.diag(posterior.covariance)[: model.n_nodes]
        < torch.diag(prior.covariance)[: model.n_nodes]
    )


@pytest.mark.parametrize(
    ("sensor_std", "innovation_gate_sigma"),
    [
        (0.0, 0.15),
        (float("nan"), 0.15),
        (0.15, float("inf")),
    ],
)
def test_observer_rejects_nonfinite_configuration(
    sensor_std: float, innovation_gate_sigma: float
) -> None:
    with pytest.raises(ValueError, match="finite"):
        KalmanObserver(
            conduction_model(),
            sensor_std=sensor_std,
            innovation_gate_sigma=innovation_gate_sigma,
        )


def test_observer_cache_depends_on_dynamics_not_source_command() -> None:
    spec = ThermalSystemSpec(
        node_names=("node",),
        heat_capacity=(2.0,),
        edges=(),
        actuators=(ActuatorSpec("heater", tau=0.0),),
        sources=(SourceSpec("heat", (1.0,), PositivePartLawSpec("heater", 1.0)),),
    )
    observer = KalmanObserver(ThermalRCModel(spec))
    observer._operators(1.0, torch.tensor([0.0], dtype=DTYPE))
    observer._operators(1.0, torch.tensor([100.0], dtype=DTYPE))

    assert len(observer._operator_cache) == 1


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


def test_implicit_observer_uses_matching_backward_euler_process_covariance() -> None:
    spec = ThermalSystemSpec(
        node_names=("node",),
        heat_capacity=(1.0,),
        edges=(),
        actuators=(),
        boundaries=(
            BoundarySpec(
                "ambient",
                (1.0,),
                ReservoirTemperatureSpec(0.0),
                ConstantLawSpec(1.0, learnable=False),
            ),
        ),
    )
    observer = KalmanObserver(
        ThermalRCModel(spec, integrator="implicit"),
        disturbance_process_std=0.2,
        bias_process_std=0.0,
    )
    transition, process_covariance = observer._operators(5.0, torch.empty(0, dtype=DTYPE))
    density = observer._process_spectral_density()
    expected = transition @ (density * 5.0) @ transition.T

    torch.testing.assert_close(process_covariance, expected)
