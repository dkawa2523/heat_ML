"""Domain objects: interval alignment, missing values, and topology invariants."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

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
    Trajectory,
)


def test_sampled_controls_use_left_interval_convention_by_default() -> None:
    sampled = np.array([[0.0], [10.0], [20.0]])
    trajectory = Trajectory.from_sampled_controls(
        case_id="step",
        time=np.array([0.0, 1.0, 2.0]),
        temperature=np.array([[20.0], [20.0], [21.0]]),
        sampled_controls=sampled,
        sensor_names=["temperature"],
        control_names=["heater"],
    )
    np.testing.assert_array_equal(trajectory.commands[:, 0], [0.0, 10.0])
    np.testing.assert_array_equal(trajectory.dt, [1.0, 1.0])


def test_right_interval_convention_is_explicit() -> None:
    trajectory = Trajectory.from_sampled_controls(
        case_id="right-labelled-export",
        time=np.array([0.0, 1.0, 2.0]),
        temperature=np.zeros((3, 1)),
        sampled_controls=np.array([[0.0], [10.0], [20.0]]),
        sensor_names=["temperature"],
        control_names=["heater"],
        convention="right",
    )
    np.testing.assert_array_equal(trajectory.commands[:, 0], [10.0, 20.0])


def test_missing_temperature_is_allowed_only_when_masked() -> None:
    trajectory = Trajectory(
        case_id="missing",
        time=np.array([0.0, 1.0]),
        temperature=np.array([[20.0, np.nan], [21.0, 22.0]]),
        commands=np.empty((1, 0)),
        sensor_names=("a", "b"),
        control_names=(),
        observation_mask=np.array([[True, False], [True, True]]),
    )
    assert not trajectory.mask[0, 1]


def test_trajectory_owns_read_only_array_values() -> None:
    time = np.array([0.0, 1.0])
    temperature = np.array([[20.0], [21.0]])
    trajectory = Trajectory(
        case_id="immutable",
        time=time,
        temperature=temperature,
        commands=np.empty((1, 0)),
        sensor_names=("temperature",),
        control_names=(),
    )
    time[0] = 99.0
    temperature[0, 0] = 99.0
    assert trajectory.time[0] == 0.0
    assert trajectory.temperature[0, 0] == 20.0
    assert not trajectory.time.flags.writeable
    assert not trajectory.temperature.flags.writeable


def test_trajectory_rejects_sample_length_controls() -> None:
    with pytest.raises(ValueError, match="interval shape"):
        Trajectory(
            case_id="ambiguous",
            time=np.array([0.0, 1.0, 2.0]),
            temperature=np.zeros((3, 1)),
            commands=np.zeros((3, 1)),
            sensor_names=("temperature",),
            control_names=("heater",),
        )


def test_observation_mapping_is_independent_of_state_nodes() -> None:
    spec = ThermalSystemSpec(
        node_names=("hot", "middle", "edge"),
        heat_capacity=(1.0, 2.0, 3.0),
        edges=(
            EdgeSpec("hot", "middle", ConstantLawSpec(0.5)),
            EdgeSpec("middle", "edge", ConstantLawSpec(0.25)),
        ),
        actuators=(ActuatorSpec("heater", tau=5.0),),
        sensor_names=("middle_sensor", "edge_sensor"),
        sensor_nodes=("middle", "edge"),
    )
    np.testing.assert_array_equal(
        spec.observation_matrix,
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )


def test_duplicate_reverse_edge_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate undirected edge"):
        ThermalSystemSpec(
            node_names=("a", "b"),
            heat_capacity=(1.0, 1.0),
            edges=(
                EdgeSpec("a", "b", ConstantLawSpec(1.0)),
                EdgeSpec("b", "a", ConstantLawSpec(1.0)),
            ),
            actuators=(),
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EdgeSpec("a", "a", ConstantLawSpec(1.0)),
        lambda: ConstantLawSpec(0.0),
        lambda: ActuatorSpec("", 1.0),
        lambda: ActuatorSpec("power", -1.0),
        lambda: ActuatorSpec("power", 0.0, learnable=True),
        lambda: PositivePartLawSpec("power", -1.0),
        lambda: SourceSpec("source", (-1.0,), PositivePartLawSpec("power", 1.0)),
        lambda: SourceSpec("source", (0.0,), PositivePartLawSpec("power", 1.0)),
        lambda: PositivePartLawSpec("power", 0.0),
        lambda: PositivePartLawSpec("power", 1.0, threshold=np.inf),
        lambda: BoundarySpec(
            "ambient",
            (-1.0,),
            ReservoirTemperatureSpec(20.0),
            ConstantLawSpec(1.0),
        ),
        lambda: BoundarySpec(
            "ambient",
            (0.0,),
            ReservoirTemperatureSpec(20.0),
            ConstantLawSpec(1.0),
        ),
        lambda: ReservoirTemperatureSpec(20.0, slope=1.0),
        lambda: PowerLawSpec("flow", 0.0, 0.1, 1.0, 0.8),
    ],
)
def test_component_specs_reject_nonphysical_values(factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match=r"must|positive|non-negative|different"):
        factory()


def test_system_rejects_unknown_references_and_bad_capacity() -> None:
    with pytest.raises(ValueError, match="heat_capacity"):
        ThermalSystemSpec(("a",), (0.0,), (), ())
    with pytest.raises(ValueError, match="unknown node"):
        ThermalSystemSpec(("a",), (1.0,), (EdgeSpec("a", "b", ConstantLawSpec(1.0)),), ())
    with pytest.raises(ValueError, match="unknown actuator"):
        ThermalSystemSpec(
            ("a",),
            (1.0,),
            (),
            (),
            sources=(SourceSpec("heat", (1.0,), PositivePartLawSpec("missing", 1.0)),),
        )


def test_actuator_tau_is_learned_only_when_a_positive_prior_exists() -> None:
    assert ActuatorSpec("direct", 0.0).learnable is False
    assert ActuatorSpec("lagged", 1.0).learnable is True


def test_system_rejects_duplicate_source_names() -> None:
    actuator = ActuatorSpec("power", 0.0, learnable=False)
    source = SourceSpec("heat", (1.0,), PositivePartLawSpec("power", 1.0))
    with pytest.raises(ValueError, match="source names must be unique"):
        ThermalSystemSpec(("node",), (1.0,), (), (actuator,), sources=(source, source))


def test_system_rejects_sensor_actuator_name_collision() -> None:
    with pytest.raises(ValueError, match="sensor and actuator names must be distinct"):
        ThermalSystemSpec(
            ("node",),
            (1.0,),
            (),
            (ActuatorSpec("value", 0.0),),
            sensor_names=("value",),
            sensor_nodes=("node",),
        )


def test_system_rejects_duplicate_boundary_names() -> None:
    boundary = BoundarySpec(
        "ambient",
        (1.0,),
        ReservoirTemperatureSpec(20.0),
        ConstantLawSpec(1.0),
    )
    with pytest.raises(ValueError, match="boundary names must be unique"):
        ThermalSystemSpec(("node",), (1.0,), (), (), boundaries=(boundary, boundary))


def test_input_dependent_conductance_validates_only_domain_references() -> None:
    boundary = BoundarySpec(
        "cooling",
        (1.0,),
        ReservoirTemperatureSpec(20.0),
        PowerLawSpec("flow", 1.0, 0.1, 0.5, 0.8),
    )
    with pytest.raises(ValueError, match="unknown actuator"):
        ThermalSystemSpec(("node",), (1.0,), (), (), boundaries=(boundary,))
    spec = ThermalSystemSpec(
        ("node",),
        (1.0,),
        (),
        (ActuatorSpec("flow", tau=1.0),),
        boundaries=(boundary,),
    )
    assert spec.boundaries == (boundary,)
