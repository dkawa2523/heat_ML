"""Shared thermal-inference cases used by focused unit-test modules."""

import numpy as np

from celltemp.domain import (
    ActuatorSpec,
    BoundarySpec,
    ConstantLawSpec,
    ReservoirTemperatureSpec,
    ThermalSystemSpec,
    Trajectory,
)
from celltemp.engine import ThermalRCModel


def cooling_model() -> ThermalRCModel:
    spec = ThermalSystemSpec(
        node_names=("core",),
        heat_capacity=(2.0,),
        edges=(),
        actuators=(ActuatorSpec("coolant", tau=0.0),),
        boundaries=(
            BoundarySpec(
                "cooling",
                (1.0,),
                ReservoirTemperatureSpec(10.0),
                ConstantLawSpec(0.5),
            ),
        ),
    )
    return ThermalRCModel(spec)


def duplicate_sensor_model() -> ThermalRCModel:
    base = cooling_model().spec
    return ThermalRCModel(
        ThermalSystemSpec(
            node_names=base.node_names,
            heat_capacity=base.heat_capacity,
            edges=base.edges,
            actuators=base.actuators,
            boundaries=base.boundaries,
            sensor_names=("left", "right"),
            sensor_nodes=("core", "core"),
        )
    )


def forecast_request(
    time: np.ndarray,
    initial_temperature: float,
    *,
    control_name: str = "coolant",
) -> Trajectory:
    temperature = np.full((len(time), 1), np.nan)
    temperature[0, 0] = initial_temperature
    return Trajectory(
        case_id="case",
        time=time,
        temperature=temperature,
        commands=np.zeros((len(time) - 1, 1)),
        sensor_names=("core",),
        control_names=(control_name,),
    )
