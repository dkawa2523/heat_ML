"""Serializable definitions of a lumped thermal system."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _tuple_floats(values: tuple[float, ...] | list[float] | np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class EdgeSpec:
    """One undirected conductive path between two thermal nodes."""

    node_a: str
    node_b: str
    conductance: float
    learnable: bool = True

    def __post_init__(self) -> None:
        if self.node_a == self.node_b:
            raise ValueError("a conductive edge must connect two different nodes")
        if not np.isfinite(self.conductance) or self.conductance <= 0.0:
            raise ValueError("edge conductance must be positive and finite")


@dataclass(frozen=True)
class ActuatorSpec:
    """A commanded input with an optional first-order response delay."""

    name: str
    tau: float = 0.0
    learnable: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("actuator name must not be empty")
        if not np.isfinite(self.tau) or self.tau < 0.0:
            raise ValueError("actuator tau must be finite and non-negative")


@dataclass(frozen=True)
class SourceSpec:
    """A non-negative heat source driven by one actuator.

    ``node_weights`` describes how the heat is distributed. ``gain`` converts the
    positive part of ``actuator - threshold`` into heat rate.
    """

    name: str
    actuator: str
    node_weights: tuple[float, ...]
    gain: float
    threshold: float = 0.0
    learnable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_weights", _tuple_floats(self.node_weights))
        if not self.name:
            raise ValueError("source name must not be empty")
        if not np.isfinite(self.gain) or self.gain < 0.0:
            raise ValueError("source gain must be finite and non-negative")
        if self.learnable and self.gain == 0.0:
            raise ValueError("a learnable source gain must have a positive prior")
        if not np.isfinite(self.threshold):
            raise ValueError("source threshold must be finite")
        weights = np.asarray(self.node_weights)
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("source node weights must be finite and non-negative")


@dataclass(frozen=True)
class BoundarySpec:
    """A convective boundary with an affine boundary temperature.

    ``temperature = intercept + slope * actuator``.  Set ``actuator=None`` for a
    constant ambient boundary.  Conductance and node weights are non-negative, so
    the boundary always pulls a node toward its temperature rather than acting as
    an unbounded signed source.
    """

    name: str
    node_weights: tuple[float, ...]
    conductance: float
    temperature_intercept: float
    actuator: str | None = None
    temperature_slope: float = 0.0
    learnable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_weights", _tuple_floats(self.node_weights))
        if not self.name:
            raise ValueError("boundary name must not be empty")
        if not np.isfinite(self.conductance) or self.conductance < 0.0:
            raise ValueError("boundary conductance must be finite and non-negative")
        if self.learnable and self.conductance == 0.0:
            raise ValueError("a learnable boundary conductance must have a positive prior")
        if not np.isfinite(self.node_weights).all() or np.any(np.asarray(self.node_weights) < 0.0):
            raise ValueError("boundary node weights must be finite and non-negative")
        if not np.isfinite(self.temperature_intercept) or not np.isfinite(self.temperature_slope):
            raise ValueError("boundary temperature parameters must be finite")


@dataclass(frozen=True)
class ThermalSystemSpec:
    """Topology, inputs, and observations for a lumped thermal system."""

    node_names: tuple[str, ...]
    heat_capacity: tuple[float, ...]
    edges: tuple[EdgeSpec, ...]
    actuators: tuple[ActuatorSpec, ...]
    sources: tuple[SourceSpec, ...] = ()
    boundaries: tuple[BoundarySpec, ...] = ()
    sensor_names: tuple[str, ...] = ()
    sensor_nodes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_names", tuple(self.node_names))
        object.__setattr__(self, "heat_capacity", _tuple_floats(self.heat_capacity))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "actuators", tuple(self.actuators))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        sensors = tuple(self.sensor_names) or tuple(self.node_names)
        sensor_nodes = tuple(self.sensor_nodes) or sensors
        object.__setattr__(self, "sensor_names", sensors)
        object.__setattr__(self, "sensor_nodes", sensor_nodes)

        self._validate_dimensions()
        nodes = set(self.node_names)
        actuators = {actuator.name for actuator in self.actuators}
        self._validate_edges(nodes)
        self._validate_inputs(actuators)
        self._validate_sensors(nodes)

    def _validate_dimensions(self) -> None:
        if not self.node_names or len(set(self.node_names)) != len(self.node_names):
            raise ValueError("node_names must be non-empty and unique")
        capacity = np.asarray(self.heat_capacity)
        if capacity.shape != (len(self.node_names),):
            raise ValueError("heat_capacity must have one value per node")
        if not np.isfinite(capacity).all() or np.any(capacity <= 0.0):
            raise ValueError("heat_capacity must contain positive finite values")
        if len({actuator.name for actuator in self.actuators}) != len(self.actuators):
            raise ValueError("actuator names must be unique")
        if len({source.name for source in self.sources}) != len(self.sources):
            raise ValueError("source names must be unique")
        if len({boundary.name for boundary in self.boundaries}) != len(self.boundaries):
            raise ValueError("boundary names must be unique")
        if len(self.sensor_names) != len(self.sensor_nodes):
            raise ValueError("sensor_names and sensor_nodes must have equal length")
        if len(set(self.sensor_names)) != len(self.sensor_names):
            raise ValueError("sensor_names must be unique")

    def _validate_edges(self, nodes: set[str]) -> None:
        edge_keys: set[frozenset[str]] = set()
        for edge in self.edges:
            if edge.node_a not in nodes or edge.node_b not in nodes:
                raise ValueError(f"edge {edge.node_a}-{edge.node_b} refers to an unknown node")
            key = frozenset((edge.node_a, edge.node_b))
            if key in edge_keys:
                raise ValueError(f"duplicate undirected edge {edge.node_a}-{edge.node_b}")
            edge_keys.add(key)

    def _validate_inputs(self, actuators: set[str]) -> None:
        for source in self.sources:
            if source.actuator not in actuators:
                raise ValueError(
                    f"source {source.name} refers to unknown actuator {source.actuator}"
                )
            if len(source.node_weights) != len(self.node_names):
                raise ValueError(f"source {source.name} must have one weight per node")
        for boundary in self.boundaries:
            if boundary.actuator is not None and boundary.actuator not in actuators:
                raise ValueError(
                    f"boundary {boundary.name} refers to unknown actuator {boundary.actuator}"
                )
            if len(boundary.node_weights) != len(self.node_names):
                raise ValueError(f"boundary {boundary.name} must have one weight per node")

    def _validate_sensors(self, nodes: set[str]) -> None:
        unknown_sensor_nodes = [name for name in self.sensor_nodes if name not in nodes]
        if unknown_sensor_nodes:
            raise ValueError(f"sensor mapping refers to unknown nodes {unknown_sensor_nodes}")

    @property
    def control_names(self) -> tuple[str, ...]:
        return tuple(actuator.name for actuator in self.actuators)

    @property
    def observation_matrix(self) -> np.ndarray:
        matrix = np.zeros((len(self.sensor_names), len(self.node_names)), dtype=np.float64)
        node_index = {name: i for i, name in enumerate(self.node_names)}
        for sensor_index, node_name in enumerate(self.sensor_nodes):
            matrix[sensor_index, node_index[node_name]] = 1.0
        return matrix
