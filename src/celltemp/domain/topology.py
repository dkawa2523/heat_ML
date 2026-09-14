"""Framework-independent definitions of a lumped thermal network."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _tuple_floats(values: tuple[float, ...] | list[float] | np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class ConstantLawSpec:
    """A positive scalar that is independent of the system inputs."""

    value: float
    learnable: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.value) or self.value <= 0.0:
            raise ValueError("constant law value must be positive and finite")


@dataclass(frozen=True)
class PositivePartLawSpec:
    """A non-negative linear response above a physical input threshold.

    ``value(u) = gain * max(u - threshold, 0)``.
    """

    control: str
    gain: float
    threshold: float = 0.0
    learnable: bool = True

    def __post_init__(self) -> None:
        if not self.control:
            raise ValueError("positive-part law control must not be empty")
        if not np.isfinite(self.gain) or self.gain <= 0.0:
            raise ValueError("positive-part law gain must be positive and finite")
        if not np.isfinite(self.threshold):
            raise ValueError("positive-part law threshold must be finite")


@dataclass(frozen=True)
class PowerLawSpec:
    """A positive offset plus a monotone response to a physical input.

    ``value(u) = offset + scale * (max(u, 0) / reference) ** exponent``.
    The value may represent conductance or heat rate depending on the thermal path
    that owns the law. Individual learnability flags keep weakly identifiable
    shape parameters fixed unless the available experiments can support them.
    """

    control: str
    reference: float
    offset: float
    scale: float
    exponent: float
    offset_learnable: bool = False
    scale_learnable: bool = True
    exponent_learnable: bool = False

    def __post_init__(self) -> None:
        if not self.control:
            raise ValueError("power-law control must not be empty")
        if not np.isfinite(self.reference) or self.reference <= 0.0:
            raise ValueError("power-law reference must be positive and finite")
        if not np.isfinite(self.offset) or self.offset < 0.0:
            raise ValueError("power-law offset must be finite and non-negative")
        if not np.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError("power-law scale must be finite and non-negative")
        if self.offset == 0.0 and self.scale == 0.0:
            raise ValueError("power-law offset or scale must be positive")
        if not np.isfinite(self.exponent) or self.exponent <= 0.0:
            raise ValueError("power-law exponent must be positive and finite")
        if self.offset_learnable and self.offset == 0.0:
            raise ValueError("a learnable power-law offset must have a positive prior")
        if self.scale_learnable and self.scale == 0.0:
            raise ValueError("a learnable power-law scale must have a positive prior")


ScalarLawSpec = ConstantLawSpec | PositivePartLawSpec | PowerLawSpec


def _law_control(law: ScalarLawSpec) -> str | None:
    return None if isinstance(law, ConstantLawSpec) else law.control


def _check_law(law: object, owner: str) -> None:
    if not isinstance(law, (ConstantLawSpec, PositivePartLawSpec, PowerLawSpec)):
        raise TypeError(f"{owner} must use a scalar law specification")


@dataclass(frozen=True)
class EdgeSpec:
    """One undirected heat-transfer path between two thermal nodes."""

    node_a: str
    node_b: str
    conductance: ScalarLawSpec

    def __post_init__(self) -> None:
        if self.node_a == self.node_b:
            raise ValueError("a conductive edge must connect two different nodes")
        _check_law(self.conductance, "edge conductance")


@dataclass(frozen=True)
class ActuatorSpec:
    """A measured or commanded input with an optional first-order delay."""

    name: str
    tau: float = 0.0
    learnable: bool | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("actuator name must not be empty")
        if not np.isfinite(self.tau) or self.tau < 0.0:
            raise ValueError("actuator tau must be finite and non-negative")
        if self.learnable is None:
            object.__setattr__(self, "learnable", self.tau > 0.0)
        elif not isinstance(self.learnable, bool):
            raise TypeError("actuator learnable must be boolean")
        if self.learnable and self.tau == 0.0:
            raise ValueError("a learnable actuator tau must have a positive prior")


@dataclass(frozen=True)
class SourceSpec:
    """A non-negative heat-rate law distributed over thermal nodes."""

    name: str
    node_weights: tuple[float, ...]
    heat_rate: ScalarLawSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_weights", _tuple_floats(self.node_weights))
        if not self.name:
            raise ValueError("source name must not be empty")
        _check_law(self.heat_rate, "source heat_rate")
        weights = np.asarray(self.node_weights)
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("source node weights must be finite and non-negative")
        if not np.any(weights > 0.0):
            raise ValueError("source node weights must contain a positive value")


@dataclass(frozen=True)
class ReservoirTemperatureSpec:
    """Affine reservoir temperature in physical input units.

    ``temperature = intercept + slope * control``. A missing control represents a
    constant reservoir and therefore requires a zero slope.
    """

    intercept: float
    control: str | None = None
    slope: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.intercept) or not np.isfinite(self.slope):
            raise ValueError("reservoir temperature parameters must be finite")
        if self.control is None and self.slope != 0.0:
            raise ValueError("a constant reservoir must have zero slope")
        if self.control == "":
            raise ValueError("reservoir control must not be empty")


@dataclass(frozen=True)
class BoundarySpec:
    """Heat exchange between weighted nodes and one thermal reservoir."""

    name: str
    node_weights: tuple[float, ...]
    reservoir_temperature: ReservoirTemperatureSpec
    conductance: ScalarLawSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_weights", _tuple_floats(self.node_weights))
        if not self.name:
            raise ValueError("boundary name must not be empty")
        _check_law(self.conductance, "boundary conductance")
        weights = np.asarray(self.node_weights)
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("boundary node weights must be finite and non-negative")
        if not np.any(weights > 0.0):
            raise ValueError("boundary node weights must contain a positive value")


@dataclass(frozen=True)
class ThermalSystemSpec:
    """Topology, inputs, heat paths, and observations for a thermal system."""

    node_names: tuple[str, ...]
    heat_capacity: tuple[float, ...]
    edges: tuple[EdgeSpec, ...]
    actuators: tuple[ActuatorSpec, ...]
    sources: tuple[SourceSpec, ...] = ()
    boundaries: tuple[BoundarySpec, ...] = ()
    sensor_names: tuple[str, ...] = ()
    sensor_nodes: tuple[str, ...] = ()
    sensor_weights: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_names", tuple(self.node_names))
        object.__setattr__(self, "heat_capacity", _tuple_floats(self.heat_capacity))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "actuators", tuple(self.actuators))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        sensor_weights = tuple(_tuple_floats(row) for row in self.sensor_weights)
        if sensor_weights and not self.sensor_names:
            raise ValueError("sensor_names are required with sensor_weights")
        if sensor_weights and self.sensor_nodes:
            raise ValueError("use either sensor_nodes or sensor_weights, not both")
        sensors = tuple(self.sensor_names) or tuple(self.node_names)
        sensor_nodes = () if sensor_weights else tuple(self.sensor_nodes) or sensors
        object.__setattr__(self, "sensor_names", sensors)
        object.__setattr__(self, "sensor_nodes", sensor_nodes)
        object.__setattr__(self, "sensor_weights", sensor_weights)

        self._validate_dimensions()
        nodes = set(self.node_names)
        actuators = {actuator.name: actuator for actuator in self.actuators}
        self._validate_edges(nodes, actuators)
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
        sensor_layout_size = (
            len(self.sensor_weights) if self.sensor_weights else len(self.sensor_nodes)
        )
        if len(self.sensor_names) != sensor_layout_size:
            raise ValueError("sensor names and observation mappings must have equal length")
        if len(set(self.sensor_names)) != len(self.sensor_names):
            raise ValueError("sensor_names must be unique")
        if set(self.sensor_names) & {actuator.name for actuator in self.actuators}:
            raise ValueError("sensor and actuator names must be distinct")

    @staticmethod
    def _validate_law_control(
        law: ScalarLawSpec,
        actuators: dict[str, ActuatorSpec],
        owner: str,
    ) -> None:
        control = _law_control(law)
        if control is None:
            return
        if control not in actuators:
            raise ValueError(f"{owner} refers to unknown actuator {control}")

    def _validate_edges(self, nodes: set[str], actuators: dict[str, ActuatorSpec]) -> None:
        edge_keys: set[frozenset[str]] = set()
        for edge in self.edges:
            if edge.node_a not in nodes or edge.node_b not in nodes:
                raise ValueError(f"edge {edge.node_a}-{edge.node_b} refers to an unknown node")
            key = frozenset((edge.node_a, edge.node_b))
            if key in edge_keys:
                raise ValueError(f"duplicate undirected edge {edge.node_a}-{edge.node_b}")
            edge_keys.add(key)
            self._validate_law_control(
                edge.conductance,
                actuators,
                f"edge {edge.node_a}-{edge.node_b}",
            )

    def _validate_inputs(self, actuators: dict[str, ActuatorSpec]) -> None:
        for source in self.sources:
            self._validate_law_control(
                source.heat_rate,
                actuators,
                f"source {source.name}",
            )
            if len(source.node_weights) != len(self.node_names):
                raise ValueError(f"source {source.name} must have one weight per node")
        for boundary in self.boundaries:
            reservoir_control = boundary.reservoir_temperature.control
            if reservoir_control is not None and reservoir_control not in actuators:
                raise ValueError(
                    f"boundary {boundary.name} refers to unknown actuator {reservoir_control}"
                )
            self._validate_law_control(
                boundary.conductance,
                actuators,
                f"boundary {boundary.name}",
            )
            if len(boundary.node_weights) != len(self.node_names):
                raise ValueError(f"boundary {boundary.name} must have one weight per node")

    def _validate_sensors(self, nodes: set[str]) -> None:
        if self.sensor_weights:
            weights = np.asarray(self.sensor_weights)
            if weights.shape != (len(self.sensor_names), len(self.node_names)):
                raise ValueError("sensor_weights must have one row per sensor and column per node")
            if not np.isfinite(weights).all() or np.any(weights < 0.0):
                raise ValueError("sensor_weights must be finite and non-negative")
            if not np.allclose(weights.sum(axis=1), 1.0, rtol=1e-9, atol=1e-12):
                raise ValueError("each sensor_weights row must sum to one")
            return
        unknown_sensor_nodes = [name for name in self.sensor_nodes if name not in nodes]
        if unknown_sensor_nodes:
            raise ValueError(f"sensor mapping refers to unknown nodes {unknown_sensor_nodes}")

    @property
    def control_names(self) -> tuple[str, ...]:
        return tuple(actuator.name for actuator in self.actuators)

    @property
    def observation_matrix(self) -> np.ndarray:
        if self.sensor_weights:
            return np.asarray(self.sensor_weights, dtype=np.float64).copy()
        matrix = np.zeros((len(self.sensor_names), len(self.node_names)), dtype=np.float64)
        node_index = {name: i for i, name in enumerate(self.node_names)}
        for sensor_index, node_name in enumerate(self.sensor_nodes):
            matrix[sensor_index, node_index[node_name]] = 1.0
        return matrix
