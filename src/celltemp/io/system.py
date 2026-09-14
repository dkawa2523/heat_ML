"""YAML serialization for :class:`~celltemp.domain.ThermalSystemSpec`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from celltemp.domain import (
    ActuatorSpec,
    BoundarySpec,
    ConstantLawSpec,
    EdgeSpec,
    PositivePartLawSpec,
    PowerLawSpec,
    ReservoirTemperatureSpec,
    ScalarLawSpec,
    SourceSpec,
    ThermalSystemSpec,
)


def _mapping(value: object, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{owner} must be a mapping")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], owner: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {owner} options: {sorted(map(str, unknown))}")


def _boolean(value: object, owner: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{owner} must be boolean")
    return value


def _node_weights(
    value: Mapping[str, float] | Sequence[float], node_names: tuple[str, ...]
) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        unknown = set(value) - set(node_names)
        if unknown:
            raise ValueError(f"node_weights refers to unknown nodes {sorted(unknown)}")
        return tuple(float(value.get(name, 0.0)) for name in node_names)
    return tuple(float(item) for item in value)


def _reservoir_temperature(value: object) -> ReservoirTemperatureSpec:
    values = _mapping(value, "boundary reservoir_temperature")
    _reject_unknown(values, {"control", "intercept", "slope"}, "reservoir_temperature")
    control = values.get("control")
    return ReservoirTemperatureSpec(
        intercept=float(values["intercept"]),
        control=None if control is None else str(control),
        slope=float(values.get("slope", 0.0)),
    )


def _scalar_law(value: object, owner: str) -> ScalarLawSpec:
    values = _mapping(value, owner)
    law_type = str(values.get("type", ""))
    if law_type == "constant":
        _reject_unknown(values, {"type", "value", "learnable"}, owner)
        return ConstantLawSpec(
            value=float(values["value"]),
            learnable=_boolean(values.get("learnable", True), f"{owner}.learnable"),
        )
    if law_type == "positive_part":
        _reject_unknown(
            values,
            {"type", "control", "gain", "threshold", "learnable"},
            owner,
        )
        return PositivePartLawSpec(
            control=str(values["control"]),
            gain=float(values["gain"]),
            threshold=float(values.get("threshold", 0.0)),
            learnable=_boolean(values.get("learnable", True), f"{owner}.learnable"),
        )
    if law_type == "power_law":
        _reject_unknown(
            values,
            {
                "type",
                "control",
                "reference",
                "offset",
                "scale",
                "exponent",
                "offset_learnable",
                "scale_learnable",
                "exponent_learnable",
            },
            owner,
        )
        return PowerLawSpec(
            control=str(values["control"]),
            reference=float(values["reference"]),
            offset=float(values["offset"]),
            scale=float(values["scale"]),
            exponent=float(values["exponent"]),
            offset_learnable=_boolean(
                values.get("offset_learnable", False), f"{owner}.offset_learnable"
            ),
            scale_learnable=_boolean(
                values.get("scale_learnable", True), f"{owner}.scale_learnable"
            ),
            exponent_learnable=_boolean(
                values.get("exponent_learnable", False), f"{owner}.exponent_learnable"
            ),
        )
    raise ValueError(f"unsupported scalar law type {law_type!r} for {owner}")


def _actuator(value: object) -> ActuatorSpec:
    values = _mapping(value, "actuator")
    _reject_unknown(values, {"name", "tau", "learnable"}, "actuator")
    learnable = values.get("learnable")
    return ActuatorSpec(
        name=str(values["name"]),
        tau=float(values.get("tau", 0.0)),
        learnable=(None if learnable is None else _boolean(learnable, "actuator.learnable")),
    )


def _edge(value: object) -> EdgeSpec:
    values = _mapping(value, "edge")
    _reject_unknown(values, {"nodes", "conductance"}, "edge")
    endpoints = values["nodes"]
    if isinstance(endpoints, (str, bytes)) or not isinstance(endpoints, Sequence):
        raise ValueError("edge.nodes must be a two-item sequence")
    if len(endpoints) != 2:
        raise ValueError("edge.nodes must contain exactly two nodes")
    return EdgeSpec(
        node_a=str(endpoints[0]),
        node_b=str(endpoints[1]),
        conductance=_scalar_law(values["conductance"], "edge conductance"),
    )


def _source(value: object, node_names: tuple[str, ...]) -> SourceSpec:
    values = _mapping(value, "source")
    _reject_unknown(values, {"name", "node_weights", "heat_rate"}, "source")
    return SourceSpec(
        name=str(values["name"]),
        node_weights=_node_weights(values["node_weights"], node_names),
        heat_rate=_scalar_law(values["heat_rate"], "source heat_rate"),
    )


def _boundary(value: object, node_names: tuple[str, ...]) -> BoundarySpec:
    values = _mapping(value, "boundary")
    _reject_unknown(
        values,
        {"name", "node_weights", "reservoir_temperature", "conductance"},
        "boundary",
    )
    return BoundarySpec(
        name=str(values["name"]),
        node_weights=_node_weights(values["node_weights"], node_names),
        reservoir_temperature=_reservoir_temperature(values["reservoir_temperature"]),
        conductance=_scalar_law(values["conductance"], "boundary conductance"),
    )


def _sensor_item(
    value: object,
    node_names: tuple[str, ...],
) -> tuple[str, str | None, tuple[float, ...] | None]:
    if not isinstance(value, Mapping):
        name = str(value)
        return name, name, None
    _reject_unknown(value, {"name", "node", "node_weights"}, "sensor")
    name = str(value["name"])
    if "node" in value and "node_weights" in value:
        raise ValueError("sensor must use either node or node_weights")
    if "node_weights" in value:
        return name, None, _node_weights(value["node_weights"], node_names)
    return name, str(value.get("node", name)), None


def _sensor_layout(
    values: object,
    node_names: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[float, ...], ...]]:
    if not values:
        return (), (), ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("sensors must be a sequence")

    names: list[str] = []
    nodes: list[str | None] = []
    weights: list[tuple[float, ...] | None] = []
    for value in values:
        name, node, weight = _sensor_item(value, node_names)
        names.append(name)
        nodes.append(node)
        weights.append(weight)
    if not any(weight is not None for weight in weights):
        return tuple(names), tuple(str(node) for node in nodes), ()

    node_index = {name: index for index, name in enumerate(node_names)}
    dense_weights: list[tuple[float, ...]] = []
    for node, weight in zip(nodes, weights, strict=True):
        if weight is not None:
            dense_weights.append(weight)
            continue
        row = [0.0] * len(node_names)
        if node not in node_index:
            raise ValueError(f"sensor mapping refers to unknown node {node}")
        row[node_index[str(node)]] = 1.0
        dense_weights.append(tuple(row))
    return tuple(names), (), tuple(dense_weights)


def system_spec_from_mapping(data: Mapping[str, Any]) -> ThermalSystemSpec:
    """Build a validated system definition from a compact mapping.

    Node weights may be a dense list or a mapping keyed by node name.  The latter
    keeps hand-written system files readable and independent of node ordering.
    """
    _reject_unknown(
        data,
        {"version", "nodes", "actuators", "edges", "sources", "boundaries", "sensors"},
        "system",
    )
    version = int(data.get("version", 3))
    if version != 3:
        raise ValueError(f"unsupported system schema version {version}")

    nodes = tuple(_mapping(item, "node") for item in data.get("nodes", ()))
    if not nodes:
        raise ValueError("system definition requires a non-empty 'nodes' list")
    for node in nodes:
        _reject_unknown(node, {"name", "heat_capacity"}, "node")
    node_names = tuple(str(node["name"]) for node in nodes)
    capacity = tuple(float(node["heat_capacity"]) for node in nodes)

    actuators = tuple(_actuator(item) for item in data.get("actuators", ()))
    edges = tuple(_edge(item) for item in data.get("edges", ()))
    sources = tuple(_source(item, node_names) for item in data.get("sources", ()))
    boundaries = tuple(_boundary(item, node_names) for item in data.get("boundaries", ()))
    sensor_names, sensor_nodes, sensor_weights = _sensor_layout(data.get("sensors", ()), node_names)

    return ThermalSystemSpec(
        node_names=node_names,
        heat_capacity=capacity,
        edges=edges,
        actuators=actuators,
        sources=sources,
        boundaries=boundaries,
        sensor_names=sensor_names,
        sensor_nodes=sensor_nodes,
        sensor_weights=sensor_weights,
    )


def system_spec_to_mapping(spec: ThermalSystemSpec) -> dict[str, Any]:
    """Convert a system definition to stable, human-readable YAML data."""

    def weights(values: tuple[float, ...]) -> dict[str, float]:
        return {name: float(value) for name, value in zip(spec.node_names, values) if value != 0.0}

    def law(item: ScalarLawSpec) -> dict[str, Any]:
        if isinstance(item, ConstantLawSpec):
            return {
                "type": "constant",
                "value": item.value,
                "learnable": item.learnable,
            }
        if isinstance(item, PositivePartLawSpec):
            return {
                "type": "positive_part",
                "control": item.control,
                "gain": item.gain,
                "threshold": item.threshold,
                "learnable": item.learnable,
            }
        return {
            "type": "power_law",
            "control": item.control,
            "reference": item.reference,
            "offset": item.offset,
            "scale": item.scale,
            "exponent": item.exponent,
            "offset_learnable": item.offset_learnable,
            "scale_learnable": item.scale_learnable,
            "exponent_learnable": item.exponent_learnable,
        }

    return {
        "version": 3,
        "nodes": [
            {"name": name, "heat_capacity": float(capacity)}
            for name, capacity in zip(spec.node_names, spec.heat_capacity)
        ],
        "actuators": [
            {"name": item.name, "tau": item.tau, "learnable": bool(item.learnable)}
            for item in spec.actuators
        ],
        "edges": [
            {
                "nodes": [item.node_a, item.node_b],
                "conductance": law(item.conductance),
            }
            for item in spec.edges
        ],
        "sources": [
            {
                "name": item.name,
                "node_weights": weights(item.node_weights),
                "heat_rate": law(item.heat_rate),
            }
            for item in spec.sources
        ],
        "boundaries": [
            {
                "name": item.name,
                "node_weights": weights(item.node_weights),
                "reservoir_temperature": {
                    "intercept": item.reservoir_temperature.intercept,
                    "control": item.reservoir_temperature.control,
                    "slope": item.reservoir_temperature.slope,
                },
                "conductance": law(item.conductance),
            }
            for item in spec.boundaries
        ],
        "sensors": (
            [
                {"name": name, "node_weights": weights(row)}
                for name, row in zip(spec.sensor_names, spec.sensor_weights, strict=True)
            ]
            if spec.sensor_weights
            else [
                {"name": name, "node": node}
                for name, node in zip(spec.sensor_names, spec.sensor_nodes, strict=True)
            ]
        ),
    }


def load_system_spec(path: str | Path) -> ThermalSystemSpec:
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, Mapping):
        raise ValueError("system YAML must contain a mapping at its root")
    return system_spec_from_mapping(data)


def save_system_spec(spec: ThermalSystemSpec, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            system_spec_to_mapping(spec),
            stream,
            sort_keys=False,
            allow_unicode=True,
        )
