"""YAML serialization for :class:`~celltemp.domain.ThermalSystemSpec`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from celltemp.domain import (
    ActuatorSpec,
    BoundarySpec,
    EdgeSpec,
    SourceSpec,
    ThermalSystemSpec,
)


def _node_weights(
    value: Mapping[str, float] | Sequence[float], node_names: tuple[str, ...]
) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        unknown = set(value) - set(node_names)
        if unknown:
            raise ValueError(f"node_weights refers to unknown nodes {sorted(unknown)}")
        return tuple(float(value.get(name, 0.0)) for name in node_names)
    return tuple(float(item) for item in value)


def system_spec_from_mapping(data: Mapping[str, Any]) -> ThermalSystemSpec:
    """Build a validated system definition from a compact mapping.

    Node weights may be a dense list or a mapping keyed by node name.  The latter
    keeps hand-written system files readable and independent of node ordering.
    """
    version = int(data.get("version", 1))
    if version != 1:
        raise ValueError(f"unsupported system schema version {version}")

    nodes = tuple(data.get("nodes", ()))
    if not nodes:
        raise ValueError("system definition requires a non-empty 'nodes' list")
    node_names = tuple(str(node["name"]) for node in nodes)
    capacity = tuple(float(node["heat_capacity"]) for node in nodes)

    actuators = tuple(
        ActuatorSpec(
            name=str(item["name"]),
            tau=float(item.get("tau", 0.0)),
            learnable=bool(item.get("learnable", True)),
        )
        for item in data.get("actuators", ())
    )
    edges = tuple(
        EdgeSpec(
            node_a=str(item["nodes"][0]),
            node_b=str(item["nodes"][1]),
            conductance=float(item["conductance"]),
            learnable=bool(item.get("learnable", True)),
        )
        for item in data.get("edges", ())
    )
    sources = tuple(
        SourceSpec(
            name=str(item["name"]),
            actuator=str(item["actuator"]),
            node_weights=_node_weights(item["node_weights"], node_names),
            gain=float(item["gain"]),
            threshold=float(item.get("threshold", 0.0)),
            learnable=bool(item.get("learnable", True)),
        )
        for item in data.get("sources", ())
    )
    boundaries = tuple(
        BoundarySpec(
            name=str(item["name"]),
            node_weights=_node_weights(item["node_weights"], node_names),
            conductance=float(item["conductance"]),
            temperature_intercept=float(item["temperature_intercept"]),
            actuator=(None if item.get("actuator") is None else str(item["actuator"])),
            temperature_slope=float(item.get("temperature_slope", 0.0)),
            learnable=bool(item.get("learnable", True)),
        )
        for item in data.get("boundaries", ())
    )

    sensor_items = tuple(data.get("sensors", ()))
    if sensor_items:
        sensor_names = tuple(
            str(item["name"]) if isinstance(item, Mapping) else str(item) for item in sensor_items
        )
        sensor_nodes = tuple(
            str(item.get("node", item["name"])) if isinstance(item, Mapping) else str(item)
            for item in sensor_items
        )
    else:
        sensor_names = ()
        sensor_nodes = ()

    return ThermalSystemSpec(
        node_names=node_names,
        heat_capacity=capacity,
        edges=edges,
        actuators=actuators,
        sources=sources,
        boundaries=boundaries,
        sensor_names=sensor_names,
        sensor_nodes=sensor_nodes,
    )


def system_spec_to_mapping(spec: ThermalSystemSpec) -> dict[str, Any]:
    """Convert a system definition to stable, human-readable YAML data."""

    def weights(values: tuple[float, ...]) -> dict[str, float]:
        return {name: float(value) for name, value in zip(spec.node_names, values) if value != 0.0}

    return {
        "version": 1,
        "nodes": [
            {"name": name, "heat_capacity": float(capacity)}
            for name, capacity in zip(spec.node_names, spec.heat_capacity)
        ],
        "actuators": [
            {"name": item.name, "tau": item.tau, "learnable": item.learnable}
            for item in spec.actuators
        ],
        "edges": [
            {
                "nodes": [item.node_a, item.node_b],
                "conductance": item.conductance,
                "learnable": item.learnable,
            }
            for item in spec.edges
        ],
        "sources": [
            {
                "name": item.name,
                "actuator": item.actuator,
                "node_weights": weights(item.node_weights),
                "gain": item.gain,
                "threshold": item.threshold,
                "learnable": item.learnable,
            }
            for item in spec.sources
        ],
        "boundaries": [
            {
                "name": item.name,
                "node_weights": weights(item.node_weights),
                "conductance": item.conductance,
                "temperature_intercept": item.temperature_intercept,
                "actuator": item.actuator,
                "temperature_slope": item.temperature_slope,
                "learnable": item.learnable,
            }
            for item in spec.boundaries
        ],
        "sensors": [
            {"name": name, "node": node} for name, node in zip(spec.sensor_names, spec.sensor_nodes)
        ],
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
