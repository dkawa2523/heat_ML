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
    if not isinstance(value, Mapping):
        raise ValueError("boundary reservoir_temperature must be a mapping")
    control = value.get("control")
    return ReservoirTemperatureSpec(
        intercept=float(value["intercept"]),
        control=None if control is None else str(control),
        slope=float(value.get("slope", 0.0)),
    )


def _scalar_law(value: object, owner: str) -> ScalarLawSpec:
    if not isinstance(value, Mapping):
        raise ValueError(f"{owner} must be a scalar-law mapping")
    law_type = str(value.get("type", ""))
    if law_type == "constant":
        return ConstantLawSpec(
            value=float(value["value"]),
            learnable=bool(value.get("learnable", True)),
        )
    if law_type == "positive_part":
        return PositivePartLawSpec(
            control=str(value["control"]),
            gain=float(value["gain"]),
            threshold=float(value.get("threshold", 0.0)),
            learnable=bool(value.get("learnable", True)),
        )
    if law_type == "power_law":
        return PowerLawSpec(
            control=str(value["control"]),
            reference=float(value["reference"]),
            offset=float(value["offset"]),
            scale=float(value["scale"]),
            exponent=float(value["exponent"]),
            offset_learnable=bool(value.get("offset_learnable", False)),
            scale_learnable=bool(value.get("scale_learnable", True)),
            exponent_learnable=bool(value.get("exponent_learnable", False)),
        )
    raise ValueError(f"unsupported scalar law type {law_type!r} for {owner}")


def system_spec_from_mapping(data: Mapping[str, Any]) -> ThermalSystemSpec:
    """Build a validated system definition from a compact mapping.

    Node weights may be a dense list or a mapping keyed by node name.  The latter
    keeps hand-written system files readable and independent of node ordering.
    """
    version = int(data.get("version", 3))
    if version != 3:
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
            conductance=_scalar_law(item["conductance"], "edge conductance"),
        )
        for item in data.get("edges", ())
    )
    sources = tuple(
        SourceSpec(
            name=str(item["name"]),
            node_weights=_node_weights(item["node_weights"], node_names),
            heat_rate=_scalar_law(item["heat_rate"], "source heat_rate"),
        )
        for item in data.get("sources", ())
    )
    boundaries = tuple(
        BoundarySpec(
            name=str(item["name"]),
            node_weights=_node_weights(item["node_weights"], node_names),
            reservoir_temperature=_reservoir_temperature(item["reservoir_temperature"]),
            conductance=_scalar_law(item["conductance"], "boundary conductance"),
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
            {"name": item.name, "tau": item.tau, "learnable": item.learnable}
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
