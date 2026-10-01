"""YAML conversion for thermal-system sensor observation layouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from celltemp.domain import ThermalSystemSpec


def _node_weights(
    value: Mapping[str, float] | Sequence[float], node_names: tuple[str, ...]
) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        unknown = set(value) - set(node_names)
        if unknown:
            raise ValueError(f"node_weights refers to unknown nodes {sorted(unknown)}")
        return tuple(float(value.get(name, 0.0)) for name in node_names)
    return tuple(float(item) for item in value)


def _sensor_item(
    value: object,
    node_names: tuple[str, ...],
) -> tuple[str, str | None, tuple[float, ...] | None]:
    if not isinstance(value, Mapping):
        name = str(value)
        return name, name, None

    unknown = set(value) - {"name", "node", "node_weights"}
    if unknown:
        raise ValueError(f"unknown sensor options: {sorted(map(str, unknown))}")
    name = str(value["name"])
    if "node" in value and "node_weights" in value:
        raise ValueError("sensor must use either node or node_weights")
    if "node_weights" in value:
        return name, None, _node_weights(value["node_weights"], node_names)
    return name, str(value.get("node", name)), None


def sensor_layout_from_mapping(
    values: object,
    node_names: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[float, ...], ...]]:
    """Read compact node or weighted sensor mappings from YAML data."""

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
        row[node_index[node]] = 1.0
        dense_weights.append(tuple(row))
    return tuple(names), (), tuple(dense_weights)


def sensor_layout_to_mapping(spec: ThermalSystemSpec) -> list[dict[str, Any]]:
    """Write a validated sensor layout as stable, readable YAML data."""

    def weights(values: tuple[float, ...]) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(spec.node_names, values, strict=True)
            if value != 0.0
        }

    if spec.sensor_weights:
        return [
            {"name": name, "node_weights": weights(row)}
            for name, row in zip(spec.sensor_names, spec.sensor_weights, strict=True)
        ]
    return [
        {"name": name, "node": node}
        for name, node in zip(spec.sensor_names, spec.sensor_nodes, strict=True)
    ]
