"""Inspect fitted thermal paths and continuous-time thermal modes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from celltemp.domain import Trajectory
from celltemp.engine import ThermalRCModel


def representative_command(trajectories: Sequence[Trajectory]) -> np.ndarray:
    """Select an observed command row nearest the component-wise training median."""
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    control_names = trajectories[0].control_names
    if any(item.control_names != control_names for item in trajectories):
        raise ValueError("all trajectories must use the same controls")
    if not control_names:
        return np.empty(0, dtype=np.float64)
    commands = np.concatenate([item.commands for item in trajectories], axis=0)
    median = np.median(commands, axis=0)
    scale = np.ptp(commands, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    distance = np.sum(((commands - median) / scale) ** 2, axis=1)
    return np.asarray(commands[int(np.argmin(distance))], dtype=np.float64).copy()


def _actuator_tensor(model: ThermalRCModel, actuator: np.ndarray) -> torch.Tensor:
    values = torch.as_tensor(
        np.asarray(actuator),
        dtype=model.capacity.dtype,
        device=model.capacity.device,
    )
    if values.shape != (model.n_controls,) or not bool(torch.isfinite(values).all()):
        raise ValueError("actuator must contain one finite value per control")
    return values


def _base_path_row(operating_point: str) -> dict[str, Any]:
    return {
        "operating_point": operating_point,
        "element_type": None,
        "name": None,
        "node_from": None,
        "node_to": None,
        "distribution_weight": None,
        "heat_capacity_j_per_k": None,
        "conductance_w_per_k": None,
        "resistance_k_per_w": None,
        "heat_rate_w": None,
        "total_heat_rate_w": None,
        "reservoir_temperature_project_scale": None,
        "law_type": None,
        "law_control": None,
        "law_control_value": None,
        "law_control_unit": None,
        "law_offset": None,
        "law_scale": None,
        "law_exponent": None,
        "law_threshold": None,
        "law_reference": None,
    }


def _law_columns(
    model: ThermalRCModel,
    law: dict[str, Any],
    actuator: np.ndarray,
) -> dict[str, Any]:
    control = law.get("control")
    control_index = model.spec.control_names.index(control) if isinstance(control, str) else None
    return {
        "law_type": law["type"],
        "law_control": control,
        "law_control_value": None if control_index is None else float(actuator[control_index]),
        "law_control_unit": (
            None if control_index is None else model.spec.control_units[control_index]
        ),
        "law_offset": law.get("offset", law.get("value")),
        "law_scale": law.get("scale", law.get("gain")),
        "law_exponent": law.get("exponent"),
        "law_threshold": law.get("threshold"),
        "law_reference": law.get("reference"),
    }


@torch.no_grad()
def thermal_path_rows(
    model: ThermalRCModel,
    actuator: np.ndarray,
    *,
    operating_point: str,
) -> list[dict[str, Any]]:
    """Describe fitted C, G/R, source heat, and boundary paths at one input point."""
    effective = _actuator_tensor(model, actuator)
    actuator_values = effective.detach().cpu().numpy()
    conductance = model.conductance(effective).detach().cpu().numpy()
    source_rate = model.source_heat_rate(effective).detach().cpu().numpy()
    boundary_conductance = model.boundary_conductance(effective).detach().cpu().numpy()
    reservoir_temperature = model.boundary_temperature(effective).detach().cpu().numpy()
    edge_laws = model.edge_laws.fitted()
    source_laws = model.source_laws.fitted()
    boundary_laws = model.boundary_laws.fitted()
    rows: list[dict[str, Any]] = []

    for name, capacity in zip(model.spec.node_names, model.capacity.detach().cpu(), strict=True):
        row = _base_path_row(operating_point)
        row.update(
            element_type="node",
            name=name,
            node_from=name,
            heat_capacity_j_per_k=float(capacity),
        )
        rows.append(row)

    for edge, value, law in zip(model.spec.edges, conductance, edge_laws, strict=True):
        row = _base_path_row(operating_point)
        row.update(
            element_type="internal_edge",
            name=f"{edge.node_a}<->{edge.node_b}",
            node_from=edge.node_a,
            node_to=edge.node_b,
            distribution_weight=1.0,
            conductance_w_per_k=float(value),
            resistance_k_per_w=None if value <= 0.0 else float(1.0 / value),
        )
        row.update(_law_columns(model, law, actuator_values))
        rows.append(row)

    for source, value, law in zip(model.spec.sources, source_rate, source_laws, strict=True):
        total = float(value * sum(source.node_weights))
        for node, weight in zip(model.spec.node_names, source.node_weights, strict=True):
            if weight <= 0.0:
                continue
            row = _base_path_row(operating_point)
            row.update(
                element_type="source",
                name=source.name,
                node_to=node,
                distribution_weight=weight,
                heat_rate_w=float(value * weight),
                total_heat_rate_w=total,
            )
            row.update(_law_columns(model, law, actuator_values))
            rows.append(row)

    for index, (boundary, value, law) in enumerate(
        zip(model.spec.boundaries, boundary_conductance, boundary_laws, strict=True)
    ):
        for node, weight in zip(model.spec.node_names, boundary.node_weights, strict=True):
            if weight <= 0.0:
                continue
            node_conductance = float(value * weight)
            row = _base_path_row(operating_point)
            row.update(
                element_type="boundary",
                name=boundary.name,
                node_from=node,
                node_to=f"reservoir:{boundary.name}",
                distribution_weight=weight,
                conductance_w_per_k=node_conductance,
                resistance_k_per_w=(None if node_conductance <= 0.0 else 1.0 / node_conductance),
                reservoir_temperature_project_scale=float(reservoir_temperature[index]),
            )
            row.update(_law_columns(model, law, actuator_values))
            rows.append(row)
    return rows


@torch.no_grad()
def thermal_mode_rows(
    model: ThermalRCModel,
    actuator: np.ndarray,
    *,
    operating_point: str,
) -> list[dict[str, Any]]:
    """Return continuous-time poles and dominant nodes at one operating point."""
    effective = _actuator_tensor(model, actuator)
    matrix = model.system_matrix(effective).detach().cpu().numpy()
    poles, vectors = np.linalg.eig(matrix)
    order = np.argsort(poles.real)[::-1]
    control_values = {
        f"control_{name}": float(effective[index].detach().cpu())
        for index, name in enumerate(model.spec.control_names)
    }
    control_units = {
        f"control_{name}_unit": model.spec.control_units[index]
        for index, name in enumerate(model.spec.control_names)
    }
    rows: list[dict[str, Any]] = []
    tolerance = 1e-12
    for mode_index, eigen_index in enumerate(order, start=1):
        pole = poles[eigen_index]
        magnitude = np.abs(vectors[:, eigen_index])
        dominant_index = int(np.argmax(magnitude))
        total_magnitude = float(magnitude.sum())
        real = float(pole.real)
        if real < -tolerance:
            classification = "decaying"
            time_constant = -1.0 / real
        elif abs(real) <= tolerance and abs(float(pole.imag)) <= tolerance:
            classification = "conserved"
            time_constant = None
        else:
            classification = "unstable_or_complex"
            time_constant = None
        rows.append(
            {
                "operating_point": operating_point,
                "mode": mode_index,
                "classification": classification,
                "pole_real_per_s": real,
                "pole_imag_per_s": float(pole.imag),
                "time_constant_s": time_constant,
                "dominant_node": model.spec.node_names[dominant_index],
                "dominant_node_fraction": (
                    None
                    if total_magnitude == 0.0
                    else float(magnitude[dominant_index] / total_magnitude)
                ),
                **control_values,
                **control_units,
            }
        )
    return rows
