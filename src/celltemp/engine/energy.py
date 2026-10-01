"""Instantaneous, signed heat-flow accounting for the thermal network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .rc import ThermalRCModel


@dataclass(frozen=True)
class HeatFlowBreakdown:
    """Heat rates in W at one or more thermal states.

    ``edge_rate[..., e]`` is positive from ``edge.node_a`` to ``edge.node_b``.
    Source and boundary tensors are positive into each node. Boundary values are
    therefore negative during cooling. ``storage_rate`` is ``C*dT/dt`` and the
    balance residual is incoming heat minus storage for each node.
    """

    edge_rate: torch.Tensor
    internal_to_node: torch.Tensor
    source_to_node: torch.Tensor
    boundary_to_node: torch.Tensor
    boundary_temperature: torch.Tensor
    storage_rate: torch.Tensor
    balance_residual: torch.Tensor


def heat_flow_breakdown(
    model: ThermalRCModel,
    temperature: torch.Tensor,
    actuator: torch.Tensor,
) -> HeatFlowBreakdown:
    """Evaluate signed heat paths independently from the state-space RHS."""
    temperature = temperature.to(dtype=model.capacity.dtype, device=model.capacity.device)
    actuator = actuator.to(dtype=model.capacity.dtype, device=model.capacity.device)
    if temperature.shape[-1] != model.n_nodes:
        raise ValueError("temperature has the wrong number of nodes")
    if actuator.shape[-1] != model.n_controls:
        raise ValueError("actuator has the wrong number of controls")
    if temperature.shape[:-1] != actuator.shape[:-1]:
        raise ValueError("temperature and actuator batch shapes must match")

    batch_shape = temperature.shape[:-1]
    node_index = {name: index for index, name in enumerate(model.spec.node_names)}
    if model.spec.edges:
        node_a = torch.tensor(
            [node_index[item.node_a] for item in model.spec.edges],
            dtype=torch.long,
            device=temperature.device,
        )
        node_b = torch.tensor(
            [node_index[item.node_b] for item in model.spec.edges],
            dtype=torch.long,
            device=temperature.device,
        )
        edge_rate = model.conductance(actuator) * (
            temperature[..., node_a] - temperature[..., node_b]
        )
        incidence = temperature.new_zeros((len(model.spec.edges), model.n_nodes))
        edge_indices = torch.arange(len(model.spec.edges), device=temperature.device)
        incidence[edge_indices, node_a] = -1.0
        incidence[edge_indices, node_b] = 1.0
        internal_to_node = edge_rate @ incidence
    else:
        edge_rate = temperature.new_empty((*batch_shape, 0))
        internal_to_node = temperature.new_zeros((*batch_shape, model.n_nodes))

    if model.spec.sources:
        source_to_node = model.source_heat_rate(actuator)[..., :, None] * model.source_weights
    else:
        source_to_node = temperature.new_empty((*batch_shape, 0, model.n_nodes))

    boundary_temperature = model.boundary_temperature(actuator)
    if model.spec.boundaries:
        boundary_to_node = (
            model.boundary_conductance(actuator)[..., :, None]
            * model.boundary_weights
            * (boundary_temperature[..., :, None] - temperature[..., None, :])
        )
    else:
        boundary_to_node = temperature.new_empty((*batch_shape, 0, model.n_nodes))

    derivative = torch.einsum(
        "...ij,...j->...i", model.system_matrix(actuator), temperature
    ) + model.forcing(actuator)
    storage_rate = derivative * model.capacity
    incoming = internal_to_node + source_to_node.sum(dim=-2) + boundary_to_node.sum(dim=-2)
    return HeatFlowBreakdown(
        edge_rate=edge_rate,
        internal_to_node=internal_to_node,
        source_to_node=source_to_node,
        boundary_to_node=boundary_to_node,
        boundary_temperature=boundary_temperature,
        storage_rate=storage_rate,
        balance_residual=incoming - storage_rate,
    )
