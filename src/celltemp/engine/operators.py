"""Assemble thermal state-space operators from model parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional

from celltemp.domain.topology import ConstantLawSpec, PowerLawSpec, ThermalSystemSpec

if TYPE_CHECKING:
    from .rc import ThermalRCModel


def validate_exact_support(spec: ThermalSystemSpec) -> None:
    """Reject time-varying coefficients outside the exact affine model class."""
    actuator_tau = {item.name: item.tau for item in spec.actuators}
    incompatible: list[str] = []
    for edge in spec.edges:
        law = edge.conductance
        if not isinstance(law, ConstantLawSpec) and actuator_tau[law.control] > 0.0:
            incompatible.append(f"edge {edge.node_a}-{edge.node_b}")
    for boundary in spec.boundaries:
        law = boundary.conductance
        if not isinstance(law, ConstantLawSpec) and actuator_tau[law.control] > 0.0:
            incompatible.append(f"boundary {boundary.name}")
    for source in spec.sources:
        law = source.heat_rate
        if isinstance(law, PowerLawSpec) and actuator_tau[law.control] > 0.0:
            incompatible.append(f"source {source.name}")
    if incompatible:
        paths = ", ".join(incompatible)
        raise ValueError(
            "exact integrator requires zero-tau controls for input-dependent "
            f"conductance and nonlinear source laws: {paths}; use the implicit integrator "
            "for lagged coefficients"
        )


def edge_laplacian_basis(
    spec: ThermalSystemSpec,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return one symmetric graph-Laplacian basis matrix per internal edge."""
    node_index = {name: index for index, name in enumerate(spec.node_names)}
    basis = torch.zeros(
        (len(spec.edges), len(spec.node_names), len(spec.node_names)),
        dtype=dtype,
    )
    for edge_index, edge in enumerate(spec.edges):
        source = node_index[edge.node_a]
        destination = node_index[edge.node_b]
        basis[edge_index, source, source] = 1.0
        basis[edge_index, destination, destination] = 1.0
        basis[edge_index, source, destination] = -1.0
        basis[edge_index, destination, source] = -1.0
    return basis


def boundary_temperature(model: ThermalRCModel, actuator: torch.Tensor) -> torch.Tensor:
    """Evaluate every reservoir temperature for the effective inputs."""
    if actuator.shape[-1] != model.n_controls:
        raise ValueError("actuator has the wrong number of controls")
    selected = torch.zeros(
        (*actuator.shape[:-1], len(model.spec.boundaries)),
        dtype=actuator.dtype,
        device=actuator.device,
    )
    controlled = model.boundary_temperature_control >= 0
    if torch.any(controlled):
        selected[..., controlled] = actuator[..., model.boundary_temperature_control[controlled]]
    return model.boundary_temperature_intercept + model.boundary_temperature_slope * selected


def system_matrix(
    model: ThermalRCModel,
    actuator: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``A`` in ``dT/dt = A T + b`` for effective controls."""
    laplacian = torch.einsum(
        "...e,eij->...ij",
        model.edge_laws._evaluate(actuator),
        model.edge_basis,
    )
    boundary_total = torch.zeros(
        model.n_nodes,
        dtype=model.capacity.dtype,
        device=model.capacity.device,
    )
    if model.spec.boundaries:
        boundary_h = model.boundary_laws._evaluate(actuator)[..., :, None] * model.boundary_weights
        boundary_total = boundary_h.sum(dim=-2)
    return -(laplacian + torch.diag_embed(boundary_total)) / model.capacity[..., None]


def forcing(model: ThermalRCModel, actuator: torch.Tensor) -> torch.Tensor:
    """Return the actuator-dependent ``b`` in ``dT/dt = A T + b``."""
    if actuator.shape[-1] != model.n_controls:
        raise ValueError("actuator has the wrong number of controls")
    heat = torch.zeros(
        (*actuator.shape[:-1], model.n_nodes),
        dtype=actuator.dtype,
        device=actuator.device,
    )
    if model.spec.sources:
        heat = heat + model.source_laws._evaluate(actuator) @ model.source_weights
    if model.spec.boundaries:
        boundary_h = model.boundary_laws._evaluate(actuator)[..., :, None] * model.boundary_weights
        reservoir_temperature = boundary_temperature(model, actuator)
        heat = heat + (reservoir_temperature[..., :, None] * boundary_h).sum(dim=-2)
    return heat / model.capacity


def joint_affine_system(
    model: ThermalRCModel,
    command: torch.Tensor,
    active_sources: torch.Tensor,
    tau: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``dx/dt = F x + c`` for ``x = [temperature, actuator]``."""
    batch_shape = command.shape[:-1]
    n_state = model.n_nodes + model.n_controls
    matrix = torch.zeros(
        (*batch_shape, n_state, n_state),
        dtype=command.dtype,
        device=command.device,
    )
    affine = torch.zeros(
        (*batch_shape, n_state),
        dtype=command.dtype,
        device=command.device,
    )
    matrix[..., : model.n_nodes, : model.n_nodes] = system_matrix(model, command)

    actuator_coefficient = torch.zeros(
        (*batch_shape, model.n_nodes, model.n_controls),
        dtype=command.dtype,
        device=command.device,
    )
    thermal_offset = torch.zeros(
        (*batch_shape, model.n_nodes),
        dtype=command.dtype,
        device=command.device,
    )

    if model.spec.sources:
        positive = model.source_laws.positive_part_mask
        if torch.any(positive):
            source_rate = (
                model.source_laws.scale()[positive, None]
                * model.source_weights[positive]
                / model.capacity[None, :]
            )
            active_rate = active_sources[..., positive, None].to(dtype=command.dtype) * source_rate
            source_controls = functional.one_hot(
                model.source_laws.control_index[positive],
                num_classes=model.n_controls,
            ).to(dtype=command.dtype)
            actuator_coefficient = actuator_coefficient + torch.einsum(
                "...sn,sc->...nc", active_rate, source_controls
            )
            thermal_offset = thermal_offset - (
                active_rate * model.source_laws.threshold[positive, None]
            ).sum(dim=-2)

        direct = ~positive
        if torch.any(direct):
            direct_heat = (
                model.source_laws._evaluate(command)[..., direct] @ model.source_weights[direct]
            )
            thermal_offset = thermal_offset + direct_heat / model.capacity

    if model.spec.boundaries:
        boundary_rate = (
            model.boundary_laws._evaluate(command)[..., :, None]
            * model.boundary_weights
            / model.capacity[None, :]
        )
        thermal_offset = thermal_offset + (
            model.boundary_temperature_intercept[:, None] * boundary_rate
        ).sum(dim=-2)
        controlled = model.boundary_temperature_control >= 0
        if torch.any(controlled):
            boundary_controls = functional.one_hot(
                model.boundary_temperature_control[controlled],
                num_classes=model.n_controls,
            ).to(dtype=command.dtype)
            controlled_rate = (
                model.boundary_temperature_slope[controlled, None]
                * boundary_rate[..., controlled, :]
            )
            actuator_coefficient = actuator_coefficient + torch.einsum(
                "...bn,bc->...nc", controlled_rate, boundary_controls
            )

    matrix[..., : model.n_nodes, model.n_nodes :] = actuator_coefficient
    affine[..., : model.n_nodes] = thermal_offset

    if tau is None:
        tau = model.actuator_tau().to(dtype=command.dtype, device=command.device)
    inverse_tau = torch.where(tau > 0.0, torch.reciprocal(tau), torch.zeros_like(tau))
    matrix[..., model.n_nodes :, model.n_nodes :] = torch.diag(-inverse_tau)
    affine[..., model.n_nodes :] = command * inverse_tau
    return matrix, affine


def source_activity(model: ThermalRCModel, actuator: torch.Tensor) -> torch.Tensor:
    """Return active positive-part source states for one or more inputs."""
    if not model.spec.sources:
        return torch.empty(
            (*actuator.shape[:-1], 0),
            dtype=torch.bool,
            device=actuator.device,
        )
    activity = torch.zeros(
        (*actuator.shape[:-1], len(model.spec.sources)),
        dtype=torch.bool,
        device=actuator.device,
    )
    positive = model.source_laws.positive_part_mask
    if model.source_laws._has_positive:
        activity[..., positive] = (
            actuator[..., model.source_laws.control_index[positive]]
            > model.source_laws.threshold[positive]
        )
    return activity


def operator_context(model: ThermalRCModel, actuator: torch.Tensor) -> tuple[float, ...]:
    """Return inputs whose instantaneous values change the thermal operator."""
    controls = [
        laws.control_index[laws.dependent_mask]
        for laws in (model.edge_laws, model.boundary_laws)
        if laws._has_dependent
    ]
    if not controls:
        return ()
    return tuple(
        float(actuator[int(index)].detach().cpu().item())
        for index in torch.unique(torch.cat(controls)).tolist()
    )
