"""Stable differentiable integration of affine thermal state equations."""

from __future__ import annotations

import math

import torch

from .validation import require_finite, require_time_step


def _validated_step(dt: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    step = torch.as_tensor(dt, dtype=reference.dtype, device=reference.device)
    require_time_step(step)
    return step


def _expanded_matrix(matrix: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
    if matrix.ndim == 2:
        return matrix.expand(*batch_shape, *matrix.shape)
    if matrix.shape[:-2] != batch_shape:
        return matrix.expand(*batch_shape, *matrix.shape[-2:])
    return matrix


def exact_affine_step(
    state: torch.Tensor,
    system_matrix: torch.Tensor,
    forcing: torch.Tensor,
    dt: float | torch.Tensor,
) -> torch.Tensor:
    """Exact zero-order-hold step for ``dx/dt = A x + b``.

    An augmented matrix exponential avoids explicitly inverting ``A`` and remains
    valid when the thermal system has a zero eigenvalue, as a closed conductive
    network does.
    """
    if state.shape != forcing.shape:
        raise ValueError("state and forcing must have the same shape")
    require_finite(state, "state")
    require_finite(system_matrix, "system_matrix")
    require_finite(forcing, "forcing")
    n_state = state.shape[-1]
    batch_shape = state.shape[:-1]
    matrix = _expanded_matrix(system_matrix, batch_shape)
    if matrix.shape[-2:] != (n_state, n_state):
        raise ValueError("system_matrix has incompatible dimensions")

    top = torch.cat([matrix, forcing.unsqueeze(-1)], dim=-1)
    bottom = torch.zeros((*batch_shape, 1, n_state + 1), dtype=state.dtype, device=state.device)
    augmented = torch.cat([top, bottom], dim=-2)
    step = _validated_step(dt, state)
    step = step.expand(batch_shape) if batch_shape else step.reshape(())
    transition = torch.matrix_exp(augmented * step[..., None, None])
    homogeneous = torch.cat([state, torch.ones_like(state[..., :1])], dim=-1)
    result = (transition @ homogeneous.unsqueeze(-1))[..., :n_state, 0]
    require_finite(result, "exact state", computed=True)
    return result


def exact_affine_operators(
    system_matrix: torch.Tensor, dt: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``Phi, Gamma`` for ``x_next = Phi x + Gamma b``.

    Unlike :func:`exact_affine_step`, this separates the forcing from the matrix
    exponential.  A thermal model whose coefficients are constant over a trajectory
    can compute these operators once per distinct timestep and reuse them for every
    command interval.
    """
    if system_matrix.ndim != 2 or system_matrix.shape[0] != system_matrix.shape[1]:
        raise ValueError("system_matrix must be one square matrix")
    require_finite(system_matrix, "system_matrix")
    n_state = system_matrix.shape[0]
    identity = torch.eye(n_state, dtype=system_matrix.dtype, device=system_matrix.device)
    zero = torch.zeros_like(system_matrix)
    augmented = torch.cat(
        [
            torch.cat([system_matrix, identity], dim=1),
            torch.cat([zero, zero], dim=1),
        ],
        dim=0,
    )
    step = _validated_step(dt, system_matrix)
    if step.ndim != 0:
        raise ValueError("operator dt must be scalar")
    transition = torch.matrix_exp(augmented * step)
    require_finite(transition, "exact operators", computed=True)
    return transition[:n_state, :n_state], transition[:n_state, n_state:]


def exact_process_operators(
    system_matrix: torch.Tensor,
    spectral_density: torch.Tensor,
    dt: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discretize a linear SDE without a growing long-interval Van Loan block.

    Van Loan's auxiliary ``-A.T`` dynamics can overflow even for a stable ``A``.
    Compute its exponential only over a short interval, then compose the stable
    physical transition and covariance by repeated doubling. The composition is
    exact and retains autograd through the matrices and timestep.
    """
    if system_matrix.ndim != 2 or system_matrix.shape[0] != system_matrix.shape[1]:
        raise ValueError("system_matrix must be one square matrix")
    if spectral_density.shape != system_matrix.shape:
        raise ValueError("spectral_density must have the system_matrix shape")
    require_finite(system_matrix, "system_matrix")
    require_finite(spectral_density, "spectral_density")
    step = _validated_step(dt, system_matrix)
    if step.ndim != 0:
        raise ValueError("process operator dt must be scalar")

    # The maximum entry bounds ||A||_inf after multiplication by the dimension.
    # Taking logarithms separately also avoids overflow in ||A|| * dt itself.
    magnitude = float(system_matrix.detach().abs().max().cpu().item())
    step_value = float(step.detach().cpu().item())
    doublings = 0
    if magnitude > 0.0 and step_value > 0.0:
        log_norm = math.log2(magnitude) + math.log2(system_matrix.shape[0])
        doublings = max(0, math.ceil(log_norm + math.log2(step_value)))
    short_step = torch.ldexp(step, torch.tensor(-doublings, device=step.device))
    zero = torch.zeros_like(system_matrix)
    van_loan = torch.cat(
        [
            torch.cat([system_matrix, spectral_density], dim=1),
            torch.cat([zero, -system_matrix.T], dim=1),
        ],
        dim=0,
    )
    exponential = torch.matrix_exp(van_loan * short_step)
    n_state = system_matrix.shape[0]
    transition = exponential[:n_state, :n_state]
    covariance = exponential[:n_state, n_state:] @ transition.T
    covariance = (covariance + covariance.T) * 0.5
    for _ in range(doublings):
        covariance = covariance + transition @ covariance @ transition.T
        transition = transition @ transition
        covariance = (covariance + covariance.T) * 0.5
    require_finite(transition, "process transition", computed=True)
    require_finite(covariance, "process covariance", computed=True)
    return transition, covariance


def implicit_euler_step(
    state: torch.Tensor,
    system_matrix: torch.Tensor,
    forcing: torch.Tensor,
    dt: float | torch.Tensor,
) -> torch.Tensor:
    """A-stable fallback for ``dx/dt = A x + b``."""
    if state.shape != forcing.shape:
        raise ValueError("state and forcing must have the same shape")
    require_finite(state, "state")
    require_finite(system_matrix, "system_matrix")
    require_finite(forcing, "forcing")
    n_state = state.shape[-1]
    batch_shape = state.shape[:-1]
    matrix = _expanded_matrix(system_matrix, batch_shape)
    if matrix.shape[-2:] != (n_state, n_state):
        raise ValueError("system_matrix has incompatible dimensions")
    step = _validated_step(dt, state)
    step = step.expand(batch_shape) if batch_shape else step.reshape(())
    identity = torch.eye(n_state, dtype=state.dtype, device=state.device)
    identity = identity.expand(*batch_shape, n_state, n_state)
    lhs = identity - step[..., None, None] * matrix
    rhs = state + step[..., None] * forcing
    result = torch.linalg.solve(lhs, rhs.unsqueeze(-1)).squeeze(-1)
    require_finite(result, "implicit state", computed=True)
    return result
