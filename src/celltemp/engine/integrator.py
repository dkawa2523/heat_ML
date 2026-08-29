"""Stable differentiable integration of affine thermal state equations."""

from __future__ import annotations

import torch


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
    n_state = state.shape[-1]
    batch_shape = state.shape[:-1]
    matrix = _expanded_matrix(system_matrix, batch_shape)
    if matrix.shape[-2:] != (n_state, n_state):
        raise ValueError("system_matrix has incompatible dimensions")

    top = torch.cat([matrix, forcing.unsqueeze(-1)], dim=-1)
    bottom = torch.zeros((*batch_shape, 1, n_state + 1), dtype=state.dtype, device=state.device)
    augmented = torch.cat([top, bottom], dim=-2)
    step = torch.as_tensor(dt, dtype=state.dtype, device=state.device)
    step = step.expand(batch_shape) if batch_shape else step.reshape(())
    transition = torch.matrix_exp(augmented * step[..., None, None])
    homogeneous = torch.cat([state, torch.ones_like(state[..., :1])], dim=-1)
    return (transition @ homogeneous.unsqueeze(-1))[..., :n_state, 0]


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
    step = torch.as_tensor(dt, dtype=system_matrix.dtype, device=system_matrix.device)
    transition = torch.matrix_exp(augmented * step)
    return transition[:n_state, :n_state], transition[:n_state, n_state:]


def implicit_euler_step(
    state: torch.Tensor,
    system_matrix: torch.Tensor,
    forcing: torch.Tensor,
    dt: float | torch.Tensor,
) -> torch.Tensor:
    """A-stable fallback for ``dx/dt = A x + b``."""
    if state.shape != forcing.shape:
        raise ValueError("state and forcing must have the same shape")
    n_state = state.shape[-1]
    batch_shape = state.shape[:-1]
    matrix = _expanded_matrix(system_matrix, batch_shape)
    if matrix.shape[-2:] != (n_state, n_state):
        raise ValueError("system_matrix has incompatible dimensions")
    step = torch.as_tensor(dt, dtype=state.dtype, device=state.device)
    step = step.expand(batch_shape) if batch_shape else step.reshape(())
    identity = torch.eye(n_state, dtype=state.dtype, device=state.device)
    identity = identity.expand(*batch_shape, n_state, n_state)
    lhs = identity - step[..., None, None] * matrix
    rhs = state + step[..., None] * forcing
    return torch.linalg.solve(lhs, rhs.unsqueeze(-1)).squeeze(-1)
