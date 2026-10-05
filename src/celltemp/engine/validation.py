"""Shared checks at numerical input and output boundaries."""

from __future__ import annotations

import torch


def require_finite(value: torch.Tensor, name: str, *, computed: bool = False) -> None:
    """Reject invalid inputs or an unrepresentable numerical result explicitly."""
    if not bool(torch.isfinite(value).all()):
        if computed:
            raise FloatingPointError(f"{name} became non-finite")
        raise ValueError(f"{name} must be finite")


def require_time_step(step: torch.Tensor) -> None:
    """Zero is valid at command boundaries; backward time is not."""
    if not bool(torch.isfinite(step).all()) or bool(torch.any(step < 0.0)):
        raise ValueError("dt must be non-negative and finite")
