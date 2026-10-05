"""State carried between thermal transitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ThermalState:
    """Physical node temperatures and effective actuator values."""

    temperature: torch.Tensor
    actuator: torch.Tensor

    @classmethod
    def _from_validated(cls, temperature: torch.Tensor, actuator: torch.Tensor) -> ThermalState:
        """Carry a checked integration result without checking the same values twice."""
        state = object.__new__(cls)
        object.__setattr__(state, "temperature", temperature)
        object.__setattr__(state, "actuator", actuator)
        return state

    def __post_init__(self) -> None:
        if self.temperature.ndim < 1 or self.actuator.ndim < 1:
            raise ValueError("temperature and actuator tensors need a feature dimension")
        if self.temperature.shape[:-1] != self.actuator.shape[:-1]:
            raise ValueError("temperature and actuator batch shapes must match")
        if not torch.isfinite(self.temperature).all() or not torch.isfinite(self.actuator).all():
            raise ValueError("thermal state must be finite")
