from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .thermal_graph import ThermalGraph


@dataclass
class ModelSpec:
    """Minimal model construction context shared by all model families."""

    n_sensors: int
    n_controls: int
    history: int
    sensor_names: list[str]
    control_names: list[str]
    hidden_dim: int = 64
    dropout: float = 0.05
    residual_scale: float = 0.1
    graph: ThermalGraph | None = None
    graph_cfg: dict | None = None


class BaseCellTempModel(nn.Module):
    """All models return standardized ΔT with shape [B, N]."""

    family: str = "base"

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
