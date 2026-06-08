from __future__ import annotations

import torch
from torch import nn

from .models_base import BaseCellTempModel, ModelSpec


class LinearRC(BaseCellTempModel):
    """Linear baseline: ΔT = W[T_cur, u_next] + b."""

    family = "sequence"

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.linear = nn.Linear(spec.n_sensors + spec.n_controls, spec.n_sensors)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        t = batch["state_hist"][:, -1, :]
        u = batch["control_next"]
        return self.linear(torch.cat([t, u], dim=-1))


class MLPUpdate(BaseCellTempModel):
    """Small MLP using flattened history."""

    family = "sequence"

    def __init__(self, spec: ModelSpec):
        super().__init__()
        in_dim = spec.history * (spec.n_sensors + spec.n_controls) + spec.n_controls
        self.net = nn.Sequential(
            nn.Linear(in_dim, spec.hidden_dim),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_dim, spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.hidden_dim, spec.n_sensors),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([batch["state_hist"], batch["control_hist"]], dim=-1).flatten(1)
        x = torch.cat([x, batch["control_next"]], dim=-1)
        return self.net(x)


class GRUUpdate(BaseCellTempModel):
    """History-dependent update using GRU."""

    family = "sequence"

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.gru = nn.GRU(spec.n_sensors + spec.n_controls, spec.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(spec.hidden_dim + spec.n_controls, spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.hidden_dim, spec.n_sensors),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([batch["state_hist"], batch["control_hist"]], dim=-1)
        _, h = self.gru(x)
        z = h[-1]
        return self.head(torch.cat([z, batch["control_next"]], dim=-1))


class LSTMUpdate(BaseCellTempModel):
    """Optional recurrent baseline. Prefer GRU/TCN unless LSTM gives clear rollout gains."""

    family = "sequence"

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.lstm = nn.LSTM(spec.n_sensors + spec.n_controls, spec.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(spec.hidden_dim + spec.n_controls, spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.hidden_dim, spec.n_sensors),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([batch["state_hist"], batch["control_hist"]], dim=-1)
        _, (h, _) = self.lstm(x)
        z = h[-1]
        return self.head(torch.cat([z, batch["control_next"]], dim=-1))


class CNN1DUpdate(BaseCellTempModel):
    """Compact 1D-CNN baseline for fixed-length temperature/control histories."""

    family = "sequence"

    def __init__(self, spec: ModelSpec):
        super().__init__()
        in_channels = spec.n_sensors + spec.n_controls
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, spec.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Conv1d(spec.hidden_dim, spec.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(spec.hidden_dim + spec.n_controls, spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.hidden_dim, spec.n_sensors),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([batch["state_hist"], batch["control_hist"]], dim=-1)
        z = self.conv(x.transpose(1, 2)).mean(dim=-1)
        return self.head(torch.cat([z, batch["control_next"]], dim=-1))


class TCNUpdate(BaseCellTempModel):
    """Simple dilated temporal convolution model."""

    family = "sequence"

    def __init__(self, spec: ModelSpec):
        super().__init__()
        c = spec.n_sensors + spec.n_controls
        layers = []
        for dilation in [1, 2, 4]:
            layers.extend(
                [
                    nn.Conv1d(c, spec.hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.ReLU(),
                    nn.Dropout(spec.dropout),
                ]
            )
            c = spec.hidden_dim
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(spec.hidden_dim + spec.n_controls, spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.hidden_dim, spec.n_sensors),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([batch["state_hist"], batch["control_hist"]], dim=-1)
        z = self.conv(x.transpose(1, 2))[:, :, -1]
        return self.head(torch.cat([z, batch["control_next"]], dim=-1))
