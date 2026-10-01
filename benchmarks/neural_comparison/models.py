"""Small sequence-model architectures used only by the comparison benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SequenceModelSpec:
    """Dimensions shared by every neural comparison model."""

    n_sensors: int
    n_controls: int
    history_steps: int
    hidden_dim: int = 32
    dropout: float = 0.05

    @property
    def history_features(self) -> int:
        """Temperature, command, and log time-step channels at each past row."""
        return self.n_sensors + self.n_controls + 1

    @property
    def query_features(self) -> int:
        """Known command and log time step for the interval being predicted."""
        return self.n_controls + 1


class SequenceUpdateModel(nn.Module):
    """Return a standardized sensor-temperature rate for one future interval."""

    def forward(self, history: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MLPUpdate(SequenceUpdateModel):
    """Flatten a fixed history window, then apply a two-layer perceptron."""

    def __init__(self, spec: SequenceModelSpec):
        super().__init__()
        input_dim = spec.history_steps * spec.history_features + spec.query_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, spec.hidden_dim),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_dim, spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.hidden_dim, spec.n_sensors),
        )

    def forward(self, history: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([history.flatten(1), query], dim=-1))


class CNN1DUpdate(SequenceUpdateModel):
    """Extract local patterns from a fixed-length multichannel history."""

    def __init__(self, spec: SequenceModelSpec):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(spec.history_features, spec.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Conv1d(spec.hidden_dim, spec.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = _prediction_head(spec)

    def forward(self, history: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(history.transpose(1, 2)).squeeze(-1)
        return self.head(torch.cat([encoded, query], dim=-1))


class _CausalConv(nn.Module):
    """A left-padded convolution that never reads beyond the history endpoint."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class TCNUpdate(SequenceUpdateModel):
    """Use dilated causal convolutions for short and long history patterns."""

    def __init__(self, spec: SequenceModelSpec):
        super().__init__()
        layers: list[nn.Module] = []
        channels = spec.history_features
        for dilation in (1, 2, 4):
            layers.extend(
                [
                    _CausalConv(channels, spec.hidden_dim, dilation),
                    nn.ReLU(),
                    nn.Dropout(spec.dropout),
                ]
            )
            channels = spec.hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.head = _prediction_head(spec)

    def forward(self, history: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(history.transpose(1, 2))[:, :, -1]
        return self.head(torch.cat([encoded, query], dim=-1))


class GRUUpdate(SequenceUpdateModel):
    """Compress the causal history with a gated recurrent unit."""

    def __init__(self, spec: SequenceModelSpec):
        super().__init__()
        self.encoder = nn.GRU(spec.history_features, spec.hidden_dim, batch_first=True)
        self.head = _prediction_head(spec)

    def forward(self, history: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(history)
        return self.head(torch.cat([hidden[-1], query], dim=-1))


class LSTMUpdate(SequenceUpdateModel):
    """Compress the causal history with separate hidden and memory states."""

    def __init__(self, spec: SequenceModelSpec):
        super().__init__()
        self.encoder = nn.LSTM(spec.history_features, spec.hidden_dim, batch_first=True)
        self.head = _prediction_head(spec)

    def forward(self, history: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(history)
        return self.head(torch.cat([hidden[-1], query], dim=-1))


def _prediction_head(spec: SequenceModelSpec) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(spec.hidden_dim + spec.query_features, spec.hidden_dim),
        nn.ReLU(),
        nn.Linear(spec.hidden_dim, spec.n_sensors),
    )


MODEL_TYPES: dict[str, type[SequenceUpdateModel]] = {
    "mlp": MLPUpdate,
    "cnn1d": CNN1DUpdate,
    "tcn": TCNUpdate,
    "gru": GRUUpdate,
    "lstm": LSTMUpdate,
}


def build_model(name: str, spec: SequenceModelSpec) -> SequenceUpdateModel:
    """Build one explicitly registered comparison model."""
    try:
        model_type = MODEL_TYPES[name]
    except KeyError as error:
        raise ValueError(f"unknown neural model {name!r}; choices={tuple(MODEL_TYPES)}") from error
    return model_type(spec)
