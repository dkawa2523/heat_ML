from __future__ import annotations

from pathlib import Path

from .models_base import BaseCellTempModel, ModelSpec
from .models_physics import GraphRC, ThermalStateSpace
from .models_sequence import CNN1DUpdate, GRUUpdate, LSTMUpdate, LinearRC, MLPUpdate, TCNUpdate
from .thermal_graph import load_thermal_graph


# Keep the registry explicit and short so model choices are easy to audit.
MODELS = {
    "linear_rc": LinearRC,
    "mlp": MLPUpdate,
    "cnn1d": CNN1DUpdate,
    "gru": GRUUpdate,
    "lstm": LSTMUpdate,
    "tcn": TCNUpdate,
    "thermal_state_space": ThermalStateSpace,
    "graph_rc": GraphRC,
}

MODEL_FAMILIES = {
    "sequence": ["linear_rc", "mlp", "cnn1d", "gru", "lstm", "tcn"],
    "physics": ["thermal_state_space", "graph_rc"],
}


def build_model(
    cfg: dict,
    n_sensors: int,
    n_controls: int,
    history: int,
    sensor_cols: list[str] | None = None,
    control_cols: list[str] | None = None,
    root: str | Path | None = None,
) -> BaseCellTempModel:
    name = cfg.get("name", "thermal_state_space")
    if name not in MODELS:
        raise ValueError(f"unknown model.name={name}. choices={list(MODELS)}")

    sensor_names = sensor_cols or [f"sensor_{i}" for i in range(n_sensors)]
    control_names = control_cols or [f"control_{i}" for i in range(n_controls)]
    graph_cfg = cfg.get("graph", {}) or {}
    graph = load_thermal_graph(graph_cfg, Path(root or "."), sensor_names, control_names) if name == "graph_rc" else None
    spec = ModelSpec(
        n_sensors=n_sensors,
        n_controls=n_controls,
        history=history,
        sensor_names=sensor_names,
        control_names=control_names,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        dropout=float(cfg.get("dropout", 0.05)),
        residual_scale=float(cfg.get("residual_scale", 0.1)),
        graph=graph,
        graph_cfg=graph_cfg,
    )
    return MODELS[name](spec)
