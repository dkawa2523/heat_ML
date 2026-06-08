from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .models_base import BaseCellTempModel, ModelSpec
from .thermal_graph import default_graph


class ThermalStateSpace(BaseCellTempModel):
    """Thermal state-space model.

    T_next = T_eq(u) + a(u) * (T_cur - T_eq(u)) + residual(T_cur, u)
    """

    family = "physics"

    def __init__(self, spec: ModelSpec):
        super().__init__()
        cfg = spec.graph_cfg or {}
        self.residual_scale = spec.residual_scale
        self.decay_min = float(cfg.get("decay_min", 0.0))
        self.decay_max = float(cfg.get("decay_max", 0.995))
        self.eq = nn.Sequential(
            nn.Linear(spec.n_controls, spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.hidden_dim, spec.n_sensors),
        )
        self.decay_raw = nn.Sequential(
            nn.Linear(spec.n_controls, spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.hidden_dim, spec.n_sensors),
            nn.Sigmoid(),
        )
        self.res = nn.Sequential(
            nn.Linear(spec.n_sensors + spec.n_controls, spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.hidden_dim, spec.n_sensors),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        t = batch["state_hist"][:, -1, :]
        u = batch["control_next"]
        t_eq = self.eq(u)
        a01 = self.decay_raw(u)
        a = self.decay_min + (self.decay_max - self.decay_min) * a01
        residual = self.residual_scale * self.res(torch.cat([t, u], dim=-1))
        return t_eq + a * (t - t_eq) + residual - t


class GraphRC(BaseCellTempModel):
    """Graph-RC using known Cell topology and thermal resistance priors."""

    family = "physics"

    def __init__(self, spec: ModelSpec):
        super().__init__()
        graph = spec.graph or default_graph(spec.sensor_names, spec.control_names)
        cfg = spec.graph_cfg or {}
        self.sensor_names = spec.sensor_names
        self.control_names = spec.control_names
        self.residual_scale = spec.residual_scale
        self.conduction_scale = float(cfg.get("conduction_scale", 0.05))
        self.source_prior_scale = float(cfg.get("source_prior_scale", 0.05))
        self.use_source_prior = bool(cfg.get("use_source_prior", True))

        self.register_buffer("edge_mask", torch.tensor(graph.edge_mask, dtype=torch.float32))
        self.register_buffer("conductance_prior", torch.tensor(graph.conductance_prior, dtype=torch.float32))
        self.register_buffer("thermal_mass", torch.tensor(graph.thermal_mass, dtype=torch.float32).clamp_min(1e-6))
        self.register_buffer("source_weight", torch.tensor(graph.source_weight, dtype=torch.float32))

        log_g = torch.log(self.conductance_prior.clamp_min(1e-6))
        if bool(cfg.get("learn_conductance", True)):
            self.log_g = nn.Parameter(log_g)
        else:
            self.register_buffer("log_g", log_g)

        self.source_nn = nn.Sequential(
            nn.Linear(spec.n_controls, spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.hidden_dim, spec.n_sensors),
        )
        self.res = nn.Sequential(
            nn.Linear(spec.n_sensors + spec.n_controls, spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.hidden_dim, spec.n_sensors),
        )

    def conductance_matrix(self) -> np.ndarray:
        with torch.no_grad():
            g = torch.exp(self.log_g.detach().cpu()) * self.edge_mask.detach().cpu()
        return g.numpy()

    def graph_prior_loss(self) -> torch.Tensor:
        """Softly keep learned conductance near the thermal-resistance prior."""
        mask = self.edge_mask > 0
        if not torch.any(mask):
            return torch.tensor(0.0, device=self.edge_mask.device)
        g = torch.exp(self.log_g)
        prior = self.conductance_prior.to(g.device)
        return ((torch.log(g[mask].clamp_min(1e-8)) - torch.log(prior[mask].clamp_min(1e-8))) ** 2).mean()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        t = batch["state_hist"][:, -1, :]
        u = batch["control_next"]
        g = torch.exp(self.log_g).to(t.device) * self.edge_mask.to(t.device)
        mass = self.thermal_mass.to(t.device)

        # diff[b,i,j] = T_j - T_i
        diff = t[:, None, :] - t[:, :, None]
        conduction = (g[None, :, :] * diff).sum(dim=2) / mass[None, :]
        conduction = self.conduction_scale * conduction

        if self.use_source_prior:
            source_prior = u @ self.source_weight.to(t.device).T
            source_prior = self.source_prior_scale * source_prior / mass[None, :]
        else:
            source_prior = torch.zeros_like(t)
        source = source_prior + self.source_nn(u)
        residual = self.residual_scale * self.res(torch.cat([t, u], dim=-1))
        return conduction + source + residual
