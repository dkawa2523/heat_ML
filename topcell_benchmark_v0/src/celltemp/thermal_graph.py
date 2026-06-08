from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import as_path


@dataclass
class ThermalGraph:
    sensor_names: list[str]
    control_names: list[str]
    edge_mask: np.ndarray          # [N, N], 1 if heat path j -> i exists
    conductance_prior: np.ndarray  # [N, N], relative G_ij prior
    thermal_mass: np.ndarray       # [N]
    source_weight: np.ndarray      # [N, C]


def default_graph(sensor_cols: list[str], control_cols: list[str]) -> ThermalGraph:
    n = len(sensor_cols)
    mask = np.ones((n, n), dtype=float) - np.eye(n, dtype=float)
    return ThermalGraph(
        sensor_names=sensor_cols,
        control_names=control_cols,
        edge_mask=mask,
        conductance_prior=mask.copy(),
        thermal_mass=np.ones(n, dtype=float),
        source_weight=np.zeros((n, len(control_cols)), dtype=float),
    )


def _check_positive_finite(values: np.ndarray, name: str, path: Path) -> None:
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{name} must be positive finite values: {path}")


def load_thermal_graph(
    graph_cfg: dict | None,
    root: Path,
    sensor_cols: list[str],
    control_cols: list[str],
) -> ThermalGraph:
    """Load known Cell thermal topology for Graph-RC.

    node_table columns:
        sensor, thermal_mass, <control>_weight

    edge_table columns:
        src, dst, r_th_K_per_W or conductance

    Values are relative priors after standardization, not strict SI-unit
    simulation constants.
    """
    graph_cfg = graph_cfg or {}
    if not graph_cfg.get("node_table") and not graph_cfg.get("edge_table"):
        return default_graph(sensor_cols, control_cols)

    n = len(sensor_cols)
    c = len(control_cols)
    index = {s: i for i, s in enumerate(sensor_cols)}
    mass = np.ones(n, dtype=float)
    source_weight = np.zeros((n, c), dtype=float)

    if graph_cfg.get("node_table"):
        node_path = as_path(graph_cfg["node_table"], root)
        if not node_path.exists():
            raise FileNotFoundError(f"node_table not found: {node_path}")
        nodes = pd.read_csv(node_path)
        if "sensor" not in nodes.columns:
            raise ValueError(f"node_table must include sensor: {node_path}")
        if nodes["sensor"].duplicated().any():
            dup = nodes.loc[nodes["sensor"].duplicated(), "sensor"].tolist()
            raise ValueError(f"duplicated sensors in node_table {node_path}: {dup}")
        nodes = nodes.set_index("sensor", drop=False)
        for s, i in index.items():
            if s not in nodes.index:
                raise ValueError(f"sensor '{s}' missing in node_table: {node_path}")
            row = nodes.loc[s]
            if "thermal_mass" in nodes.columns:
                mass[i] = float(row["thermal_mass"])
            for j, ctrl in enumerate(control_cols):
                col = f"{ctrl}_weight"
                if col in nodes.columns:
                    source_weight[i, j] = float(row[col])
        _check_positive_finite(mass, "thermal_mass", node_path)
        if graph_cfg.get("use_source_prior", True) and not np.any(np.abs(source_weight) > 0):
            raise ValueError(f"source weights are all zero in node_table: {node_path}")

    signs = graph_cfg.get("control_signs", {}) or {}
    for j, ctrl in enumerate(control_cols):
        source_weight[:, j] *= float(signs.get(ctrl, 1.0))

    mask = np.zeros((n, n), dtype=float)
    g_prior = np.zeros((n, n), dtype=float)
    if graph_cfg.get("edge_table"):
        edge_path = as_path(graph_cfg["edge_table"], root)
        if not edge_path.exists():
            raise FileNotFoundError(f"edge_table not found: {edge_path}")
        edges = pd.read_csv(edge_path)
        if not {"src", "dst"}.issubset(edges.columns):
            raise ValueError(f"edge_table must include src,dst: {edge_path}")
        if edges[["src", "dst"]].duplicated().any():
            dup = edges.loc[edges[["src", "dst"]].duplicated(), ["src", "dst"]].values.tolist()
            raise ValueError(f"duplicated edges in edge_table {edge_path}: {dup}")
        for _, row in edges.iterrows():
            src = str(row["src"])
            dst = str(row["dst"])
            if src not in index or dst not in index:
                if bool(graph_cfg.get("ignore_unknown_edges", False)):
                    continue
                raise ValueError(
                    f"unknown edge {src}->{dst}; sensors={sensor_cols}. "
                    "For 3-point subgraphs, set model.graph.ignore_unknown_edges=true."
                )
            j = index[src]
            i = index[dst]
            if bool(graph_cfg.get("init_from_resistance", True)) and "r_th_K_per_W" in edges.columns:
                r = float(row["r_th_K_per_W"])
                if not np.isfinite(r) or r <= 0:
                    raise ValueError(f"r_th_K_per_W must be positive for edge {src}->{dst}: {edge_path}")
                g = 1.0 / r
            elif "conductance" in edges.columns:
                g = float(row["conductance"])
                if not np.isfinite(g) or g <= 0:
                    raise ValueError(f"conductance must be positive for edge {src}->{dst}: {edge_path}")
            else:
                g = 1.0
            mask[i, j] = 1.0
            g_prior[i, j] = g
            if bool(graph_cfg.get("symmetric", True)):
                mask[j, i] = 1.0
                g_prior[j, i] = g
    else:
        mask = np.ones((n, n), dtype=float) - np.eye(n, dtype=float)
        g_prior = mask.copy()

    if n > 1:
        degree = mask.sum(axis=0) + mask.sum(axis=1)
        isolated = [sensor_cols[i] for i, d in enumerate(degree) if d <= 0]
        if isolated:
            raise ValueError(f"isolated sensor(s) in graph: {isolated}")

    g_prior *= mask
    max_g = float(g_prior.max()) if g_prior.size else 0.0
    if max_g > 0:
        g_prior = g_prior / max_g
    g_prior = np.where(mask > 0, np.maximum(g_prior, 1e-6), 0.0)

    return ThermalGraph(sensor_cols, control_cols, mask, g_prior, mass, source_weight)
