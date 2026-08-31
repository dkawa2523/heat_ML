"""Self-contained deployment artifact for the thermal network engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from celltemp.engine import ThermalRCModel
from celltemp.io import load_system_spec, save_system_spec


@dataclass(frozen=True)
class ThermalArtifact:
    model: ThermalRCModel
    metadata: dict[str, Any]
    path: Path

    @property
    def sensor_names(self) -> tuple[str, ...]:
        return self.model.spec.sensor_names

    @property
    def control_names(self) -> tuple[str, ...]:
        return self.model.spec.control_names


def fitted_parameters(model: ThermalRCModel) -> dict[str, list[dict[str, Any]]]:
    """Return physical fitted values with their engineering names."""
    tau = model.actuator_tau().detach().cpu().tolist()
    return {
        "edges": [
            {
                "node_a": item.node_a,
                "node_b": item.node_b,
                "conductance": law,
            }
            for item, law in zip(model.spec.edges, model.edge_laws.fitted(), strict=True)
        ],
        "actuators": [
            {"name": item.name, "tau": float(value)}
            for item, value in zip(model.spec.actuators, tau)
        ],
        "sources": [
            {"name": item.name, "heat_rate": law}
            for item, law in zip(model.spec.sources, model.source_laws.fitted(), strict=True)
        ],
        "boundaries": [
            {"name": item.name, "conductance": law}
            for item, law in zip(
                model.spec.boundaries,
                model.boundary_laws.fitted(),
                strict=True,
            )
        ],
    }


def save_artifact(
    path: str | Path,
    model: ThermalRCModel,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write only the state, system definition, and provenance needed at runtime."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target / "model.pt")
    save_system_spec(model.spec, target / "system.yaml")
    document = {
        **(metadata or {}),
        "schema_version": 4,
        "model_type": "thermal_network",
        "integrator": model.integrator,
        "dtype": str(model.capacity.dtype).removeprefix("torch."),
        "fitted_parameters": fitted_parameters(model),
    }
    (target / "metadata.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target


def load_artifact(path: str | Path, *, device: str | torch.device = "cpu") -> ThermalArtifact:
    target = Path(path)
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 4:
        raise ValueError(f"unsupported artifact schema {metadata.get('schema_version')}")
    if metadata.get("model_type") != "thermal_network":
        raise ValueError(f"unsupported model type {metadata.get('model_type')}")
    dtypes = {"float32": torch.float32, "float64": torch.float64}
    dtype_name = str(metadata.get("dtype"))
    if dtype_name not in dtypes:
        raise ValueError(f"unsupported artifact dtype {dtype_name}")
    dtype = dtypes[dtype_name]
    model = ThermalRCModel(
        load_system_spec(target / "system.yaml"),
        integrator=str(metadata.get("integrator", "exact")),
        dtype=dtype,
    ).to(device)
    state = torch.load(target / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return ThermalArtifact(model=model, metadata=metadata, path=target)
