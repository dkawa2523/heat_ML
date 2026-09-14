"""Self-contained deployment artifact for the thermal network engine."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version() -> str:
    """Return the installed distribution version or a source-tree marker."""
    try:
        return version("thermal-cell-practical")
    except PackageNotFoundError:
        return "source-tree"


def _metadata_equivalent(saved: object, actual: object) -> bool:
    """Compare fitted-parameter metadata with tolerance for device arithmetic."""
    if isinstance(actual, Mapping):
        return (
            isinstance(saved, Mapping)
            and set(saved) == set(actual)
            and all(_metadata_equivalent(saved[key], value) for key, value in actual.items())
        )
    if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes)):
        return (
            isinstance(saved, Sequence)
            and not isinstance(saved, (str, bytes))
            and len(saved) == len(actual)
            and all(
                _metadata_equivalent(saved_value, actual_value)
                for saved_value, actual_value in zip(saved, actual, strict=True)
            )
        )
    if isinstance(actual, float):
        return (
            isinstance(saved, (int, float))
            and not isinstance(saved, bool)
            and math.isclose(float(saved), actual, rel_tol=1e-9, abs_tol=1e-12)
        )
    return saved == actual


def _validate_file_hashes(target: Path, metadata: Mapping[str, Any]) -> None:
    hashes = metadata.get("file_sha256")
    if hashes is None:
        raise ValueError("artifact file_sha256 is required by schema 4")
    if not isinstance(hashes, Mapping):
        raise ValueError("artifact file_sha256 must be a mapping")
    for name in ("model.pt", "system.yaml"):
        expected_hash = hashes.get(name)
        if not isinstance(expected_hash, str) or _sha256(target / name) != expected_hash:
            raise ValueError(f"artifact checksum mismatch for {name}")


def _validate_fitted_parameters(model: ThermalRCModel, metadata: Mapping[str, Any]) -> None:
    if not _metadata_equivalent(metadata.get("fitted_parameters"), fitted_parameters(model)):
        raise ValueError("artifact fitted_parameters metadata conflicts with model.pt")


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
    document: dict[str, Any] = {
        **(metadata or {}),
        "schema_version": 4,
        "model_type": "thermal_network",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package": {
            "distribution": "thermal-cell-practical",
            "version": package_version(),
        },
        "integrator": model.integrator,
        "dtype": str(model.capacity.dtype).removeprefix("torch."),
        "fitted_parameters": fitted_parameters(model),
    }
    # Validate metadata before creating the target, preserving all-or-nothing behavior
    # for values such as NaN that JSON cannot represent.
    json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
    target.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target / "model.pt")
    save_system_spec(model.spec, target / "system.yaml")
    document["file_sha256"] = {name: _sha256(target / name) for name in ("model.pt", "system.yaml")}
    serialized_metadata = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
    (target / "metadata.json").write_text(serialized_metadata, encoding="utf-8")
    return target


def load_artifact(path: str | Path, *, device: str | torch.device = "cpu") -> ThermalArtifact:
    target = Path(path)
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 4:
        raise ValueError(f"unsupported artifact schema {metadata.get('schema_version')}")
    if metadata.get("model_type") != "thermal_network":
        raise ValueError(f"unsupported model type {metadata.get('model_type')}")
    _validate_file_hashes(target, metadata)
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
    for name, expected in model.named_buffers():
        saved = state.get(name)
        if saved is None:
            continue
        comparable = saved.to(dtype=expected.dtype, device=expected.device)
        if comparable.shape != expected.shape or not torch.equal(comparable, expected):
            raise ValueError(
                f"artifact model state conflicts with system.yaml at derived buffer {name!r}"
            )
    model.load_state_dict(state)
    model.eval()
    _validate_fitted_parameters(model, metadata)
    return ThermalArtifact(model=model, metadata=metadata, path=target)
