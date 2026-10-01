from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from celltemp.artifact import load_artifact, save_artifact
from celltemp.inference import (
    forecast,
)
from tests.unit._inference_cases import cooling_model, forecast_request


def test_artifact_round_trip_preserves_forecast(tmp_path: Path) -> None:
    model = cooling_model()
    model.boundary_laws.log_offset_multiplier.data.fill_(0.2)
    save_artifact(tmp_path / "artifact", model, metadata={"purpose": "test"})
    loaded = load_artifact(tmp_path / "artifact")
    request = forecast_request(np.array([0.0, 1.0]), 20.0)
    expected = forecast(model, request).sensor_temperature
    actual = forecast(loaded.model, request).sensor_temperature
    np.testing.assert_allclose(actual, expected)
    assert loaded.metadata["purpose"] == "test"
    fitted_boundary = loaded.metadata["fitted_parameters"]["boundaries"][0]
    assert fitted_boundary["conductance"]["type"] == "constant"
    assert set(loaded.metadata["file_sha256"]) == {"model.pt", "system.yaml"}


def test_artifact_metadata_cannot_override_runtime_format(tmp_path: Path) -> None:
    target = save_artifact(
        tmp_path / "artifact",
        cooling_model(),
        metadata={"schema_version": 999, "model_type": "stale", "dtype": "int32"},
    )
    loaded = load_artifact(target)
    assert loaded.metadata["schema_version"] == 4
    assert loaded.metadata["model_type"] == "thermal_network"
    assert loaded.metadata["dtype"] == "float64"


def test_artifact_rejects_nonfinite_metadata_before_writing(tmp_path: Path) -> None:
    target = tmp_path / "artifact"
    with pytest.raises(ValueError, match="Out of range float values"):
        save_artifact(target, cooling_model(), metadata={"score": float("nan")})
    assert not target.exists()


def test_artifact_rejects_system_and_state_disagreement(tmp_path: Path) -> None:
    target = save_artifact(tmp_path / "artifact", cooling_model())
    system_path = target / "system.yaml"
    system = yaml.safe_load(system_path.read_text(encoding="utf-8"))
    system["nodes"][0]["heat_capacity"] = 9.0
    system_path.write_text(yaml.safe_dump(system, sort_keys=False), encoding="utf-8")
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["file_sha256"]["system.yaml"] = hashlib.sha256(system_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"conflicts with system\.yaml"):
        load_artifact(target)


def test_artifact_rejects_fitted_parameter_metadata_disagreement(tmp_path: Path) -> None:
    target = save_artifact(tmp_path / "artifact", cooling_model())
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["fitted_parameters"]["boundaries"][0]["conductance"]["value"] *= 2.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="fitted_parameters metadata conflicts"):
        load_artifact(target)


def test_artifact_rejects_semantically_unchanged_file_replacement(tmp_path: Path) -> None:
    target = save_artifact(tmp_path / "artifact", cooling_model())
    system_path = target / "system.yaml"
    system_path.write_text(
        system_path.read_text(encoding="utf-8") + "\n# replaced outside the artifact writer\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"checksum mismatch for system\.yaml"):
        load_artifact(target)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 99, "unsupported artifact schema"),
        ("model_type", "unknown_model", "unsupported model type"),
    ],
)
def test_artifact_rejects_unknown_formats(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    target = save_artifact(tmp_path / "artifact", cooling_model())
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_artifact(target)


def test_artifact_rejects_unknown_dtype(tmp_path: Path) -> None:
    target = save_artifact(tmp_path / "artifact", cooling_model())
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dtype"] = "int32"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported artifact dtype"):
        load_artifact(target)
