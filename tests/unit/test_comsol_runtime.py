import subprocess
from pathlib import Path

import pytest

from external_tools import comsol_runtime
from external_tools.comsol_chip_cooling import comsol_runtime as legacy_runtime
from external_tools.comsol_chip_cooling.comsol_runtime import provenance_path


def test_provenance_path_supports_external_data_root(tmp_path: Path) -> None:
    tool_root = tmp_path / "tool"
    inside = tool_root / "data" / "case.csv"
    outside = tmp_path / "published" / "case.csv"

    assert provenance_path(inside, tool_root) == "data/case.csv"
    assert provenance_path(outside, tool_root) == outside.resolve().as_posix()


def test_legacy_runtime_exports_are_the_shared_public_api() -> None:
    assert legacy_runtime.__all__ == [
        "APPLICATION_MODEL",
        "ComsolRuntime",
        "provenance_path",
        "select_comsol",
    ]
    for name in legacy_runtime.__all__:
        assert getattr(legacy_runtime, name) is getattr(comsol_runtime, name)


@pytest.fixture
def runtime_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "comsol"
    binary = root / "bin" / "win64"
    binary.mkdir(parents=True)
    for name in ("comsolbatch.exe", "comsolcompile.exe"):
        (binary / name).write_bytes(b"mock executable")
    monkeypatch.delenv("COMSOL_ROOT", raising=False)
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path / "no-installations"))
    return root


def _license_response(
    monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int = 0
) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout, "")

    monkeypatch.setattr(comsol_runtime.subprocess, "run", run)
    return calls


def test_supplied_model_uses_its_license_without_a_library_or_heat_transfer_token(
    runtime_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "custom.mph"
    model.write_bytes(b"mock model")
    calls = _license_response(monkeypatch, "Model license available: COMSOL")
    runtime = comsol_runtime.select_comsol(str(runtime_root), source_model=model)

    assert not (runtime_root / comsol_runtime.APPLICATION_MODEL).exists()
    assert runtime.source_model == model.resolve()
    assert calls == [[str(runtime.batch), "-checklicense", str(model.resolve())]]


@pytest.mark.parametrize(
    ("stdout", "returncode"), [("Model license unavailable", 1), ("Error checking model", 0)]
)
def test_supplied_model_still_requires_a_successful_license_check(
    stdout: str,
    returncode: int,
    runtime_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "custom.mph"
    model.write_bytes(b"mock model")
    calls = _license_response(monkeypatch, stdout, returncode)
    with pytest.raises(RuntimeError, match="No usable COMSOL"):
        comsol_runtime.select_comsol(str(runtime_root), source_model=model)
    assert calls[0][1:] == ["-checklicense", str(model.resolve())]


@pytest.mark.parametrize(
    ("stdout", "accepted"),
    [("HEATTRANSFER license available", True), ("COMSOL license available", False)],
)
def test_default_library_retains_its_heat_transfer_license_requirement(
    stdout: str, accepted: bool, runtime_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = runtime_root / comsol_runtime.APPLICATION_MODEL
    model.parent.mkdir(parents=True)
    model.write_bytes(b"mock library model")
    calls = _license_response(monkeypatch, stdout)
    if accepted:
        runtime = comsol_runtime.select_comsol(str(runtime_root))
        assert runtime.source_model == model
    else:
        with pytest.raises(RuntimeError, match="No usable COMSOL"):
            comsol_runtime.select_comsol(str(runtime_root))
    batch = runtime_root / "bin" / "win64" / "comsolbatch.exe"
    assert calls == [[str(batch), "-checklicense", str(model)]]


def test_default_library_requires_the_library_file(
    runtime_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _license_response(monkeypatch, "HEATTRANSFER license available")
    with pytest.raises(RuntimeError, match="no complete COMSOL installation found"):
        comsol_runtime.select_comsol(str(runtime_root))
    assert not calls
