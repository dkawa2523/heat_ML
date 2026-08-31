"""COMSOL discovery and Java-runner execution shared by dataset workflows."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

APPLICATION_MODEL = Path(
    "applications/Heat_Transfer_Module/Tutorials,_Forced_and_Natural_Convection/chip_cooling.mph"
)


@dataclass(frozen=True)
class ComsolRuntime:
    """Resolved local COMSOL executables and the immutable library source model."""

    root: Path
    batch: Path
    compiler: Path
    source_model: Path

    def compile(self, java_source: Path) -> Path:
        """Compile one Java model runner and return its class file."""
        class_file = java_source.with_suffix(".class")
        result = subprocess.run(  # noqa: S603 - resolved trusted local executable
            [str(self.compiler), str(java_source)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = f"{result.stdout}\n{result.stderr}"
        failed = (
            result.returncode != 0
            or not class_file.is_file()
            or "Failed to compile" in output
            or "Compilation error" in output
        )
        if failed:
            raise RuntimeError(f"COMSOL Java compilation failed:\n{output}")
        return class_file

    def run_java(
        self,
        java_class: Path,
        arguments: Sequence[str | Path],
        *,
        log_path: Path,
        expected_output: Path,
        label: str,
    ) -> str:
        """Run one compiled model task, persist its log, and require its output."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.batch),
            "-inputfile",
            str(java_class),
            *(str(value) for value in arguments),
            "-nosave",
        ]
        result = subprocess.run(  # noqa: S603 - resolved trusted local executable
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = f"{result.stdout}\n{result.stderr}"
        log_path.write_text(output, encoding="utf-8")
        if (
            result.returncode != 0
            or "Error running java class" in output
            or not expected_output.is_file()
        ):
            tail = "\n".join(output.splitlines()[-100:])
            raise RuntimeError(f"COMSOL {label} failed. See {log_path}.\n{tail}")
        return output


def _candidate_roots(explicit: str | None) -> list[Path]:
    candidates: list[Path] = []
    for value in (explicit, os.getenv("COMSOL_ROOT")):
        if value:
            candidates.append(Path(value))
    program_files = Path(os.getenv("PROGRAMFILES", "C:/Program Files"))
    install_parent = program_files / "COMSOL" / "COMSOL64"
    candidates.extend([install_parent / "Multiphysics_copy1", install_parent / "Multiphysics"])
    if install_parent.is_dir():
        candidates.extend(sorted(install_parent.glob("Multiphysics*"), reverse=True))
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def select_comsol(explicit: str | None) -> ComsolRuntime:
    """Select an installation that can check out the required Heat Transfer license."""
    failures: list[str] = []
    for root in _candidate_roots(explicit):
        binary = root / "bin" / "win64"
        runtime = ComsolRuntime(
            root=root,
            batch=binary / "comsolbatch.exe",
            compiler=binary / "comsolcompile.exe",
            source_model=root / APPLICATION_MODEL,
        )
        if not (
            runtime.batch.is_file()
            and runtime.compiler.is_file()
            and runtime.source_model.is_file()
        ):
            continue
        result = subprocess.run(  # noqa: S603 - resolved trusted local executable
            [str(runtime.batch), "-checklicense", str(runtime.source_model)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = f"{result.stdout}\n{result.stderr}"
        if result.returncode == 0 and "HEATTRANSFER" in output and "Error" not in output:
            return runtime
        failures.append(f"{root}: {output.strip()[-300:]}")
    detail = "\n".join(failures) if failures else "no complete COMSOL installation found"
    raise RuntimeError(f"No usable COMSOL Heat Transfer license was found.\n{detail}")
