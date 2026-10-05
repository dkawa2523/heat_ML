#!/usr/bin/env python
"""Small local quality gate.

    python quality.py fast          format, lint, types, unit/property tests
    python quality.py architecture  Ruff, Pyrefly, import boundaries, complexity
    python quality.py pr            full tests, coverage, and architecture boundaries

The exit code is the verdict. Dependency auditing stays as a separate CI step so
this script only checks the repository itself.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parent
PATHS = [
    "src/celltemp",
    "tests",
    "quality.py",
    "benchmarks",
    "external_tools/comsol_chip_cooling",
]


def _console_script(name: str) -> str:
    scripts = Path(sysconfig.get_path("scripts"))
    for candidate in (scripts / name, scripts / f"{name}.exe"):
        if candidate.exists():
            return str(candidate)
    return shutil.which(name) or name


def run(tool: str, *args: str, timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    command = (
        [_console_script(tool), *args]
        if tool == "lint-imports"
        else [sys.executable, "-m", tool, *args]
    )
    source_path = str(REPO / "src")
    python_path = os.environ.get("PYTHONPATH")
    environment = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "RADONFILESENCODING": "utf-8",
        "PYTHONPATH": source_path if not python_path else source_path + os.pathsep + python_path,
    }
    try:
        return subprocess.run(
            command,
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=environment,
            check=False,
        )
    except OSError as error:
        return subprocess.CompletedProcess(command, 127, "", f"cannot run {tool}: {error}")


def report(name: str, result: subprocess.CompletedProcess[str]) -> bool:
    passed = result.returncode == 0
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    if not passed:
        detail = (result.stdout + result.stderr).strip().splitlines()[-40:]
        encoding = sys.stdout.encoding or "utf-8"
        printable = "\n".join(f"        {line}" for line in detail)
        print(printable.encode(encoding, errors="replace").decode(encoding))
    return passed


def check_format() -> bool:
    return report("format", run("ruff", "format", "--check", "--no-cache", *PATHS))


def check_lint() -> bool:
    return report(
        "lint",
        run("ruff", "check", "--no-cache", "--output-format", "concise", *PATHS),
    )


def check_types() -> bool:
    return report(
        "types",
        run("pyrefly", "check", "--python-interpreter-path", sys.executable, "--summarize-errors"),
    )


def check_architecture() -> bool:
    return report("architecture", run("lint-imports"))


def check_complexity() -> bool:
    """Reject only high-risk core complexity; report C-rank hotspots in design reviews."""
    results = [
        run("radon", "cc", "--show-complexity", "--min", "D", "src/celltemp"),
        run("radon", "mi", "--show", "--min", "C", "src/celltemp"),
    ]
    failed = any(result.returncode != 0 or result.stdout.strip() for result in results)
    combined = subprocess.CompletedProcess(
        args=["radon", "cc/mi", "src/celltemp"],
        returncode=1 if failed else 0,
        stdout="\n".join(result.stdout.strip() for result in results if result.stdout.strip()),
        stderr="\n".join(result.stderr.strip() for result in results if result.stderr.strip()),
    )
    return report("complexity", combined)


def check_tests(*, coverage: bool) -> bool:
    arguments = ["-q"]
    name = "tests + coverage" if coverage else "unit/property tests"
    if coverage:
        arguments += ["--cov=celltemp", "--cov-branch", "--cov-report=term:skip-covered"]
    else:
        arguments += ["-m", "not integration"]
    return report(name, run("pytest", *arguments, timeout=5400))


def run_checks(checks: list[Callable[[], bool]]) -> int:
    results = [check() for check in checks]
    return 0 if all(results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fast", "architecture", "pr"))
    command = parser.parse_args(argv).command
    if command == "architecture":
        checks = [check_lint, check_types, check_architecture, check_complexity]
    else:
        checks = [check_format, check_lint, check_types]
    if command == "pr":
        checks += [check_architecture, check_complexity, lambda: check_tests(coverage=True)]
    elif command == "fast":
        checks += [lambda: check_tests(coverage=False)]

    print(f"=== quality {command} ===\n")
    exit_code = run_checks(checks)
    print(f"\n=== quality {command}: {'PASS' if exit_code == 0 else 'FAIL'} ===")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
