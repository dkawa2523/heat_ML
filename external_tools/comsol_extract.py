"""Read stored COMSOL solutions without solving or changing the source model."""

from __future__ import annotations

import base64
import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from celltemp.config import temperature_unit_label
from external_tools.comsol_runtime import select_comsol

JAVA_SOURCE = Path(__file__).with_name("comsol") / "ExtractThermalModel.java"


def _encoded(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _request_rows(
    regions: Mapping[str, str],
    controls: Mapping[str, str],
    control_units: Mapping[str, str | None],
) -> str:
    rows = [
        "\t".join(("region", _encoded(name), _encoded(selection), ""))
        for name, selection in regions.items()
    ]
    for name, expression in controls.items():
        unit = control_units.get(name)
        unit = "K" if unit == "degC" else unit or ""
        rows.append("\t".join(("control", _encoded(name), _encoded(expression), _encoded(unit))))
    return _encoded("\n".join(rows))


def _convert_table(
    frame: pd.DataFrame,
    regions: Mapping[str, str],
    controls: Mapping[str, str],
    control_units: Mapping[str, str | None],
    temperature_unit: str,
) -> pd.DataFrame:
    """Match integrated storage and representative temperature, then label the scale."""
    temperature_unit_label(temperature_unit)
    result = frame.copy()
    for name in regions:
        capacity = result[f"node.{name}.heat_capacity"].to_numpy(dtype=float)
        integral_column = f"node.{name}.temperature_integral"
        integral = result[integral_column].to_numpy(dtype=float)
        if not np.isfinite(capacity).all() or np.any(capacity <= 0.0):
            raise ValueError(f"COMSOL region {name}: heat capacity must be positive and finite")
        temperature = integral / capacity
        if temperature_unit == "degC":
            temperature = temperature - 273.15
        result[f"node.{name}.temperature"] = temperature
        result = result.drop(columns=integral_column)
    for name in controls:
        if control_units.get(name) == "degC":
            column = f"control.{name}.command"
            result[column] = result[column] - 273.15
    return result


def extract_model(
    model_path: Path,
    *,
    regions: Mapping[str, str],
    controls: Mapping[str, str],
    control_units: Mapping[str, str | None],
    work_dir: Path,
    dataset: str | None = None,
    component: str = "comp1",
    capacity_expression: str = "ht.rho*ht.Cp",
    temperature_unit: str = "degC",
    comsol_root: str | None = None,
) -> pd.DataFrame:
    """Evaluate a stored transient solution into a temporary numeric table."""
    temperature_unit_label(temperature_unit)
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    runtime = select_comsol(comsol_root, source_model=model_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    java_source = work_dir / JAVA_SOURCE.name
    shutil.copyfile(JAVA_SOURCE, java_source)
    java_class = runtime.compile(java_source)
    table_path = work_dir / "extracted.txt"
    runtime.run_java(
        java_class,
        [
            model_path,
            table_path,
            _encoded(component),
            _encoded(dataset or ""),
            _encoded(capacity_expression),
            _request_rows(regions, controls, control_units),
        ],
        log_path=work_dir / "extract.log",
        expected_output=table_path,
        label="stored thermal-model extraction",
    )
    columns = ["time"]
    for name in regions:
        columns.extend((f"node.{name}.heat_capacity", f"node.{name}.temperature_integral"))
    columns.extend(f"control.{name}.command" for name in controls)
    frame = pd.DataFrame(np.loadtxt(table_path, comments="%", ndmin=2), columns=columns)
    return _convert_table(frame, regions, controls, control_units, temperature_unit)
