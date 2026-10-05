"""Import a solved COMSOL model into the normal thermal-system and trajectory formats.

Nodes use capacity-weighted regional temperatures and capacities at the first saved
time. Template sensors are representative observations of those nodes; this command
does not extract physical point probes or infer thermal conductances. Template
reservoir temperatures must already use the requested temperature scale.
Controls retain their values at stored solution times; configure data.control_convention
to match the source expression's interval labeling (right for completed-interval endpoints).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import yaml

from celltemp.config import temperature_unit_label
from celltemp.domain import ThermalSystemSpec
from celltemp.io import save_system_spec, system_spec_from_mapping, trajectory_from_frame
from celltemp.workflows.common import output_target, staged_output_directory
from external_tools.comsol_extract import extract_model


def _read_template(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        template = yaml.safe_load(stream)
    if not isinstance(template, Mapping):
        raise ValueError("system template must contain a mapping")
    return dict(template)


def _entity_names(template: Mapping[str, Any], field: str) -> tuple[str, ...]:
    entities = template.get(field, ())
    if isinstance(entities, (str, bytes)) or not isinstance(entities, Sequence):
        raise ValueError(f"template {field} must be a sequence")
    names = []
    for entity in entities:
        if not isinstance(entity, Mapping):
            raise ValueError(f"template {field} entries must be mappings")
        name = entity.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"template {field} names must be non-empty strings")
        names.append(name)
    if len(names) != len(set(names)) or (field == "nodes" and not names):
        raise ValueError(f"template {field} names must be unique; nodes must be non-empty")
    return tuple(names)


def _require_bindings(bindings: Mapping[str, str], names: tuple[str, ...], label: str) -> None:
    if set(bindings) != set(names):
        raise ValueError(f"{label} must specify every template entity exactly once: {list(names)}")
    if any(not isinstance(value, str) or not value.strip() for value in bindings.values()):
        raise ValueError(f"{label} values must be non-empty strings")


def _validate_extraction(
    extracted: pd.DataFrame, nodes: tuple[str, ...], controls: tuple[str, ...]
) -> None:
    columns = [
        "time",
        *(f"node.{name}.heat_capacity" for name in nodes),
        *(f"node.{name}.temperature" for name in nodes),
        *(f"control.{name}.command" for name in controls),
    ]
    if not extracted.columns.is_unique or not set(columns) <= set(extracted.columns):
        raise ValueError("COMSOL extraction must contain unique time, capacity, and value columns")
    if len(extracted) < 2 or not np.isfinite(extracted[columns].to_numpy(dtype=float)).all():
        raise ValueError("COMSOL extraction must have at least two rows of finite values")


def _draft_outputs(
    template: Mapping[str, Any], extracted: pd.DataFrame
) -> tuple[ThermalSystemSpec, pd.DataFrame]:
    nodes = _entity_names(template, "nodes")
    controls = _entity_names(template, "actuators")
    _validate_extraction(extracted, nodes, controls)
    draft = deepcopy(dict(template))
    for node in draft["nodes"]:
        node["heat_capacity"] = float(extracted[f"node.{node['name']}.heat_capacity"].iloc[0])
    spec = system_spec_from_mapping(draft)
    node_temperature = extracted[[f"node.{name}.temperature" for name in nodes]].to_numpy(
        dtype=float
    )
    sensor_temperature = node_temperature @ spec.observation_matrix.T
    frame = pd.DataFrame({"time": extracted["time"].to_numpy(dtype=float)})
    for index, sensor in enumerate(spec.sensor_names):
        frame[sensor] = sensor_temperature[:, index]
    for control in spec.control_names:
        frame[control] = extracted[f"control.{control}.command"].to_numpy(dtype=float)
    trajectory_from_frame(
        case_id="trajectory",
        frame=frame,
        time_col="time",
        sensor_cols=spec.sensor_names,
        control_cols=spec.control_names,
    )
    return spec, frame


def _validate_temperature_controls(
    template: Mapping[str, Any], control_units: Mapping[str, str | None], temperature_unit: str
) -> None:
    temperature_unit_label(temperature_unit)
    for boundary in template.get("boundaries", ()):
        if not isinstance(boundary, Mapping):
            continue
        reservoir = boundary.get("reservoir_temperature", {})
        if not isinstance(reservoir, Mapping):
            continue
        control = reservoir.get("control")
        unit = control_units.get(control) if isinstance(control, str) else None
        if unit in {"degC", "K"} and unit != temperature_unit:
            raise ValueError(
                f"reservoir temperature control {control!r} uses {unit}; "
                f"template and extraction must use the same temperature unit {temperature_unit}"
            )


def import_comsol_model(
    model_path: Path,
    template_path: Path,
    output_dir: Path,
    *,
    regions: Mapping[str, str],
    controls: Mapping[str, str],
    dataset: str | None = None,
    component: str = "comp1",
    capacity_expression: str = "ht.rho*ht.Cp",
    temperature_unit: str = "degC",
    comsol_root: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Publish a validated draft only after extracting and converting a solved model."""
    model_path, template_path = model_path.resolve(), template_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"COMSOL model does not exist: {model_path}")
    template = _read_template(template_path)
    _require_bindings(regions, _entity_names(template, "nodes"), "regions")
    _require_bindings(controls, _entity_names(template, "actuators"), "controls")
    control_units = {item["name"]: item.get("unit") for item in template.get("actuators", ())}
    _validate_temperature_controls(template, control_units, temperature_unit)
    target, overwrite = output_target(
        {"output_dir": str(output_dir.resolve()), "overwrite": overwrite},
        Path.cwd(),
        protected_paths=(model_path, template_path),
    )
    with (
        staged_output_directory(target, overwrite=overwrite) as staging,
        TemporaryDirectory(prefix="comsol-", dir=staging) as work_dir,
    ):
        extracted = extract_model(
            model_path,
            regions=regions,
            controls=controls,
            control_units=control_units,
            dataset=dataset,
            component=component,
            capacity_expression=capacity_expression,
            temperature_unit=temperature_unit,
            comsol_root=comsol_root,
            work_dir=Path(work_dir),
        )
        spec, frame = _draft_outputs(template, extracted)
        save_system_spec(spec, staging / "system.yaml")
        frame.to_csv(staging / "trajectory.csv", index=False)
    return target


def _assignments(values: Sequence[str], label: str) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, expression = value.partition("=")
        name, expression = name.strip(), expression.strip()
        if not separator or not name or not expression:
            raise ValueError(f"{label} requires name=value: {value!r}")
        if name in result:
            raise ValueError(f"{label} repeats the template name {name!r}")
        result[name] = expression
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Solved COMSOL .mph model")
    parser.add_argument("--template", type=Path, required=True, help="Existing system YAML schema")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--region", action="append", default=[], help="node=domain_selection or node=2,3; repeat"
    )
    parser.add_argument(
        "--control", action="append", default=[], help="actuator=COMSOL_expression at saved times"
    )
    parser.add_argument("--dataset", help="Transient solution dataset; required if ambiguous")
    parser.add_argument("--component", default="comp1")
    parser.add_argument("--capacity-expression", default="ht.rho*ht.Cp")
    parser.add_argument("--temperature-unit", choices=("degC", "K"), default="degC")
    parser.add_argument("--comsol-root", help="COMSOL Multiphysics installation root")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(None if argv is None else list(argv))
    target = import_comsol_model(
        args.model,
        args.template,
        args.output_dir,
        regions=_assignments(args.region, "--region"),
        controls=_assignments(args.control, "--control"),
        dataset=args.dataset,
        component=args.component,
        capacity_expression=args.capacity_expression,
        temperature_unit=args.temperature_unit,
        comsol_root=args.comsol_root,
        overwrite=args.overwrite,
    )
    print(f"Saved system.yaml and trajectory.csv: {target}")
    print(f"Use project.temperature_unit: {args.temperature_unit}.")
    print("Set data.control_convention to match the COMSOL expression's saved endpoint values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
