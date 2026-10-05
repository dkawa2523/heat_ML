"""Stored-solution extraction preserves physical units and the runtime boundary."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from external_tools import comsol_extract


def _raw_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0.0, 2.0],
            # Equal-volume materials with capacities 1 and 3 J/K at 280 and
            # 320 K produce 310 K, rather than the 300 K volume average.
            "node.body.heat_capacity": [4.0, 8.0],
            "node.body.temperature_integral": [1240.0, 2520.0],
            "node.shell.part.heat_capacity": [2.0, 6.0],
            "node.shell.part.temperature_integral": [560.0, 1740.0],
            "control.coolant.command": [293.15, 298.15],
            "control.power.command": [10.0, 15.0],
            "control.reference.command": [300.0, 301.0],
        }
    )


@pytest.mark.parametrize(("temperature_unit", "offset"), [("degC", 273.15), ("K", 0.0)])
def test_capacity_weighted_node_temperature_and_independent_control_units(
    temperature_unit: str, offset: float
) -> None:
    raw = _raw_table()
    original = raw.copy(deep=True)
    result = comsol_extract._convert_table(
        raw,
        {"body": "1,2", "shell.part": "shell"},
        {"coolant": "Tc", "power": "P", "reference": "Tref"},
        {"coolant": "degC", "power": "W", "reference": "K"},
        temperature_unit,
    )

    np.testing.assert_allclose(result["node.body.temperature"], np.array([310.0, 315.0]) - offset)
    np.testing.assert_allclose(
        result["node.shell.part.temperature"], np.array([280.0, 290.0]) - offset
    )
    np.testing.assert_allclose(result["control.coolant.command"], [20.0, 25.0])
    np.testing.assert_array_equal(result["control.power.command"], [10.0, 15.0])
    np.testing.assert_array_equal(result["control.reference.command"], [300.0, 301.0])
    np.testing.assert_array_equal(result["time"], raw["time"])
    np.testing.assert_array_equal(result["node.body.heat_capacity"], [4.0, 8.0])
    assert not any(str(column).endswith(".temperature_integral") for column in result)
    pd.testing.assert_frame_equal(raw, original)


@pytest.mark.parametrize("capacity", [0.0, -1.0, np.nan, np.inf])
def test_nonpositive_or_nonfinite_capacity_is_rejected_at_any_saved_time(capacity: float) -> None:
    frame = _raw_table()
    frame.loc[1, "node.body.heat_capacity"] = capacity
    with pytest.raises(ValueError, match="heat capacity must be positive and finite"):
        comsol_extract._convert_table(frame, {"body": "1"}, {}, {}, "degC")


def _decode(value: str) -> str:
    return base64.b64decode(value, validate=True).decode("utf-8")


def _decoded_rows(request: str) -> list[tuple[str, str, str, str]]:
    rows = []
    for line in _decode(request).splitlines():
        kind, name, value, unit = line.split("\t")
        rows.append((kind, _decode(name), _decode(value), _decode(unit)))
    return rows


def test_request_transport_keeps_utf8_expressions_delimiters_and_unit_requests() -> None:
    selection = "筐体\n内部\t層"
    expression = '熱源関数(t) + if(t>=2, 1[W], 0[W])\n + a\t*"quoted"'
    request = comsol_extract._request_rows(
        {"body.端": selection},
        {"coolant": "Tc(t)", "heater.温度": expression, "raw": "raw_value(t)"},
        {"coolant": "degC", "heater.温度": "W/m²", "raw": None},
    )

    assert request.isascii()
    assert _decoded_rows(request) == [
        ("region", "body.端", selection, ""),
        ("control", "coolant", "Tc(t)", "K"),
        ("control", "heater.温度", expression, "W/m²"),
        ("control", "raw", "raw_value(t)", ""),
    ]


def test_extract_model_connects_source_copy_compile_request_and_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "solved.mph"
    model.write_bytes(b"stored model remains unchanged")
    java_source = tmp_path / "java/ExtractThermalModel.java"
    java_source.parent.mkdir()
    java_source.write_text("// mock collector source", encoding="utf-8")
    work_dir = tmp_path / "work"
    events = []

    class Runtime:
        def compile(self, source: Path) -> Path:
            assert source == work_dir / java_source.name
            assert source.read_bytes() == java_source.read_bytes()
            compiled = source.with_suffix(".class")
            compiled.write_bytes(b"mock compiled collector")
            events.append("compile")
            return compiled

        def run_java(
            self,
            java_class: Path,
            arguments: Sequence[str | Path],
            *,
            log_path: Path,
            expected_output: Path,
            label: str,
        ) -> str:
            assert events == ["select", "compile"]
            assert java_class == work_dir / "ExtractThermalModel.class"
            assert arguments[:2] == [model.resolve(), work_dir / "extracted.txt"]
            assert [_decode(str(value)) for value in arguments[2:5]] == [
                "comp2",
                "dset5",
                "solid.rho*solid.Cp",
            ]
            assert _decoded_rows(str(arguments[5])) == [
                ("region", "body", "1,2", ""),
                ("region", "shell.part", "shell", ""),
                ("control", "coolant", "Tc(t)", "K"),
                ("control", "power", "P(t)", "W"),
                ("control", "reference", "Tref", "K"),
            ]
            assert log_path == work_dir / "extract.log"
            assert expected_output == work_dir / "extracted.txt"
            assert label == "stored thermal-model extraction"
            np.savetxt(
                expected_output,
                _raw_table().to_numpy(),
                delimiter="\t",
                header="Model: stored solution\nTime and derived values",
                comments="% ",
            )
            events.append("run")
            return "mock extraction completed"

    def select(explicit: str | None, *, source_model: Path | None = None) -> Runtime:
        assert explicit == "installed-comsol"
        assert source_model == model.resolve()
        events.append("select")
        return Runtime()

    monkeypatch.setattr(comsol_extract, "JAVA_SOURCE", java_source)
    monkeypatch.setattr(comsol_extract, "select_comsol", select)
    result = comsol_extract.extract_model(
        model,
        regions={"body": "1,2", "shell.part": "shell"},
        controls={"coolant": "Tc(t)", "power": "P(t)", "reference": "Tref"},
        control_units={"coolant": "degC", "power": "W", "reference": "K"},
        work_dir=work_dir,
        dataset="dset5",
        component="comp2",
        capacity_expression="solid.rho*solid.Cp",
        temperature_unit="degC",
        comsol_root="installed-comsol",
    )

    assert events == ["select", "compile", "run"]
    assert model.read_bytes() == b"stored model remains unchanged"
    np.testing.assert_allclose(result["node.body.temperature"], [36.85, 41.85])
    np.testing.assert_allclose(result["control.coolant.command"], [20.0, 25.0])
