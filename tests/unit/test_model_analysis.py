"""Fitted-path and thermal-mode summaries for engineering review."""

import numpy as np
import pytest

from celltemp.analysis import representative_command, thermal_mode_rows, thermal_path_rows
from celltemp.domain import (
    ActuatorSpec,
    BoundarySpec,
    ConstantLawSpec,
    ReservoirTemperatureSpec,
    SourceSpec,
    ThermalSystemSpec,
    Trajectory,
)
from celltemp.engine import ThermalRCModel


def _single_node_model() -> ThermalRCModel:
    return ThermalRCModel(
        ThermalSystemSpec(
            node_names=("part",),
            heat_capacity=(4.0,),
            edges=(),
            actuators=(ActuatorSpec("heater", unit="W", role="heat_input"),),
            sources=(SourceSpec("absorbed", (1.0,), ConstantLawSpec(2.0)),),
            boundaries=(
                BoundarySpec(
                    "coolant",
                    (1.0,),
                    ReservoirTemperatureSpec(20.0),
                    ConstantLawSpec(1.0),
                ),
            ),
        )
    )


def test_representative_command_is_an_observed_row_near_the_median() -> None:
    trajectory = Trajectory(
        case_id="case",
        time=np.arange(4.0),
        temperature=np.full((4, 1), 20.0),
        commands=np.asarray([[0.0], [8.0], [10.0]]),
        sensor_names=("part",),
        control_names=("heater",),
    )

    selected = representative_command([trajectory])

    np.testing.assert_array_equal(selected, np.asarray([8.0]))


def test_thermal_path_rows_expose_capacity_conductance_resistance_and_heat() -> None:
    rows = thermal_path_rows(
        _single_node_model(),
        np.asarray([8.0]),
        operating_point="training_representative",
    )
    by_type = {str(row["element_type"]): row for row in rows}

    assert by_type["node"]["heat_capacity_j_per_k"] == pytest.approx(4.0)
    assert by_type["boundary"]["conductance_w_per_k"] == pytest.approx(1.0)
    assert by_type["boundary"]["resistance_k_per_w"] == pytest.approx(1.0)
    assert by_type["source"]["heat_rate_w"] == pytest.approx(2.0)


def test_thermal_mode_rows_report_expected_single_node_time_constant() -> None:
    rows = thermal_mode_rows(
        _single_node_model(),
        np.asarray([8.0]),
        operating_point="training_representative",
    )

    assert len(rows) == 1
    assert rows[0]["pole_real_per_s"] == pytest.approx(-0.25)
    assert rows[0]["time_constant_s"] == pytest.approx(4.0)
    assert rows[0]["dominant_node"] == "part"
