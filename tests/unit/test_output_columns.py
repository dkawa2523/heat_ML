"""Saved waveform columns preserve arbitrary entity names and their quantities."""

from pathlib import Path

import numpy as np
import pandas as pd

from celltemp.artifact import ThermalArtifact, save_artifact
from celltemp.config import save_yaml
from celltemp.domain import (
    ActuatorSpec,
    BoundarySpec,
    ConstantLawSpec,
    EdgeSpec,
    ReservoirTemperatureSpec,
    SourceSpec,
    ThermalSystemSpec,
    Trajectory,
)
from celltemp.engine import ThermalRCModel
from celltemp.inference import forecast, monitor
from celltemp.io import trajectory_from_frame
from celltemp.workflows import run_monitor
from celltemp.workflows.forecast_diagnostics import _energy_balance_frame
from celltemp.workflows.forecast_output import forecast_frame
from celltemp.workflows.train_output import _CasePrediction, _write_test_predictions


def _model() -> ThermalRCModel:
    return ThermalRCModel(
        ThermalSystemSpec(
            node_names=("n", "std_n", "n.temperature"),
            heat_capacity=(1.0, 2.0, 3.0),
            edges=(),
            actuators=(ActuatorSpec("heater.command", tau=0.0, learnable=False),),
            sensor_names=("tc", "std_tc", "tc.std"),
            sensor_nodes=("n", "std_n", "n.temperature"),
        )
    )


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "tc": [20.0, 21.0, np.nan],
            "std_tc": [30.0, 31.0, 32.0],
            "tc.std": [40.0, np.nan, 42.0],
            "heater.command": [0.0, 1.0, 999.0],
        }
    )


def _trajectory(model: ThermalRCModel, source: pd.DataFrame) -> Trajectory:
    return trajectory_from_frame(
        case_id="names",
        frame=source,
        time_col="time",
        sensor_cols=model.spec.sensor_names,
        control_cols=model.spec.control_names,
    )


def test_forecast_columns_keep_temperature_uncertainty_and_dotted_names() -> None:
    model = _model()
    source = _source()
    source.loc[1:, list(model.spec.sensor_names)] = np.nan
    request = _trajectory(model, source)
    predicted = forecast(model, request)
    artifact = ThermalArtifact(model, {}, Path("unused"))

    frame = forecast_frame(artifact, request, predicted)

    assert len(frame.columns) == 1 + 3 * 4 + 3 * 2 + 2 + 3
    assert frame.columns.is_unique
    for index, sensor in enumerate(model.spec.sensor_names):
        np.testing.assert_array_equal(
            frame[f"sensor.{sensor}.temperature"], predicted.sensor_temperature[:, index]
        )
        np.testing.assert_array_equal(
            frame[f"sensor.{sensor}.std"], predicted.sensor_temperature_std[:, index]
        )
    for index, node in enumerate(model.spec.node_names):
        np.testing.assert_array_equal(
            frame[f"node.{node}.temperature"], predicted.node_temperature[:, index]
        )
        np.testing.assert_array_equal(
            frame[f"node.{node}.std"], predicted.node_temperature_std[:, index]
        )
    energy = _energy_balance_frame(model, case_id="names", frame=frame)
    np.testing.assert_allclose(energy["total.balance_residual_w"], 0.0)


def test_energy_columns_keep_total_entities_and_distinct_edge_paths() -> None:
    model = ThermalRCModel(
        ThermalSystemSpec(
            node_names=("a_to_b", "c", "a", "b_to_c", "total"),
            heat_capacity=(1.0,) * 5,
            edges=(
                EdgeSpec("a_to_b", "c", ConstantLawSpec(2.0)),
                EdgeSpec("a", "b_to_c", ConstantLawSpec(3.0)),
            ),
            actuators=(),
            sources=(
                SourceSpec("total", (1.0, 0.0, 0.0, 0.0, 0.0), ConstantLawSpec(2.0)),
                SourceSpec("other", (0.0, 1.0, 0.0, 0.0, 0.0), ConstantLawSpec(3.0)),
            ),
            boundaries=(
                BoundarySpec(
                    "total",
                    (1.0, 0.0, 0.0, 0.0, 0.0),
                    ReservoirTemperatureSpec(10.0),
                    ConstantLawSpec(0.5),
                ),
                BoundarySpec(
                    "cool",
                    (0.0, 1.0, 0.0, 0.0, 0.0),
                    ReservoirTemperatureSpec(0.0),
                    ConstantLawSpec(0.1),
                ),
            ),
        )
    )
    frame = pd.DataFrame({"time": [0.0, 1.0]})
    for name, temperature in zip(
        model.spec.node_names, (20.0, 30.0, 40.0, 50.0, 60.0), strict=True
    ):
        frame[f"node.{name}.temperature"] = temperature

    energy = _energy_balance_frame(model, case_id="names", frame=frame)

    assert len(energy.columns) == 29
    assert energy.columns.is_unique
    np.testing.assert_allclose(energy["source.total.heat_w"], 2.0)
    np.testing.assert_allclose(energy["total.source_heat_w"], 5.0)
    np.testing.assert_allclose(energy["boundary.total.heat_w"], -5.0)
    np.testing.assert_allclose(energy["total.boundary_heat_w"], -8.0)
    np.testing.assert_allclose(energy["boundary.total.reservoir_temperature"], 10.0)
    np.testing.assert_allclose(energy["node.total.storage_w"], 0.0)
    np.testing.assert_allclose(energy["total.storage_w"], -3.0)
    np.testing.assert_allclose(energy["edge.0.heat_w"], -20.0)
    np.testing.assert_allclose(energy["edge.1.heat_w"], -30.0)
    np.testing.assert_allclose(energy["total.balance_residual_w"], 0.0)


def test_monitor_columns_preserve_innovation_std_and_commands(tmp_path: Path) -> None:
    model = _model()
    source = _source()
    request = _trajectory(model, source)
    expected = monitor(model, request)
    save_artifact(tmp_path / "artifact", model)
    (tmp_path / "inputs").mkdir()
    source.to_csv(tmp_path / "inputs/names.csv", index=False)
    cfg = {"artifact": "artifact", "monitor": {"input_dir": "inputs", "output_dir": "monitor"}}
    config_path = tmp_path / "config.yaml"
    save_yaml(cfg, config_path)

    output = run_monitor(cfg, config_path)
    frame = pd.read_csv(output / "cases/names.csv")

    assert len(frame.columns) == 1 + 3 * 8 + 3 * 2 + 2 + 3
    assert frame.columns.is_unique
    for index, sensor in enumerate(model.spec.sensor_names):
        np.testing.assert_allclose(
            frame[f"sensor.{sensor}.innovation"], expected.innovation[:, index], equal_nan=True
        )
        np.testing.assert_allclose(
            frame[f"sensor.{sensor}.innovation_std"],
            expected.innovation_std[:, index],
            equal_nan=True,
        )
    np.testing.assert_array_equal(frame["control.heater.command.command"], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(frame["control.heater.command.effective"], [0.0, 1.0, 1.0])


def test_training_prediction_columns_preserve_the_mask_and_distinct_quantities(
    tmp_path: Path,
) -> None:
    trajectory = Trajectory(
        case_id="names",
        time=np.array([0.0, 1.0]),
        temperature=np.array([[20.0, 30.0], [21.0, 999.0]]),
        commands=np.empty((1, 0)),
        sensor_names=("tc", "from_initial_observation_tc"),
        control_names=(),
        observation_mask=np.array([[True, True], [True, False]]),
    )
    conditional = np.array([[20.0, 30.0], [21.1, 31.0]])
    causal = conditional + 0.2

    _write_test_predictions((_CasePrediction(trajectory, "test", conditional, causal),), tmp_path)
    frame = pd.read_csv(tmp_path / "names.csv")

    assert len(frame.columns) == 1 + 2 * 4
    np.testing.assert_array_equal(frame["sensor.tc.causal"], causal[:, 0])
    np.testing.assert_array_equal(
        frame["sensor.from_initial_observation_tc.conditional"], conditional[:, 1]
    )
    assert np.isnan(frame["sensor.from_initial_observation_tc.observed"].iat[1])
    assert np.isnan(frame["sensor.from_initial_observation_tc.error"].iat[1])
