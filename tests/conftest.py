"""A small self-contained CAE project shared by data and workflow tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

SENSORS = ["CP", "Center", "middle", "edge"]
CONTROLS = ["brine", "heater", "plasma"]
DT = 1.0
N_STEPS = 40

THERMAL_MASS = np.array([0.95, 1.15, 1.35, 1.55])
PLASMA_WEIGHT = np.array([1.00, 0.75, 0.55, 0.35])
HEATER_WEIGHT = np.array([0.25, 0.85, 0.65, 0.45])
BRINE_WEIGHT = np.array([0.15, 0.25, 0.55, 0.90])
EDGE_RESISTANCE = {(0, 1): 0.75, (1, 2): 0.50, (2, 3): 0.60, (1, 3): 1.60}


def conductance_matrix() -> np.ndarray:
    matrix = np.zeros((4, 4))
    for (node_a, node_b), resistance in EDGE_RESISTANCE.items():
        matrix[node_a, node_b] = matrix[node_b, node_a] = 0.045 / resistance
    return matrix


def simulate(brine: float, heater: float, plasma: float, init: np.ndarray) -> np.ndarray:
    """Independent forward-Euler reference used to create fixture data."""
    conductance = conductance_matrix()
    temperature = np.asarray(init, dtype=float).copy()
    output = np.zeros((N_STEPS, len(SENSORS)))
    for index in range(N_STEPS):
        output[index] = temperature
        if index == N_STEPS - 1:
            break
        conduction = (conductance * (temperature[None, :] - temperature[:, None])).sum(axis=1)
        plasma_source = 0.010 * plasma * PLASMA_WEIGHT
        heater_source = 0.014 * max(heater - 70.0, 0.0) * HEATER_WEIGHT
        brine_cooling = 0.030 * BRINE_WEIGHT * ((55.0 - 0.45 * brine) - temperature)
        ambient = 0.004 * (45.0 - temperature)
        temperature = (
            temperature
            + (conduction + plasma_source + heater_source + brine_cooling + ambient) / THERMAL_MASS
        )
    return output


def resolve_seed() -> int:
    return int(os.environ.get("CELLTEMP_TEST_SEED", "0"))


CONDITIONS = [
    (10.0, 80.0, 0.0),
    (10.0, 160.0, 100.0),
    (20.0, 120.0, 50.0),
    (20.0, 200.0, 150.0),
    (30.0, 80.0, 100.0),
    (30.0, 160.0, 0.0),
    (40.0, 120.0, 150.0),
    (40.0, 200.0, 50.0),
]


def _write_cae_csv(
    path: Path,
    temperatures: np.ndarray,
    controls: tuple[float, float, float],
) -> None:
    frame = pd.DataFrame({"time": np.arange(N_STEPS, dtype=float) * DT})
    for index, sensor in enumerate(SENSORS):
        frame[sensor] = temperatures[:, index]
    for name, value in zip(CONTROLS, controls):
        frame[name] = value
    frame.to_csv(path, index=False, float_format="%.6f")


def system_config() -> dict:
    return {
        "version": 1,
        "nodes": [
            {"name": name, "heat_capacity": float(capacity)}
            for name, capacity in zip(SENSORS, THERMAL_MASS)
        ],
        "actuators": [
            {"name": "brine", "tau": 0.0, "learnable": False},
            {"name": "heater", "tau": 0.0, "learnable": False},
            {"name": "plasma", "tau": 0.0, "learnable": False},
        ],
        "edges": [
            {
                "nodes": [SENSORS[node_a], SENSORS[node_b]],
                "conductance": 0.040 / resistance,
            }
            for (node_a, node_b), resistance in EDGE_RESISTANCE.items()
        ],
        "sources": [
            {
                "name": "plasma_heating",
                "actuator": "plasma",
                "node_weights": dict(zip(SENSORS, PLASMA_WEIGHT.tolist())),
                "gain": 0.008,
            },
            {
                "name": "heater_heating",
                "actuator": "heater",
                "node_weights": dict(zip(SENSORS, HEATER_WEIGHT.tolist())),
                "gain": 0.011,
                "threshold": 70.0,
            },
        ],
        "boundaries": [
            {
                "name": "brine_cooling",
                "actuator": "brine",
                "temperature_intercept": 55.0,
                "temperature_slope": -0.45,
                "node_weights": dict(zip(SENSORS, BRINE_WEIGHT.tolist())),
                "conductance": 0.024,
            },
            {
                "name": "ambient",
                "temperature_intercept": 45.0,
                "node_weights": dict.fromkeys(SENSORS, 1.0),
                "conductance": 0.003,
            },
        ],
        "sensors": [{"name": name, "node": name} for name in SENSORS],
    }


def train_config() -> dict:
    return {
        "seed": 42,
        "project": {
            "run_name": "test_run",
            "output_dir": "outputs/runs",
            "overwrite_run": True,
        },
        "data": {
            "directory": "data/raw",
            "pattern": "*.csv",
            "time_col": "time",
            "control_convention": "left",
            "sep": ",",
            "dt": DT,
            "temp_min": -50.0,
            "temp_max": 300.0,
        },
        "split": {
            "method": "random",
            "group_by_controls": True,
            "train_ratio": 0.70,
            "val_ratio": 0.15,
        },
        "system": "system.yaml",
        "engine": {"integrator": "exact"},
        "training": {
            "epochs": 8,
            "steps_per_epoch": 2,
            "batch_size": 4,
            "horizon": 25,
            "learning_rate": 0.04,
            "validation_every": 2,
            "patience": 5,
        },
    }


def runtime_config() -> dict:
    return {
        "forecast": {
            "input_dir": "data/pred/forecast",
            "pattern": "*.csv",
            "output_dir": "outputs/forecast",
            "overwrite": True,
            "device": "cpu",
        },
        "monitor": {
            "input_dir": "data/pred/monitor",
            "pattern": "*.csv",
            "output_dir": "outputs/monitor",
            "overwrite": True,
            "device": "cpu",
            "observer": {"sensor_std": 0.15},
        },
    }


def build_cae_project(root: Path) -> Path:
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True)
    initial = np.array([50.0, 52.0, 51.0, 54.0])
    for brine, heater, plasma in CONDITIONS:
        temperatures = simulate(brine, heater, plasma, initial)
        name = f"temp_{brine:g}_{heater:g}_{plasma:g}.csv"
        _write_cae_csv(raw_dir / name, temperatures, (brine, heater, plasma))
    (root / "system.yaml").write_text(
        yaml.safe_dump(system_config(), sort_keys=False), encoding="utf-8"
    )
    config = {**train_config(), **runtime_config()}
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root


@pytest.fixture
def cae_project(tmp_path: Path) -> Path:
    return build_cae_project(tmp_path)


@pytest.fixture
def data_cfg() -> dict:
    return {
        **train_config()["data"],
        "sensor_cols": list(SENSORS),
        "control_cols": list(CONTROLS),
    }


def write_forecast_request(root: Path) -> Path:
    target = root / "data" / "pred" / "forecast"
    target.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"time": np.arange(11, dtype=float) * DT})
    for sensor, initial in zip(SENSORS, [50.0, 52.0, 51.0, 54.0]):
        frame[sensor] = np.nan
        frame.loc[0, sensor] = initial
    frame["brine"] = 20.0
    frame["heater"] = 120.0
    frame["plasma"] = 50.0
    path = target / "const_case.csv"
    frame.to_csv(path, index=False)
    return path


def write_monitor_log(root: Path, *, noise: float = 0.0) -> Path:
    logs = root / "data" / "pred" / "monitor"
    logs.mkdir(parents=True, exist_ok=True)
    temperatures = simulate(20.0, 120.0, 50.0, np.array([50.0, 52.0, 51.0, 54.0]))
    if noise:
        rng = np.random.default_rng(resolve_seed())
        temperatures = temperatures + rng.normal(0.0, noise, size=temperatures.shape)
    frame = pd.DataFrame({"time": np.arange(N_STEPS, dtype=float)})
    for index, sensor in enumerate(SENSORS):
        frame[sensor] = temperatures[:, index]
    frame["brine"] = 20.0
    frame["heater"] = 120.0
    frame["plasma"] = 50.0
    log = logs / "monitor_case.csv"
    frame.to_csv(log, index=False)
    return log
