"""Single source of synthetic TopCell truth used by generation and evaluation."""

from __future__ import annotations

import numpy as np

SENSORS = ("CP", "Center", "middle", "edge")
CONTROLS = ("brine", "heater", "plasma")

THERMAL_MASS = np.array([0.95, 1.15, 1.35, 1.55], dtype=float)
PLASMA_WEIGHT = np.array([1.00, 0.75, 0.55, 0.35], dtype=float)
HEATER_WEIGHT = np.array([0.25, 0.85, 0.65, 0.45], dtype=float)
BRINE_WEIGHT = np.array([0.15, 0.25, 0.55, 0.90], dtype=float)

EDGE_RESISTANCE = {
    (0, 1): 0.75,
    (1, 2): 0.50,
    (2, 3): 0.60,
    (1, 3): 1.60,
}
CONDUCTANCE = np.zeros((len(SENSORS), len(SENSORS)), dtype=float)
for (node_a, node_b), resistance in EDGE_RESISTANCE.items():
    CONDUCTANCE[node_a, node_b] = CONDUCTANCE[node_b, node_a] = 0.045 / resistance

CONTROL_TAU = np.array([4.0, 8.0, 1.0], dtype=float)
PLASMA_GAIN = 0.010
HEATER_GAIN = 0.014
HEATER_THRESHOLD = 70.0
BRINE_TEMPERATURE_INTERCEPT = 55.0
BRINE_TEMPERATURE_SLOPE = -0.45
BRINE_CONDUCTANCE = 0.030
AMBIENT_TEMPERATURE = 45.0
AMBIENT_CONDUCTANCE = 0.004
SENSOR_NOISE_STD = 0.15

PARAMETER_TRUTH = {
    **{
        ("edge", f"{SENSORS[node_a]}-{SENSORS[node_b]}"): CONDUCTANCE[node_a, node_b]
        for node_a, node_b in EDGE_RESISTANCE
    },
    **{("actuator", control): float(tau) for control, tau in zip(CONTROLS, CONTROL_TAU)},
    ("source", "plasma_heating"): PLASMA_GAIN,
    ("source", "heater_heating"): HEATER_GAIN,
    ("boundary", "brine_cooling"): BRINE_CONDUCTANCE,
    ("boundary", "ambient"): AMBIENT_CONDUCTANCE,
}
