"""Generate a purpose-driven thermal identification benchmark.

Training data contains only identification experiments. Forecast and monitor
cases live in a separate external-evaluation directory and are never discovered
by the training workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .definition import (
    AMBIENT_CONDUCTANCE,
    AMBIENT_TEMPERATURE,
    BRINE_CONDUCTANCE,
    BRINE_TEMPERATURE_INTERCEPT,
    BRINE_TEMPERATURE_SLOPE,
    BRINE_WEIGHT,
    CONDUCTANCE,
    CONTROL_TAU,
    CONTROLS,
    HEATER_GAIN,
    HEATER_THRESHOLD,
    HEATER_WEIGHT,
    PLASMA_GAIN,
    PLASMA_WEIGHT,
    SENSOR_NOISE_STD,
    SENSORS,
    THERMAL_MASS,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "work" / "data" / "raw"
EVAL_DIR = ROOT / "work" / "data" / "eval"
FORECAST_DIR = EVAL_DIR / "forecast"
MONITOR_DIR = EVAL_DIR / "monitor"

IDENTIFICATION_KINDS = [
    "step_brine",
    "step_heater",
    "step_plasma",
    "ramp_brine",
    "ramp_heater",
    "ramp_plasma",
]
T_END = 180.0
REGULAR_TIMES = np.arange(0.0, T_END + 1.0, 1.0)


@dataclass(frozen=True)
class ForecastCase:
    case_id: str
    group: str
    purpose: str
    times: np.ndarray
    controls: np.ndarray
    initial_temperature: np.ndarray
    initial_sensor_mask: np.ndarray
    history_end_time: float = 0.0
    ambient_temperature_coefficient: float = 0.0


def _clean_csv_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.csv"):
        path.unlink()


def simulate(
    raw_controls: np.ndarray,
    initial_temperature: np.ndarray,
    times: np.ndarray,
    *,
    extra_heat: np.ndarray | None = None,
    ambient_temperature_coefficient: float = 0.0,
) -> np.ndarray:
    """Integrate the synthetic truth with causal first-order actuator states."""
    controls = np.asarray(raw_controls, dtype=float)
    time = np.asarray(times, dtype=float)
    if controls.shape != (len(time), len(CONTROLS)):
        raise ValueError("controls must have one row per timestamp")
    injected_heat = (
        np.zeros((len(time), len(SENSORS)), dtype=float)
        if extra_heat is None
        else np.asarray(extra_heat, dtype=float)
    )
    if injected_heat.shape != (len(time), len(SENSORS)):
        raise ValueError("extra_heat must have one row per timestamp and sensor node")

    temperature = np.asarray(initial_temperature, dtype=float).copy()
    actuator = controls[0].copy()
    output = np.zeros((len(time), len(SENSORS)), dtype=float)
    for index in range(len(time)):
        output[index] = temperature
        if index == len(time) - 1:
            break
        dt = time[index + 1] - time[index]
        command = controls[index]
        decay_half = np.exp(-0.5 * dt / CONTROL_TAU)
        decay_full = decay_half * decay_half
        actuator_midpoint = command + (actuator - command) * decay_half
        brine, heater, plasma = actuator_midpoint

        conduction = (CONDUCTANCE * (temperature[None, :] - temperature[:, None])).sum(axis=1)
        plasma_source = PLASMA_GAIN * plasma * PLASMA_WEIGHT
        heater_source = HEATER_GAIN * max(heater - HEATER_THRESHOLD, 0.0) * HEATER_WEIGHT
        brine_temperature = BRINE_TEMPERATURE_INTERCEPT + BRINE_TEMPERATURE_SLOPE * brine
        brine_cooling = BRINE_CONDUCTANCE * BRINE_WEIGHT * (brine_temperature - temperature)
        ambient_h = AMBIENT_CONDUCTANCE * (
            1.0
            + ambient_temperature_coefficient * np.maximum(temperature - AMBIENT_TEMPERATURE, 0.0)
        )
        ambient_loss = ambient_h * (AMBIENT_TEMPERATURE - temperature)
        rate = (
            conduction
            + plasma_source
            + heater_source
            + brine_cooling
            + ambient_loss
            + injected_heat[index]
        ) / THERMAL_MASS
        temperature = temperature + dt * rate
        actuator = command + (actuator - command) * decay_full
    return output


def constant_controls(
    brine: float,
    heater: float,
    plasma: float,
    times: np.ndarray = REGULAR_TIMES,
) -> np.ndarray:
    values = np.array([[brine, heater, plasma]], dtype=float)
    return np.repeat(values, len(times), axis=0)


def _trapezoid(
    times: np.ndarray,
    rise_from: float,
    rise_to: float,
    fall_from: float,
    fall_to: float,
) -> np.ndarray:
    up = np.clip((times - rise_from) / (rise_to - rise_from), 0.0, 1.0)
    down = np.clip((fall_to - times) / (fall_to - fall_from), 0.0, 1.0)
    return np.minimum(up, down)


def identification_schedule(
    base: tuple[float, float, float], kind: str, times: np.ndarray = REGULAR_TIMES
) -> np.ndarray:
    controls = np.repeat(np.array([base], dtype=float), len(times), axis=0)
    actuator_name = kind.removeprefix("step_").removeprefix("ramp_")
    control_index = CONTROLS.index(actuator_name)
    baseline = {"brine": 25.0, "heater": 100.0, "plasma": 0.0}[actuator_name]
    target = base[control_index]
    controls[:, control_index] = baseline
    if kind.startswith("step_"):
        active = (times >= 35.0) & (times < 125.0)
        controls[active, control_index] = target
    elif kind.startswith("ramp_"):
        profile = _trapezoid(times, 20.0, 65.0, 115.0, 170.0)
        controls[:, control_index] = baseline + (target - baseline) * profile
    else:
        raise ValueError(f"unknown identification schedule: {kind}")
    return controls


def _novel_recipe(times: np.ndarray) -> np.ndarray:
    controls = constant_controls(30.0, 180.0, 0.0, times)
    controls[(times >= 15.0) & (times < 55.0)] = [20.0, 180.0, 120.0]
    controls[(times >= 55.0) & (times < 90.0)] = [35.0, 140.0, 60.0]
    controls[(times >= 90.0) & (times < 135.0)] = [10.0, 200.0, 150.0]
    controls[times >= 135.0] = [40.0, 80.0, 0.0]
    return controls


def _short_pulses(times: np.ndarray) -> np.ndarray:
    controls = constant_controls(25.0, 140.0, 0.0, times)
    active = (times >= 20.0) & (times < 140.0)
    pulse_number = np.floor((times - 20.0) / 10.0).astype(int)
    controls[active & (pulse_number % 2 == 0), 2] = 150.0
    return controls


def _smooth_multiaxis_controls(times: np.ndarray) -> np.ndarray:
    controls = np.empty((len(times), len(CONTROLS)), dtype=float)
    controls[:, 0] = 35.0 - 20.0 * _trapezoid(times, 10.0, 55.0, 120.0, 175.0)
    controls[:, 1] = 100.0 + 90.0 * _trapezoid(times, 15.0, 60.0, 115.0, 165.0)
    controls[:, 2] = 140.0 * _trapezoid(times, 25.0, 50.0, 105.0, 145.0)
    return controls


def _irregular_times() -> np.ndarray:
    increments = [0.5, 1.5, 0.75, 1.25]
    values = [0.0]
    index = 0
    while values[-1] < T_END:
        values.append(min(T_END, values[-1] + increments[index % len(increments)]))
        index += 1
    return np.asarray(values, dtype=float)


def _write_training_csv(
    path: Path, times: np.ndarray, temperature: np.ndarray, controls: np.ndarray
) -> None:
    frame = pd.DataFrame({"time": times})
    for sensor_index, sensor in enumerate(SENSORS):
        frame[sensor] = temperature[:, sensor_index]
    for control_index, control in enumerate(CONTROLS):
        frame[control] = controls[:, control_index]
    frame.to_csv(path, index=False, float_format="%.6f")


def generate_identification_data() -> None:
    """Create steady-level coverage plus isolated actuator excitation."""
    _clean_csv_directory(TRAIN_DIR)
    initial_patterns = {
        "cold": np.array([45.0, 45.0, 45.0, 45.0]),
        "nominal": np.array([55.0, 55.0, 55.0, 55.0]),
        "gradient": np.array([54.0, 56.0, 59.0, 62.0]),
    }
    for brine in [10, 20, 30, 40]:
        for heater in [80, 120, 160, 200]:
            for plasma in [0, 50, 100, 150]:
                controls = constant_controls(brine, heater, plasma)
                for pattern_name, initial in initial_patterns.items():
                    temperature = simulate(controls, initial, REGULAR_TIMES)
                    filename = f"id_const_b{brine}_h{heater}_p{plasma}_{pattern_name}.csv"
                    _write_training_csv(TRAIN_DIR / filename, REGULAR_TIMES, temperature, controls)

    excitation_bases = [
        (10.0, 160.0, 50.0),
        (10.0, 200.0, 150.0),
        (30.0, 80.0, 100.0),
        (30.0, 160.0, 150.0),
        (40.0, 120.0, 50.0),
        (40.0, 200.0, 100.0),
    ]
    for dynamic_brine, dynamic_heater, dynamic_plasma in excitation_bases:
        for kind in IDENTIFICATION_KINDS:
            controls = identification_schedule(
                (dynamic_brine, dynamic_heater, dynamic_plasma), kind
            )
            initial = initial_patterns["nominal"]
            temperature = simulate(controls, initial, REGULAR_TIMES)
            filename = f"id_{kind}_b{dynamic_brine:g}_h{dynamic_heater:g}_p{dynamic_plasma:g}.csv"
            _write_training_csv(TRAIN_DIR / filename, REGULAR_TIMES, temperature, controls)


def forecast_cases() -> list[ForecastCase]:
    observed_all = np.ones(len(SENSORS), dtype=bool)
    irregular = _irregular_times()
    return [
        ForecastCase(
            "F01_level_interpolation",
            "interpolation",
            "unseen levels inside the training control range",
            REGULAR_TIMES,
            constant_controls(25.0, 140.0, 75.0),
            np.array([50.0, 54.0, 58.0, 61.0]),
            observed_all,
        ),
        ForecastCase(
            "F02_coupled_interpolation",
            "interpolation",
            "simultaneous intermediate heating and cooling levels",
            REGULAR_TIMES,
            constant_controls(35.0, 180.0, 125.0),
            np.array([58.0, 56.0, 53.0, 50.0]),
            observed_all,
        ),
        ForecastCase(
            "F03_high_power_extrapolation",
            "extrapolation",
            "high-power operation outside all trained command maxima",
            REGULAR_TIMES,
            constant_controls(5.0, 220.0, 175.0),
            np.array([55.0, 55.0, 55.0, 55.0]),
            observed_all,
        ),
        ForecastCase(
            "F04_strong_cooling_extrapolation",
            "extrapolation",
            "cooldown outside the trained brine and heater range",
            REGULAR_TIMES,
            constant_controls(50.0, 70.0, 20.0),
            np.array([75.0, 72.0, 68.0, 65.0]),
            observed_all,
        ),
        ForecastCase(
            "F05_short_plasma_pulses",
            "dynamics",
            "repeated switching faster than identification steps",
            REGULAR_TIMES,
            _short_pulses(REGULAR_TIMES),
            np.array([55.0, 55.0, 55.0, 55.0]),
            observed_all,
        ),
        ForecastCase(
            "F06_multiaxis_recipe",
            "dynamics",
            "unseen phase order with all commands changing together",
            REGULAR_TIMES,
            _novel_recipe(REGULAR_TIMES),
            np.array([54.0, 57.0, 60.0, 63.0]),
            observed_all,
        ),
        ForecastCase(
            "F07_smooth_multiaxis_ramp",
            "dynamics",
            "unseen smooth and overlapping command ramps",
            REGULAR_TIMES,
            _smooth_multiaxis_controls(REGULAR_TIMES),
            np.array([52.0, 55.0, 58.0, 60.0]),
            observed_all,
        ),
        ForecastCase(
            "F08_hot_start_cooldown",
            "initialization",
            "initial temperature far outside trained initial patterns",
            REGULAR_TIMES,
            constant_controls(40.0, 80.0, 0.0),
            np.array([85.0, 80.0, 75.0, 70.0]),
            observed_all,
        ),
        ForecastCase(
            "F09_sparse_initial_observation",
            "initialization",
            "open-loop start with only two of four sensors available",
            REGULAR_TIMES,
            _novel_recipe(REGULAR_TIMES),
            np.array([52.0, 58.0, 64.0, 70.0]),
            np.array([False, True, True, False]),
        ),
        ForecastCase(
            "F10_variable_dt_recipe",
            "sampling",
            "same physical integration without fixed-step resampling",
            irregular,
            _novel_recipe(irregular),
            np.array([51.0, 55.0, 59.0, 62.0]),
            observed_all,
        ),
        ForecastCase(
            "F11_nonlinear_heat_loss",
            "model_gap",
            "negative control exposing temperature-dependent heat loss",
            REGULAR_TIMES,
            constant_controls(20.0, 160.0, 120.0),
            np.array([115.0, 108.0, 100.0, 92.0]),
            observed_all,
            ambient_temperature_coefficient=0.012,
        ),
        ForecastCase(
            "F12_history_initialized_sparse",
            "initialization",
            "causal sensor history reconstructs hidden temperatures before open-loop handoff",
            REGULAR_TIMES,
            _novel_recipe(REGULAR_TIMES),
            np.array([52.0, 58.0, 64.0, 70.0]),
            np.array([False, True, True, False]),
            history_end_time=20.0,
        ),
    ]


def generate_forecast_evaluation() -> None:
    """Write request columns and sequestered truth columns in the same case file."""
    _clean_csv_directory(FORECAST_DIR)
    for case in forecast_cases():
        truth = simulate(
            case.controls,
            case.initial_temperature,
            case.times,
            ambient_temperature_coefficient=case.ambient_temperature_coefficient,
        )
        frame = pd.DataFrame({"time": case.times})
        history = case.times <= case.history_end_time
        for sensor_index, sensor in enumerate(SENSORS):
            request = np.full(len(case.times), np.nan)
            if case.initial_sensor_mask[sensor_index]:
                request[history] = truth[history, sensor_index]
            frame[sensor] = request
        for control_index, control in enumerate(CONTROLS):
            frame[control] = case.controls[:, control_index]
        for sensor_index, sensor in enumerate(SENSORS):
            frame[f"truth_{sensor}"] = truth[:, sensor_index]
        frame["benchmark_group"] = case.group
        frame["benchmark_purpose"] = case.purpose
        frame.to_csv(FORECAST_DIR / f"{case.case_id}.csv", index=False, float_format="%.6f")


def _write_monitor_case(
    *,
    case_id: str,
    group: str,
    purpose: str,
    controls: np.ndarray,
    initial_temperature: np.ndarray,
    rng: np.random.Generator,
    deterministic_bias: np.ndarray | None = None,
    missing: np.ndarray | None = None,
    extra_heat: np.ndarray | None = None,
    event_start: float | None = None,
) -> None:
    truth = simulate(
        controls,
        initial_temperature,
        REGULAR_TIMES,
        extra_heat=extra_heat,
    )
    bias = (
        np.zeros_like(truth)
        if deterministic_bias is None
        else np.asarray(deterministic_bias, dtype=float)
    )
    disturbance = (
        np.zeros_like(truth) if extra_heat is None else np.asarray(extra_heat, dtype=float)
    )
    measured = truth + bias + rng.normal(0.0, SENSOR_NOISE_STD, size=truth.shape)
    if missing is not None:
        measured[np.asarray(missing, dtype=bool)] = np.nan

    frame = pd.DataFrame({"time": REGULAR_TIMES})
    for sensor_index, sensor in enumerate(SENSORS):
        frame[sensor] = measured[:, sensor_index]
    for control_index, control in enumerate(CONTROLS):
        frame[control] = controls[:, control_index]
    for sensor_index, sensor in enumerate(SENSORS):
        frame[f"truth_{sensor}"] = truth[:, sensor_index]
        frame[f"truth_bias_{sensor}"] = bias[:, sensor_index]
        frame[f"truth_disturbance_{sensor}_w"] = disturbance[:, sensor_index]
    frame["benchmark_group"] = group
    frame["benchmark_purpose"] = purpose
    if event_start is not None:
        frame["event_start"] = event_start
    frame.to_csv(MONITOR_DIR / f"{case_id}.csv", index=False, float_format="%.6f")


def generate_monitor_evaluation(seed: int) -> None:
    """Create nominal, estimation, data-loss, and detection scenarios."""
    _clean_csv_directory(MONITOR_DIR)
    rng = np.random.default_rng(seed)
    nominal_initial = np.array([54.0, 57.0, 60.0, 63.0])

    _write_monitor_case(
        case_id="M01_nominal_noise",
        group="baseline",
        purpose="innovation scale under measurement noise only",
        controls=_smooth_multiaxis_controls(REGULAR_TIMES),
        initial_temperature=nominal_initial,
        rng=rng,
    )

    progress = (REGULAR_TIMES / T_END)[:, None]
    drift = progress * np.array([0.55, 0.30, -0.20, 0.40])[None, :]
    _write_monitor_case(
        case_id="M02_sensor_bias_drift",
        group="bias_tracking",
        purpose="separate slow sensor bias from physical temperature",
        controls=_novel_recipe(REGULAR_TIMES),
        initial_temperature=nominal_initial,
        rng=rng,
        deterministic_bias=drift,
    )

    step_bias = np.zeros((len(REGULAR_TIMES), len(SENSORS)), dtype=float)
    step_bias[REGULAR_TIMES >= 90.0, 0] = 3.0
    _write_monitor_case(
        case_id="M03_sensor_step_fault",
        group="fault_detection",
        purpose="detect an abrupt single-sensor offset before bias adaptation",
        controls=_short_pulses(REGULAR_TIMES),
        initial_temperature=np.array([55.0, 55.0, 55.0, 55.0]),
        rng=rng,
        deterministic_bias=step_bias,
        event_start=90.0,
    )

    missing = np.zeros((len(REGULAR_TIMES), len(SENSORS)), dtype=bool)
    missing[(REGULAR_TIMES >= 55.0) & (REGULAR_TIMES < 110.0), 1] = True
    missing[(REGULAR_TIMES >= 95.0) & (REGULAR_TIMES < 145.0), 3] = True
    _write_monitor_case(
        case_id="M04_missing_sensor_windows",
        group="missing_data",
        purpose="continue state estimation through staggered sensor outages",
        controls=_novel_recipe(REGULAR_TIMES),
        initial_temperature=nominal_initial,
        rng=rng,
        missing=missing,
    )

    extra_heat = np.zeros((len(REGULAR_TIMES), len(SENSORS)), dtype=float)
    heat_event = (REGULAR_TIMES >= 80.0) & (REGULAR_TIMES < 115.0)
    extra_heat[heat_event, 0] = 0.60
    extra_heat[heat_event, 1] = 0.30
    _write_monitor_case(
        case_id="M05_unmodeled_heat_load",
        group="disturbance_detection",
        purpose="detect physical heat input absent from commands and model",
        controls=constant_controls(25.0, 140.0, 75.0),
        initial_temperature=np.array([55.0, 55.0, 55.0, 55.0]),
        rng=rng,
        extra_heat=extra_heat,
        event_start=80.0,
    )


def main(*, seed: int = 42) -> None:
    generate_identification_data()
    generate_forecast_evaluation()
    generate_monitor_evaluation(seed)
    print("generated identification, forecast, and monitor benchmark cases")


if __name__ == "__main__":
    main()
