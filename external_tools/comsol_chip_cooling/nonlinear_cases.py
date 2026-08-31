"""Compact nonlinear COMSOL case set for flow and radiation adequacy tests."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class NonlinearCase:
    """One self-contained transient conjugate heat-transfer experiment."""

    case_id: str
    role: str
    group: str
    purpose: str
    fidelity: str
    time: np.ndarray
    chip_power: np.ndarray
    coolant_temperature: np.ndarray
    inlet_air_velocity: np.ndarray
    hidden_power: np.ndarray
    effective_air_velocity: np.ndarray

    @property
    def radiation_enabled(self) -> bool:
        return self.fidelity == "conjugate_laminar_radiation"


def _time(duration: float = 900.0, dt: float = 10.0) -> np.ndarray:
    return np.arange(0.0, duration + 0.5 * dt, dt, dtype=np.float64)


def _levels(
    time: np.ndarray,
    initial: float,
    changes: list[tuple[float, float]],
    *,
    transition: float = 10.0,
) -> np.ndarray:
    """Piecewise levels connected by short finite ramps."""
    knot_time = [float(time[0])]
    knot_value = [float(initial)]
    previous = float(initial)
    for start, target in changes:
        knot_time.extend((float(start), float(start + transition)))
        knot_value.extend((previous, float(target)))
        previous = float(target)
    knot_time.append(float(time[-1]))
    knot_value.append(previous)
    return np.interp(time, knot_time, knot_value)


def _ramps(time: np.ndarray, knots: list[tuple[float, float]]) -> np.ndarray:
    return np.interp(
        time,
        np.asarray([item[0] for item in knots], dtype=np.float64),
        np.asarray([item[1] for item in knots], dtype=np.float64),
    )


def _constant(time: np.ndarray, value: float) -> np.ndarray:
    return np.full_like(time, value, dtype=np.float64)


def _case(
    case_id: str,
    role: str,
    group: str,
    purpose: str,
    time: np.ndarray,
    power: np.ndarray,
    coolant: np.ndarray,
    velocity: np.ndarray,
    *,
    hidden_power: np.ndarray | None = None,
    effective_velocity: np.ndarray | None = None,
    fidelity: str = "conjugate_laminar",
) -> NonlinearCase:
    zeros = np.zeros_like(time)
    return NonlinearCase(
        case_id=case_id,
        role=role,
        group=group,
        purpose=purpose,
        fidelity=fidelity,
        time=time,
        chip_power=power,
        coolant_temperature=coolant,
        inlet_air_velocity=velocity,
        hidden_power=zeros if hidden_power is None else hidden_power,
        effective_air_velocity=velocity if effective_velocity is None else effective_velocity,
    )


def _training_cases() -> list[NonlinearCase]:
    cases: list[NonlinearCase] = []
    time = _time()

    cases.extend(
        [
            _case(
                "NT01_power_levels",
                "train",
                "power",
                "Identify level-dependent thermal resistance and time constants.",
                time,
                _levels(time, 0.0, [(60, 4), (300, 8), (600, 12)]),
                _constant(time, 25),
                _constant(time, 0.10),
            ),
            _case(
                "NT02_power_reverse",
                "train",
                "power",
                "Separate heating and cooling dynamics after a high-power state.",
                time,
                _levels(time, 0.0, [(60, 12), (320, 3), (620, 10)]),
                _constant(time, 25),
                _constant(time, 0.10),
            ),
            _case(
                "NT03_inlet_temperature_levels",
                "train",
                "inlet_temperature",
                "Identify air-temperature-level dependence at fixed flow.",
                time,
                _levels(time, 0.0, [(30, 8)]),
                _levels(time, 25.0, [(300, 15), (600, 35)]),
                _constant(time, 0.10),
            ),
            _case(
                "NT04_air_velocity_levels",
                "train",
                "air_velocity",
                "Identify cooling-conductance changes caused by inlet velocity.",
                time,
                _levels(time, 0.0, [(30, 8)]),
                _constant(time, 25),
                _levels(time, 0.10, [(300, 0.05), (600, 0.20)], transition=30),
            ),
            _case(
                "NT05_power_ramp",
                "train",
                "ramps",
                "Identify nonlinear thermal response without relying on steps.",
                time,
                _ramps(time, [(0, 0), (60, 0), (420, 12), (780, 4), (900, 4)]),
                _constant(time, 25),
                _constant(time, 0.10),
            ),
            _case(
                "NT06_combined_levels",
                "train",
                "combined",
                "Disambiguate asynchronous changes in all three inputs.",
                time,
                _levels(time, 0.0, [(60, 6), (330, 12), (660, 4)]),
                _levels(time, 25.0, [(210, 18), (510, 32)]),
                _levels(time, 0.10, [(420, 0.20), (720, 0.075)], transition=30),
            ),
            _case(
                "NT07_combined_ramps",
                "train",
                "combined",
                "Cover the interior operating region with simultaneous smooth inputs.",
                time,
                _ramps(time, [(0, 0), (120, 0), (420, 10), (650, 5), (900, 12)]),
                _ramps(time, [(0, 25), (180, 25), (420, 18), (680, 32), (900, 24)]),
                _ramps(time, [(0, 0.10), (240, 0.10), (480, 0.18), (720, 0.06), (900, 0.12)]),
            ),
            _case(
                "NT08_hot_low_flow",
                "train",
                "operating_corner",
                "Include the hot, low-flow edge of the intended training domain.",
                time,
                _levels(time, 0.0, [(60, 10), (600, 4)]),
                _constant(time, 35),
                _constant(time, 0.05),
            ),
            _case(
                "NT09_cold_high_flow",
                "train",
                "operating_corner",
                "Include the cold, high-flow edge of the intended training domain.",
                time,
                _levels(time, 0.0, [(60, 12), (600, 6)]),
                _constant(time, 15),
                _constant(time, 0.20),
            ),
            _case(
                "NT10_stationary_hot_start",
                "train",
                "initialization",
                "Start from a nonuniform stationary powered temperature field.",
                time,
                _levels(time, 10.0, [(120, 2), (650, 9)]),
                _levels(time, 30.0, [(300, 20)]),
                _levels(time, 0.10, [(450, 0.20)], transition=30),
            ),
        ]
    )
    return cases


def _forecast_cases() -> list[NonlinearCase]:
    time = _time()
    cases = [
        _case(
            "NF01_fixed_flow_interpolation",
            "forecast",
            "interpolation",
            "Interpolate power and inlet temperature at the nominal velocity.",
            time,
            _levels(time, 0.0, [(80, 7), (600, 9)]),
            _levels(time, 25.0, [(300, 22.5)]),
            _constant(time, 0.10),
        ),
        _case(
            "NF02_unseen_three_input_recipe",
            "forecast",
            "combined",
            "Test an unseen ordering and overlap of all three inputs.",
            time,
            _levels(time, 0.0, [(60, 5), (240, 11), (510, 3), (720, 9)]),
            _levels(time, 25.0, [(180, 31), (450, 17), (690, 27)]),
            _levels(time, 0.10, [(330, 0.15), (600, 0.075)], transition=30),
        ),
        _case(
            "NF03_power_extrapolation",
            "forecast",
            "extrapolation",
            "Expose hot-side model error above the 12 W training limit.",
            time,
            _levels(time, 0.0, [(60, 16)]),
            _constant(time, 25),
            _constant(time, 0.10),
        ),
        _case(
            "NF04_hot_low_flow_corner",
            "forecast",
            "extrapolation",
            "Combine hot inlet air and low flow at high power.",
            time,
            _levels(time, 0.0, [(60, 12)]),
            _constant(time, 40),
            _constant(time, 0.05),
        ),
        _case(
            "NF05_cold_inlet_extrapolation",
            "forecast",
            "extrapolation",
            "Extrapolate below the inlet-temperature training range.",
            time,
            _levels(time, 0.0, [(30, 9)]),
            _levels(time, 25.0, [(300, 10)]),
            _constant(time, 0.10),
        ),
        _case(
            "NF06_air_velocity_interpolation",
            "forecast",
            "air_velocity",
            "Interpolate unseen cooling velocities inside the training range.",
            time,
            _levels(time, 0.0, [(30, 9)]),
            _constant(time, 25),
            _levels(time, 0.10, [(260, 0.075), (560, 0.15)], transition=30),
        ),
        _case(
            "NF07_air_velocity_extrapolation",
            "forecast",
            "air_velocity",
            "Extrapolate to 0.30 m/s while remaining in the laminar regime.",
            time,
            _levels(time, 0.0, [(60, 12)]),
            _constant(time, 25),
            _levels(time, 0.10, [(300, 0.30)], transition=30),
        ),
        _case(
            "NF09_stationary_hot_start",
            "forecast",
            "initialization",
            "Forecast from an unseen nonuniform hot stationary field.",
            time,
            _levels(time, 12.0, [(120, 3), (520, 10)]),
            _levels(time, 35.0, [(120, 25)]),
            _levels(time, 0.05, [(120, 0.10)], transition=30),
        ),
    ]

    pulse_time = _time(600.0, 1.0)
    cases.insert(
        7,
        _case(
            "NF08_short_power_pulses",
            "forecast",
            "dynamics",
            "Expose fast modes with repeated 20-second, 15 W pulses.",
            pulse_time,
            _levels(
                pulse_time,
                0.0,
                [(80, 15), (100, 0), (200, 15), (220, 0), (320, 15), (340, 0)],
                transition=1,
            ),
            _constant(pulse_time, 25),
            _constant(pulse_time, 0.10),
        ),
    )
    return cases


def _monitor_cases() -> list[NonlinearCase]:
    time = _time()
    power = _levels(time, 0.0, [(40, 7), (300, 11), (600, 4)])
    coolant = _levels(time, 25.0, [(200, 20), (500, 30)])
    velocity = _levels(time, 0.10, [(400, 0.20), (700, 0.075)], transition=30)
    return [
        _case(
            "NM01_nominal_nonlinear",
            "monitor",
            "monitoring",
            "Measure false alerts during normal nonlinear operation.",
            time,
            power,
            coolant,
            velocity,
        ),
        _case(
            "NM02_uncommanded_heat",
            "monitor",
            "monitoring",
            "Detect a real 3 W heat load absent from the public command.",
            time,
            power,
            coolant,
            velocity,
            hidden_power=_levels(time, 0.0, [(420, 3), (520, 0)]),
        ),
        _case(
            "NM03_uncommanded_cooling_loss",
            "monitor",
            "monitoring",
            "Detect a temporary airflow loss absent from the public velocity command.",
            time,
            _levels(time, 0.0, [(40, 8)]),
            _constant(time, 25),
            _levels(time, 0.10, [(180, 0.15)], transition=30),
            effective_velocity=_levels(
                time,
                0.10,
                [(180, 0.15), (420, 0.05), (620, 0.15)],
                transition=30,
            ),
        ),
    ]


def _radiation_cases(base: list[NonlinearCase]) -> list[NonlinearCase]:
    by_id = {case.case_id: case for case in base}
    definitions = [
        (
            "NR01_power_levels",
            "NT01_power_levels",
            "Measure radiation contribution by power level.",
        ),
        (
            "NR02_power_ramp",
            "NT05_power_ramp",
            "Measure radiation along a continuous temperature sweep.",
        ),
        (
            "NR03_combined_levels",
            "NT06_combined_levels",
            "Measure radiation during asynchronous three-input operation.",
        ),
        (
            "NR04_hot_low_flow_corner",
            "NF04_hot_low_flow_corner",
            "Measure radiation where convection is weak and temperatures are high.",
        ),
        (
            "NR05_stationary_hot_start",
            "NF09_stationary_hot_start",
            "Measure radiation during cooldown from a nonuniform hot field.",
        ),
    ]
    return [
        replace(
            by_id[source_id],
            case_id=case_id,
            role="model_gap",
            group="radiation",
            purpose=purpose,
            fidelity="conjugate_laminar_radiation",
        )
        for case_id, source_id, purpose in definitions
    ]


def all_nonlinear_cases() -> list[NonlinearCase]:
    """Return all physical solves in deterministic execution order."""
    base = [*_training_cases(), *_forecast_cases(), *_monitor_cases()]
    return [*base, *_radiation_cases(base)]


def mesh_convergence_cases() -> list[NonlinearCase]:
    """Return compact steady operating points used only for mesh qualification."""
    time = np.array([0.0, 1.0], dtype=np.float64)
    return [
        _case(
            "MC01_nominal",
            "mesh_convergence",
            "mesh_convergence",
            "Qualify heat transfer and pressure loss at rated power and nominal flow.",
            time,
            _constant(time, 12.0),
            _constant(time, 25.0),
            _constant(time, 0.10),
        ),
        _case(
            "MC02_hot_low_flow_radiation",
            "mesh_convergence",
            "mesh_convergence",
            "Qualify the hot, low-flow corner including surface-to-surface radiation.",
            time,
            _constant(time, 12.0),
            _constant(time, 40.0),
            _constant(time, 0.05),
            fidelity="conjugate_laminar_radiation",
        ),
    ]


def high_fidelity_cases() -> list[NonlinearCase]:
    """Return one information-dense transient and its radiation-only pair."""
    time = _time(220.0, 20.0)
    base = _case(
        "HV01_composite_conjugate",
        "forecast",
        "high_fidelity_validation",
        "Validate sequential power, airflow, and inlet-temperature ramps on the qualified mesh.",
        time,
        _ramps(time, [(0, 0), (40, 0), (100, 12), (220, 12)]),
        _ramps(time, [(0, 25), (140, 25), (200, 35), (220, 35)]),
        _ramps(time, [(0, 0.10), (100, 0.10), (160, 0.05), (220, 0.05)]),
    )
    radiation = replace(
        base,
        case_id="HV02_composite_radiation",
        role="model_gap",
        purpose="Measure radiation-only model gap for the identical composite transient.",
        fidelity="conjugate_laminar_radiation",
    )
    return [base, radiation]
