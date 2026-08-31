"""Purpose-driven COMSOL cases for identification and external evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Case:
    """One self-contained transient experiment."""

    case_id: str
    role: str
    group: str
    purpose: str
    time: np.ndarray
    chip_power: np.ndarray
    coolant_temperature: np.ndarray
    hidden_power: np.ndarray
    initial_temperature: float = 25.0


def _time(duration: float = 600.0, dt: float = 2.0) -> np.ndarray:
    return np.arange(0.0, duration + 0.5 * dt, dt, dtype=np.float64)


def _levels(
    time: np.ndarray,
    initial: float,
    changes: list[tuple[float, float]],
    transition: float = 4.0,
) -> np.ndarray:
    """Piecewise levels with short finite ramps at each command corner."""
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


def _case(
    case_id: str,
    role: str,
    group: str,
    purpose: str,
    time: np.ndarray,
    power: np.ndarray,
    coolant: np.ndarray,
    *,
    hidden: np.ndarray | None = None,
    initial_temperature: float = 25.0,
) -> Case:
    zeros = np.zeros_like(time)
    return Case(
        case_id=case_id,
        role=role,
        group=group,
        purpose=purpose,
        time=time,
        chip_power=power,
        coolant_temperature=coolant,
        hidden_power=zeros if hidden is None else hidden,
        initial_temperature=initial_temperature,
    )


def _training_cases() -> list[Case]:
    cases: list[Case] = []
    for index, target in enumerate((4.0, 6.0, 8.0, 10.0, 12.0), start=1):
        time = _time()
        cases.append(
            _case(
                f"T{index:02d}_power_step_{target:g}w",
                "train",
                "power_steps",
                f"Identify source-to-chip gain and thermal time constants at {target:g} W.",
                time,
                _levels(time, 0.0, [(60.0, target)]),
                np.full_like(time, 25.0),
            )
        )

    for index, target in enumerate((20.0, 15.0, 30.0, 35.0), start=6):
        time = _time(720.0)
        cases.append(
            _case(
                f"T{index:02d}_coolant_step_{target:g}c",
                "train",
                "coolant_steps",
                f"Separate the coolant boundary response using a {target:g} degC level.",
                time,
                _levels(time, 0.0, [(20.0, 8.0)]),
                _levels(time, 25.0, [(300.0, target)]),
            )
        )

    time = _time()
    cases.extend(
        [
            _case(
                "T10_power_ramp",
                "train",
                "ramps",
                "Identify distributed thermal lag without relying only on steps.",
                time,
                _ramps(time, [(0, 0), (80, 0), (300, 12), (480, 4), (600, 4)]),
                np.full_like(time, 25.0),
            ),
            _case(
                "T11_coolant_ramp",
                "train",
                "ramps",
                "Identify the boundary path under a continuous coolant-temperature sweep.",
                time,
                _levels(time, 0.0, [(20.0, 8.0)]),
                _ramps(time, [(0, 25), (180, 25), (360, 17), (540, 33), (600, 33)]),
            ),
            _case(
                "T12_power_multilevel",
                "train",
                "multilevel",
                "Excite chip storage and both conductive links at several power levels.",
                time,
                _levels(time, 0.0, [(40, 5), (180, 11), (320, 3), (460, 9)]),
                np.full_like(time, 25.0),
            ),
            _case(
                "T13_coolant_multilevel",
                "train",
                "multilevel",
                "Excite both convective paths while chip power remains independently fixed.",
                time,
                _levels(time, 0.0, [(20, 8)]),
                _levels(time, 25.0, [(160, 18), (300, 32), (440, 22)]),
            ),
            _case(
                "T14_combined_recipe",
                "train",
                "combined",
                "Disambiguate source and boundary effects under asynchronous changes.",
                time,
                _levels(time, 0.0, [(40, 6), (210, 11), (390, 4), (520, 9)]),
                _levels(time, 25.0, [(130, 20), (300, 29), (470, 24)]),
            ),
            _case(
                "T15_combined_ramps",
                "train",
                "combined",
                "Check superposition across simultaneous, continuously changing inputs.",
                time,
                _ramps(time, [(0, 0), (80, 0), (260, 10), (430, 5), (600, 12)]),
                _ramps(time, [(0, 25), (120, 25), (300, 19), (450, 31), (600, 23)]),
            ),
            _case(
                "T16_warm_start",
                "train",
                "initialization",
                "Prevent identification from depending on a single initial temperature.",
                time,
                _levels(time, 0.0, [(120, 7), (400, 3)]),
                np.full_like(time, 30.0),
                initial_temperature=40.0,
            ),
        ]
    )
    return cases


def _forecast_cases() -> list[Case]:
    time = _time()
    regular = [
        _case(
            "F01_power_interpolation",
            "forecast",
            "interpolation",
            "Interpolate a power level absent from the training steps.",
            time,
            _levels(time, 0.0, [(70, 7.0)]),
            np.full_like(time, 25.0),
        ),
        _case(
            "F02_joint_interpolation",
            "forecast",
            "interpolation",
            "Test superposition at unseen in-range power and coolant levels.",
            time,
            _levels(time, 0.0, [(60, 9.0), (360, 5.0)]),
            _levels(time, 25.0, [(220, 22.5), (440, 27.5)]),
        ),
        _case(
            "F03_power_extrapolation",
            "forecast",
            "extrapolation",
            "Test stable hot-side extrapolation beyond the 12 W training maximum.",
            time,
            _levels(time, 0.0, [(60, 16.0)]),
            np.full_like(time, 25.0),
        ),
        _case(
            "F04_coolant_extrapolation",
            "forecast",
            "extrapolation",
            "Test cold-side extrapolation below the 15 degC training minimum.",
            time,
            _levels(time, 0.0, [(20, 9.0)]),
            _levels(time, 25.0, [(260, 10.0)]),
        ),
        _case(
            "F05_short_pulses",
            "forecast",
            "dynamics",
            "Expose missing fast modes using repeated short power pulses.",
            time,
            _levels(
                time,
                0.0,
                [(80, 12), (100, 0), (180, 12), (200, 0), (280, 12), (300, 0)],
                transition=2.0,
            ),
            np.full_like(time, 25.0),
        ),
        _case(
            "F06_unseen_recipe",
            "forecast",
            "dynamics",
            "Test an unseen ordering and overlap of both commands.",
            time,
            _levels(time, 0.0, [(40, 5), (160, 12), (310, 2), (450, 10)]),
            _levels(time, 25.0, [(100, 31), (250, 18), (390, 27), (530, 21)]),
        ),
        _case(
            "F07_hot_initial_state",
            "forecast",
            "initialization",
            "Test cooling and reheating from a uniform state outside training starts.",
            time,
            _levels(time, 0.0, [(300, 6)]),
            np.full_like(time, 25.0),
            initial_temperature=50.0,
        ),
    ]

    increments = np.resize(np.asarray([1.0, 1.0, 2.0, 3.0, 2.0, 1.0]), 1000)
    variable_time = np.concatenate(([0.0], np.cumsum(increments)))
    variable_time = variable_time[variable_time < 600.0]
    variable_time = np.append(variable_time, 600.0)
    regular.append(
        _case(
            "F08_variable_sampling",
            "forecast",
            "sampling",
            "Verify direct variable-dt integration without resampling the CAE truth.",
            variable_time,
            _ramps(variable_time, [(0, 0), (60, 0), (240, 11), (420, 4), (600, 9)]),
            _levels(variable_time, 25.0, [(180, 20), (370, 30)]),
        )
    )
    return regular


def _monitor_truth_cases() -> list[Case]:
    time = _time()
    power = _levels(time, 0.0, [(30, 7), (190, 11), (350, 4), (500, 9)])
    coolant = _levels(time, 25.0, [(140, 20), (310, 29), (470, 23)])
    hidden = _levels(time, 0.0, [(280, 3), (380, 0)], transition=2.0)
    return [
        _case(
            "R01_monitor_reference",
            "monitor_truth",
            "monitor_reference",
            (
                "Provide one shared physical trajectory for noise, drift, fault, "
                "and missing-data tests."
            ),
            time,
            power,
            coolant,
        ),
        _case(
            "R02_uncommanded_heat",
            "monitor_truth",
            "physical_disturbance",
            "Add localized heat absent from the commanded input for disturbance detection.",
            time,
            power,
            coolant,
            hidden=hidden,
        ),
    ]


def all_cases() -> list[Case]:
    """Return the complete deterministic case set in execution order."""
    return [*_training_cases(), *_forecast_cases(), *_monitor_truth_cases()]
