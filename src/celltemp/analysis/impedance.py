"""Thermal impedance from explicitly identified, isolated heat-rate steps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_MIN_WINDOW_SAMPLES = 5
_COMMAND_RTOL = 1e-9
_COMMAND_ATOL = 1e-12


@dataclass(frozen=True)
class _StepWindows:
    command_before: float
    command_after: float
    baseline: np.ndarray
    response: np.ndarray
    terminal: np.ndarray
    common_reasons: tuple[str, ...]


def _linear_slope(time: np.ndarray, values: np.ndarray) -> float:
    centered_time = time - np.mean(time)
    denominator = float(centered_time @ centered_time)
    if denominator == 0.0:
        return float("nan")
    return float(centered_time @ (values - np.mean(values)) / denominator)


def _constant_level(values: np.ndarray) -> tuple[float, bool]:
    if len(values) == 0:
        return float("nan"), False
    level = float(np.mean(values))
    constant = bool(np.allclose(values, level, rtol=_COMMAND_RTOL, atol=_COMMAND_ATOL))
    return level, constant


def _joined(reasons: list[str]) -> str:
    return ";".join(dict.fromkeys(reasons))


def _validate_step(
    time: np.ndarray,
    control_names: Sequence[str],
    *,
    case_id: str,
    power_control: str,
    transition_start_s: float,
    transition_end_s: float,
    heat_step_w: float,
) -> None:
    if power_control not in control_names:
        raise ValueError(f"{case_id}: unknown power control {power_control!r}")
    if transition_start_s > transition_end_s:
        raise ValueError(f"{case_id}: transition_start_s must not exceed transition_end_s")
    if transition_start_s < time[0] or transition_end_s >= time[-1]:
        raise ValueError(f"{case_id}: power-step transition must lie inside the trajectory")
    if not np.isfinite(heat_step_w) or heat_step_w == 0.0:
        raise ValueError(f"{case_id}: heat_step_w must be finite and nonzero")


def _other_control_reasons(
    commands: np.ndarray,
    control_names: Sequence[str],
    selected_index: int,
    window: np.ndarray,
) -> list[str]:
    reasons: list[str] = []
    for index, name in enumerate(control_names):
        if index != selected_index and not _constant_level(commands[window, index])[1]:
            reasons.append(f"other_control_changed:{name}")
    return reasons


def _step_windows(
    time: np.ndarray,
    commands: np.ndarray,
    control_names: Sequence[str],
    *,
    power_control: str,
    transition_start_s: float,
    transition_end_s: float,
    baseline_window_s: float,
    terminal_window_s: float,
) -> _StepWindows:
    control_index = tuple(control_names).index(power_control)
    command_time = time[:-1]
    baseline_start_s = transition_start_s - baseline_window_s
    pre_command = (command_time >= baseline_start_s) & (command_time < transition_start_s)
    post_command = command_time >= transition_end_s
    before, before_constant = _constant_level(commands[pre_command, control_index])
    after, after_constant = _constant_level(commands[post_command, control_index])

    reasons: list[str] = []
    if int(np.count_nonzero(pre_command)) < _MIN_WINDOW_SAMPLES:
        reasons.append("insufficient_pre_step_command_samples")
    if int(np.count_nonzero(post_command)) < _MIN_WINDOW_SAMPLES:
        reasons.append("insufficient_post_step_command_samples")
    if not before_constant:
        reasons.append("power_control_not_constant_before_transition")
    if not after_constant:
        reasons.append("power_control_not_constant_after_transition")
    if (
        before_constant
        and after_constant
        and np.isclose(before, after, rtol=_COMMAND_RTOL, atol=_COMMAND_ATOL)
    ):
        reasons.append("power_control_has_no_step")
    reasons.extend(
        _other_control_reasons(
            commands,
            control_names,
            control_index,
            command_time >= baseline_start_s,
        )
    )

    terminal_start_s = max(transition_end_s, float(time[-1]) - terminal_window_s)
    return _StepWindows(
        command_before=before,
        command_after=after,
        baseline=(time >= baseline_start_s) & (time <= transition_start_s),
        response=time >= transition_start_s,
        terminal=time >= terminal_start_s,
        common_reasons=tuple(reasons),
    )


def _baseline_evidence(
    time: np.ndarray,
    temperature: np.ndarray,
    mask: np.ndarray,
    *,
    max_drift: float,
    max_std: float,
) -> tuple[int, float, float, float, list[str]]:
    finite = mask & np.isfinite(temperature)
    count = int(np.count_nonzero(finite))
    if count < _MIN_WINDOW_SAMPLES:
        return (
            count,
            float("nan"),
            float("nan"),
            float("nan"),
            ["insufficient_baseline_temperature_samples"],
        )
    values = temperature[finite]
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    drift = _linear_slope(time[finite], values)
    reasons: list[str] = []
    if standard_deviation > max_std:
        reasons.append("baseline_noise_above_limit")
    if abs(drift) > max_drift:
        reasons.append("baseline_drift_above_limit")
    return count, mean, standard_deviation, drift, reasons


def _response_rows(
    time: np.ndarray,
    temperature: np.ndarray,
    mask: np.ndarray,
    *,
    case_id: str,
    sensor: str,
    baseline_temperature: float,
    transition_start_s: float,
    heat_step_w: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_index in np.flatnonzero(mask & np.isfinite(temperature)):
        delta_temperature = float(temperature[sample_index] - baseline_temperature)
        rows.append(
            {
                "case_id": case_id,
                "sensor": sensor,
                "time_s": float(time[sample_index]),
                "time_since_step_s": float(time[sample_index] - transition_start_s),
                "delta_temperature_k": delta_temperature,
                "heat_step_w": heat_step_w,
                "thermal_impedance_k_per_w": delta_temperature / heat_step_w,
            }
        )
    return rows


def _terminal_evidence(
    time: np.ndarray,
    temperature: np.ndarray,
    mask: np.ndarray,
    *,
    baseline_temperature: float,
    heat_step_w: float,
    max_drift: float,
) -> tuple[int, float, float, list[str]]:
    finite = mask & np.isfinite(temperature)
    count = int(np.count_nonzero(finite))
    if count < _MIN_WINDOW_SAMPLES:
        return count, float("nan"), float("nan"), ["insufficient_terminal_temperature_samples"]
    values = (temperature[finite] - baseline_temperature) / heat_step_w
    mean = float(np.mean(values))
    drift = _linear_slope(time[finite], values)
    reasons = ["terminal_zth_drift_above_limit"] if abs(drift) > max_drift else []
    return count, mean, drift, reasons


def power_step_thermal_impedance(
    time: np.ndarray,
    temperature: np.ndarray,
    sensor_names: Sequence[str],
    commands: np.ndarray,
    control_names: Sequence[str],
    *,
    case_id: str,
    power_control: str,
    transition_start_s: float,
    transition_end_s: float,
    heat_step_w: float,
    baseline_window_s: float,
    terminal_window_s: float,
    max_baseline_drift_k_per_s: float,
    max_baseline_std_k: float,
    max_terminal_zth_drift_k_per_w_s: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return transient Zth rows and per-sensor qualification evidence.

    ``heat_step_w`` is the independently known change in absorbed heat rate. It is
    intentionally not inferred from the selected command because a command may be
    dimensionless or may pass through a calibrated source law.
    """
    _validate_step(
        time,
        control_names,
        case_id=case_id,
        power_control=power_control,
        transition_start_s=transition_start_s,
        transition_end_s=transition_end_s,
        heat_step_w=heat_step_w,
    )
    windows = _step_windows(
        time,
        commands,
        control_names,
        power_control=power_control,
        transition_start_s=transition_start_s,
        transition_end_s=transition_end_s,
        baseline_window_s=baseline_window_s,
        terminal_window_s=terminal_window_s,
    )

    impedance_rows: list[dict[str, object]] = []
    qualification_rows: list[dict[str, object]] = []
    for sensor_index, sensor in enumerate(sensor_names):
        sensor_temperature = temperature[:, sensor_index]
        response_finite = windows.response & np.isfinite(sensor_temperature)
        response_count = int(np.count_nonzero(response_finite))
        baseline_count, baseline_temperature, baseline_std_k, baseline_drift, reasons = (
            _baseline_evidence(
                time,
                sensor_temperature,
                windows.baseline,
                max_drift=max_baseline_drift_k_per_s,
                max_std=max_baseline_std_k,
            )
        )
        zth_reasons = [*windows.common_reasons, *reasons]
        if response_count < _MIN_WINDOW_SAMPLES:
            zth_reasons.append("insufficient_response_temperature_samples")
        zth_qualified = not zth_reasons
        if zth_qualified:
            impedance_rows.extend(
                _response_rows(
                    time,
                    sensor_temperature,
                    windows.response,
                    case_id=case_id,
                    sensor=sensor,
                    baseline_temperature=baseline_temperature,
                    transition_start_s=transition_start_s,
                    heat_step_w=heat_step_w,
                )
            )
            terminal_count, terminal_zth, terminal_zth_drift, rth_reasons = _terminal_evidence(
                time,
                sensor_temperature,
                windows.terminal,
                baseline_temperature=baseline_temperature,
                heat_step_w=heat_step_w,
                max_drift=max_terminal_zth_drift_k_per_w_s,
            )
        else:
            terminal_count = int(
                np.count_nonzero(windows.terminal & np.isfinite(sensor_temperature))
            )
            terminal_zth = float("nan")
            terminal_zth_drift = float("nan")
            rth_reasons = ["zth_not_qualified"]
        rth_qualified = not rth_reasons

        qualification_rows.append(
            {
                "case_id": case_id,
                "sensor": sensor,
                "power_control": power_control,
                "transition_start_s": transition_start_s,
                "transition_end_s": transition_end_s,
                "command_before": windows.command_before,
                "command_after": windows.command_after,
                "heat_step_w": heat_step_w,
                "minimum_window_samples": _MIN_WINDOW_SAMPLES,
                "baseline_window_s": baseline_window_s,
                "terminal_window_s": terminal_window_s,
                "max_baseline_drift_k_per_s": max_baseline_drift_k_per_s,
                "max_baseline_std_k": max_baseline_std_k,
                "max_terminal_zth_drift_k_per_w_s": max_terminal_zth_drift_k_per_w_s,
                "baseline_samples": baseline_count,
                "response_samples": response_count,
                "terminal_samples": terminal_count,
                "baseline_temperature": baseline_temperature,
                "baseline_std_k": baseline_std_k,
                "baseline_drift_k_per_s": baseline_drift,
                "terminal_zth_k_per_w": terminal_zth,
                "terminal_zth_drift_k_per_w_s": terminal_zth_drift,
                "zth_qualified": zth_qualified,
                "zth_reason": _joined(zth_reasons),
                "rth_qualified": rth_qualified,
                "rth_reason": _joined(rth_reasons),
                "effective_rth_k_per_w": terminal_zth if rth_qualified else None,
            }
        )

    return impedance_rows, qualification_rows
