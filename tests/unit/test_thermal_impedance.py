from __future__ import annotations

import numpy as np
import pytest

from celltemp.analysis import power_step_thermal_impedance
from celltemp.workflows.analysis_impedance import resolve_thermal_impedance


def _isolated_step() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.arange(21, dtype=np.float64)
    temperature = 25.0 + np.minimum(np.maximum(time - 5.0, 0.0) * 2.0, 10.0)
    commands = np.zeros((len(time) - 1, 2), dtype=np.float64)
    commands[5, 0] = 5.0
    commands[6:, 0] = 10.0
    commands[:, 1] = 2.0
    return time, temperature[:, None], commands


def test_power_step_impedance_separates_transient_zth_and_steady_rth() -> None:
    time, temperature, commands = _isolated_step()

    rows, qualifications = power_step_thermal_impedance(
        time,
        temperature,
        ("part",),
        commands,
        ("power_command", "coolant"),
        case_id="step",
        power_control="power_command",
        transition_start_s=5.0,
        transition_end_s=6.0,
        heat_step_w=10.0,
        baseline_window_s=5.0,
        terminal_window_s=5.0,
        max_baseline_drift_k_per_s=0.0,
        max_baseline_std_k=0.0,
        max_terminal_zth_drift_k_per_w_s=0.0,
    )

    assert rows[0]["time_since_step_s"] == 0.0
    assert rows[-1]["thermal_impedance_k_per_w"] == pytest.approx(1.0)
    assert qualifications[0]["zth_qualified"] is True
    assert qualifications[0]["rth_qualified"] is True
    assert qualifications[0]["effective_rth_k_per_w"] == pytest.approx(1.0)


def test_power_step_impedance_rejects_a_nonisolated_case() -> None:
    time, temperature, commands = _isolated_step()
    commands[10:, 1] = 3.0

    rows, qualifications = power_step_thermal_impedance(
        time,
        temperature,
        ("part",),
        commands,
        ("power_command", "coolant"),
        case_id="combined_recipe",
        power_control="power_command",
        transition_start_s=5.0,
        transition_end_s=6.0,
        heat_step_w=10.0,
        baseline_window_s=5.0,
        terminal_window_s=5.0,
        max_baseline_drift_k_per_s=0.0,
        max_baseline_std_k=0.0,
        max_terminal_zth_drift_k_per_w_s=0.0,
    )

    assert rows == []
    assert qualifications[0]["zth_qualified"] is False
    assert qualifications[0]["zth_reason"] == "other_control_changed:coolant"


def test_thermal_impedance_config_requires_explicit_physical_limits() -> None:
    with pytest.raises(ValueError, match="max_baseline_std_k is required"):
        resolve_thermal_impedance(
            {
                "baseline_window_s": 5.0,
                "terminal_window_s": 5.0,
                "max_baseline_drift_k_per_s": 0.01,
                "max_terminal_zth_drift_k_per_w_s": 0.001,
                "steps": {
                    "step": {
                        "control": "power",
                        "transition_start_s": 5.0,
                        "transition_end_s": 5.0,
                        "heat_step_w": 10.0,
                    }
                },
            }
        )
