"""Observer settings shared by forecast and monitor entry points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from celltemp.engine import KalmanObserver, ThermalRCModel

DEFAULT_DISTURBANCE_PROCESS_STD = 0.02
DEFAULT_BIAS_PROCESS_STD = 0.005
DEFAULT_SENSOR_STD = 0.15
DEFAULT_INNOVATION_GATE_SIGMA = 4.0
DEFAULT_BIAS_REFERENCE: str | None = None

MONITOR_OBSERVER_DEFAULTS: dict[str, float | str | None] = {
    "disturbance_process_std": DEFAULT_DISTURBANCE_PROCESS_STD,
    "bias_process_std": DEFAULT_BIAS_PROCESS_STD,
    "sensor_std": DEFAULT_SENSOR_STD,
    "innovation_gate_sigma": DEFAULT_INNOVATION_GATE_SIGMA,
    "initial_temperature_std": 1.0,
    "initial_disturbance_std": 0.5,
    "initial_bias_std": 0.5,
    "bias_reference": DEFAULT_BIAS_REFERENCE,
}
FORECAST_OBSERVER_DEFAULTS: dict[str, float | str | None] = {
    **MONITOR_OBSERVER_DEFAULTS,
    "bias_process_std": 0.0,
    "initial_temperature_std": 100.0,
    "initial_disturbance_std": 0.0,
    "initial_bias_std": 0.0,
}
FORECAST_OBSERVER_OPTIONS = {
    "disturbance_process_std",
    "initial_temperature_std",
    "innovation_gate_sigma",
    "sensor_std",
}


def resolve_observer_settings(
    mode: str,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, float | str | None]:
    """Resolve one explicit observer profile for a deployment workflow."""
    if mode == "forecast":
        defaults = FORECAST_OBSERVER_DEFAULTS
        allowed = FORECAST_OBSERVER_OPTIONS
    elif mode == "monitor":
        defaults = MONITOR_OBSERVER_DEFAULTS
        allowed = set(defaults)
    else:
        raise ValueError(f"unknown observer mode: {mode}")
    supplied = dict(overrides or {})
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError(f"unknown {mode}.observer options: {unknown}")
    resolved = dict(defaults)
    for name, value in supplied.items():
        if name == "bias_reference":
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{mode}.observer.bias_reference must be a sensor name or null")
            resolved[name] = value
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{mode}.observer.{name} must be numeric")
            resolved[name] = float(value)
    return resolved


def build_observer(
    model: ThermalRCModel,
    settings: Mapping[str, float | str | None],
) -> KalmanObserver:
    """Construct an observer from resolved, serializable settings."""
    return KalmanObserver(
        model,
        disturbance_process_std=cast(float, settings["disturbance_process_std"]),
        bias_process_std=cast(float, settings["bias_process_std"]),
        sensor_std=cast(float, settings["sensor_std"]),
        innovation_gate_sigma=cast(float, settings["innovation_gate_sigma"]),
        initial_temperature_std=cast(float, settings["initial_temperature_std"]),
        initial_disturbance_std=cast(float, settings["initial_disturbance_std"]),
        initial_bias_std=cast(float, settings["initial_bias_std"]),
        bias_reference=cast(str | None, settings["bias_reference"]),
    )
