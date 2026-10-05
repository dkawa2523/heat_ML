"""Array metrics; fitted-model inspection is loaded only when requested."""

from typing import TYPE_CHECKING, Any

from .applicability import coverage_status, range_coverage, temporal_coverage
from .impedance import power_step_thermal_impedance
from .metrics import (
    control_metrics,
    control_waveform_rows,
    mae,
    max_abs_error,
    response_metrics,
    rmse,
    sensor_response_rows,
    sensor_waveform_rows,
    thermal_case_metrics,
    uniformity_metrics,
    uniformity_trace,
)

if TYPE_CHECKING:
    from .model import representative_command, thermal_mode_rows, thermal_path_rows
from .prediction import (
    persistence_prediction,
    prediction_comparison_rows,
    prediction_error_metrics,
    prediction_sensor_rows,
    residual_dependence_rows,
    validate_prediction_arrays,
)

__all__ = [
    "control_metrics",
    "control_waveform_rows",
    "coverage_status",
    "mae",
    "max_abs_error",
    "persistence_prediction",
    "power_step_thermal_impedance",
    "prediction_comparison_rows",
    "prediction_error_metrics",
    "prediction_sensor_rows",
    "range_coverage",
    "representative_command",
    "residual_dependence_rows",
    "response_metrics",
    "rmse",
    "sensor_response_rows",
    "sensor_waveform_rows",
    "temporal_coverage",
    "thermal_case_metrics",
    "thermal_mode_rows",
    "thermal_path_rows",
    "uniformity_metrics",
    "uniformity_trace",
    "validate_prediction_arrays",
]


def __getattr__(name: str) -> Any:
    if name in {"representative_command", "thermal_mode_rows", "thermal_path_rows"}:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
