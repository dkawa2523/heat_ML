from __future__ import annotations

import numpy as np
import pandas as pd


def _bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def effective_control_series(
    times: np.ndarray,
    controls: np.ndarray,
    control_cols: list[str],
    feature_cfg: dict | None = None,
) -> np.ndarray:
    """Return control series used by the model.

    If features.use_effective_controls is enabled, selected control columns are
    passed through a first-order lag filter.  This is a compact way to represent
    heater/brine response delay without adding another model class.

        u_eff[k] = u_eff[k-1] + alpha * (u_raw[k] - u_eff[k-1])
        alpha = 1 - exp(-dt / tau)

    tau <= 0 means no lag for that control.
    """
    feature_cfg = feature_cfg or {}
    raw = np.asarray(controls, dtype=float)
    times = np.asarray(times, dtype=float)
    if raw.ndim != 2:
        raise ValueError("controls must be [time, n_controls]")
    if len(times) != len(raw):
        raise ValueError("times and controls length mismatch")
    if len(raw) == 0:
        return raw.copy()
    if not _bool(feature_cfg.get("use_effective_controls", False)):
        return raw.copy()

    tau_cfg = feature_cfg.get("control_lag_tau", {}) or {}
    tau = np.array([float(tau_cfg.get(c, 0.0)) for c in control_cols], dtype=float)
    eff = np.empty_like(raw, dtype=float)
    eff[0] = raw[0]
    for k in range(1, len(raw)):
        dt = max(float(times[k] - times[k - 1]), 0.0)
        eff[k] = raw[k]
        for j, tj in enumerate(tau):
            if tj > 0.0:
                alpha = 1.0 - np.exp(-dt / tj)
                eff[k, j] = eff[k - 1, j] + alpha * (raw[k, j] - eff[k - 1, j])
    return eff


def previous_interpolate(query_times: np.ndarray, base_times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Zero-order hold interpolation for schedule setpoints."""
    query_times = np.asarray(query_times, dtype=float)
    base_times = np.asarray(base_times, dtype=float)
    values = np.asarray(values, dtype=float)
    idx = np.searchsorted(base_times, query_times, side="right") - 1
    idx = np.clip(idx, 0, len(base_times) - 1)
    return values[idx]


def interpolate_table(
    table: pd.DataFrame,
    query_times: np.ndarray,
    cols: list[str],
    method: str = "previous",
) -> np.ndarray:
    """Interpolate schedule/log columns at query_times.

    Controls normally use method='previous' because recipes are often stepwise
    setpoints. Temperatures normally use method='linear'.
    """
    method = str(method or "previous").lower()
    t = table["time"].to_numpy(float)
    if method in {"previous", "hold", "zoh", "step"}:
        return np.column_stack([previous_interpolate(query_times, t, table[c].to_numpy(float)) for c in cols])
    if method == "linear":
        return np.column_stack([np.interp(query_times, t, table[c].to_numpy(float)) for c in cols])
    raise ValueError(f"unknown interpolation method: {method}; use previous or linear")


def resample_log_table(
    table: pd.DataFrame,
    sensor_cols: list[str],
    control_cols: list[str],
    dt: float,
    control_method: str = "previous",
    temperature_method: str = "linear",
) -> pd.DataFrame:
    """Resample measured monitor log to the model dt.

    Temperatures are usually linear-interpolated; control setpoints are usually
    held by previous value.
    """
    table = table.sort_values("time").reset_index(drop=True)
    t = table["time"].to_numpy(float)
    if len(t) < 2:
        raise ValueError("monitor log requires at least two time rows")
    query = np.arange(float(t[0]), float(t[-1]) + 0.5 * float(dt), float(dt))
    out = pd.DataFrame({"time": query})
    temps = interpolate_table(table, query, sensor_cols, method=temperature_method)
    controls = interpolate_table(table, query, control_cols, method=control_method)
    for i, s in enumerate(sensor_cols):
        out[s] = temps[:, i]
    for i, c in enumerate(control_cols):
        out[c] = controls[:, i]
    return out
