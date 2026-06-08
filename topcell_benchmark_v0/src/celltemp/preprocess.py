from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control import effective_control_series
from .data import case_control_series


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=float)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.std + self.mean


@dataclass
class Preprocessor:
    temp: Standardizer
    delta: Standardizer
    control: Standardizer
    control_min: np.ndarray
    control_max: np.ndarray
    temp_min: np.ndarray
    temp_max: np.ndarray
    control_step_max: np.ndarray
    time_end_max: float
    raw_control_min: np.ndarray
    raw_control_max: np.ndarray
    raw_control_step_max: np.ndarray

    def t(self, x: np.ndarray) -> np.ndarray:
        return self.temp.transform(x)

    def dt(self, x: np.ndarray) -> np.ndarray:
        return self.delta.transform(x)

    def u(self, x: np.ndarray) -> np.ndarray:
        return self.control.transform(x)

    def inv_t(self, z: np.ndarray) -> np.ndarray:
        return self.temp.inverse(z)

    def inv_dt(self, z: np.ndarray) -> np.ndarray:
        return self.delta.inverse(z)

    def control_out_of_range(self, u: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Range check for model controls, usually effective controls."""
        u = np.asarray(u, dtype=float)
        return (u < self.control_min - margin) | (u > self.control_max + margin)

    def raw_control_out_of_range(self, u: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Range check for physical recipe/log controls."""
        u = np.asarray(u, dtype=float)
        return (u < self.raw_control_min - margin) | (u > self.raw_control_max + margin)

    def temp_out_of_range(self, temp: np.ndarray, margin: float = 0.0) -> np.ndarray:
        temp = np.asarray(temp, dtype=float)
        return (temp < self.temp_min - margin) | (temp > self.temp_max + margin)


def fit_preprocessor(
    cases,
    sensor_cols: list[str],
    control_cols: list[str],
    feature_cfg: dict | None = None,
) -> Preprocessor:
    feature_cfg = feature_cfg or {}
    temps, deltas = [], []
    model_controls, model_steps = [], []
    raw_controls, raw_steps = [], []
    time_end_max = 0.0
    for c in cases:
        t = c.frame[sensor_cols].to_numpy(float)
        times = c.frame.iloc[:, 0].to_numpy(float)
        u_raw = case_control_series(c, control_cols)
        u_model = effective_control_series(times, u_raw, control_cols, feature_cfg)

        temps.append(t)
        deltas.append(np.diff(t, axis=0))
        model_controls.append(u_model)
        raw_controls.append(u_raw)
        if len(u_model) > 1:
            model_steps.append(np.abs(np.diff(u_model, axis=0)))
            raw_steps.append(np.abs(np.diff(u_raw, axis=0)))
        time_end_max = max(time_end_max, float(times[-1]))

    all_temps = np.vstack(temps)
    all_model_controls = np.vstack(model_controls)
    all_raw_controls = np.vstack(raw_controls)
    if model_steps:
        model_step_max = np.vstack(model_steps).max(axis=0)
        raw_step_max = np.vstack(raw_steps).max(axis=0)
    else:
        model_step_max = np.zeros(len(control_cols), dtype=float)
        raw_step_max = np.zeros(len(control_cols), dtype=float)

    return Preprocessor(
        temp=Standardizer.fit(all_temps),
        delta=Standardizer.fit(np.vstack(deltas)),
        control=Standardizer.fit(all_model_controls),
        control_min=all_model_controls.min(axis=0),
        control_max=all_model_controls.max(axis=0),
        temp_min=all_temps.min(axis=0),
        temp_max=all_temps.max(axis=0),
        control_step_max=model_step_max,
        time_end_max=float(time_end_max),
        raw_control_min=all_raw_controls.min(axis=0),
        raw_control_max=all_raw_controls.max(axis=0),
        raw_control_step_max=raw_step_max,
    )
