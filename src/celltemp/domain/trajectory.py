"""A trajectory with unambiguous time and control-interval semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np


def _validate_arrays(
    time: np.ndarray,
    temperature: np.ndarray,
    commands: np.ndarray,
    mask: np.ndarray,
    n_sensors: int,
    n_controls: int,
) -> None:
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("time must be one-dimensional with at least two samples")
    if not np.isfinite(time).all() or not np.all(np.diff(time) > 0.0):
        raise ValueError("time must contain finite, strictly increasing values")
    if temperature.shape != (len(time), n_sensors):
        raise ValueError(
            f"temperature must have shape ({len(time)}, {n_sensors}), got {temperature.shape}"
        )
    expected_commands = (len(time) - 1, n_controls)
    if commands.shape != expected_commands:
        raise ValueError(
            f"commands must have interval shape {expected_commands}, got {commands.shape}"
        )
    if mask.shape != temperature.shape:
        raise ValueError(f"observation_mask must have shape {temperature.shape}, got {mask.shape}")
    if not np.isfinite(temperature[mask]).all():
        raise ValueError("observed temperatures must be finite")
    if not np.isfinite(commands).all():
        raise ValueError("commands must be finite")


@dataclass(frozen=True)
class Trajectory:
    """Temperature observations and the controls applied between observations.

    ``commands[k]`` is applied on ``[time[k], time[k + 1])`` and therefore drives
    the transition from ``temperature[k]`` to ``temperature[k + 1]``.  Keeping
    commands at interval length ``N - 1`` removes the left/right ambiguity that
    existed when temperatures and setpoints were stored in same-length tables.

    Missing temperatures are represented by ``observation_mask=False``.  Their
    numeric placeholders are ignored and may be NaN.  Array inputs are copied and
    stored read-only so external mutation cannot invalidate these invariants.
    """

    case_id: str
    time: np.ndarray
    temperature: np.ndarray
    commands: np.ndarray
    sensor_names: tuple[str, ...]
    control_names: tuple[str, ...]
    observation_mask: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = np.array(self.time, dtype=np.float64, copy=True)
        temperature = np.array(self.temperature, dtype=np.float64, copy=True)
        commands = np.array(self.commands, dtype=np.float64, copy=True)
        mask = (
            np.isfinite(temperature)
            if self.observation_mask is None
            else np.array(self.observation_mask, dtype=bool, copy=True)
        )

        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        _validate_arrays(
            time,
            temperature,
            commands,
            mask,
            len(self.sensor_names),
            len(self.control_names),
        )
        if len(set(self.sensor_names)) != len(self.sensor_names):
            raise ValueError("sensor_names must be unique")
        if len(set(self.control_names)) != len(self.control_names):
            raise ValueError("control_names must be unique")

        for array in (time, temperature, commands, mask):
            array.setflags(write=False)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "commands", commands)
        object.__setattr__(self, "observation_mask", mask)
        object.__setattr__(self, "sensor_names", tuple(self.sensor_names))
        object.__setattr__(self, "control_names", tuple(self.control_names))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def dt(self) -> np.ndarray:
        """One duration per control interval."""
        return np.diff(self.time)

    @property
    def mask(self) -> np.ndarray:
        """The validated non-optional observation mask."""
        return cast(np.ndarray, self.observation_mask)

    def require_layout(
        self,
        sensor_names: tuple[str, ...],
        control_names: tuple[str, ...],
    ) -> None:
        """Require the same ordered sensor/control layout as a thermal system."""
        if self.sensor_names != sensor_names:
            raise ValueError(
                f"trajectory sensors {self.sensor_names} do not match system sensors {sensor_names}"
            )
        if self.control_names != control_names:
            raise ValueError(
                f"trajectory controls {self.control_names} do not match system controls "
                f"{control_names}"
            )

    @classmethod
    def from_sampled_controls(
        cls,
        *,
        case_id: str,
        time: np.ndarray,
        temperature: np.ndarray,
        sampled_controls: np.ndarray,
        sensor_names: tuple[str, ...] | list[str],
        control_names: tuple[str, ...] | list[str],
        convention: str = "left",
        observation_mask: np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Trajectory:
        """Build a trajectory from a table that stores controls at every timestamp.

        ``left`` means the row at ``t[k]`` is active until ``t[k+1]``.  This is the
        normal zero-order-hold convention and the convention used by the bundled
        benchmark generator.  ``right`` is available only for explicitly documented
        exports where the row labels the interval ending at that timestamp.
        """
        sampled = np.asarray(sampled_controls, dtype=np.float64)
        n_time = len(np.asarray(time))
        expected = (n_time, len(control_names))
        if sampled.shape != expected:
            raise ValueError(f"sampled_controls must have shape {expected}, got {sampled.shape}")
        convention = convention.lower()
        if convention == "left":
            commands = sampled[:-1]
        elif convention == "right":
            commands = sampled[1:]
        else:
            raise ValueError("control convention must be 'left' or 'right'")
        return cls(
            case_id=case_id,
            time=time,
            temperature=temperature,
            commands=commands,
            sensor_names=tuple(sensor_names),
            control_names=tuple(control_names),
            observation_mask=observation_mask,
            metadata=metadata or {},
        )
