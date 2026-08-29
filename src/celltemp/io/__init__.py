"""Input/output adapters for generic thermal trajectories and system definitions."""

from .system import load_system_spec, save_system_spec, system_spec_from_mapping
from .trajectory import (
    load_split_assignments,
    load_trajectories,
    trajectory_from_frame,
)

__all__ = [
    "load_split_assignments",
    "load_system_spec",
    "load_trajectories",
    "save_system_spec",
    "system_spec_from_mapping",
    "trajectory_from_frame",
]
