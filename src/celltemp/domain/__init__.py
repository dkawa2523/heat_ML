"""Domain objects shared by IO, simulation, learning, and workflows."""

from .topology import ActuatorSpec, BoundarySpec, EdgeSpec, SourceSpec, ThermalSystemSpec
from .trajectory import Trajectory

__all__ = [
    "ActuatorSpec",
    "BoundarySpec",
    "EdgeSpec",
    "SourceSpec",
    "ThermalSystemSpec",
    "Trajectory",
]
