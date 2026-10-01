"""Shared thermal-engine cases used by focused unit-test modules."""

import torch

from celltemp.domain import ConstantLawSpec, EdgeSpec, ThermalSystemSpec
from celltemp.engine import ThermalRCModel

DTYPE = torch.float64


def conduction_model() -> ThermalRCModel:
    spec = ThermalSystemSpec(
        node_names=("a", "b"),
        heat_capacity=(2.0, 1.0),
        edges=(EdgeSpec("a", "b", ConstantLawSpec(0.5)),),
        actuators=(),
    )
    return ThermalRCModel(spec)
