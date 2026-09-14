"""Reusable positive scalar response laws for thermal paths."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from celltemp.domain import (
    ConstantLawSpec,
    PositivePartLawSpec,
    ScalarLawSpec,
)

_CONSTANT = 0
_POSITIVE_PART = 1
_POWER_LAW = 2


class ScalarLawSet(nn.Module):
    """Evaluate and fit a heterogeneous collection of non-negative scalar laws.

    The same implementation is used for internal conductance, reservoir
    conductance, and source heat rate. Units belong to the owning thermal path;
    this module only maps physical inputs to a non-negative scalar.
    """

    kind: torch.Tensor
    control_index: torch.Tensor
    threshold: torch.Tensor
    reference: torch.Tensor
    offset_prior: torch.Tensor
    scale_prior: torch.Tensor
    exponent_prior: torch.Tensor
    offset_learn_mask: torch.Tensor
    scale_learn_mask: torch.Tensor
    exponent_learn_mask: torch.Tensor

    def __init__(
        self,
        specs: Sequence[ScalarLawSpec],
        control_names: Sequence[str],
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.specs = tuple(specs)
        self.n_controls = len(control_names)
        controls = {name: index for index, name in enumerate(control_names)}

        kind: list[int] = []
        control_index: list[int] = []
        threshold: list[float] = []
        reference: list[float] = []
        offset: list[float] = []
        scale: list[float] = []
        exponent: list[float] = []
        offset_learnable: list[bool] = []
        scale_learnable: list[bool] = []
        exponent_learnable: list[bool] = []

        for spec in self.specs:
            if isinstance(spec, ConstantLawSpec):
                kind.append(_CONSTANT)
                control_index.append(-1)
                threshold.append(0.0)
                reference.append(1.0)
                offset.append(spec.value)
                scale.append(0.0)
                exponent.append(1.0)
                offset_learnable.append(spec.learnable)
                scale_learnable.append(False)
                exponent_learnable.append(False)
            elif isinstance(spec, PositivePartLawSpec):
                kind.append(_POSITIVE_PART)
                control_index.append(controls[spec.control])
                threshold.append(spec.threshold)
                reference.append(1.0)
                offset.append(0.0)
                scale.append(spec.gain)
                exponent.append(1.0)
                offset_learnable.append(False)
                scale_learnable.append(spec.learnable)
                exponent_learnable.append(False)
            else:
                kind.append(_POWER_LAW)
                control_index.append(controls[spec.control])
                threshold.append(0.0)
                reference.append(spec.reference)
                offset.append(spec.offset)
                scale.append(spec.scale)
                exponent.append(spec.exponent)
                offset_learnable.append(spec.offset_learnable)
                scale_learnable.append(spec.scale_learnable)
                exponent_learnable.append(spec.exponent_learnable)

        self.register_buffer("kind", torch.tensor(kind, dtype=torch.long))
        self.register_buffer("control_index", torch.tensor(control_index, dtype=torch.long))
        self.register_buffer("threshold", torch.tensor(threshold, dtype=dtype))
        self.register_buffer("reference", torch.tensor(reference, dtype=dtype))
        self.register_buffer("offset_prior", torch.tensor(offset, dtype=dtype))
        self.register_buffer("scale_prior", torch.tensor(scale, dtype=dtype))
        self.register_buffer("exponent_prior", torch.tensor(exponent, dtype=dtype))
        self.register_buffer("offset_learn_mask", torch.tensor(offset_learnable, dtype=torch.bool))
        self.register_buffer("scale_learn_mask", torch.tensor(scale_learnable, dtype=torch.bool))
        self.register_buffer(
            "exponent_learn_mask", torch.tensor(exponent_learnable, dtype=torch.bool)
        )
        self.log_offset_multiplier = nn.Parameter(torch.zeros(len(self.specs), dtype=dtype))
        self.log_scale_multiplier = nn.Parameter(torch.zeros(len(self.specs), dtype=dtype))
        self.log_exponent_multiplier = nn.Parameter(torch.zeros(len(self.specs), dtype=dtype))

    def __len__(self) -> int:
        return len(self.specs)

    @staticmethod
    def _positive(
        prior: torch.Tensor,
        raw: torch.Tensor,
        learnable: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.any(learnable):
            return prior
        value = prior.clone()
        value[learnable] = prior[learnable] * torch.exp(raw[learnable])
        return value

    def offset(self) -> torch.Tensor:
        return self._positive(
            self.offset_prior,
            self.log_offset_multiplier,
            self.offset_learn_mask,
        )

    def scale(self) -> torch.Tensor:
        return self._positive(
            self.scale_prior,
            self.log_scale_multiplier,
            self.scale_learn_mask,
        )

    def exponent(self) -> torch.Tensor:
        return self._positive(
            self.exponent_prior,
            self.log_exponent_multiplier,
            self.exponent_learn_mask,
        )

    @property
    def dependent_mask(self) -> torch.Tensor:
        return self.control_index >= 0

    @property
    def positive_part_mask(self) -> torch.Tensor:
        return self.kind == _POSITIVE_PART

    def forward(self, actuator: torch.Tensor | None = None) -> torch.Tensor:
        """Return values with shape ``[..., n_laws]``."""
        dependent = self.dependent_mask
        if actuator is None:
            if torch.any(dependent):
                raise ValueError("actuator is required for an input-dependent scalar law")
            selected = self.offset_prior.new_zeros((len(self),))
        else:
            if actuator.shape[-1] != self.n_controls:
                raise ValueError("actuator has the wrong number of controls")
            if torch.any(dependent):
                safe_index = torch.clamp(self.control_index, min=0)
                selected = actuator[..., safe_index]
            else:
                selected = actuator.new_zeros((*actuator.shape[:-1], len(self)))

        response = torch.zeros_like(selected)
        positive = self.positive_part_mask
        if torch.any(positive):
            positive_response = torch.relu(selected[..., positive] - self.threshold[positive])
            response[..., positive] = positive_response

        power = self.kind == _POWER_LAW
        if torch.any(power):
            drive = torch.clamp(selected[..., power], min=0.0) / self.reference[power]
            safe_drive = torch.clamp(drive, min=torch.finfo(drive.dtype).tiny)
            power_response = torch.where(
                drive > 0.0,
                torch.exp(self.exponent()[power] * torch.log(safe_drive)),
                torch.zeros_like(drive),
            )
            response[..., power] = power_response
        return self.offset() + self.scale() * response

    def log_parameter_multipliers(self) -> tuple[torch.Tensor, ...]:
        return (
            self.log_offset_multiplier,
            self.log_scale_multiplier,
            self.log_exponent_multiplier,
        )

    def learnable_log_parameter_values(self) -> tuple[torch.Tensor, ...]:
        """Return only multiplier entries that affect a physical coefficient."""
        return (
            self.log_offset_multiplier[self.offset_learn_mask],
            self.log_scale_multiplier[self.scale_learn_mask],
            self.log_exponent_multiplier[self.exponent_learn_mask],
        )

    def fitted(self) -> list[dict[str, Any]]:
        """Describe fitted laws without exposing their runtime representation."""
        offsets = self.offset().detach().cpu().tolist()
        scales = self.scale().detach().cpu().tolist()
        exponents = self.exponent().detach().cpu().tolist()
        result: list[dict[str, Any]] = []
        for spec, offset, scale, exponent in zip(
            self.specs,
            offsets,
            scales,
            exponents,
            strict=True,
        ):
            if isinstance(spec, ConstantLawSpec):
                result.append({"type": "constant", "value": float(offset)})
            elif isinstance(spec, PositivePartLawSpec):
                result.append(
                    {
                        "type": "positive_part",
                        "control": spec.control,
                        "gain": float(scale),
                        "threshold": spec.threshold,
                    }
                )
            else:
                result.append(
                    {
                        "type": "power_law",
                        "control": spec.control,
                        "reference": spec.reference,
                        "offset": float(offset),
                        "scale": float(scale),
                        "exponent": float(exponent),
                    }
                )
        return result
