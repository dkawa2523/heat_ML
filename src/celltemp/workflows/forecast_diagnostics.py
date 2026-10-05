"""Optional forecast diagnostics derived from saved predictions, one case at a time."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import pandas as pd
import torch

from celltemp.config import temperature_unit_label

if TYPE_CHECKING:
    from celltemp.artifact import ThermalArtifact
    from celltemp.engine import ThermalRCModel

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _energy_balance_frame(
    model: ThermalRCModel,
    *,
    case_id: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build one self-describing instantaneous heat-rate table in W."""
    with torch.no_grad():
        flow = model.heat_flow_breakdown(
            torch.tensor(
                frame[[f"node.{name}.temperature" for name in model.spec.node_names]].to_numpy(),
                dtype=model.capacity.dtype,
                device=model.capacity.device,
            ),
            torch.tensor(
                frame[
                    [f"control.{name}.effective" for name in model.spec.control_names]
                ].to_numpy(),
                dtype=model.capacity.dtype,
                device=model.capacity.device,
            ),
        )
    edge_rate = flow.edge_rate.cpu().numpy()
    internal = flow.internal_to_node.cpu().numpy()
    source = flow.source_to_node.sum(dim=-1).cpu().numpy()
    boundary = flow.boundary_to_node.sum(dim=-1).cpu().numpy()
    boundary_temperature = flow.boundary_temperature.cpu().numpy()
    storage = flow.storage_rate.cpu().numpy()
    residual = flow.balance_residual.cpu().numpy()
    frame = pd.DataFrame({"case_id": case_id, "time": frame["time"].to_numpy()})
    for index in range(len(model.spec.edges)):
        frame[f"edge.{index}.heat_w"] = edge_rate[:, index]
    for index, item in enumerate(model.spec.sources):
        frame[f"source.{item.name}.heat_w"] = source[:, index]
    for index, item in enumerate(model.spec.boundaries):
        frame[f"boundary.{item.name}.heat_w"] = boundary[:, index]
        frame[f"boundary.{item.name}.reservoir_temperature"] = boundary_temperature[:, index]
    for index, node in enumerate(model.spec.node_names):
        frame[f"node.{node}.internal_heat_w"] = internal[:, index]
        frame[f"node.{node}.storage_w"] = storage[:, index]
        frame[f"node.{node}.balance_residual_w"] = residual[:, index]
    frame["total.source_heat_w"] = source.sum(axis=1)
    frame["total.boundary_heat_w"] = boundary.sum(axis=1)
    frame["total.storage_w"] = storage.sum(axis=1)
    frame["total.balance_residual_w"] = residual.sum(axis=1)
    return frame


def _plot_energy_balance(frame: pd.DataFrame, target: Path) -> None:
    """Plot the signed external heat rates, storage, and numerical residual."""
    time = frame["time"].to_numpy(dtype=np.float64)
    source = frame["total.source_heat_w"].to_numpy(dtype=np.float64)
    boundary = frame["total.boundary_heat_w"].to_numpy(dtype=np.float64)
    storage = frame["total.storage_w"].to_numpy(dtype=np.float64)
    residual = frame["total.balance_residual_w"].to_numpy(dtype=np.float64)
    external = source + boundary

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, 7.5),
        sharex=True,
        constrained_layout=True,
        height_ratios=(2.0, 2.0, 1.2),
    )
    exchange_axis, storage_axis, residual_axis = axes
    exchange_axis.plot(time, source, color="#d95f02", linewidth=1.6, label="Source heat")
    exchange_axis.plot(
        time,
        boundary,
        color="#1b9e77",
        linewidth=1.6,
        label="Boundary heat",
    )
    exchange_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
    exchange_axis.set_ylabel("Heat into nodes [W]")
    exchange_axis.set_title(str(frame["case_id"].iat[0]))
    exchange_axis.grid(alpha=0.25)
    exchange_axis.legend(loc="best")

    storage_axis.plot(
        time,
        external,
        color="#7570b3",
        linewidth=1.8,
        label="Source + boundary",
    )
    storage_axis.plot(
        time,
        storage,
        color="#222222",
        linestyle="--",
        linewidth=1.5,
        label="Storage",
    )
    storage_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
    storage_axis.set_ylabel("Net heat rate [W]")
    storage_axis.grid(alpha=0.25)
    storage_axis.legend(loc="best")

    residual_axis.plot(time, residual, color="#e7298a", linewidth=1.4)
    residual_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
    residual_axis.set_ylabel("Residual [W]")
    residual_axis.set_xlabel("Time [s]")
    residual_axis.grid(alpha=0.25)

    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180)
    plt.close(figure)


def _plot_forecast_case(
    frame: pd.DataFrame, artifact: ThermalArtifact, case_id: str, target: Path
) -> None:
    """Plot the primary engineering view directly from the forecast table."""
    time = frame["time"].to_numpy(dtype=np.float64)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, 8.2),
        sharex=True,
        constrained_layout=True,
        height_ratios=(2.4, 1.2, 1.0),
    )
    temperature_axis, command_axis, span_axis = axes

    for sensor in artifact.sensor_names:
        temperature = frame[f"sensor.{sensor}.temperature"].to_numpy(dtype=np.float64)
        lower = frame[f"sensor.{sensor}.lower95"].to_numpy(dtype=np.float64)
        upper = frame[f"sensor.{sensor}.upper95"].to_numpy(dtype=np.float64)
        (line,) = temperature_axis.plot(time, temperature, linewidth=1.6, label=sensor)
        temperature_axis.fill_between(
            time,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.14,
            linewidth=0.0,
        )
    unit = temperature_unit_label(artifact.metadata.get("temperature_unit", "degC"))
    temperature_axis.set_ylabel(f"Temperature [{unit}]")
    temperature_axis.set_title("Predicted temperature with 95% latent-state intervals")
    temperature_axis.grid(alpha=0.25)
    temperature_axis.legend(loc="best", ncols=min(3, len(artifact.sensor_names)))

    for control, unit in zip(
        artifact.control_names, artifact.model.spec.control_units, strict=True
    ):
        label = control if unit is None else f"{control} [{unit}]"
        command_axis.step(
            time,
            frame[f"control.{control}.command"].to_numpy(dtype=np.float64),
            where="post",
            linewidth=1.35,
            label=label,
        )
    command_axis.set_ylabel("Command")
    command_axis.grid(alpha=0.25)
    if artifact.control_names:
        command_axis.legend(loc="best", ncols=min(3, len(artifact.control_names)))

    span_axis.plot(
        time,
        frame["sensor_span"].to_numpy(dtype=np.float64),
        color="#7b2cbf",
        linewidth=1.6,
    )
    span_axis.set_ylabel("Sensor span [K]")
    span_axis.set_xlabel("Time [s]")
    span_axis.grid(alpha=0.25)

    figure.suptitle(case_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180)
    plt.close(figure)


def write_forecast_diagnostics(target: Path, source: Path, artifact: ThermalArtifact) -> None:
    """Write heat-rate CSV and plots without recomputing the forecast."""
    summary = pd.read_csv(source / "forecast_summary.csv")
    for index, row in enumerate(summary.to_dict("records")):
        case_id = str(row["case_id"])
        frame = pd.read_csv(source / str(row["output"]))
        energy = _energy_balance_frame(artifact.model, case_id=case_id, frame=frame)
        energy.to_csv(
            target / "energy_balance.csv",
            index=False,
            mode="w" if index == 0 else "a",
            header=index == 0,
        )
        _plot_forecast_case(
            frame, artifact, case_id, target / "figures" / f"forecast_{case_id}.png"
        )
        _plot_energy_balance(energy, target / "figures" / f"energy_balance_{case_id}.png")
