# ruff: noqa: RUF001
"""Generate the static problem-setting figures used by the benchmark documentation.

This module draws explanatory schematics only.  It does not run a solver, train a
model, read benchmark metrics, or build a report.  The numeric evaluation remains
owned by each benchmark workflow and its JSON/CSV outputs.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUTPUT_DIR = Path(__file__).resolve().parent

INK = "#172033"
MUTED = "#5D6878"
LIGHT = "#F4F7FA"
GRID = "#D7DEE8"
HEAT = "#E4572E"
HEAT_LIGHT = "#F8C4B2"
HEATER = "#F4A62A"
COOL = "#247BA0"
COOL_LIGHT = "#BFE3F1"
SENSOR = "#00A6A6"
MODEL = "#4059AD"
MODEL_LIGHT = "#DCE3FA"
RADIATION = "#8E5EA2"
SOLID = "#697386"
SUCCESS = "#2A9D68"
WARN = "#D88412"


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["BIZ UDGothic", "Meiryo", "Yu Gothic", "sans-serif"],
            "font.size": 10,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
        }
    )


def _canvas(title: str, subtitle: str) -> tuple[Figure, Axes]:
    figure, axis = plt.subplots(figsize=(16, 9), facecolor="white")
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 9)
    axis.axis("off")
    axis.text(0.45, 8.55, title, fontsize=23, fontweight="bold", color=INK, va="top")
    axis.text(0.47, 8.12, subtitle, fontsize=11.5, color=MUTED, va="top")
    return figure, axis


def _rounded_box(
    axis: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str = "white",
    edgecolor: str = GRID,
    linewidth: float = 1.4,
    radius: float = 0.16,
    linestyle: str = "-",
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def _arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    width: float = 2.0,
    style: str = "-|>",
    connection: str = "arc3",
    linestyle: str = "-",
    alpha: float = 1.0,
    zorder: int = 4,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=width,
            color=color,
            connectionstyle=connection,
            linestyle=linestyle,
            alpha=alpha,
            shrinkA=2,
            shrinkB=2,
            zorder=zorder,
        )
    )


def _sensor(axis: Axes, xy: tuple[float, float], label: str, *, align: str = "center") -> None:
    axis.add_patch(Circle(xy, 0.105, facecolor="white", edgecolor=SENSOR, linewidth=2.6, zorder=7))
    axis.add_patch(Circle(xy, 0.032, facecolor=SENSOR, edgecolor="none", zorder=8))
    dx = 0 if align == "center" else (0.18 if align == "left" else -0.18)
    ha = "center" if align == "center" else align
    axis.text(xy[0] + dx, xy[1] + 0.20, label, fontsize=9.2, color=INK, ha=ha, va="bottom")


def _section_label(axis: Axes, x: float, y: float, number: str, title: str) -> None:
    axis.add_patch(Circle((x, y), 0.18, facecolor=MODEL, edgecolor="none", zorder=5))
    axis.text(
        x, y - 0.01, number, color="white", fontsize=10, fontweight="bold", ha="center", va="center"
    )
    axis.text(x + 0.30, y, title, color=INK, fontsize=13, fontweight="bold", va="center")


def _model_node(axis: Axes, x: float, y: float, label: str, capacity: str) -> None:
    axis.add_patch(
        Circle((x, y), 0.49, facecolor=MODEL_LIGHT, edgecolor=MODEL, linewidth=2.0, zorder=4)
    )
    axis.text(
        x,
        y + 0.07,
        label,
        ha="center",
        va="center",
        color=INK,
        fontsize=10.5,
        fontweight="bold",
        zorder=6,
    )
    axis.text(
        x,
        y - 0.18,
        capacity,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=8.5,
        zorder=6,
    )


def _save(figure: Figure, stem: str) -> None:
    figure.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.12)
    figure.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def _legend(axis: Axes, y: float = 0.34) -> None:
    entries = [
        (HEAT, "入熱"),
        (COOL, "冷却・流れ"),
        (SENSOR, "温度計測（領域平均）"),
        (RADIATION, "表面間放射"),
        (MODEL, "評価対象のRCモデル"),
    ]
    x = 0.65
    for color, label in entries:
        axis.plot([x, x + 0.38], [y, y], color=color, linewidth=4, solid_capstyle="round")
        axis.text(x + 0.48, y, label, va="center", fontsize=9.2, color=MUTED)
        x += 2.9


def draw_topcell() -> None:
    figure, axis = _canvas(
        "TopCell / quickstart — 4領域の合成伝熱問題",
        "実CADではなく、入熱分布・冷却分布・熱結合を明示したlumped-networkの概念図",
    )
    _section_label(axis, 0.72, 7.47, "1", "問題設定：4つの温度領域をすべて計測")
    _rounded_box(axis, (0.45, 1.05), 10.45, 6.0, facecolor="#FCFDFE")

    node_x = [2.0, 4.45, 6.9, 9.35]
    labels = ["CP", "Center", "middle", "edge"]
    fills = ["#F8D8CA", "#F9DFC7", "#F5E7CB", "#DCEAF0"]
    for left, label, fill in zip(node_x, labels, fills, strict=True):
        _rounded_box(
            axis, (left - 0.72, 3.45), 1.44, 1.12, facecolor=fill, edgecolor=SOLID, linewidth=1.8
        )
        axis.text(
            left, 4.06, label, ha="center", va="center", fontsize=12, fontweight="bold", color=INK
        )
        axis.text(left, 3.72, "thermal zone", ha="center", va="center", fontsize=8.2, color=MUTED)
        _sensor(axis, (left, 4.84), f"T{label}")

    for left, right in pairwise(node_x):
        axis.plot([left + 0.75, right - 0.75], [4.0, 4.0], color=SOLID, linewidth=4, zorder=2)
    _arrow(
        axis,
        (4.9, 3.55),
        (8.85, 3.55),
        color=SOLID,
        width=1.6,
        style="<->",
        connection="arc3,rad=0.36",
        alpha=0.8,
        zorder=2,
    )
    axis.text(6.9, 2.98, "Center ↔ edge の追加熱結合", ha="center", fontsize=8.8, color=MUTED)

    axis.add_patch(
        Rectangle((1.25, 6.35), 8.83, 0.28, facecolor=HEAT, edgecolor="none", alpha=0.92)
    )
    axis.text(
        1.38,
        6.74,
        "plasma heating（CP側が強く、edge側へ弱まる）",
        color=HEAT,
        fontsize=10.5,
        fontweight="bold",
    )
    plasma_weights = [1.0, 0.75, 0.55, 0.35]
    for left, weight in zip(node_x, plasma_weights, strict=True):
        _arrow(axis, (left, 6.35), (left, 5.10), color=HEAT, width=1.2 + 4.0 * weight)

    axis.add_patch(
        Rectangle((1.25, 1.58), 8.83, 0.28, facecolor=HEATER, edgecolor="none", alpha=0.95)
    )
    axis.text(
        1.38,
        1.25,
        "heater heating（Center / middle側が強い）",
        color="#A65C00",
        fontsize=10.5,
        fontweight="bold",
    )
    heater_weights = [0.25, 0.85, 0.65, 0.45]
    for left, weight in zip(node_x, heater_weights, strict=True):
        _arrow(axis, (left, 1.86), (left, 3.18), color=HEATER, width=1.2 + 3.6 * weight)

    axis.add_patch(Rectangle((10.25, 2.05), 0.26, 3.9, facecolor=COOL, edgecolor="none"))
    axis.text(
        10.60,
        5.9,
        "brine cooling",
        color=COOL,
        fontsize=10.5,
        fontweight="bold",
        rotation=90,
        va="top",
    )
    brine_weights = [0.15, 0.25, 0.55, 0.90]
    for left, weight in zip(node_x, brine_weights, strict=True):
        _arrow(
            axis,
            (left + 0.60, 3.65),
            (10.18, 2.4 + 0.82 * weight),
            color=COOL,
            width=1.0 + 3.5 * weight,
            alpha=0.82,
        )

    _section_label(axis, 11.35, 7.47, "2", "評価したモデル")
    _rounded_box(
        axis, (11.15, 4.62), 4.35, 2.43, facecolor=MODEL_LIGHT, edgecolor=MODEL, linewidth=1.7
    )
    axis.text(11.45, 6.63, "4-node physical RC", color=MODEL, fontsize=14, fontweight="bold")
    axis.text(11.45, 6.25, "固定：4領域の熱容量・観測位置", color=INK, fontsize=9.7)
    axis.text(11.45, 5.88, "学習：熱結合、入熱gain、冷却結合、入力遅れ", color=INK, fontsize=9.7)
    axis.text(11.45, 5.48, "積分：exact / causal open-loop forecast", color=INK, fontsize=9.7)
    axis.text(11.45, 5.03, "比較：engineering prior RC / persistence", color=MUTED, fontsize=9.4)

    _rounded_box(axis, (11.15, 2.64), 4.35, 1.52, facecolor="white", edgecolor=GRID)
    axis.text(11.42, 3.82, "quickstart", color=INK, fontsize=11.5, fontweight="bold")
    axis.text(12.67, 3.82, "workflow smoke（独立benchmarkではない）", color=MUTED, fontsize=9.0)
    axis.text(11.42, 3.35, "TopCell", color=INK, fontsize=11.5, fontweight="bold")
    axis.text(12.42, 3.35, "228学習軌道 → 12外部forecast + 5 monitor", color=MUTED, fontsize=9.0)
    axis.text(
        11.42,
        2.94,
        "forecast truthはモデル入力へ渡さない",
        color=SUCCESS,
        fontsize=9.4,
        fontweight="bold",
    )

    _rounded_box(axis, (11.15, 1.05), 4.35, 1.13, facecolor="#FFF8EB", edgecolor=WARN)
    axis.text(11.42, 1.83, "注意", color=WARN, fontweight="bold", fontsize=10.5)
    axis.text(12.15, 1.83, "CP / Center / middle / edgeに実寸座標はない", color=INK, fontsize=9.3)
    axis.text(
        11.42, 1.42, "図は定義済み熱結合と分布を可視化した概念配置", color=MUTED, fontsize=9.0
    )
    _legend(axis)
    _save(figure, "01_topcell_problem_setup")


def _draw_cooling(axis: Axes, *, airflow: bool) -> None:
    if airflow:
        for y in [3.55, 4.25, 4.95, 5.65]:
            _arrow(axis, (1.12, y), (9.50, y), color=COOL, width=2.4, alpha=0.82, zorder=2)
        axis.text(
            1.25,
            6.48,
            "air inlet: Tin = coolant_temperature,  v = inlet_air_velocity",
            color=COOL,
            fontsize=10.5,
            fontweight="bold",
        )
        axis.text(8.35, 6.48, "outlet", color=COOL, fontsize=9.4)
        return
    for x, y in [(3.0, 3.7), (4.1, 5.55), (5.5, 5.55), (7.45, 5.55), (8.0, 3.7)]:
        _arrow(axis, (x, y), (x, y + 0.47), color=COOL, width=2.4)
    axis.text(
        1.25,
        6.48,
        "prescribed convection: h = 10 W/(m² K), T∞ = coolant_temperature",
        color=COOL,
        fontsize=10.2,
        fontweight="bold",
    )


def _draw_radiation(axis: Axes) -> None:
    paths = [
        ((3.5, 5.40), (2.4, 5.85)),
        ((5.9, 5.40), (5.4, 5.95)),
        ((7.2, 5.40), (8.4, 5.85)),
    ]
    for start, end in paths:
        _arrow(axis, start, end, color=RADIATION, width=1.8, linestyle="--", alpha=0.9)
    axis.text(
        7.15,
        6.05,
        "surface-to-surface radiation",
        color=RADIATION,
        fontsize=9.3,
        fontweight="bold",
    )


def _draw_mesh_overlay(axis: Axes) -> None:
    for x in [3.05, 3.45, 3.85, 4.25, 4.65, 5.05, 5.45, 5.85, 6.25, 6.65, 7.05, 7.45]:
        axis.plot([x, x + 0.25], [3.24, 5.34], color="#8FA3B8", linewidth=0.55, alpha=0.8, zorder=4)
    for offset in [0.07, 0.14, 0.21, 0.29]:
        axis.plot(
            [2.96, 7.84],
            [3.22 + offset, 3.22 + offset],
            color=COOL,
            linewidth=0.7,
            alpha=0.78,
            zorder=5,
        )
    axis.plot([4.04, 6.76], [2.67, 2.67], color=WARN, linewidth=3.0, alpha=0.9, zorder=5)
    axis.text(
        1.25,
        6.82,
        "localized tetrahedral sizes + separated wall boundary layers",
        color=MODEL,
        fontsize=10.1,
        fontweight="bold",
    )


def _draw_chip_geometry(axis: Axes, *, airflow: bool, radiation: bool, mesh: bool = False) -> None:
    axis.add_patch(
        Rectangle((0.9, 1.65), 9.0, 4.55, facecolor="#F7FAFC", edgecolor=GRID, linewidth=1.6)
    )
    axis.add_patch(
        Rectangle((1.2, 1.78), 8.4, 0.23, facecolor="#CAD1DB", edgecolor=SOLID, linewidth=1.1)
    )
    axis.text(1.25, 1.48, "support / board", color=MUTED, fontsize=8.8)
    axis.add_patch(
        Rectangle(
            (4.05, 2.02), 2.7, 0.52, facecolor=HEAT_LIGHT, edgecolor=HEAT, linewidth=2.0, zorder=3
        )
    )
    axis.text(
        5.4,
        2.28,
        "silicon chip",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    axis.add_patch(
        Rectangle(
            (4.05, 2.54),
            2.7,
            0.10,
            facecolor="#D9B48F",
            edgecolor="#9B7149",
            linewidth=1.0,
            zorder=3,
        )
    )
    axis.text(6.88, 2.58, "50 µm grease contact", va="center", fontsize=8.2, color=MUTED)
    axis.add_patch(
        Rectangle(
            (2.9, 2.64), 5.0, 0.58, facecolor="#BFC7D4", edgecolor=SOLID, linewidth=1.7, zorder=3
        )
    )
    axis.text(
        5.4,
        2.94,
        "aluminum heat-sink base",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )
    for x in [3.25, 4.45, 5.65, 6.85]:
        axis.add_patch(
            Rectangle(
                (x, 3.22), 0.60, 2.15, facecolor="#D4DAE3", edgecolor=SOLID, linewidth=1.5, zorder=3
            )
        )
    axis.text(8.15, 4.34, "4 fins", fontsize=10, color=INK, fontweight="bold")

    for x in [4.6, 5.4, 6.2]:
        _arrow(axis, (x, 1.72), (x, 2.00), color=HEAT, width=3.0)
    axis.text(3.00, 2.20, "q = chip power [W]", color=HEAT, fontsize=10.5, fontweight="bold")

    _sensor(axis, (5.4, 2.30), "chip volume avg")
    _sensor(axis, (5.4, 2.94), "base volume avg")
    _sensor(axis, (6.15, 4.30), "4-fin volume avg", align="left")

    _draw_cooling(axis, airflow=airflow)
    if radiation:
        _draw_radiation(axis)
    if mesh:
        _draw_mesh_overlay(axis)


def _draw_three_node_model(axis: Axes, *, flow_dependent: bool, radiation_gap: bool) -> None:
    _rounded_box(
        axis, (10.65, 1.52), 4.85, 5.68, facecolor="#FBFCFF", edgecolor=MODEL, linewidth=1.7
    )
    axis.text(
        10.98, 6.80, "評価対象：3-node physical RC", fontsize=13.5, fontweight="bold", color=MODEL
    )
    x_values = [11.55, 13.10, 14.65]
    names = ["chip", "sink_base", "fins"]
    capacities = ["C = 10.43 J/K", "C = 24.30 J/K", "C = 46.17 J/K"]
    for x, name, capacity in zip(x_values, names, capacities, strict=True):
        _model_node(axis, x, 5.55, name, capacity)
    axis.plot([12.04, 12.61], [5.55, 5.55], color=MODEL, linewidth=3.2)
    axis.plot([13.59, 14.16], [5.55, 5.55], color=MODEL, linewidth=3.2)
    axis.text(12.32, 5.82, "fit G₁", fontsize=8.4, color=MODEL, ha="center")
    axis.text(13.88, 5.82, "fit G₂", fontsize=8.4, color=MODEL, ha="center")
    _arrow(axis, (11.55, 4.55), (11.55, 5.02), color=HEAT, width=3.2)
    axis.text(10.97, 4.25, "known 1 W/W chip source", fontsize=8.8, color=HEAT)
    _arrow(axis, (13.10, 5.04), (13.10, 4.05), color=COOL, width=2.0)
    _arrow(axis, (14.65, 5.04), (14.65, 4.05), color=COOL, width=3.2)
    boundary = "fit Gconv (constant)" if not flow_dependent else "fit scale + exponent of Gconv(v)"
    axis.text(13.88, 3.76, boundary, fontsize=8.8, color=COOL, ha="center", fontweight="bold")
    axis.text(10.98, 3.18, "固定：熱容量、chip入熱gain、入力遅れ0", fontsize=9.0, color=INK)
    axis.text(10.98, 2.79, "予測：初期観測 + 時系列commandだけ", fontsize=9.0, color=INK)
    axis.text(10.98, 2.40, "比較：engineering prior RC / persistence", fontsize=9.0, color=MUTED)
    if radiation_gap:
        axis.text(
            10.98,
            1.92,
            "放射項はRCに含めない → paired model-gap評価",
            fontsize=9.0,
            color=RADIATION,
            fontweight="bold",
        )
    else:
        axis.text(
            10.98,
            1.92,
            "流体・放射はsolveしない線形dataset v1",
            fontsize=9.0,
            color=SUCCESS,
            fontweight="bold",
        )


def draw_linear_comsol() -> None:
    figure, axis = _canvas(
        "線形COMSOL評価 — chip発熱と一定対流冷却",
        "COMSOL 3D領域平均を3温度へ縮約し、固体伝導・contact・一定対流に対するRC同定を評価",
    )
    _section_label(
        axis,
        0.72,
        7.47,
        "1",
        "COMSOL問題：solid conduction + grease contact + prescribed convection",
    )
    _draw_chip_geometry(axis, airflow=False, radiation=False)
    _section_label(axis, 10.72, 7.47, "2", "学習・評価したモデル")
    _draw_three_node_model(axis, flow_dependent=False, radiation_gap=False)
    axis.text(
        0.95,
        1.08,
        "計測は点温度ではなく、chip / base / 4 fins の各3D volume average",
        color=SENSOR,
        fontsize=10.3,
        fontweight="bold",
    )
    _legend(axis)
    _save(figure, "02_linear_comsol_problem_setup")


def draw_nonlinear_comsol() -> None:
    figure, axis = _canvas(
        "非線形COMSOL評価 — 共役熱流動・流速依存冷却・放射gap",
        "同じchip / base / fins計測境界のまま、入口空気の温度・流速と表面間放射を独立に検証",
    )
    _section_label(axis, 0.72, 7.47, "1", "COMSOL問題：conjugate forced convection")
    _draw_chip_geometry(axis, airflow=True, radiation=True)
    _section_label(axis, 10.72, 7.47, "2", "学習・評価したモデル")
    _draw_three_node_model(axis, flow_dependent=True, radiation_gap=True)
    axis.text(
        0.95,
        1.08,
        "global-8 CAE: 10 training / 9 non-radiation forecast / 5 paired radiation gap",
        color=INK,
        fontsize=9.9,
        fontweight="bold",
    )
    _legend(axis)
    _save(figure, "03_nonlinear_comsol_problem_setup")


def draw_high_fidelity_comsol() -> None:
    figure, axis = _canvas(
        "High-fidelity COMSOL外部評価 — 局所meshの2ケースpair",
        "global-8で学習済みの同一3-node RCを再学習せず、local-mediumの共役流動 / 放射pairへ適用",
    )
    _section_label(axis, 0.72, 7.47, "1", "局所mesh問題：支配領域を選択的に細分化")
    _draw_chip_geometry(axis, airflow=True, radiation=True, mesh=True)
    _section_label(axis, 10.72, 7.47, "2", "外部評価（refitなし）")
    _rounded_box(
        axis, (10.65, 4.82), 4.85, 2.30, facecolor=MODEL_LIGHT, edgecolor=MODEL, linewidth=1.7
    )
    axis.text(
        10.98,
        6.72,
        "学習済み flow-dependent 3-node RC",
        fontsize=13,
        fontweight="bold",
        color=MODEL,
    )
    axis.text(10.98, 6.30, "学習データ：global-8 nonlinear train × 10", fontsize=9.4, color=INK)
    axis.text(10.98, 5.90, "HV01：local-medium conjugate flow", fontsize=9.4, color=INK)
    axis.text(
        10.98,
        5.52,
        "HV02：同一入力 + surface radiation",
        fontsize=9.4,
        color=RADIATION,
        fontweight="bold",
    )
    axis.text(10.98, 5.14, "入力範囲内でmodel-form transferを確認", fontsize=9.4, color=SUCCESS)

    _rounded_box(
        axis, (10.65, 2.70), 4.85, 1.68, facecolor="#FFF8EB", edgecolor=WARN, linewidth=1.5
    )
    axis.text(10.98, 4.00, "現在の資格状態", fontsize=11.5, color=WARN, fontweight="bold")
    axis.text(10.98, 3.61, "mesh: benchmark基準 pass / strict基準 fail", fontsize=9.1, color=INK)
    axis.text(10.98, 3.26, "time step 2 s vs 1 s: 8量中4量 fail", fontsize=9.1, color=INK)
    axis.text(
        10.98, 2.91, "用途：volume-average model-form screeningのみ", fontsize=9.1, color=MUTED
    )

    _rounded_box(axis, (10.65, 1.05), 4.85, 1.22, facecolor="white", edgecolor=GRID)
    axis.text(
        10.98,
        1.87,
        "同じ計測：chip / base / 4-fin volume average",
        fontsize=9.2,
        color=SENSOR,
        fontweight="bold",
    )
    axis.text(
        10.98,
        1.48,
        "同じ入力：chip power / inlet T / inlet velocity",
        fontsize=9.2,
        color=COOL,
        fontweight="bold",
    )
    _legend(axis)
    _save(figure, "04_high_fidelity_comsol_problem_setup")


def _flow_box(axis: Axes, x: float, y: float, width: float, text: str, color: str) -> None:
    _rounded_box(
        axis, (x, y), width, 0.74, facecolor="white", edgecolor=color, linewidth=1.5, radius=0.12
    )
    axis.text(
        x + width / 2,
        y + 0.37,
        text,
        ha="center",
        va="center",
        fontsize=9.1,
        color=INK,
        linespacing=1.25,
    )


def draw_model_map() -> None:
    figure, axis = _canvas(
        "各評価で使用したモデル — データ源から判定まで",
        "主モデルは解釈可能な物理RC。別枠でニューラル5モデルを同条件比較（product未採用）",
    )
    headers = [
        (0.55, "評価"),
        (3.15, "真値・データ"),
        (6.35, "同定モデル"),
        (10.00, "比較モデル"),
        (13.10, "役割・境界"),
    ]
    for x, label in headers:
        axis.text(x, 7.50, label, fontsize=11.5, fontweight="bold", color=MUTED)
    axis.plot([0.45, 15.55], [7.26, 7.26], color=GRID, linewidth=1.5)

    rows = [
        (
            "quickstart",
            "合成CSV\n4温度・3入力",
            "同定済み4ノードRC",
            "未学習RC\n温度保持",
            "公開workflow動作確認\n独立benchmarkではない",
        ),
        (
            "TopCell",
            "独立した合成真値\n228学習 / 12外部予測",
            "同定済み4ノードRC",
            "未学習RC・温度保持\n+ NN5（別枠）",
            "外部case + 適用外case\n監視はRC上のobserver",
        ),
        (
            "線形COMSOL",
            "固体・接触・一定対流\n16学習 / 8外部予測",
            "同定済み3ノードRC\n一定の境界熱伝達",
            "未学習RC・温度保持\n+ NN5（別枠）",
            "3D領域平均screening\n監視はRC上のobserver",
        ),
        (
            "非線形COMSOL",
            "global-8共役熱流動\n10学習 / 14外部評価",
            "同定済み3ノードRC\n風速依存の境界熱伝達",
            "未学習RC・温度保持\n+ NN5（別枠）",
            "放射なし9 cases\n放射gap 5 cases",
        ),
        (
            "高忠実度COMSOL",
            "local-medium COMSOL\nHV01 / HV02 pair",
            "同じ非線形RC\n再学習なし",
            "未学習RC・温度保持\n+ NN5（別枠）",
            "外部転移screening\n厳格CAE資格なし",
        ),
    ]
    y_values = [6.25, 5.10, 3.95, 2.80, 1.65]
    row_colors = [SOLID, MODEL, COOL, RADIATION, WARN]
    for row, y, row_color in zip(rows, y_values, row_colors, strict=True):
        name, data, model, comparison, role = row
        axis.text(
            0.55, y + 0.37, name, fontsize=11, fontweight="bold", color=row_color, va="center"
        )
        _flow_box(axis, 2.65, y, 2.70, data, row_color)
        _arrow(axis, (5.42, y + 0.37), (5.90, y + 0.37), color=GRID, width=1.8)
        _flow_box(axis, 5.98, y, 3.05, model, MODEL)
        _arrow(axis, (9.10, y + 0.37), (9.56, y + 0.37), color=GRID, width=1.8)
        _flow_box(axis, 9.65, y, 2.35, comparison, SOLID)
        _arrow(axis, (12.07, y + 0.37), (12.53, y + 0.37), color=GRID, width=1.8)
        _flow_box(axis, 12.62, y, 2.88, role, row_color)

    _rounded_box(
        axis, (0.52, 0.48), 15.0, 0.72, facecolor="#EEF8F5", edgecolor=SUCCESS, linewidth=1.3
    )
    axis.text(0.82, 0.84, "共通原則", color=SUCCESS, fontsize=10.5, fontweight="bold", va="center")
    axis.text(
        2.05,
        0.84,
        "将来真値は入力しない / case平均 + 最大case + peakを評価 / R²は補助図",
        color=INK,
        fontsize=10.1,
        va="center",
    )
    _save(figure, "05_evaluation_model_map")


def _file_box(
    axis: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: list[str],
    color: str,
) -> None:
    _rounded_box(axis, (x, y), width, height, facecolor="white", edgecolor=color, linewidth=1.6)
    axis.text(
        x + 0.22, y + height - 0.28, title, fontsize=10.7, fontweight="bold", color=color, va="top"
    )
    for index, line in enumerate(lines):
        axis.text(
            x + 0.22,
            y + height - 0.72 - 0.31 * index,
            line,
            fontsize=8.35,
            color=INK,
            va="top",
            family="monospace",
        )


def draw_comsol_lineage() -> None:
    figure, axis = _canvas(
        "COMSOLファイルの所在と評価データの系譜",
        "Application Library原本はインストール領域、"
        "repo内の.mphは代表ケースだけを保存した作業コピー",
    )
    _file_box(
        axis,
        0.55,
        5.27,
        4.45,
        2.13,
        "① immutable source model (outside repo)",
        [
            "C:/Program Files/COMSOL/COMSOL64/",
            "Multiphysics_copy1/applications/",
            "Heat_Transfer_Module/",
            "Tutorials,_Forced_and_Natural_Convection/",
            "chip_cooling.mph",
        ],
        SOLID,
    )
    axis.text(
        0.78,
        4.93,
        "ModelUtil.loadCopyで開くため、原本は上書きしない",
        fontsize=9.2,
        color=SUCCESS,
        fontweight="bold",
    )

    _arrow(axis, (5.05, 6.34), (5.72, 6.34), color=GRID, width=2.2)
    _file_box(
        axis,
        5.82,
        5.48,
        4.15,
        1.72,
        "② Java runner (repo)",
        [
            "comsol/RunChipCoolingCase.java",
            "comsol/RunChipCoolingNonlinearCase.java",
            "input schedule → COMSOL solve → raw table",
        ],
        MODEL,
    )

    _arrow(axis, (10.03, 6.34), (10.67, 6.34), color=GRID, width=2.2)
    _file_box(
        axis,
        10.78,
        5.18,
        4.70,
        2.32,
        "③ saved representative .mph copies (repo work/)",
        [
            "linear: work/models/",
            "  electronic_chip_cooling_dataset.mph",
            "global-8 nonlinear: work/nonlinear/mesh_8/models/",
            "  electronic_chip_cooling_conjugate.mph",
            "  electronic_chip_cooling_radiation.mph",
        ],
        HEAT,
    )

    branch_y = 3.72
    axis.plot([2.78, 2.78], [5.22, branch_y], color=GRID, linewidth=2.0)
    axis.plot([2.78, 12.98], [branch_y, branch_y], color=GRID, linewidth=2.0)
    _arrow(axis, (7.88, branch_y), (7.88, 3.25), color=GRID, width=2.0)
    _file_box(
        axis,
        4.93,
        1.70,
        5.92,
        1.50,
        "④ high-fidelity local-medium: .mph copyは保存していない",
        [
            "work/nonlinear/mesh_local_medium/raw/*.txt",
            "work/nonlinear/mesh_local_medium/logs/*.log",
            "data/nonlinear_high_fidelity/dynamic/**/*.csv",
        ],
        WARN,
    )
    axis.text(
        5.18,
        1.34,
        "元のchip_cooling.mph + schedule + runner + raw/logから再現する構成",
        fontsize=9.0,
        color=MUTED,
    )

    _arrow(axis, (12.98, branch_y), (12.98, 3.20), color=GRID, width=2.0)
    _file_box(
        axis,
        11.15,
        1.70,
        4.33,
        1.50,
        "⑤ benchmarkが読むもの",
        [
            "published trajectory CSV",
            "→ train artifact",
            "→ forecast prediction / metrics / figures",
        ],
        SUCCESS,
    )

    _rounded_box(axis, (0.55, 0.44), 14.93, 0.61, facecolor="#F7F8FA", edgecolor=GRID)
    axis.text(
        0.78,
        0.75,
        "補足：work/chip_cooling_dataset_model.mph と mesh_7 / mesh_9 のコピーは"
        "現行runnerの主評価先ではない作業履歴",
        fontsize=8.9,
        color=MUTED,
        va="center",
    )
    _save(figure, "06_comsol_file_lineage")


def main() -> None:
    """Write all benchmark setup figures next to this generator."""
    _configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_topcell()
    draw_linear_comsol()
    draw_nonlinear_comsol()
    draw_high_fidelity_comsol()
    draw_model_map()
    draw_comsol_lineage()


if __name__ == "__main__":
    main()
