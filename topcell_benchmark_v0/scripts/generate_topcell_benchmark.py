"""Generate TopCell-ICP-Thermal Benchmark v0 example data.

The generated files are intentionally small enough for quick training while
covering constant CAE responses, time-varying recipes, sparse sensor variants,
and pseudo-measurement monitor logs.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PRED_DIR = ROOT / "data" / "pred"
SCHED_DIR = PRED_DIR / "schedules"
MONITOR_DIR = PRED_DIR / "monitor_logs"
CFG_DIR = ROOT / "configs"

SENSORS = ["CP", "Center", "middle", "edge"]
CONTROLS = ["brine", "heater", "plasma"]
DT = 1.0
T_END = 180.0
TIMES = np.arange(0.0, T_END + 0.5 * DT, DT)

# Compact 4-node thermal network used only to create the synthetic benchmark.
THERMAL_MASS = np.array([0.95, 1.15, 1.35, 1.55], dtype=float)
PLASMA_WEIGHT = np.array([1.00, 0.75, 0.55, 0.35], dtype=float)
HEATER_WEIGHT = np.array([0.25, 0.85, 0.65, 0.45], dtype=float)
BRINE_WEIGHT = np.array([0.15, 0.25, 0.55, 0.90], dtype=float)
EDGES_R = {
    (0, 1): 0.75,
    (1, 2): 0.50,
    (2, 3): 0.60,
    (1, 3): 1.60,
}
CONDUCTANCE = np.zeros((4, 4), dtype=float)
for (i, j), r in EDGES_R.items():
    CONDUCTANCE[i, j] = CONDUCTANCE[j, i] = 0.045 / r

# Internal actuator lags used by the synthetic "truth" generator.  The model may
# learn a similar lag through features.use_effective_controls.
TRUTH_CONTROL_TAU = {"brine": 4.0, "heater": 8.0, "plasma": 1.0}


def ensure_dirs() -> None:
    for d in [RAW_DIR, SCHED_DIR, MONITOR_DIR, CFG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def effective_controls(raw: np.ndarray) -> np.ndarray:
    tau = np.array([TRUTH_CONTROL_TAU[c] for c in CONTROLS], dtype=float)
    eff = np.empty_like(raw, dtype=float)
    eff[0] = raw[0]
    for k in range(1, len(raw)):
        dt = TIMES[k] - TIMES[k - 1]
        eff[k] = raw[k]
        for j, tj in enumerate(tau):
            if tj > 0:
                alpha = 1.0 - np.exp(-dt / tj)
                eff[k, j] = eff[k - 1, j] + alpha * (raw[k, j] - eff[k - 1, j])
    return eff


def simulate(raw_controls: np.ndarray, init_temp: np.ndarray) -> np.ndarray:
    """Synthetic CAE-like transient thermal response."""
    controls_eff = effective_controls(raw_controls)
    temp = np.asarray(init_temp, dtype=float).copy()
    out = np.zeros((len(TIMES), len(SENSORS)), dtype=float)
    for k in range(len(TIMES)):
        out[k] = temp
        if k == len(TIMES) - 1:
            break
        brine, heater, plasma = controls_eff[k]
        conduction = (CONDUCTANCE * (temp[None, :] - temp[:, None])).sum(axis=1)
        plasma_src = 0.010 * plasma * PLASMA_WEIGHT
        heater_src = 0.014 * max(heater - 70.0, 0.0) * HEATER_WEIGHT
        brine_sink_temp = 55.0 - 0.45 * brine
        brine_cooling = 0.030 * BRINE_WEIGHT * (brine_sink_temp - temp)
        ambient_loss = 0.004 * (45.0 - temp)
        dtemp = (conduction + plasma_src + heater_src + brine_cooling + ambient_loss) / THERMAL_MASS
        temp = temp + DT * dtemp
    return out


def write_cae_csv(path: Path, temps: np.ndarray, controls: np.ndarray, header: bool = True) -> None:
    df = pd.DataFrame({"time": TIMES})
    for i, s in enumerate(SENSORS):
        df[s] = temps[:, i]
    for j, c in enumerate(CONTROLS):
        df[c] = controls[:, j]
    df = df[["time", *SENSORS, *CONTROLS]]
    df.to_csv(path, index=False, header=header, float_format="%.6f")


def const_controls(brine: float, heater: float, plasma: float) -> np.ndarray:
    return np.repeat(np.array([[brine, heater, plasma]], dtype=float), len(TIMES), axis=0)


def step_schedule(base: tuple[float, float, float], kind: str) -> np.ndarray:
    br, ht, pl = base
    u = np.repeat(np.array([[br, ht, pl]], dtype=float), len(TIMES), axis=0)
    if kind == "step_plasma":
        u[:, 2] = 0.0
        u[(TIMES >= 20) & (TIMES < 120), 2] = pl
        u[TIMES >= 120, 2] = 0.0
    elif kind == "step_heater":
        u[:, 1] = 80.0
        u[(TIMES >= 40) & (TIMES < 120), 1] = ht
        u[TIMES >= 120, 1] = 120.0
    elif kind == "step_brine":
        u[:, 0] = 20.0
        u[(TIMES >= 60) & (TIMES < 140), 0] = br
        u[TIMES >= 140, 0] = 20.0
    elif kind == "recipe":
        # preheat -> main etch -> over-etch -> cooldown
        u[:, :] = np.array([20.0, 170.0, 0.0])
        u[(TIMES >= 20) & (TIMES < 120), :] = np.array([br, ht, pl])
        u[(TIMES >= 120) & (TIMES < 150), :] = np.array([max(br, 35.0), 130.0, 0.55 * pl])
        u[TIMES >= 150, :] = np.array([40.0, 80.0, 0.0])
    else:
        raise ValueError(kind)
    return u


def init_patterns() -> dict[str, np.ndarray]:
    return {
        "cold": np.array([45.0, 45.0, 45.0, 45.0]),
        "nominal": np.array([55.0, 55.0, 55.0, 55.0]),
        "gradient": np.array([54.0, 56.0, 59.0, 62.0]),
    }


def write_node_edge_tables() -> None:
    pd.DataFrame(
        [
            ["CP", 0.0, 0.0, 0.0, 0.95, 1.00, 0.25, 0.15],
            ["Center", 0.0, 0.0, 25.0, 1.15, 0.75, 0.85, 0.25],
            ["middle", 70.0, 0.0, 25.0, 1.35, 0.55, 0.65, 0.55],
            ["edge", 140.0, 0.0, 25.0, 1.55, 0.35, 0.45, 0.90],
        ],
        columns=["sensor", "x_mm", "y_mm", "z_mm", "thermal_mass", "plasma_weight", "heater_weight", "brine_weight"],
    ).to_csv(CFG_DIR / "cell_nodes.csv", index=False)
    pd.DataFrame(
        [
            ["CP", "Center", 0.75, "vertical", "plasma-side CP to upper center"],
            ["Center", "middle", 0.50, "radial", "center to middle"],
            ["middle", "edge", 0.60, "radial", "middle to edge"],
            ["Center", "edge", 1.60, "radial_long", "weak long-range path"],
        ],
        columns=["src", "dst", "r_th_K_per_W", "contact_type", "note"],
    ).to_csv(CFG_DIR / "cell_edges.csv", index=False)


def generate_raw_cae() -> None:
    for p in RAW_DIR.glob("temp_*.csv"):
        p.unlink()
    brines = [10, 20, 30, 40]
    heaters = [80, 120, 160, 200]
    plasmas = [0, 50, 100, 150]
    patterns = init_patterns()
    # Dataset A: constant-condition response.
    for br in brines:
        for ht in heaters:
            for pl in plasmas:
                for pname, init in patterns.items():
                    controls = const_controls(br, ht, pl)
                    temps = simulate(controls, init)
                    path = RAW_DIR / f"temp_{br}_{ht}_{pl}_const_{pname}.csv"
                    write_cae_csv(path, temps, controls, header=True)
    # Dataset B: dynamic recipes; representative points covering each axis.
    dynamic_bases = [
        (20, 120, 100),
        (30, 160, 150),
        (40, 120, 100),
        (20, 200, 150),
    ]
    for br, ht, pl in dynamic_bases:
        for kind in ["step_plasma", "step_heater", "step_brine", "recipe"]:
            controls = step_schedule((br, ht, pl), kind)
            init = patterns["nominal"] if kind != "recipe" else patterns["gradient"]
            temps = simulate(controls, init)
            path = RAW_DIR / f"temp_{br}_{ht}_{pl}_{kind}.csv"
            write_cae_csv(path, temps, controls, header=True)


def write_prediction_schedules() -> None:
    # Explicit schedules for forecast examples.
    schedules = {
        "schedule_plasma_onoff.csv": step_schedule((20, 120, 100), "step_plasma"),
        "schedule_heater_step.csv": step_schedule((20, 180, 100), "step_heater"),
        "schedule_brine_step.csv": step_schedule((40, 120, 100), "step_brine"),
        "schedule_recipe.csv": step_schedule((30, 160, 150), "recipe"),
    }
    rows = []
    for name, controls in schedules.items():
        df = pd.DataFrame({"time": TIMES})
        for j, c in enumerate(CONTROLS):
            df[c] = controls[:, j]
        df.to_csv(SCHED_DIR / name, index=False, float_format="%.6f")
    rows.extend(
        [
            {
                "case_id": "forecast_const_nominal",
                "t_end": T_END,
                "dt": DT,
                "init_CP": 55.0,
                "init_Center": 55.0,
                "init_middle": 55.0,
                "init_edge": 55.0,
                "brine": 20.0,
                "heater": 120.0,
                "plasma": 100.0,
                "schedule_csv": "",
            },
            {
                "case_id": "forecast_plasma_onoff",
                "t_end": T_END,
                "dt": DT,
                "init_CP": 55.0,
                "init_Center": 55.0,
                "init_middle": 55.0,
                "init_edge": 55.0,
                "brine": 20.0,
                "heater": 120.0,
                "plasma": 100.0,
                "schedule_csv": "data/pred/schedules/schedule_plasma_onoff.csv",
            },
            {
                "case_id": "forecast_recipe",
                "t_end": T_END,
                "dt": DT,
                "init_CP": 54.0,
                "init_Center": 56.0,
                "init_middle": 59.0,
                "init_edge": 62.0,
                "brine": 30.0,
                "heater": 160.0,
                "plasma": 150.0,
                "schedule_csv": "data/pred/schedules/schedule_recipe.csv",
            },
        ]
    )
    pd.DataFrame(rows).to_csv(PRED_DIR / "conditions.csv", index=False)


def write_monitor_logs(seed: int = 123) -> None:
    rng = np.random.default_rng(seed)
    monitor_specs = [
        ("monitor_recipe_toolA", (30, 160, 150), "recipe", np.array([54.0, 56.0, 59.0, 62.0])),
        ("monitor_plasma_onoff_toolB", (20, 120, 100), "step_plasma", np.array([55.0, 55.0, 55.0, 55.0])),
    ]
    cases = []
    sensor_offset = np.array([0.15, 0.45, -0.25, 0.10])
    for case_id, base, kind, init in monitor_specs:
        raw_controls = step_schedule(base, kind)
        temps_true = simulate(raw_controls, init)
        slow_drift = np.linspace(0.0, 0.50, len(TIMES))[:, None]
        machine_bias = 0.30 if "toolA" in case_id else -0.20
        noise = rng.normal(0.0, 0.15, size=temps_true.shape)
        measured = temps_true + sensor_offset[None, :] + machine_bias + slow_drift + noise
        df = pd.DataFrame({"time": TIMES})
        for i, s in enumerate(SENSORS):
            df[s] = measured[:, i]
        for j, c in enumerate(CONTROLS):
            df[c] = raw_controls[:, j]
        log_path = MONITOR_DIR / f"{case_id}.csv"
        df.to_csv(log_path, index=False, float_format="%.6f")
        cases.append({"case_id": case_id, "dt": DT, "log_csv": f"data/pred/monitor_logs/{case_id}.csv"})
    pd.DataFrame(cases).to_csv(PRED_DIR / "monitor_cases.csv", index=False)


def write_manifest() -> None:
    manifest = {
        "name": "TopCell-ICP-Thermal Benchmark v0",
        "description": "Synthetic but physics-inspired upper Cell transient thermal benchmark for v6 code.",
        "sensors": SENSORS,
        "controls": CONTROLS,
        "dt_s": DT,
        "t_end_s": T_END,
        "datasets": {
            "A_constant": "4x4x4 controls x 3 initial patterns = 192 trajectories",
            "B_dynamic": "4 base conditions x 4 schedule types = 16 trajectories",
            "D_pseudo_monitor": "2 pseudo-measurement logs with offset, drift, noise",
        },
        "truth_generator": {
            "equation": "lumped RC network with plasma/heater/brine sources and first-order actuator lag",
            "not_for_training": "The learner only sees CSV trajectories and optional graph prior tables.",
        },
    }
    with open(ROOT / "data" / "topcell_benchmark_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    ensure_dirs()
    write_node_edge_tables()
    generate_raw_cae()
    write_prediction_schedules()
    write_monitor_logs()
    write_manifest()
    print(f"generated benchmark under: {ROOT}")


if __name__ == "__main__":
    main()
