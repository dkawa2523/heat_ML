from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import as_path


@dataclass
class Case:
    case_id: str
    path: Path
    condition: dict[str, float]
    frame: pd.DataFrame


def parse_condition(path: str | Path, regex: str) -> dict[str, float]:
    p = Path(path)
    m = re.fullmatch(regex, p.name)
    if m is None:
        raise ValueError(f"file name does not match filename_regex: {p.name}")
    return {k: float(v) for k, v in m.groupdict().items()}


def _as_bool_or_auto(value) -> bool | str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "auto":
            return "auto"
        return v in {"1", "true", "yes", "on"}
    return bool(value)


def _looks_like_header(path: Path, sep: str) -> bool:
    # First non-comment line: if every token can be parsed as float, it is data;
    # otherwise it is a header.  This is safer than checking alphabetic chars
    # because numeric scientific notation may contain "e".
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = [t for t in re.split(sep, line) if t]
            try:
                for t in tokens:
                    float(t)
                return False
            except ValueError:
                return True
    return False


def _column_names_for_file(df: pd.DataFrame, data_cfg: dict, path: Path) -> list[str]:
    """Return column names for a header-less CAE CSV.

    Most files are: time + temperatures only.  If a CAE file appends
    time-varying brine/heater/plasma columns, we accept that without changing
    the rest of the code.
    """
    base_cols = list(data_cfg["columns"])
    control_cols = list(data_cfg.get("control_cols", []))
    n = df.shape[1]
    if n == len(base_cols):
        return base_cols
    if n == len(base_cols) + len(control_cols):
        return [*base_cols, *control_cols]
    raise ValueError(
        f"{path.name}: expected {len(base_cols)} columns"
        f" or {len(base_cols) + len(control_cols)} columns with controls, got {n}"
    )


def _read_case_frame(path: Path, data_cfg: dict) -> pd.DataFrame:
    sep = data_cfg.get("sep", r"[\s,]+")
    header_cfg = _as_bool_or_auto(data_cfg.get("header", False))
    header = _looks_like_header(path, sep) if header_cfg == "auto" else bool(header_cfg)

    if header:
        df = pd.read_csv(path, header=0, sep=sep, engine="python", comment="#")
        df.columns = [str(c).strip() for c in df.columns]
        required = list(data_cfg["columns"])
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: header CSV missing required columns {missing}")
        keep = [*required, *[c for c in data_cfg.get("control_cols", []) if c in df.columns]]
        # Keep only columns used by this lightweight pipeline. Extra log columns
        # can remain in the original file but are not required here.
        df = df[keep]
    else:
        df = pd.read_csv(path, header=None, sep=sep, engine="python", comment="#")
        df.columns = _column_names_for_file(df, data_cfg, path)
    return df.astype(float)


def read_case(path: str | Path, data_cfg: dict) -> Case:
    p = Path(path)
    cond = parse_condition(p, data_cfg["filename_regex"])
    df = _read_case_frame(p, data_cfg)

    # If CAE CSV already contains time-varying control columns, do not
    # overwrite them with filename constants.  Otherwise add constants from
    # temp_<brine>_<heater>_<plasma>.csv.
    for k, v in cond.items():
        if k not in df.columns:
            df[k] = v
    return Case(case_id=p.stem, path=p, condition=cond, frame=df)


def load_cases(data_cfg: dict, root: Path) -> list[Case]:
    raw_dir = as_path(data_cfg["raw_dir"], root)
    paths = sorted(raw_dir.glob(data_cfg.get("file_pattern", "temp_*.csv")))
    if not paths:
        raise FileNotFoundError(f"no CAE CSV files found under: {raw_dir}")
    cases = [read_case(p, data_cfg) for p in paths]
    validate_cases(cases, data_cfg)
    return cases


def validate_cases(cases: list[Case], data_cfg: dict) -> None:
    time_col = data_cfg["columns"][0]
    sensor_cols = list(data_cfg["sensor_cols"])
    control_cols = list(data_cfg.get("control_cols", []))
    dt_expected = data_cfg.get("dt")
    temp_min = data_cfg.get("temp_min")
    temp_max = data_cfg.get("temp_max")

    for c in cases:
        missing = [col for col in [time_col, *sensor_cols, *control_cols] if col not in c.frame.columns]
        if missing:
            raise ValueError(f"{c.path.name}: missing columns {missing}")
        values = c.frame[[time_col, *sensor_cols, *control_cols]].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{c.path.name}: NaN or inf detected")
        t = c.frame[time_col].to_numpy(float)
        if len(t) < 2:
            raise ValueError(f"{c.path.name}: at least two time rows are required")
        if not np.all(np.diff(t) > 0):
            raise ValueError(f"{c.path.name}: time must be strictly increasing")
        if dt_expected is not None and not np.allclose(np.diff(t), float(dt_expected), rtol=1e-4, atol=1e-8):
            raise ValueError(f"{c.path.name}: dt is not constant {dt_expected}")
        temps = c.frame[sensor_cols].to_numpy(float)
        if temp_min is not None and temps.min() < float(temp_min):
            raise ValueError(f"{c.path.name}: temperature below temp_min")
        if temp_max is not None and temps.max() > float(temp_max):
            raise ValueError(f"{c.path.name}: temperature above temp_max")


def infer_common_dt(cases: list[Case], time_col: str) -> float | None:
    dts: list[float] = []
    for c in cases:
        t = c.frame[time_col].to_numpy(float)
        d = np.diff(t)
        if len(d) == 0:
            continue
        if not np.allclose(d, d[0], rtol=1e-4, atol=1e-8):
            return None
        dts.append(float(d[0]))
    if not dts:
        return None
    if not np.allclose(dts, dts[0], rtol=1e-4, atol=1e-8):
        return None
    return float(dts[0])


def _random_train_val_test(cases: list[Case], split_cfg: dict, seed: int) -> dict[str, list[Case]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cases))
    rng.shuffle(idx)
    n = len(idx)
    n_train = max(1, int(round(n * float(split_cfg.get("train_ratio", 0.7)))))
    n_val = int(round(n * float(split_cfg.get("val_ratio", 0.15))))
    if n >= 3:
        n_val = max(1, n_val)
    if n_train + n_val >= n and n >= 3:
        n_train = max(1, n - 2)
        n_val = 1
    return {
        "train": [cases[i] for i in idx[:n_train]],
        "val": [cases[i] for i in idx[n_train : n_train + n_val]],
        "test": [cases[i] for i in idx[n_train + n_val :]],
    }


def _random_train_val_only(cases: list[Case], split_cfg: dict, seed: int) -> dict[str, list[Case]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cases))
    rng.shuffle(idx)
    n = len(idx)
    n_val = int(round(n * float(split_cfg.get("val_ratio", 0.15))))
    if n >= 2:
        n_val = max(1, min(n - 1, n_val))
    else:
        n_val = 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return {"train": [cases[i] for i in train_idx], "val": [cases[i] for i in val_idx]}


def _split_holdout_extreme(cases: list[Case], split_cfg: dict, seed: int, mode: str) -> dict[str, list[Case]]:
    control = split_cfg.get("holdout_control")
    if not control:
        raise ValueError(f"split.method={mode} requires split.holdout_control")
    vals = np.array([float(c.condition[control]) for c in cases])
    extreme = vals.max() if mode == "holdout_max" else vals.min()
    test = [c for c in cases if np.isclose(float(c.condition[control]), extreme)]
    test_ids = {id(c) for c in test}
    rest = [c for c in cases if id(c) not in test_ids]
    if len(rest) < 2:
        raise ValueError(f"holdout leaves too few train/val cases for control={control}")
    base = _random_train_val_only(rest, split_cfg, seed)
    return {"train": base["train"], "val": base["val"], "test": test}


def _split_holdout_corner(cases: list[Case], split_cfg: dict, seed: int) -> dict[str, list[Case]]:
    controls = list(split_cfg.get("corner_controls") or cases[0].condition.keys())
    direction = str(split_cfg.get("corner_direction", "max"))
    n_test = max(1, int(round(len(cases) * float(split_cfg.get("test_ratio", 0.15)))))
    mat = np.array([[float(c.condition[k]) for k in controls] for c in cases], dtype=float)
    mn, mx = mat.min(axis=0), mat.max(axis=0)
    scaled = (mat - mn) / np.where(mx - mn < 1e-12, 1.0, mx - mn)
    score = scaled.sum(axis=1) if direction == "max" else (1.0 - scaled).sum(axis=1)
    test_idx = set(np.argsort(score)[-n_test:])
    test = [c for i, c in enumerate(cases) if i in test_idx]
    rest = [c for i, c in enumerate(cases) if i not in test_idx]
    if len(rest) < 2:
        raise ValueError("holdout_corner leaves too few train/val cases")
    base = _random_train_val_only(rest, split_cfg, seed)
    return {"train": base["train"], "val": base["val"], "test": test}


def split_cases(cases: list[Case], split_cfg: dict, seed: int) -> dict[str, list[Case]]:
    """Split by whole CAE file. Never split rows from one trajectory."""
    method = str(split_cfg.get("method", "random"))
    if method in {"random", "condition_file"}:
        return _random_train_val_test(cases, split_cfg, seed)
    if method in {"holdout_max", "holdout_min"}:
        return _split_holdout_extreme(cases, split_cfg, seed, method)
    if method == "holdout_corner":
        return _split_holdout_corner(cases, split_cfg, seed)
    raise ValueError(
        f"unknown split.method={method}; choices are random, holdout_max, holdout_min, holdout_corner"
    )


def save_split(split: dict[str, list[Case]], out_path: str | Path) -> None:
    rows = []
    for split_name, cases in split.items():
        for c in cases:
            rows.append({"split": split_name, "case_id": c.case_id, "path": str(c.path), **c.condition})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def save_data_summary(cases: list[Case], sensor_cols: list[str], out_path: str | Path) -> None:
    rows = []
    for c in cases:
        temps = c.frame[sensor_cols].to_numpy(float)
        rows.append(
            {
                "case_id": c.case_id,
                "n_rows": len(c.frame),
                "time_start": float(c.frame.iloc[0, 0]),
                "time_end": float(c.frame.iloc[-1, 0]),
                "temp_min": float(temps.min()),
                "temp_max": float(temps.max()),
                **c.condition,
            }
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def case_control_series(case: Case, control_cols: list[str]) -> np.ndarray:
    """Return raw [time, control] array.

    If CAE files include time-varying control columns, those are used.
    Otherwise controls are filename constants.
    """
    if all(c in case.frame.columns for c in control_cols):
        return case.frame[control_cols].to_numpy(float)
    u = np.array([case.condition[c] for c in control_cols], dtype=float)
    return np.repeat(u[None, :], len(case.frame), axis=0)
