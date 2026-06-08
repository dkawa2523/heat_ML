from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import as_path


def _clip(x: np.ndarray, limit: float) -> np.ndarray:
    if limit <= 0:
        return x
    return np.clip(x, -limit, limit)


def _load_offset_artifact(path: str | Path, root: Path, sensor_cols: list[str]) -> np.ndarray:
    p = as_path(path, root)
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    offsets = obj.get("sensor_offsets", obj.get("offsets", obj))
    return np.array([float(offsets.get(s, 0.0)) for s in sensor_cols], dtype=float)


class WeakCorrector:
    """Small, interpretable post-hoc corrector for measured-machine residuals.

    It is intentionally weak: static sensor offset plus optional slow EMA bias.
    The base CAE surrogate remains the main model.
    """

    def __init__(self, sensor_cols: list[str], cfg: dict[str, Any] | None = None, root: Path | None = None):
        cfg = cfg or {}
        self.sensor_cols = sensor_cols
        self.enabled = bool(cfg.get("enabled", False))
        self.mode = str(cfg.get("mode", "none")).lower()
        self.max_abs = float(cfg.get("max_abs_correction", 0.0))
        self.alpha = float((cfg.get("ema", {}) or {}).get("alpha", cfg.get("ema_alpha", 0.03)))
        self.warmup_steps = int((cfg.get("ema", {}) or {}).get("warmup_steps", 0))
        self.n_update = 0
        self.bias = np.zeros(len(sensor_cols), dtype=float)

        offset = np.zeros(len(sensor_cols), dtype=float)
        if cfg.get("artifact") and root is not None:
            offset = _load_offset_artifact(cfg["artifact"], root, sensor_cols)
        if cfg.get("offset_values"):
            vals = cfg["offset_values"] or {}
            offset = np.array([float(vals.get(s, offset[i])) for i, s in enumerate(sensor_cols)], dtype=float)
        self.offset = _clip(offset, self.max_abs)

    def active(self) -> bool:
        return self.enabled and self.mode not in {"none", "off", "false"}

    def current(self) -> np.ndarray:
        if not self.active():
            return np.zeros(len(self.sensor_cols), dtype=float)
        if self.mode == "offset":
            corr = self.offset
        elif self.mode in {"offset_ema", "ema", "ema_bias"}:
            corr = self.offset + self.bias
        else:
            # Unknown modes intentionally fail softly at runtime config review.
            # Ridge/Huber residuals should be added only after residual analysis shows need.
            raise ValueError(f"unsupported correction.mode={self.mode}; use none, offset, or offset_ema")
        return _clip(corr, self.max_abs)

    def update(self, base_pred: np.ndarray, measured: np.ndarray) -> None:
        if not self.active() or self.mode not in {"offset_ema", "ema", "ema_bias"}:
            return
        # Use only information available after the current measurement arrives.
        residual_after_offset = np.asarray(measured, dtype=float) - np.asarray(base_pred, dtype=float) - self.offset
        self.n_update += 1
        if self.n_update <= self.warmup_steps:
            return
        self.bias = (1.0 - self.alpha) * self.bias + self.alpha * residual_after_offset
        self.bias = _clip(self.bias, self.max_abs)


def residual_long_from_monitor_files(files: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in files:
        df = pd.read_csv(path)
        sensors = sorted({c.replace("meas_", "") for c in df.columns if c.startswith("meas_")})
        for _, r in df.iterrows():
            for s in sensors:
                meas = float(r[f"meas_{s}"])
                base_col = f"base_{s}" if f"base_{s}" in df.columns else f"pred_{s}"
                final_col = f"final_{s}" if f"final_{s}" in df.columns else f"pred_{s}"
                base = float(r[base_col])
                final = float(r[final_col])
                row = {
                    "case_id": path.stem.replace("_monitor", ""),
                    "source_file": str(path),
                    "time": float(r.get("time", np.nan)),
                    "sensor": s,
                    "T_meas": meas,
                    "T_base": base,
                    "T_final": final,
                    # residual is measured - prediction. Positive means prediction is low.
                    "residual_base": meas - base,
                    "residual_final": meas - final,
                    "abs_err_base": abs(base - meas),
                    "abs_err_final": abs(final - meas),
                }
                for c in df.columns:
                    if c in {"time"} or c.startswith(("meas_", "base_", "final_", "pred_", "err_", "corr_")):
                        continue
                    val = r[c]
                    if isinstance(val, (int, float, np.integer, np.floating)) and np.isfinite(val):
                        row[c] = float(val)
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_residuals(residual_long: pd.DataFrame, quantile: float = 0.90) -> pd.DataFrame:
    if residual_long.empty:
        return pd.DataFrame()
    rows = []
    for sensor, g in residual_long.groupby("sensor"):
        rb = g["residual_base"].to_numpy(float)
        rf = g["residual_final"].to_numpy(float)
        rows.append(
            {
                "sensor": sensor,
                "n": int(len(g)),
                "median_residual_base": float(np.median(rb)),
                "mean_residual_base": float(np.mean(rb)),
                "mae_base": float(np.mean(np.abs(rb))),
                "rmse_base": float(np.sqrt(np.mean(rb**2))),
                "mae_final": float(np.mean(np.abs(rf))),
                "rmse_final": float(np.sqrt(np.mean(rf**2))),
                f"abs_residual_q{int(quantile * 100)}_final": float(np.quantile(np.abs(rf), quantile)),
            }
        )
    return pd.DataFrame(rows)


def write_offset_artifact(summary: pd.DataFrame, out_path: Path, max_abs: float = 2.0) -> Path:
    offsets = {}
    for _, r in summary.iterrows():
        offsets[str(r["sensor"])] = float(np.clip(r["median_residual_base"], -max_abs, max_abs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "type": "sensor_offset",
                "definition": "offset added to base prediction; fitted as median(T_meas - T_base)",
                "max_abs_correction": max_abs,
                "sensor_offsets": offsets,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return out_path
