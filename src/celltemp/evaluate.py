from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .features import make_step_batch, model_control_series
from .plots import plot_error_heatmap, plot_rollout


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(np.max(np.abs(err))),
        "endpoint_mae": float(np.mean(np.abs(err[-1]))),
    }


def per_sensor_metrics(y_true: np.ndarray, y_pred: np.ndarray, sensors: list[str], case_id: str) -> list[dict]:
    rows = []
    err = y_pred - y_true
    for i, s in enumerate(sensors):
        e = err[:, i]
        rows.append(
            {
                "case_id": case_id,
                "sensor": s,
                "mae": float(np.mean(np.abs(e))),
                "rmse": float(np.sqrt(np.mean(e**2))),
                "max_abs_error": float(np.max(np.abs(e))),
                "endpoint_error": float(e[-1]),
            }
        )
    return rows


@torch.no_grad()
def rollout_case(model, case, pre, sensor_cols, control_cols, history: int, device: torch.device, feature_cfg: dict | None = None):
    model.eval()
    truth = case.frame[sensor_cols].to_numpy(float)
    controls = model_control_series(case, control_cols, feature_cfg or {})
    pred = [truth[0].copy()]
    temp_hist = np.repeat(truth[0][None, :], history, axis=0)
    control_hist = np.repeat(controls[0][None, :], history, axis=0)

    for step in range(1, len(truth)):
        u_next = controls[step]
        batch = make_step_batch(temp_hist, control_hist, u_next, pre)
        batch = {k: v.to(device) for k, v in batch.items()}
        delta_scaled = model(batch).detach().cpu().numpy()[0]
        delta = pre.inv_dt(delta_scaled[None, :])[0]
        next_temp = pred[-1] + delta
        pred.append(next_temp)
        temp_hist = np.vstack([temp_hist[1:], next_temp])
        control_hist = np.vstack([control_hist[1:], u_next])
    return np.asarray(pred), truth


def _case_kind(case_id: str, keywords: list[str]) -> str:
    s = case_id.lower()
    return "dynamic" if any(k.lower() in s for k in keywords) else "constant_or_unknown"


def evaluate_rollout(model, cases, pre, cfg: dict, run_dir: Path, device: torch.device, split_name: str) -> dict[str, float]:
    sensor_cols = cfg["data"]["sensor_cols"]
    control_cols = cfg["data"]["control_cols"]
    history = int(cfg["features"].get("history_steps", 3))
    feature_cfg = cfg.get("features", {})

    rows = []
    sensor_rows = []
    horizon_errors = []
    pred_dir = run_dir / "eval_predictions" / split_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = cfg.get("evaluation", {})
    max_plots = int(eval_cfg.get("max_plots_per_split", 3))
    dynamic_keywords = list(eval_cfg.get("dynamic_case_keywords", []))

    for case_index, c in enumerate(cases):
        pred, truth = rollout_case(model, c, pre, sensor_cols, control_cols, history, device, feature_cfg)
        m = metrics(truth, pred)
        m["case_id"] = c.case_id
        if dynamic_keywords:
            m["case_kind"] = _case_kind(c.case_id, dynamic_keywords)
        rows.append(m)
        sensor_rows.extend(per_sensor_metrics(truth, pred, sensor_cols, c.case_id))
        horizon_errors.append(pred - truth)

        time = c.frame[cfg["data"]["columns"][0]].to_numpy(float)
        out = pd.DataFrame({"time": time})
        for i, s in enumerate(sensor_cols):
            out[f"true_{s}"] = truth[:, i]
            out[f"pred_{s}"] = pred[:, i]
            out[f"err_{s}"] = pred[:, i] - truth[:, i]
        out.to_csv(pred_dir / f"{c.case_id}.csv", index=False)

        if eval_cfg.get("make_plots", True) and case_index < max_plots:
            plot_rollout(time, truth, pred, sensor_cols, run_dir / "plots" / split_name / f"rollout_{c.case_id}.png")
            plot_error_heatmap(pred - truth, sensor_cols, run_dir / "plots" / split_name / f"error_{c.case_id}.png")

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    by_case = pd.DataFrame(rows)
    by_sensor = pd.DataFrame(sensor_rows)
    by_case.to_csv(metrics_dir / f"rollout_by_case_{split_name}.csv", index=False)
    by_sensor.to_csv(metrics_dir / f"rollout_by_sensor_{split_name}.csv", index=False)

    summary = {
        f"{split_name}_rollout_{k}": float(by_case[k].mean())
        for k in ["mae", "rmse", "max_abs_error", "endpoint_mae"]
    }
    pd.DataFrame([summary]).to_csv(metrics_dir / f"rollout_summary_{split_name}.csv", index=False)

    if dynamic_keywords and "case_kind" in by_case.columns:
        by_kind = by_case.groupby("case_kind")[["mae", "rmse", "max_abs_error", "endpoint_mae"]].mean().reset_index()
        by_kind.to_csv(metrics_dir / f"rollout_summary_by_kind_{split_name}.csv", index=False)

    min_len = min(e.shape[0] for e in horizon_errors)
    err_stack = np.stack([e[:min_len] for e in horizon_errors], axis=0)
    horizon = []
    for k in range(min_len):
        e = err_stack[:, k, :]
        horizon.append(
            {
                "time_index": k,
                "mae": float(np.mean(np.abs(e))),
                "rmse": float(np.sqrt(np.mean(e**2))),
                "max_abs_error": float(np.max(np.abs(e))),
            }
        )
    pd.DataFrame(horizon).to_csv(metrics_dir / f"horizon_error_{split_name}.csv", index=False)
    return summary
