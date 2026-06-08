from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import as_path, load_yaml, project_root_from_config
from .control import effective_control_series, interpolate_table, resample_log_table
from .correction import WeakCorrector
from .features import make_step_batch
from .models import build_model
from .plots import plot_error_heatmap, plot_prediction, plot_rollout


def _optional_str(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _append_warning(warnings: list[str], text: str) -> None:
    if text and text not in warnings:
        warnings.append(text)


def _load_schedule(path: str, root: Path, control_cols: list[str]) -> pd.DataFrame:
    p = as_path(path, root)
    table = pd.read_csv(p)
    if "time" not in table.columns:
        raise ValueError(f"schedule_csv must contain time: {p}")
    missing = [c for c in control_cols if c not in table.columns]
    if missing:
        raise ValueError(f"schedule_csv missing {missing}: {p}")
    table = table[["time", *control_cols]].astype(float).sort_values("time")
    if not np.isfinite(table.to_numpy(float)).all():
        raise ValueError(f"schedule_csv contains NaN or inf: {p}")
    if len(table) < 1 or (len(table) > 1 and not np.all(np.diff(table["time"]) > 0)):
        raise ValueError(f"schedule_csv time must be strictly increasing: {p}")
    return table


def _raw_control_series_for_times(
    row: pd.Series,
    times: np.ndarray,
    root: Path,
    control_cols: list[str],
    cache: dict[str, pd.DataFrame],
    pred_cfg: dict,
) -> np.ndarray:
    schedule = _optional_str(row.get("schedule_csv", ""))
    if schedule:
        if schedule not in cache:
            cache[schedule] = _load_schedule(schedule, root, control_cols)
        method = str(pred_cfg.get("schedule_interpolation", "previous"))
        return interpolate_table(cache[schedule], times, control_cols, method=method)
    return np.repeat(np.array([[float(row[c]) for c in control_cols]], dtype=float), len(times), axis=0)


def _ood_warning(pre, u: np.ndarray, control_cols: list[str]) -> str:
    mask = pre.raw_control_out_of_range(u) if hasattr(pre, "raw_control_out_of_range") else pre.control_out_of_range(u)
    mn = getattr(pre, "raw_control_min", pre.control_min)
    mx = getattr(pre, "raw_control_max", pre.control_max)
    if not np.any(mask):
        return ""
    parts = []
    for j, outside in enumerate(mask):
        if outside:
            parts.append(f"{control_cols[j]}={u[j]:.4g} outside train range [{mn[j]:.4g}, {mx[j]:.4g}]")
    return "; ".join(parts)


def _temp_warning(pre, temp: np.ndarray, sensor_cols: list[str], prefix: str = "init") -> str:
    mask = pre.temp_out_of_range(temp)
    if not np.any(mask):
        return ""
    parts = []
    for j, outside in enumerate(mask):
        if outside:
            parts.append(f"{prefix}_{sensor_cols[j]}={temp[j]:.4g} outside train range [{pre.temp_min[j]:.4g}, {pre.temp_max[j]:.4g}]")
    return "; ".join(parts)


def _control_step_threshold(pred_cfg: dict, pre, control_cols: list[str]) -> np.ndarray:
    cfg = pred_cfg.get("max_control_step")
    learned = getattr(pre, "raw_control_step_max", getattr(pre, "control_step_max", np.zeros(len(control_cols))))
    if cfg is None:
        return np.asarray(learned, dtype=float)
    if isinstance(cfg, dict):
        return np.array([float(cfg.get(c, learned[i])) for i, c in enumerate(control_cols)], dtype=float)
    return np.repeat(float(cfg), len(control_cols))


def _step_warning(prev_u: np.ndarray, u: np.ndarray, threshold: np.ndarray, control_cols: list[str]) -> str:
    diff = np.abs(np.asarray(u) - np.asarray(prev_u))
    mask = diff > threshold + 1e-12
    if not np.any(mask):
        return ""
    parts = []
    for j, outside in enumerate(mask):
        if outside:
            parts.append(f"Δ{control_cols[j]}={diff[j]:.4g} exceeds learned/specified step limit {threshold[j]:.4g}")
    return "; ".join(parts)


def _check_dt(dt: float, metadata: dict, pred_cfg: dict) -> None:
    train_dt = metadata.get("train_dt")
    if train_dt is None:
        return
    if abs(float(dt) - float(train_dt)) <= 1e-9:
        return
    if bool(pred_cfg.get("allow_dt_mismatch", False)):
        return
    raise ValueError(
        f"prediction dt={dt} differs from trained dt={train_dt}. "
        "Use the training dt, resample the schedule/log, or set prediction.allow_dt_mismatch=true after reviewing the risk."
    )


def _history_until(arr: np.ndarray, end_index: int, history: int) -> np.ndarray:
    """Return padded history ending at end_index inclusive."""
    start = max(0, end_index - history + 1)
    hist = arr[start : end_index + 1]
    if len(hist) < history:
        pad = np.repeat(hist[0][None, :], history - len(hist), axis=0)
        hist = np.vstack([pad, hist])
    return hist


def _load_package(cfg: dict, root: Path):
    pkg = as_path(cfg["model_package"]["path"], root)
    train_cfg = load_yaml(pkg / "config.yaml")
    with open(pkg / "preprocessor.pkl", "rb") as f:
        pre = pickle.load(f)
    metadata = {}
    meta_path = pkg / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return pkg, train_cfg, pre, metadata


def _load_model(train_cfg: dict, pkg: Path, sensor_cols: list[str], control_cols: list[str], history: int, device):
    model = build_model(
        train_cfg["model"],
        n_sensors=len(sensor_cols),
        n_controls=len(control_cols),
        history=history,
        sensor_cols=sensor_cols,
        control_cols=control_cols,
        root=pkg,
    ).to(device)
    model.load_state_dict(torch.load(pkg / "model.pt", map_location=device))
    model.eval()
    return model


def _forecast_case(
    model,
    row: pd.Series,
    idx: int,
    root: Path,
    pre,
    metadata: dict,
    train_cfg: dict,
    pred_cfg: dict,
    out_dir: Path,
    device,
    make_plots: bool,
    schedule_cache: dict[str, pd.DataFrame],
) -> dict:
    sensor_cols = list(train_cfg["data"]["sensor_cols"])
    control_cols = list(train_cfg["data"]["control_cols"])
    history = int(train_cfg["features"].get("history_steps", 3))
    feature_cfg = train_cfg.get("features", {})

    case_id = str(row.get("case_id", f"case_{idx}"))
    dt = float(row.get("dt", pred_cfg.get("default_dt", 1.0)))
    _check_dt(dt, metadata, pred_cfg)
    t_end = float(row.get("t_end", pred_cfg.get("default_t_end", 10.0)))
    times = np.arange(0.0, t_end + 1e-9, dt)
    raw_controls = _raw_control_series_for_times(row, times, root, control_cols, schedule_cache, pred_cfg)
    model_controls = effective_control_series(times, raw_controls, control_cols, feature_cfg)

    cur = np.array([float(row[f"init_{s}"]) for s in sensor_cols], dtype=float)
    temp_hist = np.repeat(cur[None, :], history, axis=0)
    control_hist = np.repeat(model_controls[0][None, :], history, axis=0)
    rows = [
        {
            "time": float(times[0]),
            **{s: cur[i] for i, s in enumerate(sensor_cols)},
            **{c: raw_controls[0, j] for j, c in enumerate(control_cols)},
        }
    ]
    warnings: list[str] = []

    _append_warning(warnings, _temp_warning(pre, cur, sensor_cols))
    w = _ood_warning(pre, raw_controls[0], control_cols)
    if w:
        _append_warning(warnings, f"t=0: {w}")
    train_tmax = float(metadata.get("train_time_end_max", getattr(pre, "time_end_max", 0.0)))
    if bool(pred_cfg.get("warn_if_longer_than_train", True)) and train_tmax > 0 and t_end > train_tmax + 1e-9:
        _append_warning(warnings, f"t_end={t_end:g} exceeds max training time {train_tmax:g}; long rollout may drift")

    step_limit = _control_step_threshold(pred_cfg, pre, control_cols)
    for k, t in enumerate(times[1:], start=1):
        raw_u = raw_controls[k]
        model_u = model_controls[k]
        w = _ood_warning(pre, raw_u, control_cols)
        if w:
            _append_warning(warnings, f"t={t:g}: {w}")
        w = _step_warning(raw_controls[k - 1], raw_u, step_limit, control_cols)
        if w:
            _append_warning(warnings, f"t={t:g}: {w}")
        batch = make_step_batch(temp_hist, control_hist, model_u, pre)
        batch = {key: val.to(device) for key, val in batch.items()}
        delta_scaled = model(batch).detach().cpu().numpy()[0]
        delta = pre.inv_dt(delta_scaled[None, :])[0]
        cur = cur + delta
        clamp_cfg = pred_cfg.get("clamp_temperature", {})
        if bool(clamp_cfg.get("enabled", False)):
            cur = np.clip(cur, float(clamp_cfg.get("min", -1e12)), float(clamp_cfg.get("max", 1e12)))
        rows.append(
            {
                "time": float(t),
                **{s: cur[i] for i, s in enumerate(sensor_cols)},
                **{c: raw_u[j] for j, c in enumerate(control_cols)},
            }
        )
        temp_hist = np.vstack([temp_hist[1:], cur])
        control_hist = np.vstack([control_hist[1:], model_u])

    out = pd.DataFrame(rows)
    csv_path = out_dir / f"{case_id}_prediction.csv"
    out.to_csv(csv_path, index=False)
    if warnings:
        (out_dir / "warnings" / f"{case_id}.txt").write_text("\n".join(warnings), encoding="utf-8")
    if make_plots:
        plot_prediction(out["time"].to_numpy(float), out[sensor_cols].to_numpy(float), sensor_cols, out_dir / "plots" / f"{case_id}.png")
    return {
        "case_id": case_id,
        "mode": "forecast",
        "csv": str(csv_path),
        "n_warnings": len(warnings),
        "warning": " | ".join(warnings[:5]),
        **{f"final_{s}": float(out[s].iloc[-1]) for s in sensor_cols},
    }


def _load_monitor_log(path: str, root: Path, sensor_cols: list[str], control_cols: list[str]) -> pd.DataFrame:
    p = as_path(path, root)
    df = pd.read_csv(p)
    required = ["time", *sensor_cols, *control_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"monitor log missing {missing}: {p}")
    df = df[required].astype(float).sort_values("time")
    if len(df) < 2:
        raise ValueError(f"monitor log requires at least two rows: {p}")
    if not np.isfinite(df.to_numpy(float)).all():
        raise ValueError(f"monitor log contains NaN/inf: {p}")
    if not np.all(np.diff(df["time"].to_numpy(float)) > 0):
        raise ValueError(f"monitor log time must be strictly increasing: {p}")
    return df


def _prepare_monitor_log(raw_log: pd.DataFrame, pred_cfg: dict, sensor_cols: list[str], control_cols: list[str], dt: float) -> pd.DataFrame:
    resample_cfg = pred_cfg.get("resample", {}) or {}
    if bool(resample_cfg.get("enabled", False)):
        return resample_log_table(
            raw_log,
            sensor_cols,
            control_cols,
            dt=dt,
            control_method=str(resample_cfg.get("control", pred_cfg.get("schedule_interpolation", "previous"))),
            temperature_method=str(resample_cfg.get("temperature", "linear")),
        )
    time = raw_log["time"].to_numpy(float)
    if not np.allclose(np.diff(time), dt, rtol=1e-4, atol=1e-8):
        raise ValueError("monitor log dt does not match prediction dt; enable prediction.resample.enabled=true")
    return raw_log


def _monitor_case(
    model,
    row: pd.Series,
    idx: int,
    root: Path,
    pre,
    metadata: dict,
    train_cfg: dict,
    pred_cfg: dict,
    out_dir: Path,
    device,
    make_plots: bool,
) -> dict:
    sensor_cols = list(train_cfg["data"]["sensor_cols"])
    control_cols = list(train_cfg["data"]["control_cols"])
    history = int(train_cfg["features"].get("history_steps", 3))
    feature_cfg = train_cfg.get("features", {})
    case_id = str(row.get("case_id", f"monitor_{idx}"))
    log_path = _optional_str(row.get("log_csv", row.get("monitor_csv", "")))
    if not log_path:
        raise ValueError("prediction.mode=monitor requires log_csv column in input_table")

    dt = float(row.get("dt", pred_cfg.get("default_dt", metadata.get("train_dt", 1.0))))
    _check_dt(dt, metadata, pred_cfg)
    raw_log = _load_monitor_log(log_path, root, sensor_cols, control_cols)
    log = _prepare_monitor_log(raw_log, pred_cfg, sensor_cols, control_cols, dt)
    times = log["time"].to_numpy(float)
    truth = log[sensor_cols].to_numpy(float)
    raw_controls = log[control_cols].to_numpy(float)
    model_controls = effective_control_series(times, raw_controls, control_cols, feature_cfg)
    corrector = WeakCorrector(sensor_cols, pred_cfg.get("correction", {}), root)

    warnings: list[str] = []
    _append_warning(warnings, _temp_warning(pre, truth[0], sensor_cols, prefix="meas0"))
    step_limit = _control_step_threshold(pred_cfg, pre, control_cols)
    for k in range(len(times)):
        w = _ood_warning(pre, raw_controls[k], control_cols)
        if w:
            _append_warning(warnings, f"t={times[k]:g}: {w}")
        if k > 0:
            w = _step_warning(raw_controls[k - 1], raw_controls[k], step_limit, control_cols)
            if w:
                _append_warning(warnings, f"t={times[k]:g}: {w}")
    train_tmax = float(metadata.get("train_time_end_max", getattr(pre, "time_end_max", 0.0)))
    if bool(pred_cfg.get("warn_if_longer_than_train", True)) and train_tmax > 0 and (times[-1] - times[0]) > train_tmax + 1e-9:
        _append_warning(warnings, f"monitor duration={times[-1] - times[0]:g} exceeds max training time {train_tmax:g}")

    base_pred = np.full_like(truth, np.nan, dtype=float)
    final_pred = np.full_like(truth, np.nan, dtype=float)
    corr_used = np.zeros_like(truth, dtype=float)
    base_pred[0] = truth[0]
    final_pred[0] = truth[0]
    rows = []
    for k in range(1, len(times)):
        temp_hist = _history_until(truth, k - 1, history)
        control_hist = _history_until(model_controls, k - 1, history)
        batch = make_step_batch(temp_hist, control_hist, model_controls[k], pre)
        batch = {key: val.to(device) for key, val in batch.items()}
        delta_scaled = model(batch).detach().cpu().numpy()[0]
        delta = pre.inv_dt(delta_scaled[None, :])[0]
        base_pred[k] = truth[k - 1] + delta
        correction = corrector.current()
        corr_used[k] = correction
        final_pred[k] = base_pred[k] + correction

        err_base = base_pred[k] - truth[k]
        err_final = final_pred[k] - truth[k]
        out_row = {
            "time": float(times[k]),
            **{c: raw_controls[k, j] for j, c in enumerate(control_cols)},
            "residual_mae_base": float(np.mean(np.abs(err_base))),
            "residual_rmse_base": float(np.sqrt(np.mean(err_base**2))),
            "residual_mae": float(np.mean(np.abs(err_final))),
            "residual_rmse": float(np.sqrt(np.mean(err_final**2))),
            "mean_abs_correction": float(np.mean(np.abs(correction))),
        }
        for i, s in enumerate(sensor_cols):
            out_row[f"meas_{s}"] = truth[k, i]
            out_row[f"base_{s}"] = base_pred[k, i]
            out_row[f"corr_{s}"] = correction[i]
            out_row[f"final_{s}"] = final_pred[k, i]
            # Backward-compatible aliases: pred/err mean corrected prediction/error.
            out_row[f"pred_{s}"] = final_pred[k, i]
            out_row[f"err_base_{s}"] = err_base[i]
            out_row[f"err_final_{s}"] = err_final[i]
            out_row[f"err_{s}"] = err_final[i]
        rows.append(out_row)
        corrector.update(base_pred[k], truth[k])

    out = pd.DataFrame(rows)
    csv_path = out_dir / f"{case_id}_monitor.csv"
    out.to_csv(csv_path, index=False)
    if warnings:
        (out_dir / "warnings" / f"{case_id}.txt").write_text("\n".join(warnings), encoding="utf-8")
    if make_plots:
        plot_rollout(times, truth, final_pred, sensor_cols, out_dir / "plots" / f"{case_id}_monitor.png")
        plot_error_heatmap(final_pred - truth, sensor_cols, out_dir / "plots" / f"{case_id}_monitor_error.png")
    return {
        "case_id": case_id,
        "mode": "monitor",
        "csv": str(csv_path),
        "n_warnings": len(warnings),
        "warning": " | ".join(warnings[:5]),
        "residual_mae_base": float(out["residual_mae_base"].mean()) if len(out) else np.nan,
        "residual_rmse_base": float(out["residual_rmse_base"].mean()) if len(out) else np.nan,
        "residual_mae": float(out["residual_mae"].mean()) if len(out) else np.nan,
        "residual_rmse": float(out["residual_rmse"].mean()) if len(out) else np.nan,
        "mean_abs_correction": float(out["mean_abs_correction"].mean()) if len(out) else np.nan,
    }


@torch.no_grad()
def run_predict(cfg: dict, config_path: str | Path) -> Path:
    root = project_root_from_config(config_path)
    pred_cfg = cfg["prediction"]
    if pred_cfg.get("num_threads") is not None:
        torch.set_num_threads(int(pred_cfg.get("num_threads", 1)))

    pkg, train_cfg, pre, metadata = _load_package(cfg, root)
    sensor_cols = list(train_cfg["data"]["sensor_cols"])
    control_cols = list(train_cfg["data"]["control_cols"])
    history = int(train_cfg["features"].get("history_steps", 3))

    device_name = pred_cfg.get("device", "cpu")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model = _load_model(train_cfg, pkg, sensor_cols, control_cols, history, device)

    input_table = as_path(pred_cfg["input_table"], root)
    conds = pd.read_csv(input_table)
    out_dir = as_path(pred_cfg.get("output_dir", "outputs/predictions"), root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "warnings").mkdir(parents=True, exist_ok=True)

    make_plots = bool(cfg.get("postprocess", {}).get("make_plots", True))
    mode = str(pred_cfg.get("mode", "forecast")).lower()
    if mode not in {"forecast", "monitor"}:
        raise ValueError("prediction.mode must be forecast or monitor")

    summaries = []
    cache: dict[str, pd.DataFrame] = {}
    for idx, row in conds.iterrows():
        if mode == "monitor":
            summaries.append(_monitor_case(model, row, idx, root, pre, metadata, train_cfg, pred_cfg, out_dir, device, make_plots))
        else:
            summaries.append(_forecast_case(model, row, idx, root, pre, metadata, train_cfg, pred_cfg, out_dir, device, make_plots, cache))

    pd.DataFrame(summaries).to_csv(out_dir / "prediction_summary.csv", index=False)
    if metadata:
        with open(out_dir / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    return out_dir
