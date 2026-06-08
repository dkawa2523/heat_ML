from __future__ import annotations

from pathlib import Path

from .config import as_path, project_root_from_config
from .correction import residual_long_from_monitor_files, summarize_residuals, write_offset_artifact


def run_residuals(cfg: dict, config_path: str | Path) -> Path:
    """Analyze monitor outputs and create residual tables plus a weak offset artifact."""
    root = project_root_from_config(config_path)
    res_cfg = cfg.get("residuals", {}) or {}
    pred_cfg = cfg.get("prediction", {}) or {}
    input_dir = as_path(res_cfg.get("input_dir", pred_cfg.get("output_dir", "outputs/predictions")), root)
    out_dir = as_path(res_cfg.get("output_dir", str(input_dir / "residual_analysis")), root)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = res_cfg.get("file_pattern", "*_monitor.csv")
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no monitor CSV files found: {input_dir}/{pattern}")

    residual_long = residual_long_from_monitor_files(files)
    residual_long.to_csv(out_dir / "residual_long.csv", index=False)
    summary = summarize_residuals(residual_long, quantile=float(res_cfg.get("interval_quantile", 0.90)))
    summary.to_csv(out_dir / "residual_summary.csv", index=False)
    write_offset_artifact(
        summary,
        out_dir / "offset_correction.json",
        max_abs=float(res_cfg.get("max_abs_correction", 2.0)),
    )
    return out_dir
