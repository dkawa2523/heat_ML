from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .config import as_path, project_root_from_config


def run_compare(cfg: dict, config_path: str | Path) -> Path:
    root = project_root_from_config(config_path)
    cmp_cfg = cfg.get("compare", {})
    runs_dir = as_path(cmp_cfg.get("runs_dir", "outputs/runs"), root)
    out_path = as_path(cmp_cfg.get("output", "outputs/model_leaderboard.csv"), root)

    rows = []
    for run_dir in sorted(p for p in runs_dir.glob("*") if p.is_dir()):
        row = {"run_name": run_dir.name, "run_dir": str(run_dir)}
        cfg_path = run_dir / "resolved_config.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                rcfg = yaml.safe_load(f) or {}
            row["model"] = rcfg.get("model", {}).get("name", "")
        for summary_path in sorted((run_dir / "metrics").glob("rollout_summary_*.csv")):
            # rollout_summary_by_kind_* is useful for diagnostics but should not
            # overwrite the main model-selection metrics in the leaderboard.
            if "_by_kind_" in summary_path.name:
                continue
            df = pd.read_csv(summary_path)
            for col, val in df.iloc[0].items():
                row[col] = val
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # In this project, rollout performance is the main selection criterion.
    for col in ["test_rollout_rmse", "val_rollout_rmse", "test_rollout_endpoint_mae", "val_rollout_endpoint_mae"]:
        if col in df.columns:
            df = df.sort_values(col, ascending=True)
            break
    df.to_csv(out_path, index=False)
    return out_path
