from __future__ import annotations

import copy
import json
import pickle
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import as_path, project_root_from_config, save_yaml
from .data import infer_common_dt, load_cases, save_data_summary, save_split, split_cases
from .evaluate import evaluate_rollout
from .features import WindowDataset
from .models import GraphRC, build_model
from .plots import plot_graph_matrix, plot_loss
from .preprocess import fit_preprocessor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _package_config(cfg: dict, root: Path, pkg: Path) -> dict:
    out = copy.deepcopy(cfg)
    graph = out.get("model", {}).get("graph", {}) or {}
    if graph:
        graph_dir = pkg / "graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        for key in ["node_table", "edge_table"]:
            if graph.get(key):
                src = as_path(graph[key], root)
                if src.exists():
                    dst = graph_dir / src.name
                    shutil.copy2(src, dst)
                    graph[key] = f"graph/{src.name}"
        out.setdefault("model", {})["graph"] = graph
    return out


def _save_graph_outputs(model: torch.nn.Module, cfg: dict, run_dir: Path) -> None:
    if not isinstance(model, GraphRC):
        return
    labels = list(cfg["data"]["sensor_cols"])
    prior = model.conductance_prior.detach().cpu().numpy() * model.edge_mask.detach().cpu().numpy()
    learned = model.conductance_matrix()

    graph_dir = run_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    diag_rows = []
    for i, dst in enumerate(labels):
        for j, src in enumerate(labels):
            if model.edge_mask[i, j].item() <= 0:
                continue
            ratio = float(learned[i, j] / max(prior[i, j], 1e-12))
            status = "ok"
            if ratio > 10.0:
                status = "learned_much_larger_than_prior"
            elif ratio < 0.1:
                status = "learned_much_smaller_than_prior"
            row = {
                "src": src,
                "dst": dst,
                "conductance_prior": float(prior[i, j]),
                "conductance_learned": float(learned[i, j]),
                "learned_over_prior": ratio,
                "status": status,
            }
            rows.append(row)
            if status != "ok":
                diag_rows.append(row)
    pd.DataFrame(rows).to_csv(graph_dir / "learned_conductance.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(graph_dir / "graph_diagnostics.csv", index=False)

    source_rows = []
    source_weight = model.source_weight.detach().cpu().numpy()
    mass = model.thermal_mass.detach().cpu().numpy()
    for i, sensor in enumerate(labels):
        row = {"sensor": sensor, "thermal_mass": float(mass[i])}
        for j, ctrl in enumerate(cfg["data"]["control_cols"]):
            row[f"{ctrl}_weight"] = float(source_weight[i, j])
        source_rows.append(row)
    pd.DataFrame(source_rows).to_csv(graph_dir / "source_weight.csv", index=False)

    plot_graph_matrix(prior, labels, run_dir / "plots" / "graph_conductance_prior.png", "Graph-RC prior conductance")
    plot_graph_matrix(learned, labels, run_dir / "plots" / "graph_conductance_learned.png", "Graph-RC learned conductance")


def _ranges_from_arrays(names: list[str], mn: np.ndarray, mx: np.ndarray) -> dict[str, list[float]]:
    return {name: [float(mn[i]), float(mx[i])] for i, name in enumerate(names)}


def _rollout_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor], pre, steps: int) -> torch.Tensor:
    """Small optional rollout loss in scaled-temperature space.

    It is deliberately lightweight.  It reuses the one-step model recursively
    for a few future steps and compares predicted scaled temperatures against
    future CAE temperatures carried by WindowDataset.
    """
    steps = min(int(steps), batch["future_controls"].shape[1])
    device = batch["state_hist"].device
    temp_mean = torch.tensor(pre.temp.mean, dtype=torch.float32, device=device)
    temp_std = torch.tensor(pre.temp.std, dtype=torch.float32, device=device)
    delta_mean = torch.tensor(pre.delta.mean, dtype=torch.float32, device=device)
    delta_std = torch.tensor(pre.delta.std, dtype=torch.float32, device=device)

    temp_hist = batch["state_hist"]
    control_hist = batch["control_hist"]
    loss_sum = torch.tensor(0.0, device=device)
    denom = torch.tensor(0.0, device=device)

    for k in range(steps):
        u_next = batch["future_controls"][:, k, :]
        step_batch = {"state_hist": temp_hist, "control_hist": control_hist, "control_next": u_next}
        delta_scaled = model(step_batch)
        cur_scaled = temp_hist[:, -1, :]
        next_phys = cur_scaled * temp_std + temp_mean + delta_scaled * delta_std + delta_mean
        next_scaled = (next_phys - temp_mean) / temp_std

        mask = batch["future_mask"][:, k].view(-1, 1)
        err = (next_scaled - batch["future_temps"][:, k, :]) ** 2
        loss_sum = loss_sum + (err * mask).sum()
        denom = denom + mask.sum() * err.shape[1]

        temp_hist = torch.cat([temp_hist[:, 1:, :], next_scaled.unsqueeze(1)], dim=1)
        control_hist = torch.cat([control_hist[:, 1:, :], u_next.unsqueeze(1)], dim=1)
    return loss_sum / torch.clamp(denom, min=1.0)





def _train_dt(cfg: dict, cases) -> float | None:
    if cfg.get("data", {}).get("dt") is not None:
        return float(cfg["data"]["dt"])
    return infer_common_dt(cases, cfg["data"]["columns"][0])


def run_train(cfg: dict, config_path: str | Path) -> Path:
    root = project_root_from_config(config_path)
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_seed(seed)
    if cfg.get("train", {}).get("num_threads") is not None:
        torch.set_num_threads(int(cfg["train"].get("num_threads", 1)))

    sensor_cols = list(cfg["data"]["sensor_cols"])
    control_cols = list(cfg["data"]["control_cols"])
    history = int(cfg["features"].get("history_steps", 3))

    run_name = cfg.get("project", {}).get("run_name", cfg.get("model", {}).get("name", "run"))
    run_dir = as_path(cfg.get("project", {}).get("output_dir", "outputs/runs"), root) / run_name
    if run_dir.exists() and bool(cfg.get("project", {}).get("overwrite_run", False)):
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "resolved_config.yaml")

    cases = load_cases(cfg["data"], root)
    train_dt = _train_dt(cfg, cases)
    save_data_summary(cases, sensor_cols, run_dir / "data_summary.csv")
    split = split_cases(cases, cfg.get("split", {}), seed=seed)
    save_split(split, run_dir / "split.csv")

    pre = fit_preprocessor(split["train"], sensor_cols, control_cols, cfg.get("features", {}))
    rollout_weight = float(cfg.get("train", {}).get("rollout_loss_weight", 0.0))
    rollout_steps = int(cfg.get("train", {}).get("rollout_loss_steps", 3)) if rollout_weight > 0 else 1
    graph_prior_weight = float(cfg.get("train", {}).get("graph_prior_loss_weight", 0.0))
    train_ds = WindowDataset(split["train"], sensor_cols, control_cols, pre, history, rollout_steps=rollout_steps, feature_cfg=cfg.get("features", {}))
    val_source = split["val"] or split["train"]
    val_ds = WindowDataset(val_source, sensor_cols, control_cols, pre, history, rollout_steps=1, feature_cfg=cfg.get("features", {}))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"].get("batch_size", 32)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["train"].get("batch_size", 32)), shuffle=False)

    device = choose_device(cfg.get("train", {}).get("device", "auto"))
    model = build_model(
        cfg["model"],
        n_sensors=len(sensor_cols),
        n_controls=len(control_cols),
        history=history,
        sensor_cols=sensor_cols,
        control_cols=control_cols,
        root=root,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    loss_fn = torch.nn.MSELoss()
    max_epochs = int(cfg["train"].get("epochs", 200))
    patience = int(cfg["train"].get("patience", 30))
    grad_clip = cfg["train"].get("gradient_clip", 1.0)
    print_every = int(cfg["train"].get("print_every", 50))

    best_val = float("inf")
    best_state = None
    wait = 0
    history_rows = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        train_step_losses = []
        train_rollout_losses = []
        train_graph_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            step_loss = loss_fn(model(batch), batch["target_delta"])
            if rollout_weight > 0:
                roll_loss = _rollout_loss(model, batch, pre, rollout_steps)
            else:
                roll_loss = torch.tensor(0.0, device=device)
            if graph_prior_weight > 0 and hasattr(model, "graph_prior_loss"):
                graph_loss = model.graph_prior_loss()
            else:
                graph_loss = torch.tensor(0.0, device=device)
            loss = step_loss + rollout_weight * roll_loss + graph_prior_weight * graph_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_step_losses.append(float(step_loss.detach().cpu()))
            train_rollout_losses.append(float(roll_loss.detach().cpu()))
            train_graph_losses.append(float(graph_loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_losses.append(float(loss_fn(model(batch), batch["target_delta"]).detach().cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_step_loss": float(np.mean(train_step_losses)),
                "train_rollout_loss": float(np.mean(train_rollout_losses)),
                "train_graph_prior_loss": float(np.mean(train_graph_losses)),
                "val_loss": val_loss,
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if print_every > 0 and (epoch == 1 or epoch % print_every == 0):
            print(f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if wait >= patience:
            print(f"early stopping at epoch={epoch}; best val={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(history_rows).to_csv(run_dir / "train_history.csv", index=False)
    plot_loss(history_rows, run_dir / "plots" / "loss_curve.png")

    summary = {}
    if split["val"]:
        summary.update(evaluate_rollout(model, split["val"], pre, cfg, run_dir, device, "val"))
    if split["test"]:
        summary.update(evaluate_rollout(model, split["test"], pre, cfg, run_dir, device, "test"))
    if not summary:
        summary.update(evaluate_rollout(model, split["train"], pre, cfg, run_dir, device, "train"))
    _save_graph_outputs(model, cfg, run_dir)

    pkg = run_dir / "model_package"
    pkg.mkdir(parents=True, exist_ok=True)
    package_cfg = _package_config(cfg, root, pkg)
    torch.save(model.state_dict(), pkg / "model.pt")
    with open(pkg / "preprocessor.pkl", "wb") as f:
        pickle.dump(pre, f)
    save_yaml(package_cfg, pkg / "config.yaml")
    with open(pkg / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": cfg["model"]["name"],
                "sensor_cols": sensor_cols,
                "control_cols": control_cols,
                "history_steps": history,
                "train_dt": train_dt,
                "train_time_end_max": float(pre.time_end_max),
                "train_temperature_ranges": _ranges_from_arrays(sensor_cols, pre.temp_min, pre.temp_max),
                "train_control_ranges": _ranges_from_arrays(control_cols, pre.raw_control_min, pre.raw_control_max),
                "train_effective_control_ranges": _ranges_from_arrays(control_cols, pre.control_min, pre.control_max),
                "train_control_step_max": {control_cols[i]: float(pre.raw_control_step_max[i]) for i in range(len(control_cols))},
                "features": cfg.get("features", {}),
                "split_method": cfg.get("split", {}).get("method", "random"),
                "rollout_summary": summary,
                "note": "Scaler is fit on train cases only. Split is by whole CAE CSV file.",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return run_dir
