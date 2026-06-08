from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_loss(history: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    fig = plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE on scaled delta T")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_rollout(time: np.ndarray, truth: np.ndarray, pred: np.ndarray, sensors: list[str], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 5))
    for i, s in enumerate(sensors):
        plt.plot(time, truth[:, i], linestyle="--", label=f"CAE {s}")
        plt.plot(time, pred[:, i], label=f"pred {s}")
    plt.xlabel("time")
    plt.ylabel("temperature")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_error_heatmap(errors: np.ndarray, sensors: list[str], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 3.5))
    plt.imshow(errors.T, aspect="auto")
    plt.yticks(range(len(sensors)), sensors)
    plt.xlabel("time index")
    plt.ylabel("sensor")
    plt.colorbar(label="prediction error")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_graph_matrix(matrix: np.ndarray, labels: list[str], out_path: str | Path, title: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5, 4))
    plt.imshow(matrix, aspect="equal")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("src / neighbor j")
    plt.ylabel("dst / node i")
    plt.title(title)
    plt.colorbar(label="relative conductance")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_prediction(time: np.ndarray, pred: np.ndarray, sensors: list[str], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 5))
    for i, s in enumerate(sensors):
        plt.plot(time, pred[:, i], label=s)
    plt.xlabel("time")
    plt.ylabel("predicted temperature")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
