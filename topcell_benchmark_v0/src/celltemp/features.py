from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .control import effective_control_series
from .data import Case, case_control_series
from .preprocess import Preprocessor


def model_control_series(case: Case, control_cols: list[str], feature_cfg: dict | None = None) -> np.ndarray:
    """Controls used as model input, after optional actuator lag filtering."""
    raw = case_control_series(case, control_cols)
    times = case.frame.iloc[:, 0].to_numpy(float)
    return effective_control_series(times, raw, control_cols, feature_cfg or {})


class WindowDataset(Dataset):
    """One-step windows from CAE trajectories.

    The model learns one-step ΔT.  When rollout_steps > 1, each sample also
    carries a short future sequence used by the optional rollout loss.
    """

    def __init__(
        self,
        cases,
        sensor_cols: list[str],
        control_cols: list[str],
        pre: Preprocessor,
        history: int,
        rollout_steps: int = 1,
        feature_cfg: dict | None = None,
    ):
        if history < 1:
            raise ValueError("history_steps must be >= 1")
        rollout_steps = max(1, int(rollout_steps))
        feature_cfg = feature_cfg or {}
        self.samples: list[dict[str, np.ndarray]] = []
        for c in cases:
            temp = c.frame[sensor_cols].to_numpy(float)
            control = model_control_series(c, control_cols, feature_cfg)
            for i in range(history - 1, len(temp) - 1):
                t_hist = temp[i - history + 1 : i + 1]
                u_hist = control[i - history + 1 : i + 1]
                u_next = control[i + 1]
                target_delta = temp[i + 1] - temp[i]

                future_temps = np.zeros((rollout_steps, temp.shape[1]), dtype=float)
                future_controls = np.zeros((rollout_steps, control.shape[1]), dtype=float)
                future_mask = np.zeros(rollout_steps, dtype=float)
                for k in range(rollout_steps):
                    j = i + 1 + k
                    if j < len(temp):
                        future_temps[k] = temp[j]
                        future_controls[k] = control[j]
                        future_mask[k] = 1.0
                    else:
                        future_temps[k] = temp[-1]
                        future_controls[k] = control[-1]

                self.samples.append(
                    {
                        "state_hist": pre.t(t_hist).astype("float32"),
                        "control_hist": pre.u(u_hist).astype("float32"),
                        "control_next": pre.u(u_next[None, :])[0].astype("float32"),
                        "target_delta": pre.dt(target_delta[None, :])[0].astype("float32"),
                        "target_temp": pre.t(temp[i + 1][None, :])[0].astype("float32"),
                        "future_temps": pre.t(future_temps).astype("float32"),
                        "future_controls": pre.u(future_controls).astype("float32"),
                        "future_mask": future_mask.astype("float32"),
                    }
                )
        if not self.samples:
            raise ValueError("no windows created; check history_steps and time length")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: torch.tensor(v) for k, v in self.samples[idx].items()}


def make_step_batch(
    temp_history: np.ndarray,
    control_history: np.ndarray,
    control_next: np.ndarray,
    pre: Preprocessor,
) -> dict[str, torch.Tensor]:
    return {
        "state_hist": torch.tensor(pre.t(temp_history)[None, :, :], dtype=torch.float32),
        "control_hist": torch.tensor(pre.u(control_history)[None, :, :], dtype=torch.float32),
        "control_next": torch.tensor(pre.u(control_next[None, :]), dtype=torch.float32),
    }
