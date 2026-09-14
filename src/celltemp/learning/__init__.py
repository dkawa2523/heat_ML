"""Trajectory-level identification for the physical thermal engine."""

from .objective import (
    initial_observation_rmse,
    predict_from_initial_observation,
    predict_trajectory,
    trajectory_loss,
    trajectory_rmse,
)
from .split import split_trajectories
from .trainer import TrainingConfig, TrainingResult, fit_thermal_model

__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "fit_thermal_model",
    "initial_observation_rmse",
    "predict_from_initial_observation",
    "predict_trajectory",
    "split_trajectories",
    "trajectory_loss",
    "trajectory_rmse",
]
