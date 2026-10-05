"""User-facing thermal identification, prediction, monitoring, and analysis workflows."""

from pathlib import Path


def run_analysis(cfg: dict, config_path: str | Path) -> Path:
    from .analysis import run_analysis as execute

    return execute(cfg, config_path)


def run_forecast(cfg: dict, config_path: str | Path) -> Path:
    from .forecast import run_forecast as execute

    return execute(cfg, config_path)


def run_monitor(cfg: dict, config_path: str | Path) -> Path:
    from .monitor import run_monitor as execute

    return execute(cfg, config_path)


def run_train(cfg: dict, config_path: str | Path) -> Path:
    from .train import run_train as execute

    return execute(cfg, config_path)


__all__ = ["run_analysis", "run_forecast", "run_monitor", "run_train"]
