"""Regenerate, fit, and evaluate the complete TopCell benchmark."""

from __future__ import annotations

from pathlib import Path

from scripts.evaluate_benchmark import main as evaluate
from scripts.generate_topcell_benchmark import main as generate

from celltemp.config import load_config
from celltemp.workflows import run_forecast, run_monitor, run_train

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.yaml"


def main() -> None:
    """Run the benchmark from generated inputs through acceptance checks."""
    generate()
    config = load_config(CONFIG)
    run_train(config, CONFIG)
    run_forecast(config, CONFIG)
    run_monitor(config, CONFIG)
    evaluate()


if __name__ == "__main__":
    main()
