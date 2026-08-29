from __future__ import annotations

import argparse

from .config import load_config
from .workflows import run_forecast, run_monitor, run_train


def main() -> None:
    parser = argparse.ArgumentParser("celltemp")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="identify one thermal RC model")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("overrides", nargs="*", help="dot overrides, e.g. training.epochs=50")

    p_forecast = sub.add_parser("forecast", help="open-loop forecast with a thermal artifact")
    p_forecast.add_argument("--config", required=True)
    p_forecast.add_argument("overrides", nargs="*")

    p_monitor = sub.add_parser("monitor", help="causal state and sensor-bias monitoring")
    p_monitor.add_argument("--config", required=True)
    p_monitor.add_argument("overrides", nargs="*")

    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.command == "train":
        run_dir = run_train(cfg, args.config)
        print(f"saved run: {run_dir}")
    elif args.command == "forecast":
        out_dir = run_forecast(cfg, args.config)
        print(f"saved forecasts: {out_dir}")
    elif args.command == "monitor":
        out_dir = run_monitor(cfg, args.config)
        print(f"saved monitoring results: {out_dir}")


if __name__ == "__main__":
    main()
