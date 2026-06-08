from __future__ import annotations

import argparse

from .compare import run_compare
from .config import load_config
from .predict import run_predict
from .residuals import run_residuals
from .train import run_train


def main() -> None:
    parser = argparse.ArgumentParser("celltemp")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a selected model")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("overrides", nargs="*", help="dot overrides, e.g. model.name=tcn train.epochs=50")

    p_pred = sub.add_parser("predict", help="predict multiple cases from a condition table")
    p_pred.add_argument("--config", required=True)
    p_pred.add_argument("overrides", nargs="*")

    p_cmp = sub.add_parser("compare", help="collect run metrics into a leaderboard CSV")
    p_cmp.add_argument("--config", required=True)
    p_cmp.add_argument("overrides", nargs="*")

    p_res = sub.add_parser("residuals", help="analyze monitor residuals and fit weak offset correction")
    p_res.add_argument("--config", required=True)
    p_res.add_argument("overrides", nargs="*")

    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.command == "train":
        run_dir = run_train(cfg, args.config)
        print(f"saved run: {run_dir}")
    elif args.command == "predict":
        out_dir = run_predict(cfg, args.config)
        print(f"saved predictions: {out_dir}")
    elif args.command == "compare":
        out_path = run_compare(cfg, args.config)
        print(f"saved leaderboard: {out_path}")
    elif args.command == "residuals":
        out_dir = run_residuals(cfg, args.config)
        print(f"saved residual analysis: {out_dir}")


if __name__ == "__main__":
    main()
