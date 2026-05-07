from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from uwb_localization.artifacts import get_run_dir, stage_dir
from uwb_localization.config import load_config
from uwb_localization.plotting import generate_all_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-ready plots from evaluation outputs.")
    parser.add_argument("--config", default="configs/uwb_pipeline.yaml")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = get_run_dir(config, args.run_dir)
    eval_dir = run_dir / "06_evaluation"
    pred_path = eval_dir / "predictions.csv"
    metrics_path = eval_dir / "metrics.csv"
    if not pred_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Evaluation outputs not found. Run stage 06 first.")

    predictions = pd.read_csv(pred_path)
    metrics = pd.read_csv(metrics_path)
    out_dir = stage_dir(run_dir, "07_plots")
    generate_all_plots(predictions, metrics, out_dir, config)
    print(f"[07] Plots saved to {out_dir}")


if __name__ == "__main__":
    main()

