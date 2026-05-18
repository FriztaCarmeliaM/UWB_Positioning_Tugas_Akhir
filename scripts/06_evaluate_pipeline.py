from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from uwb_localization.artifacts import get_run_dir, stage_dir
from uwb_localization.config import load_config
from uwb_localization.constraints import apply_trajectory_constraint
from uwb_localization.lstm import apply_lstm_residual
from uwb_localization.metrics import evaluate_predictions, evaluate_predictions_by_group, save_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EKF, EKF+LSTM, and optional constrained output.")
    parser.add_argument("--config", default="configs/uwb_pipeline.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--allow-missing-lstm", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = get_run_dir(config, args.run_dir)
    ekf_path = run_dir / "04_ekf" / "all_samples_ekf.csv"
    if not ekf_path.exists():
        raise FileNotFoundError(f"EKF output not found: {ekf_path}")

    ekf_df = pd.read_csv(ekf_path)
    model_dir = run_dir / "05_lstm_residual"
    if (model_dir / "lstm_metadata.json").exists():
        predictions = apply_lstm_residual(ekf_df, config, model_dir)
    elif args.allow_missing_lstm:
        print("[06] LSTM artifacts missing; copying EKF as LSTM output for EKF-only dry evaluation.")
        predictions = ekf_df.copy()
        predictions["lstm_x"] = predictions["ekf_x"]
        predictions["lstm_y"] = predictions["ekf_y"]
        predictions["lstm_residual_x"] = 0.0
        predictions["lstm_residual_y"] = 0.0
    else:
        raise FileNotFoundError(
            f"LSTM artifacts not found in {model_dir}. Run stage 05 first or pass --allow-missing-lstm."
        )

    predictions = apply_trajectory_constraint(predictions, config)
    out_dir = stage_dir(run_dir, "06_evaluation")
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    metrics = evaluate_predictions(predictions)
    save_metrics(metrics, out_dir)
    segment_metrics = evaluate_predictions_by_group(predictions, "gt_segment")
    if not segment_metrics.empty:
        segment_metrics.to_csv(out_dir / "segment_metrics.csv", index=False)
    print(f"[06] Predictions saved to {out_dir / 'predictions.csv'}")
    print(f"[06] Metrics saved to {out_dir / 'metrics.csv'}")
    print(f"[06] Next: python scripts/07_plot_results.py --config {args.config}")


if __name__ == "__main__":
    main()
