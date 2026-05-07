from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from uwb_localization.artifacts import get_run_dir, stage_dir
from uwb_localization.config import load_config
from uwb_localization.lstm import train_lstm_residual


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM residual corrector without data leakage.")
    parser.add_argument("--config", default="configs/uwb_pipeline.yaml")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = get_run_dir(config, args.run_dir)
    source_path = run_dir / "04_ekf" / "all_samples_ekf.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"EKF output not found: {source_path}")

    df = pd.read_csv(source_path)
    out_dir = stage_dir(run_dir, "05_lstm_residual")
    print("[05] Training LSTM residual corrector.")
    print("[05] Train scalers are fit on train split only; validation is used for early stopping.")
    train_lstm_residual(df, config, out_dir)
    print(f"[05] LSTM artifacts saved to {out_dir}")
    print(f"[05] Next: python scripts/06_evaluate_pipeline.py --config {args.config}")


if __name__ == "__main__":
    main()

