from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from uwb_localization.artifacts import get_run_dir, stage_dir
from uwb_localization.config import anchor_dict, load_config, load_json
from uwb_localization.ekf import run_ekf_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EKF range-based localization.")
    parser.add_argument("--config", default="configs/uwb_pipeline.yaml")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = get_run_dir(config, args.run_dir)
    source_path = run_dir / "02_range_calibration" / "all_samples_calibrated.csv"
    if not source_path.exists():
        source_path = run_dir / "01_prepared" / "all_samples.csv"
    if not source_path.exists():
        raise FileNotFoundError("No prepared/calibrated dataset found. Run stages 01 and 02 first.")

    optimized_path = run_dir / "03_anchor_optimization" / "optimized_anchors.json"
    if optimized_path.exists():
        anchor_payload = load_json(optimized_path)
        anchors = anchor_payload["anchors"]
        print(f"[04] Using optimized anchors: {optimized_path}")
    else:
        anchors = anchor_dict(config)
        print("[04] Optimized anchors not found; using anchors from config.")

    df = pd.read_csv(source_path)
    print("[04] Running EKF directly from UWB ranges...")
    ekf_df = run_ekf_dataset(df, anchors, config)

    out_dir = stage_dir(run_dir, "04_ekf")
    ekf_df.to_csv(out_dir / "all_samples_ekf.csv", index=False)
    for split, part in ekf_df.groupby("split", sort=False):
        part.to_csv(out_dir / f"{split}.csv", index=False)
    print(f"[04] EKF output saved to {out_dir / 'all_samples_ekf.csv'}")
    print(f"[04] Next: python scripts/05_train_lstm_residual.py --config {args.config}")


if __name__ == "__main__":
    main()

