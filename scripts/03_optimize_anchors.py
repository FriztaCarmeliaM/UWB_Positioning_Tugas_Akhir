from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from uwb_localization.anchor_optimization import optimize_anchors, save_optimized_anchors
from uwb_localization.artifacts import get_run_dir, stage_dir
from uwb_localization.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize anchor coordinates and optional biases on train split only.")
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

    df = pd.read_csv(source_path)
    train_df = df[df["split"] == "train"].copy()
    out_dir = stage_dir(run_dir, "03_anchor_optimization")

    print(f"[03] Optimizing anchors from train split only: n={len(train_df)}")
    optimized = optimize_anchors(train_df, config)
    save_optimized_anchors(optimized, out_dir / "optimized_anchors.json")
    print(f"[03] Anchor configuration saved to {out_dir / 'optimized_anchors.json'}")
    print(f"[03] Next: python scripts/04_run_ekf.py --config {args.config}")


if __name__ == "__main__":
    main()

