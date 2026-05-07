from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from uwb_localization.artifacts import get_run_dir, stage_dir
from uwb_localization.calibration import apply_range_calibration, fit_range_calibration, save_range_calibration
from uwb_localization.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit per-anchor range calibration on train split only.")
    parser.add_argument("--config", default="configs/uwb_pipeline.yaml")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = get_run_dir(config, args.run_dir)
    prepared_path = run_dir / "01_prepared" / "all_samples.csv"
    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {prepared_path}")

    df = pd.read_csv(prepared_path)
    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("Cannot fit range calibration: train split is empty.")

    out_dir = stage_dir(run_dir, "02_range_calibration")
    print(f"[02] Fitting range calibration from train split only: n={len(train_df)}")
    calibration = fit_range_calibration(train_df, config)
    save_range_calibration(calibration, out_dir / "range_calibration.json")

    calibrated = apply_range_calibration(df, calibration)
    calibrated.to_csv(out_dir / "all_samples_calibrated.csv", index=False)
    for split, part in calibrated.groupby("split", sort=False):
        part.to_csv(out_dir / f"{split}.csv", index=False)

    print(f"[02] Calibration saved to {out_dir / 'range_calibration.json'}")
    print(f"[02] Next: python scripts/03_optimize_anchors.py --config {args.config}")


if __name__ == "__main__":
    main()

