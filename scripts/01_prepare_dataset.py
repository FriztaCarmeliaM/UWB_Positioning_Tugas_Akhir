from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from uwb_localization.artifacts import create_run_dir, stage_dir
from uwb_localization.config import load_config
from uwb_localization.data import assign_splits, load_all_tracks, write_prepared_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and split UWB trajectory CSV files.")
    parser.add_argument("--config", default="configs/uwb_pipeline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = create_run_dir(config)
    out_dir = stage_dir(run_dir, "01_prepared")

    print(f"[01] Run directory: {run_dir}")
    print("[01] Loading CSV files and validating columns...")
    df = load_all_tracks(config)
    df = assign_splits(df, config)
    write_prepared_dataset(df, out_dir, config)

    print("[01] Prepared dataset saved.")
    print(df.groupby(["split", "trajectory"]).size().to_string())
    print(f"[01] Next: python scripts/02_calibrate_ranges.py --config {args.config}")


if __name__ == "__main__":
    main()

