from __future__ import annotations

import argparse
import copy
import itertools

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from uwb_localization.artifacts import get_run_dir, stage_dir
from uwb_localization.config import anchor_dict, load_config, load_json, save_json
from uwb_localization.ekf import run_ekf_dataset
from uwb_localization.metrics import evaluate_predictions


def _candidate_grid(config: dict) -> list[dict[str, float | bool]]:
    tuning = config.get("ekf_tuning", {})
    grid = {
        "process_noise_accel": tuning.get("process_noise_accel", [0.05, 0.10, 0.20, 0.35]),
        "measurement_noise": tuning.get("measurement_noise", [0.08, 0.12, 0.16, 0.20, 0.28]),
        "gating_threshold": tuning.get("gating_threshold", [6.63, 9.21, 16.27]),
        "enable_gating": tuning.get("enable_gating", [True]),
    }
    keys = list(grid.keys())
    candidates = []
    for values in itertools.product(*(grid[key] for key in keys)):
        candidates.append(dict(zip(keys, values)))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune EKF hyperparameters on validation split only.")
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
        anchors = load_json(optimized_path)["anchors"]
        print(f"[04-tune] Using optimized anchors: {optimized_path}")
    else:
        anchors = anchor_dict(config)
        print("[04-tune] Optimized anchors not found; using anchors from config.")

    df = pd.read_csv(source_path)
    val_df = df[df["split"] == "val"].copy()
    if val_df.empty:
        raise ValueError("EKF tuning requires a non-empty validation split.")

    out_dir = stage_dir(run_dir, "04_ekf_tuning")
    rows = []
    best: dict | None = None
    candidates = _candidate_grid(config)
    print(f"[04-tune] Evaluating {len(candidates)} EKF candidates on validation split...")

    for index, params in enumerate(candidates, start=1):
        trial_config = copy.deepcopy(config)
        trial_config.setdefault("ekf", {}).update(params)
        ekf_df = run_ekf_dataset(val_df, anchors, trial_config)
        metrics = evaluate_predictions(ekf_df)
        ekf_metric = metrics[
            (metrics["split"] == "val")
            & (metrics["trajectory"] == "all")
            & (metrics["model"] == "EKF only")
        ].iloc[0]
        row = {
            "candidate": index,
            **params,
            "val_rmse_2d_m": float(ekf_metric["rmse_2d_m"]),
            "val_mae_2d_m": float(ekf_metric["mae_2d_m"]),
            "val_p95_error_m": float(ekf_metric["p95_error_m"]),
            "val_pct_below_5cm": float(ekf_metric["pct_below_5cm"]),
        }
        rows.append(row)
        if best is None or row["val_rmse_2d_m"] < best["val_rmse_2d_m"]:
            best = row

    result = pd.DataFrame(rows).sort_values("val_rmse_2d_m").reset_index(drop=True)
    result.to_csv(out_dir / "ekf_tuning_grid.csv", index=False)
    if best is None:
        raise RuntimeError("No EKF tuning candidate was evaluated.")

    best_params = {
        key: best[key]
        for key in ["process_noise_accel", "measurement_noise", "gating_threshold", "enable_gating"]
    }
    save_json(
        {
            "selection_metric": "validation EKF only rmse_2d_m",
            "validation_rmse_2d_m": best["val_rmse_2d_m"],
            "validation_mae_2d_m": best["val_mae_2d_m"],
            "validation_p95_error_m": best["val_p95_error_m"],
            "params": best_params,
        },
        out_dir / "best_ekf_params.json",
    )
    print(f"[04-tune] Best validation EKF RMSE: {best['val_rmse_2d_m']:.4f} m")
    print(f"[04-tune] Best params saved to {out_dir / 'best_ekf_params.json'}")
    print(f"[04-tune] Next: python scripts/04_run_ekf.py --config {args.config}")


if __name__ == "__main__":
    main()
