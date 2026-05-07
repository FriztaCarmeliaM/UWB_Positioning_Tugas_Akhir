from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def position_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    error_2d = np.linalg.norm(err, axis=1)
    return {
        "rmse_x_m": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "rmse_y_m": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "rmse_2d_m": float(np.sqrt(np.mean(err[:, 0] ** 2 + err[:, 1] ** 2))),
        "mae_x_m": float(np.mean(abs_err[:, 0])),
        "mae_y_m": float(np.mean(abs_err[:, 1])),
        "mae_2d_m": float(np.mean(error_2d)),
        "median_error_m": float(np.median(error_2d)),
        "p90_error_m": float(np.percentile(error_2d, 90)),
        "p95_error_m": float(np.percentile(error_2d, 95)),
        "max_error_m": float(np.max(error_2d)),
        "pct_below_5cm": float(np.mean(error_2d <= 0.05) * 100.0),
        "pct_below_10cm": float(np.mean(error_2d <= 0.10) * 100.0),
        "pct_below_20cm": float(np.mean(error_2d <= 0.20) * 100.0),
        "pct_below_50cm": float(np.mean(error_2d <= 0.50) * 100.0),
        "n_samples": int(len(error_2d)),
    }


def evaluate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    methods = [
        ("Raw trilateration", "raw_x", "raw_y", "raw_x" in df.columns and "raw_y" in df.columns),
        (
            "Legacy position KF",
            "legacy_kf_x",
            "legacy_kf_y",
            "legacy_kf_x" in df.columns and "legacy_kf_y" in df.columns,
        ),
        ("EKF only", "ekf_x", "ekf_y", True),
        ("EKF + LSTM residual", "lstm_x", "lstm_y", "lstm_x" in df.columns and "lstm_y" in df.columns),
        (
            "EKF + LSTM + trajectory constraint",
            "constraint_x",
            "constraint_y",
            "constraint_x" in df.columns and "constraint_y" in df.columns,
        ),
    ]
    rows = []
    group_cols = ["split", "trajectory"]
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        if split_df.empty:
            continue
        groups = [("__all__", split_df)] + list(split_df.groupby("trajectory", sort=False))
        for track_name, part in groups:
            y_true = part[["gt_x", "gt_y"]].to_numpy(dtype=float)
            for model_name, x_col, y_col, available in methods:
                if not available:
                    continue
                y_pred = part[[x_col, y_col]].to_numpy(dtype=float)
                row = {
                    "split": split,
                    "trajectory": "all" if track_name == "__all__" else track_name,
                    "model": model_name,
                    "constraint_enabled": bool(part["constraint_enabled"].any())
                    if "constraint_enabled" in part.columns
                    else False,
                    **position_metrics(y_true, y_pred),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def save_metrics(metrics: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics.to_dict(orient="records"), file, indent=2)
