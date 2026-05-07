from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor

from .config import anchor_dict, save_json


def geometric_distance(df: pd.DataFrame, anchor: dict[str, float]) -> np.ndarray:
    return np.hypot(df["gt_x"].to_numpy() - anchor["x"], df["gt_y"].to_numpy() - anchor["y"])


def _fit_linear(raw_range: np.ndarray, true_range: np.ndarray, method: str, min_robust_samples: int) -> tuple[float, float, str]:
    x = raw_range.reshape(-1, 1)
    y = true_range

    if method == "huber" and len(y) >= min_robust_samples:
        model = HuberRegressor(epsilon=1.35, alpha=0.0)
        fitted_method = "huber"
    elif method == "ransac" and len(y) >= min_robust_samples:
        model = RANSACRegressor(estimator=LinearRegression(), random_state=42)
        fitted_method = "ransac"
    else:
        model = LinearRegression()
        fitted_method = "linear"

    model.fit(x, y)
    if fitted_method == "ransac":
        estimator = model.estimator_
        slope = float(estimator.coef_[0])
        intercept = float(estimator.intercept_)
    else:
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
    return slope, intercept, fitted_method


def fit_range_calibration(train_df: pd.DataFrame, config: dict) -> dict[str, Any]:
    cal_cfg = config.get("range_calibration", {})
    anchors = anchor_dict(config)
    enabled = bool(cal_cfg.get("enabled", True))
    method = cal_cfg.get("method", "huber")
    min_samples = int(cal_cfg.get("min_samples_per_anchor", 50))
    min_robust_samples = int(cal_cfg.get("min_robust_samples", 200))

    params: dict[str, Any] = {
        "model": "d_corrected = a * d_raw + b",
        "fit_split": "train",
        "method_requested": method,
        "anchors": {},
    }

    for anchor_id, anchor in anchors.items():
        raw_col = f"range_{anchor_id}"
        if raw_col not in train_df.columns:
            raise ValueError(f"Missing range column for anchor {anchor_id}: {raw_col}")

        part = train_df[[raw_col, "gt_x", "gt_y"]].dropna()
        if len(part) < min_samples:
            raise ValueError(
                f"Not enough training samples to calibrate anchor {anchor_id}: "
                f"{len(part)} < {min_samples}"
            )

        raw = part[raw_col].to_numpy(dtype=float)
        true = geometric_distance(part, anchor)
        if enabled:
            a, b, fitted_method = _fit_linear(raw, true, method, min_robust_samples)
        else:
            a, b, fitted_method = 1.0, 0.0, "identity_disabled"
        corrected = a * raw + b
        before = float(np.sqrt(np.mean((raw - true) ** 2)))
        after = float(np.sqrt(np.mean((corrected - true) ** 2)))

        params["anchors"][anchor_id] = {
            "a": a,
            "b": b,
            "method": fitted_method,
            "n_samples": int(len(part)),
            "rmse_before_m": before,
            "rmse_after_m": after,
        }

    return params


def apply_range_calibration(df: pd.DataFrame, calibration: dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    for anchor_id, params in calibration["anchors"].items():
        raw_col = f"range_{anchor_id}"
        cal_col = f"range_cal_{anchor_id}"
        if raw_col not in df.columns:
            raise ValueError(f"Missing raw range column: {raw_col}")
        df[cal_col] = float(params["a"]) * df[raw_col] + float(params["b"])
    return df


def save_range_calibration(calibration: dict[str, Any], out_path: str | Path) -> None:
    save_json(calibration, out_path)
