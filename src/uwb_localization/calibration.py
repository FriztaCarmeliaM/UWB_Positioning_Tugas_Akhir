from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import anchor_dict, save_json


def geometric_distance(df: pd.DataFrame, anchor: dict[str, float]) -> np.ndarray:
    return np.hypot(df["gt_x"].to_numpy() - anchor["x"], df["gt_y"].to_numpy() - anchor["y"])


def _ols_fit(raw_range: np.ndarray, true_range: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    x = np.asarray(raw_range, dtype=float)
    y = np.asarray(true_range, dtype=float)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w_sum = float(np.sum(w))
        if w_sum <= 1e-12:
            w = None
        else:
            x_mean = float(np.sum(w * x) / w_sum)
            y_mean = float(np.sum(w * y) / w_sum)
            cov = float(np.sum(w * (x - x_mean) * (y - y_mean)))
            var = float(np.sum(w * (x - x_mean) ** 2))
            slope = 0.0 if var <= 1e-12 else cov / var
            intercept = y_mean - slope * x_mean
            return float(slope), float(intercept)

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    var = float(np.sum((x - x_mean) ** 2))
    cov = float(np.sum((x - x_mean) * (y - y_mean)))
    slope = 0.0 if var <= 1e-12 else cov / var
    intercept = y_mean - slope * x_mean
    return float(slope), float(intercept)


def _mad_scale(residual: np.ndarray) -> float:
    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    return float(max(1.4826 * mad, 1e-6))


def _huber_fit(raw_range: np.ndarray, true_range: np.ndarray, delta: float = 1.35, iterations: int = 25) -> tuple[float, float]:
    slope, intercept = _ols_fit(raw_range, true_range)
    for _ in range(iterations):
        residual = true_range - (slope * raw_range + intercept)
        scale = _mad_scale(residual)
        limit = delta * scale
        weights = np.minimum(1.0, limit / np.maximum(np.abs(residual), 1e-12))
        next_slope, next_intercept = _ols_fit(raw_range, true_range, weights)
        if abs(next_slope - slope) < 1e-9 and abs(next_intercept - intercept) < 1e-9:
            break
        slope, intercept = next_slope, next_intercept
    return slope, intercept


def _ransac_fit(raw_range: np.ndarray, true_range: np.ndarray, trials: int = 200) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    n = len(raw_range)
    if n < 3:
        return _ols_fit(raw_range, true_range)

    base_slope, base_intercept = _ols_fit(raw_range, true_range)
    base_residual = true_range - (base_slope * raw_range + base_intercept)
    threshold = max(2.5 * _mad_scale(base_residual), 0.05)

    best_inliers: np.ndarray | None = None
    best_count = -1
    for _ in range(trials):
        idx = rng.choice(n, size=2, replace=False)
        if abs(raw_range[idx[0]] - raw_range[idx[1]]) < 1e-9:
            continue
        slope, intercept = _ols_fit(raw_range[idx], true_range[idx])
        residual = np.abs(true_range - (slope * raw_range + intercept))
        inliers = residual <= threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        return base_slope, base_intercept
    return _ols_fit(raw_range[best_inliers], true_range[best_inliers])


def _fit_linear(raw_range: np.ndarray, true_range: np.ndarray, method: str, min_robust_samples: int) -> tuple[float, float, str]:
    raw_range = np.asarray(raw_range, dtype=float)
    true_range = np.asarray(true_range, dtype=float)

    if method == "huber" and len(true_range) >= min_robust_samples:
        slope, intercept = _huber_fit(raw_range, true_range)
        fitted_method = "huber_numpy"
    elif method == "ransac" and len(true_range) >= min_robust_samples:
        slope, intercept = _ransac_fit(raw_range, true_range)
        fitted_method = "ransac_numpy"
    else:
        slope, intercept = _ols_fit(raw_range, true_range)
        fitted_method = "linear_numpy"
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
