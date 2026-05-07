from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .config import anchor_dict, save_json


def _measurement_column(df: pd.DataFrame, anchor_id: str) -> str:
    calibrated = f"range_cal_{anchor_id}"
    raw = f"range_{anchor_id}"
    if calibrated in df.columns:
        return calibrated
    if raw in df.columns:
        return raw
    raise ValueError(f"No range column found for anchor {anchor_id}.")


def optimize_anchors(train_df: pd.DataFrame, config: dict) -> dict[str, Any]:
    opt_cfg = config.get("anchor_optimization", {})
    anchors = anchor_dict(config)
    anchor_ids = list(anchors.keys())
    enabled = bool(opt_cfg.get("enabled", True))
    optimize_bias = bool(opt_cfg.get("optimize_bias", True))

    if not enabled:
        return {
            "enabled": False,
            "source": "config anchors",
            "anchors": anchors,
            "cost": None,
        }

    max_samples = int(opt_cfg.get("max_samples", 20000))
    if len(train_df) > max_samples:
        train_df = train_df.sample(max_samples, random_state=int(config.get("random_seed", 42))).sort_index()

    prior_xy = np.array([[anchors[anchor_id]["x"], anchors[anchor_id]["y"]] for anchor_id in anchor_ids], dtype=float)
    prior_bias = np.array([anchors[anchor_id].get("bias", 0.0) for anchor_id in anchor_ids], dtype=float)
    n = len(anchor_ids)
    max_move = float(opt_cfg.get("max_anchor_move_m", 0.5))
    max_bias = float(opt_cfg.get("max_bias_m", 1.0))
    move_reg = float(opt_cfg.get("move_regularization_m", 0.2))
    bias_reg = float(opt_cfg.get("bias_regularization_m", 0.5))

    gt = train_df[["gt_x", "gt_y"]].to_numpy(dtype=float)
    ranges = np.column_stack([train_df[_measurement_column(train_df, anchor_id)].to_numpy(dtype=float) for anchor_id in anchor_ids])

    p0 = np.zeros(2 * n + (n if optimize_bias else 0), dtype=float)
    lower = np.full_like(p0, -max_move)
    upper = np.full_like(p0, max_move)
    if optimize_bias:
        lower[2 * n :] = -max_bias
        upper[2 * n :] = max_bias

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta_xy = params[: 2 * n].reshape(n, 2)
        if optimize_bias:
            bias = prior_bias + params[2 * n :]
        else:
            bias = prior_bias.copy()
        return prior_xy + delta_xy, bias

    def residuals(params: np.ndarray) -> np.ndarray:
        xy, bias = unpack(params)
        measurement_residuals = []
        for idx in range(n):
            geom = np.hypot(gt[:, 0] - xy[idx, 0], gt[:, 1] - xy[idx, 1]) + bias[idx]
            measurement_residuals.append(geom - ranges[:, idx])
        res = np.concatenate(measurement_residuals)
        delta_xy = (xy - prior_xy).ravel() / max(move_reg, 1e-9)
        reg = [delta_xy]
        if optimize_bias:
            reg.append((bias - prior_bias) / max(bias_reg, 1e-9))
        return np.concatenate([res, *reg])

    result = least_squares(residuals, p0, bounds=(lower, upper), loss=opt_cfg.get("loss", "soft_l1"))
    optimized_xy, optimized_bias = unpack(result.x)

    optimized: dict[str, Any] = {
        "enabled": True,
        "source": "least_squares_train_split",
        "cost": float(result.cost),
        "success": bool(result.success),
        "message": str(result.message),
        "anchors": {},
    }
    for idx, anchor_id in enumerate(anchor_ids):
        optimized["anchors"][anchor_id] = {
            "x": float(optimized_xy[idx, 0]),
            "y": float(optimized_xy[idx, 1]),
            "bias": float(optimized_bias[idx]),
            "prior_x": float(prior_xy[idx, 0]),
            "prior_y": float(prior_xy[idx, 1]),
            "prior_bias": float(prior_bias[idx]),
            "delta_x": float(optimized_xy[idx, 0] - prior_xy[idx, 0]),
            "delta_y": float(optimized_xy[idx, 1] - prior_xy[idx, 1]),
        }
    return optimized


def save_optimized_anchors(anchors: dict[str, Any], out_path: str | Path) -> None:
    save_json(anchors, out_path)

