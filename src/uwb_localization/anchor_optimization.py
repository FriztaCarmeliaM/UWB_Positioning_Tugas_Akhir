from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import anchor_dict, save_json
from .preprocessing import preferred_range_column


def _measurement_column(df: pd.DataFrame, anchor_id: str) -> str:
    return preferred_range_column(df, anchor_id)


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

    params = np.zeros(2 * n + (n if optimize_bias else 0), dtype=float)
    lower = np.full_like(params, -max_move)
    upper = np.full_like(params, max_move)
    if optimize_bias:
        lower[2 * n :] = -max_bias
        upper[2 * n :] = max_bias
    initial_steps = np.full_like(params, max(max_move / 2.0, 1e-3))
    if optimize_bias:
        initial_steps[2 * n :] = max(max_bias / 2.0, 1e-3)
    min_step = float(opt_cfg.get("min_coordinate_step_m", 1e-4))
    max_passes = int(opt_cfg.get("coordinate_passes", 24))
    reg_weight = float(opt_cfg.get("regularization_weight", 0.01))
    loss_scale = float(opt_cfg.get("loss_scale_m", 0.20))

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta_xy = params[: 2 * n].reshape(n, 2)
        if optimize_bias:
            bias = prior_bias + params[2 * n :]
        else:
            bias = prior_bias.copy()
        return prior_xy + delta_xy, bias

    def objective(params: np.ndarray) -> float:
        xy, bias = unpack(params)
        total_loss = 0.0
        for idx in range(n):
            geom = np.hypot(gt[:, 0] - xy[idx, 0], gt[:, 1] - xy[idx, 1]) + bias[idx]
            res = geom - ranges[:, idx]
            if opt_cfg.get("loss", "soft_l1") == "soft_l1":
                scaled = res / max(loss_scale, 1e-9)
                total_loss += float(np.mean(2.0 * (np.sqrt(1.0 + scaled**2) - 1.0) * loss_scale**2))
            else:
                total_loss += float(np.mean(res**2))
        delta_xy = (xy - prior_xy).ravel() / max(move_reg, 1e-9)
        reg_terms = [delta_xy]
        if optimize_bias:
            reg_terms.append((bias - prior_bias) / max(bias_reg, 1e-9))
        reg = np.concatenate(reg_terms)
        return total_loss / n + reg_weight * float(np.mean(reg**2))

    best_cost = objective(params)
    steps = initial_steps.copy()
    passes = 0
    while float(np.max(steps)) >= min_step and passes < max_passes:
        improved = False
        for idx in range(len(params)):
            for direction in (1.0, -1.0):
                trial = params.copy()
                trial[idx] = float(np.clip(trial[idx] + direction * steps[idx], lower[idx], upper[idx]))
                if abs(trial[idx] - params[idx]) <= 1e-12:
                    continue
                cost = objective(trial)
                if cost + 1e-12 < best_cost:
                    params = trial
                    best_cost = cost
                    improved = True
        if not improved:
            steps *= 0.5
        passes += 1

    optimized_xy, optimized_bias = unpack(params)

    optimized: dict[str, Any] = {
        "enabled": True,
        "source": "coordinate_search_train_split",
        "cost": float(best_cost),
        "success": True,
        "message": f"coordinate search completed in {passes} passes",
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
