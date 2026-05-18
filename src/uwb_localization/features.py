from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .preprocessing import preferred_range_column


TARGET_COLS = ["residual_x", "residual_y"]


def build_feature_table(df: pd.DataFrame, anchor_ids: list[str]) -> tuple[pd.DataFrame, list[str]]:
    table = df.copy()
    table["residual_x"] = table["gt_x"] - table["ekf_x"]
    table["residual_y"] = table["gt_y"] - table["ekf_y"]

    if "dt" not in table.columns:
        table["dt"] = table.groupby(["split", "trajectory"])["time"].diff().fillna(0.0) if "time" in table else 0.0

    table["ekf_dx"] = table.groupby(["split", "trajectory"])["ekf_x"].diff().fillna(0.0)
    table["ekf_dy"] = table.groupby(["split", "trajectory"])["ekf_y"].diff().fillna(0.0)
    safe_dt = table["dt"].replace(0.0, np.nan)
    table["ekf_speed"] = np.hypot(table["ekf_dx"], table["ekf_dy"]).div(safe_dt).fillna(0.0)

    feature_cols = [
        "ekf_x",
        "ekf_y",
        "ekf_vx",
        "ekf_vy",
        "ekf_cov_x",
        "ekf_cov_y",
        "ekf_cov_vx",
        "ekf_cov_vy",
        "innovation_norm",
        "valid_anchor_count",
        "rejected_anchor_count",
        "dt",
        "ekf_dx",
        "ekf_dy",
        "ekf_speed",
    ]

    for anchor_id in anchor_ids:
        range_col = preferred_range_column(table, anchor_id)
        table[f"feature_range_{anchor_id}"] = table[range_col]
        feature_cols.append(f"feature_range_{anchor_id}")
        innov_col = f"innov_{anchor_id}"
        if innov_col not in table.columns:
            table[innov_col] = 0.0
        feature_cols.append(innov_col)

    for a, b in itertools.combinations(anchor_ids, 2):
        table[f"range_diff_{a}_{b}"] = table[f"feature_range_{a}"] - table[f"feature_range_{b}"]
        feature_cols.append(f"range_diff_{a}_{b}")

    if "raw_x" in table.columns and "raw_y" in table.columns:
        table["raw_minus_ekf_x"] = table["raw_x"] - table["ekf_x"]
        table["raw_minus_ekf_y"] = table["raw_y"] - table["ekf_y"]
        feature_cols.extend(["raw_minus_ekf_x", "raw_minus_ekf_y"])

    fill_cols = feature_cols + TARGET_COLS
    table[fill_cols] = table[fill_cols].replace([np.inf, -np.inf], np.nan)
    table[fill_cols] = (
        table.groupby(["split", "trajectory"], sort=False)[fill_cols]
        .transform(lambda part: part.ffill().bfill())
        .fillna(0.0)
    )
    return table, feature_cols


def fit_scalers(train_table: pd.DataFrame, feature_cols: list[str]) -> tuple[StandardScaler, StandardScaler]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(train_table[feature_cols].to_numpy(dtype=np.float32))
    y_scaler.fit(train_table[TARGET_COLS].to_numpy(dtype=np.float32))
    return x_scaler, y_scaler


def make_windows(
    table: pd.DataFrame,
    feature_cols: list[str],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler | None,
    sequence_length: int,
    residual_clip_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    x_seq = []
    y_seq = []
    meta = []

    for (_, trajectory), part in table.groupby(["split", "trajectory"], sort=False):
        part = part.sort_values(["segment_id", "sample_index"], kind="mergesort").reset_index(drop=True)
        if len(part) < sequence_length:
            continue

        x_scaled = x_scaler.transform(part[feature_cols].to_numpy(dtype=np.float32))
        y_scaled = None
        if y_scaler is not None:
            target = part[TARGET_COLS].to_numpy(dtype=np.float32)
            if residual_clip_m is not None:
                target = np.clip(target, -float(residual_clip_m), float(residual_clip_m))
            y_scaled = y_scaler.transform(target)

        for target_idx in range(sequence_length - 1, len(part)):
            start = target_idx - sequence_length + 1
            x_seq.append(x_scaled[start : target_idx + 1])
            if y_scaled is not None:
                y_seq.append(y_scaled[target_idx])
            meta.append(part.iloc[target_idx].to_dict())

    if not x_seq:
        raise ValueError(
            f"No sequence windows created. Check sequence_length={sequence_length} and split sizes."
        )

    y_array = np.asarray(y_seq, dtype=np.float32) if y_seq else None
    return np.asarray(x_seq, dtype=np.float32), y_array, pd.DataFrame(meta)
