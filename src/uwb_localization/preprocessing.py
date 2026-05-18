from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import anchor_ids


def preferred_range_column(df: pd.DataFrame, anchor_id: str) -> str:
    candidates = [
        f"range_filt_{anchor_id}",
        f"range_cal_{anchor_id}",
        f"range_{anchor_id}",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing range column for anchor {anchor_id}. Tried {candidates}.")


def _causal_hampel(raw: pd.Series, window_size: int, sigma: float, min_threshold_m: float) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(raw, errors="coerce").astype(float)
    values = values.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if values.isna().all():
        return values, pd.Series(False, index=raw.index)

    median = values.rolling(window_size, min_periods=1).median()

    def mad(part: np.ndarray) -> float:
        med = float(np.median(part))
        return float(np.median(np.abs(part - med)))

    local_mad = values.rolling(window_size, min_periods=1).apply(mad, raw=True)
    threshold = np.maximum(float(min_threshold_m), float(sigma) * 1.4826 * local_mad)
    replaced = (values - median).abs() > threshold
    filtered = values.where(~replaced, median)
    return filtered, replaced.fillna(False)


def _limit_step(
    values: pd.Series,
    times: pd.Series | None,
    max_step_m: float | None,
    max_rate_mps: float | None,
) -> tuple[pd.Series, pd.Series]:
    if len(values) == 0:
        return values, pd.Series(False, index=values.index)

    arr = values.to_numpy(dtype=float).copy()
    time_arr = times.to_numpy(dtype=float) if times is not None else None
    replaced = np.zeros(len(arr), dtype=bool)

    for idx in range(1, len(arr)):
        if not np.isfinite(arr[idx]):
            arr[idx] = arr[idx - 1]
            replaced[idx] = True
            continue
        allowed = float(max_step_m) if max_step_m is not None else np.inf
        if max_rate_mps is not None and time_arr is not None:
            dt = float(time_arr[idx] - time_arr[idx - 1])
            if np.isfinite(dt) and dt > 0:
                allowed = min(allowed, float(max_rate_mps) * dt)
        if not np.isfinite(allowed) or allowed <= 0:
            continue
        delta = arr[idx] - arr[idx - 1]
        if abs(delta) > allowed:
            arr[idx] = arr[idx - 1] + np.sign(delta) * allowed
            replaced[idx] = True

    return pd.Series(arr, index=values.index), pd.Series(replaced, index=values.index)


def _ema(values: pd.Series, alpha: float) -> pd.Series:
    alpha = float(alpha)
    if alpha >= 0.999:
        return values
    if alpha <= 0:
        raise ValueError("range_filter.ema_alpha must be > 0.")

    arr = values.to_numpy(dtype=float)
    out = arr.copy()
    for idx in range(1, len(out)):
        out[idx] = alpha * arr[idx] + (1.0 - alpha) * out[idx - 1]
    return pd.Series(out, index=values.index)


def apply_range_preprocessing(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = config.get("preprocessing", {}).get("range_filter", {})
    if not bool(cfg.get("enabled", False)):
        return df.copy()

    ids = anchor_ids(config)
    window_size = max(int(cfg.get("window_size", 5)), 1)
    sigma = float(cfg.get("hampel_sigma", 3.0))
    min_threshold = float(cfg.get("min_threshold_m", 0.15))
    max_step = cfg.get("max_step_m", 0.30)
    max_step = float(max_step) if max_step is not None else None
    max_rate = cfg.get("max_rate_mps", None)
    max_rate = float(max_rate) if max_rate is not None else None
    ema_alpha = float(cfg.get("ema_alpha", 1.0))

    out_parts = []
    group_cols = [col for col in ["split", "trajectory"] if col in df.columns]
    groups = df.groupby(group_cols, sort=False) if group_cols else [(None, df)]

    for _, part in groups:
        part = part.sort_values(["segment_id", "sample_index"], kind="mergesort").copy()
        times = part["time"] if "time" in part.columns else None
        for anchor_id in ids:
            source_col = preferred_range_column(part, anchor_id)
            hampel, hampel_replaced = _causal_hampel(part[source_col], window_size, sigma, min_threshold)
            limited, step_replaced = _limit_step(hampel, times, max_step, max_rate)
            smoothed = _ema(limited, ema_alpha)
            part[f"range_filt_{anchor_id}"] = smoothed
            part[f"range_filter_replaced_{anchor_id}"] = (hampel_replaced | step_replaced).astype(int)
            part[f"range_filter_source_{anchor_id}"] = source_col
        out_parts.append(part)

    out = pd.concat(out_parts, ignore_index=True)
    return out.sort_index(kind="mergesort").reset_index(drop=True)


def range_preprocessing_summary(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ids = anchor_ids(config)
    rows = []
    group_cols = [col for col in ["split", "trajectory"] if col in df.columns]
    groups = df.groupby(group_cols, sort=False) if group_cols else [(("all", "all"), df)]
    for key, part in groups:
        if not isinstance(key, tuple):
            key = (key, "all")
        split, trajectory = key[0], key[1]
        row = {"split": split, "trajectory": trajectory, "rows": int(len(part))}
        for anchor_id in ids:
            flag_col = f"range_filter_replaced_{anchor_id}"
            if flag_col in part.columns:
                row[f"anchor_{anchor_id}_replaced_pct"] = float(part[flag_col].mean() * 100.0)
        rows.append(row)
    return pd.DataFrame(rows)
