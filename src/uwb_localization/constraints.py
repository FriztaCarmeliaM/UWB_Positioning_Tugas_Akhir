from __future__ import annotations

import numpy as np
import pandas as pd


def _project_point_to_segment(px: float, py: float, a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    ax, ay = a
    bx, by = b
    ab = np.array([bx - ax, by - ay], dtype=float)
    ap = np.array([px - ax, py - ay], dtype=float)
    denom = float(ab @ ab)
    if denom <= 1e-12:
        return ax, ay
    t = float(np.clip((ap @ ab) / denom, 0.0, 1.0))
    projected = np.array([ax, ay], dtype=float) + t * ab
    return float(projected[0]), float(projected[1])


def rectangle_segments(bounds: dict[str, float]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    x_min = float(bounds["x_min"])
    x_max = float(bounds["x_max"])
    y_min = float(bounds["y_min"])
    y_max = float(bounds["y_max"])
    return [
        ((x_min, y_min), (x_max, y_min)),
        ((x_max, y_min), (x_max, y_max)),
        ((x_max, y_max), (x_min, y_max)),
        ((x_min, y_max), (x_min, y_min)),
    ]


def polyline_segments(points: list) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    normalized = []
    for point in points:
        if isinstance(point, dict):
            normalized.append((float(point["x"]), float(point["y"])))
        else:
            normalized.append((float(point[0]), float(point[1])))
    if len(normalized) < 2:
        raise ValueError("Polyline constraint requires at least two points.")
    return list(zip(normalized[:-1], normalized[1:]))


def _segments_from_config(constraint_cfg: dict) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    constraint_type = constraint_cfg.get("type", "rectangle")
    if constraint_type == "rectangle":
        return rectangle_segments(constraint_cfg.get("rectangle", {}))
    if constraint_type in {"polyline", "path"}:
        return polyline_segments(constraint_cfg.get("points", []))
    raise ValueError(f"Unsupported trajectory constraint type: {constraint_type}")


def _raw_position_median(out: pd.DataFrame, window_size: int) -> tuple[pd.Series, pd.Series]:
    raw_x = pd.to_numeric(out["raw_x"], errors="coerce")
    raw_y = pd.to_numeric(out["raw_y"], errors="coerce")
    group_cols = [col for col in ["split", "trajectory"] if col in out.columns]
    if not group_cols:
        return raw_x.rolling(window_size, min_periods=1).median(), raw_y.rolling(window_size, min_periods=1).median()

    smooth_x = raw_x.copy()
    smooth_y = raw_y.copy()
    for _, part in out.groupby(group_cols, sort=False):
        idx = part.index
        smooth_x.loc[idx] = raw_x.loc[idx].rolling(window_size, min_periods=1).median()
        smooth_y.loc[idx] = raw_y.loc[idx].rolling(window_size, min_periods=1).median()
    return smooth_x, smooth_y


def _apply_hybrid_fallback(out: pd.DataFrame, constraint_cfg: dict) -> pd.DataFrame:
    fallback_cfg = constraint_cfg.get("hybrid_fallback", {})
    enabled = bool(fallback_cfg.get("enabled", False))
    out["hybrid_x"] = out["lstm_x"]
    out["hybrid_y"] = out["lstm_y"]
    out["hybrid_fallback_used"] = False
    out["hybrid_fallback_distance_m"] = 0.0

    if not enabled or "raw_x" not in out.columns or "raw_y" not in out.columns:
        return out

    window_size = max(int(fallback_cfg.get("raw_window_size", 21)), 1)
    threshold = float(fallback_cfg.get("fallback_distance_m", 0.40))
    raw_x, raw_y = _raw_position_median(out, window_size)
    distance = np.hypot(raw_x.to_numpy(dtype=float) - out["lstm_x"].to_numpy(dtype=float),
                        raw_y.to_numpy(dtype=float) - out["lstm_y"].to_numpy(dtype=float))
    use_raw = np.isfinite(distance) & (distance > threshold)

    out["raw_fallback_x"] = raw_x
    out["raw_fallback_y"] = raw_y
    out.loc[use_raw, "hybrid_x"] = raw_x.loc[use_raw]
    out.loc[use_raw, "hybrid_y"] = raw_y.loc[use_raw]
    out["hybrid_fallback_used"] = use_raw
    out["hybrid_fallback_distance_m"] = distance
    return out


def apply_trajectory_constraint(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    constraint_cfg = config.get("constraint", {})
    enabled = bool(constraint_cfg.get("enabled", False))
    out = df.copy()
    out = _apply_hybrid_fallback(out, constraint_cfg)

    if not enabled:
        out["constraint_x"] = out["hybrid_x"]
        out["constraint_y"] = out["hybrid_y"]
        out["constraint_enabled"] = False
        out["constraint_distance_m"] = 0.0
        return out

    segments = _segments_from_config(constraint_cfg)
    max_projection = float(constraint_cfg.get("max_projection_m", np.inf))
    projected_x = []
    projected_y = []
    distances = []

    for px, py in out[["hybrid_x", "hybrid_y"]].to_numpy(dtype=float):
        candidates = []
        for start, end in segments:
            qx, qy = _project_point_to_segment(px, py, start, end)
            distance = float(np.hypot(px - qx, py - qy))
            candidates.append((distance, qx, qy))
        distance, qx, qy = min(candidates, key=lambda item: item[0])
        if distance <= max_projection:
            projected_x.append(qx)
            projected_y.append(qy)
            distances.append(distance)
        else:
            projected_x.append(px)
            projected_y.append(py)
            distances.append(0.0)

    out["constraint_x"] = projected_x
    out["constraint_y"] = projected_y
    out["constraint_enabled"] = True
    out["constraint_distance_m"] = distances
    return out
