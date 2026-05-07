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


def apply_trajectory_constraint(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    constraint_cfg = config.get("constraint", {})
    enabled = bool(constraint_cfg.get("enabled", False))
    out = df.copy()

    if not enabled:
        out["constraint_x"] = out["lstm_x"]
        out["constraint_y"] = out["lstm_y"]
        out["constraint_enabled"] = False
        out["constraint_distance_m"] = 0.0
        return out

    if constraint_cfg.get("type", "rectangle") != "rectangle":
        raise ValueError("Only rectangle trajectory constraint is currently implemented.")

    segments = rectangle_segments(constraint_cfg.get("rectangle", {}))
    max_projection = float(constraint_cfg.get("max_projection_m", np.inf))
    projected_x = []
    projected_y = []
    distances = []

    for px, py in out[["lstm_x", "lstm_y"]].to_numpy(dtype=float):
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

