from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .preprocessing import preferred_range_column


@dataclass
class EKFParams:
    process_noise_accel: float = 0.2
    measurement_noise: float = 0.20
    initial_position_std: float = 1.0
    initial_velocity_std: float = 1.0
    gating_threshold: float = 9.21
    min_valid_anchors: int = 3
    default_dt: float = 0.05
    max_dt: float = 0.25
    enable_gating: bool = True


def ekf_params_from_config(config: dict[str, Any]) -> EKFParams:
    cfg = config.get("ekf", {})
    return EKFParams(
        process_noise_accel=float(cfg.get("process_noise_accel", 0.2)),
        measurement_noise=float(cfg.get("measurement_noise", 0.20)),
        initial_position_std=float(cfg.get("initial_position_std", 1.0)),
        initial_velocity_std=float(cfg.get("initial_velocity_std", 1.0)),
        gating_threshold=float(cfg.get("gating_threshold", 9.21)),
        min_valid_anchors=int(cfg.get("min_valid_anchors", 3)),
        default_dt=float(cfg.get("default_dt", 0.05)),
        max_dt=float(cfg.get("max_dt", 0.25)),
        enable_gating=bool(cfg.get("enable_gating", True)),
    )


def measurement_column(df: pd.DataFrame, anchor_id: str) -> str:
    return preferred_range_column(df, anchor_id)


def _initial_position(row: pd.Series, anchors: dict[str, dict[str, float]]) -> tuple[float, float]:
    if "legacy_kf_x" in row.index and "legacy_kf_y" in row.index:
        if np.isfinite(row["legacy_kf_x"]) and np.isfinite(row["legacy_kf_y"]):
            return float(row["legacy_kf_x"]), float(row["legacy_kf_y"])
    if "raw_x" in row.index and "raw_y" in row.index:
        if np.isfinite(row["raw_x"]) and np.isfinite(row["raw_y"]):
            return float(row["raw_x"]), float(row["raw_y"])
    return _least_squares_trilateration(row, anchors)


def _least_squares_trilateration(row: pd.Series, anchors: dict[str, dict[str, float]]) -> tuple[float, float]:
    anchor_ids = list(anchors.keys())
    if len(anchor_ids) < 3:
        first = anchors[anchor_ids[0]]
        return float(first["x"]), float(first["y"])

    a0 = anchors[anchor_ids[0]]
    d0 = float(row.get(preferred_range_column(pd.DataFrame([row]), anchor_ids[0])))
    rows = []
    rhs = []
    for anchor_id in anchor_ids[1:]:
        ai = anchors[anchor_id]
        di = float(row.get(preferred_range_column(pd.DataFrame([row]), anchor_id)))
        rows.append([2 * (ai["x"] - a0["x"]), 2 * (ai["y"] - a0["y"])])
        rhs.append(d0**2 - di**2 - a0["x"] ** 2 + ai["x"] ** 2 - a0["y"] ** 2 + ai["y"] ** 2)
    ata_00 = sum(float(r[0]) * float(r[0]) for r in rows)
    ata_01 = sum(float(r[0]) * float(r[1]) for r in rows)
    ata_11 = sum(float(r[1]) * float(r[1]) for r in rows)
    atb_0 = sum(float(r[0]) * float(b) for r, b in zip(rows, rhs))
    atb_1 = sum(float(r[1]) * float(b) for r, b in zip(rows, rhs))
    det = ata_00 * ata_11 - ata_01 * ata_01
    if abs(det) <= 1e-12:
        return float(a0["x"]), float(a0["y"])
    x = (atb_0 * ata_11 - ata_01 * atb_1) / det
    y = (ata_00 * atb_1 - atb_0 * ata_01) / det
    return float(x), float(y)


def _process_noise(dt: float, accel_noise: float) -> np.ndarray:
    q = accel_noise**2
    return q * np.array(
        [
            [dt**4 / 4, 0.0, dt**3 / 2, 0.0],
            [0.0, dt**4 / 4, 0.0, dt**3 / 2],
            [dt**3 / 2, 0.0, dt**2, 0.0],
            [0.0, dt**3 / 2, 0.0, dt**2],
        ],
        dtype=float,
    )


def _predict(state: np.ndarray, cov: np.ndarray, dt: float, params: EKFParams) -> tuple[np.ndarray, np.ndarray]:
    f = [
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    predicted_state = np.zeros_like(state)
    for i in range(4):
        predicted_state[i] = sum(f[i][j] * state[j] for j in range(4))

    predicted_cov = np.zeros_like(cov)
    for i in range(4):
        for j in range(4):
            value = 0.0
            for m in range(4):
                for n in range(4):
                    value += f[i][m] * cov[m, n] * f[j][n]
            predicted_cov[i, j] = value
    predicted_cov += _process_noise(dt, params.process_noise_accel)
    return predicted_state, predicted_cov


def _quad_form(row: np.ndarray, cov: np.ndarray) -> float:
    value = 0.0
    for i in range(len(row)):
        for j in range(len(row)):
            value += float(row[i]) * float(cov[i, j]) * float(row[j])
    return value


def _cov_times_row(cov: np.ndarray, row: np.ndarray) -> np.ndarray:
    result = np.zeros(cov.shape[0], dtype=float)
    for i in range(cov.shape[0]):
        result[i] = sum(float(cov[i, j]) * float(row[j]) for j in range(cov.shape[1]))
    return result


def _cov_update(cov: np.ndarray, kalman_gain: np.ndarray, h_row: np.ndarray) -> np.ndarray:
    hp = np.zeros(cov.shape[1], dtype=float)
    for j in range(cov.shape[1]):
        hp[j] = sum(float(h_row[m]) * float(cov[m, j]) for m in range(cov.shape[0]))

    updated = cov.copy()
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            updated[i, j] = cov[i, j] - kalman_gain[i] * hp[j]

    return (updated + updated.T) * 0.5


def _measurement_model(
    state: np.ndarray,
    anchors: dict[str, dict[str, float]],
    anchor_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x, y = state[0], state[1]
    h = []
    h_jac = []
    for anchor_id in anchor_ids:
        anchor = anchors[anchor_id]
        dx = x - anchor["x"]
        dy = y - anchor["y"]
        dist = max(float(np.hypot(dx, dy)), 1e-9)
        h.append(dist + float(anchor.get("bias", 0.0)))
        h_jac.append([dx / dist, dy / dist, 0.0, 0.0])
    return np.asarray(h, dtype=float), np.asarray(h_jac, dtype=float)


def _update(
    state: np.ndarray,
    cov: np.ndarray,
    z: np.ndarray,
    anchors: dict[str, dict[str, float]],
    anchor_ids: list[str],
    params: EKFParams,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    h, h_jac = _measurement_model(state, anchors, anchor_ids)
    innovation = z - h
    r_diag = np.full(len(anchor_ids), params.measurement_noise**2, dtype=float)

    keep = np.ones(len(anchor_ids), dtype=bool)
    normalized = np.zeros(len(anchor_ids), dtype=float)
    if params.enable_gating:
        for i in range(len(anchor_ids)):
            s_i = float(_quad_form(h_jac[i], cov) + r_diag[i])
            normalized[i] = float(innovation[i] ** 2 / max(s_i, 1e-12))
            keep[i] = normalized[i] <= params.gating_threshold

    if int(keep.sum()) < params.min_valid_anchors:
        return state, cov, {
            "innovation_norm": float(np.sqrt(np.sum(innovation**2))),
            "valid_anchor_count": float(keep.sum()),
            "rejected_anchor_count": float(len(anchor_ids) - keep.sum()),
            "updated": 0.0,
            **{f"innov_{anchor_ids[i]}": float(innovation[i]) for i in range(len(anchor_ids))},
        }

    innovation_keep = innovation[keep]
    kept_indices = np.flatnonzero(keep)
    for original_index in kept_indices:
        anchor_id = anchor_ids[int(original_index)]
        h_single, h_single_jac = _measurement_model(state, anchors, [anchor_id])
        measurement_residual = float(z[int(original_index)] - h_single[0])
        h_row = h_single_jac[0]
        s_scalar = float(_quad_form(h_row, cov) + r_diag[int(original_index)])
        if s_scalar <= 1e-12:
            continue
        kalman_gain = _cov_times_row(cov, h_row) / s_scalar
        state = state + kalman_gain * measurement_residual
        cov = _cov_update(cov, kalman_gain, h_row)

    return state, cov, {
        "innovation_norm": float(np.sqrt(np.sum(innovation_keep**2))),
        "valid_anchor_count": float(keep.sum()),
        "rejected_anchor_count": float(len(anchor_ids) - keep.sum()),
        "updated": 1.0,
        **{f"innov_{anchor_ids[i]}": float(innovation[i]) for i in range(len(anchor_ids))},
    }


def run_ekf_track(track_df: pd.DataFrame, anchors: dict[str, dict[str, float]], config: dict[str, Any]) -> pd.DataFrame:
    params = ekf_params_from_config(config)
    anchor_ids = list(anchors.keys())
    df = track_df.sort_values(["segment_id", "sample_index"], kind="mergesort").reset_index(drop=True).copy()
    first_x, first_y = _initial_position(df.iloc[0], anchors)
    state = np.array([first_x, first_y, 0.0, 0.0], dtype=float)
    cov = np.diag(
        [
            params.initial_position_std**2,
            params.initial_position_std**2,
            params.initial_velocity_std**2,
            params.initial_velocity_std**2,
        ]
    )

    rows = []
    previous_time = None
    for _, row in df.iterrows():
        if previous_time is None or "time" not in df.columns:
            dt = params.default_dt
        else:
            dt = float(row["time"] - previous_time)
            if not np.isfinite(dt) or dt <= 0:
                dt = params.default_dt
            dt = min(dt, params.max_dt)
        previous_time = float(row["time"]) if "time" in df.columns else None

        state, cov = _predict(state, cov, dt, params)
        z_values = []
        valid_ids = []
        for anchor_id in anchor_ids:
            value = row[measurement_column(df, anchor_id)]
            if np.isfinite(value):
                z_values.append(float(value))
                valid_ids.append(anchor_id)

        if len(valid_ids) >= params.min_valid_anchors:
            state, cov, stats = _update(state, cov, np.asarray(z_values), anchors, valid_ids, params)
        else:
            stats = {
                "innovation_norm": np.nan,
                "valid_anchor_count": float(len(valid_ids)),
                "rejected_anchor_count": float(len(anchor_ids) - len(valid_ids)),
                "updated": 0.0,
            }

        out = row.to_dict()
        out.update(
            {
                "dt": dt,
                "ekf_x": float(state[0]),
                "ekf_y": float(state[1]),
                "ekf_vx": float(state[2]),
                "ekf_vy": float(state[3]),
                "ekf_cov_x": float(cov[0, 0]),
                "ekf_cov_y": float(cov[1, 1]),
                "ekf_cov_vx": float(cov[2, 2]),
                "ekf_cov_vy": float(cov[3, 3]),
                **stats,
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def run_ekf_dataset(df: pd.DataFrame, anchors: dict[str, dict[str, float]], config: dict[str, Any]) -> pd.DataFrame:
    parts = []
    for (_, trajectory), part in df.groupby(["split", "trajectory"], sort=False):
        parts.append(run_ekf_track(part, anchors, config))
    return pd.concat(parts, ignore_index=True)
