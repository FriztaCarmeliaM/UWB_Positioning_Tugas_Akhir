"""Error-budget / noise-floor analysis for the UWB positioning thesis.

Goal: decide -- with evidence, not opinion -- whether 5 cm or 3 cm 2D error is
reachable from THIS data, by decomposing the error into:

  total^2 ~= sensor_precision^2 (irreducible noise, from static repeatability)
           + anchor/spatial_bias^2 (multipath/NLOS, partly correctable)
           + GDOP amplification (geometry)
           + GT_timing^2 (dynamic only: space-bar delay x robot speed)
           + model_error^2

Static "Diam" recordings (tag stationary at known points) give the pure sensor
floor with EXACT ground truth (no timing problem). Dynamic GT timing budget is
estimated from waypoint spacing / event times.

No test data, no model fitting -- pure measurement analysis.

Run:
    python scripts/14_error_budget_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "20260606_error_budget"

ANCHORS = {"1": (2.26, 4.60), "2": (0.0, 0.0), "3": (4.55, 0.0)}
STATIC_POINTS = {"1": (1.0, 1.0), "2": (3.0, 1.0), "3": (3.0, 3.0), "4": (1.0, 3.0), "5": (2.0, 2.0)}


def true_dists(point: tuple[float, float]) -> dict[str, float]:
    return {a: float(np.hypot(point[0] - ax, point[1] - ay)) for a, (ax, ay) in ANCHORS.items()}


def _solve2x2(a11, a12, a21, a22, b1, b2):
    """Manual 2x2 solve / inverse (avoids np.linalg LAPACK which crashes in this
    conda env). Works elementwise when b1/b2 are arrays."""
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-15:
        return None
    x = (a22 * b1 - a12 * b2) / det
    y = (-a21 * b1 + a11 * b2) / det
    return x, y


def multilaterate(el: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form linear least-squares multilateration per sample, solved via the
    normal equations with a manual 2x2 inverse (3 anchors -> overdetermined 2x2)."""
    ids = list(ANCHORS.keys())
    a0 = ANCHORS[ids[0]]
    d0 = el[ids[0]]
    A_rows = []
    rhs = []
    for a in ids[1:]:
        ax, ay = ANCHORS[a]
        di = el[a]
        A_rows.append((2 * (ax - a0[0]), 2 * (ay - a0[1])))
        rhs.append(d0**2 - di**2 - a0[0] ** 2 + ax**2 - a0[1] ** 2 + ay**2)
    # Normal equations: (A^T A) p = A^T b, with A constant (2 rows), b per-sample
    ata11 = sum(r[0] * r[0] for r in A_rows)
    ata12 = sum(r[0] * r[1] for r in A_rows)
    ata22 = sum(r[1] * r[1] for r in A_rows)
    atb1 = sum(r[0] * rhs[i] for i, r in enumerate(A_rows))
    atb2 = sum(r[1] * rhs[i] for i, r in enumerate(A_rows))
    sol = _solve2x2(ata11, ata12, ata12, ata22, atb1, atb2)
    if sol is None:
        n = len(d0)
        return np.full(n, np.nan), np.full(n, np.nan)
    return sol[0], sol[1]


def gdop(point: tuple[float, float]) -> float:
    rows = []
    for ax, ay in ANCHORS.values():
        d = np.hypot(point[0] - ax, point[1] - ay)
        rows.append(((point[0] - ax) / d, (point[1] - ay) / d))
    g11 = sum(r[0] * r[0] for r in rows)
    g12 = sum(r[0] * r[1] for r in rows)
    g22 = sum(r[1] * r[1] for r in rows)
    det = g11 * g22 - g12 * g12
    if abs(det) < 1e-15:
        return float("nan")
    # trace of inverse of [[g11,g12],[g12,g22]] = (g11+g22)/det
    return float(np.sqrt((g11 + g22) / det))


def analyse_static() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    per_file = []
    per_anchor_bias = {a: [] for a in ANCHORS}        # global-bias estimate samples (point-level means)
    point_records = {}
    for path in sorted((ROOT / "dataset_baru" / "Diam").glob("*.csv")):
        pid = path.stem.split("_")[1]
        point = STATIC_POINTS[pid]
        td = true_dists(point)
        df = pd.read_csv(path)
        el = {a: pd.to_numeric(df[f"el{a}"], errors="coerce").to_numpy(dtype=float) for a in ANCHORS}
        finite = np.all([np.isfinite(el[a]) for a in ANCHORS], axis=0)
        el = {a: el[a][finite] for a in ANCHORS}
        x, y = multilaterate(el)
        pos_ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[pos_ok], y[pos_ok]
        rec = {
            "file": path.name, "point": pid, "true_x": point[0], "true_y": point[1],
            "n": int(len(x)), "gdop": gdop(point),
            "pos_mean_x": float(np.mean(x)), "pos_mean_y": float(np.mean(y)),
            "accuracy_cm": float(np.hypot(np.mean(x) - point[0], np.mean(y) - point[1]) * 100),
            "precision_cm": float(np.sqrt(np.var(x) + np.var(y)) * 100),  # repeatability (std of position)
            "rmse_static_cm": float(np.sqrt(np.mean((x - point[0]) ** 2 + (y - point[1]) ** 2)) * 100),
        }
        for a in ANCHORS:
            bias = float(np.mean(el[a]) - td[a])
            rec[f"bias{a}_cm"] = bias * 100
            rec[f"rangestd{a}_cm"] = float(np.std(el[a]) * 100)
            per_anchor_bias[a].append((pid, bias))
        per_file.append(rec)
        point_records.setdefault(pid, []).append(rec)

    per_file_df = pd.DataFrame(per_file)

    # Leave-one-point-out global bias correction: estimate each anchor's bias as the
    # mean over OTHER points, apply, recompute static RMSE. Tests if a single global
    # per-anchor bias explains the error (correctable) vs spatial multipath (a floor).
    loo_rows = []
    for pid, point in STATIC_POINTS.items():
        td = true_dists(point)
        # global bias per anchor from points != pid
        gbias = {}
        for a in ANCHORS:
            vals = [b for (p, b) in per_anchor_bias[a] if p != pid]
            gbias[a] = float(np.mean(vals)) if vals else 0.0
        # recompute position using bias-corrected ranges for this point's files
        accs, rmses = [], []
        for path in sorted((ROOT / "dataset_baru" / "Diam").glob(f"s2_{pid}_*.csv")):
            df = pd.read_csv(path)
            el = {a: pd.to_numeric(df[f"el{a}"], errors="coerce").to_numpy(dtype=float) - gbias[a] for a in ANCHORS}
            finite = np.all([np.isfinite(el[a]) for a in ANCHORS], axis=0)
            el = {a: el[a][finite] for a in ANCHORS}
            x, y = multilaterate(el)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            accs.append(np.hypot(np.mean(x) - point[0], np.mean(y) - point[1]) * 100)
            rmses.append(np.sqrt(np.mean((x - point[0]) ** 2 + (y - point[1]) ** 2)) * 100)
        loo_rows.append({
            "point": pid, "true_x": point[0], "true_y": point[1],
            "global_bias1_cm": gbias["1"] * 100, "global_bias2_cm": gbias["2"] * 100,
            "global_bias3_cm": gbias["3"] * 100,
            "accuracy_after_globalcal_cm": float(np.mean(accs)),
            "rmse_after_globalcal_cm": float(np.mean(rmses)),
        })
    loo_df = pd.DataFrame(loo_rows)

    summary = {
        "median_precision_cm": float(per_file_df["precision_cm"].median()),
        "median_range_std_cm": float(per_file_df[["rangestd1_cm", "rangestd2_cm", "rangestd3_cm"]].stack().median()),
        "raw_static_rmse_median_cm": float(per_file_df["rmse_static_cm"].median()),
        "raw_static_rmse_max_cm": float(per_file_df["rmse_static_cm"].max()),
        "raw_static_accuracy_median_cm": float(per_file_df["accuracy_cm"].median()),
        "raw_static_accuracy_max_cm": float(per_file_df["accuracy_cm"].max()),
        "globalcal_rmse_median_cm": float(loo_df["rmse_after_globalcal_cm"].median()),
        "globalcal_rmse_max_cm": float(loo_df["rmse_after_globalcal_cm"].max()),
        "globalcal_accuracy_median_cm": float(loo_df["accuracy_after_globalcal_cm"].median()),
        "globalcal_accuracy_max_cm": float(loo_df["accuracy_after_globalcal_cm"].max()),
    }
    return per_file_df, loo_df, summary


def analyse_dynamic_gt_timing() -> pd.DataFrame:
    """Estimate GT position uncertainty caused by waypoint-click timing jitter:
    GT_uncertainty ~= robot_speed * timing_jitter. Robot speed from waypoint
    spacing / segment duration; timing jitter from the per-session offset spread
    measured in scripts/12 (std ~0.38 s across sessions, ~0.5 s as a conservative
    within-experiment value)."""
    rows = []
    files = {
        "L1": "dataset_baru/MAJU (L)/L1.csv", "L2": "dataset_baru/MAJU (L)/L2.csv",
        "L3": "dataset_baru/MAJU (L)/L3.csv", "segitiga1": "dataset_baru/segitiga/segitiga1.csv",
        "segitiga3": "dataset_baru/segitiga/segitiga3.csv",
    }
    for name, rel in files.items():
        df = pd.read_csv(ROOT / rel)
        if "target_x" not in df.columns:
            continue
        ev = df[df["target_x"].notna()][["time", "target_x", "target_y"]].astype(float)
        ev = ev.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        seg_d = np.hypot(np.diff(ev["target_x"]), np.diff(ev["target_y"]))
        seg_t = np.diff(ev["time"].to_numpy())
        valid = seg_t > 0.5
        speeds = seg_d[valid] / seg_t[valid]
        speed = float(np.median(speeds)) if len(speeds) else float("nan")
        for jitter in (0.25, 0.5, 1.0):
            rows.append({
                "track": name, "median_speed_mps": speed, "timing_jitter_s": jitter,
                "gt_uncertainty_cm": speed * jitter * 100,
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_file, loo, summary = analyse_static()
    dyn = analyse_dynamic_gt_timing()
    per_file.to_csv(OUT / "static_per_file.csv", index=False)
    loo.to_csv(OUT / "static_global_bias_correction.csv", index=False)
    dyn.to_csv(OUT / "dynamic_gt_timing_budget.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT / "noise_floor_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("=" * 78)
    print("STATIC NOISE-FLOOR (tag stationary, EXACT ground truth)")
    print("=" * 78)
    cols = ["file", "point", "gdop", "precision_cm", "accuracy_cm", "rmse_static_cm",
            "bias1_cm", "bias2_cm", "bias3_cm", "rangestd1_cm", "rangestd2_cm", "rangestd3_cm"]
    print(per_file[cols].round(2).to_string(index=False))
    print("\nPer-point mean static RMSE (cm):")
    print(per_file.groupby("point")[["gdop", "precision_cm", "accuracy_cm", "rmse_static_cm"]].mean().round(2).to_string())

    print("\n" + "=" * 78)
    print("AFTER global (leave-one-point-out) per-anchor bias correction")
    print("=" * 78)
    print(loo.round(2).to_string(index=False))

    print("\n" + "=" * 78)
    print("DYNAMIC GT TIMING BUDGET (error injected by waypoint-click jitter)")
    print("=" * 78)
    print(dyn.round(2).to_string(index=False))

    print("\n" + "=" * 78)
    print("NOISE-FLOOR SUMMARY")
    print("=" * 78)
    for k, v in summary.items():
        print(f"  {k:38s}: {v:6.2f} cm")
    print(f"\nArtifacts -> {OUT}")


if __name__ == "__main__":
    main()
