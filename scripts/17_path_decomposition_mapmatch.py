"""Cross-track vs along-track decomposition + aggressive map-matching, to chase
~5 cm for presentation with HONEST labels.

Two ideas:
  (A) Cross-track-to-commanded-path: perpendicular distance from the sensor
      estimate to the KNOWN commanded path polyline. This is timing-independent
      (it does not use the noisy GT time-parameterisation) and is the real
      "does the estimate stay in the path corridor" accuracy -- assuming the
      robot physically followed the marked path. Labelled clearly.
  (B) Map-matched 2D vs GT: project the EKF+LSTM estimate onto the path polyline
      and forward-backward smooth. Removes cross-track error; the residual is the
      along-track (timing) error. This is a trajectory-constrained DEMONSTRATION.

We also decompose each method's 2D error vs GT into along/cross components using
the GT tangent, to show how much of the 9-10 cm is timing (along) vs sensor
(cross).

No test data is used to fit anything; the path polyline is the commanded route
known from the experiment design; the smoother window is fixed from robot speed.

Run:
    python scripts/17_path_decomposition_mapmatch.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "20260606_error_budget"

PATHS = {
    "Pola L": [(1, 1), (1, 3), (1, 1), (3, 1), (1, 1)],
    "Segitiga": [(1, 1), (1, 3), (3, 3), (1, 1)],
    "Kotak 10-loop": [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)],
}
RUNS = {
    "Pola L": ("uwb_dataset_baru_l_pipeline", "L3_gt"),
    "Segitiga": ("uwb_dataset_baru_segitiga_pipeline", "segitiga3_gt"),
    "Kotak 10-loop": ("uwb_10loop_moretrain_pipeline", "10lup2_trilat_gt"),
}
SMOOTH_WINDOW = 9


def latest_run(base):
    return Path((ROOT / "outputs" / base / "latest_run.txt").read_text(encoding="utf-8").strip())


def project_to_polyline(px, py, pts):
    """Return (qx, qy, dist) nearest point on the polyline to (px,py)."""
    best = (None, None, np.inf)
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        abx, aby = bx - ax, by - ay
        denom = abx * abx + aby * aby
        if denom <= 1e-12:
            qx, qy = ax, ay
        else:
            t = ((px - ax) * abx + (py - ay) * aby) / denom
            t = min(1.0, max(0.0, t))
            qx, qy = ax + t * abx, ay + t * aby
        d = np.hypot(px - qx, py - qy)
        if d < best[2]:
            best = (qx, qy, d)
    return best


def zero_phase_ma(v, w):
    if w <= 1:
        return v
    k = np.ones(w) / w
    f = np.convolve(v, k, mode="same")
    b = np.convolve(v[::-1], k, mode="same")[::-1]
    return 0.5 * (f + b)


def ladder(err_cm):
    e = err_cm[np.isfinite(err_cm)]
    return {
        "rmse_cm": float(np.sqrt(np.mean(e**2))),
        "mae_cm": float(np.mean(e)),
        "median_cm": float(np.median(e)),
        "p90_cm": float(np.percentile(e, 90)),
        "p95_cm": float(np.percentile(e, 95)),
        "pct_below_3cm": float(np.mean(e <= 3) * 100),
        "pct_below_5cm": float(np.mean(e <= 5) * 100),
        "pct_below_10cm": float(np.mean(e <= 10) * 100),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    decomp_rows = []
    for label, (base, test_track) in RUNS.items():
        pts = PATHS[label]
        df = pd.read_csv(latest_run(base) / "06_evaluation" / "predictions.csv")
        test = df[df["split"] == "test"].copy().reset_index(drop=True)
        gt = test[["gt_x", "gt_y"]].to_numpy(dtype=float)

        # GT tangent (per test track, time-ordered) for along/cross decomposition
        tang = np.zeros_like(gt)
        for _, idx in test.groupby("trajectory", sort=False).groups.items():
            pos = test.index.get_indexer(idx)
            g = gt[pos]
            d = np.gradient(g, axis=0)
            norm = np.hypot(d[:, 0], d[:, 1])
            norm[norm < 1e-9] = 1.0
            tang[pos] = d / norm[:, None]

        methods = {
            "Raw trilateration": test[["raw_x", "raw_y"]].to_numpy(dtype=float),
            "EKF (guarded)": test[["ekf_x", "ekf_y"]].to_numpy(dtype=float),
            "EKF + LSTM": test[["lstm_x", "lstm_y"]].to_numpy(dtype=float),
            "EKF + LSTM + current constraint": test[["constraint_x", "constraint_y"]].to_numpy(dtype=float),
        }

        # ---- aggressive map-matched (project EKF+LSTM to path, then smooth) ----
        lstm = methods["EKF + LSTM"]
        proj = np.array([project_to_polyline(x, y, pts)[:2] for x, y in lstm], dtype=float)
        mm = proj.copy()
        for _, idx in test.groupby("trajectory", sort=False).groups.items():
            pos = test.index.get_indexer(idx)
            mm[pos, 0] = zero_phase_ma(proj[pos, 0], SMOOTH_WINDOW)
            mm[pos, 1] = zero_phase_ma(proj[pos, 1], SMOOTH_WINDOW)
        methods["MAP-MATCHED (project+smooth) [demo]"] = mm

        for name, pred in methods.items():
            err2d = np.linalg.norm(pred - gt, axis=1) * 100
            rows.append({"pattern": label, "method": name, "metric": "2D vs GT", **ladder(err2d)})
            # cross-track to commanded path (timing-independent)
            xtrack = np.array([project_to_polyline(x, y, pts)[2] for x, y in pred]) * 100
            rows.append({"pattern": label, "method": name, "metric": "cross-track to path", **ladder(xtrack)})
            # along/cross decomposition vs GT
            ev = pred - gt
            along = np.abs(np.sum(ev * tang, axis=1)) * 100
            normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
            cross = np.abs(np.sum(ev * normal, axis=1)) * 100
            decomp_rows.append({"pattern": label, "method": name,
                                "along_track_mae_cm": float(np.mean(along)),
                                "cross_track_mae_cm": float(np.mean(cross))})

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "mapmatch_ablation.csv", index=False)
    decomp = pd.DataFrame(decomp_rows)
    decomp.to_csv(OUT / "along_cross_decomposition.csv", index=False)

    pd.set_option("display.width", 240)
    show = ["rmse_cm", "mae_cm", "median_cm", "p90_cm", "p95_cm", "pct_below_3cm", "pct_below_5cm", "pct_below_10cm"]
    for label in RUNS:
        for metric in ["2D vs GT", "cross-track to path"]:
            print("\n" + "=" * 116)
            print(f"{label}  --  metric: {metric}")
            print("=" * 116)
            sub = table[(table["pattern"] == label) & (table["metric"] == metric)].set_index("method")[show]
            print(sub.round(2).to_string())
    print("\n" + "=" * 60)
    print("ALONG- vs CROSS-track MAE decomposition (vs GT)")
    print("=" * 60)
    print(decomp.round(2).to_string(index=False))
    print(f"\nArtifacts -> {OUT}")


if __name__ == "__main__":
    main()
