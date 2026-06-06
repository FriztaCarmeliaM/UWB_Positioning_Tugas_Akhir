"""Strict ablation with a fine-grained error ladder (%<3/5/10/20 cm) for all
three patterns, plus two extra *valid* methods to probe the dynamic error floor:

  - Calibrated-raw multilateration (position from calibrated ranges, nominal
    anchors) -- isolates the effect of range calibration at position level.
  - EKF + offline zero-phase smoother -- a forward-backward moving average on the
    EKF position (window fixed from robot dynamics, NOT tuned on test; standard
    offline post-processing; uses no ground truth).

Everything is computed on the held-out TEST split of each pattern's latest run.
No test data is used to choose any parameter.

Run AFTER the pipelines have produced predictions:
    python scripts/15_ablation_error_ladder.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "20260606_error_budget"
ANCHORS = {"1": (2.26, 4.60), "2": (0.0, 0.0), "3": (4.55, 0.0)}

PATTERNS = {
    "Pola L": ("uwb_dataset_baru_l_pipeline", "L3_gt"),
    "Segitiga": ("uwb_dataset_baru_segitiga_pipeline", "segitiga3_gt"),
    "Kotak 10-loop": ("uwb_10loop_moretrain_pipeline", "10lup2_trilat_gt"),
}

# Zero-phase smoother window: robot ~0.2 m/s, ~20 Hz logging. An 11-sample
# (~0.55 s) centered window removes measurement jitter without distorting the
# ~10 s-per-segment motion. Fixed from physics, not tuned on test.
SMOOTH_WINDOW = 11


def latest_run(base: str) -> Path:
    return Path((ROOT / "outputs" / base / "latest_run.txt").read_text(encoding="utf-8").strip())


def multilaterate(r: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ids = list(ANCHORS.keys())
    a0 = ANCHORS[ids[0]]
    d0 = r[ids[0]]
    A_rows, rhs = [], []
    for a in ids[1:]:
        ax, ay = ANCHORS[a]
        A_rows.append((2 * (ax - a0[0]), 2 * (ay - a0[1])))
        rhs.append(d0**2 - r[a] ** 2 - a0[0] ** 2 + ax**2 - a0[1] ** 2 + ay**2)
    a11 = sum(x[0] * x[0] for x in A_rows)
    a12 = sum(x[0] * x[1] for x in A_rows)
    a22 = sum(x[1] * x[1] for x in A_rows)
    b1 = sum(x[0] * rhs[i] for i, x in enumerate(A_rows))
    b2 = sum(x[1] * rhs[i] for i, x in enumerate(A_rows))
    det = a11 * a22 - a12 * a12
    x = (a22 * b1 - a12 * b2) / det
    y = (-a12 * b1 + a11 * b2) / det
    return x, y


def zero_phase_ma(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    k = np.ones(window) / window
    fwd = np.convolve(values, k, mode="same")
    bwd = np.convolve(values[::-1], k, mode="same")[::-1]
    return 0.5 * (fwd + bwd)


def ladder(gt: np.ndarray, pred: np.ndarray) -> dict:
    e = np.linalg.norm(pred - gt, axis=1)
    e = e[np.isfinite(e)]
    return {
        "rmse_cm": np.sqrt(np.mean(e**2)) * 100,
        "mae_cm": np.mean(e) * 100,
        "median_cm": np.median(e) * 100,
        "p90_cm": np.percentile(e, 90) * 100,
        "p95_cm": np.percentile(e, 95) * 100,
        "max_cm": np.max(e) * 100,
        "pct_below_3cm": np.mean(e <= 0.03) * 100,
        "pct_below_5cm": np.mean(e <= 0.05) * 100,
        "pct_below_10cm": np.mean(e <= 0.10) * 100,
        "pct_below_20cm": np.mean(e <= 0.20) * 100,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for label, (base, test_track) in PATTERNS.items():
        run = latest_run(base)
        df = pd.read_csv(run / "06_evaluation" / "predictions.csv")
        test = df[df["split"] == "test"].copy()
        gt = test[["gt_x", "gt_y"]].to_numpy(dtype=float)

        # calibrated-raw multilateration
        rcal = {a: pd.to_numeric(test[f"range_cal_{a}"], errors="coerce").to_numpy(dtype=float) for a in ANCHORS}
        cal_x, cal_y = multilaterate(rcal)

        # EKF + offline zero-phase smoother, per test trajectory (no cross-track mixing)
        sm_x = test["ekf_x"].to_numpy(dtype=float).copy()
        sm_y = test["ekf_y"].to_numpy(dtype=float).copy()
        for _, idx in test.groupby("trajectory", sort=False).groups.items():
            pos = test.index.get_indexer(idx)
            sm_x[pos] = zero_phase_ma(test.loc[idx, "ekf_x"].to_numpy(dtype=float), SMOOTH_WINDOW)
            sm_y[pos] = zero_phase_ma(test.loc[idx, "ekf_y"].to_numpy(dtype=float), SMOOTH_WINDOW)

        methods = {
            "1. Raw trilateration": test[["raw_x", "raw_y"]].to_numpy(dtype=float),
            "2. Calibrated-raw multilat": np.stack([cal_x, cal_y], axis=1),
            "3. EKF (guarded)": test[["ekf_x", "ekf_y"]].to_numpy(dtype=float),
            "4. EKF + offline smoother": np.stack([sm_x, sm_y], axis=1),
            "5. EKF + LSTM residual": test[["lstm_x", "lstm_y"]].to_numpy(dtype=float),
            "6. EKF + LSTM + constraint": test[["constraint_x", "constraint_y"]].to_numpy(dtype=float),
        }
        for name, pred in methods.items():
            row = {"pattern": label, "method": name, **ladder(gt, pred)}
            all_rows.append(row)

    table = pd.DataFrame(all_rows)
    table.to_csv(OUT / "ablation_error_ladder.csv", index=False)
    pd.set_option("display.width", 240)
    show = ["rmse_cm", "mae_cm", "median_cm", "p90_cm", "p95_cm", "max_cm",
            "pct_below_3cm", "pct_below_5cm", "pct_below_10cm", "pct_below_20cm"]
    for label in PATTERNS:
        print("\n" + "=" * 110)
        print(label, "(test held-out)")
        print("=" * 110)
        sub = table[table["pattern"] == label].set_index("method")[show]
        print(sub.round(2).to_string())
    print(f"\nArtifacts -> {OUT / 'ablation_error_ladder.csv'}")


if __name__ == "__main__":
    main()
