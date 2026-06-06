"""Diagnose waypoint (space-bar click) timing delay for the dataset_baru tracks.

Motivation
----------
Ground truth for the L and triangle patterns is built by linearly interpolating
between waypoint *events* (the rows where ``target_x``/``target_y`` are logged at
the instant the operator marks "robot is at this waypoint"). That mark carries a
human reaction delay. If the delay were a fixed constant we could subtract it
from every event time (a no-leakage correction estimated on the TRAIN tracks).

This script measures, per track, the time offset ``tau`` that best aligns the
interpolated GT to the *raw* multilateration trajectory (a model-free position
estimate). It then reports the train-only global offset.

Conclusion drawn from the numbers (see README): the optimal offset is *not*
consistent across sessions, so a single global correction would fit noise rather
than a real reaction delay. We therefore document the timing uncertainty as the
main reason the L pattern is harder, but do NOT bake an unjustified shift into
the ground truth.

Run:
    python scripts/12_diagnose_waypoint_timing.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "20260606_timing_diagnostic"

TRACKS = {
    "L1": ("dataset_baru/MAJU (L)/L1.csv", "train"),
    "L2": ("dataset_baru/MAJU (L)/L2.csv", "train"),
    "L3": ("dataset_baru/MAJU (L)/L3.csv", "test"),
    "segitiga1": ("dataset_baru/segitiga/segitiga1.csv", "train"),
    "segitiga2": ("dataset_baru/segitiga/segitiga2.csv", "train"),
    "segitiga3": ("dataset_baru/segitiga/segitiga3.csv", "test"),
}

TAUS = np.arange(-3.0, 3.01, 0.25)


def build_gt(df: pd.DataFrame, tau: float) -> tuple[np.ndarray, np.ndarray]:
    ev = df[df["target_x"].notna() & df["target_y"].notna()][["time", "target_x", "target_y"]].copy()
    ev = ev.astype(float).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    et = ev["time"].to_numpy() + tau
    ex = ev["target_x"].to_numpy()
    ey = ev["target_y"].to_numpy()
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy()
    si = np.clip(np.searchsorted(et, t, side="right") - 1, 0, len(et) - 2)
    a = np.clip((t - et[si]) / np.maximum(et[si + 1] - et[si], 1e-12), 0, 1)
    gx = ex[si] + a * (ex[si + 1] - ex[si])
    gy = ey[si] + a * (ey[si + 1] - ey[si])
    gx[t <= et[0]] = ex[0]
    gy[t <= et[0]] = ey[0]
    gx[t >= et[-1]] = ex[-1]
    gy[t >= et[-1]] = ey[-1]
    return gx, gy


def median_raw_error(df: pd.DataFrame, gx: np.ndarray, gy: np.ndarray) -> float:
    rx = pd.to_numeric(df["x"], errors="coerce").rolling(7, center=True, min_periods=1).median().to_numpy()
    ry = pd.to_numeric(df["y"], errors="coerce").rolling(7, center=True, min_periods=1).median().to_numpy()
    ev = df[df["target_x"].notna()]
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy()
    mask = (t >= ev["time"].min()) & (t <= ev["time"].max())
    err = np.hypot(rx - gx, ry - gy)[mask]
    return float(np.nanmedian(err))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (rel, split) in TRACKS.items():
        df = pd.read_csv(ROOT / rel)
        if "target_x" not in df.columns:
            continue
        errs = np.array([median_raw_error(df, *build_gt(df, tau)) for tau in TAUS])
        best_i = int(np.argmin(errs))
        zero_i = int(np.argmin(np.abs(TAUS)))
        rows.append(
            {
                "track": name,
                "split": split,
                "tau0_median_cm": errs[zero_i] * 100,
                "best_tau_s": float(TAUS[best_i]),
                "best_median_cm": errs[best_i] * 100,
                "improvement_cm": (errs[zero_i] - errs[best_i]) * 100,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "waypoint_timing_offsets.csv", index=False)

    train = table[table["split"] == "train"]
    global_tau = float(train["best_tau_s"].mean()) if not train.empty else 0.0
    print(table.to_string(index=False))
    print(f"\nTrain-only global offset estimate (mean of train best_tau): {global_tau:+.3f} s")
    print(f"Per-track spread of best_tau (std): {table['best_tau_s'].std():.3f} s")
    print(
        "Interpretation: the best offset differs markedly between sessions, so the\n"
        "space-bar delay is not a fixed constant. A single global GT shift would fit\n"
        "session noise and is therefore NOT applied. See README section on pola L."
    )
    pd.DataFrame(
        [{"train_global_tau_s": global_tau, "best_tau_std_s": float(table["best_tau_s"].std())}]
    ).to_csv(OUT / "waypoint_timing_summary.csv", index=False)


if __name__ == "__main__":
    main()
