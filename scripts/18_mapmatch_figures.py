"""Figures for the 5 cm presentation story (PIL backend):
  1) cross_track_vs_2d.png  -- per pattern, EKF+LSTM full-2D MAE vs cross-track
     MAE, with the 5 cm target line. Shows cross-track (sensor) accuracy < 5 cm.
  2) mapmatched_demo_<pattern>.png -- GT path, EKF+LSTM, and MAP-MATCHED estimate
     overlaid (equal aspect, 0.5 m grid). Demonstration that the constrained
     estimate sits on the commanded path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "20260606_error_budget"

PATHS = {
    "Pola L": [(1, 1), (1, 3), (1, 1), (3, 1), (1, 1)],
    "Segitiga": [(1, 1), (1, 3), (3, 3), (1, 1)],
    "Kotak 10-loop": [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)],
}
RUNS = {
    "Pola L": "uwb_dataset_baru_l_pipeline",
    "Segitiga": "uwb_dataset_baru_segitiga_pipeline",
    "Kotak 10-loop": "uwb_10loop_moretrain_pipeline",
}
# from scripts/17 along_cross_decomposition (EKF+LSTM rows)
TWO_D_MAE = {"Pola L": 9.71, "Segitiga": 9.89, "Kotak 10-loop": 9.89}
XTRACK_MAE = {"Pola L": 2.59, "Segitiga": 1.67, "Kotak 10-loop": 2.78}
SMOOTH_WINDOW = 9


def font(s):
    for n in ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(n, s)
        except OSError:
            continue
    return ImageFont.load_default()


def latest_run(base):
    return Path((ROOT / "outputs" / base / "latest_run.txt").read_text(encoding="utf-8").strip())


def project(px, py, pts):
    best = (px, py, np.inf)
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        abx, aby = bx - ax, by - ay
        den = abx * abx + aby * aby
        t = 0.0 if den <= 1e-12 else min(1.0, max(0.0, ((px - ax) * abx + (py - ay) * aby) / den))
        qx, qy = ax + t * abx, ay + t * aby
        d = np.hypot(px - qx, py - qy)
        if d < best[2]:
            best = (qx, qy, d)
    return best[0], best[1]


def zpma(v, w):
    k = np.ones(w) / w
    return 0.5 * (np.convolve(v, k, "same") + np.convolve(v[::-1], k, "same")[::-1])


def bar_cross_vs_2d():
    w, h = 980, 540
    plot = {"left": 80, "top": 70, "right": w - 30, "bottom": h - 100}
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    y_max = 12.0
    d.rectangle([plot["left"], plot["top"], plot["right"], plot["bottom"]], outline="#333333")
    for k in range(7):
        yv = k * y_max / 6
        y = plot["bottom"] - yv / y_max * (plot["bottom"] - plot["top"])
        if 0 < k < 6:
            d.line([(plot["left"], y), (plot["right"], y)], fill="#dddddd")
        d.text((plot["left"] - 38, y - 7), f"{yv:.0f}", fill="#333", font=font(11))
    yt = plot["bottom"] - 5.0 / y_max * (plot["bottom"] - plot["top"])
    d.line([(plot["left"], yt), (plot["right"], yt)], fill="#d32f2f", width=2)
    d.text((plot["right"] - 96, yt - 16), "Target 5 cm", fill="#d32f2f", font=font(12))
    pats = list(TWO_D_MAE.keys())
    gw = (plot["right"] - plot["left"]) / len(pats)
    bw = gw * 0.7 / 2
    for gi, p in enumerate(pats):
        gx = plot["left"] + gi * gw
        for mi, (v, col, lab) in enumerate([(TWO_D_MAE[p], "#00796b", "2D vs GT"),
                                            (XTRACK_MAE[p], "#1565c0", "cross-track")]):
            x0 = gx + gw * 0.15 + mi * bw
            y0 = plot["bottom"] - min(v, y_max) / y_max * (plot["bottom"] - plot["top"])
            d.rectangle([x0, y0, x0 + bw - 4, plot["bottom"]], fill=col)
            d.text((x0 + 2, y0 - 14), f"{v:.1f}", fill="#111", font=font(11))
        d.text((gx + gw / 2 - 36, plot["bottom"] + 8), p, fill="#111", font=font(13))
    d.text((plot["left"], 24), "Sensor-only EKF+LSTM: full-2D (timing-limited) vs cross-track (true lateral accuracy)",
           fill="#111", font=font(15))
    d.text((14, (plot["top"] + plot["bottom"]) // 2 - 10), "MAE (cm)", fill="#111", font=font(12))
    lx, ly = plot["left"] + 20, plot["bottom"] + 40
    for col, lab in [("#00796b", "Full 2D vs GT (incl. along-track timing)"), ("#1565c0", "Cross-track to commanded path (sensor)")]:
        d.rectangle([lx, ly, lx + 18, ly + 12], fill=col)
        d.text((lx + 24, ly - 2), lab, fill="#111", font=font(11))
        lx += 380
    img.save(OUT / "cross_track_vs_2d.png")


def traj_demo(label, base):
    pts = PATHS[label]
    df = pd.read_csv(latest_run(base) / "06_evaluation" / "predictions.csv")
    t = df[df["split"] == "test"].copy().reset_index(drop=True)
    lstm = t[["lstm_x", "lstm_y"]].to_numpy(dtype=float)
    proj = np.array([project(x, y, pts) for x, y in lstm])
    mm = proj.copy()
    for _, idx in t.groupby("trajectory", sort=False).groups.items():
        pos = t.index.get_indexer(idx)
        mm[pos, 0] = zpma(proj[pos, 0], SMOOTH_WINDOW)
        mm[pos, 1] = zpma(proj[pos, 1], SMOOTH_WINDOW)
    S = 900
    pad, lo, hi = 70, 0.5, 4.0
    img = Image.new("RGB", (S, S + 40), "white")
    d = ImageDraw.Draw(img)
    def to_px(x, y):
        px = pad + (x - lo) / (hi - lo) * (S - 2 * pad)
        py = (S - pad) - (y - lo) / (hi - lo) * (S - 2 * pad)
        return px, py
    d.rectangle([pad, pad, S - pad, S - pad], outline="#333")
    g = lo
    while g <= hi + 1e-6:
        x0, _ = to_px(g, lo); _, y0 = to_px(lo, g)
        d.line([(x0, pad), (x0, S - pad)], fill="#eee")
        d.line([(pad, y0), (S - pad, y0)], fill="#eee")
        d.text((x0 - 8, S - pad + 6), f"{g:.1f}", fill="#333", font=font(10))
        d.text((pad - 30, y0 - 6), f"{g:.1f}", fill="#333", font=font(10))
        g += 0.5
    def draw(series, col, wdt):
        p = [to_px(x, y) for x, y in series if np.isfinite(x) and np.isfinite(y)]
        if len(p) > 1:
            d.line(p, fill=col, width=wdt, joint="curve")
    draw(lstm[::3], "#00796b", 1)
    draw(pts, "#263238", 4)
    draw(mm[::2], "#1565c0", 2)
    d.text((pad, 22), f"Map-matched DEMO (trajectory-constrained) - {label}", fill="#111", font=font(16))
    d.text((pad, 44), "X (m)  ->            cross-track to path ~0; 2D-vs-GT still timing-limited", fill="#555", font=font(11))
    lx, ly = pad + 10, pad + 8
    for col, lab in [("#263238", "Commanded path (prior)"), ("#00796b", "EKF+LSTM (sensor-only)"), ("#1565c0", "Map-matched (demo)")]:
        d.line([(lx, ly + 7), (lx + 26, ly + 7)], fill=col, width=4)
        d.text((lx + 32, ly), lab, fill="#111", font=font(11))
        ly += 20
    img.save(OUT / f"mapmatched_demo_{label.split()[0].lower()}.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bar_cross_vs_2d()
    for label, base in RUNS.items():
        traj_demo(label, base)
    print(f"Saved figures -> {OUT}")


if __name__ == "__main__":
    main()
