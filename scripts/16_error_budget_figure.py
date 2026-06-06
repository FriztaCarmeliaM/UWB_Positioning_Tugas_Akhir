"""Error-budget figure: shows that the achieved EKF+LSTM RMSE has reached the
theoretical dynamic floor sqrt(sensor^2 + GT_timing^2) for every pattern.

PIL backend (matplotlib savefig crashes in this conda env).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "20260606_error_budget"

# From scripts/14 (static noise floor) and scripts/15 (achieved test RMSE).
SENSOR_FLOOR_CM = 4.0       # median static RMSE in good geometry (precision ~3.1 cm)
GT_TIMING_CM = 10.0         # dynamic GT uncertainty at ~0.2 m/s x ~0.5 s click jitter
DATA = {
    # pattern: achieved EKF+LSTM RMSE (cm)
    "Pola L": 11.71,
    "Segitiga": 12.48,
    "Kotak 10-loop": 11.63,
}
BARS = ["Sensor floor\n(static)", "GT timing\nfloor (0.5 s)", "Quadrature\nfloor", "Achieved\nEKF+LSTM"]
COLORS = ["#9e9e9e", "#f59e0b", "#1565c0", "#00796b"]


def font(size):
    for n in ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    quad = float(np.hypot(SENSOR_FLOOR_CM, GT_TIMING_CM))
    w, h = 1040, 560
    plot = {"left": 80, "top": 70, "right": w - 30, "bottom": h - 110}
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    y_max = 16.0
    d.rectangle([plot["left"], plot["top"], plot["right"], plot["bottom"]], outline="#333333")
    for k in range(9):
        yv = k * y_max / 8
        y = plot["bottom"] - yv / y_max * (plot["bottom"] - plot["top"])
        if 0 < k < 8:
            d.line([(plot["left"], y), (plot["right"], y)], fill="#dddddd")
        d.text((plot["left"] - 38, y - 7), f"{yv:.0f}", fill="#333333", font=font(11))

    patterns = list(DATA.keys())
    group_w = (plot["right"] - plot["left"]) / len(patterns)
    bw = group_w * 0.8 / len(BARS)
    for gi, p in enumerate(patterns):
        vals = [SENSOR_FLOOR_CM, GT_TIMING_CM, quad, DATA[p]]
        gx = plot["left"] + gi * group_w
        for mi, v in enumerate(vals):
            x0 = gx + group_w * 0.1 + mi * bw
            y0 = plot["bottom"] - min(v, y_max) / y_max * (plot["bottom"] - plot["top"])
            d.rectangle([x0, y0, x0 + bw - 2, plot["bottom"]], fill=COLORS[mi])
            d.text((x0 - 2, y0 - 14), f"{v:.1f}", fill="#111111", font=font(10))
        d.text((gx + group_w / 2 - 36, plot["bottom"] + 8), p, fill="#111111", font=font(13))

    d.text((plot["left"], 22), "Error Budget: Achieved RMSE has reached the sqrt(sensor^2 + GT_timing^2) floor",
           fill="#111111", font=font(17))
    d.text((14, (plot["top"] + plot["bottom"]) // 2 - 20), "RMSE", fill="#111111", font=font(13))
    d.text((14, (plot["top"] + plot["bottom"]) // 2 - 2), "(cm)", fill="#111111", font=font(13))
    lx, ly = plot["left"] + 20, plot["bottom"] + 40
    for mi, name in enumerate(BARS):
        d.rectangle([lx, ly, lx + 18, ly + 12], fill=COLORS[mi])
        d.text((lx + 24, ly - 2), name.replace("\n", " "), fill="#111111", font=font(11))
        lx += 230
    img.save(OUT / "error_budget_floor.png")
    print(f"quadrature floor = {quad:.2f} cm")
    print(f"Saved -> {OUT / 'error_budget_floor.png'}")


if __name__ == "__main__":
    main()
