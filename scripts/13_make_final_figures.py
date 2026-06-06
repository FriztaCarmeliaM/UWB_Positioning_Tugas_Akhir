"""Collect the final presentation figures and metrics for all three patterns.

Reads the latest run of each pipeline (Pola L, Segitiga, Kotak 10-loop), copies
the trajectory / error / CDF / method-comparison / residual plots produced by
stage 07, draws a PIL training-loss curve from the LSTM history, and builds one
combined MAE summary bar chart with the 10 cm target line.

PIL (Pillow) is used instead of matplotlib because matplotlib's Agg PNG writer
crashes on this Windows/conda environment (libpng/freetype DLL conflict), while
the project's PIL plotting backend is known to work.

Output:
    docs/results/20260606_final/<pattern>/...
    docs/results/20260606_final/summary_mae_all_patterns.png

Run AFTER the three pipelines have finished:
    python scripts/13_make_final_figures.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "20260606_final"

PATTERNS = {
    "pola_l": {"label": "Pola L", "base": "uwb_dataset_baru_l_pipeline", "test": "L3_gt"},
    "segitiga": {"label": "Segitiga", "base": "uwb_dataset_baru_segitiga_pipeline", "test": "segitiga3_gt"},
    "kotak": {"label": "Kotak 10-loop", "base": "uwb_10loop_moretrain_pipeline", "test": "10lup2_trilat_gt"},
}

METHOD_ORDER = [
    "Raw trilateration",
    "EKF only",
    "EKF + LSTM residual",
    "EKF + LSTM + trajectory constraint",
]
METHOD_SHORT = {
    "Raw trilateration": "Raw",
    "EKF only": "EKF",
    "EKF + LSTM residual": "EKF+LSTM",
    "EKF + LSTM + trajectory constraint": "Final (constraint)",
}
METHOD_COLOR = {
    "Raw trilateration": "#9e9e9e",
    "EKF only": "#d32f2f",
    "EKF + LSTM residual": "#00796b",
    "EKF + LSTM + trajectory constraint": "#1565c0",
}


def font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "calibri.ttf", "segoeui.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def latest_run(base: str) -> Path:
    marker = ROOT / "outputs" / base / "latest_run.txt"
    run_dir = Path(marker.read_text(encoding="utf-8").strip())
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir missing for {base}: {run_dir}")
    return run_dir


def safe_name(value: str) -> str:
    return value.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def plot_loss(history_csv: Path, dest: Path, title: str) -> None:
    hist = pd.read_csv(history_csv)
    w, h = 760, 470
    plot = {"left": 80, "top": 60, "right": w - 30, "bottom": h - 60}
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    series = [("Training loss", hist["loss"].to_numpy(dtype=float), "#1f77b4")]
    if "val_loss" in hist.columns:
        series.append(("Validation loss", hist["val_loss"].to_numpy(dtype=float), "#2ca02c"))
    all_y = np.concatenate([s[1] for s in series])
    y_max = float(np.nanmax(all_y)) * 1.05
    y_min = 0.0
    n = len(hist)
    d.rectangle([plot["left"], plot["top"], plot["right"], plot["bottom"]], outline="#333333")
    for k in range(6):  # gridlines + y ticks
        yv = y_min + k * (y_max - y_min) / 5
        y = plot["bottom"] - (yv - y_min) / (y_max - y_min) * (plot["bottom"] - plot["top"])
        if 0 < k < 5:
            d.line([(plot["left"], y), (plot["right"], y)], fill="#dddddd")
        d.text((plot["left"] - 55, y - 7), f"{yv:.3f}", fill="#333333", font=font(11))
    for k in range(6):  # x ticks (epoch)
        ev = 1 + k * (max(n - 1, 1)) / 5
        x = plot["left"] + (ev - 1) / max(n - 1, 1) * (plot["right"] - plot["left"])
        d.text((x - 10, plot["bottom"] + 8), f"{ev:.0f}", fill="#333333", font=font(11))
    for label, ys, color in series:
        pts = []
        for i, v in enumerate(ys):
            x = plot["left"] + i / max(n - 1, 1) * (plot["right"] - plot["left"])
            y = plot["bottom"] - (v - y_min) / (y_max - y_min) * (plot["bottom"] - plot["top"])
            pts.append((x, y))
        if len(pts) >= 2:
            d.line(pts, fill=color, width=2)
    d.text((plot["left"], 22), f"LSTM Residual Training Loss - {title}", fill="#111111", font=font(18))
    d.text(((plot["left"] + plot["right"]) // 2 - 20, plot["bottom"] + 30), "Epoch", fill="#111111", font=font(13))
    d.text((14, (plot["top"] + plot["bottom"]) // 2 - 30), "Huber", fill="#111111", font=font(13))
    d.text((14, (plot["top"] + plot["bottom"]) // 2 - 12), "loss", fill="#111111", font=font(13))
    lx, ly = plot["left"] + 20, plot["top"] + 12
    for label, _, color in series:
        d.line([(lx, ly + 7), (lx + 26, ly + 7)], fill=color, width=3)
        d.text((lx + 34, ly), label, fill="#111111", font=font(12))
        ly += 20
    img.save(dest)


def plot_summary(summary: pd.DataFrame, patterns: list[str], dest: Path) -> None:
    w, h = 1040, 560
    plot = {"left": 80, "top": 70, "right": w - 30, "bottom": h - 110}
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    y_max = 24.0  # cm, fixed so the 10 cm target is comparable across runs
    d.rectangle([plot["left"], plot["top"], plot["right"], plot["bottom"]], outline="#333333")
    for k in range(7):
        yv = k * y_max / 6
        y = plot["bottom"] - yv / y_max * (plot["bottom"] - plot["top"])
        if 0 < k < 6:
            d.line([(plot["left"], y), (plot["right"], y)], fill="#dddddd")
        d.text((plot["left"] - 40, y - 7), f"{yv:.0f}", fill="#333333", font=font(11))
    # 10 cm target line
    yt = plot["bottom"] - 10.0 / y_max * (plot["bottom"] - plot["top"])
    d.line([(plot["left"], yt), (plot["right"], yt)], fill="#d32f2f", width=2)
    d.text((plot["right"] - 110, yt - 16), "Target 10 cm", fill="#d32f2f", font=font(12))

    n_groups = len(patterns)
    n_m = len(METHOD_ORDER)
    group_w = (plot["right"] - plot["left"]) / n_groups
    bar_w = group_w * 0.8 / n_m
    for gi, p in enumerate(patterns):
        gx = plot["left"] + gi * group_w
        for mi, method in enumerate(METHOD_ORDER):
            sel = summary[(summary["pattern"] == p) & (summary["model"] == method)]
            if sel.empty:
                continue
            v = float(sel["mae_cm"].iloc[0])
            x0 = gx + group_w * 0.1 + mi * bar_w
            y0 = plot["bottom"] - min(v, y_max) / y_max * (plot["bottom"] - plot["top"])
            d.rectangle([x0, y0, x0 + bar_w - 2, plot["bottom"]], fill=METHOD_COLOR[method])
            d.text((x0 - 2, y0 - 14), f"{v:.1f}", fill="#111111", font=font(10))
        d.text((gx + group_w / 2 - 30, plot["bottom"] + 8), p, fill="#111111", font=font(13))

    d.text((plot["left"], 24), "Test-set MAE 2D per Pattern and Method (no data leakage)", fill="#111111", font=font(18))
    d.text((14, (plot["top"] + plot["bottom"]) // 2 - 20), "MAE", fill="#111111", font=font(13))
    d.text((14, (plot["top"] + plot["bottom"]) // 2 - 2), "(cm)", fill="#111111", font=font(13))
    # legend
    lx, ly = plot["left"] + 20, plot["bottom"] + 40
    for method in METHOD_ORDER:
        d.rectangle([lx, ly, lx + 18, ly + 12], fill=METHOD_COLOR[method])
        d.text((lx + 24, ly), METHOD_SHORT[method], fill="#111111", font=font(12))
        lx += 230
    img.save(dest)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for slug, info in PATTERNS.items():
        run = latest_run(info["base"])
        dest = OUT / slug
        dest.mkdir(parents=True, exist_ok=True)

        metrics = pd.read_csv(run / "06_evaluation" / "metrics.csv")
        metrics.to_csv(dest / "metrics.csv", index=False)
        test = metrics[(metrics["split"] == "test") & (metrics["trajectory"] == "all")]
        for _, r in test.iterrows():
            if r["model"] in METHOD_ORDER:
                summary_rows.append(
                    {"pattern": info["label"], "model": r["model"],
                     "mae_cm": r["mae_2d_m"] * 100, "rmse_cm": r["rmse_2d_m"] * 100}
                )

        test_safe = safe_name(info["test"])
        plot_map = {
            f"trajectory_{test_safe}.png": "trajectory_test.png",
            f"error_over_time_{test_safe}.png": "error_over_time_test.png",
            "full_trajectory_comparison.png": "full_trajectory.png",
            "test_error_cdf.png": "test_error_cdf.png",
            "test_method_comparison.png": "test_method_comparison.png",
            "lstm_residual_distribution.png": "lstm_residual_distribution.png",
        }
        for src, dst in plot_map.items():
            src_path = run / "07_plots" / src
            if src_path.exists():
                shutil.copy2(src_path, dest / dst)

        hist_csv = run / "05_lstm_residual" / "training_history.csv"
        if hist_csv.exists():
            plot_loss(hist_csv, dest / "training_loss.png", info["label"])
        print(f"[13] {info['label']}: figures -> {dest}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary_metrics_all_patterns.csv", index=False)
    patterns = [info["label"] for info in PATTERNS.values()]
    plot_summary(summary, patterns, OUT / "summary_mae_all_patterns.png")
    print(f"[13] Summary -> {OUT / 'summary_mae_all_patterns.png'}")
    print(summary.pivot(index="pattern", columns="model", values="mae_cm").to_string())


if __name__ == "__main__":
    main()
