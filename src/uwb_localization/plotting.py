from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "ground_truth": "#263238",
    "raw": "#9e9e9e",
    "ekf": "#d32f2f",
    "lstm": "#00796b",
    "constraint": "#1565c0",
    "grid": "#dddddd",
    "axis": "#333333",
}


def _labels(config: dict) -> dict[str, str]:
    lang = config.get("plotting", {}).get("language", "en")
    if lang == "id":
        return {
            "ground_truth": "Ground Truth",
            "raw": "Raw UWB",
            "ekf": "EKF",
            "lstm": "EKF + LSTM",
            "constraint": "EKF + LSTM + Constraint",
            "x": "X (m)",
            "y": "Y (m)",
            "time": "Waktu",
            "error": "Error 2D (m)",
        }
    return {
        "ground_truth": "Ground Truth",
        "raw": "Raw UWB",
        "ekf": "EKF",
        "lstm": "EKF + LSTM",
        "constraint": "EKF + LSTM + Constraint",
        "x": "X (m)",
        "y": "Y (m)",
        "time": "Time",
        "error": "2D error (m)",
    }


def _safe_name(value: str) -> str:
    return value.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def _font(size: int = 14) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "calibri.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _finite_line(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = _numeric(df[x_col])
    y = _numeric(df[y_col])
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return x[idx], y[idx]


def _bounds(series: list[tuple[np.ndarray, np.ndarray]], equal_axis: bool) -> tuple[float, float, float, float]:
    xs = np.concatenate([x for x, _ in series if len(x)]) if any(len(x) for x, _ in series) else np.array([0.0, 1.0])
    ys = np.concatenate([y for _, y in series if len(y)]) if any(len(y) for _, y in series) else np.array([0.0, 1.0])
    x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
    y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
    if abs(x_max - x_min) < 1e-9:
        x_min -= 0.5
        x_max += 0.5
    if abs(y_max - y_min) < 1e-9:
        y_min -= 0.5
        y_max += 0.5

    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    if equal_axis:
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        half = max((x_max - x_min), (y_max - y_min)) / 2
        x_min, x_max = x_mid - half, x_mid + half
        y_min, y_max = y_mid - half, y_mid + half
    return x_min, x_max, y_min, y_max


def _canvas(width: int = 1100, height: int = 820) -> tuple[Image.Image, ImageDraw.ImageDraw, dict[str, int]]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot = {"left": 90, "top": 80, "right": width - 40, "bottom": height - 90}
    return image, draw, plot


def _map_point(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    plot: dict[str, int],
) -> tuple[int, int]:
    x_min, x_max, y_min, y_max = bounds
    px = plot["left"] + (x - x_min) / (x_max - x_min) * (plot["right"] - plot["left"])
    py = plot["bottom"] - (y - y_min) / (y_max - y_min) * (plot["bottom"] - plot["top"])
    return int(round(px)), int(round(py))


def _draw_axes(draw: ImageDraw.ImageDraw, plot: dict[str, int], title: str, x_label: str, y_label: str) -> None:
    title_font = _font(20)
    text_font = _font(14)
    draw.rectangle([plot["left"], plot["top"], plot["right"], plot["bottom"]], outline=COLORS["axis"], width=1)
    for i in range(1, 5):
        x = plot["left"] + i * (plot["right"] - plot["left"]) // 5
        y = plot["top"] + i * (plot["bottom"] - plot["top"]) // 5
        draw.line([(x, plot["top"]), (x, plot["bottom"])], fill=COLORS["grid"])
        draw.line([(plot["left"], y), (plot["right"], y)], fill=COLORS["grid"])
    draw.text((plot["left"], 28), title, fill="#111111", font=title_font)
    draw.text(((plot["left"] + plot["right"]) // 2 - 35, plot["bottom"] + 45), x_label, fill="#111111", font=text_font)
    draw.text((18, (plot["top"] + plot["bottom"]) // 2), y_label, fill="#111111", font=text_font)


def _draw_legend(draw: ImageDraw.ImageDraw, items: list[tuple[str, str]], plot: dict[str, int]) -> None:
    font = _font(13)
    x = plot["left"] + 10
    y = plot["top"] + 10
    for label, color in items:
        draw.line([(x, y + 8), (x + 28, y + 8)], fill=color, width=4)
        draw.text((x + 36, y), label, fill="#111111", font=font)
        y += 22


def _draw_line(
    draw: ImageDraw.ImageDraw,
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[float, float, float, float],
    plot: dict[str, int],
    color: str,
    width: int,
) -> None:
    x, y = _downsample(x, y)
    if len(x) < 2:
        return
    points = [_map_point(float(px), float(py), bounds, plot) for px, py in zip(x, y)]
    draw.line(points, fill=color, width=width, joint="curve")


def plot_trajectory(part: pd.DataFrame, path: Path, title: str, config: dict) -> None:
    labels = _labels(config)
    series: list[tuple[str, str, str, str, int]] = []
    if "raw_x" in part.columns and "raw_y" in part.columns:
        series.append((labels["raw"], "raw_x", "raw_y", COLORS["raw"], 2))
    series.append((labels["ground_truth"], "gt_x", "gt_y", COLORS["ground_truth"], 4))
    series.append((labels["ekf"], "ekf_x", "ekf_y", COLORS["ekf"], 3))
    if "lstm_x" in part.columns:
        series.append((labels["lstm"], "lstm_x", "lstm_y", COLORS["lstm"], 3))
    if "constraint_x" in part.columns and part.get("constraint_enabled", pd.Series([False])).any():
        series.append((labels["constraint"], "constraint_x", "constraint_y", COLORS["constraint"], 3))

    line_data = [(label, *_finite_line(part, x_col, y_col), color, width) for label, x_col, y_col, color, width in series]
    bounds = _bounds([(x, y) for _, x, y, _, _ in line_data], equal_axis=True)
    image, draw, plot = _canvas()
    _draw_axes(draw, plot, title, labels["x"], labels["y"])
    for _, x, y, color, width in line_data:
        _draw_line(draw, x, y, bounds, plot, color, width)
    _draw_legend(draw, [(label, color) for label, _, _, color, _ in line_data], plot)
    image.save(path)


def _error_series(part: pd.DataFrame, x_col: str, y_col: str) -> np.ndarray:
    return np.hypot(_numeric(part[x_col]) - _numeric(part["gt_x"]), _numeric(part[y_col]) - _numeric(part["gt_y"]))


def plot_error_over_time(part: pd.DataFrame, path: Path, title: str, config: dict) -> None:
    labels = _labels(config)
    time = _numeric(part["time"]) if "time" in part.columns else np.arange(len(part), dtype=float)
    methods = [
        ("EKF", "ekf_x", "ekf_y", COLORS["ekf"]),
        ("EKF + LSTM", "lstm_x", "lstm_y", COLORS["lstm"]),
        ("EKF + LSTM + Constraint", "constraint_x", "constraint_y", COLORS["constraint"]),
    ]
    line_data = []
    for label, x_col, y_col, color in methods:
        if x_col not in part.columns:
            continue
        err = _error_series(part, x_col, y_col)
        mask = np.isfinite(time) & np.isfinite(err)
        line_data.append((label, time[mask], err[mask], color, 3))

    bounds = _bounds([(x, y) for _, x, y, _, _ in line_data], equal_axis=False)
    image, draw, plot = _canvas(width=1100, height=620)
    _draw_axes(draw, plot, title, labels["time"], labels["error"])
    for _, x, y, color, width in line_data:
        _draw_line(draw, x, y, bounds, plot, color, width)
    _draw_legend(draw, [(label, color) for label, _, _, color, _ in line_data], plot)
    image.save(path)


def plot_error_cdf(df: pd.DataFrame, path: Path, config: dict) -> None:
    line_data = []
    for label, x_col, y_col, color in [
        ("EKF", "ekf_x", "ekf_y", COLORS["ekf"]),
        ("EKF + LSTM", "lstm_x", "lstm_y", COLORS["lstm"]),
        ("EKF + LSTM + Constraint", "constraint_x", "constraint_y", COLORS["constraint"]),
    ]:
        if x_col not in df.columns:
            continue
        err = np.sort(_error_series(df, x_col, y_col))
        err = err[np.isfinite(err)]
        if len(err) == 0:
            continue
        cdf = np.linspace(0, 1, len(err))
        line_data.append((label, err, cdf, color, 3))

    bounds = _bounds([(x, y) for _, x, y, _, _ in line_data], equal_axis=False)
    image, draw, plot = _canvas(width=900, height=650)
    _draw_axes(draw, plot, "2D Error CDF", "2D error (m)", "CDF")
    for _, x, y, color, width in line_data:
        _draw_line(draw, x, y, bounds, plot, color, width)
    _draw_legend(draw, [(label, color) for label, _, _, color, _ in line_data], plot)
    image.save(path)


def plot_metric_bars(metrics: pd.DataFrame, path: Path) -> None:
    test = metrics[(metrics["split"] == "test") & (metrics["trajectory"] == "all")]
    if test.empty:
        return

    image = Image.new("RGB", (950, 620), "white")
    draw = ImageDraw.Draw(image)
    plot = {"left": 90, "top": 80, "right": 900, "bottom": 500}
    _draw_axes(draw, plot, "Method Comparison on Test Set", "Model", "2D RMSE (m)")
    values = _numeric(test["rmse_2d_m"])
    max_value = max(float(np.nanmax(values)), 1e-6)
    bar_width = max(40, (plot["right"] - plot["left"]) // max(len(values) * 2, 1))
    font = _font(12)
    for i, (_, row) in enumerate(test.reset_index(drop=True).iterrows()):
        x_center = plot["left"] + (i + 0.5) * (plot["right"] - plot["left"]) / len(test)
        value = float(row["rmse_2d_m"])
        bar_top = plot["bottom"] - value / max_value * (plot["bottom"] - plot["top"] - 30)
        color = [COLORS["ekf"], COLORS["lstm"], COLORS["constraint"]][i % 3]
        draw.rectangle([x_center - bar_width / 2, bar_top, x_center + bar_width / 2, plot["bottom"]], fill=color)
        draw.text((x_center - 25, bar_top - 22), f"{value:.3f}", fill="#111111", font=font)
        draw.text((x_center - 65, plot["bottom"] + 18), str(row["model"])[:22], fill="#111111", font=font)
    image.save(path)


def plot_residual_distribution(df: pd.DataFrame, path: Path) -> None:
    if "lstm_residual_x" not in df.columns:
        return
    image = Image.new("RGB", (1000, 500), "white")
    draw = ImageDraw.Draw(image)
    font = _font(16)
    draw.text((40, 25), "LSTM Residual Distribution", fill="#111111", font=_font(20))
    for panel, column, color in [(0, "lstm_residual_x", COLORS["lstm"]), (1, "lstm_residual_y", COLORS["constraint"])]:
        values = _numeric(df[column])
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        hist, edges = np.histogram(values, bins=40)
        left = 60 + panel * 470
        right = left + 390
        top = 90
        bottom = 430
        draw.rectangle([left, top, right, bottom], outline=COLORS["axis"])
        max_count = max(int(hist.max()), 1)
        for i, count in enumerate(hist):
            x0 = left + i * (right - left) / len(hist)
            x1 = left + (i + 1) * (right - left) / len(hist)
            y0 = bottom - count / max_count * (bottom - top - 20)
            draw.rectangle([x0, y0, x1, bottom], fill=color)
        draw.text((left, 60), column, fill="#111111", font=font)
        draw.text((left, bottom + 14), f"{edges[0]:.2f} to {edges[-1]:.2f} m", fill="#111111", font=_font(12))
    image.save(path)


def generate_all_plots(predictions: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path, config: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectory(predictions, out_dir / "full_trajectory_comparison.png", "Full Trajectory Comparison", config)
    for trajectory, part in predictions.groupby("trajectory", sort=False):
        plot_trajectory(
            part,
            out_dir / f"trajectory_{_safe_name(trajectory)}.png",
            f"Trajectory Comparison - {trajectory}",
            config,
        )
        plot_error_over_time(
            part,
            out_dir / f"error_over_time_{_safe_name(trajectory)}.png",
            f"2D Error Over Time - {trajectory}",
            config,
        )
    plot_error_cdf(predictions[predictions["split"] == "test"], out_dir / "test_error_cdf.png", config)
    plot_metric_bars(metrics, out_dir / "test_method_comparison.png")
    plot_residual_distribution(predictions, out_dir / "lstm_residual_distribution.png")
