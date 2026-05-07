from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_trajectory(part: pd.DataFrame, path: Path, title: str, config: dict) -> None:
    labels = _labels(config)
    fig, ax = plt.subplots(figsize=(8, 7))
    if "raw_x" in part.columns and "raw_y" in part.columns:
        ax.plot(part["raw_x"], part["raw_y"], color="#9e9e9e", linewidth=1.0, alpha=0.8, label=labels["raw"])
    ax.plot(part["gt_x"], part["gt_y"], color="#263238", linewidth=2.4, label=labels["ground_truth"])
    ax.plot(part["ekf_x"], part["ekf_y"], color="#d32f2f", linewidth=1.5, linestyle="--", label=labels["ekf"])
    if "lstm_x" in part.columns:
        ax.plot(part["lstm_x"], part["lstm_y"], color="#00796b", linewidth=1.8, label=labels["lstm"])
    if "constraint_x" in part.columns and part.get("constraint_enabled", pd.Series([False])).any():
        ax.plot(part["constraint_x"], part["constraint_y"], color="#1565c0", linewidth=1.8, label=labels["constraint"])
    ax.set_title(title)
    ax.set_xlabel(labels["x"])
    ax.set_ylabel(labels["y"])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_error_over_time(part: pd.DataFrame, path: Path, title: str, config: dict) -> None:
    labels = _labels(config)
    time = part["time"] if "time" in part.columns else np.arange(len(part))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    method_cols = [
        ("EKF", "ekf_x", "ekf_y", "#d32f2f"),
        ("EKF + LSTM", "lstm_x", "lstm_y", "#00796b"),
        ("EKF + LSTM + Constraint", "constraint_x", "constraint_y", "#1565c0"),
    ]
    for label, x_col, y_col, color in method_cols:
        if x_col not in part.columns:
            continue
        err = np.hypot(part[x_col] - part["gt_x"], part[y_col] - part["gt_y"])
        ax.plot(time, err, label=label, color=color, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(labels["time"])
    ax.set_ylabel(labels["error"])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_error_cdf(df: pd.DataFrame, path: Path, config: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, x_col, y_col, color in [
        ("EKF", "ekf_x", "ekf_y", "#d32f2f"),
        ("EKF + LSTM", "lstm_x", "lstm_y", "#00796b"),
        ("EKF + LSTM + Constraint", "constraint_x", "constraint_y", "#1565c0"),
    ]:
        if x_col not in df.columns:
            continue
        err = np.sort(np.hypot(df[x_col] - df["gt_x"], df[y_col] - df["gt_y"]))
        cdf = np.linspace(0, 1, len(err))
        ax.plot(err, cdf, label=label, color=color, linewidth=1.8)
    ax.set_title("2D Error CDF")
    ax.set_xlabel("2D error (m)")
    ax.set_ylabel("CDF")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame, path: Path) -> None:
    test = metrics[(metrics["split"] == "test") & (metrics["trajectory"] == "all")]
    if test.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(test))
    bars = ax.bar(x, test["rmse_2d_m"], color=["#d32f2f", "#00796b", "#1565c0"][: len(test)])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(test["model"], rotation=15, ha="right")
    ax.set_ylabel("2D RMSE (m)")
    ax.set_title("Method Comparison on Test Set")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_residual_distribution(df: pd.DataFrame, path: Path) -> None:
    if "lstm_residual_x" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["lstm_residual_x"], bins=50, color="#00796b", alpha=0.85)
    axes[0].set_title("LSTM residual X")
    axes[0].set_xlabel("Residual (m)")
    axes[1].hist(df["lstm_residual_y"], bins=50, color="#1565c0", alpha=0.85)
    axes[1].set_title("LSTM residual Y")
    axes[1].set_xlabel("Residual (m)")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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

