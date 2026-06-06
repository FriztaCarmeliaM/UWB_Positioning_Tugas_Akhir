from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "20260602_dataset_baru"


def latest_run(base_dir: str) -> Path:
    latest = ROOT / "outputs" / base_dir / "latest_run.txt"
    if not latest.exists():
        raise FileNotFoundError(f"Missing latest run marker: {latest}")
    run_dir = Path(latest.read_text(encoding="utf-8").strip())
    if not run_dir.exists():
        raise FileNotFoundError(f"Latest run directory does not exist: {run_dir}")
    return run_dir


RUNS = {
    "Pola L": latest_run("uwb_dataset_baru_l_pipeline"),
    "Segitiga": latest_run("uwb_dataset_baru_segitiga_pipeline"),
}


def font(name: str, size: int) -> ImageFont.ImageFont:
    for candidate in [name, "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def copy_artifacts() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = {
        "Pola L": {
            "trajectory_L3_gt.png": "trajectory_l_test.png",
            "error_over_time_L3_gt.png": "error_over_time_l_test.png",
            "test_error_cdf.png": "cdf_l_test.png",
            "test_method_comparison.png": "method_l_test.png",
            "full_trajectory_comparison.png": "full_trajectory_l.png",
        },
        "Segitiga": {
            "trajectory_segitiga3_gt.png": "trajectory_segitiga_test.png",
            "error_over_time_segitiga3_gt.png": "error_over_time_segitiga_test.png",
            "test_error_cdf.png": "cdf_segitiga_test.png",
            "test_method_comparison.png": "method_segitiga_test.png",
            "full_trajectory_comparison.png": "full_trajectory_segitiga.png",
        },
    }
    for label, run_dir in RUNS.items():
        shutil.copy2(run_dir / "06_evaluation" / "metrics.csv", OUT / f"metrics_{label.lower().replace(' ', '_')}.csv")
        for src_name, dst_name in mapping[label].items():
            shutil.copy2(run_dir / "07_plots" / src_name, OUT / dst_name)


def read_test_metrics() -> pd.DataFrame:
    rows = []
    for pattern, run_dir in RUNS.items():
        metrics = pd.read_csv(run_dir / "06_evaluation" / "metrics.csv")
        test = metrics[(metrics["split"] == "test") & (metrics["trajectory"] == "all")].copy()
        for _, row in test.iterrows():
            rows.append(
                {
                    "pattern": pattern,
                    "model": row["model"],
                    "rmse_cm": float(row["rmse_2d_m"]) * 100,
                    "mae_cm": float(row["mae_2d_m"]) * 100,
                    "below10": float(row["pct_below_10cm"]),
                    "below20": float(row["pct_below_20cm"]),
                }
            )
    return pd.DataFrame(rows)


def draw_summary(metrics: pd.DataFrame) -> None:
    image = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(image)
    title_font = font("segoeuib.ttf", 30)
    head_font = font("segoeuib.ttf", 20)
    text_font = font("segoeui.ttf", 16)
    small_font = font("segoeui.ttf", 14)
    bold_font = font("segoeuib.ttf", 16)
    blue, green, red = "#2563eb", "#059669", "#dc2626"
    grid, axis, text = "#d1d5db", "#94a3b8", "#111827"

    def mae(pattern: str, model: str) -> float:
        part = metrics[(metrics["pattern"] == pattern) & (metrics["model"] == model)]
        return float(part.iloc[0]["mae_cm"]) if not part.empty else float("nan")

    draw.text((60, 42), "Hasil Dataset Baru: Pola L dan Segitiga", fill=text, font=title_font)
    draw.text(
        (60, 82),
        "Ground truth dibuat dari target_x/target_y sebagai timestamp waypoint detail | test terpisah per pola",
        fill="#4b5563",
        font=text_font,
    )

    left, top, right, bottom = 110, 165, 1400, 620
    max_value = max(45.0, float(metrics["rmse_cm"].max()) * 1.1)
    for tick in range(0, int(max_value) + 1, 10):
        y = bottom - tick / max_value * (bottom - top)
        draw.line((left, y, right, y), fill=grid, width=1)
        draw.text((left - 58, y - 8), str(tick), fill="#475569", font=small_font)
    draw.line((left, top, left, bottom), fill=axis, width=2)
    draw.line((left, bottom, right, bottom), fill=axis, width=2)
    draw.text((28, 370), "Error 2D (cm)", fill="#475569", font=text_font)
    target_y = bottom - 10 / max_value * (bottom - top)
    draw.line((left, target_y, right, target_y), fill=red, width=2)
    draw.text((right - 110, target_y - 24), "Target 10 cm", fill=red, font=small_font)

    groups = [
        ("Pola L", ["Raw trilateration", "EKF + LSTM residual", "EKF + LSTM + trajectory constraint"]),
        ("Segitiga", ["EKF only", "EKF + LSTM residual", "EKF + LSTM + trajectory constraint"]),
    ]
    x = left + 70
    bar_w = 34
    colors = {"rmse": blue, "mae": green}
    for pattern, models in groups:
        draw.text((x, top - 38), pattern, fill=text, font=head_font)
        for model in models:
            part = metrics[(metrics["pattern"] == pattern) & (metrics["model"] == model)]
            if part.empty:
                continue
            row = part.iloc[0]
            for j, (key, color) in enumerate([("rmse_cm", colors["rmse"]), ("mae_cm", colors["mae"])]):
                value = float(row[key])
                bx = x + j * 42
                by = bottom - value / max_value * (bottom - top)
                draw.rounded_rectangle((bx, by, bx + bar_w, bottom), radius=5, fill=color)
                label = f"{value:.1f}"
                draw.text((bx - 2, by - 22), label, fill=text, font=small_font)
            model_label = (
                model.replace("EKF + LSTM + trajectory constraint", "Final")
                .replace("EKF + LSTM residual", "EKF+LSTM")
                .replace("Raw trilateration", "Raw")
            )
            draw.text((x - 10, bottom + 18), model_label, fill=text, font=small_font)
            x += 130
        x += 80

    legend_y = 675
    draw.rounded_rectangle((110, legend_y, 132, legend_y + 14), radius=4, fill=blue)
    draw.text((140, legend_y - 4), "RMSE 2D", fill="#475569", font=small_font)
    draw.rounded_rectangle((240, legend_y, 262, legend_y + 14), radius=4, fill=green)
    draw.text((270, legend_y - 4), "MAE 2D", fill="#475569", font=small_font)

    box = (60, 740, 1440, 845)
    draw.rounded_rectangle(box, radius=10, outline="#cbd5e1", fill="#f8fafc", width=2)
    draw.text(
        (86, 762),
        "Interpretasi: dataset baru menambah validasi pola L dan segitiga. Hasil final menurunkan MAE pada kedua pola.",
        fill=text,
        font=bold_font,
    )
    draw.text(
        (86, 792),
        (
            f"Pola L: MAE EKF+LSTM {mae('Pola L', 'EKF + LSTM residual'):.2f} cm -> "
            f"final {mae('Pola L', 'EKF + LSTM + trajectory constraint'):.2f} cm. "
            f"Segitiga: MAE EKF+LSTM {mae('Segitiga', 'EKF + LSTM residual'):.2f} cm -> "
            f"final {mae('Segitiga', 'EKF + LSTM + trajectory constraint'):.2f} cm."
        ),
        fill="#4b5563",
        font=text_font,
    )
    draw.text(
        (86, 820),
        "Target utama <10 cm tetap dicapai pada eksperimen kotak final; dataset baru dipakai untuk memperluas pembahasan lintasan.",
        fill="#4b5563",
        font=text_font,
    )
    image.save(OUT / "dataset_baru_summary.png")


def draw_loss(run_dir: Path, output_name: str, title: str) -> None:
    history = pd.read_csv(run_dir / "05_lstm_residual" / "training_history.csv")
    epochs = list(range(1, len(history) + 1))
    loss = history["loss"].astype(float).to_list()
    val = history["val_loss"].astype(float).to_list()
    image = Image.new("RGB", (1050, 720), "white")
    draw = ImageDraw.Draw(image)
    title_font = font("segoeuib.ttf", 26)
    text_font = font("segoeui.ttf", 15)
    small_font = font("segoeui.ttf", 13)
    blue, green = "#2563eb", "#059669"
    grid, axis, text = "#d1d5db", "#94a3b8", "#111827"
    draw.text((60, 36), title, fill=text, font=title_font)
    draw.text((60, 72), "Sumbu X = epoch, sumbu Y = loss", fill="#4b5563", font=text_font)

    left, top, right, bottom = 105, 125, 990, 570
    max_y = max(max(loss), max(val)) * 1.12
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        value = i * max_y / 5
        draw.line((left, y, right, y), fill=grid)
        draw.text((left - 70, y - 8), f"{value:.3f}", fill="#475569", font=small_font)
    for epoch in range(1, len(epochs) + 1, max(1, len(epochs) // 6)):
        x = left + (epoch - 1) / max(len(epochs) - 1, 1) * (right - left)
        draw.line((x, top, x, bottom), fill=grid)
        draw.text((x - 8, bottom + 10), str(epoch), fill="#475569", font=small_font)
    draw.line((left, top, left, bottom), fill=axis, width=2)
    draw.line((left, bottom, right, bottom), fill=axis, width=2)
    draw.text(((left + right) // 2 - 24, bottom + 48), "Epoch", fill="#475569", font=text_font)
    draw.text((30, (top + bottom) // 2), "Loss", fill="#475569", font=text_font)

    def point(epoch: int, value: float) -> tuple[float, float]:
        return (
            left + (epoch - 1) / max(len(epochs) - 1, 1) * (right - left),
            bottom - value / max_y * (bottom - top),
        )

    draw.line([point(e, v) for e, v in zip(epochs, loss)], fill=blue, width=3)
    draw.line([point(e, v) for e, v in zip(epochs, val)], fill=green, width=3)
    draw.rounded_rectangle((left + 10, top + 12, left + 32, top + 26), radius=4, fill=blue)
    draw.text((left + 42, top + 8), "Training loss", fill=text, font=small_font)
    draw.rounded_rectangle((left + 165, top + 12, left + 187, top + 26), radius=4, fill=green)
    draw.text((left + 197, top + 8), "Validation loss", fill=text, font=small_font)
    image.save(OUT / output_name)


def main() -> None:
    copy_artifacts()
    metrics = read_test_metrics()
    metrics.to_csv(OUT / "dataset_baru_metrics_summary.csv", index=False)
    draw_summary(metrics)
    draw_loss(RUNS["Pola L"], "loss_l.png", "Training dan Validation Loss - Pola L")
    draw_loss(RUNS["Segitiga"], "loss_segitiga.png", "Training dan Validation Loss - Segitiga")
    print(f"Wrote dataset baru result snapshot to {OUT}")


if __name__ == "__main__":
    main()
