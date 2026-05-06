"""
UWB Indoor Localization - KF residual correction with LSTM, no data leakage.

Target model:
    residual = ground_truth_position - kalman_filter_position
    corrected_position = kalman_filter_position + calibrated_predicted_residual

This script avoids the common leakage points:
    1. Train/validation/test are split before sliding windows are created.
    2. Scalers are fit on training rows only.
    3. Validation is used for early stopping; test is evaluated only at the end.
    4. Windows are created per trajectory segment, so they never cross file/split
       boundaries.
    5. Residual calibration is fit on validation predictions only, then applied to
       train/validation/test. Test labels are never used to tune the correction.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks as keras_cb
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


np.random.seed(42)
tf.random.set_seed(42)


# ===========================================================================
# CONFIG
# ===========================================================================

CSV_PATHS = [
    Path("Data hasil/data_with_velocity1 (majulurus).csv"),
    Path("Data hasil/kotak 2 loop.csv"),
    Path("Data hasil/kotak 3 loop.csv"),
    Path("Data hasil/diam.csv"),
]

OUTPUT_DIR = Path("output_lstm_no_leakage")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "best_lstm_residual_corrector.keras"

RAW_REQUIRED_COLS = ["x_kf", "y_kf", "x_true", "y_true", "d1", "d2", "d3"]
TARGET_COLS = ["err_x", "err_y"]

# Same anchor positions used in Kalman Filter.py.
ANCHORS = {
    "1": (5.92, 5.02),
    "2": (0.0, 0.0),
    "3": (10.58, 0.0),
}

FEATURE_COLS = [
    "x",
    "y",
    "x_kf",
    "y_kf",
    "d1",
    "d2",
    "d3",
    "el1",
    "el2",
    "el3",
    "raw_minus_kf_x",
    "raw_minus_kf_y",
    "raw_vs_kf_dist",
    "d1_innov",
    "d2_innov",
    "d3_innov",
    "el1_innov",
    "el2_innov",
    "el3_innov",
    "d12_diff",
    "d23_diff",
    "d13_diff",
    "kf_dx",
    "kf_dy",
    "raw_dx",
    "raw_dy",
    "kf_speed",
    "raw_speed",
    "dt",
]

# Split options:
#   "chronological_by_file": each file is split 70/15/15 in time order.
#   "by_track": selected files are held out as test trajectories.
SPLIT_MODE = "chronological_by_file"
TEST_TRACK_KEYWORDS = ["kotak 3 loop"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

TIME_STEPS = 30
EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
CALIBRATION_METRIC = "rmse_euclidean"

# Keep this False when you only want to re-evaluate the latest saved model.
# Set to True to train from scratch again.
RETRAIN_MODEL = False


# ===========================================================================
# DATA PREPARATION
# ===========================================================================

def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["err_x"] = df["x_true"] - df["x_kf"]
    df["err_y"] = df["y_true"] - df["y_kf"]

    df["raw_minus_kf_x"] = df["x"] - df["x_kf"]
    df["raw_minus_kf_y"] = df["y"] - df["y_kf"]
    df["raw_vs_kf_dist"] = np.hypot(df["raw_minus_kf_x"], df["raw_minus_kf_y"])

    for idx, (anchor_x, anchor_y) in ANCHORS.items():
        dist_from_kf = np.hypot(df["x_kf"] - anchor_x, df["y_kf"] - anchor_y)
        df[f"d{idx}_innov"] = df[f"d{idx}"] - dist_from_kf
        df[f"el{idx}_innov"] = df[f"el{idx}"] - dist_from_kf

    df["d12_diff"] = df["d1"] - df["d2"]
    df["d23_diff"] = df["d2"] - df["d3"]
    df["d13_diff"] = df["d1"] - df["d3"]

    if "time" in df.columns:
        df["dt"] = df["time"].diff().fillna(0.0).clip(lower=0.0)
    else:
        df["dt"] = 0.0

    df["kf_dx"] = df["x_kf"].diff().fillna(0.0)
    df["kf_dy"] = df["y_kf"].diff().fillna(0.0)
    df["raw_dx"] = df["x"].diff().fillna(0.0)
    df["raw_dy"] = df["y"].diff().fillna(0.0)

    safe_dt = df["dt"].replace(0.0, np.nan)
    df["kf_speed"] = np.hypot(df["kf_dx"], df["kf_dy"]).div(safe_dt).fillna(0.0)
    df["raw_speed"] = np.hypot(df["raw_dx"], df["raw_dy"]).div(safe_dt).fillna(0.0)

    df[FEATURE_COLS + TARGET_COLS] = (
        df[FEATURE_COLS + TARGET_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
    )

    return df


def load_tracks(csv_paths: list[Path]) -> list[pd.DataFrame]:
    tracks = []
    for path in csv_paths:
        df = pd.read_csv(path)
        required_cols = sorted(set(RAW_REQUIRED_COLS + ["x", "y", "el1", "el2", "el3"]))
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")

        df = df.dropna(subset=required_cols).reset_index(drop=True)
        df = add_model_features(df)
        df["source"] = path.name
        tracks.append(df)

    return tracks


def split_chronological_by_file(
    tracks: list[pd.DataFrame],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    train_parts, val_parts, test_parts = [], [], []

    for df in tracks:
        n_rows = len(df)
        train_end = int(n_rows * train_ratio)
        val_end = int(n_rows * (train_ratio + val_ratio))

        train_parts.append(df.iloc[:train_end].reset_index(drop=True))
        val_parts.append(df.iloc[train_end:val_end].reset_index(drop=True))
        test_parts.append(df.iloc[val_end:].reset_index(drop=True))

    return train_parts, val_parts, test_parts


def split_by_track(
    tracks: list[pd.DataFrame],
    test_keywords: list[str],
    val_ratio: float,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    train_val_parts, test_parts = [], []

    for df in tracks:
        source = str(df["source"].iloc[0]).lower()
        is_test = any(keyword.lower() in source for keyword in test_keywords)
        if is_test:
            test_parts.append(df.reset_index(drop=True))
        else:
            train_val_parts.append(df.reset_index(drop=True))

    if not train_val_parts or not test_parts:
        raise ValueError("SPLIT_MODE='by_track' needs at least one train file and one test file.")

    train_parts, val_parts = [], []
    for df in train_val_parts:
        val_start = int(len(df) * (1 - val_ratio))
        train_parts.append(df.iloc[:val_start].reset_index(drop=True))
        val_parts.append(df.iloc[val_start:].reset_index(drop=True))

    return train_parts, val_parts, test_parts


def fit_scalers(train_parts: list[pd.DataFrame]) -> tuple[StandardScaler, StandardScaler]:
    train_df = pd.concat(train_parts, ignore_index=True)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(train_df[FEATURE_COLS].to_numpy(np.float32))
    scaler_y.fit(train_df[TARGET_COLS].to_numpy(np.float32))

    return scaler_x, scaler_y


def make_windows(
    parts: list[pd.DataFrame],
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    time_steps: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_seq, y_seq, meta_rows = [], [], []

    for df in parts:
        if len(df) < time_steps:
            continue

        x_raw = df[FEATURE_COLS].to_numpy(np.float32)
        y_raw = df[TARGET_COLS].to_numpy(np.float32)
        x_scaled = scaler_x.transform(x_raw)
        y_scaled = scaler_y.transform(y_raw)

        for target_idx in range(time_steps - 1, len(df)):
            start = target_idx - time_steps + 1
            x_seq.append(x_scaled[start : target_idx + 1])
            y_seq.append(y_scaled[target_idx])

            meta_rows.append(
                {
                    "source": df.at[target_idx, "source"],
                    "time": df.at[target_idx, "time"] if "time" in df.columns else target_idx,
                    "x_true": df.at[target_idx, "x_true"],
                    "y_true": df.at[target_idx, "y_true"],
                    "x_kf": df.at[target_idx, "x_kf"],
                    "y_kf": df.at[target_idx, "y_kf"],
                    "err_x": df.at[target_idx, "err_x"],
                    "err_y": df.at[target_idx, "err_y"],
                }
            )

    if not x_seq:
        raise ValueError("No windows created. Reduce TIME_STEPS or add more rows.")

    return np.array(x_seq), np.array(y_seq), pd.DataFrame(meta_rows)


# ===========================================================================
# MODEL AND EVALUATION
# ===========================================================================

def build_model() -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(TIME_STEPS, len(FEATURE_COLS))),
            layers.GaussianNoise(0.01),
            layers.Bidirectional(layers.LSTM(96, return_sequences=True)),
            layers.LayerNormalization(),
            layers.Dropout(0.25),
            layers.LSTM(48),
            layers.LayerNormalization(),
            layers.Dropout(0.25),
            layers.Dense(64, activation="swish"),
            layers.Dropout(0.15),
            layers.Dense(32, activation="swish"),
            layers.Dense(2),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=LR, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    return model


def rmse_average_axis(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def position_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    euclidean = np.linalg.norm(err, axis=1)

    return {
        "rmse_x": float(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))),
        "rmse_y": float(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))),
        "rmse_avg_axis": rmse_average_axis(y_true, y_pred),
        "rmse_euclidean": float(np.sqrt(np.mean(np.sum(err**2, axis=1)))),
        "mae_avg_axis": float(mean_absolute_error(y_true, y_pred)),
        "mae_euclidean": float(np.mean(euclidean)),
        "p95_error_m": float(np.percentile(euclidean, 95)),
    }


def predict_residuals(
    model: tf.keras.Model,
    x_seq: np.ndarray,
    scaler_y: StandardScaler,
) -> np.ndarray:
    pred_residual_scaled = model.predict(x_seq, verbose=0)
    return scaler_y.inverse_transform(pred_residual_scaled)


def residual_targets(meta: pd.DataFrame) -> np.ndarray:
    return meta[["err_x", "err_y"]].to_numpy(np.float64)


def make_calibrator(
    name: str,
    alpha_x: float,
    beta_x: float,
    alpha_y: float,
    beta_y: float,
) -> dict[str, float | str]:
    return {
        "name": name,
        "alpha_x": float(alpha_x),
        "beta_x": float(beta_x),
        "alpha_y": float(alpha_y),
        "beta_y": float(beta_y),
    }


def apply_calibrator(
    pred_residual: np.ndarray,
    calibrator: dict[str, float | str],
) -> np.ndarray:
    corrected = np.empty_like(pred_residual, dtype=np.float64)
    corrected[:, 0] = calibrator["alpha_x"] * pred_residual[:, 0] + calibrator["beta_x"]
    corrected[:, 1] = calibrator["alpha_y"] * pred_residual[:, 1] + calibrator["beta_y"]
    return corrected


def fit_scalar_calibrator(
    pred_residual: np.ndarray,
    true_residual: np.ndarray,
) -> dict[str, float | str]:
    numerator = float(np.sum(pred_residual * true_residual))
    denominator = float(np.sum(pred_residual * pred_residual))
    alpha = numerator / denominator if denominator > 0 else 0.0
    return make_calibrator("Kalman + LSTM scalar calibrated", alpha, 0.0, alpha, 0.0)


def fit_per_axis_calibrator(
    pred_residual: np.ndarray,
    true_residual: np.ndarray,
) -> dict[str, float | str]:
    alphas = []
    for axis_idx in range(2):
        pred_axis = pred_residual[:, axis_idx]
        true_axis = true_residual[:, axis_idx]
        denominator = float(np.sum(pred_axis * pred_axis))
        alpha = float(np.sum(pred_axis * true_axis) / denominator) if denominator > 0 else 0.0
        alphas.append(alpha)

    return make_calibrator("Kalman + LSTM per-axis calibrated", alphas[0], 0.0, alphas[1], 0.0)


def fit_affine_calibrator(
    pred_residual: np.ndarray,
    true_residual: np.ndarray,
) -> dict[str, float | str]:
    params = []
    for axis_idx in range(2):
        pred_axis = pred_residual[:, axis_idx]
        true_axis = true_residual[:, axis_idx]
        design = np.column_stack([pred_axis, np.ones_like(pred_axis)])
        alpha, beta = np.linalg.lstsq(design, true_axis, rcond=None)[0]
        params.append((float(alpha), float(beta)))

    return make_calibrator(
        "Kalman + LSTM affine calibrated",
        params[0][0],
        params[0][1],
        params[1][0],
        params[1][1],
    )


def corrected_positions(meta: pd.DataFrame, residual: np.ndarray) -> np.ndarray:
    kf_pos = meta[["x_kf", "y_kf"]].to_numpy(np.float64)
    return kf_pos + residual


def select_residual_calibrator(
    val_pred_residual: np.ndarray,
    val_meta: pd.DataFrame,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    y_true_pos = val_meta[["x_true", "y_true"]].to_numpy(np.float64)
    y_kf_pos = val_meta[["x_kf", "y_kf"]].to_numpy(np.float64)
    true_residual = residual_targets(val_meta)

    candidates = [
        make_calibrator("Validation-selected Kalman only", 0.0, 0.0, 0.0, 0.0),
        make_calibrator("Kalman + LSTM raw", 1.0, 0.0, 1.0, 0.0),
        fit_scalar_calibrator(val_pred_residual, true_residual),
        fit_per_axis_calibrator(val_pred_residual, true_residual),
        fit_affine_calibrator(val_pred_residual, true_residual),
    ]

    rows = []
    for calibrator in candidates:
        residual = apply_calibrator(val_pred_residual, calibrator)
        y_pred_pos = corrected_positions(val_meta, residual)
        row = {
            "candidate": calibrator["name"],
            "alpha_x": calibrator["alpha_x"],
            "beta_x": calibrator["beta_x"],
            "alpha_y": calibrator["alpha_y"],
            "beta_y": calibrator["beta_y"],
            **position_metrics(y_true_pos, y_pred_pos),
        }
        rows.append(row)

    selection = pd.DataFrame(rows)
    best_idx = selection[CALIBRATION_METRIC].idxmin()
    return candidates[int(best_idx)], selection


def evaluate_split(
    pred_residual: np.ndarray,
    meta: pd.DataFrame,
    split_name: str,
    selected_calibrator: dict[str, float | str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = meta.copy()
    result["pred_err_x"] = pred_residual[:, 0]
    result["pred_err_y"] = pred_residual[:, 1]

    y_true_pos = result[["x_true", "y_true"]].to_numpy()
    y_kf_pos = result[["x_kf", "y_kf"]].to_numpy()

    raw_residual = apply_calibrator(
        pred_residual,
        make_calibrator("Kalman + LSTM raw", 1.0, 0.0, 1.0, 0.0),
    )
    calibrated_residual = apply_calibrator(pred_residual, selected_calibrator)

    y_raw_pos = corrected_positions(result, raw_residual)
    y_calibrated_pos = corrected_positions(result, calibrated_residual)

    result["calibrated_err_x"] = calibrated_residual[:, 0]
    result["calibrated_err_y"] = calibrated_residual[:, 1]
    result["x_pred_raw"] = y_raw_pos[:, 0]
    result["y_pred_raw"] = y_raw_pos[:, 1]
    result["x_pred"] = y_calibrated_pos[:, 0]
    result["y_pred"] = y_calibrated_pos[:, 1]

    metrics = pd.DataFrame(
        [
            {
                "split": split_name,
                "model": "Kalman Filter",
                **position_metrics(y_true_pos, y_kf_pos),
            },
            {
                "split": split_name,
                "model": "Kalman + LSTM raw",
                **position_metrics(y_true_pos, y_raw_pos),
            },
            {
                "split": split_name,
                "model": selected_calibrator["name"],
                **position_metrics(y_true_pos, y_calibrated_pos),
            },
        ]
    )

    result["split"] = split_name
    return result, metrics


def add_improvement_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for split, part in metrics.groupby("split", sort=False):
        kf = part.loc[part["model"] == "Kalman Filter"].iloc[0]
        compared_models = part.loc[part["model"] != "Kalman Filter"]

        for _, candidate in compared_models.iterrows():
            row = {"split": split, "model": f"Improvement vs KF - {candidate['model']} (%)"}
            for col in [
                "rmse_x",
                "rmse_y",
                "rmse_avg_axis",
                "rmse_euclidean",
                "mae_avg_axis",
                "mae_euclidean",
                "p95_error_m",
            ]:
                row[col] = (kf[col] - candidate[col]) / kf[col] * 100 if kf[col] != 0 else np.nan
            rows.append(row)

    return pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True)


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["val_loss"], label="val loss")
    ax.set_title("Training history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_history.png", dpi=150)
    plt.close(fig)


def plot_test_trajectories(result: pd.DataFrame) -> None:
    for source, part in result.groupby("source", sort=False):
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(part["x_true"], part["y_true"], color="#263238", linewidth=2.5, label="Ground Truth")
        ax.plot(part["x_kf"], part["y_kf"], color="#d32f2f", linewidth=1.5, linestyle="--", label="Kalman Filter")
        ax.plot(
            part["x_pred_raw"],
            part["y_pred_raw"],
            color="#8e24aa",
            linewidth=1.2,
            linestyle=":",
            label="Kalman + LSTM raw",
        )
        ax.plot(
            part["x_pred"],
            part["y_pred"],
            color="#00796b",
            linewidth=2,
            label="Kalman + calibrated LSTM",
        )
        ax.set_title(f"Test trajectory - {source}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.axis("equal")
        ax.legend()
        fig.tight_layout()

        safe_name = source.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(OUTPUT_DIR / f"test_trajectory_{safe_name}.png", dpi=150)
        plt.close(fig)


def plot_full_trajectories(predictions: pd.DataFrame) -> None:
    for source, part in predictions.groupby("source", sort=False):
        part = part.sort_values(["split", "time"]).copy()

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(part["x_true"], part["y_true"], color="#263238", linewidth=2.5, label="Ground Truth")
        ax.plot(part["x_kf"], part["y_kf"], color="#d32f2f", linewidth=1.4, linestyle="--", label="Kalman Filter")
        ax.plot(part["x_pred"], part["y_pred"], color="#00796b", linewidth=1.8, label="Kalman + calibrated LSTM")

        test_part = part.loc[part["split"] == "test"]
        if not test_part.empty:
            ax.scatter(
                test_part["x_true"].iloc[0],
                test_part["y_true"].iloc[0],
                color="#1565c0",
                s=38,
                label="Test start",
                zorder=4,
            )

        ax.set_title(f"Full trajectory - {source}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.axis("equal")
        ax.legend()
        fig.tight_layout()

        safe_name = source.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(OUTPUT_DIR / f"full_trajectory_{safe_name}.png", dpi=150)
        plt.close(fig)


def plot_test_error_over_time(result: pd.DataFrame) -> None:
    for source, part in result.groupby("source", sort=False):
        part = part.sort_values("time").copy()
        kf_error = np.hypot(part["x_true"] - part["x_kf"], part["y_true"] - part["y_kf"])
        raw_error = np.hypot(part["x_true"] - part["x_pred_raw"], part["y_true"] - part["y_pred_raw"])
        calibrated_error = np.hypot(part["x_true"] - part["x_pred"], part["y_true"] - part["y_pred"])

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(part["time"], kf_error, color="#d32f2f", linewidth=1.5, label="Kalman Filter")
        ax.plot(part["time"], raw_error, color="#8e24aa", linewidth=1.1, linestyle=":", label="Kalman + LSTM raw")
        ax.plot(part["time"], calibrated_error, color="#00796b", linewidth=1.8, label="Kalman + calibrated LSTM")
        ax.set_title(f"Test error over time - {source}")
        ax.set_xlabel("Time")
        ax.set_ylabel("2D error (m)")
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend()
        fig.tight_layout()

        safe_name = source.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(OUTPUT_DIR / f"test_error_over_time_{safe_name}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    tracks = load_tracks(CSV_PATHS)

    if SPLIT_MODE == "chronological_by_file":
        train_parts, val_parts, test_parts = split_chronological_by_file(
            tracks, TRAIN_RATIO, VAL_RATIO
        )
    elif SPLIT_MODE == "by_track":
        train_parts, val_parts, test_parts = split_by_track(
            tracks, TEST_TRACK_KEYWORDS, VAL_RATIO
        )
    else:
        raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

    scaler_x, scaler_y = fit_scalers(train_parts)

    x_train, y_train, train_meta = make_windows(train_parts, scaler_x, scaler_y, TIME_STEPS)
    x_val, y_val, val_meta = make_windows(val_parts, scaler_x, scaler_y, TIME_STEPS)
    x_test, y_test, test_meta = make_windows(test_parts, scaler_x, scaler_y, TIME_STEPS)

    print("Window count:")
    print(f"  train: {len(x_train)}")
    print(f"  val  : {len(x_val)}")
    print(f"  test : {len(x_test)}")
    print(f"Feature count: {len(FEATURE_COLS)}")

    history = None
    if MODEL_PATH.exists() and not RETRAIN_MODEL:
        print(f"Loading existing model from: {MODEL_PATH}")
        model = models.load_model(str(MODEL_PATH), compile=False)
        model.summary()
    else:
        model = build_model()
        model.summary()

        callbacks_list = [
            keras_cb.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
                verbose=1,
            ),
            keras_cb.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=7,
                min_lr=1e-5,
                verbose=1,
            ),
            keras_cb.ModelCheckpoint(
                str(MODEL_PATH),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            keras_cb.TerminateOnNaN(),
        ]

        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_val, y_val),
            callbacks=callbacks_list,
            shuffle=False,
            verbose=1,
        )

    train_pred_residual = predict_residuals(model, x_train, scaler_y)
    val_pred_residual = predict_residuals(model, x_val, scaler_y)
    test_pred_residual = predict_residuals(model, x_test, scaler_y)

    selected_calibrator, calibration_candidates = select_residual_calibrator(
        val_pred_residual,
        val_meta,
    )

    train_result, train_metrics = evaluate_split(
        train_pred_residual,
        train_meta,
        "train",
        selected_calibrator,
    )
    val_result, val_metrics = evaluate_split(
        val_pred_residual,
        val_meta,
        "val",
        selected_calibrator,
    )
    test_result, test_metrics = evaluate_split(
        test_pred_residual,
        test_meta,
        "test",
        selected_calibrator,
    )

    metrics = pd.concat([train_metrics, val_metrics, test_metrics], ignore_index=True)
    metrics_with_improvement = add_improvement_rows(metrics)
    predictions = pd.concat([train_result, val_result, test_result], ignore_index=True)

    metrics_with_improvement.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    calibration_candidates.to_csv(OUTPUT_DIR / "calibration_candidates.csv", index=False)
    model.save(str(OUTPUT_DIR / "last_lstm_residual_corrector.keras"))

    if history is not None:
        pd.DataFrame(history.history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "csv_paths": [str(path) for path in CSV_PATHS],
                "split_mode": SPLIT_MODE,
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "time_steps": TIME_STEPS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
                "retrain_model": RETRAIN_MODEL,
                "model_path": str(MODEL_PATH),
                "calibration_metric": CALIBRATION_METRIC,
                "selected_calibrator": selected_calibrator,
                "feature_cols": FEATURE_COLS,
                "target_cols": TARGET_COLS,
                "anchors": ANCHORS,
            },
            file,
            indent=2,
        )

    if history is not None:
        plot_training_history(history)

    plot_test_trajectories(test_result)
    plot_full_trajectories(predictions)
    plot_test_error_over_time(test_result)

    print("\nMetrics:")
    print(metrics_with_improvement.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nValidation residual calibration candidates:")
    print(calibration_candidates.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected calibrator: {selected_calibrator['name']}")
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
