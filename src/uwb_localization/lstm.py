from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import anchor_ids, save_json
from .features import TARGET_COLS, build_feature_table, fit_scalers, make_windows


def _tf():
    try:
        import tensorflow as tf
        from tensorflow.keras import callbacks, layers, models
        from tensorflow.keras.optimizers import Adam
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for the LSTM residual stage. Install the environment "
            "from environment.yml or run inside the 'tensor' conda environment."
        ) from exc
    return tf, callbacks, layers, models, Adam


def build_model(input_shape: tuple[int, int], config: dict[str, Any]):
    tf, _, layers, models, Adam = _tf()
    lstm_cfg = config.get("lstm", {})
    hidden = int(lstm_cfg.get("hidden_size", 96))
    dropout = float(lstm_cfg.get("dropout", 0.25))
    lr = float(lstm_cfg.get("learning_rate", 1e-3))

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.GaussianNoise(float(lstm_cfg.get("gaussian_noise", 0.0))),
            layers.Bidirectional(layers.LSTM(hidden, return_sequences=True)),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.LSTM(max(hidden // 2, 16)),
            layers.LayerNormalization(),
            layers.Dropout(dropout),
            layers.Dense(max(hidden // 2, 32), activation="swish"),
            layers.Dropout(float(lstm_cfg.get("dense_dropout", 0.15))),
            layers.Dense(2),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=float(lstm_cfg.get("clipnorm", 1.0))),
        loss=tf.keras.losses.Huber(delta=float(lstm_cfg.get("huber_delta", 1.0))),
        metrics=["mae"],
    )
    return model


def train_lstm_residual(ekf_df: pd.DataFrame, config: dict[str, Any], out_dir: Path) -> None:
    tf, callbacks, _, models, _ = _tf()
    seed = int(config.get("random_seed", 42))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    ids = anchor_ids(config)
    table, feature_cols = build_feature_table(ekf_df, ids)

    train_table = table[table["split"] == "train"].copy()
    val_table = table[table["split"] == "val"].copy()
    if train_table.empty or val_table.empty:
        raise ValueError("LSTM training requires non-empty train and validation splits.")

    x_scaler, y_scaler = fit_scalers(train_table, feature_cols)
    lstm_cfg = config.get("lstm", {})
    sequence_length = int(lstm_cfg.get("sequence_length", 30))
    residual_clip = float(lstm_cfg.get("residual_clip_m", 0.5))

    x_train, y_train, _ = make_windows(
        train_table, feature_cols, x_scaler, y_scaler, sequence_length, residual_clip
    )
    x_val, y_val, _ = make_windows(
        val_table, feature_cols, x_scaler, y_scaler, sequence_length, residual_clip
    )

    model = build_model((sequence_length, len(feature_cols)), config)
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(lstm_cfg.get("patience", 20)),
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, int(lstm_cfg.get("patience", 20)) // 3),
            min_lr=float(lstm_cfg.get("min_learning_rate", 1e-5)),
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            str(out_dir / "best_lstm_residual.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=int(lstm_cfg.get("epochs", 200)),
        batch_size=int(lstm_cfg.get("batch_size", 64)),
        shuffle=False,
        callbacks=callbacks_list,
        verbose=1,
    )

    model.save(out_dir / "last_lstm_residual.keras")
    pd.DataFrame(history.history).to_csv(out_dir / "training_history.csv", index=False)
    joblib.dump(x_scaler, out_dir / "feature_scaler.joblib")
    joblib.dump(y_scaler, out_dir / "target_scaler.joblib")
    save_json(
        {
            "feature_cols": feature_cols,
            "target_cols": TARGET_COLS,
            "sequence_length": sequence_length,
            "residual_clip_m": residual_clip,
            "train_windows": int(len(x_train)),
            "val_windows": int(len(x_val)),
        },
        out_dir / "lstm_metadata.json",
    )


def apply_lstm_residual(ekf_df: pd.DataFrame, config: dict[str, Any], model_dir: Path) -> pd.DataFrame:
    _, _, _, models, _ = _tf()
    metadata_path = model_dir / "lstm_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing LSTM metadata: {metadata_path}")

    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_cols = metadata["feature_cols"]
    sequence_length = int(metadata["sequence_length"])
    residual_clip = float(config.get("lstm", {}).get("residual_clip_m", metadata.get("residual_clip_m", 0.5)))
    weight = float(config.get("lstm", {}).get("lstm_weight_lambda", 1.0))

    x_scaler = joblib.load(model_dir / "feature_scaler.joblib")
    y_scaler = joblib.load(model_dir / "target_scaler.joblib")
    model_path = model_dir / "best_lstm_residual.keras"
    if not model_path.exists():
        model_path = model_dir / "last_lstm_residual.keras"
    model = models.load_model(model_path, compile=False)

    table, _ = build_feature_table(ekf_df, anchor_ids(config))
    missing = [col for col in feature_cols if col not in table.columns]
    if missing:
        raise ValueError(f"Feature columns missing for LSTM inference: {missing}")

    x_seq, _, meta = make_windows(table, feature_cols, x_scaler, None, sequence_length)
    pred_scaled = model.predict(x_seq, verbose=0)
    pred_residual = y_scaler.inverse_transform(pred_scaled)
    residual_norm = np.linalg.norm(pred_residual, axis=1)
    scale = np.minimum(1.0, residual_clip / np.maximum(residual_norm, 1e-12))
    pred_residual = pred_residual * scale[:, None] * weight

    out = meta.copy()
    out["lstm_residual_x"] = pred_residual[:, 0]
    out["lstm_residual_y"] = pred_residual[:, 1]
    out["lstm_x"] = out["ekf_x"] + out["lstm_residual_x"]
    out["lstm_y"] = out["ekf_y"] + out["lstm_residual_y"]
    out["lstm_sequence_length"] = sequence_length
    return out
