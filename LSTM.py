"""
===========================================================================
  UWB Indoor Localization — KF Error Correction with LSTM
===========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks as keras_cb
from tensorflow.keras.optimizers import Nadam
import warnings, os

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# ===========================================================================
# KONFIGURASI
# ===========================================================================
CSV_PATH    = "data_with_velocity1.csv"
OUTPUT_DIR  = "output_lstm_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fitur yang digunakan
RAW_REQUIRED_COLS = ["x_kf", "y_kf", "x_true", "y_true", "d1", "d2", "d3"]
INPUT_COLS  = ["x_kf", "y_kf", "d1", "d2", "d3"]
TARGET_COLS = ["err_x", "err_y"]

# Hyperparameter Model
TIME_STEPS   = 30
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.2
DENSE_OUT    = 2
EPOCHS       = 200
BATCH_SIZE   = 32
LR           = 0.001

# ===========================================================================
# STEP 1 — LOAD DATA & HITUNG ERROR
# ===========================================================================
print("\n[INFO] STEP 1: Memuat Data dan Menghitung Residual Error")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=RAW_REQUIRED_COLS).reset_index(drop=True)

# Menghitung target error (Deviasi antara Ground Truth dan Kalman Filter)
df["err_x"] = df["x_true"] - df["x_kf"]
df["err_y"] = df["y_true"] - df["y_kf"]

X_data = df[INPUT_COLS].values.astype(np.float32)
y_data = df[TARGET_COLS].values.astype(np.float32)
pos_data = df[["x_true", "y_true"]].values.astype(np.float32)

# ===========================================================================
# STEP 2 — NORMALISASI SKALA GLOBAL
# ===========================================================================
print("[INFO] STEP 2: Normalisasi Skala...")
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

# Fit dan transform ke seluruh data agar model mengenali nilai ekstrem noise UWB
X_sc = scaler_X.fit_transform(X_data)
y_sc = scaler_y.fit_transform(y_data)

# ===========================================================================
# STEP 3 — SLIDING WINDOW & RANDOM SPLIT
# ===========================================================================
print("[INFO] STEP 3: Membuat Sliding Window Sequence...")
def create_sequences(X, y, time_steps):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i : i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

# Ekstrak sequence untuk semua lintasan robot
X_seq_all, y_seq_all = create_sequences(X_sc, y_sc, TIME_STEPS)

# Split sequence 
X_train, X_test, y_train, y_test = train_test_split(X_seq_all, y_seq_all, test_size=0.2, random_state=42)

# ===========================================================================
# STEP 4 — BANGUN MODEL LSTM
# ===========================================================================
print("[INFO] STEP 4: Membangun Arsitektur LSTM...")
model = models.Sequential([
    layers.Input(shape=(TIME_STEPS, len(INPUT_COLS))),
    layers.LSTM(LSTM_UNITS_1, return_sequences=True),
    layers.Dropout(DROPOUT_RATE),
    layers.LSTM(LSTM_UNITS_2, return_sequences=False),
    layers.Dropout(DROPOUT_RATE),
    layers.Dense(DENSE_OUT)
])

model.compile(optimizer=Nadam(learning_rate=LR), loss="mse", metrics=["mae"])

# ===========================================================================
# STEP 5 — TRAINING
# ===========================================================================
print("\n[INFO] STEP 5: Memulai Proses Training...")
callbacks_list = [
    keras_cb.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    keras_cb.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
]

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks_list,
    verbose=1
)

# ===========================================================================
# STEP 6 — REKONSTRUKSI TRAJEKTORI PENUH (FULL ROUTE)
# ===========================================================================
print("\n[INFO] STEP 6: Merekonstruksi Seluruh Trajektori...")

# Minta model memprediksi ERROR untuk seluruh lintasan
y_pred_err_sc_all = model.predict(X_seq_all, verbose=0)
y_pred_err_all = scaler_y.inverse_transform(y_pred_err_sc_all)

# Ambil data posisi asli KF (buang TIME_STEPS pertama karena termakan window)
x_kf_all = X_data[TIME_STEPS:, 0]
y_kf_all = X_data[TIME_STEPS:, 1]

# Hitung Posisi Akhir = Posisi KF + Prediksi Error LSTM
y_pred_pos = np.zeros_like(y_pred_err_all)
y_pred_pos[:, 0] = x_kf_all + y_pred_err_all[:, 0]
y_pred_pos[:, 1] = y_kf_all + y_pred_err_all[:, 1]

# Siapkan data Ground Truth dan KF absolut untuk evaluasi
y_true_pos = pos_data[TIME_STEPS:]
y_kf_pos = X_data[TIME_STEPS:, 0:2]

# ===========================================================================
# STEP 7 — EVALUASI METRIK KESELURUHAN JALUR
# ===========================================================================
print("\n" + "=" * 50)
print("  STEP 7: HASIL EVALUASI TRAJEKTORI PENUH")
print("=" * 50)

def compute_metrics(y_t, y_p, label=""):
    rmse_x  = np.sqrt(mean_squared_error(y_t[:, 0], y_p[:, 0]))
    rmse_y  = np.sqrt(mean_squared_error(y_t[:, 1], y_p[:, 1]))
    print(f"  [{label}] \n   -> RMSE X: {rmse_x:.4f} m | RMSE Y: {rmse_y:.4f} m\n")
    return rmse_x, rmse_y

rmse_kf_x, rmse_kf_y = compute_metrics(y_true_pos, y_kf_pos, "Baseline Kalman Filter")
rmse_lstm_x, rmse_lstm_y = compute_metrics(y_true_pos, y_pred_pos, "LSTM Corrector")

rmse_kf_tot = np.sqrt((rmse_kf_x**2 + rmse_kf_y**2) / 2)
rmse_lstm_tot = np.sqrt((rmse_lstm_x**2 + rmse_lstm_y**2) / 2)
improvement = (rmse_kf_tot - rmse_lstm_tot) / rmse_kf_tot * 100

print(f"[KESIMPULAN] Peningkatan Performa: {improvement:+.2f}%")

# ===========================================================================
# STEP 8 — VISUALISASI TRAJEKTORI
# ===========================================================================
print("\n[INFO] STEP 8: Menyimpan Grafik dan Data CSV...")

fig, ax = plt.subplots(figsize=(9, 8))
ax.set_facecolor("white")
ax.grid(True, linestyle="--", alpha=0.5)

# Plot Garis
ax.plot(y_true_pos[:, 0], y_true_pos[:, 1], color="#2c3e50", lw=3, label="Ground Truth Asli")
ax.plot(y_kf_pos[:, 0], y_kf_pos[:, 1], color="#e74c3c", lw=1.5, linestyle="--", label="Kalman Filter")
ax.plot(y_pred_pos[:, 0], y_pred_pos[:, 1], color="#27ae60", lw=2, label="LSTM Terkoreksi")

# FITUR AUTO-ZOOM: Mengunci tampilan pada ukuran rute asli + margin
margin = 0.5
ax.set_xlim(y_true_pos[:, 0].min() - margin, y_true_pos[:, 0].max() + margin)
ax.set_ylim(y_true_pos[:, 1].min() - margin, y_true_pos[:, 1].max() + margin)

ax.set_xlabel("Sumbu X (meter)", fontsize=11)
ax.set_ylabel("Sumbu Y (meter)", fontsize=11)
ax.set_title(f"Full Trajectory: Hasil Koreksi LSTM (Akurasi: {improvement:+.1f}%)", fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc="upper right")
plt.tight_layout()

# Simpan Grafik
plot_path = os.path.join(OUTPUT_DIR, "full_trajectory_autozoom.png")
plt.savefig(plot_path, dpi=150)
plt.show()

# ===========================================================================
# STEP 9 — EKSPOR HASIL KE CSV
# ===========================================================================
df_result = pd.DataFrame({
    "x_true": y_true_pos[:, 0], "y_true": y_true_pos[:, 1],
    "x_kf"  : y_kf_pos[:, 0],   "y_kf"  : y_kf_pos[:, 1],
    "x_pred": y_pred_pos[:, 0], "y_pred": y_pred_pos[:, 1],
})
csv_out = os.path.join(OUTPUT_DIR, "final_prediction_results.csv")
df_result.to_csv(csv_out, index=False, float_format="%.6f")

print(f"[SELESAI] Data tersimpan di folder: {OUTPUT_DIR}")