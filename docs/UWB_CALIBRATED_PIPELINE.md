# Calibrated UWB Localization Pipeline

Dokumen ini menjelaskan pipeline baru untuk lokalisasi indoor UWB yang dibuat
agar eksperimen lebih reproducible, modular, dan bebas data leakage. Pipeline
ini tidak menghapus script lama; semua implementasi baru berada di `src/`,
`scripts/`, dan `configs/`.

## 1. Mengapa Pipeline Baru Dibuat

Script LSTM awal berpotensi menghasilkan performa terlalu tinggi karena beberapa
praktik yang tidak aman untuk time-series:

1. Scaler dapat ter-fit menggunakan seluruh data sebelum split.
2. Sliding window dapat dibuat sebelum split sehingga sample train dan test
   saling berbagi konteks waktu.
3. Random split pada sequence time-series dapat membuat data test terlalu mirip
   dengan train.
4. Test set dapat ikut digunakan sebagai validation/early stopping.

Karena itu, hasil LSTM lama tidak sebaiknya dijadikan klaim utama TA. Pipeline
baru ini memisahkan train, validation, dan test sebelum fitting calibration,
scaler, model LSTM, dan pemilihan hyperparameter.

## 2. Target Akurasi

Target error di bawah 5 cm hanya realistis dalam setup yang sangat terkontrol:
anchor terukur akurat, line-of-sight dominan, ground truth presisi, range sudah
dikalibrasi, dan lintasan uji sesuai distribusi training. Klaim di bawah 5 cm
harus dievaluasi menggunakan **2D Euclidean RMSE**:

```text
RMSE_2D = sqrt(mean((x_pred - x_true)^2 + (y_pred - y_true)^2))
```

Jangan hanya melaporkan RMSE X dan RMSE Y secara terpisah, karena error posisi
2D adalah gabungan kedua komponen tersebut.

Penjelasan singkat Bahasa Indonesia:

> Target error di bawah 5 cm hanya boleh diklaim jika diuji pada kondisi yang
> terkontrol dan metrik utamanya adalah RMSE Euclidean 2D. Jika hanya RMSE X
> atau RMSE Y yang kecil, belum tentu posisi 2D benar-benar akurat di bawah 5 cm.

## 3. Struktur Implementasi Baru

```text
configs/
  uwb_pipeline.yaml

src/uwb_localization/
  data.py
  calibration.py
  anchor_optimization.py
  ekf.py
  features.py
  lstm.py
  constraints.py
  metrics.py
  plotting.py
  artifacts.py
  config.py

scripts/
  01_prepare_dataset.py
  02_calibrate_ranges.py
  03_optimize_anchors.py
  04_run_ekf.py
  05_train_lstm_residual.py
  06_evaluate_pipeline.py
  07_plot_results.py
```

## 4. Tahapan Pipeline

### 4.1 Prepare Dataset

Script:

```bash
python scripts/01_prepare_dataset.py --config configs/uwb_pipeline.yaml
```

Fungsi:

- Membaca seluruh CSV trajectory.
- Melakukan standardisasi nama kolom.
- Memvalidasi kolom wajib.
- Menambahkan metadata `source_file`, `trajectory`, `segment_id`, dan
  `sample_index`.
- Membuat split train, validation, dan test sesuai config.

Output:

```text
outputs/uwb_calibrated_pipeline/<timestamp>/01_prepared/
  all_samples.csv
  split_manifest.csv
  schema.json
```

### 4.2 Range Calibration

Script:

```bash
python scripts/02_calibrate_ranges.py --config configs/uwb_pipeline.yaml
```

Fungsi:

- Fitting model linear per anchor:

```text
d_corrected = a * d_raw + b
```

- Ground truth distance dihitung dari posisi tag ground truth dan posisi anchor.
- Parameter hanya di-fit pada train split.
- Parameter kemudian diterapkan ke train, validation, test, dan full dataset.

Output:

```text
02_range_calibration/
  range_calibration.json
  all_samples_calibrated.csv
  train.csv
  val.csv
  test.csv
```

### 4.3 Anchor Optimization

Script:

```bash
python scripts/03_optimize_anchors.py --config configs/uwb_pipeline.yaml
```

Fungsi:

- Menggunakan `scipy.optimize.least_squares`.
- Mengestimasi koreksi kecil posisi anchor dan bias range.
- Menggunakan regularisasi dan bounds agar anchor tidak bergerak tidak realistis.
- Hanya menggunakan train split.

Output:

```text
03_anchor_optimization/
  optimized_anchors.json
```

### 4.4 EKF Range Localization

Script:

```bash
python scripts/04_run_ekf.py --config configs/uwb_pipeline.yaml
```

Fungsi:

- Menjalankan EKF langsung dari UWB ranges, bukan smoothing posisi x-y lama.
- State:

```text
[x, y, vx, vy]
```

- Motion model: constant velocity.
- Measurement model:

```text
h_i(x, y) = sqrt((x - ax_i)^2 + (y - ay_i)^2) + bias_i
```

- Mendukung Jacobian, process noise, measurement noise, innovation gating, dan
  outlier rejection.

Output:

```text
04_ekf/
  all_samples_ekf.csv
  train.csv
  val.csv
  test.csv
```

### 4.5 LSTM Residual Correction

Script:

```bash
python scripts/05_train_lstm_residual.py --config configs/uwb_pipeline.yaml
```

Fungsi:

- LSTM memprediksi residual terhadap EKF:

```text
residual_x = x_true - ekf_x
residual_y = y_true - ekf_y
```

- Input fitur mencakup calibrated ranges, EKF position, EKF velocity,
  innovation residual, covariance diagonal, range differences, delta time, dan
  motion features.
- Scaler hanya fit pada train split.
- Validation hanya untuk early stopping.
- Test tidak disentuh saat training.
- Residual output di-clip menggunakan `residual_clip_m` agar LSTM tidak
  overwrite EKF secara tidak realistis.

Output:

```text
05_lstm_residual/
  best_lstm_residual.keras
  last_lstm_residual.keras
  feature_scaler.joblib
  target_scaler.joblib
  training_history.csv
  lstm_metadata.json
```

### 4.6 Evaluation

Script:

```bash
python scripts/06_evaluate_pipeline.py --config configs/uwb_pipeline.yaml
```

Fungsi:

- Menghasilkan prediksi final.
- Mengevaluasi tiga varian:
  1. EKF only.
  2. EKF + LSTM residual.
  3. EKF + LSTM + trajectory constraint.
- Jika constraint disabled, kolom constraint tetap diberi flag
  `constraint_enabled = false` agar tidak disalahartikan sebagai metode aktif.

Metrik:

- RMSE X.
- RMSE Y.
- 2D Euclidean RMSE.
- MAE X.
- MAE Y.
- MAE 2D.
- Median error.
- P90 error.
- P95 error.
- Max error.
- Persentase sample di bawah 5 cm, 10 cm, 20 cm, dan 50 cm.
- Metrik keseluruhan dan per-track.

Output:

```text
06_evaluation/
  predictions.csv
  metrics.csv
  metrics.json
```

### 4.7 Plotting

Script:

```bash
python scripts/07_plot_results.py --config configs/uwb_pipeline.yaml
```

Plot yang dihasilkan:

- Full trajectory comparison.
- Per-track trajectory comparison.
- Error over time.
- CDF of 2D error.
- Bar chart method comparison.
- Residual distribution.

Output:

```text
07_plots/
  full_trajectory_comparison.png
  trajectory_<track>.png
  error_over_time_<track>.png
  test_error_cdf.png
  test_method_comparison.png
  lstm_residual_distribution.png
```

## 5. Cara Run Manual dari Awal

Aktifkan environment:

```bash
conda activate tensor
```

Jika environment `tensor` sudah ada dari eksperimen sebelumnya, pastikan
dependency pipeline baru tersedia:

```bash
conda install -n tensor -c conda-forge pyyaml scipy joblib -y
```

Alternatifnya, buat ulang environment dari file:

```bash
conda env update -n tensor -f environment.yml
```

Jalankan stage satu per satu:

```bash
python scripts/01_prepare_dataset.py --config configs/uwb_pipeline.yaml
python scripts/02_calibrate_ranges.py --config configs/uwb_pipeline.yaml
python scripts/03_optimize_anchors.py --config configs/uwb_pipeline.yaml
python scripts/04_run_ekf.py --config configs/uwb_pipeline.yaml
python scripts/05_train_lstm_residual.py --config configs/uwb_pipeline.yaml
python scripts/06_evaluate_pipeline.py --config configs/uwb_pipeline.yaml
python scripts/07_plot_results.py --config configs/uwb_pipeline.yaml
```

Stage pertama membuat folder timestamp baru di:

```text
outputs/uwb_calibrated_pipeline/<timestamp>/
```

Config default menggunakan test track `kotak 3 loop` sebagai lintasan yang
benar-benar held-out. Validation dibuat dari tail tiap train trajectory agar
validation tetap memiliki contoh gerak dinamis tanpa mengambil data test.

Anchor optimization default dibuat `enabled: false` karena pada dataset awal
optimasi anchor dapat bergerak sampai batas maksimum, yang merupakan indikasi
overfit terhadap train split. Jika ingin mengaktifkannya, turunkan batas gerak
anchor dan laporkan konfigurasi anchor teroptimasi secara eksplisit.

Stage berikutnya otomatis membaca run terbaru melalui:

```text
outputs/uwb_calibrated_pipeline/latest_run.txt
```

Jika ingin memakai run tertentu:

```bash
python scripts/04_run_ekf.py --config configs/uwb_pipeline.yaml --run-dir outputs/uwb_calibrated_pipeline/<timestamp>
```

## 6. Cara Melaporkan di TA

Narasi yang aman:

> Kalman Filter digunakan sebagai estimator utama berbasis range UWB. Sebelum
> EKF, range dari tiap anchor dikalibrasi menggunakan train split. Anchor
> position dan bias kemudian dapat dioptimasi menggunakan nonlinear least
> squares pada train split. LSTM digunakan sebagai residual corrector terhadap
> output EKF, bukan sebagai estimator posisi absolut. Seluruh scaler, kalibrasi,
> optimasi anchor, dan training LSTM dilakukan tanpa menggunakan test set.

Narasi yang harus dihindari:

> LSTM meningkatkan akurasi sampai sangat tinggi berdasarkan random split window.

Alasannya: random split window pada time-series dapat membuat train dan test
terlalu mirip, sehingga hasil tidak mencerminkan generalisasi lintasan.

## 7. Catatan Constraint Trajectory

Trajectory constraint adalah post-processing yang hanya boleh digunakan jika rute
eksperimen memang diketahui, misalnya lintasan rectangle dengan batas `x_min`,
`x_max`, `y_min`, dan `y_max`. Hasil constraint harus dilaporkan terpisah dari
hasil unconstrained.

Di config default:

```yaml
constraint:
  enabled: false
```

Jika diaktifkan, thesis harus menulis dengan jelas bahwa hasil tersebut memakai
pengetahuan tambahan tentang bentuk lintasan.
