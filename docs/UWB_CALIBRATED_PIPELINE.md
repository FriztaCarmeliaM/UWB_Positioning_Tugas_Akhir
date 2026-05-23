# Calibrated UWB Localization Pipeline

Dokumen ini menjelaskan pipeline baru untuk lokalisasi indoor UWB yang dibuat
agar eksperimen lebih reproducible, modular, dan bebas data leakage. Pipeline
ini tidak menghapus script lama; semua implementasi baru berada di `src/`,
`scripts/`, dan `configs/`.

Dokumen ini mengikuti konfigurasi hasil terbaru:

```text
configs/uwb_pipeline_10loop_moretrain.yaml
outputs/uwb_10loop_moretrain_pipeline/20260518_213455/
docs/results/20260518_213455/
```

Hasil utama pada test held-out `10lup2_trilat_gt` adalah RMSE 2D 11.25 cm dan
MAE 2D 9.50 cm untuk metode EKF + LSTM residual. Artinya, target alternatif di
bawah 10 cm sudah terpenuhi pada metrik MAE 2D, sedangkan RMSE 2D masih perlu
dilaporkan apa adanya.

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

Pada hasil 10-loop terbaru, target 5 cm belum tercapai. Jika target alternatif
yang diminta adalah di bawah 10 cm, klaim yang dapat dipertanggungjawabkan
adalah **MAE 2D = 9.50 cm** pada test held-out. RMSE 2D masih **11.25 cm** karena
RMSE lebih sensitif terhadap beberapa spike dan loop dengan error besar.
Keduanya perlu ditulis eksplisit agar klaim hasil tidak rancu.

Track record penurunan error pada test set:

| Tahap | RMSE 2D | MAE 2D | Penjelasan |
| --- | ---: | ---: | --- |
| Raw trilateration | 24.95 cm | 21.93 cm | Posisi langsung dari trilaterasi range datar |
| EKF only | 17.14 cm | 15.20 cm | Range sudah dikalibrasi dan posisi distabilkan model gerak |
| EKF + LSTM residual | 11.25 cm | **9.50 cm** | LSTM mengoreksi residual yang tersisa dari output EKF |

Gambar bukti ringkas tersedia di:

```text
docs/results/20260518_213455/target_10cm_evidence.png
docs/results/20260518_213455/segment_error_breakdown.png
```

## 3. Struktur Implementasi Baru

```text
configs/
  uwb_pipeline_latest.yaml
  uwb_pipeline_10loop_final.yaml
  uwb_pipeline_10loop_moretrain.yaml

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
  00_make_waypoint_ground_truth.py
  01_prepare_dataset.py
  02_calibrate_ranges.py
  03_optimize_anchors.py
  04_tune_ekf.py
  04_run_ekf.py
  05_train_lstm_residual.py
  06_evaluate_pipeline.py
  07_plot_results.py
```

## 4. Tahapan Pipeline

### 4.0 Waypoint Ground Truth

Script:

```bash
python scripts/00_make_waypoint_ground_truth.py --manual-times configs/latest_waypoint_times.yaml
```

Fungsi:

- Membaca catatan waktu waypoint dari `configs/latest_waypoint_times.yaml`.
- Membentuk ground truth `gt_x` dan `gt_y` dari waypoint fisik lintasan.
- Untuk data 10-loop, timestamp waypoint digunakan per titik lintasan sehingga
  interpolasi ground truth dilakukan per segmen.
- Untuk data 5-loop, catatan waktunya hanya awal dan akhir gerak sehingga hasil
  ground truth kurang detail dibanding 10-loop.

Output:

```text
Data eksperimen/latest_waypoint_ground_truth/
  10lup_trilat_gt.csv
  10lup1_trilat_gt.csv
  10lup2_trilat_gt.csv
  trilat5lup_gt.csv
  trilat5lup1_gt.csv
```

### 4.1 Prepare Dataset

Script:

```bash
python scripts/01_prepare_dataset.py --config configs/uwb_pipeline_10loop_moretrain.yaml
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
outputs/uwb_10loop_moretrain_pipeline/<timestamp>/01_prepared/
  all_samples.csv
  split_manifest.csv
  schema.json
```

### 4.2 Range Calibration

Script:

```bash
python scripts/02_calibrate_ranges.py --config configs/uwb_pipeline_10loop_moretrain.yaml
```

Fungsi:

- Fitting model linear per anchor:

```text
d_corrected = a * d_raw + b
```

- Ground truth distance dihitung dari posisi tag ground truth dan posisi anchor.
- Parameter hanya di-fit pada train split.
- Parameter kemudian diterapkan ke train, validation, test, dan full dataset.
- Range utama yang digunakan adalah `el1`, `el2`, dan `el3`, yaitu jarak datar
  hasil koreksi dari pembacaan UWB miring `d1`, `d2`, dan `d3`.

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
python scripts/03_optimize_anchors.py --config configs/uwb_pipeline_10loop_moretrain.yaml
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
python scripts/04_tune_ekf.py --config configs/uwb_pipeline_10loop_moretrain.yaml
python scripts/04_run_ekf.py --config configs/uwb_pipeline_10loop_moretrain.yaml
```

Fungsi:

- `04_tune_ekf.py` mencari kombinasi parameter EKF terbaik menggunakan
  validation split.
- `04_run_ekf.py` menjalankan EKF final memakai parameter hasil tuning.
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
python scripts/05_train_lstm_residual.py --config configs/uwb_pipeline_10loop_moretrain.yaml
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
python scripts/06_evaluate_pipeline.py --config configs/uwb_pipeline_10loop_moretrain.yaml
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
python scripts/07_plot_results.py --config configs/uwb_pipeline_10loop_moretrain.yaml
```

Plot yang dihasilkan:

- Full trajectory comparison.
- Per-track trajectory comparison.
- Error over time.
- CDF of 2D error.
- Bar chart method comparison.
- Bukti target alternatif 10 cm.
- Breakdown error per segmen test.
- Residual distribution.

Output:

```text
07_plots/
  full_trajectory_comparison.png
  trajectory_<track>.png
  error_over_time_<track>.png
  test_error_cdf.png
  test_method_comparison.png
  target_10cm_evidence.png
  segment_error_breakdown.png
  lstm_residual_distribution.png
```

## 5. Cara Run Manual dari Awal

Aktifkan environment:

```bash
conda activate uwb-ta
```

Jika environment belum lengkap, install dependency utama:

```bash
conda install -n uwb-ta -c conda-forge numpy pandas scipy scikit-learn matplotlib joblib pyyaml tensorflow=2.19.1 pyserial pillow -y
```

Alternatifnya, buat ulang environment dari file:

```bash
conda env update -n uwb-ta -f environment.yml
```

Jalankan stage satu per satu:

```bash
python scripts/00_make_waypoint_ground_truth.py --manual-times configs/latest_waypoint_times.yaml
python scripts/01_prepare_dataset.py --config configs/uwb_pipeline_10loop_moretrain.yaml
python scripts/02_calibrate_ranges.py --config configs/uwb_pipeline_10loop_moretrain.yaml
python scripts/03_optimize_anchors.py --config configs/uwb_pipeline_10loop_moretrain.yaml
python scripts/04_tune_ekf.py --config configs/uwb_pipeline_10loop_moretrain.yaml
python scripts/04_run_ekf.py --config configs/uwb_pipeline_10loop_moretrain.yaml
python scripts/05_train_lstm_residual.py --config configs/uwb_pipeline_10loop_moretrain.yaml
python scripts/06_evaluate_pipeline.py --config configs/uwb_pipeline_10loop_moretrain.yaml
python scripts/07_plot_results.py --config configs/uwb_pipeline_10loop_moretrain.yaml
```

Stage pertama membuat folder timestamp baru di:

```text
outputs/uwb_10loop_moretrain_pipeline/<timestamp>/
```

Config terbaru menggunakan `10lup2_trilat_gt` sebagai test held-out. Validation
dibuat dari sebagian akhir data train sehingga tuning tetap tidak melihat test
set. Anchor optimization pada config ini aktif, tetapi dibatasi oleh
`max_anchor_move_m` agar koreksi posisi anchor tidak berubah terlalu jauh dari
pengukuran manual.

Stage berikutnya otomatis membaca run terbaru melalui:

```text
outputs/uwb_10loop_moretrain_pipeline/latest_run.txt
```

Jika ingin memakai run tertentu:

```bash
python scripts/04_run_ekf.py --config configs/uwb_pipeline_10loop_moretrain.yaml --run-dir outputs/uwb_10loop_moretrain_pipeline/<timestamp>
```

## 6. Cara Melaporkan di TA

Narasi yang aman:

> Kalman Filter digunakan sebagai estimator utama berbasis range UWB. Sebelum
> EKF, range dari tiap anchor dikalibrasi menggunakan train split. Anchor
> position dan bias kemudian dapat dioptimasi menggunakan nonlinear least
> squares pada train split. LSTM digunakan sebagai residual corrector terhadap
> output EKF, bukan sebagai estimator posisi absolut. Seluruh scaler, kalibrasi,
> optimasi anchor, dan training LSTM dilakukan tanpa menggunakan test set. Pada
> test set terpisah, EKF + LSTM residual menghasilkan RMSE 2D 11.25 cm dan MAE
> 2D 9.50 cm. Dengan demikian, target alternatif di bawah 10 cm tercapai pada
> metrik MAE 2D, sedangkan RMSE 2D masih di atas 10 cm karena dipengaruhi
> spike/lonjakan.

Narasi yang harus dihindari:

> LSTM meningkatkan akurasi sampai sangat tinggi berdasarkan random split window.

Alasannya: random split window pada time-series dapat membuat train dan test
terlalu mirip, sehingga hasil tidak mencerminkan generalisasi lintasan.

## 7. Catatan Constraint Trajectory

Trajectory constraint adalah post-processing yang hanya boleh digunakan jika rute
eksperimen memang diketahui, misalnya lintasan rectangle dengan batas `x_min`,
`x_max`, `y_min`, dan `y_max`. Hasil constraint harus dilaporkan terpisah dari
hasil unconstrained.

Di config terbaru:

```yaml
constraint:
  enabled: false
```

Jika diaktifkan, thesis harus menulis dengan jelas bahwa hasil tersebut memakai
pengetahuan tambahan tentang bentuk lintasan.
