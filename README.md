# UWB Indoor Localization: Calibrated EKF and LSTM Residual Correction

Repository ini berisi penelitian lokalisasi indoor berbasis Ultra-Wideband
(UWB) untuk tugas akhir. Implementasi terbaru menggunakan pipeline yang
reproducible dan **no-data-leakage** dengan tahapan pembentukan ground truth
berbasis waypoint, kalibrasi range, preprocessing range, optimasi anchor, EKF
berbasis pengukuran jarak langsung, LSTM residual correction, evaluasi, dan
plotting.

**Status hasil terbaru:** konfigurasi terbaik yang paling layak dijadikan hasil
utama adalah eksperimen 10-loop dengan train/test terpisah. Pada test set
`10lup2_trilat_gt`, metode **EKF + LSTM residual** menurunkan RMSE 2D dari raw
trilateration sebesar **24.95 cm** menjadi **11.25 cm**, serta menghasilkan
**MAE 2D sebesar 9.50 cm**. Dengan demikian, target alternatif di bawah 10 cm
sudah tercapai jika metrik yang digunakan adalah rata-rata error/MAE 2D,
sedangkan RMSE 2D masih 11.25 cm karena lebih sensitif terhadap beberapa spike
dan loop dengan error besar. Evaluasi dilakukan tanpa data leakage.

Cara membaca hasil utama:

1. **RMSE 2D** adalah metrik konservatif karena memberi penalti lebih besar
   pada error/lonjakan besar.
2. **MAE 2D** adalah rata-rata error posisi per sampel.
3. Klaim "di bawah 10 cm" pada hasil terbaru merujuk pada **MAE 2D = 9.50
   cm** di test set terpisah, bukan RMSE 2D.
4. Seluruh angka utama di README ini berasal dari snapshot
   `docs/results/20260518_213455/`, terutama file `metrics.csv` dan
   `segment_metrics.csv`.

---

## Daftar Isi

- [1. Abstrak](#1-abstrak)
  - [1.1 Kata Kunci](#11-kata-kunci)
- [2. Pendahuluan](#2-pendahuluan)
  - [2.1 Latar Belakang](#21-latar-belakang)
  - [2.2 Tujuan Penelitian](#22-tujuan-penelitian)
  - [2.3 Catatan Target Akurasi 5 cm](#23-catatan-target-akurasi-5-cm)
- [3. Dataset dan Struktur Repository](#3-dataset-dan-struktur-repository)
  - [3.1 Dataset](#31-dataset)
  - [3.2 Struktur Kode](#32-struktur-kode)
  - [3.3 Snapshot Hasil Terbaru](#33-snapshot-hasil-terbaru)
- [4. Metodologi](#4-metodologi)
  - [4.1 Data Loading dan Split No-Leakage](#41-data-loading-dan-split-no-leakage)
  - [4.2 Kalibrasi Range per Anchor](#42-kalibrasi-range-per-anchor)
  - [4.3 Anchor Optimization](#43-anchor-optimization)
  - [4.4 Extended Kalman Filter Berbasis Range](#44-extended-kalman-filter-berbasis-range)
  - [4.5 LSTM Residual Correction](#45-lstm-residual-correction)
  - [4.6 Evaluasi dan Visualisasi](#46-evaluasi-dan-visualisasi)
- [5. Hasil Eksperimen Terbaru](#5-hasil-eksperimen-terbaru)
  - [5.1 Split Eksperimen](#51-split-eksperimen)
  - [5.2 Hasil Kalibrasi Range](#52-hasil-kalibrasi-range)
  - [5.3 Hasil Kuantitatif Test Set](#53-hasil-kuantitatif-test-set)
  - [5.4 Hasil Per Lintasan](#54-hasil-per-lintasan)
  - [5.5 Visualisasi](#55-visualisasi)
- [6. Diskusi](#6-diskusi)
  - [6.1 Interpretasi Hasil](#61-interpretasi-hasil)
  - [6.2 Mengapa Target 5 cm Belum Realistis](#62-mengapa-target-5-cm-belum-realistis)
  - [6.3 Implikasi untuk Tugas Akhir](#63-implikasi-untuk-tugas-akhir)
- [7. Cara Menjalankan Pipeline](#7-cara-menjalankan-pipeline)
  - [7.1 Environment](#71-environment)
  - [7.2 Urutan Eksekusi](#72-urutan-eksekusi)
  - [7.3 Output Pipeline](#73-output-pipeline)
- [8. Kesimpulan](#8-kesimpulan)
- [9. Referensi File Penting](#9-referensi-file-penting)

---

## 1. Abstrak

Penelitian ini mengevaluasi sistem lokalisasi indoor berbasis UWB menggunakan
pendekatan berlapis, yaitu pembentukan ground truth berbasis waypoint, kalibrasi
range per anchor, preprocessing range untuk mengurangi lonjakan pembacaan,
Extended Kalman Filter (EKF) berbasis range, dan LSTM residual correction.
Pipeline terbaru dirancang untuk menghindari data leakage dengan memisahkan
train, validation, dan test sebelum proses fitting kalibrasi, optimasi anchor,
scaler, sequence window, tuning EKF, dan training model LSTM.

Hasil terbaru menunjukkan bahwa raw trilateration masih memiliki error test
sekitar 24.95 cm RMSE 2D. Setelah diproses menggunakan EKF dan LSTM residual
correction, error test turun menjadi 11.25 cm RMSE 2D dan 9.50 cm MAE 2D. Hasil
ini menunjukkan bahwa metode yang digunakan mampu memperbaiki estimasi posisi
secara signifikan. Target 5 cm belum tercapai secara valid pada dataset saat
ini, tetapi target alternatif di bawah 10 cm sudah tercapai pada metrik MAE 2D.

### 1.1 Kata Kunci

UWB indoor localization, waypoint ground truth, range calibration, Extended
Kalman Filter, LSTM residual correction, no data leakage, trajectory evaluation.

---

## 2. Pendahuluan

### 2.1 Latar Belakang

Ultra-Wideband (UWB) sering digunakan untuk lokalisasi indoor karena mampu
memberikan estimasi jarak antar perangkat dengan resolusi waktu yang tinggi.
Namun, pada praktiknya pengukuran UWB tetap rentan terhadap bias, multipath,
NLOS, perbedaan tinggi antara tag dan anchor, kesalahan posisi anchor, noise
temporal, serta lonjakan pembacaan pada kondisi tertentu. Oleh karena itu,
estimasi posisi langsung dari trilaterasi raw range sering belum cukup stabil
untuk menghasilkan posisi robot yang akurat.

Pada eksperimen terbaru, robot bergerak pada lintasan persegi dengan waypoint
fisik `(1,1)`, `(3,1)`, `(3,3)`, dan `(1,3)`. Ground truth tidak dibentuk dari
plot prediksi, tetapi dari informasi waypoint dan timestamp saat robot mulai
bergerak atau sampai pada waypoint. Pendekatan ini lebih dapat dipertanggungjawabkan
karena ground truth berasal dari titik lintasan yang diukur saat eksperimen.

### 2.2 Tujuan Penelitian

Tujuan penelitian ini adalah:

1. Membangun pipeline lokalisasi UWB yang reproducible dan no-data-leakage.
2. Membentuk ground truth lintasan berdasarkan waypoint dan catatan waktu
   eksperimen.
3. Mengevaluasi performa raw trilateration, EKF berbasis range, dan LSTM
   residual correction.
4. Melaporkan hasil menggunakan metrik posisi 2D, terutama
   **2D Euclidean RMSE**.
5. Menentukan apakah dataset saat ini sudah cukup untuk memenuhi target error
   5 cm secara valid.

### 2.3 Catatan Target Akurasi 5 cm

Target error di bawah 5 cm hanya valid jika diuji pada setup yang sangat
terkontrol: posisi anchor presisi, kondisi line-of-sight dominan, sampling
stabil, kalibrasi range kuat, dan ground truth akurat sampai level sentimeter.
Metrik utama harus menggunakan RMSE 2D:

```text
RMSE_2D = sqrt(mean((x_pred - x_true)^2 + (y_pred - y_true)^2))
```

Perbedaan metrik yang digunakan:

```text
MAE_2D  = rata-rata jarak error setiap sampel
RMSE_2D = rata-rata kuadrat error, lalu diakar
```

Jika masih ada beberapa titik yang melonjak jauh dari ground truth, nilai RMSE
akan naik lebih besar daripada MAE. Karena itu, MAE dapat berada di bawah 10 cm
walaupun RMSE masih sedikit di atas 10 cm.

Pada dataset terbaru, hasil terbaik test set adalah 11.25 cm RMSE 2D dan 9.50
cm MAE 2D. Dengan demikian, target 5 cm belum dapat diklaim secara valid. Untuk
target alternatif di bawah 10 cm, klaim yang paling aman adalah **MAE 2D sudah
di bawah 10 cm**, sedangkan **RMSE 2D belum di bawah 10 cm**. Perbedaan ini
terjadi karena RMSE memberi penalti lebih besar pada beberapa spike/lonjakan dan
loop yang errornya besar, sementara MAE merepresentasikan rata-rata error
absolut per sampel.

---

## 3. Dataset dan Struktur Repository

### 3.1 Dataset

Dataset raw terbaru berada di folder `Data hasil/`. Ground truth berbasis
waypoint dihasilkan ke folder `Data eksperimen/latest_waypoint_ground_truth/`.

| File raw | File ground truth | Deskripsi | Peran pada config terbaik |
| --- | --- | --- | --- |
| `10lup+trilat.csv` | `10lup_trilat_gt.csv` | Lintasan persegi 10 loop | Train |
| `10lup1+trilat.csv` | `10lup1_trilat_gt.csv` | Lintasan persegi 10 loop sesi lain | Train, dengan sebagian akhir menjadi validation |
| `10lup2+trilat.csv` | `10lup2_trilat_gt.csv` | Lintasan persegi 10 loop sesi test | Test held-out |
| `trilat5lup.csv` | `trilat5lup_gt.csv` | Lintasan persegi 5 loop | Eksperimen tambahan |
| `trilat5lup1.csv` | `trilat5lup1_gt.csv` | Lintasan persegi 5 loop sesi lain | Eksperimen tambahan |

Kolom utama pada dataset terbaru:

| Kelompok | Kolom |
| --- | --- |
| Waktu | `time` |
| Range UWB miring | `d1`, `d2`, `d3` |
| Range datar hasil koreksi tinggi | `el1`, `el2`, `el3` |
| Raw trilateration | `x`, `y` |
| Ground truth waypoint | `gt_x`, `gt_y`, `gt_segment`, `gt_loop_id` |

Koordinat anchor yang digunakan:

| Anchor | Koordinat |
| --- | --- |
| Anchor 1 | `(2.26, 4.60)` |
| Anchor 2 | `(0.00, 0.00)` |
| Anchor 3 | `(4.55, 0.00)` |

Pipeline EKF menggunakan `el1`, `el2`, dan `el3` sebagai input range karena EKF
2D memodelkan jarak datar antara tag dan anchor. Nilai `el` berasal dari
koreksi Pythagoras terhadap pembacaan UWB miring dengan selisih tinggi perangkat.

Alur pembacaan data yang digunakan:

1. Sensor UWB membaca jarak miring `d1`, `d2`, dan `d3`.
2. Karena tag dan anchor memiliki beda tinggi, jarak miring dikoreksi menjadi
   jarak datar `el1`, `el2`, dan `el3`.
3. Pipeline kalibrasi, EKF, dan LSTM menggunakan `el1`, `el2`, dan `el3`
   sebagai range utama.
4. Kolom `x` dan `y` hasil trilaterasi raw tetap disimpan sebagai pembanding
   baseline dan fitur tambahan, tetapi bukan ground truth.
5. Ground truth dibuat dari waypoint fisik dan catatan waktu eksperimen, bukan
   dari hasil prediksi model.

### 3.2 Struktur Kode

Implementasi pipeline baru:

```text
configs/
  uwb_pipeline_latest.yaml
  uwb_pipeline_10loop_final.yaml
  uwb_pipeline_10loop_moretrain.yaml

src/uwb_localization/
  data.py
  calibration.py
  preprocessing.py
  anchor_optimization.py
  ekf.py
  features.py
  lstm.py
  constraints.py
  metrics.py
  plotting.py

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

Dokumentasi pipeline detail tersedia di:

[docs/UWB_CALIBRATED_PIPELINE.md](docs/UWB_CALIBRATED_PIPELINE.md)

### 3.3 Snapshot Hasil Terbaru

Snapshot ringan hasil terbaru yang diringkas di README ini berada di:

```text
docs/results/20260518_213455/
```

Folder tersebut berisi metrics, parameter kalibrasi range, hasil optimasi
anchor, hasil tuning EKF, configuration snapshot, dan plot. Full output
pipeline tetap berada secara lokal di:

```text
outputs/uwb_10loop_moretrain_pipeline/20260518_213455/
```

Konfigurasi yang digunakan adalah:

```text
configs/uwb_pipeline_10loop_moretrain.yaml
```

---

## 4. Metodologi

### 4.1 Data Loading dan Split No-Leakage

Pipeline membaca seluruh file CSV, menstandardisasi nama kolom, memvalidasi
kolom wajib, dan menambahkan metadata:

```text
source_file, trajectory, segment_id, sample_index, split,
gt_segment, gt_loop_id, gt_segment_progress
```

Strategi split pada konfigurasi terbaik:

1. `10lup_trilat_gt` digunakan sebagai data train.
2. `10lup1_trilat_gt` digunakan sebagai data train, dengan 15% bagian akhir
   menjadi validation.
3. `10lup2_trilat_gt` digunakan sebagai **test held-out**.
4. File test tidak digunakan untuk fitting kalibrasi, optimasi anchor, scaler,
   early stopping, tuning EKF, atau training LSTM.

Pendekatan ini menjaga evaluasi tetap no-data-leakage karena test set berasal
dari sesi pengambilan data terpisah dan tidak ikut dipakai saat training.

Split tidak dilakukan random per baris karena data UWB adalah time-series.
Jika baris diacak, sampel train dan test bisa berasal dari loop yang sama dan
berdekatan waktunya. Kondisi tersebut membuat hasil terlihat sangat bagus,
tetapi tidak membuktikan kemampuan model pada sesi pengambilan data baru yang
benar-benar terpisah.

### 4.2 Kalibrasi Range per Anchor

Kalibrasi range dilakukan per anchor menggunakan model linear:

```text
d_corrected = a * d_raw + b
```

Parameter `a` dan `b` hanya di-fit pada train split. Ground-truth distance
dihitung dari posisi ground truth tag terhadap posisi anchor. Setelah kalibrasi,
range diproses menggunakan filter robust untuk mengurangi lonjakan pembacaan
UWB yang tidak masuk akal secara temporal.

Preprocessing range yang digunakan:

1. Hampel/median filtering untuk mendeteksi spike lokal.
2. Pembatasan perubahan range antar-sampel.
3. Exponential moving average ringan agar data tidak terlalu bergetar.

Tujuan tahap ini bukan memaksa lintasan menjadi persegi ideal, tetapi
mengurangi pembacaan range yang tidak mungkin secara fisik. Parameter kalibrasi
dan preprocessing dipelajari dari train split, lalu diterapkan apa adanya ke
validation dan test.

### 4.3 Anchor Optimization

Pipeline mendukung anchor optimization untuk mengoreksi kecil posisi anchor dan
bias range berdasarkan train split. Optimasi dilakukan dengan batas perpindahan
agar posisi anchor tidak berubah secara tidak realistis.

Pada konfigurasi terbaik, anchor optimization aktif:

```yaml
anchor_optimization:
  enabled: true
  max_anchor_move_m: 0.15
  max_bias_m: 0.50
```

Hasil optimasi anchor tetap dibatasi dan hanya di-fit pada train split. Dengan
demikian, koreksi anchor tidak menggunakan informasi dari test set.

Tahap ini diperlukan karena posisi anchor hasil pengukuran manual dapat memiliki
selisih beberapa sentimeter. Jika anchor dibiarkan tanpa koreksi, error tersebut
akan masuk ke perhitungan range dan posisi. Namun, batas `max_anchor_move_m`
tetap dipasang agar optimasi tidak menggeser anchor terlalu jauh hanya demi
menyesuaikan data train.

### 4.4 Extended Kalman Filter Berbasis Range

EKF terbaru tidak hanya smoothing posisi `x,y` lama, tetapi memproses range UWB
secara langsung.

State EKF:

```text
[x, y, vx, vy]
```

Measurement model:

```text
h_i(x, y) = sqrt((x - ax_i)^2 + (y - ay_i)^2) + bias_i
```

EKF mendukung:

1. Constant-velocity prediction model.
2. Jacobian measurement model.
3. Process noise dan measurement noise configurable.
4. Innovation gating dan outlier rejection.
5. Minimum valid anchor count.
6. Tuning parameter EKF berdasarkan validation split.

Parameter EKF terbaik pada run terbaru:

| Parameter | Nilai |
| --- | ---: |
| `process_noise_accel` | `0.35` |
| `measurement_noise` | `0.08` |
| `gating_threshold` | `16.27` |
| `enable_gating` | `true` |

EKF dipakai sebagai estimator utama karena robot bergerak kontinu, sehingga
posisi antar-sampel seharusnya tidak berubah terlalu ekstrem. Prediction step
menjaga gerakan tetap halus, sedangkan update step tetap mengikuti pembacaan
range UWB. Innovation gating membantu menolak pengukuran yang terlalu jauh dari
prediksi model gerak.

### 4.5 LSTM Residual Correction

LSTM tidak memprediksi posisi absolut. LSTM dilatih untuk memprediksi residual
terhadap output EKF:

```text
residual_x = x_true - ekf_x
residual_y = y_true - ekf_y
```

Estimasi akhir:

```text
x_lstm = x_ekf + residual_x_pred
y_lstm = y_ekf + residual_y_pred
```

Fitur input LSTM mencakup posisi EKF, kecepatan EKF, covariance, innovation
residual, calibrated/filtered range, range difference, delta time, dan motion
features. Scaler hanya di-fit pada train split, validation hanya digunakan untuk
early stopping, dan test tidak disentuh selama training.

Untuk mencegah koreksi tidak realistis, residual output di-clip menggunakan
`residual_clip_m`.

LSTM dibuat sebagai residual corrector agar model tidak belajar ulang seluruh
posisi dari nol. EKF sudah memberikan estimasi posisi yang stabil, lalu LSTM
hanya mempelajari pola sisa error yang konsisten, misalnya bias lokal pada
segmen tertentu. Strategi ini lebih aman daripada meminta LSTM langsung
memprediksi `x,y` absolut dari data UWB mentah.

### 4.6 Evaluasi dan Visualisasi

Metrik yang dilaporkan:

1. RMSE X.
2. RMSE Y.
3. RMSE 2D Euclidean.
4. MAE X.
5. MAE Y.
6. MAE 2D.
7. Median error.
8. P90 dan P95 error.
9. Max error.
10. Persentase sample di bawah 5 cm, 10 cm, 20 cm, dan 50 cm.

Metrik utama untuk klaim akurasi adalah **RMSE 2D Euclidean**, karena posisi
robot berada pada bidang 2D dan error harus dihitung dari gabungan error sumbu
X dan Y.

Untuk target alternatif dari dosen, metrik pendamping yang dilaporkan adalah
**MAE 2D**. README ini menulis keduanya agar pembaca dapat membedakan hasil
konservatif RMSE dan rata-rata error MAE secara jelas.

---

## 5. Hasil Eksperimen Terbaru

### 5.1 Split Eksperimen

Konfigurasi hasil terbaik menggunakan `configs/uwb_pipeline_10loop_moretrain.yaml`.

| Split | Trajectory | Jumlah Sample |
| --- | --- | ---: |
| Train | `10lup_trilat_gt` | 5,814 |
| Train | `10lup1_trilat_gt` | 6,398 |
| Validation | `10lup_trilat_gt` | 1,027 |
| Validation | `10lup1_trilat_gt` | 1,130 |
| Test | `10lup2_trilat_gt` | 8,055 |

Jumlah sample yang masuk ke evaluasi LSTM sedikit lebih kecil dari jumlah raw
sample karena LSTM membutuhkan sequence window sepanjang 30 sample.

### 5.2 Hasil Kalibrasi Range

Parameter kalibrasi range di-fit menggunakan train split saja. Setelah itu
parameter diterapkan ke train, validation, dan test.

Pada run terbaik, preprocessing range mengganti sebagian kecil data yang
terdeteksi sebagai lonjakan:

| Split | Trajectory | Anchor 1 replaced | Anchor 2 replaced | Anchor 3 replaced |
| --- | --- | ---: | ---: | ---: |
| Train | `10lup_trilat_gt` | 0.00% | 2.32% | 0.00% |
| Train | `10lup1_trilat_gt` | 0.08% | 1.90% | 0.00% |
| Test | `10lup2_trilat_gt` | 0.06% | 2.64% | 0.00% |

Hasil ini menunjukkan bahwa mayoritas data tetap dipertahankan, sedangkan
preprocessing hanya menangani spike yang tampak tidak konsisten secara temporal.

### 5.3 Hasil Kuantitatif Test Set

Evaluasi pada test held-out `10lup2_trilat_gt`:

| Model | RMSE X | RMSE Y | RMSE 2D | MAE 2D | Median Error | P95 Error | < 5 cm | < 10 cm | < 20 cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw trilateration | 0.176 m | 0.164 m | 0.249 m | 0.219 m | 0.212 m | 0.426 m | 4.97% | 16.35% | 45.83% |
| EKF only | 0.120 m | 0.123 m | 0.171 m | 0.152 m | 0.149 m | 0.293 m | 10.05% | 30.36% | 72.27% |
| EKF + LSTM residual | **0.077 m** | **0.082 m** | **0.113 m** | **0.095 m** | **0.085 m** | **0.209 m** | **25.64%** | **59.05%** | **93.65%** |

Perbandingan utama:

| Perbandingan | Perubahan RMSE 2D |
| --- | ---: |
| EKF only vs Raw trilateration | 31.30% lebih baik |
| EKF + LSTM residual vs EKF only | 34.36% lebih baik |
| EKF + LSTM residual vs Raw trilateration | 54.91% lebih baik |

Interpretasi utama: LSTM residual correction berhasil memperbaiki output EKF
pada test set yang tidak digunakan saat training. Error akhir adalah 11.25 cm
RMSE 2D dan 9.50 cm MAE 2D. Artinya, target 5 cm belum tercapai secara valid,
tetapi target alternatif di bawah 10 cm sudah tercapai jika dosen menerima MAE
2D sebagai metrik rata-rata error.

Track record penurunan error sampai target alternatif 10 cm:

| Tahap | Input utama | Fungsi tahap | RMSE 2D | MAE 2D | Status |
| --- | --- | --- | ---: | ---: | --- |
| Raw trilateration | `el1`, `el2`, `el3` langsung | Baseline posisi dari trilaterasi | 24.95 cm | 21.93 cm | Error masih besar |
| EKF only | Range terkalibrasi + model gerak | Menstabilkan posisi dan menahan spike | 17.14 cm | 15.20 cm | Error turun, belum <10 cm |
| EKF + LSTM residual | Output EKF + fitur residual | Mengoreksi pola residual EKF | 11.25 cm | **9.50 cm** | MAE sudah <10 cm |

Penurunan error terjadi bertahap. Kalibrasi dan EKF mengurangi error dari
pembacaan range yang bias dan berisik. LSTM residual kemudian memperbaiki
kesalahan sisa yang masih berulang pada output EKF. Karena test set tetap
terpisah, penurunan ini dapat dilaporkan sebagai hasil generalisasi pipeline,
bukan hasil dari kebocoran data.

### 5.4 Hasil Per Lintasan

| Split | Trajectory | Model Terbaik | RMSE 2D Terbaik |
| --- | --- | --- | ---: |
| Train | `10lup_trilat_gt` | EKF + LSTM residual | 0.142 m |
| Train | `10lup1_trilat_gt` | EKF + LSTM residual | 0.076 m |
| Validation | `all` | EKF + LSTM residual | 0.121 m |
| Test | `10lup2_trilat_gt` | EKF + LSTM residual | 0.113 m |

Sebagai pembanding, eksperimen yang melibatkan data 5-loop menghasilkan performa
test lebih rendah karena catatan waktunya hanya berupa interval awal-gerak dan
akhir-gerak, bukan timestamp waypoint yang detail. Oleh karena itu, hasil
10-loop lebih kuat untuk dijadikan hasil utama.

Breakdown error EKF + LSTM residual pada test set `10lup2_trilat_gt`:

| Segmen Test | RMSE 2D | MAE 2D | Median | P95 | Jumlah Sample |
| --- | ---: | ---: | ---: | ---: | ---: |
| Start/stop di P1 | 19.56 cm | 18.68 cm | 19.75 cm | 26.86 cm | 169 |
| P1 ke P2 | 11.50 cm | 9.27 cm | 7.92 cm | 22.33 cm | 1,830 |
| P2 ke P3 | 10.63 cm | 9.25 cm | 8.67 cm | 18.56 cm | 2,030 |
| P3 ke P4 | 12.43 cm | 10.51 cm | 9.62 cm | 25.60 cm | 2,108 |
| P4 ke P1 | 9.03 cm | 8.05 cm | 7.53 cm | 15.25 cm | 1,889 |

Tabel ini menunjukkan bahwa sebagian besar segmen utama sudah berada di sekitar
atau di bawah 10 cm pada MAE/median. Error terbesar masih muncul pada bagian
start/stop dan beberapa segmen yang terkena lonjakan UWB, sehingga RMSE total
masih tertahan di 11.25 cm.

### 5.5 Visualisasi

Gambar berikut ditambahkan sebagai bukti ringkas bahwa hasil terbaru sudah
sesuai dengan kesepakatan target alternatif: MAE 2D berada di bawah 10 cm,
sedangkan RMSE 2D masih dilaporkan apa adanya.

#### Bukti Target Alternatif 10 cm

![Bukti Target Alternatif 10 cm](docs/results/20260518_213455/target_10cm_evidence.png)

#### Breakdown Error per Segmen Test

![Breakdown Error per Segmen](docs/results/20260518_213455/segment_error_breakdown.png)

#### 5.5.1 Perbandingan Metode pada Test Set

![Test Method Comparison](docs/results/20260518_213455/test_method_comparison.png)

#### 5.5.2 CDF Error 2D pada Test Set

![Test Error CDF](docs/results/20260518_213455/test_error_cdf.png)

#### 5.5.3 Full Trajectory Comparison

![Full Trajectory Comparison](docs/results/20260518_213455/full_trajectory_comparison.png)

#### 5.5.4 Trajectory per Lintasan

| Train `10lup` | Train/Validation `10lup1` |
| --- | --- |
| ![Trajectory 10lup](docs/results/20260518_213455/trajectory_10lup_trilat_gt.png) | ![Trajectory 10lup1](docs/results/20260518_213455/trajectory_10lup1_trilat_gt.png) |

| Test `10lup2` | Full Comparison |
| --- | --- |
| ![Trajectory 10lup2](docs/results/20260518_213455/trajectory_10lup2_trilat_gt.png) | ![Full Trajectory Comparison](docs/results/20260518_213455/full_trajectory_comparison.png) |

#### 5.5.5 Error Over Time per Lintasan

| Train `10lup` | Train/Validation `10lup1` |
| --- | --- |
| ![Error Over Time 10lup](docs/results/20260518_213455/error_over_time_10lup_trilat_gt.png) | ![Error Over Time 10lup1](docs/results/20260518_213455/error_over_time_10lup1_trilat_gt.png) |

| Test `10lup2` | Test CDF |
| --- | --- |
| ![Error Over Time 10lup2](docs/results/20260518_213455/error_over_time_10lup2_trilat_gt.png) | ![Test Error CDF](docs/results/20260518_213455/test_error_cdf.png) |

#### 5.5.6 Residual Distribution LSTM

![LSTM Residual Distribution](docs/results/20260518_213455/lstm_residual_distribution.png)

---

## 6. Diskusi

### 6.1 Interpretasi Hasil

Hasil terbaru menunjukkan bahwa pipeline berlapis mampu meningkatkan akurasi
lokalisasi UWB. Raw trilateration pada test set menghasilkan RMSE 2D sebesar
24.95 cm. Setelah diproses menggunakan EKF, RMSE 2D turun menjadi 17.14 cm.
Setelah ditambahkan LSTM residual correction, RMSE 2D turun lagi menjadi 11.25
cm dan MAE 2D menjadi 9.50 cm.

Peningkatan ini menunjukkan bahwa EKF mampu menstabilkan estimasi posisi dari
range UWB, sedangkan LSTM residual correction mampu mempelajari pola residual
yang masih tersisa dari output EKF. Karena test set tidak digunakan saat
training, peningkatan pada test set dapat dianggap sebagai peningkatan
generalisasi, bukan akibat data leakage. Klaim di bawah 10 cm perlu ditulis
spesifik sebagai **MAE 2D 9.50 cm**, bukan RMSE 2D, karena RMSE masih 11.25 cm.

Jika kalimat "error di bawah 10 cm" digunakan di laporan, penulisannya harus
diperjelas sebagai berikut:

1. **Sudah tercapai untuk MAE 2D:** rata-rata error test adalah 9.50 cm.
2. **Belum tercapai untuk RMSE 2D:** RMSE test masih 11.25 cm.
3. **Belum semua titik di bawah 10 cm:** 59.05% sampel test berada di bawah 10
   cm, sedangkan sisanya masih dipengaruhi spike dan segmen dengan error besar.

### 6.2 Mengapa Target 5 cm Belum Realistis

Target 5 cm belum realistis pada dataset saat ini karena beberapa alasan:

1. Ground truth waypoint masih berbasis timestamp detik, belum timestamp presisi
   sampai pecahan detik.
2. UWB masih menunjukkan spike dan bias berbeda pada area lintasan tertentu.
3. Perbedaan tinggi tag-anchor perlu dikoreksi, sehingga error kecil pada
   pembacaan jarak dapat memengaruhi jarak datar.
4. Raw trilateration test masih memiliki RMSE 2D sekitar 24.95 cm, sehingga
   input awal belum cukup bersih untuk menghasilkan posisi akhir 5 cm secara
   valid.
5. Klaim 5 cm hanya aman jika didukung ground truth presisi dan kondisi
   eksperimen yang sangat terkontrol.

Dengan kata lain, keterbatasan utama bukan hanya algoritma, tetapi juga kualitas
range UWB dan presisi ground truth.

### 6.3 Implikasi untuk Tugas Akhir

Narasi yang paling aman untuk tugas akhir:

> Pipeline lokalisasi UWB telah dibuat dengan prinsip no-data-leakage melalui
> pemisahan train, validation, dan test sebelum proses kalibrasi, tuning, dan
> training. Pada test set terpisah, EKF + LSTM residual correction menurunkan
> RMSE 2D dari 24.95 cm pada raw trilateration menjadi 11.25 cm, dan menghasilkan
> MAE 2D sebesar 9.50 cm. Hasil ini menunjukkan adanya peningkatan akurasi yang
> signifikan. Target 5 cm belum tercapai, tetapi target alternatif di bawah 10 cm
> sudah terpenuhi pada metrik MAE 2D. RMSE 2D masih di atas 10 cm karena adanya
> beberapa spike/lonjakan dan error besar pada loop tertentu.

Narasi yang sebaiknya dihindari:

> Sistem mencapai akurasi 5 cm menggunakan LSTM.

Klaim tersebut belum didukung oleh hasil evaluasi test set saat ini.

---

## 7. Cara Menjalankan Pipeline

### 7.1 Environment

Aktifkan conda environment:

```bash
conda activate uwb-ta
```

Jika dependency belum lengkap:

```bash
conda install -n uwb-ta -c conda-forge numpy pandas scipy scikit-learn matplotlib joblib pyyaml tensorflow=2.19.1 pyserial pillow -y
```

Alternatif:

```bash
conda env update -n uwb-ta -f environment.yml
```

### 7.2 Urutan Eksekusi

Jalankan dari root repository:

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

Urutan tersebut harus dijalankan berurutan karena setiap stage membaca output
dari stage sebelumnya:

| Stage | Script | Fungsi utama | Output yang dipakai stage berikutnya |
| --- | --- | --- | --- |
| 0 | `00_make_waypoint_ground_truth.py` | Membentuk ground truth waypoint dari catatan waktu | CSV ground truth |
| 1 | `01_prepare_dataset.py` | Membaca data dan membuat split train/val/test | Dataset tersplit |
| 2 | `02_calibrate_ranges.py` | Mengoreksi bias range per anchor | Range terkalibrasi |
| 3 | `03_optimize_anchors.py` | Koreksi kecil posisi anchor dan bias | Anchor/bias teroptimasi |
| 4a | `04_tune_ekf.py` | Mencari parameter EKF terbaik memakai validation | Parameter EKF |
| 4b | `04_run_ekf.py` | Menjalankan EKF pada semua split | Output EKF |
| 5 | `05_train_lstm_residual.py` | Melatih LSTM residual dari output EKF | Model dan scaler |
| 6 | `06_evaluate_pipeline.py` | Menghitung metrik test | `metrics.csv` dan prediksi |
| 7 | `07_plot_results.py` | Membuat gambar laporan | Plot hasil |

Untuk mengecek angka utama setelah pipeline selesai:

```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/uwb_10loop_moretrain_pipeline/20260518_213455/06_evaluation/metrics.csv'); print(df[(df['split']=='test') & (df['trajectory']=='all')][['model','rmse_2d_m','mae_2d_m','pct_below_10cm']])"
```

Angka yang dijadikan snapshot README:

```text
Raw trilateration      RMSE 2D 0.2495 m, MAE 2D 0.2193 m
EKF only               RMSE 2D 0.1714 m, MAE 2D 0.1520 m
EKF + LSTM residual    RMSE 2D 0.1125 m, MAE 2D 0.0950 m
```

### 7.3 Output Pipeline

Setiap run membuat folder timestamp:

```text
outputs/uwb_10loop_moretrain_pipeline/<timestamp>/
```

Output penting:

| Folder | Isi |
| --- | --- |
| `01_prepared/` | Dataset tersplit dan manifest |
| `02_range_calibration/` | Parameter kalibrasi range dan summary preprocessing |
| `03_anchor_optimization/` | Anchor config yang digunakan |
| `04_ekf_tuning/` | Grid tuning EKF dan parameter terbaik |
| `04_ekf/` | Output EKF per sample |
| `05_lstm_residual/` | Model LSTM, scaler, training history |
| `06_evaluation/` | Metrics, segment metrics, dan predictions |
| `07_plots/` | Plot trajectory, CDF, bar chart, residual |

---

## 8. Kesimpulan

Pipeline terbaru sudah memenuhi prinsip no-data-leakage dan memberikan evaluasi
yang lebih valid. Pada hasil terbaik, metode **EKF + LSTM residual correction**
menjadi metode terbaik pada test held-out `10lup2_trilat_gt` dengan RMSE 2D
sebesar **0.1125 m** atau sekitar **11.25 cm**, serta MAE 2D sebesar **0.0950
m** atau sekitar **9.50 cm**. Hasil ini lebih baik daripada raw trilateration
sebesar **0.2495 m** dan EKF only sebesar **0.1714 m**.

Target akurasi 5 cm belum tercapai secara valid untuk dataset saat ini. Target
alternatif di bawah 10 cm sudah tercapai pada metrik MAE 2D, tetapi belum pada
RMSE 2D. Pipeline yang dibuat sudah menunjukkan peningkatan akurasi yang jelas,
dapat direproduksi, dan dapat dipertanggungjawabkan karena seluruh proses
kalibrasi, tuning, training, dan evaluasi dilakukan dengan pemisahan data yang
benar.

---

## 9. Referensi File Penting

| File / Folder | Keterangan |
| --- | --- |
| `configs/uwb_pipeline_10loop_moretrain.yaml` | Konfigurasi utama hasil terbaik |
| `configs/latest_waypoint_times.yaml` | Catatan waktu waypoint dan interval gerak |
| `configs/uwb_pipeline_latest.yaml` | Konfigurasi eksperimen semua dataset terbaru |
| `Data eksperimen/latest_waypoint_ground_truth/` | CSV hasil ground truth waypoint |
| `src/uwb_localization/` | Source code modular |
| `scripts/` | Script CLI per stage |
| `docs/UWB_CALIBRATED_PIPELINE.md` | Dokumentasi pipeline detail |
| `docs/results/20260518_213455/metrics.csv` | Metrics hasil terbaik |
| `docs/results/20260518_213455/` | Snapshot gambar dan artifact ringan hasil terbaik |
| `output_lstm_no_leakage/` | Output eksperimen LSTM lama/no-leakage awal |
