# UWB Positioning Tugas Akhir

Repository ini berisi eksperimen lokalisasi indoor berbasis Ultra-Wideband (UWB)
dengan dua tahap utama:

1. Estimasi posisi awal menggunakan Kalman Filter.
2. Koreksi residual error menggunakan LSTM.

Pada eksperimen hybrid, LSTM tidak menggantikan Kalman Filter. LSTM digunakan
untuk memprediksi residual error dari hasil estimasi Kalman Filter:

```text
residual = posisi ground truth - posisi Kalman Filter
posisi akhir = posisi Kalman Filter + residual LSTM
```

## Struktur Data dan Kode

```text
Data hasil/
  data_with_velocity1 (majulurus).csv
  kotak 2 loop.csv
  kotak 3 loop.csv
  diam.csv

Kalman Filter.py
LSTM.py
LSTM_no_leakage.py

output_lstm_no_leakage/
  metrics.csv
  predictions.csv
  calibration_candidates.csv
  training_history.csv
  training_history.png
  test_trajectory_*.png
  full_trajectory_*.png
  test_error_over_time_*.png
  best_lstm_residual_corrector.keras
```

File utama yang digunakan untuk eksperimen terbaru adalah
`LSTM_no_leakage.py`.

## Metodologi Eksperimen LSTM

Eksperimen terbaru dibuat untuk menghindari data leakage pada data time-series.
Beberapa langkah pencegahannya:

1. Data dibagi menjadi train, validation, dan test secara kronologis.
2. Sliding window dibuat setelah pembagian data, sehingga window tidak melewati
   batas split.
3. Scaler fitur dan target hanya di-fit menggunakan data train.
4. Validation digunakan untuk early stopping dan kalibrasi residual.
5. Test hanya digunakan untuk evaluasi akhir.

Fitur yang digunakan berjumlah 29, mencakup posisi raw UWB, posisi Kalman,
jarak ke anchor, innovation terhadap anchor, selisih antar jarak, delta posisi,
kecepatan, dan `dt`.

Target LSTM adalah residual:

```text
err_x = x_true - x_kf
err_y = y_true - y_kf
```

## Kalibrasi Residual

Hasil LSTM raw tidak langsung digunakan sebagai hasil akhir, karena pada data
test residual mentah justru memperburuk estimasi Kalman Filter. Oleh karena itu,
ditambahkan tahap kalibrasi residual menggunakan validation set.

Beberapa kandidat kalibrasi dievaluasi:

1. Kalman only.
2. Kalman + LSTM raw.
3. Kalman + LSTM scalar calibrated.
4. Kalman + LSTM per-axis calibrated.
5. Kalman + LSTM affine calibrated.

Kandidat terbaik berdasarkan validation RMSE Euclidean adalah:

```text
Kalman + LSTM affine calibrated
```

Parameter kalibrasi yang terpilih:

```text
alpha_x = 2.7830
beta_x  = 0.3807
alpha_y = 2.6250
beta_y  = 0.1741
```

## Hasil Utama

Evaluasi test set:

| Model | RMSE X (m) | RMSE Y (m) | RMSE Euclidean (m) | MAE Euclidean (m) | P95 Error (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kalman Filter | 0.3009 | 0.9486 | 0.9952 | 0.8789 | 1.8460 |
| Kalman + LSTM raw | 0.3134 | 0.9706 | 1.0200 | 0.9116 | 1.8686 |
| Kalman + LSTM affine calibrated | 0.3461 | 0.8598 | 0.9269 | 0.8139 | 1.7788 |

Peningkatan test set dibanding Kalman Filter:

| Model | RMSE Euclidean | MAE Euclidean | P95 Error |
| --- | ---: | ---: | ---: |
| Kalman + LSTM raw | -2.4896% | -3.7303% | -1.2231% |
| Kalman + LSTM affine calibrated | 6.8662% | 7.3882% | 3.6379% |

Interpretasi:

- LSTM raw belum mampu memperbaiki estimasi Kalman secara langsung.
- Kalibrasi residual menggunakan validation set membuat hasil hybrid lebih baik
  pada metrik posisi 2D secara keseluruhan.
- Peningkatan terbesar terjadi pada komponen `y`, sedangkan komponen `x` pada
  test set memburuk.
- Secara kualitatif, hasil trajectory masih belum sepenuhnya konsisten pada
  semua lintasan, sehingga metode hybrid ini masih perlu pengembangan lebih
  lanjut.

## Hasil Per Lintasan Test

| Lintasan | KF RMSE Euclidean (m) | Calibrated RMSE Euclidean (m) |
| --- | ---: | ---: |
| Majulurus | 0.5386 | 0.4949 |
| Kotak 2 loop | 0.7589 | 0.7349 |
| Kotak 3 loop | 1.3178 | 1.2154 |
| Diam | 0.2041 | 0.2383 |

Hasil hybrid calibrated membaik pada lintasan majulurus, kotak 2 loop, dan
kotak 3 loop, tetapi memburuk pada data diam. Hal ini menunjukkan bahwa model
masih belum general untuk semua kondisi gerak.

## Cara Menjalankan

Aktifkan environment conda:

```bash
conda activate tensor
cd /home/ucl/Documents/UWB_Positioning_Tugas_Akhir
python LSTM_no_leakage.py
```

Secara default, script akan memuat model yang sudah tersimpan:

```python
RETRAIN_MODEL = False
```

Untuk training ulang dari awal, ubah konfigurasi menjadi:

```python
RETRAIN_MODEL = True
```

Output akan disimpan ke:

```text
output_lstm_no_leakage/
```

## File Output Penting

| File | Keterangan |
| --- | --- |
| `metrics.csv` | Ringkasan metrik train, validation, dan test. |
| `predictions.csv` | Hasil prediksi per sample, termasuk Kalman, LSTM raw, dan LSTM calibrated. |
| `calibration_candidates.csv` | Perbandingan kandidat kalibrasi residual pada validation set. |
| `training_history.csv` | Riwayat training LSTM. |
| `training_history.png` | Plot training dan validation loss. |
| `test_trajectory_*.png` | Plot trajectory khusus test split. |
| `full_trajectory_*.png` | Plot trajectory gabungan train, validation, dan test. |
| `test_error_over_time_*.png` | Plot error test terhadap waktu. |

## Kesimpulan Sementara

Kalman Filter masih menjadi estimator utama yang paling stabil. LSTM residual
correction dapat menurunkan error posisi 2D setelah dilakukan kalibrasi
menggunakan validation set, namun hasil kualitatif belum konsisten pada seluruh
lintasan. Dengan demikian, pendekatan hybrid Kalman Filter + LSTM menunjukkan
potensi, tetapi masih membutuhkan pengembangan pada kualitas dataset, strategi
split, fitur temporal, dan arsitektur model agar hasil trajectory lebih stabil.
