# 🦋 LepiVision — Butterfly Species Classification with Deep Learning

> Klasifikasi spesies kupu-kupu menggunakan Convolutional Neural Network (CNN) berbasis arsitektur Sequential dan Transfer Learning (InceptionResNetV2), dilatih pada Google Colab dengan dataset Kaggle.

---

## 📋 Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Demo & Hasil](#demo--hasil)
- [Dataset](#dataset)
- [Arsitektur Model](#arsitektur-model)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Struktur Proyek](#struktur-proyek)
- [Cara Menjalankan](#cara-menjalankan)
- [Konfigurasi Training](#konfigurasi-training)
- [Hasil Pelatihan](#hasil-pelatihan)
- [Lisensi](#lisensi)

---

## 📌 Tentang Proyek

**LepiVision** adalah proyek *image classification* untuk mengenali spesies kupu-kupu dari gambar menggunakan teknik *deep learning*. Proyek ini dibangun sebagai Tugas Besar mata kuliah **Sistem Cerdas dan Pembelajaran (SCL)**.

Dua pendekatan model diimplementasikan dan dibandingkan:
1. **Model CNN Sequential** — dibangun dari awal (*from scratch*) dengan beberapa lapisan konvolusi.
2. **Model Transfer Learning (InceptionResNetV2)** — memanfaatkan bobot pra-latih dari ImageNet untuk meningkatkan performa.

Target utama: mencapai akurasi **> 95%** pada data training dengan loss **< 0.001**.

---

## 🎯 Demo & Hasil

| Model | Optimizer | Epochs | Target Akurasi |
|---|---|---|---|
| CNN Sequential | RMSprop | Maks. 60 | > 95% |
| InceptionResNetV2 | RMSprop | Maks. 60 | > 95% |

Kurva training disimpan sebagai `history.csv` di masing-masing folder model (`model[0]/` dan `model[1]/`) untuk analisis lebih lanjut.

---

## 🗂️ Dataset

Dataset yang digunakan adalah **Butterfly & Moths Image Classification** dari Kaggle, yang berisi gambar kupu-kupu dari **100 spesies** (subset 50 kelas digunakan dalam proyek ini).

- **Sumber:** [Kaggle — Butterfly Images 40 Species](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species)
- **Pembuat:** `gpiosenka`
- **Resolusi Input:** Semua gambar di-*resize* ke `224 × 224 piksel`
- **Pembagian Data:**
  - `butterflies/train` — Data pelatihan
  - `butterflies/valid` — Data validasi
  - `butterflies/test`  — Data pengujian & eksplorasi

---

## 🧠 Arsitektur Model

### Model 1 — CNN Sequential (From Scratch)

```
Input (224, 224, 3)
    ↓
Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)
Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)
Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)
Conv2D(128, 3×3, ReLU) → MaxPooling2D(2×2)
Conv2D(256, 3×3, ReLU) → MaxPooling2D(2×2)
Dropout(0.2)
    ↓
Flatten
Dense(256, ReLU)
Dense(50, Sigmoid)   ← Output: 50 kelas
```

### Model 2 — Transfer Learning (InceptionResNetV2)

```
Input (224, 224, 3)
    ↓
InceptionResNetV2 (pre-trained ImageNet, include_top=False)
    ↓
GlobalAveragePooling2D
Flatten
Dense(50, Softmax)   ← Output: 50 kelas
```

Pre-trained weights dari **ImageNet** digunakan untuk mengekstrak fitur, sementara lapisan klasifikasi disesuaikan dengan jumlah kelas pada dataset ini.

---

## ⚙️ Teknologi yang Digunakan

| Teknologi | Versi / Keterangan |
|---|---|
| Python | 3.8+ |
| TensorFlow / Keras | 2.x |
| scikit-learn | Model evaluation |
| NumPy | Operasi numerik |
| Matplotlib | Visualisasi gambar & grafik |
| Pandas | Logging history training |
| Google Colab | Platform pelatihan (GPU) |
| Kaggle API | Download dataset otomatis |

---

## 📁 Struktur Proyek

```
LepiVision/
│
├── lepivision.ipynb          # Notebook utama (training & evaluasi)
│
├── model[0]/                 # Output Model 1 — CNN Sequential
│   ├── bestValLoss.hdf5      # Bobot model terbaik (val_loss)
│   ├── bestLoss.hdf5         # Bobot model terbaik (loss)
│   └── history.csv           # Riwayat akurasi & loss per epoch
│
├── model[1]/                 # Output Model 2 — InceptionResNetV2
│   ├── bestValLoss.hdf5
│   └── history.csv
│
├── lastModel/                # Model Sequential terakhir (SavedModel)
├── inception/                # Model Inception terakhir (SavedModel)
│
└── README.md
```

> **Catatan:** Folder `butterflies/`, `model[0]/`, `model[1]/`, `lastModel/`, dan `inception/` **tidak disertakan** dalam repositori ini karena ukurannya yang besar. Ikuti langkah-langkah di bawah untuk mengunduh dataset dan melatih ulang model.

---

## 🚀 Cara Menjalankan

### Prasyarat

1. Akun **Google** dengan akses ke [Google Colab](https://colab.research.google.com) dan **Google Drive**.
2. Akun **Kaggle** dengan API Token (`kaggle.json`).
   - Unduh dari: `kaggle.com` → Profil → *Settings* → *API* → *Create New Token*.

### Langkah-Langkah

**1. Buka Notebook di Google Colab**

Upload file `lepivision.ipynb` ke Google Colab, atau klik tombol di bawah:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**2. Mount Google Drive**

Sebelum menjalankan sel pertama, pastikan Google Drive sudah di-*mount*:
```python
from google.colab import drive
drive.mount('/content/drive')
```

**3. Upload `kaggle.json`**

Jalankan **Cell 1**. Sebuah dialog upload akan muncul — pilih file `kaggle.json` dari komputer Anda. Skrip akan otomatis:
- Memindahkan `kaggle.json` ke `~/.kaggle/`
- Men-download dataset `butterfly-images40-species` dari Kaggle
- Mengekstrak dan menyalinnya ke Google Drive

**4. Jalankan Sel Secara Berurutan**

| Cell | Konten |
|---|---|
| Cell 1 | Persiapan lingkungan, download dataset, import library |
| Cell 2 | Eksplorasi dataset (menampilkan sampel gambar per kelas) |
| Cell 3 | Augmentasi data & konfigurasi generator |
| Cell 4 | Persiapan callbacks (checkpoint, early stopping, custom callback) |
| Cell 5 | Training Model 1 — CNN Sequential |
| Cell 6 | Training Model 2 — InceptionResNetV2 |

---

## 🔧 Konfigurasi Training

### Augmentasi Data

```python
ImageDataGenerator(
    rescale        = 1./255,   # Normalisasi piksel ke [0, 1]
    rotation_range = 20,       # Rotasi acak ±20°
    zoom_range     = 0.2,      # Zoom acak ±20%
    shear_range    = 0.2,      # Shear transform
    fill_mode      = 'nearest'
)
```

### Callbacks

| Callback | Konfigurasi |
|---|---|
| `ModelCheckpoint` | Simpan bobot terbaik berdasarkan `val_loss` |
| `EarlyStopping` | Hentikan training jika `val_loss` tidak membaik selama 10 epoch |
| `myCallback` | Hentikan training jika `accuracy > 95%` dan `loss < 0.001` |

### Hyperparameter

| Parameter | Nilai |
|---|---|
| Input Size | 224 × 224 × 3 |
| Batch Size | 32 |
| Epochs (maks.) | 60 |
| Optimizer | RMSprop |
| Loss Function | Categorical Crossentropy |
| Metrik | Accuracy |

---

## 📊 Hasil Pelatihan

Setelah training selesai, riwayat akurasi dan loss tersimpan dalam file CSV:

```
model[0]/history.csv   → Riwayat CNN Sequential
model[1]/history.csv   → Riwayat InceptionResNetV2
```

Kolom yang tersimpan:
- `accuracy` — Akurasi pada data training
- `val_accuracy` — Akurasi pada data validasi
- `loss` — Loss pada data training
- `val_loss` — Loss pada data validasi

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademis. Dataset digunakan sesuai lisensi yang tercantum pada halaman Kaggle dataset [gpiosenka/butterfly-images40-species](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species).

---

<p align="center">
  Dibuat dengan ❤️ untuk Tugas Besar Sistem Cerdas dan Pembelajaran (SCL)
</p>
