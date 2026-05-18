# Penjelasan Dataset & Model — Titanic Classifier

## 1. Dataset: Titanic

### Asal Usul
Dataset Titanic adalah salah satu dataset paling terkenal di dunia machine learning. Berisi data penumpang kapal RMS Titanic yang tenggelam pada 15 April 1912 setelah menabrak gunung es di Samudra Atlantik Utara. Dari 2.224 penumpang dan kru, sekitar 1.500 jiwa meninggal dunia.

Dataset ini populer karena:
- Ukurannya sedang (cocok untuk pembelajaran)
- Memiliki campuran fitur numerik dan kategorikal
- Target variabelnya binary (selamat / tidak)
- Konteks ceritanya menarik dan mudah dipahami

**Sumber asli:** Kompetisi Kaggle — [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

---

### Fitur yang Digunakan

Dari dataset asli, dipilih **6 fitur** yang paling relevan dan tidak memiliki banyak nilai kosong:

| Fitur | Tipe | Deskripsi | Alasan Dipilih |
|-------|------|-----------|----------------|
| `Pclass` | Ordinal (1,2,3) | Kelas tiket penumpang | Kelas tiket sangat memengaruhi akses ke sekoci |
| `Sex` | Binary (0,1) | Jenis kelamin | "Wanita dan anak-anak dahulu" — kebijakan evakuasi nyata |
| `Age` | Numerik (float) | Usia penumpang | Anak-anak diprioritaskan, lansia lebih rentan |
| `SibSp` | Numerik (int) | Jumlah saudara/pasangan di kapal | Ukuran keluarga memengaruhi keputusan evakuasi |
| `Parch` | Numerik (int) | Jumlah orang tua/anak di kapal | Idem, terkait dinamika keluarga |
| `Fare` | Numerik (float) | Harga tiket | Berkorelasi dengan kelas & lokasi kabin |

**Target variabel:**
- `Survived = 0` → Tidak Selamat
- `Survived = 1` → Selamat

### Preprocessing

1. **Hapus baris dengan nilai kosong (`dropna()`)** — terutama kolom `Age` yang sering kosong
2. **Encoding `Sex`** — ubah teks menjadi angka: `male → 0`, `female → 1`
3. **Tidak ada fitur tambahan (feature engineering)** — sengaja disederhanakan untuk keperluan pembelajaran

---

### Distribusi Data (estimasi dari dataset standar)

| Keterangan | Jumlah |
|-----------|--------|
| Total penumpang (dataset) | ~891 |
| Setelah `dropna()` | ~714 |
| Data training (80%) | ~571 |
| Data testing (20%) | ~143 |

| Kategori | Persentase |
|----------|-----------|
| Tidak Selamat (0) | ~61% |
| Selamat (1) | ~39% |

> Dataset ini **imbalanced** — lebih banyak yang tidak selamat. Ini merupakan refleksi dari kejadian nyata.

---

## 2. Model: Random Forest Classifier

### Apa itu Random Forest?

Random Forest adalah algoritma **ensemble learning** yang membangun banyak decision tree secara paralel, lalu menggabungkan hasil prediksi masing-masing tree melalui voting mayoritas.

```
Input Features
      │
      ▼
┌──────────────────────────────────────────────┐
│           Random Forest (100 trees)          │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐      │
│  │Tree 1│  │Tree 2│  │Tree 3│  │Tree N│      │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘      │
│     │         │          │         │         │
│     └─────────┴──────────┴─────────┘         │
│              Voting Mayoritas                │
└──────────────────────────────────────────────┘
      │
      ▼
Prediksi Final (0 atau 1) + Probabilitas
```

### Konfigurasi Model

```python
RandomForestClassifier(
    n_estimators=100,   # 100 decision tree
    random_state=42     # seed untuk reprodusibilitas
)
```

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `n_estimators` | 100 | Jumlah tree — semakin banyak, semakin stabil (tapi lebih lambat) |
| `random_state` | 42 | Seed random — hasil training selalu sama setiap dijalankan |

### Preprocessing: StandardScaler

Sebelum masuk ke model, semua fitur dinormalisasi menggunakan `StandardScaler`:

```
X_scaled = (X - mean) / std_deviation
```

Ini penting karena:
- Fitur seperti `Fare` (0–512) jauh lebih besar skala-nya dari `Pclass` (1–3)
- Mencegah fitur bernilai besar mendominasi hasil prediksi

### Performa Model

Menggunakan train-test split 80:20 dengan `random_state=42`:

| Metrik | Nilai (estimasi) |
|--------|-----------------|
| **Accuracy** | ~82–85% |
| **Precision (Selamat)** | ~80–84% |
| **Recall (Selamat)** | ~75–80% |
| **F1-Score** | ~78–82% |

> Nilai persis bervariasi tergantung versi dataset. Lihat output `python train.py` untuk nilai aktual.

### Mengapa Random Forest?

| Keunggulan | Penjelasan |
|-----------|-----------|
| Akurasi tinggi | Umumnya lebih baik dari single decision tree |
| Robust terhadap outlier | Agregasi banyak tree meredam pengaruh outlier |
| Feature importance | Bisa melihat fitur mana yang paling berpengaruh |
| Tidak mudah overfit | Karena menggunakan banyak tree independen |
| Mudah digunakan | Sedikit hyperparameter yang perlu diatur |

---

## 3. Alur Inferensi

```
Request masuk: [Pclass=3, Sex=0, Age=22, SibSp=1, Parch=0, Fare=7.25]
        │
        ▼
  Validasi range (FEATURE_BOUNDS)
        │
        ▼
  Cek cache Redis (hash MD5 dari features)
        │
   ┌────┴────┐
  Cache     Cache
  hit       miss
   │         │
   │         ▼
   │   StandardScaler.transform()
   │         │
   │         ▼
   │   RandomForest.predict() → [0]
   │   RandomForest.predict_proba() → [[0.87, 0.13]]
   │         │
   │         ▼
   │   Simpan ke Redis (TTL 5 menit)
   │         │
   └────┬────┘
        ▼
Response: { prediction: 0, label: "Tidak Selamat", confidence: 0.87 }
```

---

## 4. File Model yang Disimpan

| File | Isi | Ukuran Estimasi |
|------|-----|-----------------|
| `model/model.pkl` | Object `RandomForestClassifier` terlatih | ~5–15 MB |
| `model/scaler.pkl` | Object `StandardScaler` dengan mean & std dari data training | < 1 KB |
| `model/classes.pkl` | Label class: `["Tidak Selamat", "Selamat"]` | < 1 KB |

> File `.pkl` menggunakan format binary Python (joblib). **Jangan buka file ini di production dari sumber tidak terpercaya** — `.pkl` bisa mengandung kode berbahaya.

---

## 5. Keterbatasan Model

1. **Dataset kecil** — hanya ~714 sampel setelah preprocessing
2. **Missing value dihapus** — bukan di-impute, sehingga kehilangan informasi
3. **Fitur terbatas** — fitur seperti `Cabin`, `Embarked`, `Ticket` diabaikan
4. **Tidak ada cross-validation** — hanya single train-test split
5. **Hyperparameter default** — belum dioptimasi dengan GridSearch/RandomSearch

Untuk production nyata, model ini perlu ditingkatkan dengan teknik seperti:
- Imputation untuk nilai yang hilang
- Feature engineering (misalnya ekstrak gelar dari nama: Mr, Mrs, Miss)
- Hyperparameter tuning
- Cross-validation untuk evaluasi yang lebih robust