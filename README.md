# Titanic ML Service — Pertemuan 11

Sistem inferensi Machine Learning berbasis dataset Titanic, terdiri dari dua service:

- **`python-ml/`** — FastAPI ML service (model inference, JWT auth, Redis cache)
- **`express-gateway/`** — Express.js API Gateway (JWT auth terpisah, circuit breaker, proxy ke ML service)

---

## Struktur Folder

```
project/
├── python-ml/
│   ├── main.py               # FastAPI app
│   ├── train.py              # Script training model
│   ├── Titanic.csv           # Dataset
│   ├── model/
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   └── classes.pkl
│   └── requirements.txt
│
├── express-gateway/
│   ├── index.js              # Entry point Express
│   ├── middleware/
│   │   └── auth.js           # JWT middleware
│   ├── routes/
│   │   └── predict.js        # Route + circuit breaker
│   ├── .env
│   └── package.json
│
└── README.md
```

---

## Prasyarat

| Tool | Versi Minimum |
|------|--------------|
| Python | 3.9+ |
| Node.js | 18+ |
| Redis | 6+ (opsional, cache) |

---

## Setup Python ML Service

### 1. Install dependencies

```bash
cd python-ml
pip install -r requirements.txt
```

**requirements.txt:**
```
fastapi
uvicorn[standard]
scikit-learn
numpy
pandas
joblib
redis
python-jose[cryptography]
```

### 2. Siapkan dataset & train model

Letakkan file `Titanic.csv` di folder `python-ml/`, lalu jalankan:

```bash
python train.py
```

Output: file `model/model.pkl`, `model/scaler.pkl`, `model/classes.pkl`

### 3. Jalankan ML service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Service berjalan di: `http://localhost:8000`

### 4. Akses Swagger UI

Buka browser:
```
http://localhost:8000/docs
```

Atau ReDoc (alternatif):
```
http://localhost:8000/redoc
```

---

## Setup Express Gateway

### 1. Install dependencies

```bash
cd express-gateway
npm install
```

### 2. Buat file `.env`

```env
PORT=3000
JWT_SECRET=rahasia-ganti-di-production

# URL ML service
ML_SERVICE_URL=http://localhost:8000

# Kredensial untuk autentikasi ke ML service
ML_SERVICE_USER=gateway
ML_SERVICE_PASS=alitkicau
```

### 3. Jalankan Express gateway

```bash
node index.js
# atau pakai nodemon (development)
npx nodemon index.js
```

Gateway berjalan di: `http://localhost:3000`

---

## Menjalankan Semua Service Sekaligus

Buka **3 terminal terpisah**:

**Terminal 1 — Redis:**
```bash
redis-server
```

**Terminal 2 — ML Service:**
```bash
cd python-ml
uvicorn main:app --port 8000 --reload
```

**Terminal 3 — Express Gateway:**
```bash
cd express-gateway
node index.js
```

---

## Alur Autentikasi

Terdapat **dua sistem JWT yang terpisah**:

```
Client → [POST /auth/login] → Express Gateway → dapat token Express
Client → [POST /api/predict + Bearer token] → Express Gateway
         → verifikasi token Express
         → ambil/refresh ML token (internal)
         → forward ke FastAPI
         → return hasil prediksi
```

| Service | Endpoint Login | Password Default |
|---------|---------------|-----------------|
| Express Gateway | `POST /auth/login` | `alitkicau` |
| FastAPI ML | `POST /auth/token` | `alitkicau` |

> **Catatan:** Client hanya berinteraksi dengan Express Gateway. Token ML dikelola otomatis oleh gateway.

---

## Daftar Endpoint

### Express Gateway (`localhost:3000`)

| Method | Endpoint | Auth | Deskripsi |
|--------|----------|------|-----------|
| POST | `/auth/login` | X | Login, dapat JWT Express |
| POST | `/api/predict` | Bearer | Prediksi single penumpang |
| POST | `/api/batch-predict` | Bearer | Prediksi batch |
| GET | `/api/ml-health` | Bearer | Cek status ML service |

### FastAPI ML Service (`localhost:8000`)

| Method | Endpoint | Auth | Deskripsi |
|--------|----------|------|-----------|
| GET | `/health` | X | Health check |
| POST | `/auth/token` | X | Login ML service |
| POST | `/predict` | Bearer | Prediksi (dengan cache) |
| POST | `/batch-predict` | Bearer | Prediksi batch |
| GET | `/docs` | X | Swagger UI |
| GET | `/redoc` | X | ReDoc UI |

---

## Format Features

Prediksi membutuhkan **6 fitur** dalam urutan berikut:

| Index | Nama | Tipe | Range | Keterangan |
|-------|------|------|-------|------------|
| 0 | `Pclass` | int | 1–3 | Kelas tiket (1=first, 2=second, 3=third) |
| 1 | `Sex` | int | 0–1 | Jenis kelamin (0=male, 1=female) |
| 2 | `Age` | float | 0.42–80 | Usia penumpang |
| 3 | `SibSp` | int | 0–8 | Jumlah saudara/pasangan di kapal |
| 4 | `Parch` | int | 0–6 | Jumlah orang tua/anak di kapal |
| 5 | `Fare` | float | 0–512.33 | Harga tiket |

**Contoh request body:**
```json
{
  "features": [3, 0, 22, 1, 0, 7.25]
}
```

---

## Circuit Breaker

Express Gateway mengimplementasikan circuit breaker sederhana:

- **Threshold:** 3 kegagalan berturut-turut → circuit **OPEN**
- **Reset:** Otomatis setelah **30 detik**
- **Response saat open:** `503 ML Service sedang tidak tersedia (circuit open)`

---

## Troubleshooting

| Error | Penyebab | Solusi |
|-------|----------|--------|
| `ModuleNotFoundError` | Dependency belum install | `pip install -r requirements.txt` |
| `FileNotFoundError: model/model.pkl` | Model belum di-train | Jalankan `python train.py` |
| `Token tidak valid` | Token expired atau salah | Login ulang untuk dapat token baru |
| `503 ML Service tidak dapat dihubungi` | ML service belum jalan | Pastikan FastAPI sudah running di port 8000 |
| `Redis connection refused` | Redis tidak jalan | Cache otomatis dinonaktifkan, service tetap jalan |
| `ECONNREFUSED` di Express | ML_SERVICE_URL salah | Cek `.env`, pastikan port sesuai |

--

# LINK REPO GITHUB

[Titanic: Machine Learning](https://github.com/Nonom0no/titanic-ml.git)