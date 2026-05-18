import os
import json
import hashlib
from typing import List, Optional
from datetime import datetime, timedelta

import numpy as np
import joblib
import redis

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import JWTError, jwt

# --- Config ---
SECRET_KEY = os.getenv("JWT_SECRET", "secret-key-ganti-di-production")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# --- App ---
app = FastAPI(
    title="Python ML Service - Titanic Classifier",
    description="AI/ML Inference Service menggunakan dataset Titanic. Pertemuan 11.",
    version="1.0.0",
)

# --- Load model ---
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
classes = joblib.load("model/classes.pkl")

# --- Redis ---
try:
    cache = redis.from_url(REDIS_URL, decode_responses=True)
    cache.ping()
    CACHE_ENABLED = True
except Exception:
    cache = None
    CACHE_ENABLED = False

# --- JWT ---
security = HTTPBearer()

def create_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token tidak valid")

# --- Feature bounds ---
FEATURE_BOUNDS = [
    ("Pclass", 1, 3),
    ("Sex", 0, 1),
    ("Age", 0.42, 80),
    ("SibSp", 0, 8),
    ("Parch", 0, 6),
    ("Fare", 0, 512.33),
]

def validate_features(features: List[float]):
    if len(features) != 6:
        raise HTTPException(
            status_code=400,
            detail="Butuh tepat 6 features: Pclass, Sex(0/1), Age, SibSp, Parch, Fare"
        )
    for i, (name, min_val, max_val) in enumerate(FEATURE_BOUNDS):
        if not (min_val <= features[i] <= max_val):
            raise HTTPException(
                status_code=400,
                detail=f"{name} harus antara {min_val} dan {max_val}, nilai yang diterima: {features[i]}"
            )

# --- Schema ---
class LoginRequest(BaseModel):
    username: str
    password: str

class PredictRequest(BaseModel):
    features: List[float]  # [Pclass, Sex, Age, SibSp, Parch, Fare]
    user_id: Optional[str] = None

class PredictResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    cached: bool = False
    service: str = "python-ml-fastapi"

class BatchPredictRequest(BaseModel):
    items: List[PredictRequest]

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int

# --- Helper ---
def _predict_single(features: List[float]) -> dict:
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)
    return {
        "prediction": int(pred[0]),
        "label": classes[int(pred[0])],
        "confidence": float(proba.max()),
    }

def _cache_key(features: List[float]) -> str:
    key = hashlib.md5(json.dumps(features).encode()).hexdigest()
    return f"titanic:predict:{key}"

# --- Endpoints ---
@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "service": "python-ml-fastapi",
        "cache": "connected" if CACHE_ENABLED else "disabled",
    }

@app.post("/auth/token", tags=["Auth"])
async def login(req: LoginRequest):
    if req.password != "alitkicau":
        raise HTTPException(status_code=401, detail="Password salah")
    return {"access_token": create_token(req.username), "token_type": "bearer"}

@app.post("/predict", response_model=PredictResponse, tags=["ML"])
async def predict(req: PredictRequest, username: str = Depends(verify_token)):
    validate_features(req.features)

    if CACHE_ENABLED:
        key = _cache_key(req.features)
        cached = cache.get(key)
        if cached:
            result = json.loads(cached)
            result["cached"] = True
            return PredictResponse(**result)

    result = _predict_single(req.features)

    if CACHE_ENABLED:
        cache.setex(key, 300, json.dumps(result))

    return PredictResponse(**result)

@app.post("/batch-predict", response_model=BatchPredictResponse, tags=["ML"])
async def batch_predict(req: BatchPredictRequest, username: str = Depends(verify_token)):
    for item in req.items:
        validate_features(item.features)

    results = []
    for item in req.items:
        if CACHE_ENABLED:
            key = _cache_key(item.features)
            cached = cache.get(key)
            if cached:
                r = json.loads(cached)
                r["cached"] = True
                results.append(PredictResponse(**r))
                continue

        r = _predict_single(item.features)
        if CACHE_ENABLED:
            cache.setex(key, 300, json.dumps(r))
        results.append(PredictResponse(**r))

    return BatchPredictResponse(results=results, total=len(results))