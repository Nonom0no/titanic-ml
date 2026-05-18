const express = require("express");
const axios = require("axios");
const router = express.Router();
const auth = require("../middleware/auth");

const ML_URL = process.env.ML_SERVICE_URL;

// Token ML service disimpan di memory (refresh saat expired)
let mlToken = null;
let mlTokenExpiry = null;

async function getMLToken() {
  if (mlToken && mlTokenExpiry > Date.now()) return mlToken;
  const res = await axios.post(`${ML_URL}/auth/token`, {
    username: process.env.ML_SERVICE_USER,
    password: process.env.ML_SERVICE_PASS,
  });
  mlToken = res.data.access_token;
  mlTokenExpiry = Date.now() + 55 * 60 * 1000; // 55 menit
  return mlToken;
}

async function callML(endpoint, data) {
  const token = await getMLToken();
  return axios.post(`${ML_URL}${endpoint}`, data, {
    headers: { Authorization: `Bearer ${token}` },
    timeout: 5000,
  });
}

// Circuit breaker sederhana
let mlFailCount = 0;
let mlCircuitOpen = false;
let mlCircuitResetAt = null;

function checkCircuit() {
  if (mlCircuitOpen && Date.now() > mlCircuitResetAt) {
    mlCircuitOpen = false;
    mlFailCount = 0;
  }
  if (mlCircuitOpen) throw new Error("CIRCUIT_OPEN");
}

function recordFailure() {
  mlFailCount++;
  if (mlFailCount >= 3) {
    mlCircuitOpen = true;
    mlCircuitResetAt = Date.now() + 30000; // reset 30 detik
  }
}

function recordSuccess() {
  mlFailCount = 0;
  mlCircuitOpen = false;
}

// POST /api/predict
router.post("/predict", auth, async (req, res) => {
  const { features, user_id } = req.body;
  if (!features || !Array.isArray(features)) {
    return res.status(400).json({ error: "Field 'features' (array) wajib ada" });
  }

  try {
    checkCircuit();
    const mlRes = await callML("/predict", { features, user_id });
    recordSuccess();
    res.json({
      user: req.user.sub,
      result: mlRes.data,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    if (err.message === "CIRCUIT_OPEN") {
      return res.status(503).json({ error: "ML Service sedang tidak tersedia (circuit open)" });
    }
    if (err.code === "ECONNREFUSED" || err.code === "ECONNABORTED") {
      recordFailure();
      return res.status(503).json({ error: "ML Service tidak dapat dihubungi" });
    }
    recordFailure();
    res.status(500).json({ error: err.response?.data?.detail || err.message });
  }
});

// POST /api/batch-predict
router.post("/batch-predict", auth, async (req, res) => {
  const { items } = req.body;
  if (!items || !Array.isArray(items)) {
    return res.status(400).json({ error: "Field 'items' (array) wajib ada" });
  }

  try {
    checkCircuit();
    const mlRes = await callML("/batch-predict", { items });
    recordSuccess();
    res.json({
      user: req.user.sub,
      ...mlRes.data,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    if (err.message === "CIRCUIT_OPEN") {
      return res.status(503).json({ error: "ML Service sedang tidak tersedia (circuit open)" });
    }
    if (err.code === "ECONNREFUSED" || err.code === "ECONNABORTED") {
      recordFailure();
      return res.status(503).json({ error: "ML Service tidak dapat dihubungi" });
    }
    recordFailure();
    res.status(500).json({ error: err.response?.data?.detail || err.message });
  }
});

// GET /api/ml-health - cek status ML service
router.get("/ml-health", auth, async (req, res) => {
  try {
    const r = await axios.get(`${ML_URL}/health`, { timeout: 3000 });
    res.json(r.data);
  } catch {
    res.status(503).json({ error: "ML Service tidak tersedia" });
  }
});

module.exports = router;