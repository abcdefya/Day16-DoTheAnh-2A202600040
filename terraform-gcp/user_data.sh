#!/bin/bash
set -e

# Install Python and ML packages
apt-get update -y
apt-get install -y python3 python3-pip python3-venv curl

python3 -m pip install --upgrade pip
pip3 install lightgbm scikit-learn pandas numpy fastapi uvicorn

# Create working directory and benchmark script
mkdir -p /root/ml-benchmark
cat > /root/ml-benchmark/benchmark.py << 'PYEOF'
import time
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb

print("=== LightGBM CPU Benchmark on GCP n2-standard-8 ===\n")

# Generate synthetic dataset similar to Credit Card Fraud (284807 rows, imbalanced)
print("Generating dataset (284807 rows, 30 features, imbalance 0.17%)...")
t0 = time.time()
X, y = make_classification(
    n_samples=284807,
    n_features=30,
    n_informative=20,
    n_redundant=5,
    weights=[0.9983, 0.0017],
    flip_y=0,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
load_time = time.time() - t0
print(f"Data ready in {load_time:.2f}s — train: {len(X_train)}, test: {len(X_test)}\n")

# Train LightGBM
print("Training LightGBM...")
t1 = time.time()
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": int((y == 0).sum() / (y == 1).sum()),
    "verbose": -1,
    "n_jobs": -1,
}
callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)]
model = lgb.train(params, train_data, num_boost_round=500,
                  valid_sets=[valid_data], callbacks=callbacks)
train_time = time.time() - t1
print(f"\nTraining done in {train_time:.2f}s — best iteration: {model.best_iteration}\n")

# Evaluate
y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int)
auc    = roc_auc_score(y_test, y_prob)
acc    = accuracy_score(y_test, y_pred)
f1     = f1_score(y_test, y_pred)
prec   = precision_score(y_test, y_pred)
rec    = recall_score(y_test, y_pred)

# Inference latency
t2 = time.time()
for _ in range(1000):
    model.predict(X_test[:1])
lat_1row = (time.time() - t2) / 1000 * 1000  # ms

t3 = time.time()
model.predict(X_test[:1000])
tput_1000 = (time.time() - t3) * 1000  # ms

results = {
    "load_data_sec":       round(load_time, 3),
    "training_time_sec":   round(train_time, 3),
    "best_iteration":      model.best_iteration,
    "auc_roc":             round(auc, 6),
    "accuracy":            round(acc, 6),
    "f1_score":            round(f1, 6),
    "precision":           round(prec, 6),
    "recall":              round(rec, 6),
    "inference_latency_1row_ms":      round(lat_1row, 4),
    "inference_latency_1000rows_ms":  round(tput_1000, 4),
}

print("=== Results ===")
for k, v in results.items():
    print(f"  {k}: {v}")

with open("/root/ml-benchmark/benchmark_result.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to /root/ml-benchmark/benchmark_result.json")

# Persist model for inference server
model.save_model("/root/ml-benchmark/lgbm_model.txt")
print("Model saved to /root/ml-benchmark/lgbm_model.txt")
PYEOF

# Create lightweight FastAPI inference server (keeps port 8000 alive for LB health check)
mkdir -p /opt/lgbm-server
cat > /opt/lgbm-server/server.py << 'PYEOF'
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import lightgbm as lgb
import numpy as np
import os

app = FastAPI(title="LightGBM Inference Server")
MODEL_PATH = "/root/ml-benchmark/lgbm_model.txt"
model = lgb.Booster(model_file=MODEL_PATH) if os.path.exists(MODEL_PATH) else None

class PredictRequest(BaseModel):
    features: List[List[float]]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/v1/predict")
def predict(req: PredictRequest):
    if model is None:
        return {"error": "Model not loaded. Run benchmark.py first."}
    arr = np.array(req.features)
    probs = model.predict(arr).tolist()
    return {"predictions": probs}
PYEOF

# Systemd service: start simple health-check HTTP server immediately,
# upgrade to full model server after benchmark.py runs
cat > /etc/systemd/system/lgbm-server.service << 'EOF'
[Unit]
Description=LightGBM Inference Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/lgbm-server
ExecStart=/usr/bin/python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable lgbm-server
systemctl start lgbm-server

echo "=== CPU instance setup complete ==="
echo "Run benchmark: python3 /root/ml-benchmark/benchmark.py"
echo "Inference API: http://localhost:8000/v1/predict"
