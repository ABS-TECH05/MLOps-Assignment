import time
from contextlib import asynccontextmanager
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_ARTIFACT = joblib.load("fraud_model.joblib")
MODEL = MODEL_ARTIFACT["model"]
MODEL_TYPE = MODEL_ARTIFACT["model_type"]
MODEL_VERSION = MODEL_ARTIFACT["version"]
FEATURES = MODEL_ARTIFACT["features"]

prediction_count = 0
fraud_count = 0
total_latency_ms = 0.0


class Transaction(BaseModel):
    amount: float = Field(..., example=150.0)
    num_transactions_24h: int = Field(..., example=3)
    distance_from_home_km: float = Field(..., example=25.0)
    is_weekend: int = Field(..., example=0)

    def to_feature_list(self):
        return [
            self.amount,
            self.num_transactions_24h,
            self.distance_from_home_km,
            self.is_weekend,
        ]


class BatchRequest(BaseModel):
    transactions: List[Transaction]


def get_risk_level(prob: float) -> str:
    if prob < 0.30:
        return "LOW"
    elif prob < 0.70:
        return "MEDIUM"
    return "HIGH"


def predict_one(txn: Transaction):
    global prediction_count, fraud_count, total_latency_ms

    start = time.perf_counter()

    X = np.array([txn.to_feature_list()], dtype=np.float32)
    fraud_prob = float(MODEL.predict_proba(X)[0][1])
    is_fraud = fraud_prob >= 0.50
    risk_level = get_risk_level(fraud_prob)

    latency_ms = (time.perf_counter() - start) * 1000.0

    prediction_count += 1
    total_latency_ms += latency_ms
    if is_fraud:
        fraud_count += 1

    return {
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud": bool(is_fraud),
        "verdict": "FRAUD" if is_fraud else "legit",
        "risk_level": risk_level,
        "latency_ms": round(latency_ms, 2),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_TYPE}


@app.get("/model-info")
def model_info():
    return {
        "model_type": MODEL_TYPE,
        "version": MODEL_VERSION,
        "features": FEATURES,
    }


@app.post("/predict")
def predict(txn: Transaction):
    return predict_one(txn)


@app.post("/predict/batch")
def predict_batch(batch: BatchRequest):
    return {
        "count": len(batch.transactions),
        "predictions": [predict_one(txn) for txn in batch.transactions],
    }


@app.get("/metrics")
def metrics():
    avg_latency = total_latency_ms / prediction_count if prediction_count else 0.0
    fraud_percentage = (fraud_count / prediction_count * 100.0) if prediction_count else 0.0

    return {
        "predictions_served": prediction_count,
        "average_latency_ms": round(avg_latency, 2),
        "fraud_percentage": round(fraud_percentage, 2),
    }