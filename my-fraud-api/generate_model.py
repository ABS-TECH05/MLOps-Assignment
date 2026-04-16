import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "amount",
    "num_transactions_24h",
    "distance_from_home_km",
    "is_weekend",
]

MODEL_VERSION = "1.0.0"
MODEL_TYPE = "RandomForest"


def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42):
    rng = np.random.default_rng(random_state)

    amount = rng.uniform(1, 2000, n_samples)
    num_transactions_24h = rng.integers(0, 15, n_samples)
    distance_from_home_km = rng.uniform(0, 150, n_samples)
    is_weekend = rng.integers(0, 2, n_samples)

    X = np.column_stack([
        amount,
        num_transactions_24h,
        distance_from_home_km,
        is_weekend,
    ])

    risk_score = (
        (amount > 500).astype(int)
        + (num_transactions_24h > 5).astype(int)
        + (distance_from_home_km > 50).astype(int)
        + (is_weekend == 1).astype(int)
    )

    fraud_probability = np.clip(0.05 + 0.22 * risk_score, 0, 0.98)
    y = (rng.random(n_samples) < fraud_probability).astype(int)

    return X, y


def main():
    X, y = generate_synthetic_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X, y)

    artifact = {
        "model": model,
        "model_type": MODEL_TYPE,
        "version": MODEL_VERSION,
        "features": FEATURES,
    }

    joblib.dump(artifact, "fraud_model.joblib")
    print("Saved fraud_model.joblib")


if __name__ == "__main__":
    main()