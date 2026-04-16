import os
import warnings

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DB_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "my_fraud_models"
REGISTERED_MODEL_NAME = "my_fraud_detector"


def generate_data():
    X, y = make_classification(
        n_samples=4000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        n_classes=2,
        weights=[0.95, 0.05],
        flip_y=0.01,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_and_log_model(model_name, model, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc_roc = roc_auc_score(y_test, y_proba)
        auc_pr = average_precision_score(y_test, y_proba)

        mlflow.log_metrics({
            "auc_roc": auc_roc,
            "auc_pr": auc_pr
        })

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"{model_name} logged | AUC-ROC={auc_roc:.4f} | AUC-PR={auc_pr:.4f}")


def select_best_model():
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print("Experiment not found.")
        return

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.auc_pr DESC"],
    )

    if runs.empty:
        print("No runs found.")
        return

    winner = runs.iloc[0]

    run_id = winner["run_id"]
    auc_pr = winner["metrics.auc_pr"]
    auc_roc = winner["metrics.auc_roc"]

    latest_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")

    winner_version = None
    for mv in latest_versions:
        if mv.run_id == run_id:
            winner_version = mv.version
            break

    if winner_version is None:
        print("No registered model version found for winning run.")
        return

    # Archive any version already in Staging
    for mv in latest_versions:
        current_stage = getattr(mv, "current_stage", None)
        if current_stage == "Staging" and mv.version != winner_version:
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=mv.version,
                stage="Archived"
            )

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=winner_version,
        stage="Staging"
    )

    print("\n" + "="*40)
    print("Best Model")
    print("="*40)
    print(f"Run ID      : {run_id}")
    print(f"AUC-PR      : {auc_pr:.4f}")
    print(f"AUC-ROC     : {auc_roc:.4f}")
    print(f"Model       : {REGISTERED_MODEL_NAME}")
    print(f"Version     : {winner_version}")
    print("Stage       : STAGING")
    print("="*40 + "\n")
    


def main():
    mlflow.set_tracking_uri(DB_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = generate_data()

    candidates = [
    (
        "logreg_c_0_1",
        LogisticRegression(C=0.1, max_iter=1000, solver="liblinear", random_state=42),
        {"model_type": "LogisticRegression", "C": 0.1},
    ),
    (
        "logreg_c_10_0",
        LogisticRegression(C=10.0, max_iter=1000, solver="liblinear", random_state=42),
        {"model_type": "LogisticRegression", "C": 10.0},
    ),
    (
        "rf_50",
        RandomForestClassifier(n_estimators=50, random_state=42),
        {"model_type": "RandomForest", "n_estimators": 50},
    ),
    (
        "rf_200",
        RandomForestClassifier(n_estimators=200, random_state=42),
        {"model_type": "RandomForest", "n_estimators": 200},
    ),
]

    for model_name, model, params in candidates:
        train_and_log_model(model_name, model, params, X_train, X_test, y_train, y_test)

    select_best_model()


if __name__ == "__main__":
    main()