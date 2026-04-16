# MLOps-Assignment

📁 Project Structure
my-fraud-pipeline/
│
├── data/
│   ├── raw.csv
│   ├── clean.csv
│
├── create_data.py
├── clean_data.py
├── dvc.yaml
│
├── my-mlflow-experiments/
│   ├── train_models.py
│   ├── select_best_model.py
│
├── my-fraud-api/
│   ├── app.py
│   ├── generate_model.py
│   ├── test_api.py
│   ├── fraud_model.joblib
🧩 Assignment 1 — DVC Pipeline
🎯 Objective

Build a reproducible data pipeline using DVC.

🔹 Steps
Generated synthetic fraud dataset (raw.csv)
Tracked dataset using DVC
Created cleaning script → removes negative amounts
Built pipeline (dvc.yaml)
Executed pipeline using dvc repro
Visualized pipeline using dvc dag
Demonstrated pipeline re-run on data change
📌 Key Features
Data versioning with .dvc file (includes md5 hash)
Dependency tracking
Efficient pipeline execution (only changed stages rerun)
🧠 Assignment 2 — MLflow Experiment Tracking
🎯 Objective

Train multiple models and select the best one using MLflow.

🔹 Models Trained
Logistic Regression (C = 0.1)
Logistic Regression (C = 10.0)
Random Forest (n_estimators = 50)
✅ Bonus: Random Forest (n_estimators = 200)
🔹 Logged in MLflow
Hyperparameters (mlflow.log_params)
Metrics:
AUC-ROC
AUC-PR (used for selection)
Model artifacts
Registered model → my_fraud_detector
🔹 Best Model Selection
Function: select_best_model()
Criteria: Highest AUC-PR
Automatically:
Finds best run
Registers model
Moves it to Staging
📊 MLflow Features Used
Compare view
Model registry
Experiment tracking
Visualization plots (Parallel Coordinates)
⚡ Assignment 3 — FastAPI Model Serving
🎯 Objective

Deploy fraud detection model as an API.

🔹 Endpoints Implemented
✅ 1. Health Check
GET /health

Response:

{"status": "ok", "model": "RandomForest"}
✅ 2. Model Info
GET /model-info

Returns:

Model type
Version
Expected features
✅ 3. Single Prediction
POST /predict

Input:

{
  "amount": 150.0,
  "num_transactions_24h": 3,
  "distance_from_home_km": 25.0,
  "is_weekend": 0
}

Output:

{
  "fraud_probability": 0.23,
  "is_fraud": false,
  "verdict": "legit",
  "risk_level": "LOW",
  "latency_ms": 3.2
}
✅ 4. Batch Prediction
POST /predict/batch
Accepts multiple transactions
Returns predictions for all
✅ 5. BONUS — Metrics Endpoint
GET /metrics

Returns:

{
  "predictions_served": 5,
  "average_latency_ms": 3.12,
  "fraud_percentage": 40.0
}
🔹 API Features
Real-time fraud prediction
Risk classification:
LOW
MEDIUM
HIGH
Latency tracking
Batch processing
🧪 Testing
🔹 Swagger UI
http://127.0.0.1:8010/docs

Used to:

Test endpoints interactively
Validate responses
🔹 test_api.py
Sends 5 transactions to /predict
Displays formatted output:
Txn  Amount    Txns/24h  Distance    Fraud%    Verdict
------------------------------------------------------------
1    $850      9         88km        87.3      FRAUD
...
⚙️ How to Run
1. Start API
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8010
2. Run test script
python test_api.py
🏆 Key Learnings
Data versioning using DVC
Pipeline automation
Experiment tracking with MLflow
Model selection using AUC-PR
Model deployment with FastAPI
REST API development & testing
🚀 Final Outcome

A complete production-style ML pipeline:

Data → DVC → MLflow → Model Selection → API Deployment → Monitoring
