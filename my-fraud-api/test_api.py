import requests

URL = "http://127.0.0.1:8010/predict"

transactions = [
    {"amount": 850.0, "num_transactions_24h": 9, "distance_from_home_km": 88.0, "is_weekend": 1},
    {"amount": 22.0, "num_transactions_24h": 1, "distance_from_home_km": 3.0, "is_weekend": 0},
    {"amount": 430.0, "num_transactions_24h": 4, "distance_from_home_km": 60.0, "is_weekend": 0},
    {"amount": 85.0, "num_transactions_24h": 2, "distance_from_home_km": 12.0, "is_weekend": 0},
    {"amount": 1200.0, "num_transactions_24h": 12, "distance_from_home_km": 110.0, "is_weekend": 1},
]

print(f"{'Txn':<5}{'Amount':<10}{'Txns/24h':<10}{'Distance':<12}{'Fraud%':<10}{'Verdict'}")
print("-"*60)

for i, txn in enumerate(transactions, 1):
    res = requests.post(URL, json=txn).json()

    print(f"{i:<5}${txn['amount']:<9.0f}{txn['num_transactions_24h']:<10}{txn['distance_from_home_km']:<12.0f}{res['fraud_probability']*100:<10.1f}{res['verdict']}")