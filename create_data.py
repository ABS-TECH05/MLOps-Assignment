import csv
import os
import random

os.makedirs("data", exist_ok=True)

with open("data/raw.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["transaction_id", "amount", "is_fraud"])

    for i in range(1, 2001):
        amount = round(random.uniform(-1000, 5000), 2)
        is_fraud = 1 if random.random() < 0.05 else 0
        writer.writerow([i, amount, is_fraud])

print("Done")