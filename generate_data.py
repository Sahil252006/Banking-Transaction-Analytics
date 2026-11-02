
import os
import random
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)

os.makedirs("data", exist_ok=True)

transaction_types = ["Deposit", "Withdrawal", "Transfer", "Payment"]
channels = ["ATM", "Mobile Banking", "Internet Banking", "Branch"]
statuses = ["Success", "Failed", "Pending"]
locations = ["Mumbai", "Pune", "Delhi", "Bengaluru", "Hyderabad"]

records = []

start_date = datetime(2026, 1, 1)

for i in range(1, 5001):
    transaction_date = start_date + timedelta(
        days=random.randint(0, 300),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )

    records.append({
        "transaction_id": f"TXN{i:06d}",
        "customer_id": f"CUST{random.randint(1, 500):04d}",
        "transaction_date": transaction_date.strftime("%Y-%m-%d %H:%M:%S"),
        "amount": round(random.uniform(100, 25000), 2),
        "transaction_type": random.choice(transaction_types),
        "channel": random.choice(channels),
        "location": random.choice(locations),
        "status": random.choices(
            statuses,
            weights=[85, 10, 5]
        )[0]
    })

# Add some deliberately high-value transactions for testing
for i in range(10):
    records.append({
        "transaction_id": f"HIGH{i + 1:03d}",
        "customer_id": f"CUST{100 + i:04d}",
        "transaction_date": "2026-10-20 12:30:00",
        "amount": round(random.uniform(60000, 100000), 2),
        "transaction_type": "Transfer",
        "channel": "Internet Banking",
        "location": "Mumbai",
        "status": "Success"
    })

df = pd.DataFrame(records)

df.to_csv("data/banking_transactions.csv", index=False)

print("Dataset generated successfully!")
print("Total records:", len(df))
print(df.head())
