"""
Step 1: Synthetic Payment Data Generator
---------------------------------------
This script creates realistic fake payment transactions
for Card, ACH, and Wallet.

We will use this data later for:
- Real-time dashboard
- Fraud detection model
- GenAI features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid

# -----------------------------
# Settings (you can change these)
# -----------------------------
NUM_TRANSACTIONS = 5000          # How many transactions to create
FRAUD_RATE = 0.015               # 1.5% of transactions will be fraud
START_DATE = datetime(2026, 9, 1)

# Possible values
PAYMENT_METHODS = ["Card", "ACH", "Wallet"]
REGIONS = ["Western Europe", "Northern Europe", "Southern Europe", "Eastern Europe", "North America"]
MERCHANTS = [
    "Amazon", "Shopify Store", "Uber", "Netflix", "Spotify",
    "Local Coffee Shop", "Electronics Hub", "Fashion Outlet",
    "Grocery Mart", "Travel Booking", "GameStore", "Pharmacy Plus"
]
CURRENCIES = ["EUR", "USD"]

def generate_transactions(n=NUM_TRANSACTIONS):
    data = []

    for i in range(n):
        # Basic transaction info
        transaction_id = str(uuid.uuid4())[:8]
        timestamp = START_DATE + timedelta(minutes=random.randint(0, 60*24*20))  # 20 days
        payment_method = random.choice(PAYMENT_METHODS)
        amount = round(random.uniform(5, 800), 2)
        currency = random.choice(CURRENCIES)
        merchant = random.choice(MERCHANTS)
        region = random.choice(REGIONS)
        customer_id = f"CUST-{random.randint(1000, 9999)}"

        # Decide if this transaction is fraudulent
        is_fraud = 1 if random.random() < FRAUD_RATE else 0

        # Make fraud transactions look a bit different
        if is_fraud:
            amount = round(random.uniform(300, 2500), 2)  # higher amounts
            # sometimes unusual region or merchant patterns can be added later

        data.append({
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "payment_method": payment_method,
            "amount": amount,
            "currency": currency,
            "merchant": merchant,
            "region": region,
            "customer_id": customer_id,
            "is_fraud": is_fraud
        })

    df = pd.DataFrame(data)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Generating synthetic payment data...")
    df = generate_transactions()

    # Save to CSV
    output_file = "payments_data.csv"
    df.to_csv(output_file, index=False)

    print(f"Done! Created {len(df)} transactions")
    print(f"Saved to: {output_file}")
    print("\nFraud distribution:")
    print(df["is_fraud"].value_counts())
    print("\nSample of the data:")
    print(df.head(8))
