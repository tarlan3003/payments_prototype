"""
Step 3 (Final Correct Version)
------------------------------
- Business metrics (KPIs) → calculated on the FULL dataset
- Fraud model → trained only on training data
- Risk scores → calculated ONLY on unseen test data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Payments Dashboard + Fraud Scoring",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Payments Dashboard + Fraud Risk Scoring")
st.markdown("**Prototype – Step 3 (Correct version)**")

# -----------------------------
# Load the data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("payments_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

# -----------------------------
# Train model with proper split
# -----------------------------
@st.cache_resource
def train_fraud_model(data):
    df_model = data.copy()

    # Encode categorical features
    le_method = LabelEncoder()
    le_region = LabelEncoder()
    le_merchant = LabelEncoder()

    df_model["payment_method_enc"] = le_method.fit_transform(df_model["payment_method"])
    df_model["region_enc"] = le_region.fit_transform(df_model["region"])
    df_model["merchant_enc"] = le_merchant.fit_transform(df_model["merchant"])

    features = ["amount", "payment_method_enc", "region_enc", "merchant_enc"]
    X = df_model[features]
    y = df_model["is_fraud"]

    # 75% train / 25% test + keep the original indices
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_model.index,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=7,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, le_method, le_region, le_merchant, features, idx_test

model, le_method, le_region, le_merchant, features, test_indices = train_fraud_model(df)

# -----------------------------
# Create scored test set (unseen data only)
# -----------------------------
df_test = df.loc[test_indices].copy()
df_test["payment_method_enc"] = le_method.transform(df_test["payment_method"])
df_test["region_enc"] = le_region.transform(df_test["region"])
df_test["merchant_enc"] = le_merchant.transform(df_test["merchant"])

df_test["risk_score"] = (model.predict_proba(df_test[features])[:, 1] * 100).round(1)

# -----------------------------
# 1. Business Metrics → FULL dataset
# -----------------------------
st.subheader("Business Metrics (Full Dataset)")

col1, col2, col3, col4 = st.columns(4)

total_transactions = len(df)
total_value = df["amount"].sum()
actual_fraud_count = df["is_fraud"].sum()
actual_fraud_rate = (actual_fraud_count / total_transactions) * 100

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Total Value", f"€{total_value:,.0f}")
col3.metric("Actual Fraud Count", f"{actual_fraud_count}")
col4.metric("Actual Fraud Rate", f"{actual_fraud_rate:.2f}%")

st.markdown("---")

# -----------------------------
# 2. Model Metrics → TEST set only
# -----------------------------
st.subheader("Model Risk Scoring (Unseen Test Data Only)")

col1, col2, col3 = st.columns(3)

test_size = len(df_test)
high_risk_count = len(df_test[df_test["risk_score"] >= 70])
avg_risk = df_test["risk_score"].mean()

col1.metric("Test Set Size", f"{test_size:,}")
col2.metric("High Risk (≥70)", f"{high_risk_count}")
col3.metric("Average Risk Score", f"{avg_risk:.1f}")

st.info("Risk scores are calculated only on data the model has never seen (25% test set).")

st.markdown("---")

# -----------------------------
# Charts (Full data)
# -----------------------------
st.subheader("Overview (Full Dataset)")

col_left, col_right = st.columns(2)

with col_left:
    method_counts = df["payment_method"].value_counts().reset_index()
    method_counts.columns = ["Payment Method", "Count"]
    fig1 = px.pie(method_counts, names="Payment Method", values="Count",
                  title="Transactions by Payment Method")
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    region_counts = df["region"].value_counts().reset_index()
    region_counts.columns = ["Region", "Count"]
    fig2 = px.bar(region_counts, x="Region", y="Count",
                  title="Transactions by Region", color="Count",
                  color_continuous_scale="Blues")
    fig2.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# -----------------------------
# High Risk Transactions (from test set only)
# -----------------------------
st.subheader("High Risk Transactions (Risk Score ≥ 70) — Test Set Only")

high_risk = df_test[df_test["risk_score"] >= 70].sort_values("risk_score", ascending=False)

if len(high_risk) > 0:
    st.dataframe(
        high_risk[["transaction_id", "timestamp", "payment_method", "amount",
                   "merchant", "region", "risk_score", "is_fraud"]].head(20),
        use_container_width=True
    )
else:
    st.info("No transactions with risk score ≥ 70 in the test set.")

st.markdown("---")

# -----------------------------
# Sample of scored test transactions
# -----------------------------
st.subheader("Sample of Scored Transactions (Test Set)")

sample = df_test.sort_values("timestamp", ascending=False).head(30).copy()

def risk_level(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"

sample["Risk Level"] = sample["risk_score"].apply(risk_level)

st.dataframe(
    sample[["transaction_id", "timestamp", "payment_method", "amount",
            "merchant", "region", "risk_score", "Risk Level", "is_fraud"]],
    use_container_width=True
)

st.caption("Model: Random Forest | Train/Test split 75/25 | Risk scores only on unseen data")
