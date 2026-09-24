"""
Step 3: Dashboard + Simple Fraud Risk Scoring
---------------------------------------------
New features:
- Train a simple fraud detection model
- Calculate a risk score (0-100) for every transaction
- Show high-risk transactions
- Color-code the risk
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
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
st.markdown("Prototype – Step 3")

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
# Simple Fraud Model
# -----------------------------
@st.cache_resource
def train_fraud_model(data):
    """
    Train a very simple Random Forest model.
    In a real system this would be much more advanced
    and served via Azure ML Online Endpoint.
    """
    df_model = data.copy()

    # Encode categorical columns
    le_method = LabelEncoder()
    le_region = LabelEncoder()
    le_merchant = LabelEncoder()

    df_model["payment_method_enc"] = le_method.fit_transform(df_model["payment_method"])
    df_model["region_enc"] = le_region.fit_transform(df_model["region"])
    df_model["merchant_enc"] = le_merchant.fit_transform(df_model["merchant"])

    features = ["amount", "payment_method_enc", "region_enc", "merchant_enc"]
    X = df_model[features]
    y = df_model["is_fraud"]

    # Train a simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=6)
    model.fit(X, y)

    return model, le_method, le_region, le_merchant, features

model, le_method, le_region, le_merchant, features = train_fraud_model(df)

# Calculate risk scores for all transactions
df_score = df.copy()
df_score["payment_method_enc"] = le_method.transform(df_score["payment_method"])
df_score["region_enc"] = le_region.transform(df_score["region"])
df_score["merchant_enc"] = le_merchant.transform(df_score["merchant"])

# Probability of fraud * 100 → risk score
df_score["risk_score"] = (model.predict_proba(df_score[features])[:, 1] * 100).round(1)

# -----------------------------
# Key Metrics
# -----------------------------
st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

total_transactions = len(df_score)
total_value = df_score["amount"].sum()
high_risk_count = len(df_score[df_score["risk_score"] >= 70])
avg_risk = df_score["risk_score"].mean()

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Total Value", f"€{total_value:,.0f}")
col3.metric("High Risk (≥70)", f"{high_risk_count}")
col4.metric("Average Risk Score", f"{avg_risk:.1f}")

st.markdown("---")

# -----------------------------
# Charts
# -----------------------------
st.subheader("Overview")

col_left, col_right = st.columns(2)

with col_left:
    method_counts = df_score["payment_method"].value_counts().reset_index()
    method_counts.columns = ["Payment Method", "Count"]
    fig1 = px.pie(method_counts, names="Payment Method", values="Count",
                  title="Transactions by Payment Method")
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    region_counts = df_score["region"].value_counts().reset_index()
    region_counts.columns = ["Region", "Count"]
    fig2 = px.bar(region_counts, x="Region", y="Count",
                  title="Transactions by Region", color="Count",
                  color_continuous_scale="Blues")
    fig2.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# -----------------------------
# High Risk Transactions
# -----------------------------
st.subheader("🚨 High Risk Transactions (Risk Score ≥ 70)")

high_risk = df_score[df_score["risk_score"] >= 70].sort_values("risk_score", ascending=False)

if len(high_risk) > 0:
    st.dataframe(
        high_risk[["transaction_id", "timestamp", "payment_method", "amount", 
                   "merchant", "region", "risk_score", "is_fraud"]].head(15),
        use_container_width=True
    )
else:
    st.info("No high-risk transactions found with the current threshold.")

st.markdown("---")

# -----------------------------
# All Recent Transactions with Risk Score
# -----------------------------
st.subheader("Recent Transactions with Risk Score")

recent = df_score.sort_values("timestamp", ascending=False).head(25)

def risk_color(score):
    if score >= 70:
        return "🔴 High"
    elif score >= 40:
        return "🟠 Medium"
    else:
        return "🟢 Low"

recent = recent.copy()
recent["Risk Level"] = recent["risk_score"].apply(risk_color)

st.dataframe(
    recent[["transaction_id", "timestamp", "payment_method", "amount", 
            "merchant", "region", "risk_score", "Risk Level"]],
    use_container_width=True
)

st.caption("Risk Score is generated by a simple Random Forest model. In production this would be an Azure ML Online Endpoint.")
