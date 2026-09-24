"""
Step 2: First Streamlit Dashboard
---------------------------------
This is the beginning of our prototype.
It loads the synthetic payment data and shows:
- Key numbers (KPIs)
- Charts by payment method and region
- A table of recent transactions
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Payments Real-Time Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Payments Real-Time Dashboard")
st.markdown("Prototype – Step 2: Basic Dashboard")

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
# Key Metrics (KPIs)
# -----------------------------
st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

total_transactions = len(df)
total_value = df["amount"].sum()
fraud_count = df["is_fraud"].sum()
fraud_rate = (fraud_count / total_transactions) * 100

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Total Value", f"€{total_value:,.0f}")
col3.metric("Fraud Transactions", f"{fraud_count}")
col4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

st.markdown("---")

# -----------------------------
# Charts
# -----------------------------
st.subheader("Breakdown")

col_left, col_right = st.columns(2)

with col_left:
    # Transactions by Payment Method
    method_counts = df["payment_method"].value_counts().reset_index()
    method_counts.columns = ["Payment Method", "Count"]
    
    fig1 = px.pie(
        method_counts,
        names="Payment Method",
        values="Count",
        title="Transactions by Payment Method",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    # Transactions by Region
    region_counts = df["region"].value_counts().reset_index()
    region_counts.columns = ["Region", "Count"]
    
    fig2 = px.bar(
        region_counts,
        x="Region",
        y="Count",
        title="Transactions by Region",
        color="Count",
        color_continuous_scale="Blues"
    )
    fig2.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# -----------------------------
# Recent Transactions Table
# -----------------------------
st.subheader("Recent Transactions")

# Show the latest 20 transactions
recent = df.sort_values("timestamp", ascending=False).head(20)
st.dataframe(
    recent[["transaction_id", "timestamp", "payment_method", "amount", "merchant", "region", "is_fraud"]],
    use_container_width=True
)

st.caption("This is a static view for now. In the next steps we will make it feel more real-time and add fraud scoring + AI.")
