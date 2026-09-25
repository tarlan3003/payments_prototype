"""
Final Prototype
- Business metrics on full data
- Fraud model with proper train/test split
- Charts (restored)
- Real OpenAI for Dispute Summary + Chat
- Chatbot can answer both specific transaction questions AND general dataset questions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI
import os

# -----------------------------
# Load API Key securely
# -----------------------------
def load_api_key():
    key_file = "api_key.txt"
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            key = f.read().strip()
            if key and key != "PASTE_YOUR_OPENAI_API_KEY_HERE":
                return key
    return os.getenv("OPENAI_API_KEY")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Payments Dashboard + Fraud + GenAI",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Payments Real-Time Prototype")
st.markdown("**Dashboard + Fraud Scoring + OpenAI GenAI**")

# -----------------------------
# OpenAI client
# -----------------------------
api_key = load_api_key()
client = OpenAI(api_key=api_key) if api_key else None

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("payments_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

# -----------------------------
# Train fraud model
# -----------------------------
@st.cache_resource
def train_fraud_model(data):
    df_model = data.copy()

    le_method = LabelEncoder()
    le_region = LabelEncoder()
    le_merchant = LabelEncoder()

    df_model["payment_method_enc"] = le_method.fit_transform(df_model["payment_method"])
    df_model["region_enc"] = le_region.fit_transform(df_model["region"])
    df_model["merchant_enc"] = le_merchant.fit_transform(df_model["merchant"])

    features = ["amount", "payment_method_enc", "region_enc", "merchant_enc"]
    X = df_model[features]
    y = df_model["is_fraud"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_model.index, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=80, max_depth=7, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    return model, le_method, le_region, le_merchant, features, idx_test

model, le_method, le_region, le_merchant, features, test_indices = train_fraud_model(df)

# Score test set
df_test = df.loc[test_indices].copy().reset_index(drop=True)
df_test["payment_method_enc"] = le_method.transform(df_test["payment_method"])
df_test["region_enc"] = le_region.transform(df_test["region"])
df_test["merchant_enc"] = le_merchant.transform(df_test["merchant"])
df_test["risk_score"] = (model.predict_proba(df_test[features])[:, 1] * 100).round(1)

df_test["label"] = df_test.apply(
    lambda r: f"{r['transaction_id']} | {r['merchant']} | €{r['amount']:.2f} | Risk {r['risk_score']}",
    axis=1
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("GenAI Assistant")

selected_label = st.sidebar.selectbox(
    "Select a transaction:",
    options=df_test["label"].tolist()
)

selected_row = df_test[df_test["label"] == selected_label].iloc[0]

st.sidebar.markdown("---")
st.sidebar.markdown("### Selected Transaction")
st.sidebar.write(f"**ID:** {selected_row['transaction_id']}")
st.sidebar.write(f"**Merchant:** {selected_row['merchant']}")
st.sidebar.write(f"**Amount:** €{selected_row['amount']:.2f}")
st.sidebar.write(f"**Method:** {selected_row['payment_method']}")
st.sidebar.write(f"**Region:** {selected_row['region']}")
st.sidebar.write(f"**Risk Score:** {selected_row['risk_score']}")
st.sidebar.write(f"**Actual Fraud:** {'Yes' if selected_row['is_fraud'] == 1 else 'No'}")

# -----------------------------
# OpenAI helpers
# -----------------------------
def call_openai(system_prompt: str, user_prompt: str) -> str:
    if client is None:
        return "⚠️ OpenAI API key not found. Please put your key inside **api_key.txt**"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=700
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling OpenAI: {str(e)}"


def generate_dispute_summary(row) -> str:
    system_prompt = """You are an expert assistant working for a global digital payments company.
Write clear, professional, and neutral dispute summaries.
Do not invent information."""

    user_prompt = f"""
Write a concise dispute summary for this transaction:

Transaction ID: {row['transaction_id']}
Timestamp: {row['timestamp']}
Customer ID: {row['customer_id']}
Payment Method: {row['payment_method']}
Merchant: {row['merchant']}
Amount: €{row['amount']:.2f} {row['currency']}
Region: {row['region']}
Risk Score (0-100): {row['risk_score']}
Actual Fraud Label: {'Yes' if row['is_fraud'] == 1 else 'No'}

Use these sections:
1. Transaction Details
2. Risk Assessment
3. Summary
4. Recommended Action
"""
    return call_openai(system_prompt, user_prompt)


def generate_chat_response(question: str, row, full_df) -> str:
    """Can answer both specific transaction questions and general dataset questions."""

    # Basic dataset statistics for general questions
    total_tx = len(full_df)
    total_value = full_df["amount"].sum()
    fraud_count = full_df["is_fraud"].sum()
    fraud_rate = (fraud_count / total_tx) * 100
    avg_amount = full_df["amount"].mean()
    methods = full_df["payment_method"].value_counts().to_dict()
    regions = full_df["region"].value_counts().to_dict()

    system_prompt = """You are a helpful AI assistant for a payments company.
You can answer two types of questions:
1. Questions about the currently selected transaction
2. General questions about the overall dataset

Answer clearly and based only on the provided information.
If the question is general, use the dataset statistics.
If the question is about the selected transaction, use the transaction data.
"""

    user_prompt = f"""
=== SELECTED TRANSACTION ===
Transaction ID: {row['transaction_id']}
Timestamp: {row['timestamp']}
Customer ID: {row['customer_id']}
Payment Method: {row['payment_method']}
Merchant: {row['merchant']}
Amount: €{row['amount']:.2f} {row['currency']}
Region: {row['region']}
Risk Score: {row['risk_score']}
Actual Fraud Label: {'Yes' if row['is_fraud'] == 1 else 'No'}

=== FULL DATASET STATISTICS ===
Total Transactions: {total_tx:,}
Total Value: €{total_value:,.2f}
Actual Fraud Count: {fraud_count}
Fraud Rate: {fraud_rate:.2f}%
Average Amount: €{avg_amount:.2f}
Payment Methods distribution: {methods}
Regions distribution: {regions}

User question: {question}

Answer the question helpfully.
"""
    return call_openai(system_prompt, user_prompt)

# -----------------------------
# Business Metrics
# -----------------------------
st.subheader("Business Metrics (Full Dataset)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Total Value", f"€{df['amount'].sum():,.0f}")
col3.metric("Actual Fraud Count", f"{df['is_fraud'].sum()}")
col4.metric("Actual Fraud Rate", f"{(df['is_fraud'].sum() / len(df) * 100):.2f}%")

st.markdown("---")

# -----------------------------
# Charts 
# -----------------------------
st.subheader("Overview")

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
# GenAI Section
# -----------------------------
st.subheader("GenAI Features (OpenAI gpt-4o-mini)")

if client is None:
    st.warning("⚠️ OpenAI API key not found. Please put your key inside **api_key.txt**")
else:
    st.success("OpenAI client is ready")

tab1, tab2 = st.tabs(["Dispute Summary", "AI Assistant Chat"])

with tab1:
    st.markdown("### Automatic Dispute Summary")
    if st.button("Generate Dispute Summary", type="primary"):
        with st.spinner("Generating summary with OpenAI..."):
            summary = generate_dispute_summary(selected_row)
        st.markdown(summary)

with tab2:
    st.markdown("### AI Assistant")
    st.markdown("You can ask about the **selected transaction** or about the **overall dataset**.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the transaction or the whole dataset..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_chat_response(prompt, selected_row, df)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")

# -----------------------------
# High Risk Table
# -----------------------------
st.subheader("High Risk Transactions (Test Set – Score ≥ 70)")
high_risk = df_test[df_test["risk_score"] >= 70].sort_values("risk_score", ascending=False)

if len(high_risk) > 0:
    st.dataframe(
        high_risk[["transaction_id", "timestamp", "payment_method", "amount",
                   "merchant", "region", "risk_score", "is_fraud"]].head(15),
        use_container_width=True
    )
else:
    st.info("No high-risk transactions found in the test set.")

st.caption("Random Forest (train/test split) + OpenAI gpt-4o-mini | Key loaded from api_key.txt")
