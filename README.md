# Real-Time Payments Analytics & Fraud Detection Prototype

Azure-inspired end-to-end prototype for real-time payment analytics, fraud risk scoring, and GenAI-powered assistance.

Built as a case study solution for a global digital payments provider handling **Card**, **ACH**, and **Wallet** transactions.

---

## Project Overview

This prototype demonstrates:

- Real-time style dashboard (volume, value, by merchant & region)
- Fraud risk scoring using a Machine Learning model
- Automatic dispute summary generation
- AI chatbot that can answer questions about individual transactions and the overall dataset

The solution is designed to map closely to a real **Microsoft Azure** architecture.

---

## Architecture Mapping (Prototype → Azure)

| Prototype Component              | Azure Service                          |
|----------------------------------|----------------------------------------|
| Synthetic payment stream         | Azure Event Hubs                       |
| Aggregations & Dashboard         | Azure Stream Analytics + Power BI      |
| Fraud Model                      | Azure Machine Learning Online Endpoint |
| Risk Scores storage              | Azure Cosmos DB                        |
| Historical data                  | Azure Data Lake Storage Gen2           |
| GenAI (Chat + Dispute Summary)   | Azure OpenAI + Azure AI Search (RAG)   |

---

## Features

- Interactive dashboard with key business metrics
- Fraud risk score (0–100) for every transaction
- Model performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- High-risk transaction monitoring
- AI-powered Dispute Summary generation
- AI Chatbot (supports both transaction-specific and dataset-level questions)
- Secure API key handling (`api_key.txt` is git-ignored)

---

## Tech Stack

- **Python**
- **Streamlit** – Interactive web app
- **Pandas & NumPy** – Data processing
- **Scikit-learn** – Fraud detection model (Random Forest)
- **Plotly** – Charts
- **OpenAI API** (`gpt-4o-mini`) – GenAI features

---

## Project Structure

```bash
payments_prototype/
├── app.py                 # Main Streamlit application
├── generate_data.py       # Synthetic data generator
├── payments_data.csv      # Generated payment data
├── requirements.txt       # Python dependencies
├── api_key.txt            # OpenAI API key (not committed)
└── .gitignore
