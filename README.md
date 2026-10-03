# Real-Time Payments Analytics & Fraud Detection Prototype

Interactive prototype for real-time payment analytics, fraud risk scoring, and AI-powered assistance.

Built for a global digital payments use case handling **Card**, **ACH**, and **Wallet** transactions.

---

## Features

- Interactive dashboard with key business metrics
- Fraud risk scoring (0–100) using Machine Learning
- Model performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- High-risk transaction monitoring
- Automatic Dispute Summary generation using AI
- AI Chatbot that answers questions about individual transactions and the overall dataset

---

## Tech Stack

- **Python**
- **Streamlit** – Web application
- **Pandas & NumPy** – Data processing
- **Scikit-learn** – Fraud detection model (Random Forest)
- **Plotly** – Interactive charts
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
