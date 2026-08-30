"""
extract_samples.py

Pulls a small set of REAL transactions from the database (not fake/hand-typed
values) to use as selectable demo scenarios in the Streamlit app. This keeps
the demo credible -- every scenario is an actual row from the dataset, not a
made-up value nobody can explain.

Run this once from the project root:
    python src/extract_samples.py

Output: data/sample_transactions.csv (small, safe to commit to git --
only a handful of rows, no bulk raw data).
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv

load_dotenv(override=True)

url = URL.create(
    "postgresql+psycopg2",
    username=os.environ.get("DB_USER", "postgres"),
    password=os.environ["DB_PASSWORD"],
    host=os.environ.get("DB_HOST", "localhost"),
    port=os.environ.get("DB_PORT", "5432"),
    database=os.environ.get("DB_NAME", "finguard_db"),
)
engine = create_engine(url)

# Pull the engineered feature table (same one the model was trained on)
df = pd.read_sql("SELECT * FROM transactions_features", engine)

feature_cols = [f"V{i}" for i in range(1, 29)] + [
    "Hour", "Is_High_Risk_Hour", "Amount_Log", "Amount_Zscore",
    "Is_Round_Amount", "Txn_Count_Last_Hour", "Amount", "Class"
]
# Only keep columns that actually exist (in case naming differs slightly)
feature_cols = [c for c in feature_cols if c in df.columns]

samples = []

# 1. A couple of clearly legitimate, everyday-looking transactions
legit = df[df["Class"] == 0].sample(3, random_state=42)
for i, (_, row) in enumerate(legit.iterrows()):
    samples.append({"label": f"Typical purchase #{i+1}", **row[feature_cols].to_dict()})

# 2. A couple of confirmed fraud cases, picked across a range of model confidence
fraud = df[df["Class"] == 1].sample(3, random_state=42)
for i, (_, row) in enumerate(fraud.iterrows()):
    samples.append({"label": f"Confirmed fraud case #{i+1}", **row[feature_cols].to_dict()})

samples_df = pd.DataFrame(samples)
os.makedirs("data", exist_ok=True)
samples_df.to_csv("data/sample_transactions.csv", index=False)
print(f"Saved {len(samples_df)} sample transactions to data/sample_transactions.csv")
print(samples_df[["label", "Amount", "Class"]])
