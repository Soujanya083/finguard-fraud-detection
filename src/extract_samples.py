"""
extract_samples.py (v2)

Pulls a curated set of REAL transactions, organized by risk tier (using the
model's own predicted probability), instead of a flat random list. This
mirrors how a real fraud analyst would want to browse cases: by risk level,
not a random flat dropdown.

Run this once from the project root:
    python src/extract_samples.py
"""

import os
import joblib
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

df = pd.read_sql("SELECT * FROM transactions_features", engine)

feature_cols = [f"V{i}" for i in range(1, 29)] + [
    "Hour", "Is_High_Risk_Hour", "Amount_Log", "Amount_Zscore",
    "Is_Round_Amount", "Txn_Count_Last_Hour", "Amount", "Class"
]
feature_cols = [c for c in feature_cols if c in df.columns]

model = joblib.load("models/final_rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

model_feature_cols = [c for c in feature_cols if c not in ("Amount", "Class")]
X_all_scaled = scaler.transform(df[model_feature_cols])
df["model_score"] = model.predict_proba(X_all_scaled)[:, 1]

samples = []

# High risk (>= 80%) — clear block cases
high_risk = df[df["model_score"] >= 0.80].sample(min(4, len(df[df["model_score"] >= 0.80])), random_state=42)
for i, (_, row) in enumerate(high_risk.iterrows()):
    samples.append({"label": f"🚫 High risk #{i+1}", **row[feature_cols].to_dict()})

# Medium risk (40-80%) — review cases, the most interesting/ambiguous ones
med_risk = df[(df["model_score"] >= 0.40) & (df["model_score"] < 0.80)]
med_risk = med_risk.sample(min(4, len(med_risk)), random_state=42)
for i, (_, row) in enumerate(med_risk.iterrows()):
    samples.append({"label": f"🔎 Borderline / review #{i+1}", **row[feature_cols].to_dict()})

# Low risk (< 40%) but still real, everyday transactions
low_risk = df[df["model_score"] < 0.40].sample(4, random_state=42)
for i, (_, row) in enumerate(low_risk.iterrows()):
    samples.append({"label": f"✅ Typical / low risk #{i+1}", **row[feature_cols].to_dict()})

samples_df = pd.DataFrame(samples)
os.makedirs("data", exist_ok=True)
samples_df.to_csv("data/sample_transactions.csv", index=False)
print(f"Saved {len(samples_df)} curated sample transactions to data/sample_transactions.csv")
print(samples_df[["label", "Amount", "Class"]])