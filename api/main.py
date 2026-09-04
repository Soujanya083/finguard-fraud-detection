from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from scoring import RF_WEIGHT, ISO_WEIGHT, get_combined_score, get_risk_bucket

try:
    import shap
except ImportError:
    shap = None

try:
    import joblib
except ImportError:
    joblib = None


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_FILE = ROOT / "data" / "processed" / "transactions_features.csv"


app = FastAPI(
    title="FinGuard Fraud Risk API",
    description="AI-powered fraud risk scoring API for FinGuard.",
    version="1.0.0",
)


# ---------------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------------

class Transaction(BaseModel):
    """
    You can either send:

    1. Raw transaction:
       Time + V1...V28 + Amount

    OR

    2. Already engineered 34 model features.

    Txn_Count_Last_Hour is optional for raw transactions because
    a single API request cannot know the true live transaction count.
    """

    features: Dict[str, float] = Field(
        ...,
        description="Transaction features."
    )

    txn_count_last_hour: Optional[float] = Field(
        default=0,
        description="Optional transaction count in the previous hour."
    )


class PredictionResponse(BaseModel):
    risk_score: float
    risk_percent: float
    rf_score: float
    anomaly_score: Optional[float] = None
    risk_level: str
    decision: str
    model_available: bool
    explanation: List[Dict[str, Any]]
    note: str


# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------

def _load(path: Path):
    if not path.exists():
        return None

    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        if joblib:
            return joblib.load(path)
        raise


ARTIFACTS = {
    "rf": _load(MODEL_DIR / "final_rf_model.pkl"),
    "scaler": _load(MODEL_DIR / "scaler.pkl"),
    "iso": _load(MODEL_DIR / "isolation_forest.pkl"),
    "iso_scaler": _load(MODEL_DIR / "iso_score_scaler.pkl"),
    "pca_col_idx": _load(MODEL_DIR / "pca_col_idx.pkl"),
}


# ---------------------------------------------------------
# MODEL FEATURE ORDER
# ---------------------------------------------------------

MODEL_FEATURES = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20", "V21", "V22",
    "V23", "V24", "V25", "V26", "V27", "V28",
    "Hour",
    "Is_High_Risk_Hour",
    "Amount_Log",
    "Amount_Zscore",
    "Is_Round_Amount",
    "Txn_Count_Last_Hour",
]


# ---------------------------------------------------------
# AMOUNT STATISTICS
# ---------------------------------------------------------

def get_amount_statistics():
    """
    Uses the same processed dataset used by the project
    to calculate the Amount mean and standard deviation.
    """

    if not DATA_FILE.exists():
        raise RuntimeError(
            "Processed transaction file not found: "
            "data/processed/transactions_features.csv"
        )

    df = pd.read_csv(DATA_FILE)

    if "Amount" not in df.columns:
        raise RuntimeError("Amount column not found in processed dataset.")

    mean = float(df["Amount"].mean())
    std = float(df["Amount"].std())

    return mean, std


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

def prepare_features(
    features: Dict[str, float],
    txn_count_last_hour: float = 0
):
    """
    Converts raw transaction data into the exact 34 features
    expected by the trained Random Forest model.
    """

    if ARTIFACTS["rf"] is None:
        raise RuntimeError(
            "Random Forest model not found at "
            "models/final_rf_model.pkl"
        )

    if ARTIFACTS["scaler"] is None:
        raise RuntimeError(
            "Scaler not found at models/scaler.pkl"
        )

    # -----------------------------------------------------
    # OPTION 1:
    # User already supplied all 34 engineered features
    # -----------------------------------------------------

    if all(feature in features for feature in MODEL_FEATURES):

        df = pd.DataFrame(
            [[features[f] for f in MODEL_FEATURES]],
            columns=MODEL_FEATURES
        )

        return df

    # -----------------------------------------------------
    # OPTION 2:
    # User supplied raw transaction
    # Time + V1...V28 + Amount
    # -----------------------------------------------------

    required_raw = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount"]
    )

    missing = [
        feature for feature in required_raw
        if feature not in features
    ]

    if missing:
        raise ValueError(
            "Missing required raw features: "
            + ", ".join(missing)
        )

    time_value = float(features["Time"])
    amount = float(features["Amount"])

    # V1-V28
    engineered = {
        f"V{i}": float(features[f"V{i}"])
        for i in range(1, 29)
    }

    # Hour
    hour = int((time_value // 3600) % 24)

    engineered["Hour"] = hour

    # High-risk hour
    engineered["Is_High_Risk_Hour"] = (
        1 if hour in [1, 2, 3, 4, 5] else 0
    )

    # Log amount
    engineered["Amount_Log"] = float(np.log1p(amount))

    # Amount Z-score
    amount_mean, amount_std = get_amount_statistics()

    if amount_std == 0:
        amount_zscore = 0.0
    else:
        amount_zscore = (
            (amount - amount_mean) / amount_std
        )

    engineered["Amount_Zscore"] = float(amount_zscore)

    # Round amount
    engineered["Is_Round_Amount"] = (
        1 if amount % 1 == 0 else 0
    )

    # Transaction count
    engineered["Txn_Count_Last_Hour"] = float(
        txn_count_last_hour
    )

    # Exact model order
    df = pd.DataFrame(
        [[engineered[f] for f in MODEL_FEATURES]],
        columns=MODEL_FEATURES
    )

    return df


# ---------------------------------------------------------
# SCALE FEATURES
# ---------------------------------------------------------

def scale_features(X: pd.DataFrame):
    """
    Apply the same StandardScaler used during model training.
    """

    scaler = ARTIFACTS["scaler"]

    try:
        return scaler.transform(X)
    except Exception as exc:
        raise RuntimeError(
            f"Feature scaling failed: {type(exc).__name__}: {exc}"
        )


# ---------------------------------------------------------
# COMBINED SCORE (RF + Isolation Forest)
# ---------------------------------------------------------

def get_combined_risk_score(X_scaled):
    """
    Mirrors app/streamlit_app.py's get_combined_risk_score() exactly, so the
    API and the dashboard return the same score for the same transaction.

    rf_score:      Random Forest fraud probability, 0-1
    anomaly_score: Isolation Forest anomaly score (fit on PCA columns only),
                    normalized to 0-1, or None if the Isolation Forest
                    artifacts weren't found -- in that case the combined
                    score falls back to the RF score alone.
    """
    rf = ARTIFACTS["rf"]
    rf_score = float(rf.predict_proba(X_scaled)[0, 1])

    iso = ARTIFACTS["iso"]
    iso_scaler = ARTIFACTS["iso_scaler"]
    pca_col_idx = ARTIFACTS["pca_col_idx"]

    if iso is None or iso_scaler is None or pca_col_idx is None:
        # Degrade gracefully instead of pretending we blended when we didn't.
        return rf_score, rf_score, None

    X_pca_only = X_scaled[:, pca_col_idx]
    iso_raw_score = -iso.decision_function(X_pca_only)
    anomaly_score = float(
        iso_scaler.transform(iso_raw_score.reshape(-1, 1)).flatten()[0]
    )
    anomaly_score = float(np.clip(anomaly_score, 0, 1))

    combined = get_combined_score(rf_score, anomaly_score)
    return combined, rf_score, anomaly_score


# ---------------------------------------------------------
# SHAP EXPLANATION
# ---------------------------------------------------------

def get_shap_explanation(
    X_scaled,
    feature_names,
    top_n=5
):
    if shap is None:
        return []

    if ARTIFACTS["rf"] is None:
        return []

    try:
        explainer = shap.TreeExplainer(
            ARTIFACTS["rf"]
        )

        values = explainer.shap_values(X_scaled)

        if isinstance(values, list):
            vals = np.asarray(values[-1])[0]

        else:
            arr = np.asarray(values)

            if arr.ndim == 3:
                vals = arr[0, :, -1]

            elif arr.ndim == 2:
                vals = arr[0]

            else:
                vals = arr.reshape(-1)

        pairs = sorted(
            zip(feature_names, vals),
            key=lambda x: abs(float(x[1])),
            reverse=True
        )[:top_n]

        return [
            {
                "feature": name,
                "shap_value": round(float(value), 6),
                "direction": (
                    "increases_risk"
                    if value > 0
                    else "decreases_risk"
                )
            }
            for name, value in pairs
        ]

    except Exception as exc:
        return [
            {
                "feature": "SHAP",
                "shap_value": 0.0,
                "direction": f"unavailable: {type(exc).__name__}"
            }
        ]


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "random_forest_loaded": ARTIFACTS["rf"] is not None,
        "isolation_forest_loaded": ARTIFACTS["iso"] is not None,
        "shap_available": shap is not None,
        "scaler_loaded": ARTIFACTS["scaler"] is not None,
    }


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):

    try:

        # 1. Feature engineering
        X = prepare_features(
            transaction.features,
            transaction.txn_count_last_hour or 0
        )

        # 2. Scaling
        X_scaled = scale_features(X)

        # 3. Combined Random Forest + Isolation Forest prediction
        #    (matches app/streamlit_app.py's get_combined_risk_score exactly)
        risk_score, rf_score, anomaly_score = get_combined_risk_score(X_scaled)

        # 4. SHAP explanation (computed from the RF model, same as dashboard)
        explanation = get_shap_explanation(
            X_scaled,
            MODEL_FEATURES
        )

        # 5. Business decision
        risk_level, decision = get_risk_bucket(
            risk_score
        )

        note = (
            f"FinGuard risk_score = {RF_WEIGHT}\u00d7Random Forest + "
            f"{ISO_WEIGHT}\u00d7Isolation Forest, matching the dashboard. "
            "Txn_Count_Last_Hour is supplied by the caller "
            "when available; a single API request cannot "
            "calculate a true live transaction-stream count."
        )
        if anomaly_score is None:
            note = (
                "Isolation Forest artifacts not found -- risk_score is the "
                "Random Forest probability only (not blended). "
                "Txn_Count_Last_Hour is supplied by the caller "
                "when available; a single API request cannot "
                "calculate a true live transaction-stream count."
            )

        return PredictionResponse(
            risk_score=round(risk_score, 6),
            risk_percent=round(risk_score * 100, 2),
            rf_score=round(rf_score, 6),
            anomaly_score=(
                round(anomaly_score, 6) if anomaly_score is not None else None
            ),
            risk_level=risk_level,
            decision=decision,
            model_available=True,
            explanation=explanation,
            note=note,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: "
                f"{type(exc).__name__}: {exc}"
            )
        )
