import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="FinGuard - Fraud Detection", layout="wide")

# --- Load model, scaler, and real sample transactions ---
@st.cache_resource
def load_model():
    model = joblib.load("models/final_rf_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

@st.cache_data
def load_samples():
    return pd.read_csv("data/sample_transactions.csv")

model, scaler = load_model()
samples_df = load_samples()

feature_cols = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14',
                 'V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28',
                 'Hour','Is_High_Risk_Hour','Amount_Log','Amount_Zscore','Is_Round_Amount','Txn_Count_Last_Hour']

st.title("🛡️ FinGuard — Fraud Detection System")
st.markdown(
    "Select a real transaction from the held-out test set to see its fraud risk score, "
    "the model's reasoning, and a recommended action."
)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Select a Transaction")
    st.caption(
        "These are real, held-out transactions — not synthetic examples. "
        "V1–V28 are anonymized PCA features from the original dataset (Kaggle's privacy-preserving "
        "transform), so individual values aren't human-interpretable on their own; "
        "SHAP below explains their *relative* contribution to this specific prediction."
    )

    scenario_label = st.selectbox("Transaction scenario", samples_df["label"].tolist())
    selected_row = samples_df[samples_df["label"] == scenario_label].iloc[0]

    st.metric("Transaction Amount", f"₹{selected_row['Amount']:.2f}")
    st.caption(f"Hour of day: {int(selected_row['Hour'])}:00")

    predict_btn = st.button("🔍 Assess Fraud Risk", type="primary", use_container_width=True)

with col2:
    if predict_btn:
        X_input = pd.DataFrame([selected_row])[feature_cols]
        X_input_scaled = scaler.transform(X_input)

        proba = model.predict_proba(X_input_scaled)[0][1]

        # --- Recommended action based on risk tier (simple, explainable thresholds) ---
        if proba >= 0.80:
            action = "🚫 Auto-block & escalate to fraud team"
            action_color = "error"
        elif proba >= 0.40:
            action = "🔎 Flag for manual review"
            action_color = "warning"
        else:
            action = "✅ Auto-approve"
            action_color = "success"

        st.subheader("Risk Assessment")
        c1, c2 = st.columns(2)
        c1.metric("Fraud Probability", f"{proba*100:.2f}%")
        c2.metric("Actual Label (ground truth)", "FRAUD" if selected_row["Class"] == 1 else "LEGIT")

        st.subheader("Recommended Action")
        getattr(st, action_color)(f"**{action}**")
        st.caption(
            "Thresholds: ≥80% → auto-block, 40–80% → manual review, <40% → auto-approve. "
            "These are illustrative starting points, not tuned against real cost data."
        )

        # SHAP explanation for this single prediction
        st.subheader("Why this prediction? (SHAP Explanation)")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_input_scaled)
        shap_vals_fraud = shap_vals[:, :, 1] if np.array(shap_vals).ndim == 3 else shap_vals[1]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_vals_fraud, X_input, feature_names=feature_cols, plot_type="bar", show=False)
        st.pyplot(fig)
    else:
        st.info("👈 Select a transaction and click Assess to see results.")

st.markdown("---")
st.caption("FinGuard | Track 2: AI Risk Manager | Model: Tuned Random Forest (Precision 0.71, Recall 0.79, PR-AUC 0.80)")