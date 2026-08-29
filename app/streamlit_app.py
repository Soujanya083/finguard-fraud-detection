import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="FinGuard - Fraud Detection", layout="wide")

# --- Load model and scaler ---
@st.cache_resource
def load_model():
    model = joblib.load("models/final_rf_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

feature_cols = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14',
                 'V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28',
                 'Hour','Is_High_Risk_Hour','Amount_Log','Amount_Zscore','Is_Round_Amount','Txn_Count_Last_Hour']

st.title("🛡️ FinGuard — Fraud Detection System")
st.markdown("Enter transaction details to check fraud probability, or use randomized test values.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Transaction Input")
    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=1.0)
    hour = st.slider("Hour of Day", 0, 23, 12)

    st.caption("Advanced: anonymized PCA features (V1-V28). Defaults are typical/legit-like values — adjust to simulate different scenarios.")
    use_random_fraud_like = st.checkbox("Use example fraud-like values")

    v_values = {}
    if use_random_fraud_like:
        # Example values roughly mimicking a suspicious pattern based on our SHAP findings
        fraud_like_defaults = {'V14': -8.8, 'V10': -2.9, 'V12': -3.7, 'V4': 3.75, 'V11': 1.85, 'V17': -1.6}
        for v in [f"V{i}" for i in range(1, 29)]:
            v_values[v] = fraud_like_defaults.get(v, 0.0)
    else:
        for v in [f"V{i}" for i in range(1, 29)]:
            v_values[v] = 0.0

    predict_btn = st.button("🔍 Predict Fraud Risk", type="primary", use_container_width=True)

with col2:
    if predict_btn:
        # Build feature row matching training format
        amount_log = np.log1p(amount)
        amount_zscore = (amount - 88.29) / 250.10  # using approx train mean/std from EDA
        is_round = 1 if amount % 1 == 0 else 0
        is_high_risk_hour = 1 if hour in [1,2,3,4,5] else 0
        txn_count_last_hour = 7000  # approx average, since we can't compute true rolling count for a single new input

        row = {**v_values, 'Hour': hour, 'Is_High_Risk_Hour': is_high_risk_hour,
               'Amount_Log': amount_log, 'Amount_Zscore': amount_zscore,
               'Is_Round_Amount': is_round, 'Txn_Count_Last_Hour': txn_count_last_hour}

        X_input = pd.DataFrame([row])[feature_cols]
        X_input_scaled = scaler.transform(X_input)

        proba = model.predict_proba(X_input_scaled)[0][1]
        prediction = "🚨 FRAUD" if proba > 0.5 else "✅ LEGIT"

        st.subheader("Prediction Result")
        st.metric("Fraud Probability", f"{proba*100:.2f}%")
        if proba > 0.5:
            st.error(f"**{prediction}** — This transaction is flagged as high risk.")
        else:
            st.success(f"**{prediction}** — This transaction appears normal.")

        # SHAP explanation for this single prediction
        st.subheader("Why this prediction? (SHAP Explanation)")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_input_scaled)
        shap_vals_fraud = shap_vals[:, :, 1] if np.array(shap_vals).ndim == 3 else shap_vals[1]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_vals_fraud, X_input, feature_names=feature_cols, plot_type="bar", show=False)
        st.pyplot(fig)
    else:
        st.info("👈 Enter transaction details and click Predict to see results.")

st.markdown("---")
st.caption("FinGuard | Final Year Data Analytics Major Project | Model: Tuned Random Forest (PR-AUC 0.80)")