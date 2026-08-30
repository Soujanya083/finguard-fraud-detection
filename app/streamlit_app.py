import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from llm_narrative import generate_fraud_narrative

st.set_page_config(page_title="FinGuard - Fraud Detection", layout="wide")

# --- Load model, scaler, samples, and full test predictions (with error handling) ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/final_rf_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error(
            "⚠️ Model files not found. Make sure `models/final_rf_model.pkl` and "
            "`models/scaler.pkl` exist — run `03_modeling.ipynb` first to train and save them."
        )
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Unexpected error loading model: {e}")
        st.stop()

@st.cache_data
def load_samples():
    try:
        return pd.read_csv("data/sample_transactions.csv")
    except FileNotFoundError:
        st.error(
            "⚠️ Sample transactions file not found. Run `python src/extract_samples.py` "
            "first to generate `data/sample_transactions.csv`."
        )
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Unexpected error loading sample transactions: {e}")
        st.stop()

@st.cache_data
def load_test_predictions():
    try:
        return pd.read_csv("data/test_predictions.csv")
    except FileNotFoundError:
        return None
    except Exception:
        return None

model, scaler = load_model()
samples_df = load_samples()
test_preds_df = load_test_predictions()

feature_cols = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14',
                 'V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28',
                 'Hour','Is_High_Risk_Hour','Amount_Log','Amount_Zscore','Is_Round_Amount','Txn_Count_Last_Hour']

st.title("🛡️ FinGuard — Fraud Detection System")

tab1, tab2 = st.tabs(["🔍 Transaction Assessment", "🎚️ Risk Threshold Simulator"])

# ============================================================
# TAB 1: Existing single-transaction assessment (unchanged)
# ============================================================
with tab1:
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

        if samples_df.empty:
            st.warning("No sample transactions available.")
            st.stop()

        scenario_label = st.selectbox("Transaction scenario", samples_df["label"].tolist())
        selected_row = samples_df[samples_df["label"] == scenario_label].iloc[0]

        st.metric("Transaction Amount", f"₹{selected_row['Amount']:.2f}")
        st.caption(f"Hour of day: {int(selected_row['Hour'])}:00")

        predict_btn = st.button("🔍 Assess Fraud Risk", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            missing_cols = [c for c in feature_cols if c not in selected_row.index]
            if missing_cols:
                st.error(
                    f"⚠️ This transaction is missing required features: {', '.join(missing_cols)}. "
                    "The sample data may be out of sync with the model — try regenerating it with "
                    "`python src/extract_samples.py`."
                )
                st.stop()

            try:
                X_input = pd.DataFrame([selected_row])[feature_cols]
                X_input_scaled = scaler.transform(X_input)
                proba = model.predict_proba(X_input_scaled)[0][1]
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
                st.info("This usually means the sample data's columns don't match what the model expects.")
                st.stop()

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

            st.subheader("Why this prediction? (SHAP Explanation)")
            top_features = []
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_input_scaled)
                shap_vals_fraud = shap_vals[:, :, 1] if np.array(shap_vals).ndim == 3 else shap_vals[1]

                impacts = list(zip(feature_cols, shap_vals_fraud[0]))
                top_features = sorted(impacts, key=lambda x: abs(x[1]), reverse=True)[:4]

                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_vals_fraud, X_input, feature_names=feature_cols, plot_type="bar", show=False)
                st.pyplot(fig)
            except Exception as e:
                st.warning(
                    f"⚠️ Could not generate SHAP explanation ({e}). "
                    "The risk score and recommendation above are still valid."
                )

            st.subheader("🤖 AI-Generated Risk Summary")
            if top_features:
                with st.spinner("Generating natural-language explanation..."):
                    narrative = generate_fraud_narrative(
                        risk_probability=proba,
                        top_features=top_features,
                        amount=selected_row["Amount"],
                        recommended_action=action,
                    )
                st.info(narrative)
            else:
                st.caption("AI narrative unavailable — SHAP feature data wasn't generated above.")
        else:
            st.info("👈 Select a transaction and click Assess to see results.")

# ============================================================
# TAB 2: Risk Threshold Simulator (new)
# ============================================================
with tab2:
    st.markdown(
        "Every fraud system has to pick a threshold: above what probability do we flag a transaction? "
        "Move the slider to see how that single choice trades off fraud caught, false alarms, and "
        "manual review workload — computed live on the **actual held-out test set** (56,962 real transactions)."
    )

    if test_preds_df is None:
        st.warning(
            "⚠️ Test predictions file not found. Run the export cell in `03_modeling.ipynb` "
            "(saves `data/test_predictions.csv`) to enable this simulator."
        )
    else:
        threshold = st.slider(
            "Fraud probability threshold — transactions scoring above this are flagged",
            min_value=0.0, max_value=1.0, value=0.40, step=0.01,
        )

        y_true = test_preds_df["y_true"].values
        y_proba = test_preds_df["y_pred_proba"].values
        amounts = test_preds_df["Amount"].values
        y_flagged = (y_proba >= threshold).astype(int)

        tp = ((y_true == 1) & (y_flagged == 1)).sum()
        fp = ((y_true == 0) & (y_flagged == 1)).sum()
        fn = ((y_true == 1) & (y_flagged == 0)).sum()
        tn = ((y_true == 0) & (y_flagged == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        fraud_caught_amt = amounts[(y_true == 1) & (y_flagged == 1)].sum()
        fraud_missed_amt = amounts[(y_true == 1) & (y_flagged == 0)].sum()

        COST_PER_REVIEW = 150
        review_cost = fp * COST_PER_REVIEW
        net_benefit = fraud_caught_amt - review_cost

        st.subheader(f"At threshold {threshold:.2f}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{precision*100:.1f}%")
        m2.metric("Recall (fraud caught)", f"{recall*100:.1f}%")
        m3.metric("False Positives", f"{fp:,}")
        m4.metric("Frauds Missed", f"{fn:,}")

        m5, m6, m7 = st.columns(3)
        m5.metric("Fraud Amount Caught", f"₹{fraud_caught_amt:,.0f}")
        m6.metric("Est. Review Cost (@₹150/case)", f"₹{review_cost:,.0f}")
        m7.metric("Net Benefit", f"₹{net_benefit:,.0f}")

        st.caption(
            "Lower the threshold → catch more fraud, but more false alarms (more manual review workload). "
            "Raise it → fewer false alarms, but more fraud slips through. "
            "₹150/review is a stated assumption, not derived from real cost data. "
            "Currency shown as ₹ for illustration; the source dataset does not specify a currency."
        )

st.markdown("---")
st.caption("FinGuard | Track 2: AI Risk Manager | Model: Tuned Random Forest (Precision 0.71, Recall 0.79, PR-AUC 0.80) | AI narrative: Google Gemini")