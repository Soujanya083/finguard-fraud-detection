import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from llm_narrative import generate_fraud_narrative

load_dotenv(override=True)

st.set_page_config(page_title="FinGuard - Fraud Detection", layout="wide")

RF_WEIGHT = 0.7
ISO_WEIGHT = 0.3

# --- DB connection for audit logging (separate, lightweight, non-fatal if it fails) ---
@st.cache_resource
def get_db_engine():
    url = URL.create(
        "postgresql+psycopg2",
        username=os.environ.get("DB_USER", "postgres"),
        password=os.environ["DB_PASSWORD"],
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432"),
        database=os.environ.get("DB_NAME", "finguard_db"),
    )
    return create_engine(url)

db_engine = get_db_engine()


def log_to_audit_trail(label, amount, rf_score, anomaly_score, combined_score, action, actual_label):
    """Write one row to the audit_log table. Never raises -- a logging
    failure should never break the actual risk assessment the user is
    trying to see."""
    if db_engine is None:
        return False
    try:
        with db_engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO audit_log
                    (transaction_label, amount, rf_score, anomaly_score, combined_score, recommended_action, actual_label)
                    VALUES (:label, :amount, :rf_score, :anomaly_score, :combined_score, :action, :actual_label)
                """),
                {
                    "label": label, "amount": float(amount), "rf_score": float(rf_score),
                    "anomaly_score": float(anomaly_score), "combined_score": float(combined_score),
                    "action": action, "actual_label": actual_label,
                },
            )
            conn.commit()
        return True
    except Exception:
        return False

# --- Load model, scaler, isolation forest, and real sample transactions ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/final_rf_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        iso_forest = joblib.load("models/isolation_forest.pkl")
        iso_score_scaler = joblib.load("models/iso_score_scaler.pkl")
        pca_col_idx = joblib.load("models/pca_col_idx.pkl")
        return model, scaler, iso_forest, iso_score_scaler, pca_col_idx
    except FileNotFoundError as e:
        st.error(
            f"⚠️ Model files not found ({e}). Run `03_modeling.ipynb` fully "
            "(including the Isolation Forest cells) to generate all required model files."
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
    except Exception:
        return None

model, scaler, iso_forest, iso_score_scaler, pca_col_idx = load_model()
samples_df = load_samples()
test_preds_df = load_test_predictions()

feature_cols = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14',
                 'V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28',
                 'Hour','Is_High_Risk_Hour','Amount_Log','Amount_Zscore','Is_Round_Amount','Txn_Count_Last_Hour']


def get_combined_risk_score(X_input_scaled):
    """Blend the supervised Random Forest score with the Isolation Forest
    anomaly score (fit on PCA features only). Weights (0.7 RF / 0.3 ISO)
    were chosen based on measured precision/recall trade-offs on the held-out
    test set -- see README for the full comparison table."""
    rf_proba = model.predict_proba(X_input_scaled)[0][1]

    X_pca_only = X_input_scaled[:, pca_col_idx]
    iso_raw_score = -iso_forest.decision_function(X_pca_only)
    iso_score_norm = iso_score_scaler.transform(iso_raw_score.reshape(-1, 1)).flatten()[0]
    iso_score_norm = np.clip(iso_score_norm, 0, 1)  # guard against out-of-range values on new data

    combined = (RF_WEIGHT * rf_proba) + (ISO_WEIGHT * iso_score_norm)
    return combined, rf_proba, iso_score_norm


st.title("🛡️ FinGuard — Fraud Detection System")

tab1, tab2, tab3 = st.tabs(["🔍 Transaction Assessment", "🎚️ Risk Threshold Simulator", "📋 Audit Log"])

# ============================================================
# TAB 1: Transaction assessment (now using combined RF + Isolation Forest score)
# ============================================================
with tab1:
    st.markdown(
        "Select a real transaction from the held-out test set to see its fraud risk score, "
        "the model's reasoning, and a recommended action."
    )
    st.caption(
        f"Risk score = {RF_WEIGHT} × Random Forest probability + {ISO_WEIGHT} × Isolation Forest "
        "anomaly score (weights chosen from measured precision/recall trade-offs — see README)."
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
                    "Try regenerating sample data with `python src/extract_samples.py`."
                )
                st.stop()

            try:
                X_input = pd.DataFrame([selected_row])[feature_cols]
                X_input_scaled = scaler.transform(X_input)
                combined_proba, rf_proba, iso_score = get_combined_risk_score(X_input_scaled)
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
                st.stop()

            if combined_proba >= 0.80:
                action = "🚫 Auto-block & escalate to fraud team"
                action_color = "error"
            elif combined_proba >= 0.40:
                action = "🔎 Flag for manual review"
                action_color = "warning"
            else:
                action = "✅ Auto-approve"
                action_color = "success"

            st.subheader("Risk Assessment")
            c1, c2, c3 = st.columns(3)
            c1.metric("Combined Risk Score", f"{combined_proba*100:.2f}%")
            c2.metric("— RF component", f"{rf_proba*100:.1f}%")
            c3.metric("— Anomaly component", f"{iso_score*100:.1f}%")
            st.metric("Actual Label (ground truth)", "FRAUD" if selected_row["Class"] == 1 else "LEGIT")

            logged = log_to_audit_trail(
                label=scenario_label, amount=selected_row["Amount"],
                rf_score=rf_proba, anomaly_score=iso_score, combined_score=combined_proba,
                action=action, actual_label="FRAUD" if selected_row["Class"] == 1 else "LEGIT",
            )
            if logged:
                st.caption("✅ This assessment was logged to the audit trail.")
            else:
                st.caption("⚠️ Audit logging unavailable (assessment result above is still valid).")

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
                st.caption("Note: SHAP explains the Random Forest component. The anomaly component is a separate, unsupervised signal blended in above.")
            except Exception as e:
                st.warning(f"⚠️ Could not generate SHAP explanation ({e}). Risk score above is still valid.")

            st.subheader("🤖 AI-Generated Risk Summary")
            if top_features:
                with st.spinner("Generating natural-language explanation..."):
                    narrative = generate_fraud_narrative(
                        risk_probability=combined_proba,
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
# TAB 2: Risk Threshold Simulator (unchanged — still uses RF-only test predictions)
# ============================================================
with tab2:
    st.markdown(
        "Every fraud system has to pick a threshold: above what probability do we flag a transaction? "
        "Move the slider to see how that single choice trades off fraud caught, false alarms, and "
        "manual review workload — computed live on the **actual held-out test set** (56,962 real transactions)."
    )
    st.caption(
        "Note: this simulator currently uses the Random Forest score only (not the combined RF+Isolation "
        "Forest score used in the Transaction Assessment tab)."
    )

    if test_preds_df is None:
        st.warning(
            "⚠️ Test predictions file not found. Run the export cell in `03_modeling.ipynb` "
            "to enable this simulator."
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

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        fraud_caught_amt = amounts[(y_true == 1) & (y_flagged == 1)].sum()

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
            "₹150/review is a stated assumption, not derived from real cost data. "
            "Currency shown as ₹ for illustration; the source dataset does not specify a currency."
        )

# ============================================================
# TAB 3: Audit Log (new) — shows every logged risk assessment
# ============================================================
with tab3:
    st.markdown(
        "Every assessment made in the **Transaction Assessment** tab is logged here — "
        "timestamp, scores, and the action taken. This is what a real risk system needs "
        "for compliance and later review: a record of every decision, not just the final one."
    )

    if db_engine is None:
        st.warning("⚠️ Database connection unavailable — cannot load audit log.")
    else:
        try:
            with db_engine.connect() as conn:
                audit_df = pd.read_sql(
                    text("SELECT timestamp, transaction_label, amount, rf_score, anomaly_score, "
                         "combined_score, recommended_action, actual_label FROM audit_log "
                         "ORDER BY timestamp DESC LIMIT 100"),
                    conn,
                )
            if audit_df.empty:
                st.info("No assessments logged yet — go make one in the Transaction Assessment tab.")
            else:
                st.dataframe(audit_df, use_container_width=True)
                st.caption(f"Showing the {len(audit_df)} most recent logged assessment(s).")

                # Quick summary stats
                st.subheader("Summary")
                s1, s2, s3 = st.columns(3)
                s1.metric("Total Logged Assessments", len(audit_df))
                s2.metric("Auto-blocked", (audit_df["recommended_action"].str.contains("block")).sum())
                s3.metric("Flagged for Review", (audit_df["recommended_action"].str.contains("review")).sum())
        except Exception as e:
            st.error(f"⚠️ Could not load audit log: {e}")

st.markdown("---")
st.caption("FinGuard | Track 2: AI Risk Manager | Risk Score: 0.7×Random Forest + 0.3×Isolation Forest | AI narrative: Google Gemini")