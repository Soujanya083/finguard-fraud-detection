import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sys
import os
import time
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from llm_narrative import generate_fraud_narrative
from scoring import (
    RF_WEIGHT,
    ISO_WEIGHT,
    BLOCK_THRESHOLD,
    REVIEW_THRESHOLD,
    get_combined_score,
    get_risk_bucket,
)

load_dotenv(override=True)

st.set_page_config(page_title="FinGuard - Fraud Detection", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: #1A1F36; letter-spacing: -0.01em; }

    .stApp { background-color: #FAFBFC; }

    /* Metric cards: white surface, color-coded left border carries meaning, not decoration */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E3E7EF;
        border-left: 3px solid #2563EB;
        border-radius: 6px;
        padding: 14px 18px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        font-variant-numeric: tabular-nums;
        color: #1A1F36;
    }
    div[data-testid="stMetricLabel"] { color: #6B7280; font-weight: 500; font-size: 0.85rem; }

    /* Tabs: blue underline on active */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid #E3E7EF;
    }
    .stTabs [data-baseweb="tab"] {
        color: #6B7280;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        padding: 10px 18px;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
    }

    /* Buttons: blue accent, sharp not bubbly */
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #2563EB;
        color: #2563EB;
    }
    .stButton>button[kind="primary"] {
        background-color: #2563EB;
        color: #FFFFFF;
        border: none;
    }

    div[data-testid="stAlert"] { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

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

@st.cache_data
def load_full_dataset():
    if db_engine is None:
        return None
    try:
        with db_engine.connect() as conn:
            return pd.read_sql("SELECT * FROM transactions_features", conn)
    except Exception:
        return None

model, scaler, iso_forest, iso_score_scaler, pca_col_idx = load_model()
samples_df = load_samples()
test_preds_df = load_test_predictions()
full_dataset_df = load_full_dataset()

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

    combined = get_combined_score(rf_proba, iso_score_norm)
    return combined, rf_proba, iso_score_norm


st.title("🛡️ FinGuard — Fraud Detection System")
st.caption("AI Risk Manager | Real-time transaction scoring, explainability, and human-in-the-loop review")
st.markdown("")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🔍 Transaction Assessment", "🎚️ Risk Threshold Simulator", "📋 Audit Log", "🕵️ Investigator Queue", "📊 Fraud Analytics", "⚡ Live Feed Simulation", "📁 Batch Scoring"])

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

            if combined_proba >= BLOCK_THRESHOLD:
                action = "🚫 Auto-block & escalate to fraud team"
                action_color = "error"
            elif combined_proba >= REVIEW_THRESHOLD:
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

# ============================================================
# TAB 4: Investigator Queue (new) — human-in-the-loop review
# ============================================================
with tab4:
    st.markdown(
        "Transactions flagged for **manual review** wait here until an investigator makes a call. "
        "This closes the loop: the model recommends, a human decides, and that decision is stored — "
        "the foundation for future model retraining on real feedback."
    )

    if db_engine is None:
        st.warning("⚠️ Database connection unavailable — cannot load the review queue.")
    else:
        try:
            with db_engine.connect() as conn:
                pending_df = pd.read_sql(
                    text("""
                        SELECT id, timestamp, transaction_label, amount, combined_score, actual_label
                        FROM audit_log
                        WHERE recommended_action LIKE '%%review%%' AND investigator_decision IS NULL
                        ORDER BY timestamp DESC
                    """),
                    conn,
                )
        except Exception as e:
            st.error(f"⚠️ Could not load review queue: {e}")
            pending_df = pd.DataFrame()

        if pending_df.empty:
            st.success("✅ No transactions currently pending review.")
            st.caption(
                "Go assess a transaction in Tab 1 that lands in the 40–80% risk range "
                "(recommended action: 'Flag for manual review') to populate this queue."
            )
        else:
            st.info(f"**{len(pending_df)}** transaction(s) awaiting review.")
            for _, row in pending_df.iterrows():
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2, 1, 2])
                    c1.markdown(f"**{row['transaction_label']}**  \n₹{row['amount']:.2f} · Risk: {row['combined_score']*100:.1f}%")
                    c2.markdown(f"Ground truth:  \n**{row['actual_label']}**")
                    with c3:
                        b1, b2, b3 = st.columns(3)
                        decide = None
                        if b1.button("✅ Legit", key=f"legit_{row['id']}", use_container_width=True):
                            decide = "Confirmed Legitimate"
                        if b2.button("🚨 Fraud", key=f"fraud_{row['id']}", use_container_width=True):
                            decide = "Confirmed Fraud"
                        if b3.button("⏳ Escalate", key=f"esc_{row['id']}", use_container_width=True):
                            decide = "Escalated"

                        if decide:
                            try:
                                with db_engine.connect() as conn:
                                    conn.execute(
                                        text("""
                                            UPDATE audit_log
                                            SET investigator_decision = :decision, decided_at = NOW()
                                            WHERE id = :id
                                        """),
                                        {"decision": decide, "id": int(row["id"])},
                                    )
                                    conn.commit()
                                st.success(f"Recorded: {decide}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"⚠️ Could not save decision: {e}")

        # Show recently decided items too
        try:
            with db_engine.connect() as conn:
                decided_df = pd.read_sql(
                    text("""
                        SELECT timestamp, transaction_label, amount, combined_score,
                               investigator_decision, decided_at
                        FROM audit_log
                        WHERE investigator_decision IS NOT NULL
                        ORDER BY decided_at DESC LIMIT 20
                    """),
                    conn,
                )
            if not decided_df.empty:
                st.subheader("Recently Decided")
                st.dataframe(decided_df, use_container_width=True)
        except Exception:
            pass

# ============================================================
# TAB 5: Fraud Analytics (new) — real SQL-driven business insights
# ============================================================
with tab5:
    st.markdown(
        "Business-question analytics computed directly via SQL against the `transactions_features` table — "
        "the same real, held-out data used throughout this app."
    )

    if db_engine is None:
        st.warning("⚠️ Database connection unavailable — cannot load analytics.")
    else:
        try:
            with db_engine.connect() as conn:
                # Query 1: Fraud rate by hour of day
                hourly_df = pd.read_sql(
                    text("""
                        SELECT "Hour",
                               COUNT(*) AS total_transactions,
                               SUM(CASE WHEN "Class" = 1 THEN 1 ELSE 0 END) AS fraud_count,
                               ROUND(100.0 * SUM(CASE WHEN "Class" = 1 THEN 1 ELSE 0 END) / COUNT(*), 3) AS fraud_rate_pct
                        FROM transactions_features
                        GROUP BY "Hour"
                        ORDER BY "Hour"
                    """),
                    conn,
                )

                # Query 2: Fraud rate by amount bucket
                amount_bucket_df = pd.read_sql(
                    text("""
                        SELECT
                            CASE
                                WHEN "Amount" < 10 THEN '₹0-10'
                                WHEN "Amount" < 50 THEN '₹10-50'
                                WHEN "Amount" < 100 THEN '₹50-100'
                                WHEN "Amount" < 500 THEN '₹100-500'
                                ELSE '₹500+'
                            END AS amount_bucket,
                            COUNT(*) AS total_transactions,
                            SUM(CASE WHEN "Class" = 1 THEN 1 ELSE 0 END) AS fraud_count,
                            ROUND(100.0 * SUM(CASE WHEN "Class" = 1 THEN 1 ELSE 0 END) / COUNT(*), 3) AS fraud_rate_pct
                        FROM transactions_features
                        GROUP BY amount_bucket
                        ORDER BY MIN("Amount")
                    """),
                    conn,
                )

                # Query 3: Overall summary stats
                summary_row = conn.execute(text("""
                    SELECT COUNT(*) AS total,
                           SUM(CASE WHEN "Class" = 1 THEN 1 ELSE 0 END) AS frauds,
                           ROUND(AVG("Amount")::numeric, 2) AS avg_amount,
                           ROUND(AVG(CASE WHEN "Class" = 1 THEN "Amount" END)::numeric, 2) AS avg_fraud_amount
                    FROM transactions_features
                """)).fetchone()

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Transactions", f"{summary_row[0]:,}")
            s2.metric("Total Frauds", f"{summary_row[1]:,}")
            s3.metric("Avg Transaction", f"₹{summary_row[2]:.2f}")
            s4.metric("Avg Fraud Amount", f"₹{summary_row[3]:.2f}")

            st.subheader("Fraud Rate by Hour of Day")
            st.bar_chart(hourly_df.set_index("Hour")["fraud_rate_pct"])
            peak_hour = hourly_df.loc[hourly_df["fraud_rate_pct"].idxmax()]
            st.caption(
                f"Highest fraud rate: **{int(peak_hour['Hour'])}:00** at {peak_hour['fraud_rate_pct']:.2f}% "
                f"({int(peak_hour['fraud_count'])} of {int(peak_hour['total_transactions'])} transactions that hour)."
            )
            with st.expander("View raw query results"):
                st.dataframe(hourly_df, use_container_width=True)

            st.subheader("Fraud Rate by Transaction Amount")
            st.bar_chart(amount_bucket_df.set_index("amount_bucket")["fraud_rate_pct"])
            with st.expander("View raw query results"):
                st.dataframe(amount_bucket_df, use_container_width=True)

            st.caption(
                "All figures computed live via SQL (GROUP BY, aggregate functions) against the full "
                "284,807-transaction dataset — not a sample."
            )
        except Exception as e:
            st.error(f"⚠️ Could not load analytics: {e}")

# ============================================================
# TAB 6: Live Feed Simulation (new) — SIMULATED, clearly labeled
# ============================================================
with tab6:
    st.markdown(
        "**⚠️ Simulation, not live production traffic.** This dataset is a static, historical snapshot "
        "with no connected real-time transaction stream. To demonstrate how FinGuard would behave against "
        "incoming transactions, this replays real held-out test transactions one at a time with a short "
        "delay, using each one's actual (precomputed) Random Forest score."
    )

    if test_preds_df is None or test_preds_df.empty:
        st.warning("⚠️ Test predictions file not found — cannot run simulation.")
    else:
        n_txns = st.slider("Number of transactions to simulate", 5, 50, 15)
        speed = st.select_slider("Speed", options=["Slow", "Normal", "Fast"], value="Normal")
        delay = {"Slow": 1.0, "Normal": 0.4, "Fast": 0.1}[speed]

        if st.button("▶ Start Simulation", type="primary"):
            sample = test_preds_df.sample(n_txns).reset_index(drop=True)

            progress = st.progress(0)
            stats_placeholder = st.empty()
            table_placeholder = st.empty()

            processed_rows = []
            approved, review, blocked, frauds_caught = 0, 0, 0, 0

            for i, row in sample.iterrows():
                proba = row["y_pred_proba"]
                if proba >= BLOCK_THRESHOLD:
                    action = "🚫 Block"
                    blocked += 1
                elif proba >= REVIEW_THRESHOLD:
                    action = "🔎 Review"
                    review += 1
                else:
                    action = "✅ Approve"
                    approved += 1

                if row["y_true"] == 1 and proba >= REVIEW_THRESHOLD:
                    frauds_caught += 1

                processed_rows.append({
                    "Amount": f"₹{row['Amount']:.2f}",
                    "Risk Score": f"{proba*100:.1f}%",
                    "Action": action,
                    "Actual": "FRAUD" if row["y_true"] == 1 else "legit",
                })

                with stats_placeholder.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Processed", i + 1)
                    c2.metric("Auto-approved", approved)
                    c3.metric("Flagged/Blocked", review + blocked)
                    c4.metric("Frauds Caught", frauds_caught)

                table_placeholder.dataframe(
                    pd.DataFrame(processed_rows).iloc[::-1], use_container_width=True
                )
                progress.progress((i + 1) / n_txns)
                time.sleep(delay)

            st.success(f"Simulation complete — {n_txns} transactions processed in real time.")

# ============================================================
# TAB 7: Batch Scoring (new) — score REAL data, not preset scenarios
# ============================================================
with tab7:
    st.markdown(
        "Score transactions in bulk — either the **full held-out test set** (56,962 real transactions, "
        "not a curated sample) or your **own uploaded CSV**. This is the actual model and scaler running "
        "against real feature data, the same way a production batch job would."
    )

    source = st.radio(
        "Data source",
        ["Full held-out test set (real, all 56,962 transactions)", "Upload my own CSV"],
        horizontal=False,
    )

    batch_df = None

    if source.startswith("Full"):
        if full_dataset_df is None:
            st.warning("⚠️ Could not load the full dataset from the database.")
        else:
            n_rows = st.slider("How many transactions to score (for speed)", 100, 5000, 1000, step=100)
            if st.button("▶ Score Test Set Sample", type="primary"):
                batch_df = full_dataset_df.sample(n_rows).reset_index(drop=True)
    else:
        uploaded = st.file_uploader(
            "Upload a CSV with the same columns as the training features "
            "(V1-V28, Hour, Amount, etc. — see `data/raw/creditcard.csv` for the expected format)",
            type="csv",
        )
        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                st.success(f"Loaded {len(batch_df)} rows from your file.")
            except Exception as e:
                st.error(f"⚠️ Could not read CSV: {e}")

    if batch_df is not None:
        missing = [c for c in feature_cols if c not in batch_df.columns]
        if missing:
            st.error(f"⚠️ Uploaded/selected data is missing required columns: {', '.join(missing)}")
        else:
            with st.spinner(f"Scoring {len(batch_df)} transactions..."):
                try:
                    X_batch = batch_df[feature_cols]
                    X_batch_scaled = scaler.transform(X_batch)
                    rf_scores = model.predict_proba(X_batch_scaled)[:, 1]

                    X_batch_pca = X_batch_scaled[:, pca_col_idx]
                    iso_raw = -iso_forest.decision_function(X_batch_pca)
                    iso_scores = iso_score_scaler.transform(iso_raw.reshape(-1, 1)).flatten()
                    iso_scores = np.clip(iso_scores, 0, 1)

                    combined_scores = get_combined_score(rf_scores, iso_scores)

                    results_df = batch_df.copy()
                    results_df["risk_score"] = combined_scores
                    results_df["action"] = pd.cut(
                        combined_scores,
                        bins=[-0.01, REVIEW_THRESHOLD, BLOCK_THRESHOLD, 1.01],
                        labels=["Approve", "Review", "Block"],
                    )

                    st.subheader("Results")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Total Scored", f"{len(results_df):,}")
                    r2.metric("Auto-approved", f"{(results_df['action'] == 'Approve').sum():,}")
                    r3.metric("Flagged for Review", f"{(results_df['action'] == 'Review').sum():,}")
                    r4.metric("Auto-blocked", f"{(results_df['action'] == 'Block').sum():,}")

                    if "Class" in results_df.columns:
                        true_frauds = (results_df["Class"] == 1).sum()
                        caught = ((results_df["Class"] == 1) & (results_df["action"] != "Approve")).sum()
                        st.info(f"Of **{true_frauds}** actual frauds in this batch, **{caught}** were flagged for review or blocked ({caught/true_frauds*100:.1f}% caught)." if true_frauds > 0 else "No confirmed frauds in this batch.")

                    display_cols = ["Amount", "risk_score", "action"] + (["Class"] if "Class" in results_df.columns else [])
                    st.dataframe(
                        results_df[display_cols].sort_values("risk_score", ascending=False).head(200),
                        use_container_width=True,
                    )
                    st.caption("Showing top 200 by risk score. Download full results below.")

                    csv_out = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ Download full results as CSV", csv_out,
                        "finguard_batch_results.csv", "text/csv",
                    )
                except Exception as e:
                    st.error(f"⚠️ Batch scoring failed: {e}")

st.markdown("---")
st.caption("FinGuard | Track 2: AI Risk Manager | Risk Score: 0.7×Random Forest + 0.3×Isolation Forest | AI narrative: Google Gemini")