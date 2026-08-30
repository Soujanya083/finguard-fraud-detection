# 🛡️ FinGuard — AI-Powered Fraud Risk Manager

**Razorpay AI Buildathon 2026 — Track 2: AI Risk Manager**

FinGuard is an end-to-end fraud detection system that scores incoming transactions for fraud risk, explains *why* a transaction was flagged using SHAP, and recommends a specific action (auto-approve, manual review, or auto-block) — not just a probability number.

---

## The Problem

Payment platforms process thousands of transactions where a small fraction are fraudulent. Two failure modes cost money either way:
- **Miss a fraud** → direct financial loss
- **Wrongly flag a legitimate transaction** → customer friction, support cost, lost trust

A useful risk system has to do more than predict — it needs to explain its reasoning (for audit/trust) and translate a score into a concrete next action.

## What FinGuard Does

```
Transaction → Feature Engineering → ML Risk Model → Risk Score
                                                          ↓
                                        SHAP Explanation (why this score?)
                                                          ↓
                                   Recommended Action (approve / review / block)
```

1. **Data & SQL layer**: Transactions are stored in PostgreSQL; feature engineering and querying happen via SQL, not just pandas.
2. **Model**: 6 models were trained and compared (Logistic Regression, Decision Tree, Random Forest, 2x XGBoost variants, LightGBM). The best performer, a **tuned Random Forest**, was selected via `RandomizedSearchCV` optimizing PR-AUC (the right metric for a ~0.17%-fraud, heavily imbalanced dataset).
3. **Explainability**: Real `shap.TreeExplainer` — both global feature importance and per-transaction explanations, not a stand-in.
4. **Product**: A working Streamlit app lets you pick a real, held-out transaction and see its live risk score, SHAP explanation, and recommended action.
5. **Business case**: False-positive cost and fraud-prevented value are calculated directly from the confusion matrix on the held-out test set.

## Dataset

[Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) — 284,807 real anonymized transactions, 492 confirmed frauds (~0.17%). Features V1–V28 are PCA-anonymized for privacy by the original dataset publisher; `Amount` and `Time` are the only non-anonymized fields. This is a real public dataset, not synthetic data.

**Note on currency**: the dataset does not specify a currency for the `Amount` field. This project displays amounts with a ₹ symbol for illustrative purposes only — treat all monetary figures below as relative/illustrative, not literal INR values.

## Results (Tuned Random Forest, held-out test set)

| Metric | Value |
|---|---|
| Precision (Fraud class) | 0.71 |
| Recall (Fraud class) | 0.79 |
| F1 (Fraud class) | 0.75 |
| ROC-AUC | 0.983 |
| PR-AUC | 0.802 |
| False Positives | 22 |
| False Negatives | 16 |

## Business Impact (held-out test set, illustrative)

Using the confusion matrix above and an assumed ₹150 cost per false-positive review:

| | Amount |
|---|---|
| Fraud successfully caught | ₹5,090.99 |
| Fraud missed | ₹2,638.27 |
| False-positive review cost (22 × ₹150) | ₹3,300.00 |
| **Net benefit** | **₹1,790.99** |

*The ₹150-per-false-positive figure is an assumption (estimated manual review/support cost), not derived from the dataset — stated explicitly rather than presented as fact.* Framed as a ratio (scale-independent): **for every ₹1 spent on false-positive reviews, the model recovers ~₹1.54 in caught fraud.**

## Architecture & Tech Stack

- **Data storage**: PostgreSQL (SQL queries for feature engineering, not just pandas)
- **ML**: scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE)
- **Explainability**: SHAP (`TreeExplainer`)
- **Dashboard**: Streamlit
- **Environment/secrets**: python-dotenv (`.env`, gitignored — never hardcoded credentials)

## How to Run

1. Clone this repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the project root (see `.env.example`) with your PostgreSQL credentials.
3. Load the raw dataset into PostgreSQL:
   ```
   python src/data_pipeline.py
   ```
4. Run the notebooks in order (`01_eda.ipynb` → `02_feature_engineering.ipynb` → `03_modeling.ipynb`) to reproduce the full pipeline, or use the pre-trained model in `models/`.
5. Generate demo scenarios and launch the dashboard:
   ```
   python src/extract_samples.py
   streamlit run app/streamlit_app.py
   ```

## Limitations & What I'd Do Differently at Scale

- V1-V28 features are PCA-anonymized by the dataset publisher — real deployment would use interpretable raw features (merchant category, device fingerprint, IP geolocation, etc.) for richer, more actionable SHAP explanations.
- The ₹150 false-positive cost is a stated assumption, not derived from real operational data — a production system would calibrate this from actual support/review costs.
- `Txn_Count_Last_Hour` (a velocity feature) can't be computed live for a single new transaction in the current demo without a running transaction stream — it's approximated. A production system would maintain a real-time rolling window per account.
- Action thresholds (80%/40%) are illustrative starting points, not tuned against a real cost-benefit curve.

## Author

Built by Soujanya U as a submission for Razorpay AI Buildathon 2026, Track 2 (AI Risk Manager).