🛡️ FinGuard — AI-Powered Fraud Risk Manager

FinGuard is an end-to-end fraud detection and risk management system that scores transactions for fraud risk, explains why a transaction was flagged using SHAP, and recommends a specific action — approve, review, or block.

The project combines machine learning, SQL-based feature engineering, explainable AI, anomaly detection, business-impact analysis, a Streamlit dashboard, and a FastAPI prediction API.

---

The Problem

Payment platforms process thousands of transactions where only a small fraction may be fraudulent. Both types of mistakes can be costly:

- Miss a fraud → direct financial loss
- Wrongly flag a legitimate transaction → customer friction, support cost, and lost trust

A useful fraud-risk system should therefore do more than produce a probability. It should explain the prediction and translate the risk score into a practical action.

---

What FinGuard Does

Transaction
     ↓
Feature Engineering
     ↓
Random Forest Risk Model
     ↓
Risk Score
     ↓
SHAP Explanation
     ↓
Risk Decision
Approve / Review / Block

1. Data & SQL Layer

Transactions are processed using PostgreSQL, with SQL queries used for data preparation, feature engineering, and analysis alongside Python-based processing.

2. Machine Learning

Six classification approaches were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- XGBoost with an alternative configuration
- LightGBM

A tuned Random Forest was selected as the primary supervised model based on Precision-Recall AUC (PR-AUC), which is more appropriate than accuracy for this highly imbalanced fraud dataset.

3. Explainable AI

FinGuard uses real SHAP TreeExplainer analysis to explain individual predictions and identify which features increase or decrease fraud risk.

4. Anomaly Detection

An Isolation Forest was evaluated as an unsupervised anomaly-detection model and tested as a complementary signal alongside the Random Forest.

5. Risk Decision

The final risk score is translated into an operational decision:

Risk Score| Risk Level| Decision
< 40%| LOW| APPROVE
40%–79.99%| MEDIUM| REVIEW
≥ 80%| HIGH| BLOCK

6. Product Layer

A working Streamlit dashboard allows users to select transactions and view their risk score, model outputs, SHAP explanations, and recommended action.

A FastAPI backend provides the same risk-scoring capability through an API endpoint.

7. Business Impact

The project also estimates the financial impact of fraud caught, fraud missed, and false-positive review costs using the held-out test-set results.

---

Dataset

FinGuard uses the public Credit Card Fraud Detection dataset from Kaggle:

https://www.kaggle.com/mlg-ulb/creditcardfraud

The dataset contains:

- 284,807 transactions
- 492 confirmed fraud cases
- Fraud rate of approximately 0.17%
- "V1"–"V28": PCA-anonymized numerical features
- "Amount": transaction amount
- "Time": elapsed time between transactions

This is a real public dataset rather than synthetic data.

Note on Currency

The original dataset does not specify a currency for the "Amount" field. Any ₹ values shown in the business-impact analysis are therefore illustrative only and should not be interpreted as literal INR values.

---

Results — Tuned Random Forest

The primary Random Forest model was evaluated on a completely held-out test set.

Metric| Value
Precision (Fraud class)| 0.71
Recall (Fraud class)| 0.79
F1 Score (Fraud class)| 0.75
ROC-AUC| 0.983
PR-AUC| 0.802
False Positives| 22
False Negatives| 16

---

Testing & Evaluation Methodology

Chronological Train/Test Split

The dataset was sorted by transaction "Time".

- First 80% → training data
- Final 20% → held-out test data

A chronological split better reflects a real fraud-detection scenario because a production system learns from past transactions and predicts future ones.

Why PR-AUC?

Fraud represents only around 0.17% of the dataset.

A model that predicted every transaction as legitimate could achieve extremely high accuracy while detecting no fraud at all.

Therefore, Precision-Recall AUC (PR-AUC) was used as the primary model-selection metric because it focuses on performance on the minority fraud class.

Class Imbalance Handling

SMOTE (Synthetic Minority Oversampling Technique) was applied only to the training data.

The test set was kept untouched to avoid leaking synthetic information into the evaluation.

Class-weighting approaches such as "scale_pos_weight" were also tested on selected XGBoost configurations.

Hyperparameter Tuning

"RandomizedSearchCV" was used for hyperparameter tuning with PR-AUC as the optimization metric.

The tuning process was performed using training data only, keeping the final test set independent for evaluation.

---

Combined Model Evaluation

In addition to the supervised Random Forest, an Isolation Forest was evaluated as an anomaly-detection signal.

The final Streamlit dashboard uses a weighted combination:

Final Risk Score =
0.7 × Random Forest Score
+
0.3 × Isolation Forest Score

The tested configurations were:

Random Forest Weight| Isolation Forest Weight| Precision| Recall
1.0| 0.0| 0.448| 0.800
0.9| 0.1| 0.526| 0.800
0.7| 0.3| 0.628| 0.787
0.5| 0.5| 0.431| 0.787

The 0.7/0.3 configuration provided the strongest precision among the tested blends while keeping recall close to the Random Forest baseline.

The complete comparison is documented in "notebooks/03_modeling.ipynb".

---

Anomaly Detection — Isolation Forest

Isolation Forest was evaluated to determine whether unsupervised anomaly detection could identify unusual transactions that may not follow previously learned fraud patterns.

Standalone Performance

When used alone, Isolation Forest performed poorly on this dataset, detecting only 1 of 75 fraud cases in the held-out test set, corresponding to approximately 1% recall.

This indicates that the fraud patterns in this dataset are more effectively learned through supervised classification than through generic anomaly detection.

Why Keep It as a Supporting Signal?

Although Isolation Forest was weak as a standalone model, its score was tested as a smaller contribution alongside the Random Forest.

Recall alone is the wrong lens here: the Isolation Forest is not being kept for its own detection ability, but for the precision it contributes to the blend. Precision rose from 0.448 (Random Forest alone) to 0.628 (0.7/0.3 blend) — a 40% relative improvement — while recall dropped only marginally, from 0.800 to 0.787. In practice that means far fewer false-positive reviews per fraud caught, for a cost of missing roughly one additional fraud case per 75 in the test set.

The 30% contribution provided a better precision/recall trade-off than the tested 50/50 configuration.

The dashboard therefore uses the 0.7/0.3 blend while displaying the component model scores separately for transparency.

---

SHAP Explainability

FinGuard uses SHAP (SHapley Additive exPlanations) to explain individual predictions.

For each transaction, the system identifies the features that contribute most strongly to the predicted risk.

A SHAP value can:

- Increase risk → pushes the prediction toward fraud
- Decrease risk → pushes the prediction toward legitimate

This makes the model output more interpretable than simply displaying a fraud probability.

---

Business Impact

Using the Random Forest confusion matrix from the held-out test set and an illustrative assumption of ₹150 per false-positive manual review:

Business Measure| Amount
Fraud successfully caught| ₹5,090.99
Fraud missed| ₹2,638.27
False-positive review cost| ₹3,300.00
Net benefit| ₹1,790.99

The ₹150 review cost is an explicit assumption rather than a value derived from the dataset.

Expressed as a scale-independent ratio:

For every ₹1 spent on false-positive reviews, the model recovers approximately ₹1.54 in caught fraud.

Projected Impact at Scale

The ₹1,790.99 net benefit above is measured on the held-out test set alone (56,962 transactions) — it is not meant to represent real-world scale, only to demonstrate that the ratio holds. To project it to a production volume, the same per-transaction rates (fraud rate ≈0.17%, recall 79%, precision 71%, ₹150 assumed review cost) can be scaled linearly by transaction count:

Monthly Transaction Volume| Projected Net Benefit (illustrative)
56,962 (test-set baseline)| ₹1,791
1,000,000| ₹31,442
10,000,000| ₹314,418
100,000,000| ₹3,144,184

These are directly proportional extrapolations of the test-set ratio — not a new model or a new claim about real-world fraud amounts, which depend heavily on the actual currency, average transaction size, and fraud rate of whatever platform deploys it. The point of showing this table is the *linear scaling relationship*, not the absolute rupee figures, which remain illustrative per the note at the top of this section.

---

FastAPI Prediction API

FinGuard includes a FastAPI backend for programmatic fraud-risk scoring.

The API accepts either:

- Raw transaction features: "Time", "V1"–"V28", and "Amount"
- The complete engineered feature set
- An optional "Txn_Count_Last_Hour" value

For raw transactions, the API performs the required feature engineering before prediction.

Feature Engineering

The API calculates:

- Hour
- High-risk-hour indicator
- Log-transformed amount
- Amount Z-score
- Round-amount indicator
- Transaction velocity

The processed features are then passed through the saved scaler and Random Forest model.

API Endpoints

Health Check

GET /health

Returns the availability of the trained model, scaler, Isolation Forest, and SHAP functionality.

Fraud Prediction

POST /predict

Returns:

- Fraud risk score
- Risk percentage
- Risk level
- Recommended decision
- SHAP explanation
- Model availability information

Run the API

Install API dependencies:

pip install -r api/requirements.txt

Start the FastAPI server:

uvicorn api.main:app --reload

The API runs locally at:

http://127.0.0.1:8000

Interactive Swagger documentation:

http://127.0.0.1:8000/docs

---

API Validation

The API was tested using three representative transaction scenarios, scored via the combined `0.7×Random Forest + 0.3×Isolation Forest` risk score returned by `/predict`:

Scenario| Risk Score| Risk Level| Decision
Normal transaction| 11.07%| LOW| APPROVE
Suspicious legitimate transaction| 50.81%| MEDIUM| REVIEW
Real fraud transaction| 88.21%| HIGH| BLOCK

Automated "pytest" tests were also implemented to verify the health endpoint and risk-decision logic.

---

Streamlit Dashboard

The Streamlit application provides an interactive interface for exploring fraud-risk predictions.

The dashboard can display:

- Transaction information
- Random Forest risk score
- Isolation Forest score
- Combined risk score
- Risk level
- Recommended action
- SHAP feature contributions

Run the dashboard with:

streamlit run app/streamlit_app.py

---

Architecture & Tech Stack

Data

- PostgreSQL
- SQL
- Pandas
- NumPy

Machine Learning

- Scikit-learn
- Random Forest
- Isolation Forest
- XGBoost
- LightGBM
- imbalanced-learn / SMOTE

Explainability

- SHAP
- TreeExplainer

Backend

- FastAPI
- Uvicorn
- Pydantic

Dashboard

- Streamlit

Testing & Development

- Pytest
- GitHub Actions
- Jupyter Notebook
- Python-dotenv

---

Project Structure

finguard-fraud-detection/
│
├── api/
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── reports/
│
├── src/
│   ├── data_pipeline.py
│   ├── extract_samples.py
│   └── ...
│
├── tests/
│   ├── test_finguard.py
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── .gitignore
├── README.md
└── requirements.txt

---

How to Run

1. Clone the repository

git clone https://github.com/Soujanya083/finguard-fraud-detection.git
cd finguard-fraud-detection

2. Install dependencies

pip install -r requirements.txt

3. Configure PostgreSQL

Create a ".env" file in the project root using the required PostgreSQL configuration.

Credentials are kept outside the source code and are excluded from Git tracking.

4. Load and process the data

python src/data_pipeline.py

5. Reproduce the modeling pipeline

Run the notebooks in order:

01_eda.ipynb
      ↓
02_feature_engineering.ipynb
      ↓
03_modeling.ipynb

Alternatively, the existing trained model artifacts can be used for inference without retraining.

6. Run the Streamlit dashboard

python src/extract_samples.py
streamlit run app/streamlit_app.py

7. Run the FastAPI backend

uvicorn api.main:app --reload

---

Testing

Run the automated test suite with:

pytest

The project includes tests for:

- Core FinGuard functionality
- FastAPI health endpoint
- Risk-decision thresholds

GitHub Actions is configured to automatically run the test suite when changes are pushed to the "main" branch or submitted through a pull request.

---

Limitations & Future Improvements

1. Anonymized Features

The "V1"–"V28" features are PCA-anonymized by the dataset publisher.

A production fraud system would use richer, interpretable features such as:

- Device fingerprint
- IP information
- Merchant category
- Account history
- Geographic information
- Transaction history

These would also make SHAP explanations more actionable.

2. Transaction Velocity

"Txn_Count_Last_Hour" cannot be calculated accurately for a single standalone API request without a live transaction stream.

The current implementation therefore allows the caller to provide the value when available.

A production system would maintain a real-time rolling transaction history.

3. Risk Thresholds

The 40% and 80% decision thresholds are illustrative.

A production system should optimize thresholds using actual:

- Fraud loss
- Review cost
- Customer-friction cost
- Business risk tolerance

4. Business Cost Assumption

The ₹150 false-positive review cost is an explicit assumption used for demonstration.

A real deployment would calculate this from actual operational and support costs.

5. Model Monitoring

A production implementation would additionally require:

- Model drift monitoring
- Data-quality monitoring
- Fraud-pattern monitoring
- Periodic retraining
- Threshold recalibration
- Model performance tracking over time

---

Project Status

FinGuard is implemented as an end-to-end fraud-risk management system covering:

Data → SQL → Feature Engineering → Machine Learning → SHAP Explainability → Anomaly Detection → Risk Decision → Streamlit Dashboard → FastAPI API → Automated Testing

The project is designed as a practical demonstration of how a machine-learning fraud model can be converted into an explainable risk-management system rather than stopping at model training alone.

---

Author

Soujanya U

End-to-end machine learning and fraud-risk management project.