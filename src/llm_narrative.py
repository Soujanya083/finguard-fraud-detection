"""
llm_narrative.py

Generates a natural-language fraud risk narrative using Google's Gemini API,
based on the model's risk score and top SHAP-driven features for a transaction.

This is a REAL API call, not a stubbed/mocked function -- it requires a valid
GEMINI_API_KEY in .env to run.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
_model = genai.GenerativeModel("gemini-3.6-flash")


def generate_fraud_narrative(risk_probability: float, top_features: list, amount: float, recommended_action: str) -> str:
    """
    Generate a plain-language fraud risk narrative + reasoning.

    Args:
        risk_probability: model's fraud probability (0-1)
        top_features: list of (feature_name, shap_value) tuples, the top
            SHAP-driving features for this specific transaction, most
            important first
        amount: transaction amount
        recommended_action: the rule-based action already decided
            (e.g. "Auto-block & escalate", "Flag for manual review", "Auto-approve")

    Returns:
        A short natural-language explanation string. Falls back to a
        clear error message (not a crash) if the API call fails, so a
        transient network/API issue never takes down the whole app.
    """
    feature_summary = ", ".join(f"{name} (impact: {val:+.3f})" for name, val in top_features)

    prompt = f"""You are a fraud risk analyst assistant. Write a SHORT (2-3 sentences max),
plain-language summary of why a transaction was flagged, for a non-technical reviewer.

Transaction details:
- Fraud risk probability: {risk_probability*100:.1f}%
- Transaction amount: {amount:.2f}
- Top contributing model features (from SHAP explainability, anonymized PCA components): {feature_summary}
- System's recommended action: {recommended_action}

Write a concise, professional explanation a fraud investigator could read in 5 seconds.
Do not repeat the raw feature names verbatim (like "V14") since they are anonymized and meaningless
to a human reader -- instead, describe the *pattern* in general terms (e.g. "an unusual combination
of transaction characteristics not typical of this account's normal behavior").
Do not invent facts not given above."""

    try:
        response = _model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return (
            f"⚠️ AI narrative unavailable right now ({e}). "
            f"Risk score and recommended action above are still valid and unaffected."
        )