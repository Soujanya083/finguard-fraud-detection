"""
scoring.py

Single source of truth for FinGuard's risk-score blending and decision
thresholds. Both app/streamlit_app.py and api/main.py import from here so
the dashboard and the API can never silently drift apart on the numbers
that decide block / review / approve -- and tests/test_finguard.py imports
the real functions instead of re-implementing them by hand.
"""

RF_WEIGHT = 0.7
ISO_WEIGHT = 0.3

# Project's documented thresholds:
# >= 0.80 -> BLOCK
# >= 0.40 -> REVIEW
# <  0.40 -> APPROVE
BLOCK_THRESHOLD = 0.80
REVIEW_THRESHOLD = 0.40


def get_combined_score(rf_proba: float, iso_score: float) -> float:
    """Blend the Random Forest probability with the normalized Isolation
    Forest anomaly score. Weights (0.7 RF / 0.3 ISO) were chosen based on
    measured precision/recall trade-offs on the held-out test set -- see
    README for the full comparison table."""
    return (RF_WEIGHT * rf_proba) + (ISO_WEIGHT * iso_score)


def get_risk_bucket(combined_score: float):
    """Map a combined score to (risk_level, decision)."""
    if combined_score >= BLOCK_THRESHOLD:
        return "HIGH", "BLOCK"
    if combined_score >= REVIEW_THRESHOLD:
        return "MEDIUM", "REVIEW"
    return "LOW", "APPROVE"
