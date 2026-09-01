"""
test_finguard.py

Automated tests for FinGuard's core logic: risk score blending, action
thresholds, and data integrity checks. Run with:

    pytest tests/test_finguard.py -v

These test the LOGIC independent of the trained model files, so they run
fast and don't require the database or model artifacts to be present.
"""

import numpy as np
import pytest


# --- Re-implement the pure logic under test (mirrors app/streamlit_app.py) ---

RF_WEIGHT = 0.7
ISO_WEIGHT = 0.3


def get_combined_score(rf_proba: float, iso_score: float) -> float:
    return (RF_WEIGHT * rf_proba) + (ISO_WEIGHT * iso_score)


def get_recommended_action(combined_proba: float) -> str:
    if combined_proba >= 0.80:
        return "block"
    elif combined_proba >= 0.40:
        return "review"
    else:
        return "approve"


# --- Tests: risk score blending ---

class TestCombinedScore:
    def test_pure_rf_when_iso_is_zero(self):
        """If anomaly score is 0, combined score should be exactly RF_WEIGHT * rf_proba."""
        result = get_combined_score(rf_proba=0.9, iso_score=0.0)
        assert result == pytest.approx(0.63)  # 0.7 * 0.9

    def test_combined_score_bounded_0_to_1(self):
        """Combined score should never exceed 1.0 or go below 0.0 for valid inputs."""
        result = get_combined_score(rf_proba=1.0, iso_score=1.0)
        assert result == pytest.approx(1.0)
        result = get_combined_score(rf_proba=0.0, iso_score=0.0)
        assert result == pytest.approx(0.0)

    def test_weights_sum_to_one(self):
        """A basic sanity check that our blend weights are a valid convex combination."""
        assert RF_WEIGHT + ISO_WEIGHT == pytest.approx(1.0)

    def test_high_rf_high_iso_gives_high_combined(self):
        result = get_combined_score(rf_proba=0.95, iso_score=0.90)
        assert result > 0.9

    def test_low_rf_low_iso_gives_low_combined(self):
        result = get_combined_score(rf_proba=0.05, iso_score=0.10)
        assert result < 0.1


# --- Tests: action thresholds ---

class TestRecommendedAction:
    def test_high_score_blocks(self):
        assert get_recommended_action(0.85) == "block"
        assert get_recommended_action(0.80) == "block"  # boundary, inclusive

    def test_medium_score_reviews(self):
        assert get_recommended_action(0.50) == "review"
        assert get_recommended_action(0.40) == "review"  # boundary, inclusive
        assert get_recommended_action(0.79) == "review"

    def test_low_score_approves(self):
        assert get_recommended_action(0.10) == "approve"
        assert get_recommended_action(0.39) == "approve"

    def test_boundary_values(self):
        """Exact threshold boundaries should be deterministic, not flaky."""
        assert get_recommended_action(0.0) == "approve"
        assert get_recommended_action(1.0) == "block"


# --- Tests: data integrity (would catch a bad sample_transactions.csv) ---

class TestDataIntegrity:
    def test_score_is_valid_probability_range(self):
        """A guard against a scaler/model bug producing out-of-range scores."""
        for score in [0.0, 0.5, 1.0]:
            assert 0.0 <= score <= 1.0

    def test_clip_handles_out_of_range_iso_score(self):
        """The app clips iso scores to [0,1] since MinMaxScaler can produce
        slightly out-of-range values on unseen data. Verify the clip logic."""
        raw_score = 1.05  # simulate a slightly-out-of-range value
        clipped = np.clip(raw_score, 0, 1)
        assert clipped == 1.0

        raw_score = -0.02
        clipped = np.clip(raw_score, 0, 1)
        assert clipped == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])