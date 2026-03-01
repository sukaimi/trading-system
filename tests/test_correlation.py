"""Tests for tools/correlation.py"""

import pytest

from tools.correlation import CorrelationAnalyzer


@pytest.fixture
def ca():
    return CorrelationAnalyzer()


class TestPairwiseCorrelation:
    def test_perfect_positive(self, ca):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert ca.pairwise_correlation(a, b) == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative(self, ca):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [50.0, 40.0, 30.0, 20.0, 10.0]
        assert ca.pairwise_correlation(a, b) == pytest.approx(-1.0, abs=0.01)

    def test_short_series_returns_zero(self, ca):
        assert ca.pairwise_correlation([1, 2], [3, 4]) == 0.0

    def test_constant_series_returns_zero(self, ca):
        assert ca.pairwise_correlation([5.0] * 10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 0.0

    def test_different_lengths_trimmed(self, ca):
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        b = [10.0, 20.0, 30.0, 40.0, 50.0]
        corr = ca.pairwise_correlation(a, b)
        assert -1.0 <= corr <= 1.0


class TestPortfolioCorrelationMatrix:
    def test_single_position(self, ca):
        result = ca.portfolio_correlation_matrix([{"asset": "BTC"}])
        assert result["max_correlation"] == 0.0

    def test_empty_positions(self, ca):
        result = ca.portfolio_correlation_matrix([])
        assert result["matrix"] == {}


class TestDetectRegime:
    def test_risk_off_high_vix_gold_up(self, ca):
        assert ca.detect_regime({"vix": 30, "gold_change_7d": 2.0}) == "risk_off"

    def test_risk_on_low_vix_crypto_up(self, ca):
        assert ca.detect_regime({"vix": 15, "btc_change_7d": 5.0}) == "risk_on"

    def test_transitional(self, ca):
        assert ca.detect_regime({"vix": 22}) == "transitional"

    def test_unknown_no_data(self, ca):
        assert ca.detect_regime({}) == "unknown"
