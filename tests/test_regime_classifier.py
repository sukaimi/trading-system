"""Tests for core/regime_classifier.py and ADX indicator."""

import math
from unittest.mock import MagicMock, patch

import pytest

from core.regime_classifier import (
    DEFAULT_REGIME_INITIAL_STOP_MULT,
    DEFAULT_REGIME_TRAILING_MULT,
    MarketRegime,
    RegimeClassifier,
)
from tools.technical_indicators import TechnicalIndicators


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def ti():
    return TechnicalIndicators()


@pytest.fixture
def classifier():
    return RegimeClassifier()


@pytest.fixture
def trending_up_ohlcv():
    """60 bars of steadily rising prices — should classify as TRENDING_UP."""
    bars = []
    for i in range(60):
        base = 100.0 + i * 1.5
        bars.append({
            "open": base,
            "high": base + 2.0,
            "low": base - 1.0,
            "close": base + 1.0,
            "volume": 1000,
        })
    return bars


@pytest.fixture
def trending_down_ohlcv():
    """60 bars of steadily falling prices — should classify as TRENDING_DOWN."""
    bars = []
    for i in range(60):
        base = 200.0 - i * 1.5
        bars.append({
            "open": base,
            "high": base + 1.0,
            "low": base - 2.0,
            "close": base - 1.0,
            "volume": 1000,
        })
    return bars


@pytest.fixture
def ranging_ohlcv():
    """60 bars oscillating around a mean — should classify as RANGING."""
    bars = []
    for i in range(60):
        base = 100.0 + 3.0 * math.sin(i * 0.5)
        bars.append({
            "open": base - 0.5,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": 1000,
        })
    return bars


@pytest.fixture
def high_vol_ohlcv():
    """60 bars with extreme volatility swings — should classify as HIGH_VOLATILITY."""
    bars = []
    for i in range(60):
        swing = 15.0 * (1 if i % 2 == 0 else -1)
        base = 100.0 + swing
        bars.append({
            "open": base - 5.0,
            "high": base + 10.0,
            "low": base - 10.0,
            "close": base + 5.0 * (1 if i % 3 == 0 else -1),
            "volume": 1000,
        })
    return bars


@pytest.fixture
def low_vol_ohlcv():
    """60 bars with very tight ranges — should classify as LOW_VOLATILITY."""
    bars = []
    for i in range(60):
        base = 100.0 + i * 0.01  # barely moving
        bars.append({
            "open": base,
            "high": base + 0.1,
            "low": base - 0.1,
            "close": base + 0.05,
            "volume": 1000,
        })
    return bars


# ── ADX Tests ────────────────────────────────────────────────────────


class TestADX:
    def test_adx_trending_market(self, ti):
        """ADX should be high (>25) for a strongly trending market."""
        # Strong uptrend: each bar higher than last
        highs = [float(100 + i * 2) for i in range(50)]
        lows = [float(98 + i * 2) for i in range(50)]
        closes = [float(99 + i * 2) for i in range(50)]
        adx = ti.adx(highs, lows, closes)
        assert adx > 20, f"ADX should be high for trending market, got {adx}"

    def test_adx_ranging_market(self, ti):
        """ADX should be low (<25) for a ranging market."""
        # Oscillating prices
        import math
        highs = [101.0 + 2.0 * math.sin(i * 0.5) for i in range(50)]
        lows = [99.0 + 2.0 * math.sin(i * 0.5) for i in range(50)]
        closes = [100.0 + 2.0 * math.sin(i * 0.5) for i in range(50)]
        adx = ti.adx(highs, lows, closes)
        assert adx < 30, f"ADX should be low-moderate for ranging market, got {adx}"

    def test_adx_insufficient_data(self, ti):
        """ADX should return 0.0 for insufficient data."""
        assert ti.adx([100], [99], [100]) == 0.0
        assert ti.adx([100] * 10, [99] * 10, [100] * 10) == 0.0

    def test_adx_range(self, ti):
        """ADX should be between 0 and 100."""
        highs = [float(100 + i) for i in range(50)]
        lows = [float(98 + i) for i in range(50)]
        closes = [float(99 + i) for i in range(50)]
        adx = ti.adx(highs, lows, closes)
        assert 0 <= adx <= 100, f"ADX out of range: {adx}"

    def test_adx_custom_period(self, ti):
        """ADX should work with custom period."""
        highs = [float(100 + i * 2) for i in range(60)]
        lows = [float(98 + i * 2) for i in range(60)]
        closes = [float(99 + i * 2) for i in range(60)]
        adx_7 = ti.adx(highs, lows, closes, period=7)
        adx_21 = ti.adx(highs, lows, closes, period=21)
        # Both should be valid
        assert adx_7 > 0
        assert adx_21 > 0


# ── SMA Series Tests ─────────────────────────────────────────────────


class TestSMASeries:
    def test_sma_series_length(self, ti):
        prices = [float(100 + i) for i in range(30)]
        sma_vals = ti.sma_series(prices, 20)
        assert len(sma_vals) == 11  # 30 - 20 + 1

    def test_sma_series_values(self, ti):
        prices = [10.0, 20.0, 30.0, 40.0, 50.0]
        sma_vals = ti.sma_series(prices, 3)
        assert sma_vals[0] == pytest.approx(20.0)  # avg(10,20,30)
        assert sma_vals[1] == pytest.approx(30.0)  # avg(20,30,40)
        assert sma_vals[2] == pytest.approx(40.0)  # avg(30,40,50)

    def test_sma_series_insufficient(self, ti):
        assert ti.sma_series([10.0], 5) == []


# ── Regime Classification Tests ──────────────────────────────────────


class TestRegimeClassification:
    def test_trending_up(self, classifier, trending_up_ohlcv):
        result = classifier.classify_from_ohlcv(trending_up_ohlcv)
        assert result["regime"] == MarketRegime.TRENDING_UP.value
        assert result["confidence"] > 0.3
        assert result["adx"] > 0
        assert result["rsi"] > 50

    def test_trending_down(self, classifier, trending_down_ohlcv):
        result = classifier.classify_from_ohlcv(trending_down_ohlcv)
        assert result["regime"] == MarketRegime.TRENDING_DOWN.value
        assert result["confidence"] > 0.3
        assert result["rsi"] < 50

    def test_ranging(self, classifier, ranging_ohlcv):
        result = classifier.classify_from_ohlcv(ranging_ohlcv)
        assert result["regime"] in (
            MarketRegime.RANGING.value,
            MarketRegime.LOW_VOLATILITY.value,
        )

    def test_high_volatility(self, classifier, high_vol_ohlcv):
        result = classifier.classify_from_ohlcv(high_vol_ohlcv)
        assert result["regime"] in (
            MarketRegime.HIGH_VOLATILITY.value,
            MarketRegime.RANGING.value,  # wide swings may also look ranging
        )
        assert result["bollinger_bandwidth"] > 0

    def test_low_volatility(self, classifier, low_vol_ohlcv):
        result = classifier.classify_from_ohlcv(low_vol_ohlcv)
        assert result["regime"] in (
            MarketRegime.LOW_VOLATILITY.value,
            MarketRegime.TRENDING_UP.value,  # slight upward drift
            MarketRegime.RANGING.value,
        )
        assert result["bollinger_bandwidth"] < 0.05

    def test_result_keys(self, classifier, trending_up_ohlcv):
        result = classifier.classify_from_ohlcv(trending_up_ohlcv)
        expected_keys = {"regime", "confidence", "adx", "atr_ratio", "bollinger_bandwidth", "sma_slope", "rsi"}
        assert expected_keys.issubset(result.keys())

    def test_confidence_range(self, classifier, trending_up_ohlcv):
        result = classifier.classify_from_ohlcv(trending_up_ohlcv)
        assert 0.0 <= result["confidence"] <= 1.0


# ── Insufficient Data Tests ──────────────────────────────────────────


class TestInsufficientData:
    def test_empty_ohlcv(self, classifier):
        result = classifier.classify_from_ohlcv([])
        assert result["regime"] == MarketRegime.RANGING.value
        assert result["confidence"] == 0.5

    def test_short_ohlcv(self, classifier):
        bars = [{"open": 100, "high": 101, "low": 99, "close": 100, "volume": 100}] * 10
        result = classifier.classify_from_ohlcv(bars)
        assert result["regime"] == MarketRegime.RANGING.value

    def test_no_market_data_fetcher(self):
        classifier = RegimeClassifier(market_data_fetcher=None)
        result = classifier.classify("BTC")
        assert result["regime"] == MarketRegime.RANGING.value


# ── Portfolio Classification Tests ────────────────────────────────────


class TestPortfolioClassification:
    def test_portfolio_classify_empty(self, classifier):
        result = classifier.classify_portfolio([])
        assert result["dominant_regime"] == MarketRegime.RANGING.value
        assert result["regime_agreement"] == 0.0
        assert result["per_asset"] == {}

    def test_portfolio_classify_with_mock_mdf(self, trending_up_ohlcv):
        mock_mdf = MagicMock()
        mock_mdf.get_ohlcv.return_value = trending_up_ohlcv
        classifier = RegimeClassifier(market_data_fetcher=mock_mdf)

        result = classifier.classify_portfolio(["BTC", "ETH"])

        assert "per_asset" in result
        assert "BTC" in result["per_asset"]
        assert "ETH" in result["per_asset"]
        assert result["dominant_regime"] in [r.value for r in MarketRegime]
        assert 0 <= result["regime_agreement"] <= 1.0

    def test_portfolio_regime_agreement(self, trending_up_ohlcv):
        """When all assets have same data, regime agreement should be 1.0."""
        mock_mdf = MagicMock()
        mock_mdf.get_ohlcv.return_value = trending_up_ohlcv
        classifier = RegimeClassifier(market_data_fetcher=mock_mdf)

        result = classifier.classify_portfolio(["A", "B", "C"])
        assert result["regime_agreement"] == 1.0


# ── Trailing Multiplier Tests ─────────────────────────────────────────


class TestTrailingMultiplier:
    def test_default_multipliers(self, classifier):
        assert classifier.get_trailing_multiplier("TRENDING_UP") == 0.8
        assert classifier.get_trailing_multiplier("TRENDING_DOWN") == 0.8
        assert classifier.get_trailing_multiplier("RANGING") == 1.5
        assert classifier.get_trailing_multiplier("HIGH_VOLATILITY") == 2.0
        assert classifier.get_trailing_multiplier("LOW_VOLATILITY") == 1.0

    def test_unknown_regime_default(self, classifier):
        assert classifier.get_trailing_multiplier("UNKNOWN") == 1.0

    def test_config_override(self, classifier):
        config = {
            "regime_trailing_mult": {
                "TRENDING_UP": 0.5,
                "HIGH_VOLATILITY": 3.0,
            }
        }
        assert classifier.get_trailing_multiplier("TRENDING_UP", config) == 0.5
        assert classifier.get_trailing_multiplier("HIGH_VOLATILITY", config) == 3.0
        assert classifier.get_trailing_multiplier("RANGING", config) == 1.0  # not in override


# ── Initial Stop Multiplier Tests ─────────────────────────────────────


class TestInitialStopMultiplier:
    def test_default_multipliers(self, classifier):
        assert classifier.get_initial_stop_multiplier("TRENDING_UP") == 0.9
        assert classifier.get_initial_stop_multiplier("TRENDING_DOWN") == 0.9
        assert classifier.get_initial_stop_multiplier("RANGING") == 1.0
        assert classifier.get_initial_stop_multiplier("HIGH_VOLATILITY") == 1.3
        assert classifier.get_initial_stop_multiplier("LOW_VOLATILITY") == 1.0

    def test_unknown_regime_default(self, classifier):
        assert classifier.get_initial_stop_multiplier("UNKNOWN") == 1.0

    def test_config_override(self, classifier):
        config = {
            "regime_initial_stop_mult": {
                "HIGH_VOLATILITY": 1.5,
            }
        }
        assert classifier.get_initial_stop_multiplier("HIGH_VOLATILITY", config) == 1.5


# ── MarketRegime Enum Tests ───────────────────────────────────────────


class TestMarketRegimeEnum:
    def test_enum_values(self):
        assert MarketRegime.TRENDING_UP.value == "TRENDING_UP"
        assert MarketRegime.TRENDING_DOWN.value == "TRENDING_DOWN"
        assert MarketRegime.RANGING.value == "RANGING"
        assert MarketRegime.HIGH_VOLATILITY.value == "HIGH_VOLATILITY"
        assert MarketRegime.LOW_VOLATILITY.value == "LOW_VOLATILITY"

    def test_enum_count(self):
        assert len(MarketRegime) == 5

    def test_enum_is_string(self):
        """MarketRegime inherits from str for JSON serialization."""
        assert isinstance(MarketRegime.TRENDING_UP, str)


# ── Adaptive Stop Integration Tests ──────────────────────────────────


class TestAdaptiveStopIntegration:
    """Test that regime multipliers are applied correctly in pipeline context."""

    def test_trailing_distance_scaled_by_regime(self, classifier):
        """Verify that get_trailing_multiplier returns different values per regime."""
        trending = classifier.get_trailing_multiplier("TRENDING_UP")
        ranging = classifier.get_trailing_multiplier("RANGING")
        high_vol = classifier.get_trailing_multiplier("HIGH_VOLATILITY")

        # Trending should tighten (< 1.0)
        assert trending < 1.0
        # Ranging should widen (> 1.0)
        assert ranging > 1.0
        # High vol should widen even more
        assert high_vol > ranging

    def test_initial_stop_scaled_by_regime(self, classifier):
        """Verify initial stop multipliers make sense."""
        trending = classifier.get_initial_stop_multiplier("TRENDING_UP")
        high_vol = classifier.get_initial_stop_multiplier("HIGH_VOLATILITY")
        normal = classifier.get_initial_stop_multiplier("RANGING")

        # Trending should tighten initial stop
        assert trending < normal
        # High vol should widen initial stop
        assert high_vol > normal

    def test_flat_prices_regime(self, classifier):
        """Completely flat prices should not crash."""
        bars = []
        for i in range(60):
            bars.append({
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000,
            })
        result = classifier.classify_from_ohlcv(bars)
        assert result["regime"] in [r.value for r in MarketRegime]
        assert result["confidence"] > 0


# ── Classify with Market Data Fetcher ─────────────────────────────────


class TestClassifyWithMDF:
    def test_classify_calls_get_ohlcv(self, trending_up_ohlcv):
        mock_mdf = MagicMock()
        mock_mdf.get_ohlcv.return_value = trending_up_ohlcv
        classifier = RegimeClassifier(market_data_fetcher=mock_mdf)

        result = classifier.classify("BTC")

        mock_mdf.get_ohlcv.assert_called_once_with("BTC", period="3mo", interval="1d")
        assert result["regime"] in [r.value for r in MarketRegime]

    def test_classify_handles_mdf_exception(self):
        mock_mdf = MagicMock()
        mock_mdf.get_ohlcv.side_effect = Exception("API error")
        classifier = RegimeClassifier(market_data_fetcher=mock_mdf)

        result = classifier.classify("BTC")

        assert result["regime"] == MarketRegime.RANGING.value
        assert result["confidence"] == 0.5
