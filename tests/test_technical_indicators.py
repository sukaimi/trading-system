"""Tests for tools/technical_indicators.py"""

import pytest

from tools.technical_indicators import TechnicalIndicators


@pytest.fixture
def ti():
    return TechnicalIndicators()


@pytest.fixture
def rising_prices():
    """Steadily rising prices — RSI should be high."""
    return [float(100 + i) for i in range(50)]


@pytest.fixture
def falling_prices():
    """Steadily falling prices — RSI should be low."""
    return [float(150 - i) for i in range(50)]


@pytest.fixture
def flat_prices():
    """Flat prices."""
    return [100.0] * 50


class TestRSI:
    def test_rsi_rising(self, ti, rising_prices):
        rsi = ti.rsi(rising_prices)
        assert rsi > 70, f"RSI should be high for rising prices, got {rsi}"

    def test_rsi_falling(self, ti, falling_prices):
        rsi = ti.rsi(falling_prices)
        assert rsi < 30, f"RSI should be low for falling prices, got {rsi}"

    def test_rsi_insufficient_data(self, ti):
        assert ti.rsi([100, 101, 102]) == 50.0

    def test_rsi_range(self, ti, rising_prices):
        rsi = ti.rsi(rising_prices)
        assert 0 <= rsi <= 100


class TestMACD:
    def test_macd_returns_keys(self, ti, rising_prices):
        result = ti.macd(rising_prices)
        assert "macd_line" in result
        assert "signal_line" in result
        assert "histogram" in result

    def test_macd_insufficient_data(self, ti):
        result = ti.macd([100.0] * 10)
        assert result["macd_line"] == 0.0
        assert result["signal_line"] == 0.0

    def test_macd_rising_positive(self, ti, rising_prices):
        result = ti.macd(rising_prices)
        assert result["macd_line"] > 0, "MACD line should be positive for uptrend"


class TestBollingerBands:
    def test_bb_keys(self, ti, rising_prices):
        result = ti.bollinger_bands(rising_prices)
        assert all(k in result for k in ("upper", "middle", "lower", "bandwidth"))

    def test_bb_upper_above_lower(self, ti, rising_prices):
        result = ti.bollinger_bands(rising_prices)
        assert result["upper"] > result["lower"]

    def test_bb_insufficient_data(self, ti):
        result = ti.bollinger_bands([100.0] * 5, period=20)
        assert result["upper"] == 0.0

    def test_bb_flat_narrow_bands(self, ti, flat_prices):
        result = ti.bollinger_bands(flat_prices)
        assert result["bandwidth"] == pytest.approx(0.0, abs=0.01)


class TestATR:
    def test_atr_positive(self, ti):
        highs = [float(102 + i) for i in range(20)]
        lows = [float(98 + i) for i in range(20)]
        closes = [float(100 + i) for i in range(20)]
        atr = ti.atr(highs, lows, closes)
        assert atr > 0

    def test_atr_insufficient_data(self, ti):
        assert ti.atr([100], [99], [100]) == 0.0


class TestSMA:
    def test_sma_calculation(self, ti):
        prices = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert ti.sma(prices, 5) == pytest.approx(30.0)

    def test_sma_insufficient(self, ti):
        assert ti.sma([10.0, 20.0], 5) == 0.0
