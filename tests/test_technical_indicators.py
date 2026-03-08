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


class TestLiquiditySweep:
    def test_bullish_sweep_detected(self, ti):
        """Price dips below recent swing low then closes above it."""
        # 20 bars of normal price action with a swing low around 95
        highs = [float(105 + (i % 3)) for i in range(20)]
        lows = [float(95 + (i % 3)) for i in range(20)]
        closes = [float(100 + (i % 3)) for i in range(20)]
        # 21st bar: sweeps below the swing low of 95 (low=93) but closes above it (close=97)
        highs.append(100.0)
        lows.append(93.0)
        closes.append(97.0)

        result = ti.liquidity_sweep(highs, lows, closes)
        assert result["detected"] is True
        assert result["type"] == "bullish"
        assert result["sweep_level"] == min(lows[:20])
        assert 0.0 < result["reclaim_pct"] <= 1.0

    def test_bearish_sweep_detected(self, ti):
        """Price spikes above recent swing high then closes below it."""
        highs = [float(105 + (i % 3)) for i in range(20)]
        lows = [float(95 + (i % 3)) for i in range(20)]
        closes = [float(100 + (i % 3)) for i in range(20)]
        # 21st bar: sweeps above the swing high of 107 (high=109) but closes below it (close=104)
        highs.append(109.0)
        lows.append(102.0)
        closes.append(104.0)

        result = ti.liquidity_sweep(highs, lows, closes)
        assert result["detected"] is True
        assert result["type"] == "bearish"
        assert result["sweep_level"] == max(highs[:20])
        assert 0.0 < result["reclaim_pct"] <= 1.0

    def test_no_sweep_normal_action(self, ti):
        """Normal price action within range — no sweep."""
        highs = [float(105 + (i % 3)) for i in range(21)]
        lows = [float(95 + (i % 3)) for i in range(21)]
        closes = [float(100 + (i % 3)) for i in range(21)]

        result = ti.liquidity_sweep(highs, lows, closes)
        assert result["detected"] is False
        assert result["type"] == "none"

    def test_insufficient_data(self, ti):
        """Returns safe defaults with too few bars."""
        result = ti.liquidity_sweep([100.0] * 5, [99.0] * 5, [100.0] * 5, lookback=20)
        assert result["detected"] is False
        assert result["sweep_level"] == 0.0
        assert result["reclaim_pct"] == 0.0


class TestVolumeAnomaly:
    def test_normal_volume(self, ti):
        """Volume at average level — ratio ~1.0, not anomaly."""
        volumes = [1000.0] * 20
        result = ti.volume_anomaly(volumes)
        assert result["ratio"] == pytest.approx(1.0)
        assert result["is_anomaly"] is False
        assert result["level"] == "normal"

    def test_spike_volume(self, ti):
        """Volume at 3x average — is_anomaly=True, level='spike'."""
        volumes = [1000.0] * 19 + [3000.0]
        result = ti.volume_anomaly(volumes)
        assert result["ratio"] > 2.5
        assert result["is_anomaly"] is True
        assert result["level"] == "spike"
        assert result["z_score"] > 0

    def test_elevated_volume(self, ti):
        """Volume at ~2x average — is_anomaly=True, level='elevated'."""
        # 30 bars of 1000, then latest bar at 2200 → ratio = 2200/avg ≈ 2.06
        # avg of last 20 = (19*1000 + 2200)/20 = 1060, ratio = 2200/1060 ≈ 2.08
        volumes = [1000.0] * 29 + [2200.0]
        result = ti.volume_anomaly(volumes)
        assert result["ratio"] >= 2.0
        assert result["is_anomaly"] is True
        assert result["level"] == "elevated"

    def test_insufficient_data(self, ti):
        """Returns safe defaults with too few bars."""
        result = ti.volume_anomaly([1000.0] * 5, period=20)
        assert result["ratio"] == 1.0
        assert result["is_anomaly"] is False
        assert result["level"] == "normal"
