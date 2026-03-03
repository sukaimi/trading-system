"""Tests for agents/chart_analyst.py"""

from unittest.mock import MagicMock, patch

import pytest

from agents.chart_analyst import ChartAnalyst
from core.llm_client import LLMClient


@pytest.fixture
def mock_llm():
    return LLMClient(mock_mode=True)


@pytest.fixture
def mock_market_data():
    mdf = MagicMock()
    # Return 60 days of fake OHLCV data
    bars = []
    base_price = 65000.0
    for i in range(60):
        price = base_price + (i * 100)  # Uptrend
        bars.append({
            "date": f"2026-01-{i+1:02d}",
            "open": price - 50,
            "high": price + 200,
            "low": price - 200,
            "close": price,
            "volume": 1000000 + (i * 10000),
        })
    mdf.get_ohlcv.return_value = bars
    return mdf


@pytest.fixture
def analyst(mock_llm, mock_market_data):
    return ChartAnalyst(llm_client=mock_llm, market_data=mock_market_data)


class TestAnalyze:
    def test_returns_dict(self, analyst):
        result = analyst.analyze("BTC")
        assert isinstance(result, dict)
        assert "pattern_found" in result
        assert "direction" in result
        assert "confidence" in result

    def test_handles_no_data(self, mock_llm):
        mdf = MagicMock()
        mdf.get_ohlcv.return_value = []
        analyst = ChartAnalyst(llm_client=mock_llm, market_data=mdf)
        result = analyst.analyze("BTC")
        assert result["pattern_found"] is False
        assert result["direction"] == "neutral"

    def test_handles_insufficient_data(self, mock_llm):
        mdf = MagicMock()
        mdf.get_ohlcv.return_value = [
            {"date": "2026-01-01", "open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}
            for _ in range(10)  # Only 10 bars, need 20
        ]
        analyst = ChartAnalyst(llm_client=mock_llm, market_data=mdf)
        result = analyst.analyze("BTC")
        assert result["pattern_found"] is False

    def test_handles_llm_error(self, mock_market_data):
        llm = MagicMock()
        llm.call_deepseek.return_value = {"error": "API timeout"}
        analyst = ChartAnalyst(llm_client=llm, market_data=mock_market_data)
        result = analyst.analyze("BTC")
        assert result["pattern_found"] is False

    def test_handles_exception(self, mock_llm):
        mdf = MagicMock()
        mdf.get_ohlcv.side_effect = Exception("network error")
        analyst = ChartAnalyst(llm_client=mock_llm, market_data=mdf)
        result = analyst.analyze("BTC")
        assert result["pattern_found"] is False
        assert result["direction"] == "neutral"


class TestFormatOHLCV:
    def test_limits_to_60_bars(self, analyst):
        bars = [
            {"date": f"d{i}", "open": i, "high": i+1, "low": i-1, "close": i, "volume": 100}
            for i in range(100)
        ]
        formatted = analyst._format_ohlcv(bars)
        assert len(formatted) == 60

    def test_rounds_prices(self, analyst):
        bars = [{"date": "d1", "open": 1.23456, "high": 2.34567, "low": 0.12345, "close": 1.56789, "volume": 100}]
        formatted = analyst._format_ohlcv(bars)
        assert formatted[0]["open"] == 1.23
        assert formatted[0]["close"] == 1.57


class TestNormalizeResult:
    def test_clamps_confidence(self, analyst):
        result = analyst._normalize_result({"confidence": 1.5, "pattern_found": True}, "BTC")
        assert result["confidence"] == 1.0

        result = analyst._normalize_result({"confidence": -0.5, "pattern_found": True}, "BTC")
        assert result["confidence"] == 0.0

    def test_truncates_description(self, analyst):
        result = analyst._normalize_result({"description": "x" * 500, "pattern_found": True}, "BTC")
        assert len(result["description"]) <= 300

    def test_preserves_fields(self, analyst):
        result = analyst._normalize_result({
            "pattern_found": True,
            "pattern_name": "ascending triangle",
            "direction": "long",
            "confidence": 0.75,
            "support_levels": [60000, 62000],
            "resistance_levels": [70000],
            "volume_confirms": True,
            "trend": "uptrend",
            "description": "Breakout incoming",
        }, "BTC")
        assert result["pattern_name"] == "ascending triangle"
        assert result["direction"] == "long"
        assert result["volume_confirms"] is True
        assert result["asset"] == "BTC"


class TestEmptyResult:
    def test_empty_result_structure(self):
        result = ChartAnalyst._empty_result()
        assert result["pattern_found"] is False
        assert result["direction"] == "neutral"
        assert result["confidence"] == 0.0
        assert result["trend"] == "sideways"
