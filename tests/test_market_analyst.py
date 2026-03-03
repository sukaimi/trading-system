"""Tests for agents/market_analyst.py"""

from unittest.mock import MagicMock, patch

import pytest

from agents.market_analyst import MarketAnalyst
from core.llm_client import LLMClient
from core.schemas import (
    ConfirmingSignal,
    ConfirmingSignals,
    Direction,
    Sentiment,
    SignalAlert,
    SignalCategory,
    TimeHorizon,
    TradeThesis,
    Urgency,
)


@pytest.fixture
def mock_llm():
    return LLMClient(mock_mode=True)


@pytest.fixture
def mock_market_data():
    mdf = MagicMock()
    mdf.get_ohlcv.return_value = [
        {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000}
        for _ in range(60)
    ]
    mdf.get_market_context.return_value = {"vix": 20, "dxy": 104}
    return mdf


@pytest.fixture
def mock_indicators():
    ti = MagicMock()
    ti.rsi.return_value = 55.0
    ti.macd.return_value = {"macd_line": 0.5, "signal_line": 0.3, "histogram": 0.2}
    ti.bollinger_bands.return_value = {"upper": 110, "middle": 100, "lower": 90, "bandwidth": 0.2}
    ti.atr.return_value = 3.5
    ti.sma.return_value = 100.0
    return ti


@pytest.fixture
def mock_correlation():
    ca = MagicMock()
    ca.detect_regime.return_value = "risk_on"
    return ca


@pytest.fixture
def analyst(mock_llm, mock_market_data, mock_indicators, mock_correlation):
    return MarketAnalyst(
        llm_client=mock_llm,
        market_data=mock_market_data,
        indicators=mock_indicators,
        correlation=mock_correlation,
    )


@pytest.fixture
def btc_signal():
    return SignalAlert(
        asset="BTC",
        signal_strength=0.8,
        headline="BTC breaks 100k",
        sentiment=Sentiment.BULLISH,
        category=SignalCategory.CRYPTO_SPECIFIC,
        new_information="New ATH",
        urgency=Urgency.HIGH,
        confidence_in_classification=0.8,
    )


class TestAnalyzeSignal:
    def test_returns_none_for_macro(self, analyst):
        signal = SignalAlert(
            asset="MACRO", signal_strength=0.5, headline="Macro",
            sentiment=Sentiment.NEUTRAL, category=SignalCategory.MACRO,
            new_information="", urgency=Urgency.LOW, confidence_in_classification=0.5,
        )
        assert analyst.analyze_signal(signal) is None

    def test_mock_mode_returns_none_or_thesis(self, analyst, btc_signal):
        # Mock LLM returns generic dict — no valid thesis fields → returns None
        result = analyst.analyze_signal(btc_signal)
        # In mock mode, the LLM returns a mock response without thesis fields
        # So analyze_signal should return None (no_trade or build fails)
        assert result is None or isinstance(result, TradeThesis)


class TestScheduledAnalysis:
    def test_returns_list(self, analyst):
        result = analyst.scheduled_analysis("asian_open")
        assert isinstance(result, list)


class TestShouldEscalate:
    def test_high_confidence_large_position(self, analyst):
        thesis = TradeThesis(
            asset="BTC", direction=Direction.LONG, confidence=0.75,
            thesis="test", suggested_position_pct=5.0,
        )
        assert analyst.should_escalate(thesis) is True

    def test_low_confidence_small_position(self, analyst):
        thesis = TradeThesis(
            asset="GLDM", direction=Direction.LONG, confidence=0.5,
            thesis="test", suggested_position_pct=2.0,
        )
        assert analyst.should_escalate(thesis) is False

    def test_crypto_swing(self, analyst):
        thesis = TradeThesis(
            asset="BTC", direction=Direction.LONG, confidence=0.5,
            thesis="test", suggested_position_pct=2.0, time_horizon=TimeHorizon.SWING,
        )
        assert analyst.should_escalate(thesis) is True

    def test_many_risks(self, analyst):
        thesis = TradeThesis(
            asset="SLV", direction=Direction.SHORT, confidence=0.5,
            thesis="test", suggested_position_pct=2.0,
            what_could_go_wrong=["r1", "r2", "r3"],
        )
        assert analyst.should_escalate(thesis) is True
