"""Tests for core/pipeline.py"""

from unittest.mock import MagicMock, patch

import pytest

from core.llm_client import LLMClient
from core.pipeline import TradingPipeline
from core.portfolio import PortfolioState
from core.risk_manager import RiskManager
from core.schemas import (
    ConfirmingSignal,
    ConfirmingSignals,
    DevilsVerdict,
    Direction,
    Sentiment,
    SignalAlert,
    SignalCategory,
    TradeThesis,
    Urgency,
    Verdict,
)
from tools.telegram_bot import TelegramNotifier


@pytest.fixture
def risk_config():
    return {
        "max_position_pct": 7.0,
        "max_daily_loss_pct": 5.0,
        "max_total_drawdown_pct": 15.0,
        "max_open_positions": 3,
        "max_correlation": 0.50,
        "stop_loss_atr_mult": 2.0,
        "base_risk_per_trade_pct": 2.0,
    }


@pytest.fixture
def pipeline(risk_config):
    portfolio = PortfolioState()
    risk_manager = RiskManager(config=risk_config)
    executor = MagicMock()
    telegram = MagicMock()
    llm = LLMClient(mock_mode=True)

    return TradingPipeline(
        portfolio=portfolio,
        risk_manager=risk_manager,
        executor=executor,
        telegram=telegram,
        llm_client=llm,
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


class TestRunNewsScan:
    def test_returns_list(self, pipeline):
        result = pipeline.run_news_scan()
        assert isinstance(result, list)

    def test_handles_error(self, pipeline):
        # Pipeline handles errors gracefully
        mock_scout = MagicMock()
        mock_scout.scan.side_effect = Exception("scan failed")
        pipeline._news_scout = mock_scout
        result = pipeline.run_news_scan()
        assert result == []


class TestProcessSingleSignal:
    def test_no_thesis_returns_no_trade(self, pipeline, btc_signal):
        # Mock analyst returns None
        pipeline._analyst = MagicMock()
        pipeline._analyst.analyze_signal.return_value = None
        result = pipeline.process_single_signal(btc_signal)
        assert result["outcome"] == "no_trade"

    def test_killed_trade(self, pipeline, btc_signal):
        # Analyst returns thesis, devil kills it
        thesis = TradeThesis(
            asset="BTC", direction=Direction.LONG, confidence=0.7,
            thesis="test", suggested_position_pct=5.0,
        )
        killed_verdict = DevilsVerdict(
            original_thesis_id="x", verdict=Verdict.KILLED,
            confidence_adjusted=0.0, fatal_flaws=["no edge"],
        )
        pipeline._analyst = MagicMock()
        pipeline._analyst.analyze_signal.return_value = thesis
        pipeline._devil = MagicMock()
        pipeline._devil.challenge.return_value = killed_verdict
        pipeline._journal = MagicMock()

        result = pipeline.process_single_signal(btc_signal)
        assert result["outcome"] == "killed"

    def test_risk_rejected(self, pipeline, btc_signal):
        thesis = TradeThesis(
            asset="BTC", direction=Direction.LONG, confidence=0.7,
            thesis="test", suggested_position_pct=5.0,
            invalidation_level="62000",
            supporting_data={"current_price": 65000},
        )
        approved_verdict = DevilsVerdict(
            original_thesis_id="x", verdict=Verdict.APPROVED,
            confidence_adjusted=0.65,
        )
        pipeline._analyst = MagicMock()
        pipeline._analyst.analyze_signal.return_value = thesis
        pipeline._devil = MagicMock()
        pipeline._devil.challenge.return_value = approved_verdict
        pipeline._journal = MagicMock()

        # Fill portfolio to max positions → risk rejects
        for i in range(3):
            pipeline._portfolio.add_position({
                "trade_id": f"t{i}", "asset": f"A{i}",
                "direction": "long", "entry_price": 100, "position_size_pct": 3,
            })

        result = pipeline.process_single_signal(btc_signal)
        assert result["outcome"] == "risk_rejected"

    def test_execution_error(self, pipeline, btc_signal):
        thesis = TradeThesis(
            asset="BTC", direction=Direction.LONG, confidence=0.7,
            thesis="test", suggested_position_pct=5.0,
            invalidation_level="50000",
            supporting_data={"current_price": 65000},
        )
        verdict = DevilsVerdict(
            original_thesis_id="x", verdict=Verdict.APPROVED,
            confidence_adjusted=0.65,
        )
        pipeline._analyst = MagicMock()
        pipeline._analyst.analyze_signal.return_value = thesis
        pipeline._devil = MagicMock()
        pipeline._devil.challenge.return_value = verdict
        pipeline._journal = MagicMock()
        pipeline._executor = MagicMock()
        pipeline._executor.execute.return_value = {"type": "order_error", "error": "IBKR not connected"}
        # Bypass R:R check — this test is about execution errors, not R:R
        pipeline._check_reward_risk_ratio = MagicMock(return_value=(True, 3.0, "passed"))

        result = pipeline.process_single_signal(btc_signal)
        assert result["outcome"] == "execution_error"

    def test_successful_execution(self, pipeline, btc_signal):
        thesis = TradeThesis(
            asset="BTC", direction=Direction.LONG, confidence=0.7,
            thesis="test", suggested_position_pct=5.0,
            invalidation_level="50000",
            supporting_data={"current_price": 65000},
        )
        verdict = DevilsVerdict(
            original_thesis_id="x", verdict=Verdict.APPROVED,
            confidence_adjusted=0.65,
        )
        pipeline._analyst = MagicMock()
        pipeline._analyst.analyze_signal.return_value = thesis
        pipeline._devil = MagicMock()
        pipeline._devil.challenge.return_value = verdict
        pipeline._journal = MagicMock()
        pipeline._executor = MagicMock()
        # Bypass R:R check — this test is about successful execution flow
        pipeline._check_reward_risk_ratio = MagicMock(return_value=(True, 3.0, "passed"))
        pipeline._executor.execute.return_value = {
            "type": "order_confirmation",
            "order_id": 123,
            "asset": "BTC",
            "direction": "long",
            "quantity": 0.01,
            "fill_price": 65100.0,
            "status": "Filled",
        }

        result = pipeline.process_single_signal(btc_signal)
        assert result["outcome"] == "executed"
        assert result["fill_price"] == 65100.0


class TestRunScheduledAnalysis:
    def test_returns_list(self, pipeline):
        result = pipeline.run_scheduled_analysis("asian_open")
        assert isinstance(result, list)

    def test_handles_error(self, pipeline):
        pipeline._analyst = MagicMock()
        pipeline._analyst.scheduled_analysis.side_effect = Exception("fail")
        result = pipeline.run_scheduled_analysis("asian_open")
        assert result == []


class TestRunCircuitBreakerCheck:
    def test_skips_when_halted(self, pipeline):
        pipeline._portfolio.halted = True
        pipeline._circuit_breaker = MagicMock()
        pipeline.run_circuit_breaker_check()
        pipeline._circuit_breaker.check.assert_not_called()

    def test_runs_when_not_halted(self, pipeline):
        pipeline._portfolio.halted = False
        pipeline._circuit_breaker = MagicMock()
        pipeline._circuit_breaker.check.return_value = None
        pipeline.run_circuit_breaker_check()
        pipeline._circuit_breaker.check.assert_called_once()


class TestCheckTakeProfits:
    def test_skips_when_halted(self, pipeline):
        pipeline._portfolio.halted = True
        result = pipeline.check_take_profits()
        assert result == []

    def test_skips_positions_without_tp(self, pipeline):
        pipeline._portfolio.add_position({
            "trade_id": "t1", "asset": "BTC", "direction": "long",
            "entry_price": 60000, "position_size_pct": 5.0, "quantity": 0.01,
            "stop_loss_price": 57000,
            # no take_profit_price
        })
        result = pipeline.check_take_profits()
        assert result == []

    @patch("core.pipeline.MarketDataFetcher")
    def test_closes_when_tp_hit_long(self, mock_mdf_cls, pipeline):
        mock_mdf = MagicMock()
        mock_mdf.get_price.return_value = {"price": 65000}
        mock_mdf_cls.return_value = mock_mdf

        pipeline._portfolio.add_position({
            "trade_id": "t1", "asset": "BTC", "direction": "long",
            "entry_price": 60000, "position_size_pct": 5.0, "quantity": 0.01,
            "stop_loss_price": 57000, "take_profit_price": 64500,
        })
        pipeline._executor = MagicMock()
        pipeline._executor.execute.return_value = {
            "type": "order_confirmation", "fill_price": 65000, "quantity": 0.01,
        }

        result = pipeline.check_take_profits()
        assert len(result) == 1
        assert result[0]["pnl"] > 0
        pipeline._executor.execute.assert_called_once()

    @patch("core.pipeline.MarketDataFetcher")
    def test_no_close_when_below_tp_long(self, mock_mdf_cls, pipeline):
        mock_mdf = MagicMock()
        mock_mdf.get_price.return_value = {"price": 62000}
        mock_mdf_cls.return_value = mock_mdf

        pipeline._portfolio.add_position({
            "trade_id": "t1", "asset": "BTC", "direction": "long",
            "entry_price": 60000, "position_size_pct": 5.0, "quantity": 0.01,
            "stop_loss_price": 57000, "take_profit_price": 64500,
        })

        result = pipeline.check_take_profits()
        assert result == []

    @patch("core.pipeline.MarketDataFetcher")
    def test_closes_when_tp_hit_short(self, mock_mdf_cls, pipeline):
        mock_mdf = MagicMock()
        mock_mdf.get_price.return_value = {"price": 2700}
        mock_mdf_cls.return_value = mock_mdf

        pipeline._portfolio.add_position({
            "trade_id": "t2", "asset": "ETH", "direction": "short",
            "entry_price": 3200, "position_size_pct": 5.0, "quantity": 1.0,
            "stop_loss_price": 3500, "take_profit_price": 2750,
        })
        pipeline._executor = MagicMock()
        pipeline._executor.execute.return_value = {
            "type": "order_confirmation", "fill_price": 2700, "quantity": 1.0,
        }

        result = pipeline.check_take_profits()
        assert len(result) == 1
        assert result[0]["pnl"] > 0


class TestBuildExecutionOrder:
    def test_order_has_required_fields(self, pipeline):
        thesis = TradeThesis(
            asset="ETH", direction=Direction.SHORT, confidence=0.8,
            thesis="test", suggested_position_pct=7.0,
            invalidation_level="3500",
            supporting_data={"current_price": 3200},
        )
        verdict = DevilsVerdict(
            original_thesis_id="x", verdict=Verdict.APPROVED,
            confidence_adjusted=0.75,
        )
        order = pipeline._build_execution_order(thesis, verdict)
        assert order["asset"] == "ETH"
        assert order["direction"] == "short"
        assert order["stop_loss"] == 3500.0
        assert order["quantity"] > 0

    def test_take_profit_calculated_from_rr_long(self, pipeline):
        thesis = TradeThesis(
            asset="BTC", direction=Direction.LONG, confidence=0.8,
            thesis="test", suggested_position_pct=5.0,
            invalidation_level="57000", risk_reward_ratio="1.5",
            supporting_data={"current_price": 60000},
        )
        verdict = DevilsVerdict(
            original_thesis_id="x", verdict=Verdict.APPROVED,
            confidence_adjusted=0.75,
        )
        order = pipeline._build_execution_order(thesis, verdict)
        # stop = 57000, risk = 3000, reward = 3000 * 1.5 = 4500
        # Regime defaults to RANGING (mocked classifier fails), TP mult = 0.8
        # TP distance = 4500 * 0.8 = 3600, TP = 60000 + 3600 = 63600
        # But TP floor enforces min R:R 2.0: min_tp = 60000 + 3000*2 = 66000
        assert order["take_profit"] == 66000.0
        assert order["stop_loss"] == 57000.0

    def test_take_profit_calculated_from_rr_short(self, pipeline):
        thesis = TradeThesis(
            asset="ETH", direction=Direction.SHORT, confidence=0.8,
            thesis="test", suggested_position_pct=5.0,
            invalidation_level="3500", risk_reward_ratio="2.0",
            supporting_data={"current_price": 3200},
        )
        verdict = DevilsVerdict(
            original_thesis_id="x", verdict=Verdict.APPROVED,
            confidence_adjusted=0.75,
        )
        order = pipeline._build_execution_order(thesis, verdict)
        # stop = 3500, risk = 300, reward = 300 * 2.0 = 600
        # Regime defaults to RANGING (mocked classifier fails), TP mult = 0.8
        # TP distance = 600 * 0.8 = 480, TP = 3200 - 480 = 2720
        # But TP floor enforces min R:R 2.0: min_tp = 3200 - 300*2 = 2600
        assert order["take_profit"] == 2600.0
