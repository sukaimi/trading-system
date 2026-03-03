"""Tests for the software stop-loss monitor in core/pipeline.py."""

from unittest.mock import MagicMock, patch

import pytest

from core.llm_client import LLMClient
from core.pipeline import TradingPipeline
from core.portfolio import PortfolioState
from core.risk_manager import RiskManager


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


def _add_position(pipeline, trade_id, asset, direction, entry, stop, qty=0.01, pct=5.0):
    """Helper to add a position with stop-loss fields."""
    pipeline._portfolio.add_position({
        "trade_id": trade_id,
        "asset": asset,
        "direction": direction,
        "entry_price": entry,
        "position_size_pct": pct,
        "quantity": qty,
        "stop_loss_price": stop,
    })


# ── No-Action Cases ───────────────────────────────────────────────────


class TestNoAction:

    def test_no_positions(self, pipeline):
        result = pipeline.check_stop_losses()
        assert result == []

    def test_halted_system_skips(self, pipeline):
        pipeline._portfolio.halted = True
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)
        result = pipeline.check_stop_losses()
        assert result == []
        pipeline._executor.execute.assert_not_called()

    def test_position_without_stop_loss_skipped(self, pipeline):
        pipeline._portfolio.add_position({
            "trade_id": "t1", "asset": "BTC", "direction": "long",
            "entry_price": 65000, "position_size_pct": 5.0,
        })
        with patch("core.pipeline.MarketDataFetcher") as MockMDF:
            MockMDF.return_value.get_price.return_value = {"price": 60000}
            result = pipeline.check_stop_losses()
        assert result == []
        pipeline._executor.execute.assert_not_called()

    @patch("core.pipeline.MarketDataFetcher")
    def test_long_price_above_stop(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 66000}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)
        result = pipeline.check_stop_losses()
        assert result == []
        pipeline._executor.execute.assert_not_called()

    @patch("core.pipeline.MarketDataFetcher")
    def test_short_price_below_stop(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 60000}
        _add_position(pipeline, "t1", "BTC", "short", 65000, 68250)
        result = pipeline.check_stop_losses()
        assert result == []
        pipeline._executor.execute.assert_not_called()


# ── Breach Cases ──────────────────────────────────────────────────────


class TestBreach:

    @patch("core.pipeline.MarketDataFetcher")
    def test_long_stop_breached(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 61000}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750, qty=0.01)
        pipeline._executor.execute.return_value = {
            "type": "order_confirmation",
            "order_id": 99, "asset": "BTC",
            "fill_price": 61000, "quantity": 0.01, "status": "Filled",
        }

        result = pipeline.check_stop_losses()

        assert len(result) == 1
        assert result[0]["trade_id"] == "t1"
        assert result[0]["pnl"] < 0
        # Executor called with opposite direction
        call_args = pipeline._executor.execute.call_args[0][0]
        assert call_args["direction"] == "short"
        assert call_args["asset"] == "BTC"
        # Position removed
        assert len(pipeline._portfolio.open_positions) == 0

    @patch("core.pipeline.MarketDataFetcher")
    def test_short_stop_breached(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 69000}
        _add_position(pipeline, "t1", "BTC", "short", 65000, 68250, qty=0.01)
        pipeline._executor.execute.return_value = {
            "type": "order_confirmation",
            "fill_price": 69000, "quantity": 0.01, "status": "Filled",
        }

        result = pipeline.check_stop_losses()

        assert len(result) == 1
        assert result[0]["pnl"] < 0
        call_args = pipeline._executor.execute.call_args[0][0]
        assert call_args["direction"] == "long"

    @patch("core.pipeline.MarketDataFetcher")
    def test_exact_stop_price_triggers(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 61750}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750, qty=0.01)
        pipeline._executor.execute.return_value = {
            "type": "order_confirmation",
            "fill_price": 61750, "quantity": 0.01, "status": "Filled",
        }

        result = pipeline.check_stop_losses()
        assert len(result) == 1

    @patch("core.pipeline.MarketDataFetcher")
    def test_pnl_calculation_long(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 60000}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750, qty=0.01)
        pipeline._executor.execute.return_value = {
            "type": "order_confirmation",
            "fill_price": 60000, "quantity": 0.01, "status": "Filled",
        }

        result = pipeline.check_stop_losses()
        # PnL = (60000 - 65000) * 0.01 = -50
        assert result[0]["pnl"] == -50.0

    @patch("core.pipeline.MarketDataFetcher")
    def test_pnl_calculation_short(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 69000}
        _add_position(pipeline, "t1", "BTC", "short", 65000, 68250, qty=0.01)
        pipeline._executor.execute.return_value = {
            "type": "order_confirmation",
            "fill_price": 69000, "quantity": 0.01, "status": "Filled",
        }

        result = pipeline.check_stop_losses()
        # PnL = (65000 - 69000) * 0.01 = -40
        assert result[0]["pnl"] == -40.0

    @patch("core.pipeline.MarketDataFetcher")
    def test_multiple_positions_partial_breach(self, MockMDF, pipeline):
        def price_side_effect(symbol):
            return {"BTC": {"price": 61000}, "ETH": {"price": 3500}}.get(symbol, {"price": 0})
        MockMDF.return_value.get_price.side_effect = price_side_effect

        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750, qty=0.01)
        _add_position(pipeline, "t2", "ETH", "long", 3200, 3040, qty=0.1)  # safe

        pipeline._executor.execute.return_value = {
            "type": "order_confirmation",
            "fill_price": 61000, "quantity": 0.01, "status": "Filled",
        }

        result = pipeline.check_stop_losses()

        assert len(result) == 1
        assert result[0]["asset"] == "BTC"
        assert len(pipeline._portfolio.open_positions) == 1
        assert pipeline._portfolio.open_positions[0]["asset"] == "ETH"

    @patch("core.pipeline.MarketDataFetcher")
    def test_telegram_called_on_breach(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 61000}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750, qty=0.01)
        pipeline._executor.execute.return_value = {
            "type": "order_confirmation",
            "fill_price": 61000, "quantity": 0.01, "status": "Filled",
        }

        pipeline.check_stop_losses()

        pipeline._telegram.send_alert.assert_called_once()
        alert_msg = pipeline._telegram.send_alert.call_args[0][0]
        assert "STOP-LOSS HIT" in alert_msg

    @patch("core.pipeline.MarketDataFetcher")
    def test_portfolio_persisted_after_closure(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 61000}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750, qty=0.01)
        pipeline._executor.execute.return_value = {
            "type": "order_confirmation",
            "fill_price": 61000, "quantity": 0.01, "status": "Filled",
        }

        with patch.object(pipeline._portfolio, "persist") as mock_persist:
            pipeline.check_stop_losses()
            mock_persist.assert_called_once()


# ── Error Resilience ──────────────────────────────────────────────────


class TestErrorResilience:

    @patch("core.pipeline.MarketDataFetcher")
    def test_price_fetch_failure_skips(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.side_effect = Exception("API down")
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750, qty=0.01)

        result = pipeline.check_stop_losses()

        assert result == []
        assert len(pipeline._portfolio.open_positions) == 1
        pipeline._executor.execute.assert_not_called()

    @patch("core.pipeline.MarketDataFetcher")
    def test_zero_price_skips(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 0}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750, qty=0.01)

        result = pipeline.check_stop_losses()

        assert result == []
        assert len(pipeline._portfolio.open_positions) == 1

    @patch("core.pipeline.MarketDataFetcher")
    def test_executor_exception_keeps_position(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 61000}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750, qty=0.01)
        pipeline._executor.execute.side_effect = Exception("Connection refused")

        result = pipeline.check_stop_losses()

        assert result == []
        assert len(pipeline._portfolio.open_positions) == 1

    @patch("core.pipeline.MarketDataFetcher")
    def test_executor_order_error_keeps_position(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 61000}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750, qty=0.01)
        pipeline._executor.execute.return_value = {
            "type": "order_error", "error": "Insufficient margin",
        }

        result = pipeline.check_stop_losses()

        assert result == []
        assert len(pipeline._portfolio.open_positions) == 1

    @patch("core.pipeline.MarketDataFetcher")
    def test_telegram_failure_still_closes(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 61000}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750, qty=0.01)
        pipeline._executor.execute.return_value = {
            "type": "order_confirmation",
            "fill_price": 61000, "quantity": 0.01, "status": "Filled",
        }
        pipeline._telegram.send_alert.side_effect = Exception("Telegram down")

        result = pipeline.check_stop_losses()

        assert len(result) == 1
        assert len(pipeline._portfolio.open_positions) == 0
