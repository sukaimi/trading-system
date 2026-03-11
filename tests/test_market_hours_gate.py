"""Tests for market hours gating — skip stock closes when market is closed."""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ---------- AlpacaExecutor.is_market_open() ----------

@patch.dict("os.environ", {
    "ALPACA_API_KEY": "test-key",
    "ALPACA_SECRET_KEY": "test-secret",
    "ALPACA_BASE_URL": "https://paper-api.alpaca.markets/v2",
})
class TestIsMarketOpen:
    """Tests for AlpacaExecutor.is_market_open() with caching."""

    @patch("core.alpaca_executor.requests")
    def test_market_open(self, mock_requests):
        from core.alpaca_executor import AlpacaExecutor
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"is_open": True, "next_close": "2026-03-11T16:00:00-04:00"}
        mock_requests.get.return_value = mock_resp

        executor = AlpacaExecutor()
        assert executor.is_market_open() is True

    @patch("core.alpaca_executor.requests")
    def test_market_closed(self, mock_requests):
        from core.alpaca_executor import AlpacaExecutor
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"is_open": False, "next_open": "2026-03-12T09:30:00-04:00"}
        mock_requests.get.return_value = mock_resp

        executor = AlpacaExecutor()
        assert executor.is_market_open() is False

    @patch("core.alpaca_executor.requests")
    def test_cache_avoids_repeat_api_calls(self, mock_requests):
        from core.alpaca_executor import AlpacaExecutor
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"is_open": True}
        mock_requests.get.return_value = mock_resp

        executor = AlpacaExecutor()
        executor.is_market_open()
        executor.is_market_open()
        executor.is_market_open()

        # Only one GET call (the init doesn't call clock, so 1 call total)
        get_calls = [c for c in mock_requests.get.call_args_list if "clock" in str(c)]
        assert len(get_calls) == 1, "Should cache and not re-call API"

    @patch("core.alpaca_executor.requests")
    def test_cache_expires_after_5_min(self, mock_requests):
        from core.alpaca_executor import AlpacaExecutor
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"is_open": False}
        mock_requests.get.return_value = mock_resp

        executor = AlpacaExecutor()
        executor.is_market_open()

        # Simulate cache expiry
        executor._clock_cache_ts = time.time() - 301

        mock_resp.json.return_value = {"is_open": True}
        result = executor.is_market_open()
        assert result is True

    @patch("core.alpaca_executor.requests")
    def test_api_failure_returns_false(self, mock_requests):
        from core.alpaca_executor import AlpacaExecutor
        mock_requests.get.side_effect = Exception("Connection error")

        executor = AlpacaExecutor()
        assert executor.is_market_open() is False


# ---------- Pipeline market hours gate ----------

class TestPipelineMarketHoursGate:
    """Tests for _close_broker_position market hours gating."""

    def _make_pipeline(self, executor, market_open=True):
        """Create a TradingPipeline with mocked dependencies."""
        from core.pipeline import TradingPipeline

        portfolio = MagicMock()
        portfolio.halted = False
        portfolio.open_positions = []
        portfolio.equity = 100000.0
        risk_manager = MagicMock()
        telegram = MagicMock()
        llm = MagicMock()

        with patch.object(TradingPipeline, '_backfill_legacy_positions'):
            pipeline = TradingPipeline(
                portfolio=portfolio,
                risk_manager=risk_manager,
                executor=executor,
                telegram=telegram,
                llm_client=llm,
            )

        return pipeline

    @patch.dict("os.environ", {
        "ALPACA_API_KEY": "test-key",
        "ALPACA_SECRET_KEY": "test-secret",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets/v2",
    })
    @patch("core.alpaca_executor.requests")
    def test_stock_close_deferred_when_market_closed(self, mock_requests):
        """Stock close should return order_error when market is closed."""
        from core.alpaca_executor import AlpacaExecutor

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"is_open": False}
        mock_requests.get.return_value = mock_resp

        executor = AlpacaExecutor()
        pipeline = self._make_pipeline(executor)

        result = pipeline._close_broker_position("AAPL", "trade-1", 10.0, "long", "sl_close")

        assert result["type"] == "order_error"
        assert "Market closed" in result["error"]
        # close_position should NOT have been called
        mock_requests.delete.assert_not_called()

    @patch.dict("os.environ", {
        "ALPACA_API_KEY": "test-key",
        "ALPACA_SECRET_KEY": "test-secret",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets/v2",
    })
    @patch("core.alpaca_executor.requests")
    def test_crypto_close_allowed_when_market_closed(self, mock_requests):
        """Crypto close should proceed even when stock market is closed (24/7)."""
        from core.alpaca_executor import AlpacaExecutor

        # Clock says closed
        mock_clock_resp = MagicMock()
        mock_clock_resp.status_code = 200
        mock_clock_resp.json.return_value = {"is_open": False}

        # Close position response
        mock_close_resp = MagicMock()
        mock_close_resp.status_code = 200
        mock_close_resp.content = b'{"id": "order-1"}'
        mock_close_resp.json.return_value = {"id": "order-1", "status": "filled", "filled_avg_price": "70000", "filled_qty": "0.001"}

        def side_effect_get(url, **kwargs):
            if "clock" in url:
                return mock_clock_resp
            if "orders" in url:
                # For _wait_for_fill
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"status": "filled", "filled_avg_price": "70000", "filled_qty": "0.001"}
                return resp
            if "positions" in url:
                return MagicMock(status_code=404)
            return mock_clock_resp

        mock_requests.get.side_effect = side_effect_get
        mock_requests.delete.return_value = mock_close_resp

        executor = AlpacaExecutor()
        pipeline = self._make_pipeline(executor)

        result = pipeline._close_broker_position("BTC", "trade-2", 0.001, "long", "sl_close")

        # Should attempt close (not deferred)
        assert result["type"] == "order_confirmation"

    @patch.dict("os.environ", {
        "ALPACA_API_KEY": "test-key",
        "ALPACA_SECRET_KEY": "test-secret",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets/v2",
    })
    @patch("core.alpaca_executor.requests")
    def test_stock_close_proceeds_when_market_open(self, mock_requests):
        """Stock close should proceed when market is open."""
        from core.alpaca_executor import AlpacaExecutor

        mock_clock_resp = MagicMock()
        mock_clock_resp.status_code = 200
        mock_clock_resp.json.return_value = {"is_open": True}

        mock_close_resp = MagicMock()
        mock_close_resp.status_code = 200
        mock_close_resp.content = b'{"id": "order-1"}'
        mock_close_resp.json.return_value = {"id": "order-1", "status": "filled", "filled_avg_price": "180", "filled_qty": "10"}

        def side_effect_get(url, **kwargs):
            if "clock" in url:
                return mock_clock_resp
            if "orders" in url:
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"status": "filled", "filled_avg_price": "180", "filled_qty": "10"}
                return resp
            if "positions" in url:
                return MagicMock(status_code=404)
            return mock_clock_resp

        mock_requests.get.side_effect = side_effect_get
        mock_requests.delete.return_value = mock_close_resp

        executor = AlpacaExecutor()
        pipeline = self._make_pipeline(executor)

        result = pipeline._close_broker_position("NVDA", "trade-3", 10.0, "long", "sl_close")

        assert result["type"] == "order_confirmation"

    @patch.dict("os.environ", {
        "ALPACA_API_KEY": "test-key",
        "ALPACA_SECRET_KEY": "test-secret",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets/v2",
    })
    @patch("core.alpaca_executor.requests")
    def test_deferred_close_logged_once_per_asset(self, mock_requests):
        """Deferred close should only log once per asset, not every heartbeat."""
        from core.alpaca_executor import AlpacaExecutor

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"is_open": False}
        mock_requests.get.return_value = mock_resp

        executor = AlpacaExecutor()
        pipeline = self._make_pipeline(executor)

        # First call — should add to deferred set
        pipeline._close_broker_position("AAPL", "trade-1", 10.0, "long", "sl_close")
        assert "AAPL" in pipeline._deferred_closes

        # Second call — AAPL already deferred
        pipeline._close_broker_position("AAPL", "trade-1", 10.0, "long", "sl_close")
        assert "AAPL" in pipeline._deferred_closes

    @patch.dict("os.environ", {
        "ALPACA_API_KEY": "test-key",
        "ALPACA_SECRET_KEY": "test-secret",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets/v2",
    })
    @patch("core.alpaca_executor.requests")
    def test_deferred_flag_cleared_when_market_opens(self, mock_requests):
        """Deferred flag should be cleared when market opens and close succeeds."""
        from core.alpaca_executor import AlpacaExecutor

        executor = AlpacaExecutor()
        pipeline = self._make_pipeline(executor)

        # Simulate deferred state
        pipeline._deferred_closes.add("AAPL")

        # Market opens
        mock_clock_resp = MagicMock()
        mock_clock_resp.status_code = 200
        mock_clock_resp.json.return_value = {"is_open": True}

        mock_close_resp = MagicMock()
        mock_close_resp.status_code = 200
        mock_close_resp.content = b'{"id": "order-1"}'
        mock_close_resp.json.return_value = {"id": "order-1", "status": "filled", "filled_avg_price": "180", "filled_qty": "10"}

        def side_effect_get(url, **kwargs):
            if "clock" in url:
                return mock_clock_resp
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"status": "filled", "filled_avg_price": "180", "filled_qty": "10"}
            return resp

        mock_requests.get.side_effect = side_effect_get
        mock_requests.delete.return_value = mock_close_resp

        pipeline._close_broker_position("AAPL", "trade-1", 10.0, "long", "sl_close")

        assert "AAPL" not in pipeline._deferred_closes


class TestIsCrypto:
    """Test the _is_crypto static method."""

    def test_btc_is_crypto(self):
        from core.pipeline import TradingPipeline
        assert TradingPipeline._is_crypto("BTC") is True

    def test_eth_is_crypto(self):
        from core.pipeline import TradingPipeline
        assert TradingPipeline._is_crypto("ETH") is True

    def test_stock_is_not_crypto(self):
        from core.pipeline import TradingPipeline
        assert TradingPipeline._is_crypto("AAPL") is False
        assert TradingPipeline._is_crypto("SPY") is False
        assert TradingPipeline._is_crypto("GLDM") is False
