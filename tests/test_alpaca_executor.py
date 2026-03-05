"""Tests for AlpacaExecutor — crypto sell uses qty, buy uses notional."""

from unittest.mock import MagicMock, patch

import pytest


@patch.dict("os.environ", {
    "ALPACA_API_KEY": "test-key",
    "ALPACA_SECRET_KEY": "test-secret",
    "ALPACA_BASE_URL": "https://paper-api.alpaca.markets/v2",
})
class TestCryptoOrderPayload:
    """Verify crypto sell orders use qty, buy orders use notional."""

    @patch("core.alpaca_executor.requests")
    def test_crypto_sell_uses_qty(self, mock_requests):
        """Crypto sell (close) must use qty param to avoid notional rounding mismatch."""
        from core.alpaca_executor import AlpacaExecutor

        # Mock the order response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "order-123",
            "status": "filled",
            "filled_avg_price": "72000.00",
            "filled_qty": "0.001158702",
        }
        mock_requests.post.return_value = mock_resp
        mock_requests.get.return_value = mock_resp

        executor = AlpacaExecutor()
        result = executor.execute({
            "asset": "BTC",
            "direction": "short",  # sell
            "quantity": 0.001158702,
            "order_type": "market",
            "thesis_id": "test-close",
        })

        # Verify the POST payload uses qty, NOT notional
        call_args = mock_requests.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "qty" in payload, "Crypto sell should use qty parameter"
        assert "notional" not in payload, "Crypto sell should NOT use notional"
        assert payload["qty"] == str(0.001158702)

    @patch("core.alpaca_executor.requests")
    def test_crypto_buy_uses_notional(self, mock_requests):
        """Crypto buy (open) must use notional for dollar-based position sizing."""
        from core.alpaca_executor import AlpacaExecutor

        # Mock price lookup
        mock_price_resp = MagicMock()
        mock_price_resp.status_code = 200
        mock_price_resp.json.return_value = {
            "trades": {"BTCUSD": {"p": 70000.0}},
        }

        # Mock order response
        mock_order_resp = MagicMock()
        mock_order_resp.status_code = 200
        mock_order_resp.json.return_value = {
            "id": "order-456",
            "status": "filled",
            "filled_avg_price": "70000.00",
            "filled_qty": "0.001429",
        }

        mock_requests.get.return_value = mock_price_resp
        mock_requests.post.return_value = mock_order_resp

        executor = AlpacaExecutor()
        result = executor.execute({
            "asset": "BTC",
            "direction": "long",  # buy
            "quantity": 0.001429,
            "order_type": "market",
            "thesis_id": "test-open",
        })

        call_args = mock_requests.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "notional" in payload, "Crypto buy should use notional parameter"
        assert "qty" not in payload, "Crypto buy should NOT use qty"

    @patch("core.alpaca_executor.requests")
    def test_stock_sell_uses_qty(self, mock_requests):
        """Stock orders always use qty (integer shares)."""
        from core.alpaca_executor import AlpacaExecutor

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "order-789",
            "status": "filled",
            "filled_avg_price": "180.00",
            "filled_qty": "1",
        }
        mock_requests.post.return_value = mock_resp
        mock_requests.get.return_value = mock_resp

        executor = AlpacaExecutor()
        result = executor.execute({
            "asset": "NVDA",
            "direction": "short",  # sell
            "quantity": 1.0,
            "order_type": "market",
            "thesis_id": "test-stock-sell",
        })

        call_args = mock_requests.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "qty" in payload
        assert "notional" not in payload
