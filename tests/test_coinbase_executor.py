"""Tests for CoinbaseExecutor — mock SDK client for unit testing."""

from unittest.mock import MagicMock, patch

import pytest


class TestCoinbaseInit:
    @patch("core.coinbase_executor.CoinbaseExecutor._init_client")
    def test_paper_mode_is_false(self, mock_init):
        from core.coinbase_executor import CoinbaseExecutor
        executor = CoinbaseExecutor()
        assert executor.paper_mode is False

    @patch("core.coinbase_executor.CoinbaseExecutor._init_client")
    def test_no_credentials_warns(self, mock_init):
        from core.coinbase_executor import CoinbaseExecutor
        executor = CoinbaseExecutor()
        assert executor._client is None


class TestCoinbaseExecute:
    @patch("core.coinbase_executor.CoinbaseExecutor._init_client")
    def test_no_client_returns_error(self, mock_init):
        from core.coinbase_executor import CoinbaseExecutor
        executor = CoinbaseExecutor()
        executor._client = None
        result = executor.execute({
            "asset": "BTC", "direction": "long",
            "quantity": 0.001, "thesis_id": "test",
        })
        assert result["type"] == "order_error"
        assert "not initialized" in result["error"]

    @patch("core.coinbase_executor.CoinbaseExecutor._init_client")
    def test_buy_order_success(self, mock_init):
        from core.coinbase_executor import CoinbaseExecutor
        executor = CoinbaseExecutor()
        mock_client = MagicMock()
        executor._client = mock_client

        mock_client.get_product.return_value = {"price": "65000.00"}
        mock_client.market_order_buy.return_value = {
            "success": True, "order_id": "cb-order-123",
        }
        mock_client.get_order.return_value = {
            "order": {
                "status": "FILLED",
                "average_filled_price": "65100.00",
                "filled_size": "0.001",
            }
        }

        result = executor.execute({
            "asset": "BTC", "direction": "long",
            "quantity": 0.001, "thesis_id": "test_thesis",
        })

        assert result["type"] == "order_confirmation"
        assert result["asset"] == "BTC"
        assert result["direction"] == "long"
        assert result["fill_price"] == 65100.0
        assert result["quantity"] == 0.001
        assert result["status"] == "Filled"
        mock_client.market_order_buy.assert_called_once()

    @patch("core.coinbase_executor.CoinbaseExecutor._init_client")
    def test_sell_order_uses_base_size(self, mock_init):
        from core.coinbase_executor import CoinbaseExecutor
        executor = CoinbaseExecutor()
        mock_client = MagicMock()
        executor._client = mock_client

        mock_client.get_product.return_value = {"price": "65000.00"}
        mock_client.market_order_sell.return_value = {
            "success": True, "order_id": "cb-sell-123",
        }
        mock_client.get_order.return_value = {
            "order": {
                "status": "FILLED",
                "average_filled_price": "64900.00",
                "filled_size": "0.001",
            }
        }

        result = executor.execute({
            "asset": "ETH", "direction": "short",
            "quantity": 0.5, "thesis_id": "test",
        })

        assert result["type"] == "order_confirmation"
        mock_client.market_order_sell.assert_called_once()
        # Verify base_size was passed (not quote_size)
        call_kwargs = mock_client.market_order_sell.call_args
        assert "base_size" in call_kwargs.kwargs or "base_size" in (call_kwargs[1] if len(call_kwargs) > 1 else {})

    @patch("core.coinbase_executor.CoinbaseExecutor._init_client")
    def test_order_rejected(self, mock_init):
        from core.coinbase_executor import CoinbaseExecutor
        executor = CoinbaseExecutor()
        mock_client = MagicMock()
        executor._client = mock_client

        mock_client.get_product.return_value = {"price": "65000.00"}
        mock_client.market_order_buy.return_value = {
            "success": False,
            "failure_reason": "INSUFFICIENT_FUND",
            "error_response": {"message": "Not enough balance"},
        }

        result = executor.execute({
            "asset": "BTC", "direction": "long",
            "quantity": 0.001, "thesis_id": "test",
        })

        assert result["type"] == "order_error"
        assert "Not enough balance" in result["error"]

    @patch("core.coinbase_executor.CoinbaseExecutor._init_client")
    def test_price_fetch_failure(self, mock_init):
        from core.coinbase_executor import CoinbaseExecutor
        executor = CoinbaseExecutor()
        mock_client = MagicMock()
        executor._client = mock_client

        mock_client.get_product.side_effect = Exception("API error")

        with patch("tools.market_data.MarketDataFetcher") as MockMDF:
            MockMDF.return_value.get_price.return_value = {"price": 0}
            result = executor.execute({
                "asset": "BTC", "direction": "long",
                "quantity": 0.001, "thesis_id": "test",
            })

        assert result["type"] == "order_error"
        assert "Cannot get price" in result["error"]

    @patch("core.coinbase_executor.CoinbaseExecutor._init_client")
    def test_fill_timeout_returns_error(self, mock_init):
        from core.coinbase_executor import CoinbaseExecutor
        executor = CoinbaseExecutor()
        mock_client = MagicMock()
        executor._client = mock_client

        mock_client.get_product.return_value = {"price": "65000.00"}
        mock_client.market_order_buy.return_value = {
            "success": True, "order_id": "cb-timeout",
        }
        # Order never fills
        mock_client.get_order.return_value = {
            "order": {"status": "PENDING"}
        }

        result = executor.execute({
            "asset": "BTC", "direction": "long",
            "quantity": 0.001, "thesis_id": "test",
        })

        assert result["type"] == "order_error"
        assert "timeout" in result["error"]


class TestCoinbaseGetPrice:
    @patch("core.coinbase_executor.CoinbaseExecutor._init_client")
    def test_gets_price_from_coinbase(self, mock_init):
        from core.coinbase_executor import CoinbaseExecutor
        executor = CoinbaseExecutor()
        mock_client = MagicMock()
        executor._client = mock_client

        mock_client.get_product.return_value = {"price": "65432.10"}
        price = executor._get_price("BTC")
        assert price == 65432.10

    @patch("core.coinbase_executor.CoinbaseExecutor._init_client")
    def test_falls_back_to_coingecko(self, mock_init):
        from core.coinbase_executor import CoinbaseExecutor
        executor = CoinbaseExecutor()
        mock_client = MagicMock()
        executor._client = mock_client

        mock_client.get_product.side_effect = Exception("fail")

        with patch("tools.market_data.MarketDataFetcher") as MockMDF:
            MockMDF.return_value.get_price.return_value = {"price": 65000}
            price = executor._get_price("BTC")

        assert price == 65000
