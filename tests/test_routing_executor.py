"""Tests for RoutingExecutor — asset routing logic."""

from unittest.mock import MagicMock, patch

import pytest


@patch("core.routing_executor.CoinbaseExecutor")
@patch("core.routing_executor.AlpacaExecutor")
class TestRoutingLogic:
    def test_btc_routes_to_coinbase(self, MockAlpaca, MockCoinbase):
        from core.routing_executor import RoutingExecutor
        executor = RoutingExecutor()

        mock_coinbase = MockCoinbase.return_value
        mock_coinbase.execute.return_value = {
            "type": "order_confirmation", "asset": "BTC",
        }

        result = executor.execute({
            "asset": "BTC", "direction": "long",
            "quantity": 0.001, "thesis_id": "test",
        })

        mock_coinbase.execute.assert_called_once()
        MockAlpaca.return_value.execute.assert_not_called()

    def test_eth_routes_to_coinbase(self, MockAlpaca, MockCoinbase):
        from core.routing_executor import RoutingExecutor
        executor = RoutingExecutor()

        mock_coinbase = MockCoinbase.return_value
        mock_coinbase.execute.return_value = {
            "type": "order_confirmation", "asset": "ETH",
        }

        result = executor.execute({
            "asset": "ETH", "direction": "long",
            "quantity": 0.1, "thesis_id": "test",
        })

        mock_coinbase.execute.assert_called_once()
        MockAlpaca.return_value.execute.assert_not_called()

    def test_aapl_routes_to_alpaca(self, MockAlpaca, MockCoinbase):
        from core.routing_executor import RoutingExecutor
        executor = RoutingExecutor()

        mock_alpaca = MockAlpaca.return_value
        mock_alpaca.execute.return_value = {
            "type": "order_confirmation", "asset": "AAPL",
        }

        result = executor.execute({
            "asset": "AAPL", "direction": "long",
            "quantity": 5, "thesis_id": "test",
        })

        mock_alpaca.execute.assert_called_once()
        MockCoinbase.return_value.execute.assert_not_called()

    def test_spy_routes_to_alpaca(self, MockAlpaca, MockCoinbase):
        from core.routing_executor import RoutingExecutor
        executor = RoutingExecutor()

        mock_alpaca = MockAlpaca.return_value
        mock_alpaca.execute.return_value = {
            "type": "order_confirmation", "asset": "SPY",
        }

        result = executor.execute({
            "asset": "SPY", "direction": "long",
            "quantity": 10, "thesis_id": "test",
        })

        mock_alpaca.execute.assert_called_once()

    def test_gldm_routes_to_alpaca(self, MockAlpaca, MockCoinbase):
        from core.routing_executor import RoutingExecutor
        executor = RoutingExecutor()

        mock_alpaca = MockAlpaca.return_value
        mock_alpaca.execute.return_value = {
            "type": "order_confirmation", "asset": "GLDM",
        }

        result = executor.execute({
            "asset": "GLDM", "direction": "long",
            "quantity": 10, "thesis_id": "test",
        })

        mock_alpaca.execute.assert_called_once()

    def test_paper_mode_is_false(self, MockAlpaca, MockCoinbase):
        from core.routing_executor import RoutingExecutor
        executor = RoutingExecutor()
        assert executor.paper_mode is False

    def test_return_value_passed_through(self, MockAlpaca, MockCoinbase):
        from core.routing_executor import RoutingExecutor
        executor = RoutingExecutor()

        expected = {
            "type": "order_confirmation",
            "asset": "BTC",
            "fill_price": 65000,
            "quantity": 0.001,
            "status": "Filled",
        }
        MockCoinbase.return_value.execute.return_value = expected

        result = executor.execute({
            "asset": "BTC", "direction": "long",
            "quantity": 0.001, "thesis_id": "test",
        })

        assert result == expected
