"""Tests for Executor — contract building and mock IBKR connection."""

from unittest.mock import MagicMock, patch

import pytest

from core.executor import Executor


class TestBuildContract:
    def test_btc_contract(self):
        executor = Executor()
        with patch.dict("sys.modules", {
            "ib_insync": MagicMock(),
        }):
            from ib_insync import Crypto
            contract = executor._build_contract("BTC")
            Crypto.assert_called_with("BTC", "PAXOS", "USD")

    def test_eth_contract(self):
        executor = Executor()
        with patch.dict("sys.modules", {
            "ib_insync": MagicMock(),
        }):
            from ib_insync import Crypto
            contract = executor._build_contract("ETH")
            Crypto.assert_called_with("ETH", "PAXOS", "USD")

    def test_gldm_contract(self):
        executor = Executor()
        with patch.dict("sys.modules", {
            "ib_insync": MagicMock(),
        }):
            from ib_insync import Stock
            contract = executor._build_contract("GLDM")
            Stock.assert_called_with("GLDM", "ARCA", "USD")

    def test_slv_contract(self):
        executor = Executor()
        with patch.dict("sys.modules", {
            "ib_insync": MagicMock(),
        }):
            from ib_insync import Stock
            contract = executor._build_contract("SLV")
            Stock.assert_called_with("SLV", "ARCA", "USD")

    def test_unknown_asset_raises(self):
        executor = Executor()
        with patch.dict("sys.modules", {
            "ib_insync": MagicMock(),
        }):
            with pytest.raises(ValueError, match="Unknown asset"):
                executor._build_contract("INVALID")


class TestExecutorInit:
    def test_default_paper_mode(self):
        executor = Executor()
        assert executor.paper_mode is True
        assert executor.port == 7497

    def test_live_mode(self):
        executor = Executor({"paper_mode": False, "port": 7496})
        assert executor.paper_mode is False
        assert executor.port == 7496

    def test_custom_config(self):
        executor = Executor({
            "paper_mode": True,
            "host": "192.168.1.1",
            "port": 9999,
            "client_id": 5,
        })
        assert executor.host == "192.168.1.1"
        assert executor.port == 9999
        assert executor.client_id == 5


class TestExecuteNoConnection:
    def test_execute_without_ibkr(self):
        """Execute should return an order_error when IBKR is not available."""
        executor = Executor()
        order = {
            "asset": "BTC",
            "direction": "long",
            "quantity": 0.001,
            "thesis_id": "test_thesis",
            "order_type": "market",
        }
        result = executor.execute(order)
        assert result["type"] == "order_error"
        assert "not connected" in result["error"].lower() or "Connection" in result.get("error", "")
        assert result["thesis_id"] == "test_thesis"
