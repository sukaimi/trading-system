"""Tests for PortfolioState — state mutations, persistence, snapshots."""

import json
import os
from unittest.mock import MagicMock

import pytest

from core.portfolio import PortfolioState


class TestSnapshot:
    def test_initial_snapshot(self):
        ps = PortfolioState(initial_capital=100.0)
        snap = ps.snapshot()
        assert snap["equity"] == 100.0
        assert snap["initial_capital"] == 100.0
        assert snap["open_positions"] == []
        assert snap["halted"] is False

    def test_snapshot_is_copy(self):
        ps = PortfolioState()
        snap = ps.snapshot()
        snap["equity"] = 999.0
        assert ps.equity != 999.0


class TestUpdateEquity:
    def test_equity_update(self):
        ps = PortfolioState(initial_capital=100.0)
        ps.update_equity(110.0)
        assert ps.equity == 110.0
        assert ps.peak_equity == 110.0
        assert ps.drawdown_from_peak_pct == 0.0

    def test_drawdown_calculation(self):
        ps = PortfolioState(initial_capital=100.0)
        ps.update_equity(110.0)
        ps.update_equity(100.0)
        # Drawdown = (110 - 100) / 110 * 100 ≈ 9.09%
        assert ps.drawdown_from_peak_pct == pytest.approx(9.09, abs=0.01)

    def test_peak_only_increases(self):
        ps = PortfolioState(initial_capital=100.0)
        ps.update_equity(120.0)
        ps.update_equity(110.0)
        assert ps.peak_equity == 120.0


class TestRecordTrade:
    def test_winning_trade(self):
        ps = PortfolioState(initial_capital=100.0)
        ps.record_trade(pnl=5.0)
        assert ps.total_trades == 1
        assert ps.total_wins == 1
        assert ps.total_losses == 0
        assert ps.consecutive_losses == 0
        assert ps.daily_pnl == 5.0

    def test_losing_trade(self):
        ps = PortfolioState(initial_capital=100.0)
        ps.record_trade(pnl=-3.0)
        assert ps.total_losses == 1
        assert ps.consecutive_losses == 1

    def test_consecutive_losses(self):
        ps = PortfolioState(initial_capital=100.0)
        ps.record_trade(pnl=-1.0)
        ps.record_trade(pnl=-2.0)
        ps.record_trade(pnl=-1.5)
        assert ps.consecutive_losses == 3

    def test_win_resets_consecutive_losses(self):
        ps = PortfolioState(initial_capital=100.0)
        ps.record_trade(pnl=-1.0)
        ps.record_trade(pnl=-2.0)
        ps.record_trade(pnl=5.0)
        assert ps.consecutive_losses == 0


class TestPositions:
    def test_add_position(self):
        ps = PortfolioState()
        ps.add_position({"trade_id": "t1", "asset": "BTC"})
        assert len(ps.open_positions) == 1

    def test_remove_position(self):
        ps = PortfolioState()
        ps.add_position({"trade_id": "t1", "asset": "BTC"})
        ps.add_position({"trade_id": "t2", "asset": "ETH"})
        removed = ps.remove_position("t1")
        assert removed["asset"] == "BTC"
        assert len(ps.open_positions) == 1

    def test_remove_nonexistent(self):
        ps = PortfolioState()
        result = ps.remove_position("nonexistent")
        assert result is None


class TestResetDaily:
    def test_reset_clears_daily_pnl(self):
        ps = PortfolioState(initial_capital=100.0)
        ps.record_trade(pnl=5.0)
        ps.reset_daily()
        assert ps.daily_pnl == 0.0
        assert ps.daily_pnl_pct == 0.0
        # Total trades should NOT be reset
        assert ps.total_trades == 1


class TestPersistence:
    def test_persist_and_load(self, tmp_state_file):
        ps = PortfolioState(initial_capital=100.0, state_file=tmp_state_file)
        ps.update_equity(105.0)
        ps.add_position({"trade_id": "t1", "asset": "BTC"})
        ps.record_trade(pnl=3.0)
        ps.persist()

        # Load into a new instance
        ps2 = PortfolioState(state_file=tmp_state_file)
        loaded = ps2.load()
        assert loaded is True
        assert ps2.equity == 105.0
        assert len(ps2.open_positions) == 1
        assert ps2.total_trades == 1

    def test_load_nonexistent_file(self, tmp_path):
        ps = PortfolioState(state_file=str(tmp_path / "nonexistent.json"))
        assert ps.load() is False

    def test_load_corrupt_file(self, tmp_path):
        corrupt_file = str(tmp_path / "corrupt.json")
        with open(corrupt_file, "w") as f:
            f.write("not valid json{{{")
        ps = PortfolioState(state_file=corrupt_file)
        assert ps.load() is False

    def test_persist_creates_directory(self, tmp_path):
        state_file = str(tmp_path / "subdir" / "state.json")
        ps = PortfolioState(state_file=state_file)
        ps.persist()
        assert os.path.exists(state_file)


class TestCalculateEquity:
    def test_no_positions_returns_current_equity(self):
        ps = PortfolioState(initial_capital=100.0)
        mdf = MagicMock()
        result = ps.calculate_equity(mdf)
        assert result == 100.0
        mdf.get_price.assert_not_called()

    def test_equity_reflects_unrealized_gains(self):
        ps = PortfolioState(initial_capital=100.0)
        # Simulate: bought 0.001 BTC at $50,000 ($50 cost)
        ps.add_position({
            "trade_id": "t1", "asset": "BTC", "direction": "long",
            "entry_price": 50000.0, "quantity": 0.001,
        })
        # BTC now at $60,000 → position worth $60 → gain of $10
        mdf = MagicMock()
        mdf.get_price.return_value = {"price": 60000.0}
        result = ps.calculate_equity(mdf)
        # cash = 100 - (50000 * 0.001) = 50
        # market_value = 60000 * 0.001 = 60
        # equity = 50 + 60 = 110
        assert result == pytest.approx(110.0, abs=0.01)

    def test_equity_reflects_unrealized_losses(self):
        ps = PortfolioState(initial_capital=100.0)
        ps.add_position({
            "trade_id": "t1", "asset": "ETH", "direction": "long",
            "entry_price": 2000.0, "quantity": 0.01,
        })
        # ETH dropped to $1500 → position worth $15 → loss of $5
        mdf = MagicMock()
        mdf.get_price.return_value = {"price": 1500.0}
        result = ps.calculate_equity(mdf)
        # cash = 100 - (2000 * 0.01) = 80
        # market_value = 1500 * 0.01 = 15
        # equity = 80 + 15 = 95
        assert result == pytest.approx(95.0, abs=0.01)

    def test_multiple_positions(self):
        ps = PortfolioState(initial_capital=100.0)
        ps.add_position({
            "trade_id": "t1", "asset": "BTC", "direction": "long",
            "entry_price": 50000.0, "quantity": 0.001,
        })
        ps.add_position({
            "trade_id": "t2", "asset": "ETH", "direction": "long",
            "entry_price": 2000.0, "quantity": 0.005,
        })
        mdf = MagicMock()
        mdf.get_price.side_effect = lambda sym: (
            {"price": 55000.0} if sym == "BTC" else {"price": 2200.0}
        )
        result = ps.calculate_equity(mdf)
        # cost_basis = (50000 * 0.001) + (2000 * 0.005) = 50 + 10 = 60
        # cash = 100 - 60 = 40
        # market_value = (55000 * 0.001) + (2200 * 0.005) = 55 + 11 = 66
        # equity = 40 + 66 = 106
        assert result == pytest.approx(106.0, abs=0.01)

    def test_price_fetch_failure_uses_entry_price(self):
        ps = PortfolioState(initial_capital=100.0)
        ps.add_position({
            "trade_id": "t1", "asset": "BTC", "direction": "long",
            "entry_price": 50000.0, "quantity": 0.001,
        })
        # Price fetch returns 0 → fallback to entry price
        mdf = MagicMock()
        mdf.get_price.return_value = {"price": 0.0}
        result = ps.calculate_equity(mdf)
        # Falls back to entry price → no change
        assert result == pytest.approx(100.0, abs=0.01)
