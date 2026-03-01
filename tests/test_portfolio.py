"""Tests for PortfolioState — state mutations, persistence, snapshots."""

import json
import os

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
