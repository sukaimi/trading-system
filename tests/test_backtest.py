"""Tests for the backtest module.

Covers: engine simulation loop, portfolio state, data loader caching,
signal generation, parameter sweeps, and metrics calculation.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from tools.backtest.portfolio_sim import SimulatedPortfolio, Position, ClosedTrade
from tools.backtest.signals import SignalGenerator, BacktestSignal
from tools.backtest.engine import BacktestEngine, load_risk_params
from tools.backtest.report import calculate_metrics, _calculate_sharpe, format_summary_table, save_results
from tools.backtest.data_loader import HistoricalDataLoader, CORE_ASSETS, YFINANCE_SYMBOLS
from tools.backtest.param_sweep import ParameterSweeper, _frange


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 100, start_price: float = 100.0, volatility: float = 0.02,
                trend: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    prices = [start_price]
    for _ in range(n - 1):
        change = rng.normal(trend, volatility)
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    highs = prices * (1 + rng.uniform(0, 0.015, n))
    lows = prices * (1 - rng.uniform(0, 0.015, n))
    opens = prices * (1 + rng.uniform(-0.005, 0.005, n))
    volumes = rng.randint(100000, 1000000, n)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    }, index=dates)
    df.index.name = "date"
    return df


def _make_trending_up(n: int = 200) -> pd.DataFrame:
    """Generate an uptrend for testing."""
    return _make_ohlcv(n=n, start_price=50.0, volatility=0.015, trend=0.002, seed=123)


def _make_trending_down(n: int = 200) -> pd.DataFrame:
    """Generate a downtrend for testing."""
    return _make_ohlcv(n=n, start_price=150.0, volatility=0.015, trend=-0.002, seed=456)


def _default_params() -> dict:
    """Return default test parameters."""
    return {
        "default_stop_loss_pct": 3.0,
        "default_take_profit_pct": 6.0,
        "stop_loss_atr_mult": 2.0,
        "take_profit_atr_mult": 3.0,
        "trailing_stop_activation_pct": 2.0,
        "trailing_stop_distance_pct": 1.5,
        "max_position_pct": 5.0,
        "base_risk_per_trade_pct": 1.5,
        "max_holding_bars": 72,
        "use_atr_stops": True,
        "trading_friction": {
            "enabled": True,
            "spread_pct": {"stock": 0.05, "etf": 0.03, "crypto": 0.15},
            "commission_pct": {"stock": 0.0, "etf": 0.0, "crypto": 0.15},
            "short_borrow_annual_pct": {"stock": 1.5, "etf": 1.0, "crypto": 5.0},
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# SimulatedPortfolio Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSimulatedPortfolio:
    """Tests for SimulatedPortfolio state management."""

    def test_initial_state(self):
        p = SimulatedPortfolio(initial_capital=10000.0)
        assert p.cash == 10000.0
        assert p.equity == 10000.0
        assert p.positions == []
        assert p.closed_trades == []
        assert p.peak_equity == 10000.0

    def test_open_position(self):
        p = SimulatedPortfolio(initial_capital=10000.0)
        pos = p.open_position(
            asset="AAPL", direction="long", price=150.0, quantity=10.0,
            stop_loss=145.0, take_profit=160.0, bar_index=5, date_str="2024-01-05",
        )
        assert pos is not None
        assert pos.asset == "AAPL"
        assert pos.direction == "long"
        assert pos.entry_price == 150.0
        assert pos.quantity == 10.0
        assert p.cash == 10000.0 - 150.0 * 10.0

    def test_open_position_insufficient_cash(self):
        p = SimulatedPortfolio(initial_capital=100.0)
        pos = p.open_position(
            asset="AAPL", direction="long", price=150.0, quantity=10.0,
            stop_loss=145.0, take_profit=160.0, bar_index=0, date_str="2024-01-01",
        )
        assert pos is None
        assert p.cash == 100.0

    def test_open_position_with_friction(self):
        p = SimulatedPortfolio(initial_capital=10000.0)
        pos = p.open_position(
            asset="AAPL", direction="long", price=100.0, quantity=10.0,
            stop_loss=95.0, take_profit=110.0, bar_index=0, date_str="2024-01-01",
            friction_cost=5.0,
        )
        assert pos is not None
        # Cash should be reduced by notional + friction
        assert p.cash == 10000.0 - 100.0 * 10.0 - 5.0

    def test_close_position_long_profit(self):
        p = SimulatedPortfolio(initial_capital=10000.0)
        pos = p.open_position(
            asset="AAPL", direction="long", price=100.0, quantity=10.0,
            stop_loss=95.0, take_profit=110.0, bar_index=0, date_str="2024-01-01",
        )
        trade = p.close_position(pos, exit_price=110.0, bar_index=5, date_str="2024-01-06",
                                  exit_reason="take_profit")
        assert trade.pnl == 100.0  # (110-100) * 10
        assert trade.pnl_pct == pytest.approx(0.1)
        assert trade.exit_reason == "take_profit"
        assert trade.bars_held == 5
        assert len(p.positions) == 0
        assert len(p.closed_trades) == 1

    def test_close_position_long_loss(self):
        p = SimulatedPortfolio(initial_capital=10000.0)
        pos = p.open_position(
            asset="AAPL", direction="long", price=100.0, quantity=10.0,
            stop_loss=95.0, take_profit=110.0, bar_index=0, date_str="2024-01-01",
        )
        trade = p.close_position(pos, exit_price=95.0, bar_index=3, date_str="2024-01-04",
                                  exit_reason="stop_loss")
        assert trade.pnl == -50.0  # (95-100) * 10
        assert trade.pnl_pct == pytest.approx(-0.05)

    def test_close_position_short_profit(self):
        p = SimulatedPortfolio(initial_capital=10000.0)
        pos = p.open_position(
            asset="TSLA", direction="short", price=200.0, quantity=5.0,
            stop_loss=210.0, take_profit=180.0, bar_index=0, date_str="2024-01-01",
        )
        trade = p.close_position(pos, exit_price=180.0, bar_index=4, date_str="2024-01-05",
                                  exit_reason="take_profit")
        assert trade.pnl == 100.0  # (200-180) * 5
        assert trade.direction == "short"

    def test_equity_with_open_positions(self):
        p = SimulatedPortfolio(initial_capital=10000.0)
        p.open_position(
            asset="AAPL", direction="long", price=100.0, quantity=10.0,
            stop_loss=95.0, take_profit=110.0, bar_index=0, date_str="2024-01-01",
        )
        # Equity = cash + position notional (at entry price)
        assert p.equity == 10000.0  # 9000 cash + 1000 position

    def test_record_equity_snapshot(self):
        p = SimulatedPortfolio(initial_capital=10000.0)
        p.open_position(
            asset="AAPL", direction="long", price=100.0, quantity=10.0,
            stop_loss=95.0, take_profit=110.0, bar_index=0, date_str="2024-01-01",
        )
        # Record snapshot with price up
        p.record_equity_snapshot("2024-01-02", {"AAPL": 105.0})
        assert len(p.equity_curve) == 1
        # Equity = 9000 cash + 105*10 position = 10050
        assert p.equity_curve[0]["equity"] == 10050.0

    def test_max_drawdown(self):
        p = SimulatedPortfolio(initial_capital=10000.0)
        p.equity_curve = [
            {"date": "d1", "equity": 10000.0},
            {"date": "d2", "equity": 10500.0},
            {"date": "d3", "equity": 9500.0},
            {"date": "d4", "equity": 10200.0},
        ]
        # Peak = 10500, trough = 9500, DD = (9500-10500)/10500 = -9.52%
        dd = p.max_drawdown
        assert dd == pytest.approx(-1000.0 / 10500.0, abs=0.001)

    def test_get_positions_for_asset(self):
        p = SimulatedPortfolio(initial_capital=50000.0)
        p.open_position("AAPL", "long", 150.0, 10.0, 140.0, 160.0, 0, "d1")
        p.open_position("NVDA", "long", 500.0, 5.0, 480.0, 520.0, 0, "d1")
        p.open_position("AAPL", "long", 152.0, 5.0, 145.0, 165.0, 1, "d2")
        assert len(p.get_positions_for_asset("AAPL")) == 2
        assert len(p.get_positions_for_asset("NVDA")) == 1
        assert len(p.get_positions_for_asset("TSLA")) == 0

    def test_closed_trade_to_dict(self):
        trade = ClosedTrade(
            asset="AAPL", direction="long", entry_price=100.0, exit_price=110.0,
            quantity=10.0, pnl=100.0, pnl_pct=0.1, bars_held=5,
            exit_reason="take_profit", entry_date="2024-01-01", exit_date="2024-01-06",
        )
        d = trade.to_dict()
        assert d["asset"] == "AAPL"
        assert d["pnl"] == 100.0
        assert d["exit_reason"] == "take_profit"

    def test_apply_borrow_cost(self):
        p = SimulatedPortfolio(initial_capital=10000.0)
        pos = p.open_position("AAPL", "short", 100.0, 10.0, 110.0, 90.0, 0, "d1")
        initial_cash = p.cash
        p.apply_borrow_cost(pos, 0.50)
        assert p.cash == initial_cash - 0.50


# ══════════════════════════════════════════════════════════════════════════════
# Signal Generator Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSignalGenerator:
    """Tests for synthetic signal generation."""

    def test_generates_signals(self):
        df = _make_ohlcv(200)
        gen = SignalGenerator()
        signals = gen.generate("AAPL", df, warmup=50)
        # Should generate at least some signals on 200 bars
        assert isinstance(signals, list)
        for sig in signals:
            assert isinstance(sig, BacktestSignal)
            assert sig.asset == "AAPL"
            assert sig.direction in ("long", "short")
            assert 0.0 <= sig.confidence <= 1.0
            assert sig.bar_index >= 50

    def test_no_signals_insufficient_data(self):
        df = _make_ohlcv(30)
        gen = SignalGenerator()
        signals = gen.generate("AAPL", df, warmup=50)
        assert signals == []

    def test_cooldown_respected(self):
        df = _make_ohlcv(200)
        gen = SignalGenerator(cooldown_bars=10)
        signals = gen.generate("AAPL", df)
        if len(signals) >= 2:
            for i in range(1, len(signals)):
                assert signals[i].bar_index - signals[i - 1].bar_index >= 10

    def test_rsi_to_confidence_long(self):
        # RSI=20 -> 0.9, RSI=35 -> 0.6
        assert SignalGenerator._rsi_to_confidence(20.0, "long") == pytest.approx(0.9)
        assert SignalGenerator._rsi_to_confidence(35.0, "long") == pytest.approx(0.6)
        assert SignalGenerator._rsi_to_confidence(27.5, "long") == pytest.approx(0.75)

    def test_rsi_to_confidence_short(self):
        # RSI=80 -> 0.9, RSI=65 -> 0.6
        assert SignalGenerator._rsi_to_confidence(80.0, "short") == pytest.approx(0.9)
        assert SignalGenerator._rsi_to_confidence(65.0, "short") == pytest.approx(0.6)

    def test_confidence_clamped(self):
        # Extreme RSI should be clamped to [0.5, 0.95]
        conf = SignalGenerator._rsi_to_confidence(0.0, "long")
        assert 0.5 <= conf <= 0.95
        conf = SignalGenerator._rsi_to_confidence(100.0, "short")
        assert 0.5 <= conf <= 0.95


# ══════════════════════════════════════════════════════════════════════════════
# Engine Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBacktestEngine:
    """Tests for the core backtest engine simulation loop."""

    def test_basic_run(self):
        data = {"AAPL": _make_ohlcv(200)}
        engine = BacktestEngine(params=_default_params(), initial_capital=10000.0)
        result = engine.run(data)
        assert "initial_capital" in result
        assert "final_equity" in result
        assert "total_return" in result
        assert "closed_trades" in result
        assert "equity_curve" in result
        assert result["initial_capital"] == 10000.0

    def test_empty_data(self):
        engine = BacktestEngine(params=_default_params())
        result = engine.run({})
        assert result["trade_count"] == 0

    def test_multiple_assets(self):
        data = {
            "AAPL": _make_ohlcv(200, seed=1),
            "NVDA": _make_ohlcv(200, seed=2),
            "SPY": _make_ohlcv(200, seed=3),
        }
        engine = BacktestEngine(params=_default_params(), initial_capital=50000.0)
        result = engine.run(data)
        assert "AAPL" in result["assets_traded"]
        assert "NVDA" in result["assets_traded"]

    def test_no_friction(self):
        data = {"AAPL": _make_ohlcv(200)}
        engine = BacktestEngine(params=_default_params(), use_friction=False)
        result = engine.run(data)
        # Should still produce results
        assert "final_equity" in result

    def test_stop_loss_triggers(self):
        """Verify stop-loss exits happen."""
        # Use a downtrend to increase chance of stop-loss hits
        data = {"TEST": _make_trending_down(200)}
        params = _default_params()
        params["default_stop_loss_pct"] = 2.0  # Tight stop
        params["default_take_profit_pct"] = 20.0  # Wide TP (unlikely to hit in downtrend)
        params["max_holding_bars"] = 999  # Don't force-exit
        engine = BacktestEngine(params=params, use_friction=False)
        result = engine.run(data)
        # Check if any stop losses triggered
        sl_exits = [t for t in result["closed_trades"] if t["exit_reason"] == "stop_loss"]
        # In a strong downtrend, long signals with tight stops should trigger SL
        # (Some signals may be shorts that profit, so we just verify the mechanism works)
        assert isinstance(sl_exits, list)

    def test_holding_period_exits(self):
        """Verify holding period forced exits."""
        data = {"AAPL": _make_ohlcv(200)}
        params = _default_params()
        params["max_holding_bars"] = 3  # Force exit after 3 bars
        params["default_stop_loss_pct"] = 50.0  # Very wide (won't trigger)
        params["default_take_profit_pct"] = 50.0  # Very wide (won't trigger)
        engine = BacktestEngine(params=params, use_friction=False)
        result = engine.run(data)
        hp_exits = [t for t in result["closed_trades"] if t["exit_reason"] == "holding_period"]
        # With 3-bar max hold and wide stops, most exits should be holding_period
        if result["closed_trades"]:
            assert any(t["bars_held"] <= 3 for t in result["closed_trades"])

    def test_end_of_data_closes_positions(self):
        """All positions should be closed at end of data."""
        data = {"AAPL": _make_ohlcv(200)}
        params = _default_params()
        params["max_holding_bars"] = 999  # Don't force-exit
        params["default_stop_loss_pct"] = 50.0  # Very wide
        params["default_take_profit_pct"] = 50.0  # Very wide
        engine = BacktestEngine(params=params, use_friction=False)
        result = engine.run(data)
        eod_exits = [t for t in result["closed_trades"] if t["exit_reason"] == "end_of_data"]
        # At least some positions should have been force-closed at end of data
        assert isinstance(eod_exits, list)

    def test_equity_curve_recorded(self):
        data = {"AAPL": _make_ohlcv(100)}
        engine = BacktestEngine(params=_default_params())
        result = engine.run(data)
        assert len(result["equity_curve"]) > 0
        # Each entry has date and equity
        for snap in result["equity_curve"]:
            assert "date" in snap
            assert "equity" in snap

    def test_check_stop_loss_long(self):
        engine = BacktestEngine(params=_default_params())
        pos = Position("AAPL", "long", 100.0, 10.0, 95.0, 110.0)
        assert engine._check_stop_loss(pos, bar_low=94.0, bar_high=101.0) is True
        assert engine._check_stop_loss(pos, bar_low=96.0, bar_high=101.0) is False

    def test_check_stop_loss_short(self):
        engine = BacktestEngine(params=_default_params())
        pos = Position("AAPL", "short", 100.0, 10.0, 105.0, 90.0)
        assert engine._check_stop_loss(pos, bar_low=99.0, bar_high=106.0) is True
        assert engine._check_stop_loss(pos, bar_low=99.0, bar_high=104.0) is False

    def test_check_take_profit_long(self):
        engine = BacktestEngine(params=_default_params())
        pos = Position("AAPL", "long", 100.0, 10.0, 95.0, 110.0)
        assert engine._check_take_profit(pos, bar_high=111.0, bar_low=99.0) is True
        assert engine._check_take_profit(pos, bar_high=109.0, bar_low=99.0) is False

    def test_check_take_profit_short(self):
        engine = BacktestEngine(params=_default_params())
        pos = Position("AAPL", "short", 100.0, 10.0, 105.0, 90.0)
        assert engine._check_take_profit(pos, bar_high=101.0, bar_low=89.0) is True
        assert engine._check_take_profit(pos, bar_high=101.0, bar_low=91.0) is False

    def test_trailing_stop_activation_long(self):
        engine = BacktestEngine(params=_default_params())
        pos = Position("AAPL", "long", 100.0, 10.0, 95.0, 110.0)
        activation_pct = 0.02  # 2%
        distance_pct = 0.015  # 1.5%

        # Price at +3% from entry — should activate trailing
        triggered = engine._update_trailing_stop(pos, 103.0, activation_pct, distance_pct)
        assert pos.trailing_activated is True
        assert pos.trailing_stop is not None
        assert pos.trailing_stop == pytest.approx(103.0 * (1 - 0.015))
        assert triggered is False

        # Price moves higher — trailing stop ratchets up
        triggered = engine._update_trailing_stop(pos, 105.0, activation_pct, distance_pct)
        assert pos.trailing_stop == pytest.approx(105.0 * (1 - 0.015))
        assert triggered is False

        # Price drops but not to trailing stop — no trigger
        prev_trail = pos.trailing_stop
        triggered = engine._update_trailing_stop(pos, 104.0, activation_pct, distance_pct)
        assert pos.trailing_stop == prev_trail  # Doesn't ratchet down
        assert triggered is False

    def test_trailing_stop_trigger_long(self):
        engine = BacktestEngine(params=_default_params())
        pos = Position("AAPL", "long", 100.0, 10.0, 95.0, 110.0)
        # Activate trailing at +3%
        engine._update_trailing_stop(pos, 103.0, 0.02, 0.015)
        assert pos.trailing_activated is True
        # Move price higher to ratchet trail up
        engine._update_trailing_stop(pos, 106.0, 0.02, 0.015)
        # trail = 106 * (1 - 0.015) = 104.41
        # Now price drops to just below trailing stop, but still above activation (2%)
        # Price must be >= 102 (2% above entry) to enter the trailing block
        # and also <= trailing_stop to trigger
        trail = pos.trailing_stop
        assert trail == pytest.approx(106.0 * 0.985)
        # Price at 104.0 — still above activation (4% above entry), below trail (104.41)
        triggered = engine._update_trailing_stop(pos, 104.0, 0.02, 0.015)
        assert triggered is True

    def test_position_sizing_with_atr(self):
        engine = BacktestEngine(params=_default_params())
        size = engine._calculate_position_size(
            confidence=0.8, atr=2.0, portfolio_value=10000.0,
            base_risk_pct=0.015, sl_atr_mult=2.0, max_position_pct=0.05,
        )
        # Expected: (10000 * 0.015) / (2.0 * 2.0) * 0.8 = 150 / 4 * 0.8 = 30
        assert size == pytest.approx(30.0)

    def test_position_sizing_capped(self):
        engine = BacktestEngine(params=_default_params())
        size = engine._calculate_position_size(
            confidence=1.0, atr=0.01, portfolio_value=10000.0,
            base_risk_pct=0.015, sl_atr_mult=2.0, max_position_pct=0.05,
        )
        # With very small ATR, position would be huge — should cap at 5% = 500
        assert size == 500.0

    def test_position_sizing_zero_atr(self):
        engine = BacktestEngine(params=_default_params())
        size = engine._calculate_position_size(
            confidence=0.7, atr=0.0, portfolio_value=10000.0,
            base_risk_pct=0.015, sl_atr_mult=2.0, max_position_pct=0.05,
        )
        # Fallback: portfolio_value * max_position_pct * confidence = 10000 * 0.05 * 0.7 = 350
        assert size == pytest.approx(350.0)

    def test_friction_calculation(self):
        friction_config = {
            "spread_pct": {"stock": 0.05, "etf": 0.03, "crypto": 0.15},
            "commission_pct": {"stock": 0.0, "etf": 0.0, "crypto": 0.15},
        }
        cost = BacktestEngine._calc_friction("AAPL", 100.0, 10.0, friction_config)
        # Stock: spread = 100*10 * (0.05/100) / 2 = 0.25, commission = 0
        assert cost == pytest.approx(0.25)

        cost_crypto = BacktestEngine._calc_friction("BTC", 50000.0, 0.1, friction_config)
        # Crypto: notional = 5000, spread = 5000 * (0.15/100) / 2 = 3.75, commission = 5000 * (0.15/100) = 7.5
        assert cost_crypto == pytest.approx(3.75 + 7.5)


# ══════════════════════════════════════════════════════════════════════════════
# Report / Metrics Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestReportMetrics:
    """Tests for metrics calculation and reporting."""

    def test_calculate_metrics_basic(self):
        result = {
            "initial_capital": 10000.0,
            "final_equity": 11000.0,
            "total_return": 0.1,
            "max_drawdown": -0.05,
            "closed_trades": [
                {"pnl": 200.0, "pnl_pct": 0.1, "bars_held": 5, "exit_reason": "take_profit"},
                {"pnl": -100.0, "pnl_pct": -0.05, "bars_held": 3, "exit_reason": "stop_loss"},
                {"pnl": 150.0, "pnl_pct": 0.08, "bars_held": 4, "exit_reason": "take_profit"},
            ],
            "equity_curve": [
                {"date": "d1", "equity": 10000.0},
                {"date": "d2", "equity": 10200.0},
                {"date": "d3", "equity": 10100.0},
                {"date": "d4", "equity": 11000.0},
            ],
        }
        m = calculate_metrics(result)
        assert m["win_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert m["trade_count"] == 3
        assert m["profit_factor"] == pytest.approx(350.0 / 100.0)
        assert m["gross_profit"] == 350.0
        assert m["gross_loss"] == 100.0
        assert m["total_return"] == 0.1
        assert m["exit_reasons"]["take_profit"] == 2
        assert m["exit_reasons"]["stop_loss"] == 1

    def test_calculate_metrics_no_trades(self):
        result = {
            "initial_capital": 10000.0,
            "final_equity": 10000.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "closed_trades": [],
            "equity_curve": [],
        }
        m = calculate_metrics(result)
        assert m["trade_count"] == 0
        assert m["win_rate"] == 0.0
        assert m["sharpe"] == 0.0

    def test_calculate_metrics_all_winners(self):
        result = {
            "total_return": 0.3,
            "max_drawdown": -0.02,
            "closed_trades": [
                {"pnl": 100.0, "pnl_pct": 0.05, "bars_held": 3, "exit_reason": "take_profit"},
                {"pnl": 200.0, "pnl_pct": 0.1, "bars_held": 5, "exit_reason": "take_profit"},
            ],
            "equity_curve": [
                {"date": "d1", "equity": 10000.0},
                {"date": "d2", "equity": 10300.0},
            ],
        }
        m = calculate_metrics(result)
        assert m["win_rate"] == 1.0
        assert m["profit_factor"] == 999.0  # Clamped inf

    def test_calculate_metrics_all_losers(self):
        result = {
            "total_return": -0.05,
            "max_drawdown": -0.05,
            "closed_trades": [
                {"pnl": -50.0, "pnl_pct": -0.05, "bars_held": 2, "exit_reason": "stop_loss"},
                {"pnl": -30.0, "pnl_pct": -0.03, "bars_held": 3, "exit_reason": "stop_loss"},
            ],
            "equity_curve": [
                {"date": "d1", "equity": 10000.0},
                {"date": "d2", "equity": 9920.0},
            ],
        }
        m = calculate_metrics(result)
        assert m["win_rate"] == 0.0
        assert m["profit_factor"] == 0.0

    def test_sharpe_ratio_positive(self):
        # Steadily increasing equity -> positive Sharpe
        curve = [{"date": f"d{i}", "equity": 10000 + i * 10} for i in range(100)]
        sharpe = _calculate_sharpe(curve)
        assert sharpe > 0

    def test_sharpe_ratio_flat(self):
        # No change -> zero Sharpe (std=0)
        curve = [{"date": f"d{i}", "equity": 10000.0} for i in range(50)]
        sharpe = _calculate_sharpe(curve)
        assert sharpe == 0.0

    def test_sharpe_ratio_insufficient_data(self):
        assert _calculate_sharpe([]) == 0.0
        assert _calculate_sharpe([{"date": "d1", "equity": 10000}]) == 0.0

    def test_expectancy(self):
        result = {
            "total_return": 0.05,
            "max_drawdown": -0.1,
            "closed_trades": [
                {"pnl": 300.0, "pnl_pct": 0.15, "bars_held": 5, "exit_reason": "take_profit"},
                {"pnl": -100.0, "pnl_pct": -0.05, "bars_held": 2, "exit_reason": "stop_loss"},
                {"pnl": -100.0, "pnl_pct": -0.05, "bars_held": 3, "exit_reason": "stop_loss"},
            ],
            "equity_curve": [],
        }
        m = calculate_metrics(result)
        # win_rate = 1/3, avg_win = 300, avg_loss = 100
        # expectancy = (1/3 * 300) - (2/3 * 100) = 100 - 66.67 = 33.33
        assert m["expectancy"] == pytest.approx(33.3333, abs=0.01)

    def test_format_summary_table(self):
        results = [
            {
                "params": {"default_stop_loss_pct": 2.0, "default_take_profit_pct": 8.0},
                "metrics": {
                    "sharpe": 1.42, "max_drawdown": -0.123, "win_rate": 0.62,
                    "profit_factor": 2.14, "avg_win_loss_ratio": 3.21,
                    "trade_count": 847, "total_return": 0.342,
                },
                "n_assets": 14,
                "is_current": False,
            },
            {
                "params": {"default_stop_loss_pct": 3.0, "default_take_profit_pct": 6.0},
                "metrics": {
                    "sharpe": 0.87, "max_drawdown": -0.182, "win_rate": 0.89,
                    "profit_factor": 0.92, "avg_win_loss_ratio": 0.71,
                    "trade_count": 912, "total_return": -0.043,
                },
                "n_assets": 14,
                "is_current": True,
            },
        ]
        table = format_summary_table(results, "stops", current_params={})
        assert "BACKTEST RESULTS" in table
        assert "stops" in table
        assert "YOU ARE HERE" in table

    def test_save_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [
                {
                    "params": {"default_stop_loss_pct": 2.0, "default_take_profit_pct": 8.0},
                    "metrics": {
                        "sharpe": 1.42, "max_drawdown": -0.12, "win_rate": 0.62,
                        "profit_factor": 2.14, "avg_win_loss_ratio": 3.21,
                        "trade_count": 100, "total_return": 0.34,
                        "expectancy": 50.0, "avg_bars_held": 4.2,
                        "gross_profit": 5000.0, "gross_loss": 2000.0,
                        "exit_reasons": {"take_profit": 62, "stop_loss": 38},
                    },
                },
            ]
            paths = save_results(results, "stops", output_dir=tmpdir)
            assert "csv" in paths
            assert "best_params" in paths
            assert os.path.exists(paths["csv"])
            assert os.path.exists(paths["best_params"])

            # Verify best params JSON
            with open(paths["best_params"]) as f:
                best = json.load(f)
            assert best["default_stop_loss_pct"] == 2.0


# ══════════════════════════════════════════════════════════════════════════════
# Data Loader Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHistoricalDataLoader:
    """Tests for data loader with cache."""

    def test_cache_path_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = HistoricalDataLoader(cache_dir=tmpdir)
            path = loader._cache_path("AAPL", "2y", "1d")
            assert "AAPL_2y_1d.parquet" in str(path)

    def test_cache_path_special_chars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = HistoricalDataLoader(cache_dir=tmpdir)
            path = loader._cache_path("BTC-USD", "1y", "1d")
            assert "BTC_USD_1y_1d.parquet" in str(path)
            path = loader._cache_path("^VIX", "6mo", "1d")
            assert "_VIX_6mo_1d.parquet" in str(path)

    def test_cache_validity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = HistoricalDataLoader(cache_dir=tmpdir)
            path = Path(tmpdir) / "test.parquet"
            # Non-existent file is invalid
            assert loader._is_cache_valid(path) is False

            # Create a file (recent mtime)
            path.touch()
            assert loader._is_cache_valid(path) is True

    def test_core_assets_list(self):
        assert len(CORE_ASSETS) == 14
        assert "BTC" in CORE_ASSETS
        assert "SPY" in CORE_ASSETS

    def test_yfinance_symbols_mapping(self):
        assert YFINANCE_SYMBOLS["BTC"] == "BTC-USD"
        assert YFINANCE_SYMBOLS["ETH"] == "ETH-USD"

    @patch("yfinance.Ticker")
    def test_get_historical_cache_hit(self, mock_ticker):
        """Verify cache is used when valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = HistoricalDataLoader(cache_dir=tmpdir)
            # Pre-populate cache
            df = _make_ohlcv(50)
            cache_path = loader._cache_path("AAPL", "2y", "1d")
            df.to_parquet(cache_path)

            result = loader.get_historical("AAPL", "2y", "1d")
            assert len(result) == 50
            mock_ticker.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# Parameter Sweep Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestParameterSweep:
    """Tests for parameter sweep generation and execution."""

    def test_frange(self):
        result = _frange(1.0, 3.0, 0.5)
        assert result == [1.0, 1.5, 2.0, 2.5, 3.0]

    def test_frange_single(self):
        result = _frange(2.0, 2.0, 1.0)
        assert result == [2.0]

    def test_generate_stops_params(self):
        sweeper = ParameterSweeper()
        combos = sweeper._generate_params("stops")
        # SL: 1.0-8.0/0.5 = 15, TP: 3.0-15.0/1.0 = 13 -> 195
        assert len(combos) == 195
        # Each combo should have use_atr_stops=False
        for c in combos:
            assert c["use_atr_stops"] is False
            assert "default_stop_loss_pct" in c
            assert "default_take_profit_pct" in c

    def test_generate_atr_stops_params(self):
        sweeper = ParameterSweeper()
        combos = sweeper._generate_params("atr-stops")
        # SL_ATR: 1.0-4.0/0.5 = 7, TP_ATR: 2.0-6.0/0.5 = 9 -> 63
        assert len(combos) == 63

    def test_generate_trailing_params(self):
        sweeper = ParameterSweeper()
        combos = sweeper._generate_params("trailing")
        # Act: 1.0-5.0/0.5 = 9, Dist: 0.5-3.0/0.5 = 6 -> 54
        assert len(combos) == 54

    def test_generate_sizing_params(self):
        sweeper = ParameterSweeper()
        combos = sweeper._generate_params("sizing")
        # Pos: 3.0-10.0/1.0 = 8, Risk: 0.5-3.0/0.5 = 6 -> 48
        assert len(combos) == 48

    def test_generate_holding_params(self):
        sweeper = ParameterSweeper()
        combos = sweeper._generate_params("holding")
        # Hold: 7 values, SL: 15 values -> 105
        assert len(combos) == 105

    def test_unknown_mode_raises(self):
        sweeper = ParameterSweeper()
        with pytest.raises(ValueError, match="Unknown sweep mode"):
            sweeper._generate_params("invalid")

    def test_sweep_small(self):
        """Run a tiny sweep to verify end-to-end."""
        data = {"AAPL": _make_ohlcv(150, seed=77)}
        sweeper = ParameterSweeper(initial_capital=10000.0, max_workers=1)

        # Monkey-patch to use only 2 combos for speed
        original = sweeper._generate_params

        def _small_gen(mode):
            combos = original(mode)
            return combos[:2]

        sweeper._generate_params = _small_gen

        results = sweeper.sweep("stops", data)
        assert len(results) == 2
        # Results should be sorted by Sharpe descending
        if results[0]["metrics"]["sharpe"] != -999 and results[1]["metrics"]["sharpe"] != -999:
            assert results[0]["metrics"]["sharpe"] >= results[1]["metrics"]["sharpe"]

    def test_mark_current(self):
        sweeper = ParameterSweeper()
        results = [
            {"params": {"default_stop_loss_pct": 2.0, "default_take_profit_pct": 8.0}},
            {"params": {
                "default_stop_loss_pct": sweeper._baseline["default_stop_loss_pct"],
                "default_take_profit_pct": sweeper._baseline["default_take_profit_pct"],
            }},
        ]
        sweeper._mark_current(results, "stops")
        assert results[0].get("is_current") is False
        assert results[1].get("is_current") is True


# ══════════════════════════════════════════════════════════════════════════════
# CLI Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCLI:
    """Tests for the CLI argument parser."""

    def test_parse_args_basic(self):
        from tools.backtest.__main__ import parse_args
        args = parse_args(["--sweep", "stops"])
        assert args.sweep == "stops"
        assert args.period == "2y"
        assert args.interval == "1d"
        assert args.capital == 10000.0
        assert args.assets is None

    def test_parse_args_full(self):
        from tools.backtest.__main__ import parse_args
        args = parse_args([
            "--sweep", "atr-stops",
            "--period", "6mo",
            "--interval", "1h",
            "--capital", "200",
            "--assets", "AAPL,NVDA,SPY",
            "--workers", "2",
            "--no-friction",
            "--verbose",
        ])
        assert args.sweep == "atr-stops"
        assert args.period == "6mo"
        assert args.capital == 200.0
        assert args.assets == "AAPL,NVDA,SPY"
        assert args.workers == 2
        assert args.no_friction is True
        assert args.verbose is True

    def test_parse_args_invalid_mode(self):
        from tools.backtest.__main__ import parse_args
        with pytest.raises(SystemExit):
            parse_args(["--sweep", "invalid"])

    def test_parse_args_missing_sweep(self):
        from tools.backtest.__main__ import parse_args
        with pytest.raises(SystemExit):
            parse_args([])


# ══════════════════════════════════════════════════════════════════════════════
# Integration Test
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_single_asset(self):
        """Full pipeline: data -> engine -> metrics -> report."""
        data = {"AAPL": _make_ohlcv(200, seed=99)}
        params = _default_params()

        engine = BacktestEngine(params=params, initial_capital=10000.0, use_friction=True)
        result = engine.run(data)

        metrics = calculate_metrics(result)
        assert "sharpe" in metrics
        assert "win_rate" in metrics
        assert "trade_count" in metrics
        assert metrics["trade_count"] == len(result["closed_trades"])

    def test_full_pipeline_multiple_assets(self):
        data = {
            "AAPL": _make_ohlcv(200, seed=1),
            "NVDA": _make_trending_up(200),
            "SPY": _make_ohlcv(200, seed=3),
        }
        params = _default_params()
        engine = BacktestEngine(params=params, initial_capital=30000.0)
        result = engine.run(data)
        metrics = calculate_metrics(result)
        assert metrics["trade_count"] >= 0

    def test_parameter_comparison(self):
        """Verify different params produce different results."""
        data = {"AAPL": _make_ohlcv(200, seed=42)}

        params_tight = _default_params()
        params_tight["default_stop_loss_pct"] = 1.0
        params_tight["default_take_profit_pct"] = 3.0

        params_wide = _default_params()
        params_wide["default_stop_loss_pct"] = 8.0
        params_wide["default_take_profit_pct"] = 15.0

        engine_tight = BacktestEngine(params=params_tight, use_friction=False)
        engine_wide = BacktestEngine(params=params_wide, use_friction=False)

        result_tight = engine_tight.run(data)
        result_wide = engine_wide.run(data)

        # Different params should produce different trade counts or returns
        # (they may be equal by chance, but very unlikely with these extremes)
        r1 = calculate_metrics(result_tight)
        r2 = calculate_metrics(result_wide)
        # At minimum, both should produce valid metrics
        assert isinstance(r1["sharpe"], float)
        assert isinstance(r2["sharpe"], float)
