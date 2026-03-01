"""Shared test fixtures for the trading system test suite."""

import json
import os
import tempfile

import pytest


@pytest.fixture
def risk_config():
    """Standard risk configuration for tests."""
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
def sample_portfolio_state():
    """A healthy portfolio state with one open position."""
    return {
        "equity": 100.0,
        "initial_capital": 100.0,
        "peak_equity": 105.0,
        "open_positions": [
            {
                "trade_id": "trade_001",
                "asset": "BTC",
                "direction": "long",
                "entry_price": 62500.0,
                "position_size_pct": 3.5,
            }
        ],
        "daily_pnl": 1.5,
        "daily_pnl_pct": 1.5,
        "drawdown_from_peak_pct": 4.76,
        "consecutive_losses": 0,
        "total_trades": 5,
        "total_wins": 3,
        "total_losses": 2,
        "halted": False,
    }


@pytest.fixture
def sample_execution_order():
    """A valid execution order that should pass risk checks."""
    return {
        "type": "execution_order",
        "thesis_id": "thesis_001",
        "asset": "ETH",
        "direction": "long",
        "quantity": 0.01,
        "order_type": "market",
        "stop_loss": 3000.0,
        "take_profit": 3500.0,
        "position_size_pct": 5.0,
    }


@pytest.fixture
def empty_portfolio_state():
    """An empty portfolio with no positions."""
    return {
        "equity": 100.0,
        "initial_capital": 100.0,
        "peak_equity": 100.0,
        "open_positions": [],
        "daily_pnl": 0.0,
        "daily_pnl_pct": 0.0,
        "drawdown_from_peak_pct": 0.0,
        "consecutive_losses": 0,
        "total_trades": 0,
        "total_wins": 0,
        "total_losses": 0,
        "halted": False,
    }


@pytest.fixture
def tmp_state_file(tmp_path):
    """Temporary file path for portfolio state persistence tests."""
    return str(tmp_path / "test_portfolio_state.json")
