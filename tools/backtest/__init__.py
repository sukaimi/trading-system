"""Backtest module — parameter sweep optimization against historical data.

Validates and optimizes risk parameters (stop-loss, take-profit, trailing stops,
position sizing, holding periods) using yfinance historical data. Does NOT
simulate LLM agent decisions — tests what happens after a trade is entered.

Usage:
    python -m tools.backtest --sweep stops --period 2y
"""

from tools.backtest.engine import BacktestEngine
from tools.backtest.param_sweep import ParameterSweeper

__all__ = ["BacktestEngine", "ParameterSweeper"]
