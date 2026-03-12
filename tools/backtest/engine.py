"""Backtest engine — core simulation loop.

Processes OHLCV bars chronologically, checking stops/TP/trailing on each bar,
generating signals, sizing positions, and tracking equity. This is the heart
of the backtester.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from tools.backtest.portfolio_sim import SimulatedPortfolio, Position
from tools.backtest.signals import SignalGenerator, BacktestSignal
from tools.technical_indicators import TechnicalIndicators

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config")
RISK_PARAMS_PATH = os.path.join(CONFIG_DIR, "risk_params.json")


def load_risk_params() -> dict[str, Any]:
    """Load baseline risk parameters from config."""
    with open(RISK_PARAMS_PATH) as f:
        return json.load(f)


class BacktestEngine:
    """Run a single backtest with a given set of parameters.

    Processes OHLCV data bar-by-bar:
    1. Update prices for held positions
    2. Check stop-losses (fixed % and ATR-based)
    3. Check take-profits (fixed % and ATR-based)
    4. Check trailing stops (activate, update, trigger)
    5. Check holding period forced exits
    6. Generate new entry signals
    7. Size position, apply friction, record trade
    8. Record equity snapshot
    """

    def __init__(
        self,
        params: dict[str, Any],
        initial_capital: float = 10000.0,
        use_friction: bool = True,
    ):
        self.params = params
        self.initial_capital = initial_capital
        self.use_friction = use_friction
        self._ti = TechnicalIndicators()
        self._signal_gen = SignalGenerator()

    def run(
        self,
        data: dict[str, pd.DataFrame],
        warmup: int = 50,
    ) -> dict[str, Any]:
        """Run backtest across all assets.

        Args:
            data: Dict mapping symbol -> OHLCV DataFrame.
            warmup: Bars to skip for indicator warmup.

        Returns:
            Dict with portfolio, closed_trades, equity_curve, params.
        """
        portfolio = SimulatedPortfolio(initial_capital=self.initial_capital)

        # Extract parameters
        sl_pct = self.params.get("default_stop_loss_pct", 3.0) / 100.0
        tp_pct = self.params.get("default_take_profit_pct", 6.0) / 100.0
        sl_atr_mult = self.params.get("stop_loss_atr_mult", 2.0)
        tp_atr_mult = self.params.get("take_profit_atr_mult", 3.0)
        trailing_activation = self.params.get("trailing_stop_activation_pct", 2.0) / 100.0
        trailing_distance = self.params.get("trailing_stop_distance_pct", 1.5) / 100.0
        max_position_pct = self.params.get("max_position_pct", 5.0) / 100.0
        base_risk_pct = self.params.get("base_risk_per_trade_pct", 1.5) / 100.0
        max_holding_bars = self.params.get("max_holding_bars", 72)  # In bars (days for daily data)
        use_atr_stops = self.params.get("use_atr_stops", True)

        # Friction config
        friction_config = self.params.get("trading_friction", {})
        friction_enabled = self.use_friction and friction_config.get("enabled", True)

        # Pre-generate signals for all assets
        all_signals: dict[str, list[BacktestSignal]] = {}
        for symbol, df in data.items():
            all_signals[symbol] = self._signal_gen.generate(symbol, df, warmup=warmup)

        # Build a unified timeline of all bars across all assets
        # Use the longest asset's date range as the timeline
        all_dates: set[str] = set()
        for df in data.values():
            all_dates.update(str(d) for d in df.index)
        timeline = sorted(all_dates)

        if not timeline:
            return self._build_result(portfolio, data)

        # Index signals by (asset, bar_index) for quick lookup
        signal_lookup: dict[tuple[str, int], BacktestSignal] = {}
        for symbol, sigs in all_signals.items():
            for sig in sigs:
                signal_lookup[(symbol, sig.bar_index)] = sig

        # Process each bar in chronological order
        for global_bar, date_str in enumerate(timeline):
            bar_prices: dict[str, float] = {}

            # Update prices and collect current bar data per asset
            for symbol, df in data.items():
                if date_str not in [str(d) for d in df.index]:
                    continue
                idx = list(str(d) for d in df.index).index(date_str)
                row = df.iloc[idx]
                current_price = float(row["close"])
                bar_prices[symbol] = current_price
                current_high = float(row["high"])
                current_low = float(row["low"])

                # Get ATR for this bar
                if idx >= 15:
                    highs = df["high"].iloc[:idx + 1].tolist()
                    lows = df["low"].iloc[:idx + 1].tolist()
                    closes = df["close"].iloc[:idx + 1].tolist()
                    atr = self._ti.atr(highs, lows, closes, period=14)
                else:
                    atr = 0.0

                # 1. Check exits for positions in this asset
                positions_to_check = portfolio.get_positions_for_asset(symbol)
                for pos in positions_to_check[:]:  # Copy to allow mutation
                    bars_held = global_bar - pos.entry_bar

                    # Calculate friction for exit
                    exit_friction = 0.0
                    if friction_enabled:
                        exit_friction = self._calc_friction(symbol, current_price, pos.quantity, friction_config)

                    # Check stop-loss
                    if self._check_stop_loss(pos, current_low, current_high):
                        exit_price = pos.stop_loss  # Exit at stop price
                        portfolio.close_position(pos, exit_price, global_bar, date_str, "stop_loss", exit_friction)
                        continue

                    # Check take-profit
                    if self._check_take_profit(pos, current_high, current_low):
                        exit_price = pos.take_profit
                        portfolio.close_position(pos, exit_price, global_bar, date_str, "take_profit", exit_friction)
                        continue

                    # Update and check trailing stop
                    if self._update_trailing_stop(pos, current_price, trailing_activation, trailing_distance):
                        exit_price = pos.trailing_stop if pos.trailing_stop else current_price
                        portfolio.close_position(pos, exit_price, global_bar, date_str, "trailing_stop", exit_friction)
                        continue

                    # Check holding period
                    if max_holding_bars > 0 and bars_held >= max_holding_bars:
                        portfolio.close_position(pos, current_price, global_bar, date_str, "holding_period", exit_friction)
                        continue

                    # Apply daily borrow cost for shorts
                    if pos.direction == "short" and friction_enabled:
                        borrow = self._calc_borrow(symbol, current_price, pos.quantity, friction_config)
                        portfolio.apply_borrow_cost(pos, borrow)

                # 2. Check for new entry signals at this bar
                # Find the bar index in this asset's DataFrame
                sig = signal_lookup.get((symbol, idx))
                if sig is not None:
                    # Calculate ATR-based stops if available
                    if use_atr_stops and atr > 0:
                        if sig.direction == "long":
                            stop_loss = current_price - atr * sl_atr_mult
                            take_profit = current_price + atr * tp_atr_mult
                        else:
                            stop_loss = current_price + atr * sl_atr_mult
                            take_profit = current_price - atr * tp_atr_mult
                    else:
                        # Fixed percentage stops
                        if sig.direction == "long":
                            stop_loss = current_price * (1 - sl_pct)
                            take_profit = current_price * (1 + tp_pct)
                        else:
                            stop_loss = current_price * (1 + sl_pct)
                            take_profit = current_price * (1 - tp_pct)

                    # Position sizing
                    position_value = self._calculate_position_size(
                        confidence=sig.confidence,
                        atr=atr,
                        portfolio_value=portfolio.equity,
                        base_risk_pct=base_risk_pct,
                        sl_atr_mult=sl_atr_mult,
                        max_position_pct=max_position_pct,
                    )

                    if position_value <= 0 or current_price <= 0:
                        continue

                    quantity = position_value / current_price

                    # Entry friction
                    entry_friction = 0.0
                    if friction_enabled:
                        entry_friction = self._calc_friction(symbol, current_price, quantity, friction_config)

                    portfolio.open_position(
                        asset=symbol,
                        direction=sig.direction,
                        price=current_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        bar_index=global_bar,
                        date_str=date_str,
                        friction_cost=entry_friction,
                    )

            # Record equity snapshot
            portfolio.record_equity_snapshot(date_str, bar_prices)

        # Close any remaining positions at end of data
        for pos in portfolio.positions[:]:
            symbol = pos.asset
            if symbol in data:
                last_price = float(data[symbol]["close"].iloc[-1])
            else:
                last_price = pos.entry_price
            exit_friction = 0.0
            if friction_enabled:
                exit_friction = self._calc_friction(symbol, last_price, pos.quantity, friction_config)
            portfolio.close_position(
                pos, last_price, len(timeline) - 1, timeline[-1], "end_of_data", exit_friction
            )

        return self._build_result(portfolio, data)

    def _check_stop_loss(self, pos: Position, bar_low: float, bar_high: float) -> bool:
        """Check if stop-loss was hit during this bar."""
        if pos.direction == "long":
            return bar_low <= pos.stop_loss
        else:
            return bar_high >= pos.stop_loss

    def _check_take_profit(self, pos: Position, bar_high: float, bar_low: float) -> bool:
        """Check if take-profit was hit during this bar."""
        if pos.direction == "long":
            return bar_high >= pos.take_profit
        else:
            return bar_low <= pos.take_profit

    def _update_trailing_stop(
        self,
        pos: Position,
        current_price: float,
        activation_pct: float,
        distance_pct: float,
    ) -> bool:
        """Update trailing stop and return True if triggered.

        Trailing stop activates when price moves favorably beyond activation_pct.
        Once active, the stop ratchets to lock in profits, trailing behind
        the highest (long) or lowest (short) price by distance_pct.
        """
        if pos.direction == "long":
            move_pct = (current_price - pos.entry_price) / pos.entry_price
            if move_pct >= activation_pct:
                new_trail = current_price * (1 - distance_pct)
                if not pos.trailing_activated:
                    pos.trailing_activated = True
                    pos.trailing_stop = new_trail
                elif new_trail > (pos.trailing_stop or 0):
                    pos.trailing_stop = new_trail

                # Check if trailing stop was hit
                if pos.trailing_stop and current_price <= pos.trailing_stop:
                    return True
        else:
            move_pct = (pos.entry_price - current_price) / pos.entry_price
            if move_pct >= activation_pct:
                new_trail = current_price * (1 + distance_pct)
                if not pos.trailing_activated:
                    pos.trailing_activated = True
                    pos.trailing_stop = new_trail
                elif new_trail < (pos.trailing_stop or float("inf")):
                    pos.trailing_stop = new_trail

                if pos.trailing_stop and current_price >= pos.trailing_stop:
                    return True

        return False

    def _calculate_position_size(
        self,
        confidence: float,
        atr: float,
        portfolio_value: float,
        base_risk_pct: float,
        sl_atr_mult: float,
        max_position_pct: float,
    ) -> float:
        """ATR-based position sizing, same logic as RiskManager.calculate_position_size()."""
        if atr <= 0 or portfolio_value <= 0:
            # Fallback to simple percentage
            return portfolio_value * max_position_pct * min(max(confidence, 0.0), 1.0)

        stop_distance = atr * sl_atr_mult
        position_value = (portfolio_value * base_risk_pct) / stop_distance

        # Scale by confidence
        confidence_scalar = min(max(confidence, 0.0), 1.0)
        position_value *= confidence_scalar

        # Cap at max position size
        max_value = portfolio_value * max_position_pct
        return min(position_value, max_value)

    @staticmethod
    def _calc_friction(
        asset: str, price: float, quantity: float, friction_config: dict
    ) -> float:
        """Calculate total entry/exit friction (spread + commission)."""
        asset_type = "crypto" if asset in ("BTC", "ETH") else (
            "etf" if asset in ("SPY", "GLDM", "SLV", "TLT", "XLE", "EWS", "FXI") else "stock"
        )
        spread_pcts = friction_config.get("spread_pct", {})
        comm_pcts = friction_config.get("commission_pct", {})
        spread_pct = spread_pcts.get(asset_type, 0.05) / 100.0
        comm_pct = comm_pcts.get(asset_type, 0.0) / 100.0

        spread_cost = price * quantity * spread_pct / 2
        comm_cost = price * quantity * comm_pct
        return spread_cost + comm_cost

    @staticmethod
    def _calc_borrow(
        asset: str, price: float, quantity: float, friction_config: dict
    ) -> float:
        """Calculate daily borrow cost for shorts."""
        asset_type = "crypto" if asset in ("BTC", "ETH") else (
            "etf" if asset in ("SPY", "GLDM", "SLV", "TLT", "XLE", "EWS", "FXI") else "stock"
        )
        borrow_rates = friction_config.get("short_borrow_annual_pct", {})
        annual_pct = borrow_rates.get(asset_type, 1.5) / 100.0
        return (price * quantity * annual_pct) / 365.0

    @staticmethod
    def _build_result(portfolio: SimulatedPortfolio, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Build the result dict from a completed backtest."""
        return {
            "initial_capital": portfolio.initial_capital,
            "final_equity": round(portfolio.equity, 2),
            "total_return": round((portfolio.equity - portfolio.initial_capital) / portfolio.initial_capital, 4),
            "peak_equity": round(portfolio.peak_equity, 2),
            "max_drawdown": round(portfolio.max_drawdown, 4),
            "closed_trades": [t.to_dict() for t in portfolio.closed_trades],
            "trade_count": len(portfolio.closed_trades),
            "equity_curve": portfolio.equity_curve,
            "assets_traded": list(data.keys()),
        }
