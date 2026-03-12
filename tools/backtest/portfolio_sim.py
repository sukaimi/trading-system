"""Simulated portfolio for backtesting — tracks equity, positions, drawdown.

Pure Python, no external dependencies beyond numpy for math. Tracks positions,
applies friction, records equity curve and closed trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Position:
    """An open position in the simulated portfolio."""

    asset: str
    direction: str  # "long" or "short"
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    trailing_stop: float | None = None
    trailing_activated: bool = False
    entry_bar: int = 0
    entry_date: str = ""

    @property
    def notional(self) -> float:
        return self.entry_price * self.quantity


@dataclass
class ClosedTrade:
    """A completed trade with P&L info."""

    asset: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    bars_held: int
    exit_reason: str  # stop_loss | take_profit | trailing_stop | holding_period | end_of_data
    entry_date: str = ""
    exit_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": round(self.pnl, 4),
            "pnl_pct": round(self.pnl_pct, 4),
            "bars_held": self.bars_held,
            "exit_reason": self.exit_reason,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
        }


class SimulatedPortfolio:
    """Track simulated portfolio state during a backtest run.

    Manages cash, positions, equity curve, and closed trades.
    Friction (spread + commission) is applied on entry and exit.
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: list[Position] = []
        self.closed_trades: list[ClosedTrade] = []
        self.equity_curve: list[dict[str, Any]] = []
        self.peak_equity = initial_capital

    @property
    def equity(self) -> float:
        """Current equity = cash + sum of open position values."""
        position_value = sum(p.notional for p in self.positions)
        return self.cash + position_value

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a negative percentage (e.g., -0.12 = -12%)."""
        if self.peak_equity <= 0:
            return 0.0
        if not self.equity_curve:
            return 0.0
        max_dd = 0.0
        peak = self.initial_capital
        for snap in self.equity_curve:
            eq = snap["equity"]
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd
        return max_dd

    def open_position(
        self,
        asset: str,
        direction: str,
        price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        bar_index: int,
        date_str: str,
        friction_cost: float = 0.0,
    ) -> Position | None:
        """Open a new position. Returns the Position or None if insufficient cash."""
        cost = price * quantity + friction_cost
        if cost > self.cash:
            return None

        self.cash -= cost
        pos = Position(
            asset=asset,
            direction=direction,
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_bar=bar_index,
            entry_date=date_str,
        )
        self.positions.append(pos)
        return pos

    def close_position(
        self,
        position: Position,
        exit_price: float,
        bar_index: int,
        date_str: str,
        exit_reason: str,
        friction_cost: float = 0.0,
    ) -> ClosedTrade:
        """Close an open position and record the trade."""
        # Calculate P&L
        if position.direction == "long":
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity

        pnl -= friction_cost  # Deduct exit friction

        pnl_pct = pnl / (position.entry_price * position.quantity) if position.entry_price > 0 else 0.0

        # Return cash from sale
        proceeds = exit_price * position.quantity - friction_cost
        self.cash += proceeds

        bars_held = bar_index - position.entry_bar

        trade = ClosedTrade(
            asset=position.asset,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            bars_held=bars_held,
            exit_reason=exit_reason,
            entry_date=position.entry_date,
            exit_date=date_str,
        )
        self.closed_trades.append(trade)

        # Remove from open positions
        if position in self.positions:
            self.positions.remove(position)

        return trade

    def record_equity_snapshot(self, date_str: str, bar_prices: dict[str, float]) -> None:
        """Record current equity to the equity curve.

        Updates position values based on current prices before recording.
        """
        # Update position notional values based on current prices
        # (positions track entry_price for P&L, but equity uses current market value)
        position_value = 0.0
        for pos in self.positions:
            current_price = bar_prices.get(pos.asset, pos.entry_price)
            if pos.direction == "long":
                position_value += current_price * pos.quantity
            else:
                # Short: value = 2 * entry - current (profit when price drops)
                position_value += (2 * pos.entry_price - current_price) * pos.quantity

        eq = self.cash + position_value
        if eq > self.peak_equity:
            self.peak_equity = eq

        self.equity_curve.append({
            "date": date_str,
            "equity": round(eq, 2),
        })

    def apply_borrow_cost(self, position: Position, daily_cost: float) -> None:
        """Deduct daily borrow cost for short positions from cash."""
        self.cash -= daily_cost

    def get_positions_for_asset(self, asset: str) -> list[Position]:
        """Get all open positions for a given asset."""
        return [p for p in self.positions if p.asset == asset]
