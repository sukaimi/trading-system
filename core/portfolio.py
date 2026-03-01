"""Thread-safe portfolio state with JSON persistence.

PortfolioState tracks open positions, daily P&L, drawdown, and
provides snapshot/persist/load for crash recovery.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.portfolio")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
STATE_FILE = os.path.join(DATA_DIR, "portfolio_state.json")


class PortfolioState:
    """Shared portfolio state — thread-safe, JSON-persistent."""

    def __init__(
        self,
        initial_capital: float = 100.0,
        state_file: str = STATE_FILE,
    ):
        self._lock = threading.Lock()
        self.state_file = state_file

        self.equity: float = initial_capital
        self.initial_capital: float = initial_capital
        self.peak_equity: float = initial_capital
        self.open_positions: list[dict[str, Any]] = []
        self.daily_pnl: float = 0.0
        self.daily_pnl_pct: float = 0.0
        self.drawdown_from_peak_pct: float = 0.0
        self.consecutive_losses: int = 0
        self.total_trades: int = 0
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.halted: bool = False
        self.last_updated: str = datetime.utcnow().isoformat()

    # ── Thread-safe accessors ──────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of the current state as a plain dict."""
        with self._lock:
            return {
                "equity": self.equity,
                "initial_capital": self.initial_capital,
                "peak_equity": self.peak_equity,
                "open_positions": list(self.open_positions),
                "daily_pnl": self.daily_pnl,
                "daily_pnl_pct": self.daily_pnl_pct,
                "drawdown_from_peak_pct": self.drawdown_from_peak_pct,
                "consecutive_losses": self.consecutive_losses,
                "total_trades": self.total_trades,
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "halted": self.halted,
                "last_updated": self.last_updated,
            }

    def update_equity(self, new_equity: float) -> None:
        """Update equity and recalculate drawdown."""
        with self._lock:
            self.equity = new_equity
            if new_equity > self.peak_equity:
                self.peak_equity = new_equity
            self.drawdown_from_peak_pct = (
                ((self.peak_equity - self.equity) / self.peak_equity) * 100
                if self.peak_equity > 0
                else 0.0
            )
            self.last_updated = datetime.utcnow().isoformat()

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade and update stats."""
        with self._lock:
            self.total_trades += 1
            self.daily_pnl += pnl
            self.daily_pnl_pct = (
                (self.daily_pnl / self.equity) * 100 if self.equity > 0 else 0.0
            )
            if pnl >= 0:
                self.total_wins += 1
                self.consecutive_losses = 0
            else:
                self.total_losses += 1
                self.consecutive_losses += 1
            self.last_updated = datetime.utcnow().isoformat()

    def add_position(self, position: dict[str, Any]) -> None:
        """Add an open position."""
        with self._lock:
            self.open_positions.append(position)
            self.last_updated = datetime.utcnow().isoformat()

    def remove_position(self, trade_id: str) -> dict[str, Any] | None:
        """Remove and return a position by trade_id."""
        with self._lock:
            for i, pos in enumerate(self.open_positions):
                if pos.get("trade_id") == trade_id:
                    removed = self.open_positions.pop(i)
                    self.last_updated = datetime.utcnow().isoformat()
                    return removed
            return None

    def reset_daily(self) -> None:
        """Reset daily P&L counters (called at start of each trading day)."""
        with self._lock:
            self.daily_pnl = 0.0
            self.daily_pnl_pct = 0.0
            self.last_updated = datetime.utcnow().isoformat()

    # ── Persistence ────────────────────────────────────────────────────

    def persist(self) -> None:
        """Save state to JSON file."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        state = self.snapshot()
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
        log.info("Portfolio state persisted to %s", self.state_file)

    def load(self) -> bool:
        """Load state from JSON file. Returns True if loaded successfully."""
        if not os.path.exists(self.state_file):
            log.info("No persisted state found at %s", self.state_file)
            return False

        try:
            with open(self.state_file) as f:
                state = json.load(f)
            with self._lock:
                self.equity = state.get("equity", self.equity)
                self.initial_capital = state.get("initial_capital", self.initial_capital)
                self.peak_equity = state.get("peak_equity", self.peak_equity)
                self.open_positions = state.get("open_positions", [])
                self.daily_pnl = state.get("daily_pnl", 0.0)
                self.daily_pnl_pct = state.get("daily_pnl_pct", 0.0)
                self.drawdown_from_peak_pct = state.get("drawdown_from_peak_pct", 0.0)
                self.consecutive_losses = state.get("consecutive_losses", 0)
                self.total_trades = state.get("total_trades", 0)
                self.total_wins = state.get("total_wins", 0)
                self.total_losses = state.get("total_losses", 0)
                self.halted = state.get("halted", False)
                self.last_updated = state.get(
                    "last_updated", datetime.utcnow().isoformat()
                )
            log.info("Portfolio state loaded from %s", self.state_file)
            return True
        except (json.JSONDecodeError, KeyError) as e:
            log.error("Failed to load portfolio state: %s", e)
            return False
