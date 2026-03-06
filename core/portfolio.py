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

from core.event_bus import event_bus
from core.logger import setup_logger

log = setup_logger("trading.portfolio")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
STATE_FILE = os.path.join(DATA_DIR, "portfolio_state.json")
EQUITY_HISTORY_FILE = os.path.join(DATA_DIR, "equity_history.json")


class PortfolioState:
    """Shared portfolio state — thread-safe, JSON-persistent."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
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

    def calculate_equity(self, market_data_fetcher: Any) -> float:
        """Recalculate equity from cash + unrealized value of open positions.

        cash_basis = current equity minus the cost basis of open positions
        new_equity = cash_basis + sum(current_price * quantity) for each position
        """
        with self._lock:
            positions = list(self.open_positions)
            current_equity = self.equity

        if not positions:
            return current_equity

        # Snapshot old unrealized P&L BEFORE updating positions
        old_unrealized = sum(
            p.get("unrealized_pnl", 0.0) for p in positions
        )

        # Calculate new unrealized P&L across all positions
        total_unrealized_pnl = 0.0
        for pos in positions:
            asset = pos.get("asset", "")
            quantity = pos.get("quantity", 0.0)
            entry_price = pos.get("entry_price", 0.0)
            direction = pos.get("direction", "long")
            if not asset or quantity == 0:
                continue
            price_data = market_data_fetcher.get_price(asset)
            price = price_data.get("price", 0.0)
            if price <= 0:
                price = entry_price  # fallback
            # Longs profit when price goes up, shorts profit when price goes down
            if direction == "short":
                pnl = (entry_price - price) * quantity
            else:
                pnl = (price - entry_price) * quantity
            pnl_pct = ((pnl) / (entry_price * quantity) * 100) if entry_price and quantity else 0
            pos["current_price"] = price
            pos["unrealized_pnl"] = round(pnl, 4)
            pos["unrealized_pnl_pct"] = round(pnl_pct, 2)
            total_unrealized_pnl += pnl

            # Track MAE (max adverse excursion) and MFE (max favorable excursion)
            if pnl_pct < pos.get("mae_pct", 0.0):
                pos["mae_pct"] = round(pnl_pct, 2)
            if pnl_pct > pos.get("mfe_pct", 0.0):
                pos["mfe_pct"] = round(pnl_pct, 2)

        # Base equity = current equity minus old unrealized (gives capital + realized gains)
        base_equity = current_equity - old_unrealized
        new_equity = base_equity + total_unrealized_pnl
        log.info(
            "Equity recalculated: $%.2f (base=$%.2f, unrealized=$%.2f)",
            new_equity, base_equity, total_unrealized_pnl,
        )

        self.update_equity(new_equity)
        self._check_position_weight_drift(positions, new_equity)
        self.persist()
        event_bus.emit("portfolio", "updated", self.snapshot())
        return new_equity

    def _check_position_weight_drift(
        self, positions: list[dict[str, Any]], equity: float
    ) -> None:
        """Log warning and emit event if any position drifts above max_position_pct."""
        if equity <= 0:
            return

        try:
            config_path = os.path.join(CONFIG_DIR, "risk_params.json")
            with open(config_path) as f:
                risk_params = json.load(f)
            max_position_pct = risk_params.get("max_position_pct", 7.0)
        except Exception:
            max_position_pct = 7.0

        for pos in positions:
            current_price = pos.get("current_price", 0.0)
            quantity = pos.get("quantity", 0.0)
            asset = pos.get("asset", "unknown")
            if current_price <= 0 or quantity <= 0:
                continue
            weight_pct = (current_price * quantity) / equity * 100
            if weight_pct > max_position_pct:
                log.warning(
                    "POSITION WEIGHT DRIFT: %s is %.2f%% of equity (max %.1f%%)",
                    asset, weight_pct, max_position_pct,
                )
                event_bus.emit("risk", "position_weight_drift", {
                    "asset": asset,
                    "weight_pct": round(weight_pct, 2),
                    "max_position_pct": max_position_pct,
                    "current_price": current_price,
                    "quantity": quantity,
                    "equity": equity,
                })

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
        event_bus.emit("portfolio", "updated", self.snapshot())

    def add_position(self, position: dict[str, Any]) -> None:
        """Add an open position."""
        with self._lock:
            self.open_positions.append(position)
            self.last_updated = datetime.utcnow().isoformat()
        event_bus.emit("portfolio", "position_added", position)
        event_bus.emit("portfolio", "updated", self.snapshot())

    def remove_position(self, trade_id: str) -> dict[str, Any] | None:
        """Remove and return a position by trade_id."""
        with self._lock:
            for i, pos in enumerate(self.open_positions):
                if pos.get("trade_id") == trade_id:
                    removed = self.open_positions.pop(i)
                    self.last_updated = datetime.utcnow().isoformat()
                    break
            else:
                return None
        event_bus.emit("portfolio", "position_removed", {"trade_id": trade_id})
        event_bus.emit("portfolio", "updated", self.snapshot())
        return removed

    def reset_daily(self) -> None:
        """Reset daily P&L counters (called at start of each trading day)."""
        with self._lock:
            self.daily_pnl = 0.0
            self.daily_pnl_pct = 0.0
            self.last_updated = datetime.utcnow().isoformat()

    # ── Persistence ────────────────────────────────────────────────────

    def persist(self) -> None:
        """Save state to JSON file and append equity history snapshot."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        state = self.snapshot()
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
        self._record_equity_snapshot()
        log.info("Portfolio state persisted to %s", self.state_file)

    def _record_equity_snapshot(self) -> None:
        """Append current equity to history file (capped ring buffer)."""
        entry = {
            "t": datetime.utcnow().isoformat(),
            "equity": self.equity,
            "pnl_pct": self.daily_pnl_pct,
        }
        try:
            history: list[dict[str, Any]] = []
            if os.path.exists(EQUITY_HISTORY_FILE):
                with open(EQUITY_HISTORY_FILE) as f:
                    history = json.load(f)
            history.append(entry)
            if len(history) > 2000:
                history = history[-2000:]
            with open(EQUITY_HISTORY_FILE, "w") as f:
                json.dump(history, f)
        except Exception:
            pass

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
