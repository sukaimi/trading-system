"""Phantom Portfolio — tracks trades killed/rejected to measure opportunity cost.

Logs every signal that made it past NewsScout but was ultimately not executed.
Later, we check what would have happened to identify if our gates are too strict.
Persists to data/phantom_trades.json.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.phantom_tracker")

DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "phantom_trades.json"
)


class PhantomTracker:
    """Records missed trades and tracks what would have happened."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trades: list[dict[str, Any]] = []
        self._load()

    def record_missed(
        self,
        asset: str,
        direction: str,
        confidence: float,
        killed_by: str,
        reason: str,
        entry_price: float = 0.0,
        suggested_position_pct: float = 0.0,
        thesis: str = "",
    ) -> None:
        """Record a trade that was killed or rejected."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset": asset,
            "direction": direction,
            "confidence": confidence,
            "entry_price": entry_price,
            "suggested_position_pct": suggested_position_pct,
            "killed_by": killed_by,
            "reason": reason,
            "thesis": thesis[:200],
            "outcome_checked": False,
            "outcome_price": None,
            "outcome_pnl_pct": None,
        }

        with self._lock:
            self._trades.append(record)
            # Keep last 200 phantom trades
            if len(self._trades) > 200:
                self._trades = self._trades[-200:]

        self._persist()
        log.info(
            "Phantom trade recorded: %s %s (killed by %s, conf %.2f)",
            direction, asset, killed_by, confidence,
        )

    def get_unchecked(self) -> list[dict[str, Any]]:
        """Return phantom trades that haven't been checked against outcomes."""
        with self._lock:
            return [t for t in self._trades if not t.get("outcome_checked")]

    def update_outcome(
        self, index: int, current_price: float
    ) -> dict[str, Any] | None:
        """Update a phantom trade with what actually happened."""
        with self._lock:
            if index >= len(self._trades):
                return None
            trade = self._trades[index]
            if trade["outcome_checked"] or trade["entry_price"] <= 0:
                return None

            entry = trade["entry_price"]
            direction = trade["direction"]

            if direction == "long":
                pnl_pct = ((current_price - entry) / entry) * 100
            else:
                pnl_pct = ((entry - current_price) / entry) * 100

            trade["outcome_checked"] = True
            trade["outcome_price"] = current_price
            trade["outcome_pnl_pct"] = round(pnl_pct, 2)

        self._persist()
        return trade

    def summary(self) -> dict[str, Any]:
        """Return summary stats for the dashboard."""
        with self._lock:
            checked = [t for t in self._trades if t.get("outcome_checked")]
            unchecked = [t for t in self._trades if not t.get("outcome_checked")]

            would_have_won = [t for t in checked if (t.get("outcome_pnl_pct") or 0) > 0]
            would_have_lost = [t for t in checked if (t.get("outcome_pnl_pct") or 0) <= 0]

            # Group by killer
            by_killer: dict[str, int] = {}
            for t in self._trades:
                killer = t.get("killed_by", "unknown")
                by_killer[killer] = by_killer.get(killer, 0) + 1

            return {
                "total_missed": len(self._trades),
                "checked": len(checked),
                "unchecked": len(unchecked),
                "would_have_won": len(would_have_won),
                "would_have_lost": len(would_have_lost),
                "missed_profit_pct": round(
                    sum(t.get("outcome_pnl_pct", 0) for t in would_have_won), 2
                ),
                "avoided_loss_pct": round(
                    sum(abs(t.get("outcome_pnl_pct", 0)) for t in would_have_lost), 2
                ),
                "by_killer": by_killer,
                "recent": self._trades[-10:],
            }

    def _persist(self) -> None:
        try:
            os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
            with self._lock:
                data = list(self._trades)
            with open(DATA_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load(self) -> None:
        if not os.path.exists(DATA_FILE):
            return
        try:
            with open(DATA_FILE) as f:
                self._trades = json.load(f)
        except Exception:
            self._trades = []
