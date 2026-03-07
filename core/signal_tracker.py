"""Signal Accuracy Tracker — tracks whether NewsScout signals led to profitable trades.

Records every signal from creation through pipeline outcome to trade close,
computing signal_correct based on sentiment vs actual P&L.
Persists to data/signal_accuracy.json.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from core.event_bus import event_bus
from core.logger import setup_logger

log = setup_logger("trading.signal_tracker")

DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "signal_accuracy.json"
)

MAX_SIGNALS = 500


class SignalAccuracyTracker:
    """Tracks signal-to-trade accuracy with JSON persistence."""

    def __init__(self, data_file: str = DATA_FILE) -> None:
        self._data_file = data_file
        self._lock = threading.Lock()
        self._signals: list[dict[str, Any]] = []
        # Index: trade_id -> signal index for fast lookup on close
        self._trade_index: dict[str, int] = {}
        self._load()
        self._rebuild_trade_index()

        # Subscribe to trade close events
        event_bus.add_listener(self._on_event)

    def _on_event(self, event: dict[str, Any]) -> None:
        """Handle event bus events for trade closes."""
        category = event.get("category", "")
        event_type = event.get("event_type", "")
        data = event.get("data", {})

        if category == "stop_loss" and event_type == "triggered":
            self._handle_trade_close(data, "stop_loss")
        elif category == "take_profit" and event_type == "triggered":
            self._handle_trade_close(data, "take_profit")
        elif category == "pipeline" and event_type == "time_exit":
            self._handle_trade_close(data, "time_exit")

    def _handle_trade_close(self, data: dict[str, Any], exit_reason: str) -> None:
        """Process a trade close event from the event bus."""
        trade_id = str(data.get("trade_id", ""))
        if not trade_id:
            return

        entry_price = data.get("entry_price", 0.0)
        exit_price = data.get("exit_price", 0.0)
        pnl = data.get("pnl", 0.0)

        # Calculate pnl_pct
        direction = data.get("direction", "long")
        pnl_pct = 0.0
        if entry_price and entry_price > 0:
            if direction == "long":
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        # Calculate hold duration if timestamps available
        hold_hours = data.get("holding_hours")

        self.record_trade_close(
            trade_id=trade_id,
            exit_price=exit_price,
            pnl_usd=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            hold_duration_hours=hold_hours,
        )

    def record_signal(
        self, signal_id: str, signal: Any, source_type: str
    ) -> None:
        """Record a new signal creation."""
        record: dict[str, Any] = {
            "signal_id": signal_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset": getattr(signal, "asset", ""),
            "headline": getattr(signal, "headline", ""),
            "signal_strength": getattr(signal, "signal_strength", 0.0),
            "sentiment": getattr(signal, "sentiment", "neutral"),
            "category": getattr(signal, "category", ""),
            "urgency": getattr(signal, "urgency", ""),
            "source_type": source_type,
            # Outcome fields (filled later)
            "pipeline_outcome": None,
            "trade_id": None,
            "direction": None,
            "entry_price": None,
            "killed_by": None,
            "kill_reason": None,
            # Trade close fields (filled on exit)
            "exit_price": None,
            "pnl_usd": None,
            "pnl_pct": None,
            "exit_reason": None,
            "hold_duration_hours": None,
            "signal_correct": None,
        }

        # Convert enums to strings
        for field in ("sentiment", "category", "urgency"):
            val = record[field]
            if hasattr(val, "value"):
                record[field] = val.value

        with self._lock:
            self._signals.append(record)
            if len(self._signals) > MAX_SIGNALS:
                self._signals = self._signals[-MAX_SIGNALS:]
                self._rebuild_trade_index()

        self._persist()
        log.info(
            "Signal recorded: %s %s (%s) strength=%.2f",
            signal_id, record["asset"], source_type, record["signal_strength"],
        )

    def record_outcome(
        self,
        signal_id: str,
        pipeline_outcome: str,
        trade_id: str | None = None,
        direction: str | None = None,
        entry_price: float | None = None,
        killed_by: str | None = None,
        kill_reason: str | None = None,
    ) -> None:
        """Record what happened to a signal in the pipeline."""
        with self._lock:
            for i, sig in enumerate(self._signals):
                if sig["signal_id"] == signal_id:
                    sig["pipeline_outcome"] = pipeline_outcome
                    sig["trade_id"] = trade_id
                    sig["direction"] = direction
                    sig["entry_price"] = entry_price
                    sig["killed_by"] = killed_by
                    sig["kill_reason"] = kill_reason
                    if trade_id:
                        self._trade_index[str(trade_id)] = i
                    break

        self._persist()
        log.info("Signal outcome: %s -> %s", signal_id, pipeline_outcome)

    def record_trade_close(
        self,
        trade_id: str,
        exit_price: float,
        pnl_usd: float,
        pnl_pct: float,
        exit_reason: str,
        hold_duration_hours: float | None = None,
    ) -> None:
        """Match trade_id back to signal and fill exit fields."""
        trade_id = str(trade_id)

        with self._lock:
            idx = self._trade_index.get(trade_id)
            if idx is None:
                # Search linearly as fallback
                for i, sig in enumerate(self._signals):
                    if str(sig.get("trade_id", "")) == trade_id:
                        idx = i
                        break

            if idx is None or idx >= len(self._signals):
                log.debug("No signal found for trade_id=%s", trade_id)
                return

            sig = self._signals[idx]
            sig["exit_price"] = exit_price
            sig["pnl_usd"] = round(pnl_usd, 2)
            sig["pnl_pct"] = round(pnl_pct, 2)
            sig["exit_reason"] = exit_reason
            sig["hold_duration_hours"] = hold_duration_hours

            # Compute signal_correct
            sentiment = sig.get("sentiment", "neutral")
            if sentiment in ("bullish",) and pnl_usd > 0:
                sig["signal_correct"] = True
            elif sentiment in ("bearish",) and pnl_usd > 0:
                sig["signal_correct"] = True
            elif sentiment in ("bullish",) and pnl_usd <= 0:
                sig["signal_correct"] = False
            elif sentiment in ("bearish",) and pnl_usd <= 0:
                sig["signal_correct"] = False
            else:
                # neutral/uncertain sentiment — correct if profitable
                sig["signal_correct"] = pnl_usd > 0

        self._persist()
        log.info(
            "Trade close recorded: trade_id=%s pnl=$%.2f correct=%s",
            trade_id, pnl_usd, sig.get("signal_correct"),
        )

    def summary(self) -> dict[str, Any]:
        """Return aggregate stats for the dashboard."""
        with self._lock:
            total = len(self._signals)
            executed = [s for s in self._signals if s.get("pipeline_outcome") == "executed"]
            closed = [s for s in self._signals if s.get("exit_price") is not None]
            wins = [s for s in closed if s.get("signal_correct") is True]
            losses = [s for s in closed if s.get("signal_correct") is False]

            # By category
            by_category: dict[str, dict[str, int]] = {}
            for s in self._signals:
                cat = s.get("category", "unknown")
                if cat not in by_category:
                    by_category[cat] = {"total": 0, "executed": 0, "wins": 0, "losses": 0}
                by_category[cat]["total"] += 1
                if s.get("pipeline_outcome") == "executed":
                    by_category[cat]["executed"] += 1
                if s.get("signal_correct") is True:
                    by_category[cat]["wins"] += 1
                elif s.get("signal_correct") is False:
                    by_category[cat]["losses"] += 1

            # By asset
            by_asset: dict[str, dict[str, int]] = {}
            for s in self._signals:
                asset = s.get("asset", "unknown")
                if asset not in by_asset:
                    by_asset[asset] = {"total": 0, "executed": 0, "wins": 0, "losses": 0}
                by_asset[asset]["total"] += 1
                if s.get("pipeline_outcome") == "executed":
                    by_asset[asset]["executed"] += 1
                if s.get("signal_correct") is True:
                    by_asset[asset]["wins"] += 1
                elif s.get("signal_correct") is False:
                    by_asset[asset]["losses"] += 1

            # By strength bucket
            by_strength: dict[str, dict[str, int]] = {}
            for s in self._signals:
                strength = s.get("signal_strength", 0)
                if strength >= 0.8:
                    bucket = "high_0.8+"
                elif strength >= 0.5:
                    bucket = "medium_0.5-0.8"
                else:
                    bucket = "low_<0.5"
                if bucket not in by_strength:
                    by_strength[bucket] = {"total": 0, "executed": 0, "wins": 0, "losses": 0}
                by_strength[bucket]["total"] += 1
                if s.get("pipeline_outcome") == "executed":
                    by_strength[bucket]["executed"] += 1
                if s.get("signal_correct") is True:
                    by_strength[bucket]["wins"] += 1
                elif s.get("signal_correct") is False:
                    by_strength[bucket]["losses"] += 1

            # Outcome breakdown
            outcomes: dict[str, int] = {}
            for s in self._signals:
                outcome = s.get("pipeline_outcome", "pending")
                outcomes[outcome] = outcomes.get(outcome, 0) + 1

            win_count = len(wins)
            loss_count = len(losses)
            closed_count = len(closed)

            return {
                "total_signals": total,
                "executed": len(executed),
                "closed": closed_count,
                "wins": win_count,
                "losses": loss_count,
                "win_rate": round(win_count / closed_count * 100, 1) if closed_count > 0 else 0.0,
                "outcomes": outcomes,
                "by_category": by_category,
                "by_asset": by_asset,
                "by_strength_bucket": by_strength,
                "recent": self._signals[-10:],
            }

    def _rebuild_trade_index(self) -> None:
        """Rebuild trade_id -> index mapping."""
        self._trade_index = {}
        for i, sig in enumerate(self._signals):
            tid = sig.get("trade_id")
            if tid:
                self._trade_index[str(tid)] = i

    def _persist(self) -> None:
        """Save signals to disk."""
        try:
            os.makedirs(os.path.dirname(self._data_file), exist_ok=True)
            with self._lock:
                data = list(self._signals)
            with open(self._data_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # non-critical

    def _load(self) -> None:
        """Restore signals from disk on startup."""
        if not os.path.exists(self._data_file):
            return
        try:
            with open(self._data_file) as f:
                self._signals = json.load(f)
            # Cap on load
            if len(self._signals) > MAX_SIGNALS:
                self._signals = self._signals[-MAX_SIGNALS:]
        except Exception:
            self._signals = []
