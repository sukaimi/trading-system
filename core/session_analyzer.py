"""Session Analyzer — pure Python, no LLM calls (Tier 0).

Tracks which trading session (Asian, European, US) produces the best
results per asset. Uses closed trade data from trade_journal.json.

Session boundaries (UTC):
  Asian:    22:00 - 06:00
  European: 06:00 - 12:00
  US:       12:00 - 22:00
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import datetime
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.session_analyzer")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
JOURNAL_FILE = os.path.join(DATA_DIR, "trade_journal.json")

SESSIONS = {
    "asian": (22, 6),     # 22:00-06:00 UTC (wraps midnight)
    "european": (6, 12),  # 06:00-12:00 UTC
    "us": (12, 22),       # 12:00-22:00 UTC
}


class SessionAnalyzer:
    """Analyze trade performance by trading session."""

    def __init__(self, min_trades_per_session: int = 5):
        self._min_trades = min_trades_per_session

    def classify_session(self, utc_hour: int) -> str:
        """Classify a UTC hour (0-23) into a session name.

        Returns 'asian', 'european', or 'us'.
        """
        utc_hour = utc_hour % 24
        # Asian wraps midnight: 22:00-06:00
        if utc_hour >= 22 or utc_hour < 6:
            return "asian"
        elif utc_hour < 12:
            return "european"
        else:
            return "us"

    def _load_journal(self) -> list[dict[str, Any]]:
        """Load closed trades from trade journal."""
        if not os.path.exists(JOURNAL_FILE):
            return []
        try:
            with open(JOURNAL_FILE) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load trade journal: %s", e)
        return []

    def _is_closed_trade(self, trade: dict[str, Any]) -> bool:
        """Check if a trade entry has the fields needed for session analysis."""
        return (
            trade.get("pnl_pct") is not None
            and trade.get("timestamp_open") is not None
            and trade.get("asset") is not None
        )

    def _extract_utc_hour(self, timestamp: str) -> int | None:
        """Parse a timestamp string and return the UTC hour."""
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(timestamp, fmt)
                return dt.hour
            except ValueError:
                continue
        # Try ISO format as last resort
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.hour
        except (ValueError, AttributeError):
            return None

    def analyze(self) -> dict[str, Any]:
        """Analyze trade performance by session.

        Returns per-asset and overall session statistics with scores.
        """
        journal = self._load_journal()
        closed_trades = [t for t in journal if self._is_closed_trade(t)]

        result: dict[str, Any] = {
            "sufficient_data": False,
            "per_asset": {},
            "overall": {},
        }

        if not closed_trades:
            return result

        # Classify each trade into a session
        classified: list[tuple[dict[str, Any], str]] = []
        for trade in closed_trades:
            hour = self._extract_utc_hour(trade["timestamp_open"])
            if hour is not None:
                session = self.classify_session(hour)
                classified.append((trade, session))

        if not classified:
            return result

        # Group by asset and session
        by_asset: dict[str, dict[str, list[dict[str, Any]]]] = {}
        overall_by_session: dict[str, list[dict[str, Any]]] = {
            "asian": [], "european": [], "us": [],
        }

        for trade, session in classified:
            asset = trade["asset"]
            by_asset.setdefault(asset, {"asian": [], "european": [], "us": []})
            by_asset[asset][session].append(trade)
            overall_by_session[session].append(trade)

        # Check if any session has enough data
        has_sufficient = False
        for asset, sessions in by_asset.items():
            for session_name, trades in sessions.items():
                if len(trades) >= self._min_trades:
                    has_sufficient = True
                    break

        result["sufficient_data"] = has_sufficient

        # Per-asset analysis
        for asset, sessions in by_asset.items():
            asset_result: dict[str, Any] = {}
            best_score = -1.0
            best_session = None

            for session_name in ("asian", "european", "us"):
                trades = sessions[session_name]
                stats = self._compute_session_stats(trades)
                asset_result[session_name] = stats
                if stats["score"] > best_score:
                    best_score = stats["score"]
                    best_session = session_name

            asset_result["best_session"] = best_session
            result["per_asset"][asset] = asset_result

        # Overall analysis
        for session_name in ("asian", "european", "us"):
            trades = overall_by_session[session_name]
            result["overall"][session_name] = self._compute_session_stats(trades)

        return result

    def _compute_session_stats(
        self, trades: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute win rate, avg PnL, and score for a set of trades."""
        count = len(trades)
        if count == 0:
            return {
                "trades": 0,
                "win_rate": 0.0,
                "avg_pnl_pct": 0.0,
                "score": 0.0,
            }

        pnls = [t["pnl_pct"] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / count
        avg_pnl = statistics.mean(pnls)

        # Score: weighted combination of win rate and avg PnL
        # Win rate contributes 60%, normalized PnL contributes 40%
        # PnL is clamped to [-10, 10] range for normalization
        clamped_pnl = max(-10.0, min(10.0, avg_pnl))
        pnl_score = (clamped_pnl + 10.0) / 20.0  # Normalize to [0, 1]
        score = round(0.6 * win_rate + 0.4 * pnl_score, 2)

        return {
            "trades": count,
            "win_rate": round(win_rate, 2),
            "avg_pnl_pct": round(avg_pnl, 2),
            "score": score,
        }

    def get_session_weight(self, asset: str, session: str) -> float:
        """Return confidence weight (0.5-1.5) for asset in given session.

        Returns 1.0 if insufficient data for this asset+session combination.
        """
        analysis = self.analyze()

        if not analysis["sufficient_data"]:
            return 1.0

        asset_data = analysis["per_asset"].get(asset)
        if not asset_data:
            return 1.0

        session_data = asset_data.get(session)
        if not session_data or session_data["trades"] < self._min_trades:
            return 1.0

        # Convert score (0-1) to weight (0.5-1.5)
        # score=0.5 maps to weight=1.0 (neutral)
        # score=0.0 maps to weight=0.5 (reduce position)
        # score=1.0 maps to weight=1.5 (increase position)
        score = session_data["score"]
        weight = 0.5 + score  # Maps [0,1] → [0.5, 1.5]
        return round(weight, 2)
