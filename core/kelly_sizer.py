"""Kelly Criterion Position Sizing — pure Python, Tier 0 (no LLM calls).

Computes fractional Kelly position sizes from historical trade journal data
with a fallback hierarchy: asset -> sector -> global -> ATR sizing.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.kelly")


class KellySizer:
    """Fractional Kelly position sizer using historical trade outcomes."""

    def __init__(self, journal_file: str, config: dict[str, Any] | None = None):
        self._journal_file = journal_file
        config = config or {}
        self._kelly_alpha: float = config.get("kelly_alpha", 0.35)
        self._min_kelly_trades: int = config.get("min_kelly_trades", 20)
        self._max_kelly_pct: float = config.get("max_kelly_pct", 5.0)
        self._kelly_enabled: bool = config.get("kelly_enabled", True)

        # TTL cache for journal reads (60 second TTL)
        self._cache_ts: float = 0.0
        self._cached_trades: list[dict[str, Any]] = []
        self._cache_ttl: float = 60.0

    # ── Sector mapping (matches risk_manager.py) ──────────────────────────

    _CORE_SECTORS: dict[str, str] = {
        "BTC": "crypto", "ETH": "crypto",
        "AAPL": "tech", "NVDA": "tech", "TSLA": "tech", "AMZN": "tech", "META": "tech",
        "TLT": "bonds",
        "GLDM": "commodities", "SLV": "commodities",
        "XLE": "energy",
        "EWS": "regional", "FXI": "regional",
        "SPY": "index",
    }

    def _get_sector(self, asset: str) -> str:
        """Get sector for an asset. Hardcoded for core 14, 'other' for unknown."""
        return self._CORE_SECTORS.get(asset, "other")

    # ── Journal loading with TTL cache ────────────────────────────────────

    def _load_closed_trades(self) -> list[dict[str, Any]]:
        """Load closed trades from trade journal JSON with TTL cache."""
        now = time.time()
        if now - self._cache_ts < self._cache_ttl and self._cached_trades is not None:
            return self._cached_trades

        self._cache_ts = now
        self._cached_trades = []

        if not os.path.exists(self._journal_file):
            return self._cached_trades

        try:
            with open(self._journal_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, ValueError):
            return self._cached_trades

        if not isinstance(data, list):
            return self._cached_trades

        closed: list[dict[str, Any]] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            outcome = entry.get("outcome")
            if not isinstance(outcome, dict):
                continue
            pnl = outcome.get("pnl_usd")
            if pnl is not None:
                closed.append({
                    "asset": entry.get("asset", ""),
                    "pnl_usd": float(pnl),
                })

        self._cached_trades = closed
        return self._cached_trades

    # ── Stats computation ─────────────────────────────────────────────────

    def _compute_stats(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute win rate, avg win/loss, payoff ratio from trades."""
        if not trades:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "payoff_ratio": 0.0,
                "sample_size": 0,
            }

        wins = [t["pnl_usd"] for t in trades if t["pnl_usd"] > 0]
        losses = [abs(t["pnl_usd"]) for t in trades if t["pnl_usd"] < 0]
        # Skip pnl_usd == 0 (breakeven)
        total = len(wins) + len(losses)

        if total == 0:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "payoff_ratio": 0.0,
                "sample_size": 0,
            }

        win_rate = len(wins) / total
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        # Payoff ratio = avg_win / avg_loss
        if not wins:
            payoff_ratio = 0.0
        elif not losses:
            # No losses: cap payoff ratio at 10.0
            payoff_ratio = min(avg_win / 1.0, 10.0)
        else:
            payoff_ratio = avg_win / avg_loss

        return {
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "payoff_ratio": round(payoff_ratio, 4),
            "sample_size": total,
        }

    # ── Stats accessors ───────────────────────────────────────────────────

    def get_stats_for_asset(self, asset: str) -> dict[str, Any]:
        """Get trade stats filtered by specific asset."""
        trades = self._load_closed_trades()
        filtered = [t for t in trades if t["asset"] == asset]
        return self._compute_stats(filtered)

    def get_stats_for_sector(self, sector: str) -> dict[str, Any]:
        """Get trade stats filtered by sector."""
        trades = self._load_closed_trades()
        filtered = [t for t in trades if self._get_sector(t["asset"]) == sector]
        return self._compute_stats(filtered)

    def get_stats_global(self) -> dict[str, Any]:
        """Get trade stats across all closed trades."""
        trades = self._load_closed_trades()
        return self._compute_stats(trades)

    # ── Kelly fraction computation ────────────────────────────────────────

    def kelly_fraction(self, asset: str) -> dict[str, Any]:
        """Compute fractional Kelly position size with fallback hierarchy.

        Returns dict with: fraction, source, stats, using_fallback.
        Hierarchy: asset (n >= min) -> sector (n >= min) -> global (n >= min) -> insufficient.
        """
        if not self._kelly_enabled:
            return {
                "fraction": 0.0,
                "source": "disabled",
                "stats": {},
                "using_fallback": True,
            }

        min_n = self._min_kelly_trades

        # Try asset-level
        asset_stats = self.get_stats_for_asset(asset)
        if asset_stats["sample_size"] >= min_n:
            return self._apply_kelly(asset_stats, source="asset", using_fallback=False)

        # Try sector-level
        sector = self._get_sector(asset)
        sector_stats = self.get_stats_for_sector(sector)
        if sector_stats["sample_size"] >= min_n:
            return self._apply_kelly(sector_stats, source="sector", using_fallback=True)

        # Try global
        global_stats = self.get_stats_global()
        if global_stats["sample_size"] >= min_n:
            return self._apply_kelly(global_stats, source="global", using_fallback=True)

        # Insufficient data at all levels
        return {
            "fraction": 0.0,
            "source": "insufficient_data",
            "stats": global_stats,
            "using_fallback": True,
        }

    def _apply_kelly(
        self, stats: dict[str, Any], source: str, using_fallback: bool
    ) -> dict[str, Any]:
        """Apply Kelly formula: f* = (p * b - q) / b, then scale by alpha."""
        p = stats["win_rate"]
        b = stats["payoff_ratio"]
        q = 1.0 - p

        if b <= 0:
            return {
                "fraction": 0.0,
                "source": source,
                "stats": stats,
                "using_fallback": using_fallback,
            }

        f_star = (p * b - q) / b

        if f_star <= 0:
            # Negative edge — no position
            return {
                "fraction": 0.0,
                "source": source,
                "stats": stats,
                "using_fallback": using_fallback,
            }

        # Apply fractional Kelly (alpha scaling)
        fraction = self._kelly_alpha * f_star

        # Cap at max_kelly_pct / 100
        max_frac = self._max_kelly_pct / 100.0
        fraction = min(fraction, max_frac)

        return {
            "fraction": round(fraction, 4),
            "source": source,
            "stats": stats,
            "using_fallback": using_fallback,
        }

    # ── Dashboard summary ─────────────────────────────────────────────────

    def get_all_stats(self) -> dict[str, Any]:
        """Full stats for dashboard: global + per-asset + per-sector + config."""
        trades = self._load_closed_trades()

        # Per-asset stats
        assets_seen: set[str] = {t["asset"] for t in trades if t["asset"]}
        per_asset: dict[str, Any] = {}
        for asset in sorted(assets_seen):
            filtered = [t for t in trades if t["asset"] == asset]
            per_asset[asset] = self._compute_stats(filtered)

        # Per-sector stats
        sectors_seen: set[str] = {self._get_sector(t["asset"]) for t in trades if t["asset"]}
        per_sector: dict[str, Any] = {}
        for sector in sorted(sectors_seen):
            filtered = [t for t in trades if self._get_sector(t["asset"]) == sector]
            per_sector[sector] = self._compute_stats(filtered)

        return {
            "global": self._compute_stats(trades),
            "per_asset": per_asset,
            "per_sector": per_sector,
            "config": {
                "kelly_alpha": self._kelly_alpha,
                "min_kelly_trades": self._min_kelly_trades,
                "max_kelly_pct": self._max_kelly_pct,
                "kelly_enabled": self._kelly_enabled,
            },
        }
