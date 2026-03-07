"""Earnings calendar — pure Python, no LLM calls.

Tracks upcoming earnings dates for stock assets to prevent entering
positions right before high-binary-risk events. ETFs, crypto, and
commodities have no earnings and are always safe.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.earnings")


class EarningsCalendar:
    """Track upcoming earnings dates for traded stock assets.

    Hardcoded calendar updated quarterly by SelfOptimizer or manually.
    ETFs (SPY, GLDM, SLV, EWS, FXI, QQQ, GLD) and crypto (BTC, ETH)
    have no earnings and return None for all queries.
    """

    # Known quarterly earnings dates (approximate, updated manually)
    EARNINGS_DATES: dict[str, list[str]] = {
        "AAPL": ["2026-01-29", "2026-04-30", "2026-07-30", "2026-10-29"],
        "NVDA": ["2026-02-26", "2026-05-28", "2026-08-27", "2026-11-19"],
        "TSLA": ["2026-01-29", "2026-04-22", "2026-07-22", "2026-10-21"],
        "AMZN": ["2026-02-06", "2026-04-30", "2026-07-30", "2026-10-29"],
        "META": ["2026-01-29", "2026-04-23", "2026-07-23", "2026-10-28"],
        # SPY, GLDM, SLV, BTC, ETH, EWS, FXI, QQQ, GLD — no earnings
    }

    def days_until_earnings(self, asset: str, ref_date: date | None = None) -> int | None:
        """Return days until next earnings, or None if no earnings (ETF/crypto).

        Args:
            asset: Ticker symbol.
            ref_date: Reference date (defaults to today). Useful for testing.

        Returns:
            Number of calendar days until next earnings, or None if asset
            has no scheduled earnings.
        """
        dates = self.EARNINGS_DATES.get(asset)
        if not dates:
            return None

        today = ref_date or date.today()
        future_dates = []
        for d_str in dates:
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d").date()
                if d >= today:
                    future_dates.append(d)
            except ValueError:
                continue

        if not future_dates:
            return None

        nearest = min(future_dates)
        return (nearest - today).days

    def has_earnings_soon(self, asset: str, days: int = 3, ref_date: date | None = None) -> bool:
        """True if asset has earnings within `days` calendar days.

        Args:
            asset: Ticker symbol.
            days: Window in calendar days (default 3).
            ref_date: Reference date (defaults to today).

        Returns:
            True if earnings are within the window, False otherwise.
            Always False for ETFs/crypto (no earnings).
        """
        until = self.days_until_earnings(asset, ref_date=ref_date)
        if until is None:
            return False
        return until <= days

    def upcoming_earnings(self, days: int = 14, ref_date: date | None = None) -> list[dict[str, Any]]:
        """Return all assets with earnings in next N days, sorted by date.

        Args:
            days: Look-ahead window in calendar days (default 14).
            ref_date: Reference date (defaults to today).

        Returns:
            List of dicts: [{"asset": str, "date": str, "days_until": int}, ...]
        """
        today = ref_date or date.today()
        results: list[dict[str, Any]] = []

        for asset, date_strs in self.EARNINGS_DATES.items():
            for d_str in date_strs:
                try:
                    d = datetime.strptime(d_str, "%Y-%m-%d").date()
                except ValueError:
                    continue
                delta = (d - today).days
                if 0 <= delta <= days:
                    results.append({
                        "asset": asset,
                        "date": d_str,
                        "days_until": delta,
                    })

        results.sort(key=lambda x: x["days_until"])
        return results
