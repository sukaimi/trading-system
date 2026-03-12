"""Earnings calendar — pure Python, no LLM calls.

Tracks upcoming earnings dates for stock assets to prevent entering
positions right before high-binary-risk events. ETFs, crypto, and
commodities have no earnings and are always safe.

Data sources (in priority order):
1. Alpha Vantage EARNINGS API (cached 7 days) — covers open universe
2. Hardcoded EARNINGS_DATES dict — fallback for core stocks
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.earnings")

# Assets that never have earnings (ETFs, crypto, commodities)
_NO_EARNINGS_ASSETS = frozenset({
    "SPY", "GLDM", "SLV", "EWS", "FXI", "QQQ", "GLD", "TLT", "XLE",
    "BTC", "ETH",
})


class EarningsCalendar:
    """Track upcoming earnings dates for traded stock assets.

    Uses Alpha Vantage EARNINGS API when available, with hardcoded
    dates as fallback. ETFs/crypto/commodities always return None.
    """

    # Known quarterly earnings dates (approximate, updated manually)
    # Used as fallback when Alpha Vantage is unavailable
    EARNINGS_DATES: dict[str, list[str]] = {
        "AAPL": ["2026-01-29", "2026-04-30", "2026-07-30", "2026-10-29"],
        "NVDA": ["2026-02-26", "2026-05-28", "2026-08-27", "2026-11-19"],
        "TSLA": ["2026-01-29", "2026-04-22", "2026-07-22", "2026-10-21"],
        "AMZN": ["2026-02-06", "2026-04-30", "2026-07-30", "2026-10-29"],
        "META": ["2026-01-29", "2026-04-23", "2026-07-23", "2026-10-28"],
        # SPY, GLDM, SLV, BTC, ETH, EWS, FXI, QQQ, GLD — no earnings
    }

    def __init__(self, av_client: Any | None = None):
        """Initialize with optional Alpha Vantage client.

        Args:
            av_client: AlphaVantageClient instance. If None, attempts
                       to use the module-level singleton.
        """
        self._av_client = av_client
        # Cache of API-fetched earnings dates: ticker -> list of date strings
        self._api_dates_cache: dict[str, list[str]] = {}

    def _get_av_client(self) -> Any | None:
        """Lazy-load the Alpha Vantage client."""
        if self._av_client is not None:
            return self._av_client
        try:
            from tools.alpha_vantage import get_av_client
            self._av_client = get_av_client()
            return self._av_client
        except Exception:
            return None

    def _get_earnings_dates(self, asset: str) -> list[str] | None:
        """Get earnings dates for an asset, trying API first then fallback.

        Returns:
            List of date strings, or None if asset has no earnings.
        """
        if asset in _NO_EARNINGS_ASSETS:
            return None

        # Try API-cached dates first
        if asset in self._api_dates_cache:
            return self._api_dates_cache[asset]

        # Try Alpha Vantage API
        av = self._get_av_client()
        if av is not None:
            try:
                data = av.earnings(asset)
                if data and "quarterlyEarnings" in data:
                    dates = []
                    for q in data["quarterlyEarnings"]:
                        rd = q.get("reportedDate", "")
                        if rd:
                            dates.append(rd)
                    if dates:
                        self._api_dates_cache[asset] = sorted(dates)
                        return self._api_dates_cache[asset]
            except Exception as e:
                log.debug("AV earnings fetch failed for %s: %s", asset, e)

        # Fallback to hardcoded dates
        return self.EARNINGS_DATES.get(asset)

    def get_eps_history(self, ticker: str, quarters: int = 4) -> list[dict[str, Any]]:
        """Get recent EPS surprise history for a ticker.

        Args:
            ticker: Stock ticker symbol.
            quarters: Number of recent quarters (default 4).

        Returns:
            List of dicts with date, reported_eps, estimated_eps, surprise_pct.
            Empty list if unavailable.
        """
        av = self._get_av_client()
        if av is None:
            return []
        try:
            return av.get_eps_history(ticker, quarters=quarters)
        except Exception:
            return []

    def days_until_earnings(self, asset: str, ref_date: date | None = None) -> int | None:
        """Return days until next earnings, or None if no earnings (ETF/crypto).

        Args:
            asset: Ticker symbol.
            ref_date: Reference date (defaults to today). Useful for testing.

        Returns:
            Number of calendar days until next earnings, or None if asset
            has no scheduled earnings.
        """
        dates = self._get_earnings_dates(asset)
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

        # Merge hardcoded + API-cached assets
        all_assets = set(self.EARNINGS_DATES.keys()) | set(self._api_dates_cache.keys())
        for asset in all_assets:
            date_strs = self._get_earnings_dates(asset) or []
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
