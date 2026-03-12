"""Alpha Vantage API client — shared rate limiter, cache, and endpoints.

Centralizes all Alpha Vantage API calls with:
- Per-minute rate limiting (25 calls/min for free tier)
- Daily call budget (450/day, leaving 50-call buffer from 500 limit)
- In-memory cache with configurable TTL per endpoint
- Graceful fallback — all methods return None/empty on failure

Endpoints:
- news_sentiment(): NEWS_SENTIMENT (articles + per-ticker sentiment)
- ticker_sentiment(): aggregated sentiment score per ticker from NEWS_SENTIMENT
- earnings(): EARNINGS function (upcoming + historical EPS data)
- sector_performance(): SECTOR function (real-time sector rankings)
"""

from __future__ import annotations

import os
import time
from datetime import date
from typing import Any

import requests

from core.logger import setup_logger

log = setup_logger("trading.alpha_vantage")

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# Alpha Vantage ticker format mapping (crypto needs CRYPTO: prefix)
AV_TICKERS = {
    "BTC": "CRYPTO:BTC",
    "ETH": "CRYPTO:ETH",
    "GLDM": "GLDM",
    "SLV": "SLV",
    "AAPL": "AAPL",
    "NVDA": "NVDA",
    "TSLA": "TSLA",
    "AMZN": "AMZN",
    "SPY": "SPY",
    "META": "META",
    "EWS": "EWS",
    "FXI": "FXI",
    "QQQ": "QQQ",
    "GLD": "GLD",
}

# Cache TTLs in seconds
CACHE_TTL_NEWS = 15 * 60        # 15 minutes
CACHE_TTL_EARNINGS = 7 * 86400  # 7 days
CACHE_TTL_SECTOR = 4 * 3600     # 4 hours

# Rate limits
MAX_CALLS_PER_MINUTE = 25
MAX_CALLS_PER_DAY = 450  # Leave 50-call buffer from 500 free tier


class AlphaVantageClient:
    """Centralized Alpha Vantage API client with rate limiting and caching."""

    def __init__(self):
        self._key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self._cache: dict[str, tuple[float, Any]] = {}  # key -> (expiry_ts, data)
        self._daily_calls: int = 0
        self._daily_reset: date = date.today()
        self._call_timestamps: list[float] = []  # for per-minute rate limiting
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "TradingSystem/1.0 (+https://tradebot.codeandcraft.ai)"
        )

    @property
    def has_key(self) -> bool:
        """Check if API key is configured."""
        return bool(self._key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def news_sentiment(
        self, tickers: list[str] | None = None
    ) -> dict[str, Any] | None:
        """Fetch NEWS_SENTIMENT data including per-ticker sentiment breakdown.

        Args:
            tickers: List of ticker symbols (our format, e.g. "BTC", "AAPL").
                     Mapped to AV format internally.

        Returns:
            Raw AV response dict with 'feed' array, or None on failure.
            Each feed item includes 'ticker_sentiment' array with per-ticker scores.
        """
        if not self._key:
            return None

        if tickers is None:
            tickers = list(AV_TICKERS.keys())

        av_tickers = ",".join(AV_TICKERS.get(t, t) for t in tickers)
        cache_key = f"news:{av_tickers}"

        return self._call(
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": av_tickers,
            },
            cache_key=cache_key,
            cache_ttl=CACHE_TTL_NEWS,
        )

    def ticker_sentiment(self, ticker: str) -> dict[str, Any] | None:
        """Get aggregated sentiment score for a single ticker.

        Extracts per-ticker sentiment from the NEWS_SENTIMENT response
        and computes a relevance-weighted average.

        Args:
            ticker: Ticker symbol (our format, e.g. "AAPL").

        Returns:
            {
                "score": float (-1 to 1),
                "label": str (Bearish/Somewhat-Bearish/Neutral/Somewhat_Bullish/Bullish),
                "article_count": int,
                "avg_relevance": float (0 to 1),
            }
            or None if no data available.
        """
        av_ticker = AV_TICKERS.get(ticker, ticker)

        # Try to use cached news_sentiment data first
        news_data = self.news_sentiment([ticker])
        if not news_data or "feed" not in news_data:
            return None

        total_weight = 0.0
        weighted_score = 0.0
        article_count = 0
        relevance_sum = 0.0
        labels: list[str] = []

        for item in news_data.get("feed", []):
            for ts in item.get("ticker_sentiment", []):
                ts_ticker = ts.get("ticker", "")
                if ts_ticker != av_ticker:
                    continue

                try:
                    score = float(ts.get("ticker_sentiment_score", 0))
                    relevance = float(ts.get("relevance_score", 0))
                except (ValueError, TypeError):
                    continue

                if relevance <= 0:
                    continue

                weighted_score += score * relevance
                total_weight += relevance
                relevance_sum += relevance
                article_count += 1
                label = ts.get("ticker_sentiment_label", "Neutral")
                labels.append(label)

        if article_count == 0:
            return None

        avg_score = weighted_score / total_weight if total_weight > 0 else 0.0
        avg_relevance = relevance_sum / article_count

        # Determine aggregate label from score
        if avg_score <= -0.35:
            agg_label = "Bearish"
        elif avg_score <= -0.15:
            agg_label = "Somewhat-Bearish"
        elif avg_score < 0.15:
            agg_label = "Neutral"
        elif avg_score < 0.35:
            agg_label = "Somewhat_Bullish"
        else:
            agg_label = "Bullish"

        return {
            "score": round(avg_score, 4),
            "label": agg_label,
            "article_count": article_count,
            "avg_relevance": round(avg_relevance, 4),
        }

    def earnings(self, ticker: str) -> dict[str, Any] | None:
        """Fetch earnings data for a ticker.

        Returns:
            {
                "annual_earnings": [...],
                "quarterly_earnings": [
                    {
                        "fiscalDateEnding": "2025-12-31",
                        "reportedDate": "2026-01-29",
                        "reportedEPS": "2.40",
                        "estimatedEPS": "2.35",
                        "surprise": "0.05",
                        "surprisePercentage": "2.1277",
                    },
                    ...
                ]
            }
            or None on failure.
        """
        if not self._key:
            return None

        cache_key = f"earnings:{ticker}"
        return self._call(
            params={
                "function": "EARNINGS",
                "symbol": ticker,
            },
            cache_key=cache_key,
            cache_ttl=CACHE_TTL_EARNINGS,
        )

    def get_upcoming_earnings_date(self, ticker: str) -> str | None:
        """Get the next upcoming earnings report date for a ticker.

        Returns:
            Date string "YYYY-MM-DD" or None if unknown.
        """
        data = self.earnings(ticker)
        if not data:
            return None

        today_str = date.today().isoformat()
        quarterly = data.get("quarterlyEarnings", [])

        # quarterlyEarnings is sorted most-recent-first
        # Find the nearest future reportedDate
        future_dates: list[str] = []
        for q in quarterly:
            rd = q.get("reportedDate", "")
            if rd and rd >= today_str:
                future_dates.append(rd)

        if future_dates:
            return min(future_dates)

        # If no future dates in quarterly data, check if there's a pattern
        # (next quarter after the most recent report)
        return None

    def get_eps_history(self, ticker: str, quarters: int = 4) -> list[dict[str, Any]]:
        """Get recent EPS surprise history for a ticker.

        Args:
            ticker: Stock ticker symbol.
            quarters: Number of recent quarters to return (default 4).

        Returns:
            List of dicts with keys: date, reported_eps, estimated_eps, surprise_pct.
            Empty list on failure.
        """
        data = self.earnings(ticker)
        if not data:
            return []

        quarterly = data.get("quarterlyEarnings", [])
        today_str = date.today().isoformat()
        results: list[dict[str, Any]] = []

        for q in quarterly:
            rd = q.get("reportedDate", "")
            if not rd or rd > today_str:
                continue  # Skip future (unreported) entries

            try:
                reported = float(q.get("reportedEPS", 0))
                estimated = float(q.get("estimatedEPS", 0))
                surprise_pct = float(q.get("surprisePercentage", 0))
            except (ValueError, TypeError):
                continue

            results.append({
                "date": rd,
                "reported_eps": reported,
                "estimated_eps": estimated,
                "surprise_pct": surprise_pct,
            })

            if len(results) >= quarters:
                break

        return results

    def sector_performance(self) -> dict[str, Any] | None:
        """Fetch real-time sector performance rankings.

        Returns:
            {
                "Rank A: Real-Time Performance": {"Information Technology": "1.23%", ...},
                "Rank B: 1 Day Performance": {...},
                "Rank C: 5 Day Performance": {...},
                "Rank D: 1 Month Performance": {...},
                "Rank E: 3 Month Performance": {...},
                "Rank F: Year-to-Date (YTD) Performance": {...},
                "Rank G: 1 Year Performance": {...},
                ...
            }
            or None on failure.
        """
        if not self._key:
            return None

        return self._call(
            params={"function": "SECTOR"},
            cache_key="sector",
            cache_ttl=CACHE_TTL_SECTOR,
        )

    def get_sector_rankings(self, timeframe: str = "1mo") -> list[dict[str, Any]]:
        """Get sector rankings for a specific timeframe, sorted by performance.

        Args:
            timeframe: One of "realtime", "1d", "5d", "1mo", "3mo", "ytd", "1yr".

        Returns:
            Sorted list of {"sector": str, "performance_pct": float, "rank": int}.
            Empty list on failure.
        """
        data = self.sector_performance()
        if not data:
            return []

        # Map timeframe to AV rank key
        timeframe_map = {
            "realtime": "Rank A: Real-Time Performance",
            "1d": "Rank B: 1 Day Performance",
            "5d": "Rank C: 5 Day Performance",
            "1mo": "Rank D: 1 Month Performance",
            "3mo": "Rank E: 3 Month Performance",
            "ytd": "Rank F: Year-to-Date (YTD) Performance",
            "1yr": "Rank G: 1 Year Performance",
        }

        rank_key = timeframe_map.get(timeframe, timeframe_map["1mo"])
        sector_data = data.get(rank_key, {})

        if not sector_data or not isinstance(sector_data, dict):
            return []

        rankings: list[dict[str, Any]] = []
        for sector, pct_str in sector_data.items():
            try:
                pct = float(pct_str.replace("%", ""))
            except (ValueError, TypeError, AttributeError):
                continue
            rankings.append({"sector": sector, "performance_pct": pct})

        # Sort by performance descending
        rankings.sort(key=lambda x: x["performance_pct"], reverse=True)

        # Add rank numbers
        for i, r in enumerate(rankings, 1):
            r["rank"] = i

        return rankings

    # ------------------------------------------------------------------
    # Internal: rate-limited, cached API call
    # ------------------------------------------------------------------

    def _call(
        self,
        params: dict[str, str],
        cache_key: str,
        cache_ttl: float,
    ) -> dict[str, Any] | None:
        """Make a rate-limited, cached API call to Alpha Vantage.

        Returns parsed JSON dict or None on failure/rate-limit.
        """
        # Check cache first
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        # Rate limit check
        if not self._check_rate_limit():
            log.warning("Alpha Vantage rate limit reached — skipping call")
            return None

        # Daily budget check
        if not self._check_daily_budget():
            log.warning("Alpha Vantage daily budget exhausted — skipping call")
            return None

        # Make the API call
        try:
            params["apikey"] = self._key
            resp = self._session.get(
                ALPHA_VANTAGE_URL,
                params=params,
                timeout=15,
            )
            self._record_call()

            if resp.status_code != 200:
                log.warning("Alpha Vantage returned status %d", resp.status_code)
                return None

            data = resp.json()

            # AV returns error messages in the response body
            if "Error Message" in data or "Note" in data:
                error = data.get("Error Message") or data.get("Note", "")
                log.warning("Alpha Vantage API error: %s", error[:200])
                return None

            # Cache the result
            self._set_cache(cache_key, data, cache_ttl)
            return data

        except Exception as e:
            log.warning("Alpha Vantage API call failed: %s", e)
            return None

    def _check_rate_limit(self) -> bool:
        """Check if we can make another call (max 25/min)."""
        now = time.time()
        self._call_timestamps = [
            t for t in self._call_timestamps if now - t < 60
        ]
        return len(self._call_timestamps) < MAX_CALLS_PER_MINUTE

    def _check_daily_budget(self) -> bool:
        """Check if we're within daily call budget."""
        today = date.today()
        if today != self._daily_reset:
            self._daily_calls = 0
            self._daily_reset = today
        return self._daily_calls < MAX_CALLS_PER_DAY

    def _record_call(self) -> None:
        """Record a successful API call for rate limiting."""
        self._call_timestamps.append(time.time())
        self._daily_calls += 1

    def _get_cache(self, key: str) -> Any | None:
        """Get a cached value if it exists and hasn't expired."""
        if key in self._cache:
            expiry, data = self._cache[key]
            if time.time() < expiry:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, key: str, data: Any, ttl: float) -> None:
        """Store a value in the cache with TTL."""
        self._cache[key] = (time.time() + ttl, data)


# Module-level singleton for shared use
_client: AlphaVantageClient | None = None


def get_av_client() -> AlphaVantageClient:
    """Get the module-level AlphaVantageClient singleton."""
    global _client
    if _client is None:
        _client = AlphaVantageClient()
    return _client
