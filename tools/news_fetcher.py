"""News fetching tools — RSS feeds and news APIs.

Sources: CoinDesk RSS, Yahoo Finance, MarketWatch,
CNBC, Seeking Alpha, Investing.com, Alpha Vantage News API,
CryptoCompare News API.
All methods return standardized article dicts, never raise.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from typing import Any

import feedparser
import requests

from core.logger import setup_logger
from tools.alpha_vantage import AlphaVantageClient, get_av_client

log = setup_logger("trading.news_fetcher")

RSS_FEEDS = {
    # Crypto
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    # Equities & macro (also covers gold/silver/precious metals)
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "cnbc": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "seeking_alpha": "https://seekingalpha.com/market_currents.xml",
    "investing_com": "https://www.investing.com/rss/news.rss",
    "straits_times": "https://www.straitstimes.com/news/business/rss.xml",
    "scmp": "https://www.scmp.com/rss/91/feed",
}

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/news/"

# Alpha Vantage ticker format mapping
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

SEEN_ARTICLES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "seen_articles.json"
)


class NewsFetcher:
    """Fetch and aggregate financial news from multiple sources."""

    def __init__(self, inter_feed_delay: float = 1.5):
        self._av_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self._cc_key: str = os.getenv("CRYPTOCOMPARE_API_KEY", "")
        self._av_timestamps: list[float] = []
        self._av_max_per_min: int = 5

        # Item 1: requests.Session with User-Agent
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "TradingSystem/1.0 (+https://tradebot.codeandcraft.ai)"
        )

        # Item 2: Inter-feed delay
        self._inter_feed_delay = inter_feed_delay

        # Item 3: Persistent dedup
        self._seen_articles: dict[str, dict] = self._load_seen_articles()

        # Item 4: Per-feed failure tracking with exponential backoff
        self._feed_failures: dict[str, dict] = {}
        self._cycle_count: int = 0

        # Item 5: Stale feed detection
        self._feed_last_new_article: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public properties for self-healer monitor access
    # ------------------------------------------------------------------

    @property
    def feed_failures(self) -> dict[str, dict]:
        """Expose feed failure tracking for self-healer (returns copy)."""
        return dict(self._feed_failures)

    @property
    def feed_last_new_article(self) -> dict[str, str]:
        """Expose stale feed tracking for self-healer (returns copy)."""
        return dict(self._feed_last_new_article)

    @property
    def cycle_count(self) -> int:
        """Current fetch cycle count."""
        return self._cycle_count

    # ------------------------------------------------------------------
    # Item 3: Persistent dedup helpers
    # ------------------------------------------------------------------

    def _load_seen_articles(self) -> dict[str, dict]:
        """Load seen articles from JSON file."""
        try:
            with open(SEEN_ARTICLES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _persist_seen_articles(self) -> None:
        """Write seen articles to JSON file."""
        try:
            os.makedirs(os.path.dirname(SEEN_ARTICLES_FILE), exist_ok=True)
            with open(SEEN_ARTICLES_FILE, "w") as f:
                json.dump(self._seen_articles, f, indent=2)
        except Exception as e:
            log.warning("Failed to persist seen articles: %s", e)

    def _cleanup_expired_articles(self) -> None:
        """Remove entries where first_seen is older than 48 hours."""
        cutoff = datetime.utcnow() - timedelta(hours=48)
        expired = [
            key
            for key, val in self._seen_articles.items()
            if datetime.fromisoformat(val.get("first_seen", "2000-01-01")) < cutoff
        ]
        for key in expired:
            del self._seen_articles[key]

    # ------------------------------------------------------------------
    # Item 4: Raw RSS fetch (raises on error)
    # ------------------------------------------------------------------

    def _fetch_rss_raw(self, feed_url: str) -> list[dict[str, str]]:
        """Parse an RSS feed and return standardized articles. Raises on error."""
        feed = feedparser.parse(
            feed_url, agent=self._session.headers["User-Agent"]
        )
        if getattr(feed, "bozo", False):
            log.warning("RSS feed may be broken (bozo): %s — %s",
                        feed_url[:50], getattr(feed, "bozo_exception", "unknown"))
        articles = []
        for entry in feed.get("entries", []):
            articles.append(
                self._standardize_article(
                    title=entry.get("title", ""),
                    summary=entry.get("summary", entry.get("description", "")),
                    link=entry.get("link", ""),
                    published=entry.get("published", ""),
                    source=feed.get("feed", {}).get("title", feed_url),
                )
            )
        log.info("Fetched %d articles from %s", len(articles), feed_url[:50])
        return articles

    def fetch_rss(self, feed_url: str) -> list[dict[str, str]]:
        """Parse an RSS feed and return standardized articles."""
        try:
            return self._fetch_rss_raw(feed_url)
        except Exception as e:
            log.warning("RSS fetch failed for %s: %s", feed_url[:50], e)
            return []

    def fetch_all_rss(self) -> list[dict[str, str]]:
        """Fetch from all configured RSS feeds, deduplicate by title."""
        all_articles: list[dict[str, str]] = []
        seen_titles: set[str] = set()

        # Item 4: Increment cycle count
        self._cycle_count += 1

        for i, (name, url) in enumerate(RSS_FEEDS.items()):
            # Item 2: Inter-feed delay (skip for first feed)
            if i > 0:
                time.sleep(self._inter_feed_delay)

            # Item 4: Check backoff
            failure_info = self._feed_failures.get(name, {})
            skip_until = failure_info.get("skip_until_cycle", 0)
            if skip_until > self._cycle_count:
                log.debug(
                    "Skipping feed %s (backoff until cycle %d, current %d)",
                    name,
                    skip_until,
                    self._cycle_count,
                )
                continue

            try:
                articles = self._fetch_rss_raw(url)
                # Item 4: Success — reset failures
                self._feed_failures.pop(name, None)

                new_count = 0
                for article in articles:
                    title_lower = article["title"].lower().strip()
                    if title_lower and title_lower not in seen_titles:
                        seen_titles.add(title_lower)
                        all_articles.append(article)
                        new_count += 1

                # Item 5: Track last time feed produced new articles
                if new_count > 0:
                    self._feed_last_new_article[name] = (
                        datetime.utcnow().isoformat()
                    )

            except Exception as e:
                # Item 4: Track failure with exponential backoff
                info = self._feed_failures.get(name, {"consecutive_failures": 0})
                info["consecutive_failures"] = info["consecutive_failures"] + 1
                backoff = 2 ** min(info["consecutive_failures"], 3)
                info["skip_until_cycle"] = self._cycle_count + backoff
                self._feed_failures[name] = info
                log.warning(
                    "Feed %s failed (attempt %d, backoff %d cycles): %s",
                    name,
                    info["consecutive_failures"],
                    backoff,
                    e,
                )

        # Item 5: Stale feed detection (skip on first cycle)
        if self._cycle_count > 1:
            now = datetime.utcnow()
            for feed_name, ts_str in self._feed_last_new_article.items():
                # Don't warn for feeds currently in backoff
                if feed_name in self._feed_failures:
                    continue
                last_new = datetime.fromisoformat(ts_str)
                hours_since = (now - last_new).total_seconds() / 3600
                if hours_since > 24:
                    log.warning(
                        "STALE FEED: %s has not produced new articles in %.1f hours",
                        feed_name,
                        hours_since,
                    )

        log.info("Total unique RSS articles: %d", len(all_articles))
        return all_articles

    def fetch_alpha_vantage_news(
        self, tickers: list[str] | None = None
    ) -> list[dict[str, str]]:
        """Fetch news from Alpha Vantage News API via shared AV client.

        Delegates to AlphaVantageClient for rate limiting and caching.
        Also extracts per-ticker sentiment data for downstream use.
        """
        av_client = self._get_av_client()
        if not av_client.has_key:
            log.debug("Alpha Vantage API key not set — skipping")
            return []

        if tickers is None:
            try:
                from core.asset_registry import get_tradeable_assets
                tickers = get_tradeable_assets()
            except Exception:
                tickers = list(AV_TICKERS.keys())

        data = av_client.news_sentiment(tickers)
        if not data:
            return []

        av_tickers_set = set(AV_TICKERS.get(t, t) for t in tickers)

        articles = []
        for item in data.get("feed", []):
            article = self._standardize_article(
                title=item.get("title", ""),
                summary=item.get("summary", ""),
                link=item.get("url", ""),
                published=item.get("time_published", ""),
                source=item.get("source", "alpha_vantage"),
            )
            article["sentiment_score"] = item.get(
                "overall_sentiment_score", ""
            )
            # Find best relevance score among requested tickers
            relevance = ""
            for ts in item.get("ticker_sentiment", []):
                ticker_sym = ts.get("ticker", "")
                if ticker_sym in av_tickers_set:
                    relevance = ts.get("relevance_score", "")
                    break
            article["relevance_score"] = relevance
            articles.append(article)

        log.info("Fetched %d articles from Alpha Vantage", len(articles))
        return articles

    def _get_av_client(self) -> AlphaVantageClient:
        """Get the Alpha Vantage client (allows override for testing)."""
        if not hasattr(self, "_av_client") or self._av_client is None:
            self._av_client = get_av_client()
        return self._av_client

    def fetch_crypto_news(self) -> list[dict[str, str]]:
        """Fetch crypto news from CryptoCompare."""
        if not self._cc_key:
            log.debug("CryptoCompare API key not set — skipping")
            return []

        try:
            resp = self._session.get(
                CRYPTOCOMPARE_URL,
                params={"lang": "EN"},
                headers={"Authorization": f"Apikey {self._cc_key}"},
                timeout=15,
            )
            if resp.status_code != 200:
                log.warning("CryptoCompare returned status %d", resp.status_code)
                return []

            data = resp.json()
            articles = []
            for item in data.get("Data", []):
                articles.append(
                    self._standardize_article(
                        title=item.get("title", ""),
                        summary=item.get("body", "")[:500],
                        link=item.get("url", ""),
                        published=datetime.utcfromtimestamp(
                            item.get("published_on", 0)
                        ).isoformat()
                        if item.get("published_on")
                        else "",
                        source=item.get("source", "cryptocompare"),
                    )
                )
            log.info("Fetched %d articles from CryptoCompare", len(articles))
            return articles
        except Exception as e:
            log.warning("CryptoCompare fetch failed: %s", e)
            return []

    def fetch_fed_calendar(self) -> list[dict[str, str]]:
        """Return known upcoming FOMC meeting dates.

        Uses hardcoded dates as a reliable fallback. A scraper for
        live calendar data can be added in a later phase.
        """
        fomc_dates = [
            {"event": "FOMC Meeting", "date": "2026-03-18", "description": "FOMC rate decision"},
            {"event": "FOMC Meeting", "date": "2026-05-06", "description": "FOMC rate decision"},
            {"event": "FOMC Meeting", "date": "2026-06-17", "description": "FOMC rate decision + SEP"},
            {"event": "FOMC Meeting", "date": "2026-07-29", "description": "FOMC rate decision"},
            {"event": "FOMC Meeting", "date": "2026-09-16", "description": "FOMC rate decision + SEP"},
            {"event": "FOMC Meeting", "date": "2026-11-04", "description": "FOMC rate decision"},
            {"event": "FOMC Meeting", "date": "2026-12-16", "description": "FOMC rate decision + SEP"},
        ]
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return [d for d in fomc_dates if d["date"] >= today]

    def fetch_all(self, tickers: list[str] | None = None) -> list[dict[str, str]]:
        """Fetch news from all sources, aggregate and return.

        This is the main entry point called by NewsScout.
        """
        # Item 3: Clean up expired seen articles
        self._cleanup_expired_articles()

        all_articles = self.fetch_all_rss()
        all_articles.extend(self.fetch_alpha_vantage_news(tickers))
        all_articles.extend(self.fetch_crypto_news())

        # Deduplicate by title (in-batch)
        seen: set[str] = set()
        unique: list[dict[str, str]] = []
        for article in all_articles:
            title_lower = article["title"].lower().strip()
            if title_lower and title_lower not in seen:
                seen.add(title_lower)
                # Item 3: Persistent dedup — check against seen_articles
                dedup_key = article.get("link") or title_lower
                if dedup_key not in self._seen_articles:
                    self._seen_articles[dedup_key] = {
                        "title": article["title"],
                        "first_seen": datetime.utcnow().isoformat(),
                    }
                    unique.append(article)

        # Item 3: Persist after dedup
        self._persist_seen_articles()

        log.info("Total aggregated articles: %d", len(unique))
        return unique

    def _check_av_rate_limit(self) -> bool:
        """Check if we can make another Alpha Vantage call (5/min)."""
        now = time.time()
        self._av_timestamps = [t for t in self._av_timestamps if now - t < 60]
        return len(self._av_timestamps) < self._av_max_per_min

    @staticmethod
    def _standardize_article(
        title: str, summary: str, link: str, published: str, source: str
    ) -> dict[str, str]:
        """Normalize article fields to standard dict."""
        return {
            "title": title.strip(),
            "summary": summary.strip()[:500],
            "link": link.strip(),
            "published": published.strip(),
            "source": source.strip(),
        }
