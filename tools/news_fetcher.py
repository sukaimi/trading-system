"""News fetching tools — RSS feeds and news APIs.

Sources: Reuters RSS, CoinDesk RSS, Kitco RSS,
Alpha Vantage News API, CryptoCompare News API.
All methods return standardized article dicts, never raise.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import feedparser
import requests

from core.logger import setup_logger

log = setup_logger("trading.news_fetcher")

RSS_FEEDS = {
    "reuters": "https://feeds.reuters.com/reuters/businessNews",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "kitco": "https://www.kitco.com/feed/",
}

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/news/"

# Alpha Vantage ticker format mapping
AV_TICKERS = {
    "BTC": "CRYPTO:BTC",
    "ETH": "CRYPTO:ETH",
    "GLDM": "GLDM",
    "SLV": "SLV",
}


class NewsFetcher:
    """Fetch and aggregate financial news from multiple sources."""

    def __init__(self):
        self._av_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self._cc_key: str = os.getenv("CRYPTOCOMPARE_API_KEY", "")
        self._av_timestamps: list[float] = []
        self._av_max_per_min: int = 5

    def fetch_rss(self, feed_url: str) -> list[dict[str, str]]:
        """Parse an RSS feed and return standardized articles."""
        try:
            feed = feedparser.parse(feed_url)
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
        except Exception as e:
            log.warning("RSS fetch failed for %s: %s", feed_url[:50], e)
            return []

    def fetch_all_rss(self) -> list[dict[str, str]]:
        """Fetch from all configured RSS feeds, deduplicate by title."""
        all_articles: list[dict[str, str]] = []
        seen_titles: set[str] = set()

        for name, url in RSS_FEEDS.items():
            articles = self.fetch_rss(url)
            for article in articles:
                title_lower = article["title"].lower().strip()
                if title_lower and title_lower not in seen_titles:
                    seen_titles.add(title_lower)
                    all_articles.append(article)

        log.info("Total unique RSS articles: %d", len(all_articles))
        return all_articles

    def fetch_alpha_vantage_news(
        self, tickers: list[str] | None = None
    ) -> list[dict[str, str]]:
        """Fetch news from Alpha Vantage News API (5/min rate limit)."""
        if not self._av_key:
            log.debug("Alpha Vantage API key not set — skipping")
            return []

        if not self._check_av_rate_limit():
            log.warning("Alpha Vantage rate limit reached — skipping")
            return []

        if tickers is None:
            tickers = ["BTC", "ETH", "GLDM", "SLV"]

        av_tickers = ",".join(AV_TICKERS.get(t, t) for t in tickers)

        try:
            resp = requests.get(
                ALPHA_VANTAGE_URL,
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": av_tickers,
                    "apikey": self._av_key,
                },
                timeout=15,
            )
            self._av_timestamps.append(time.time())

            if resp.status_code != 200:
                log.warning("Alpha Vantage returned status %d", resp.status_code)
                return []

            data = resp.json()
            articles = []
            for item in data.get("feed", []):
                articles.append(
                    self._standardize_article(
                        title=item.get("title", ""),
                        summary=item.get("summary", ""),
                        link=item.get("url", ""),
                        published=item.get("time_published", ""),
                        source=item.get("source", "alpha_vantage"),
                    )
                )
            log.info("Fetched %d articles from Alpha Vantage", len(articles))
            return articles
        except Exception as e:
            log.warning("Alpha Vantage fetch failed: %s", e)
            return []

    def fetch_crypto_news(self) -> list[dict[str, str]]:
        """Fetch crypto news from CryptoCompare."""
        if not self._cc_key:
            log.debug("CryptoCompare API key not set — skipping")
            return []

        try:
            resp = requests.get(
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
        all_articles = self.fetch_all_rss()
        all_articles.extend(self.fetch_alpha_vantage_news(tickers))
        all_articles.extend(self.fetch_crypto_news())

        # Deduplicate by title
        seen: set[str] = set()
        unique: list[dict[str, str]] = []
        for article in all_articles:
            title_lower = article["title"].lower().strip()
            if title_lower and title_lower not in seen:
                seen.add(title_lower)
                unique.append(article)

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
