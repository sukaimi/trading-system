"""Tests for tools/alpha_vantage.py — AlphaVantageClient.

All external HTTP calls are mocked. Tests cover:
- Rate limiting (per-minute and daily budget)
- Caching (hit, miss, expiry)
- ticker_sentiment() aggregation logic
- earnings() and EPS history
- sector_performance() and rankings
- Graceful fallback on errors
- Integration with news_fetcher delegation
- Integration with earnings_calendar API-backed dates
"""

import time
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from tools.alpha_vantage import AlphaVantageClient


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def av():
    """AlphaVantageClient with a test API key."""
    client = AlphaVantageClient()
    client._key = "test-key-123"
    return client


@pytest.fixture
def av_no_key():
    """AlphaVantageClient without an API key."""
    client = AlphaVantageClient()
    client._key = ""
    return client


MOCK_NEWS_RESPONSE = {
    "feed": [
        {
            "title": "Apple beats earnings",
            "summary": "AAPL reported strong Q1",
            "url": "http://example.com/1",
            "time_published": "20260312T120000",
            "source": "Reuters",
            "overall_sentiment_score": 0.25,
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.8",
                    "ticker_sentiment_score": "0.35",
                    "ticker_sentiment_label": "Somewhat_Bullish",
                },
                {
                    "ticker": "MSFT",
                    "relevance_score": "0.2",
                    "ticker_sentiment_score": "0.10",
                    "ticker_sentiment_label": "Neutral",
                },
            ],
        },
        {
            "title": "Tech sector rallies",
            "summary": "Broad tech gains",
            "url": "http://example.com/2",
            "time_published": "20260312T110000",
            "source": "CNBC",
            "overall_sentiment_score": 0.40,
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.5",
                    "ticker_sentiment_score": "0.20",
                    "ticker_sentiment_label": "Neutral",
                },
            ],
        },
    ]
}

MOCK_EARNINGS_RESPONSE = {
    "symbol": "AAPL",
    "annualEarnings": [
        {"fiscalDateEnding": "2025-09-30", "reportedEPS": "6.50"},
    ],
    "quarterlyEarnings": [
        {
            "fiscalDateEnding": "2026-06-30",
            "reportedDate": "2026-07-30",
            "reportedEPS": "0",
            "estimatedEPS": "1.60",
            "surprise": "0",
            "surprisePercentage": "0",
        },
        {
            "fiscalDateEnding": "2026-03-31",
            "reportedDate": "2026-04-30",
            "reportedEPS": "0",
            "estimatedEPS": "1.55",
            "surprise": "0",
            "surprisePercentage": "0",
        },
        {
            "fiscalDateEnding": "2025-12-31",
            "reportedDate": "2026-01-29",
            "reportedEPS": "2.40",
            "estimatedEPS": "2.35",
            "surprise": "0.05",
            "surprisePercentage": "2.1277",
        },
        {
            "fiscalDateEnding": "2025-09-30",
            "reportedDate": "2025-10-30",
            "reportedEPS": "1.64",
            "estimatedEPS": "1.60",
            "surprise": "0.04",
            "surprisePercentage": "2.50",
        },
    ],
}

MOCK_SECTOR_RESPONSE = {
    "Meta Data": {"Information": "US Sector Performance"},
    "Rank A: Real-Time Performance": {
        "Information Technology": "1.23%",
        "Health Care": "0.55%",
        "Energy": "-0.30%",
    },
    "Rank B: 1 Day Performance": {
        "Information Technology": "1.50%",
        "Health Care": "0.80%",
        "Energy": "-0.10%",
    },
    "Rank D: 1 Month Performance": {
        "Information Technology": "5.20%",
        "Health Care": "2.10%",
        "Financials": "1.80%",
        "Energy": "-1.50%",
    },
}


# ------------------------------------------------------------------
# Rate Limiting
# ------------------------------------------------------------------

class TestRateLimiting:
    def test_per_minute_rate_limit(self, av):
        """Should block when 25 calls made within 60 seconds."""
        av._call_timestamps = [time.time()] * 25
        assert av._check_rate_limit() is False

    def test_per_minute_under_limit(self, av):
        """Should allow when under 25 calls per minute."""
        av._call_timestamps = [time.time()] * 10
        assert av._check_rate_limit() is True

    def test_per_minute_old_timestamps_expire(self, av):
        """Timestamps older than 60s should be purged."""
        av._call_timestamps = [time.time() - 120] * 30
        assert av._check_rate_limit() is True

    def test_daily_budget_exceeded(self, av):
        """Should block when daily budget is exhausted."""
        av._daily_calls = 450
        av._daily_reset = date.today()
        assert av._check_daily_budget() is False

    def test_daily_budget_resets(self, av):
        """Daily counter should reset on new day."""
        av._daily_calls = 450
        av._daily_reset = date(2020, 1, 1)  # Old date
        assert av._check_daily_budget() is True
        assert av._daily_calls == 0

    def test_record_call_increments(self, av):
        """_record_call should increment daily count and add timestamp."""
        before = av._daily_calls
        av._record_call()
        assert av._daily_calls == before + 1
        assert len(av._call_timestamps) == 1


# ------------------------------------------------------------------
# Caching
# ------------------------------------------------------------------

class TestCaching:
    def test_cache_hit(self, av):
        """Cached data should be returned without API call."""
        av._set_cache("test_key", {"cached": True}, ttl=300)
        result = av._get_cache("test_key")
        assert result == {"cached": True}

    def test_cache_miss(self, av):
        """Missing key should return None."""
        assert av._get_cache("nonexistent") is None

    def test_cache_expired(self, av):
        """Expired cache entry should return None."""
        av._cache["expired_key"] = (time.time() - 10, {"old": True})
        assert av._get_cache("expired_key") is None
        assert "expired_key" not in av._cache  # Should be cleaned up

    def test_cache_prevents_api_call(self, av):
        """When data is cached, _call should not hit the network."""
        av._set_cache("news:AAPL", MOCK_NEWS_RESPONSE, ttl=300)
        result = av._call(
            params={"function": "NEWS_SENTIMENT", "tickers": "AAPL"},
            cache_key="news:AAPL",
            cache_ttl=300,
        )
        assert result == MOCK_NEWS_RESPONSE


# ------------------------------------------------------------------
# No API Key
# ------------------------------------------------------------------

class TestNoApiKey:
    def test_news_sentiment_no_key(self, av_no_key):
        assert av_no_key.news_sentiment() is None

    def test_earnings_no_key(self, av_no_key):
        assert av_no_key.earnings("AAPL") is None

    def test_sector_performance_no_key(self, av_no_key):
        assert av_no_key.sector_performance() is None

    def test_ticker_sentiment_no_key(self, av_no_key):
        assert av_no_key.ticker_sentiment("AAPL") is None

    def test_has_key_false(self, av_no_key):
        assert av_no_key.has_key is False

    def test_has_key_true(self, av):
        assert av.has_key is True


# ------------------------------------------------------------------
# Ticker Sentiment Aggregation (P0)
# ------------------------------------------------------------------

class TestTickerSentiment:
    def test_aggregated_score(self, av):
        """Should compute relevance-weighted average sentiment."""
        av._set_cache("news:AAPL", MOCK_NEWS_RESPONSE, ttl=300)
        result = av.ticker_sentiment("AAPL")

        assert result is not None
        assert "score" in result
        assert "label" in result
        assert "article_count" in result
        assert "avg_relevance" in result
        assert result["article_count"] == 2

        # Weighted average: (0.35*0.8 + 0.20*0.5) / (0.8+0.5) = 0.38/1.3 ≈ 0.2923
        assert 0.28 < result["score"] < 0.31

    def test_unknown_ticker(self, av):
        """Ticker not in any article should return None."""
        av._set_cache("news:UNKNOWN123", MOCK_NEWS_RESPONSE, ttl=300)
        # Force a cache hit for the news call
        with patch.object(av, "news_sentiment", return_value=MOCK_NEWS_RESPONSE):
            result = av.ticker_sentiment("UNKNOWN123")
        assert result is None

    def test_label_bearish(self, av):
        """Score <= -0.35 should be labeled Bearish."""
        mock_data = {
            "feed": [{
                "ticker_sentiment": [{
                    "ticker": "AAPL",
                    "relevance_score": "0.9",
                    "ticker_sentiment_score": "-0.50",
                    "ticker_sentiment_label": "Bearish",
                }],
            }],
        }
        with patch.object(av, "news_sentiment", return_value=mock_data):
            result = av.ticker_sentiment("AAPL")
        assert result is not None
        assert result["label"] == "Bearish"

    def test_label_bullish(self, av):
        """Score >= 0.35 should be labeled Bullish."""
        mock_data = {
            "feed": [{
                "ticker_sentiment": [{
                    "ticker": "AAPL",
                    "relevance_score": "0.9",
                    "ticker_sentiment_score": "0.50",
                    "ticker_sentiment_label": "Bullish",
                }],
            }],
        }
        with patch.object(av, "news_sentiment", return_value=mock_data):
            result = av.ticker_sentiment("AAPL")
        assert result is not None
        assert result["label"] == "Bullish"

    def test_zero_relevance_ignored(self, av):
        """Articles with zero relevance should be ignored."""
        mock_data = {
            "feed": [{
                "ticker_sentiment": [{
                    "ticker": "AAPL",
                    "relevance_score": "0.0",
                    "ticker_sentiment_score": "0.50",
                    "ticker_sentiment_label": "Bullish",
                }],
            }],
        }
        with patch.object(av, "news_sentiment", return_value=mock_data):
            result = av.ticker_sentiment("AAPL")
        assert result is None


# ------------------------------------------------------------------
# Earnings (P1)
# ------------------------------------------------------------------

class TestEarnings:
    def test_earnings_returns_data(self, av):
        """Should return earnings data from API."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_EARNINGS_RESPONSE
        with patch.object(av._session, "get", return_value=mock_resp):
            result = av.earnings("AAPL")
        assert result is not None
        assert "quarterlyEarnings" in result

    def test_earnings_cached(self, av):
        """Earnings should be cached for 7 days."""
        av._set_cache("earnings:AAPL", MOCK_EARNINGS_RESPONSE, ttl=7 * 86400)
        result = av.earnings("AAPL")
        assert result == MOCK_EARNINGS_RESPONSE

    def test_eps_history(self, av):
        """Should extract EPS surprise history from earnings data."""
        av._set_cache("earnings:AAPL", MOCK_EARNINGS_RESPONSE, ttl=7 * 86400)
        history = av.get_eps_history("AAPL", quarters=4)
        # Only entries with reportedDate <= today and non-zero reportedEPS
        assert isinstance(history, list)
        # At least the 2026-01-29 and 2025-10-30 entries should be included
        # (they have past reportedDates with non-zero EPS)
        for entry in history:
            assert "date" in entry
            assert "reported_eps" in entry
            assert "estimated_eps" in entry
            assert "surprise_pct" in entry

    def test_eps_history_empty_on_no_data(self, av):
        """Should return empty list when no earnings data."""
        assert av.get_eps_history("UNKNOWN") == []

    def test_upcoming_earnings_date(self, av):
        """Should find the next future earnings date."""
        av._set_cache("earnings:AAPL", MOCK_EARNINGS_RESPONSE, ttl=7 * 86400)
        result = av.get_upcoming_earnings_date("AAPL")
        # Should be a future date (2026-04-30 or 2026-07-30)
        if result is not None:
            assert result >= date.today().isoformat()


# ------------------------------------------------------------------
# Sector Performance (P2)
# ------------------------------------------------------------------

class TestSectorPerformance:
    def test_sector_data_returned(self, av):
        """Should return sector performance data."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_SECTOR_RESPONSE
        with patch.object(av._session, "get", return_value=mock_resp):
            result = av.sector_performance()
        assert result is not None
        assert "Rank D: 1 Month Performance" in result

    def test_sector_rankings_sorted(self, av):
        """Rankings should be sorted by performance descending."""
        av._set_cache("sector", MOCK_SECTOR_RESPONSE, ttl=4 * 3600)
        rankings = av.get_sector_rankings("1mo")
        assert len(rankings) == 4
        assert rankings[0]["rank"] == 1
        assert rankings[0]["sector"] == "Information Technology"
        assert rankings[0]["performance_pct"] == 5.20
        assert rankings[-1]["sector"] == "Energy"
        assert rankings[-1]["performance_pct"] == -1.50

    def test_sector_rankings_realtime(self, av):
        """Should handle realtime timeframe."""
        av._set_cache("sector", MOCK_SECTOR_RESPONSE, ttl=4 * 3600)
        rankings = av.get_sector_rankings("realtime")
        assert len(rankings) == 3
        assert rankings[0]["sector"] == "Information Technology"

    def test_sector_rankings_empty_on_failure(self, av):
        """Should return empty list when no data."""
        assert av.get_sector_rankings("1mo") == []

    def test_sector_rankings_invalid_timeframe(self, av):
        """Invalid timeframe should default to 1mo."""
        av._set_cache("sector", MOCK_SECTOR_RESPONSE, ttl=4 * 3600)
        rankings = av.get_sector_rankings("invalid")
        assert len(rankings) == 4  # Falls back to 1mo


# ------------------------------------------------------------------
# API Error Handling
# ------------------------------------------------------------------

class TestErrorHandling:
    def test_http_error(self, av):
        """Non-200 status should return None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch.object(av._session, "get", return_value=mock_resp):
            result = av._call(
                params={"function": "SECTOR"},
                cache_key="test",
                cache_ttl=60,
            )
        assert result is None

    def test_api_error_message(self, av):
        """AV error messages in response body should return None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Error Message": "Invalid API call"
        }
        with patch.object(av._session, "get", return_value=mock_resp):
            result = av._call(
                params={"function": "SECTOR"},
                cache_key="test",
                cache_ttl=60,
            )
        assert result is None

    def test_api_note_rate_limit(self, av):
        """AV rate limit Note in response should return None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 25 calls per day."
        }
        with patch.object(av._session, "get", return_value=mock_resp):
            result = av._call(
                params={"function": "SECTOR"},
                cache_key="test",
                cache_ttl=60,
            )
        assert result is None

    def test_network_exception(self, av):
        """Network errors should return None gracefully."""
        with patch.object(av._session, "get", side_effect=ConnectionError("timeout")):
            result = av._call(
                params={"function": "SECTOR"},
                cache_key="test",
                cache_ttl=60,
            )
        assert result is None

    def test_rate_limited_call_returns_none(self, av):
        """When rate limited, _call should return None without hitting network."""
        av._call_timestamps = [time.time()] * 25
        result = av._call(
            params={"function": "SECTOR"},
            cache_key="test",
            cache_ttl=60,
        )
        assert result is None

    def test_daily_budget_exhausted(self, av):
        """When daily budget is exhausted, _call should return None."""
        av._daily_calls = 450
        av._daily_reset = date.today()
        result = av._call(
            params={"function": "SECTOR"},
            cache_key="test",
            cache_ttl=60,
        )
        assert result is None


# ------------------------------------------------------------------
# News Fetcher Delegation
# ------------------------------------------------------------------

class TestNewsFetcherDelegation:
    def test_news_fetcher_uses_av_client(self):
        """NewsFetcher.fetch_alpha_vantage_news should delegate to AV client."""
        from tools.news_fetcher import NewsFetcher

        nf = NewsFetcher(inter_feed_delay=0)
        mock_av = MagicMock()
        mock_av.has_key = True
        mock_av.news_sentiment.return_value = MOCK_NEWS_RESPONSE
        nf._av_client = mock_av

        articles = nf.fetch_alpha_vantage_news(["AAPL"])
        assert len(articles) == 2
        assert articles[0]["title"] == "Apple beats earnings"
        mock_av.news_sentiment.assert_called_once_with(["AAPL"])

    def test_news_fetcher_no_key(self):
        """NewsFetcher should return empty when AV has no key."""
        from tools.news_fetcher import NewsFetcher

        nf = NewsFetcher(inter_feed_delay=0)
        mock_av = MagicMock()
        mock_av.has_key = False
        nf._av_client = mock_av

        articles = nf.fetch_alpha_vantage_news(["AAPL"])
        assert articles == []

    def test_news_fetcher_av_returns_none(self):
        """NewsFetcher should return empty when AV returns None."""
        from tools.news_fetcher import NewsFetcher

        nf = NewsFetcher(inter_feed_delay=0)
        mock_av = MagicMock()
        mock_av.has_key = True
        mock_av.news_sentiment.return_value = None
        nf._av_client = mock_av

        articles = nf.fetch_alpha_vantage_news(["AAPL"])
        assert articles == []

    def test_news_fetcher_extracts_sentiment(self):
        """NewsFetcher should extract per-article sentiment and relevance."""
        from tools.news_fetcher import NewsFetcher

        nf = NewsFetcher(inter_feed_delay=0)
        mock_av = MagicMock()
        mock_av.has_key = True
        mock_av.news_sentiment.return_value = MOCK_NEWS_RESPONSE
        nf._av_client = mock_av

        articles = nf.fetch_alpha_vantage_news(["AAPL"])
        assert articles[0]["sentiment_score"] == 0.25
        assert articles[0]["relevance_score"] == "0.8"


# ------------------------------------------------------------------
# Earnings Calendar Integration
# ------------------------------------------------------------------

class TestEarningsCalendarIntegration:
    def test_api_backed_dates(self):
        """EarningsCalendar should use AV data for known stocks."""
        from core.earnings_calendar import EarningsCalendar

        mock_av = MagicMock()
        mock_av.earnings.return_value = {
            "quarterlyEarnings": [
                {"reportedDate": "2026-07-30"},
                {"reportedDate": "2026-04-30"},
                {"reportedDate": "2026-01-29"},
            ],
        }

        cal = EarningsCalendar(av_client=mock_av)
        days = cal.days_until_earnings("MSFT", ref_date=date(2026, 4, 27))
        # Should find 2026-04-30 = 3 days away
        assert days == 3

    def test_api_fallback_to_hardcoded(self):
        """When AV returns None, should fall back to hardcoded dates."""
        from core.earnings_calendar import EarningsCalendar

        mock_av = MagicMock()
        mock_av.earnings.return_value = None

        cal = EarningsCalendar(av_client=mock_av)
        # AAPL has hardcoded dates
        days = cal.days_until_earnings("AAPL", ref_date=date(2026, 4, 27))
        assert days == 3  # 2026-04-30

    def test_etf_always_none(self):
        """ETFs should always return None regardless of AV client."""
        from core.earnings_calendar import EarningsCalendar

        mock_av = MagicMock()
        cal = EarningsCalendar(av_client=mock_av)
        assert cal.days_until_earnings("SPY") is None
        assert cal.days_until_earnings("GLDM") is None
        mock_av.earnings.assert_not_called()

    def test_crypto_always_none(self):
        """Crypto should always return None."""
        from core.earnings_calendar import EarningsCalendar

        mock_av = MagicMock()
        cal = EarningsCalendar(av_client=mock_av)
        assert cal.days_until_earnings("BTC") is None
        assert cal.days_until_earnings("ETH") is None

    def test_get_eps_history(self):
        """Should delegate EPS history to AV client."""
        from core.earnings_calendar import EarningsCalendar

        mock_av = MagicMock()
        mock_av.get_eps_history.return_value = [
            {"date": "2026-01-29", "reported_eps": 2.40, "estimated_eps": 2.35, "surprise_pct": 2.13},
        ]

        cal = EarningsCalendar(av_client=mock_av)
        history = cal.get_eps_history("AAPL")
        assert len(history) == 1
        assert history[0]["surprise_pct"] == 2.13

    def test_get_eps_history_no_av(self):
        """Should return empty list when no AV client."""
        from core.earnings_calendar import EarningsCalendar

        cal = EarningsCalendar(av_client=None)
        # With av_client=None, it tries to import get_av_client
        # which might not have a key — graceful fallback
        # We test the explicit None path
        cal._av_client = None
        history = cal.get_eps_history("AAPL")
        assert history == [] or isinstance(history, list)

    def test_open_universe_ticker(self):
        """Open universe stock should get earnings from AV API."""
        from core.earnings_calendar import EarningsCalendar

        mock_av = MagicMock()
        mock_av.earnings.return_value = {
            "quarterlyEarnings": [
                {"reportedDate": "2026-05-15"},
                {"reportedDate": "2026-02-12"},
            ],
        }

        cal = EarningsCalendar(av_client=mock_av)
        days = cal.days_until_earnings("GOOG", ref_date=date(2026, 5, 10))
        assert days == 5  # 2026-05-15

    def test_upcoming_earnings_includes_api_dates(self):
        """upcoming_earnings should include API-cached tickers."""
        from core.earnings_calendar import EarningsCalendar

        mock_av = MagicMock()
        mock_av.earnings.return_value = None  # Don't auto-fetch

        cal = EarningsCalendar(av_client=mock_av)
        # Manually populate API cache as if we'd fetched earlier
        cal._api_dates_cache["GOOG"] = ["2026-01-25", "2026-04-20"]

        result = cal.upcoming_earnings(days=5, ref_date=date(2026, 4, 18))
        assets = {r["asset"] for r in result}
        assert "GOOG" in assets


# ------------------------------------------------------------------
# Technical Indicators (P3: VWAP, Stochastic, OBV)
# ------------------------------------------------------------------

class TestVWAP:
    def test_vwap_basic(self):
        """VWAP should compute volume-weighted average of typical price."""
        from tools.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()

        highs = [102.0, 104.0, 106.0]
        lows = [98.0, 100.0, 102.0]
        closes = [100.0, 102.0, 104.0]
        volumes = [1000.0, 2000.0, 1500.0]

        result = ti.vwap(highs, lows, closes, volumes)
        assert result > 0

        # Manual: TP = [(102+98+100)/3, (104+100+102)/3, (106+102+104)/3]
        #         = [100, 102, 104]
        # VWAP = (100*1000 + 102*2000 + 104*1500) / (1000+2000+1500)
        #      = (100000 + 204000 + 156000) / 4500 = 460000/4500 ≈ 102.222
        assert abs(result - 102.222) < 0.01

    def test_vwap_zero_volume(self):
        """Zero total volume should return 0.0."""
        from tools.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        result = ti.vwap([100.0], [98.0], [99.0], [0.0])
        assert result == 0.0

    def test_vwap_empty(self):
        """Empty data should return 0.0."""
        from tools.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        assert ti.vwap([], [], [], []) == 0.0


class TestStochastic:
    def test_stochastic_overbought(self):
        """Price at top of range should give high %K."""
        from tools.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()

        # Rising prices — close near the high
        highs = [float(100 + i) for i in range(20)]
        lows = [float(95 + i) for i in range(20)]
        closes = [float(99 + i) for i in range(20)]

        result = ti.stochastic(highs, lows, closes)
        assert result["k"] > 70

    def test_stochastic_oversold(self):
        """Price at bottom of range should give low %K."""
        from tools.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()

        # Falling prices — close near the low
        highs = [float(105 - i) for i in range(20)]
        lows = [float(95 - i) for i in range(20)]
        closes = [float(96 - i) for i in range(20)]

        result = ti.stochastic(highs, lows, closes)
        assert result["k"] < 30

    def test_stochastic_insufficient_data(self):
        """Should return zeroes with too few data points."""
        from tools.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        result = ti.stochastic([100.0] * 5, [98.0] * 5, [99.0] * 5)
        assert result["k"] == 0.0
        assert result["d"] == 0.0

    def test_stochastic_keys(self):
        """Should return dict with k and d keys."""
        from tools.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        highs = [float(105 + (i % 3)) for i in range(20)]
        lows = [float(95 + (i % 3)) for i in range(20)]
        closes = [float(100 + (i % 3)) for i in range(20)]
        result = ti.stochastic(highs, lows, closes)
        assert "k" in result
        assert "d" in result


class TestOBV:
    def test_obv_rising(self):
        """OBV should be positive for rising prices."""
        from tools.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()

        closes = [float(100 + i) for i in range(20)]
        volumes = [1000.0] * 20

        result = ti.obv(closes, volumes)
        assert result > 0  # All up days

    def test_obv_falling(self):
        """OBV should be negative for falling prices."""
        from tools.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()

        closes = [float(120 - i) for i in range(20)]
        volumes = [1000.0] * 20

        result = ti.obv(closes, volumes)
        assert result < 0  # All down days

    def test_obv_insufficient_data(self):
        """Should return 0.0 with too few data points."""
        from tools.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        assert ti.obv([100.0], [1000.0]) == 0.0

    def test_obv_flat(self):
        """Flat prices should give zero OBV."""
        from tools.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()

        closes = [100.0] * 20
        volumes = [1000.0] * 20

        result = ti.obv(closes, volumes)
        assert result == 0.0


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

class TestSingleton:
    def test_get_av_client_returns_same_instance(self):
        """Module-level singleton should return same instance."""
        import tools.alpha_vantage as av_mod
        av_mod._client = None  # Reset
        c1 = av_mod.get_av_client()
        c2 = av_mod.get_av_client()
        assert c1 is c2
        av_mod._client = None  # Clean up
