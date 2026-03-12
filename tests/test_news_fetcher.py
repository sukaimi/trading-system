"""Tests for tools/news_fetcher.py — all external calls mocked."""

import json
import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from tools.news_fetcher import NewsFetcher, SEEN_ARTICLES_FILE


@pytest.fixture
def nf():
    return NewsFetcher(inter_feed_delay=0)


class TestSessionUserAgent:
    def test_session_has_user_agent(self):
        nf = NewsFetcher(inter_feed_delay=0)
        assert nf._session.headers["User-Agent"].startswith("TradingSystem/")


class TestFetchRSS:
    @patch("tools.news_fetcher.feedparser.parse")
    def test_success(self, mock_parse, nf):
        mock_parse.return_value = {
            "entries": [
                {"title": "BTC hits ATH", "summary": "Bitcoin new high", "link": "http://a.com", "published": "2026-03-01"},
                {"title": "ETH upgrade", "summary": "Ethereum update", "link": "http://b.com", "published": "2026-03-01"},
            ],
            "feed": {"title": "Test Feed"},
        }
        articles = nf.fetch_rss("http://example.com/rss")
        assert len(articles) == 2
        assert articles[0]["title"] == "BTC hits ATH"

    @patch("tools.news_fetcher.feedparser.parse")
    def test_parse_failure(self, mock_parse, nf):
        mock_parse.side_effect = Exception("parse error")
        assert nf.fetch_rss("http://bad-url") == []


class TestFetchAllRSS:
    @patch("tools.news_fetcher.feedparser.parse")
    def test_deduplication(self, mock_parse, nf):
        mock_parse.return_value = {
            "entries": [
                {"title": "Same Headline", "summary": "s", "link": "http://a.com", "published": ""},
            ],
            "feed": {"title": "Feed"},
        }
        articles = nf.fetch_all_rss()
        # All feeds return same headline — should be deduped to 1
        assert len(articles) == 1


class TestFetchAlphaVantage:
    def test_no_api_key(self, nf):
        mock_av = MagicMock()
        mock_av.has_key = False
        nf._av_client = mock_av
        assert nf.fetch_alpha_vantage_news() == []

    def test_success(self, nf):
        mock_av = MagicMock()
        mock_av.has_key = True
        mock_av.news_sentiment.return_value = {
            "feed": [
                {"title": "Fed holds rates", "summary": "...", "url": "http://c.com", "time_published": "2026-03-01", "source": "Reuters"},
            ]
        }
        nf._av_client = mock_av
        articles = nf.fetch_alpha_vantage_news(["BTC"])
        assert len(articles) == 1


class TestFetchCryptoNews:
    def test_no_api_key(self, nf):
        nf._cc_key = ""
        assert nf.fetch_crypto_news() == []

    def test_success(self, nf):
        nf._cc_key = "test-key"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Data": [
                {"title": "Crypto rally", "body": "...", "url": "http://d.com", "published_on": 1740000000, "source": "cc"},
            ]
        }
        with patch.object(nf._session, "get", return_value=mock_resp):
            articles = nf.fetch_crypto_news()
        assert len(articles) == 1


class TestFetchFedCalendar:
    def test_returns_future_dates(self, nf):
        dates = nf.fetch_fed_calendar()
        assert isinstance(dates, list)
        assert all("date" in d for d in dates)


class TestFetchAll:
    @patch.object(NewsFetcher, "fetch_all_rss", return_value=[
        {"title": "A", "summary": "", "link": "http://a1.com", "published": "", "source": ""},
    ])
    @patch.object(NewsFetcher, "fetch_alpha_vantage_news", return_value=[
        {"title": "B", "summary": "", "link": "http://b1.com", "published": "", "source": ""},
    ])
    @patch.object(NewsFetcher, "fetch_crypto_news", return_value=[
        {"title": "A", "summary": "", "link": "http://a2.com", "published": "", "source": ""},
    ])
    def test_dedup_across_sources(self, mock_cc, mock_av, mock_rss, nf):
        nf._seen_articles = {}  # Clear persistent dedup for isolated test
        articles = nf.fetch_all()
        # "A" appears twice (different links) but same title — title dedup keeps 1
        # "B" is unique — kept
        assert len(articles) == 2  # "A" deduped by title


class TestFeedFailureBackoff:
    @patch("tools.news_fetcher.feedparser.parse")
    def test_feed_failure_backoff(self, mock_parse, nf):
        """Mock feedparser to raise for one specific feed, verify exponential skip."""
        call_count = {"coindesk": 0}

        def parse_side_effect(url, agent=None):
            # Fail for coindesk, succeed for others
            if "coindesk" in url:
                call_count["coindesk"] += 1
                raise ConnectionError("timeout")
            return {"entries": [], "feed": {"title": "Feed"}}

        mock_parse.side_effect = parse_side_effect

        # Cycle 1: coindesk fails (consecutive_failures=1, skip_until=cycle+2)
        nf.fetch_all_rss()
        assert "coindesk" in nf._feed_failures
        assert nf._feed_failures["coindesk"]["consecutive_failures"] == 1
        assert nf._feed_failures["coindesk"]["skip_until_cycle"] == 1 + 2  # 3

        # Cycle 2: coindesk should be skipped (cycle_count=2 < skip_until=3)
        call_count["coindesk"] = 0
        nf.fetch_all_rss()
        assert call_count["coindesk"] == 0  # was skipped

        # Cycle 3: coindesk should be retried (cycle_count=3 >= skip_until=3)
        call_count["coindesk"] = 0
        nf.fetch_all_rss()
        assert call_count["coindesk"] == 1  # was retried (and failed again)
        assert nf._feed_failures["coindesk"]["consecutive_failures"] == 2
        # skip_until = 3 + 2^2 = 7
        assert nf._feed_failures["coindesk"]["skip_until_cycle"] == 3 + 4

    @patch("tools.news_fetcher.feedparser.parse")
    def test_feed_recovery_resets_backoff(self, mock_parse, nf):
        """Fail then succeed — verify backoff is reset."""
        fail_first = [True]

        def parse_side_effect(url, agent=None):
            if "coindesk" in url and fail_first[0]:
                fail_first[0] = False
                raise ConnectionError("timeout")
            return {"entries": [], "feed": {"title": "Feed"}}

        mock_parse.side_effect = parse_side_effect

        # Cycle 1: coindesk fails
        nf.fetch_all_rss()
        assert "coindesk" in nf._feed_failures

        # Advance past backoff period
        nf._feed_failures["coindesk"]["skip_until_cycle"] = 0  # force allow

        # Cycle 2: coindesk succeeds — failures should be reset
        nf.fetch_all_rss()
        assert "coindesk" not in nf._feed_failures


class TestStaleFeedWarning:
    @patch("tools.news_fetcher.feedparser.parse")
    def test_stale_feed_warning(self, mock_parse, nf, caplog):
        """Set old timestamp, verify warning logged."""
        mock_parse.return_value = {"entries": [], "feed": {"title": "Feed"}}

        # Simulate past history: coindesk had articles 25 hours ago
        old_time = (datetime.utcnow() - timedelta(hours=25)).isoformat()
        nf._feed_last_new_article["coindesk"] = old_time

        # Set cycle > 1 so stale detection runs
        nf._cycle_count = 1

        with caplog.at_level(logging.WARNING):
            nf.fetch_all_rss()

        assert any("STALE FEED: coindesk" in msg for msg in caplog.messages)


class TestSeenArticlesPersistence:
    @patch.object(NewsFetcher, "fetch_all_rss", return_value=[
        {"title": "Unique Story", "summary": "", "link": "http://unique.com", "published": "", "source": ""},
    ])
    @patch.object(NewsFetcher, "fetch_alpha_vantage_news", return_value=[])
    @patch.object(NewsFetcher, "fetch_crypto_news", return_value=[])
    def test_seen_articles_persisted(self, mock_cc, mock_av, mock_rss, tmp_path):
        """Verify articles are persisted to disk and filtered on second call."""
        seen_file = tmp_path / "seen_articles.json"

        with patch("tools.news_fetcher.SEEN_ARTICLES_FILE", str(seen_file)):
            nf1 = NewsFetcher(inter_feed_delay=0)
            articles1 = nf1.fetch_all()
            assert len(articles1) == 1

            # File should exist now
            assert seen_file.exists()
            data = json.loads(seen_file.read_text())
            assert "http://unique.com" in data

            # Second fetcher loads from file — same article should be filtered
            nf2 = NewsFetcher(inter_feed_delay=0)
            articles2 = nf2.fetch_all()
            assert len(articles2) == 0

    def test_seen_articles_ttl_cleanup(self, tmp_path):
        """Inject old entry, verify cleanup removes it."""
        seen_file = tmp_path / "seen_articles.json"
        old_entry = {
            "http://old.com": {
                "title": "Old Article",
                "first_seen": (datetime.utcnow() - timedelta(hours=49)).isoformat(),
            },
            "http://new.com": {
                "title": "New Article",
                "first_seen": datetime.utcnow().isoformat(),
            },
        }
        seen_file.write_text(json.dumps(old_entry))

        with patch("tools.news_fetcher.SEEN_ARTICLES_FILE", str(seen_file)):
            nf = NewsFetcher(inter_feed_delay=0)
            assert len(nf._seen_articles) == 2

            nf._cleanup_expired_articles()
            assert "http://old.com" not in nf._seen_articles
            assert "http://new.com" in nf._seen_articles
