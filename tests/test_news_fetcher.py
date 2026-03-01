"""Tests for tools/news_fetcher.py — all external calls mocked."""

from unittest.mock import MagicMock, patch

import pytest

from tools.news_fetcher import NewsFetcher


@pytest.fixture
def nf():
    return NewsFetcher()


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
        # 3 feeds return same headline — should be deduped to 1
        assert len(articles) == 1


class TestFetchAlphaVantage:
    def test_no_api_key(self, nf):
        nf._av_key = ""
        assert nf.fetch_alpha_vantage_news() == []

    @patch("tools.news_fetcher.requests.get")
    def test_success(self, mock_get, nf):
        nf._av_key = "test-key"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "feed": [
                {"title": "Fed holds rates", "summary": "...", "url": "http://c.com", "time_published": "2026-03-01", "source": "Reuters"},
            ]
        }
        mock_get.return_value = mock_resp
        articles = nf.fetch_alpha_vantage_news(["BTC"])
        assert len(articles) == 1


class TestFetchCryptoNews:
    def test_no_api_key(self, nf):
        nf._cc_key = ""
        assert nf.fetch_crypto_news() == []

    @patch("tools.news_fetcher.requests.get")
    def test_success(self, mock_get, nf):
        nf._cc_key = "test-key"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Data": [
                {"title": "Crypto rally", "body": "...", "url": "http://d.com", "published_on": 1740000000, "source": "cc"},
            ]
        }
        mock_get.return_value = mock_resp
        articles = nf.fetch_crypto_news()
        assert len(articles) == 1


class TestFetchFedCalendar:
    def test_returns_future_dates(self, nf):
        dates = nf.fetch_fed_calendar()
        assert isinstance(dates, list)
        assert all("date" in d for d in dates)


class TestFetchAll:
    @patch.object(NewsFetcher, "fetch_all_rss", return_value=[
        {"title": "A", "summary": "", "link": "", "published": "", "source": ""},
    ])
    @patch.object(NewsFetcher, "fetch_alpha_vantage_news", return_value=[
        {"title": "B", "summary": "", "link": "", "published": "", "source": ""},
    ])
    @patch.object(NewsFetcher, "fetch_crypto_news", return_value=[
        {"title": "A", "summary": "", "link": "", "published": "", "source": ""},
    ])
    def test_dedup_across_sources(self, mock_cc, mock_av, mock_rss, nf):
        articles = nf.fetch_all()
        assert len(articles) == 2  # "A" deduped
