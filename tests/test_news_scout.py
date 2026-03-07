"""Tests for agents/news_scout.py"""

from unittest.mock import MagicMock, patch

import pytest

from agents.news_scout import NewsScout
from core.llm_client import LLMClient
from core.schemas import SignalAlert


@pytest.fixture
def mock_llm():
    return LLMClient(mock_mode=True)


@pytest.fixture
def mock_fetcher():
    fetcher = MagicMock()
    fetcher.fetch_all.return_value = [
        {"title": "Fed cuts rates by 50bp", "summary": "Surprise cut", "source": "Reuters", "published": "2026-03-01", "link": ""},
        {"title": "BTC breaks 100k", "summary": "New ATH", "source": "CoinDesk", "published": "2026-03-01", "link": ""},
    ]
    return fetcher


@pytest.fixture
def scout(mock_llm, mock_fetcher):
    return NewsScout(llm_client=mock_llm, news_fetcher=mock_fetcher)


class TestScan:
    def test_scan_returns_list(self, scout):
        result = scout.scan()
        assert isinstance(result, list)

    def test_scan_no_articles(self, mock_llm):
        fetcher = MagicMock()
        fetcher.fetch_all.return_value = []
        scout = NewsScout(llm_client=mock_llm, news_fetcher=fetcher)
        assert scout.scan() == []


class TestClassifyArticles:
    def test_llm_returns_list(self, scout):
        articles = [{"title": "Test", "summary": "s", "source": "r", "published": "", "link": ""}]
        result = scout.classify_articles(articles)
        assert isinstance(result, list)

    @patch.object(LLMClient, "call_deepseek")
    def test_llm_error(self, mock_ds, mock_fetcher):
        mock_ds.return_value = {"error": "timeout"}
        scout = NewsScout(llm_client=LLMClient(mock_mode=False), news_fetcher=mock_fetcher)
        result = scout.classify_articles([{"title": "T"}])
        assert result == []


class TestApplyFilters:
    def test_below_threshold_filtered(self, scout):
        raw = [{"signal_strength": 0.2, "headline": "Weak signal", "asset": "BTC", "sentiment": "neutral", "category": "macro", "new_information": "", "urgency": "low"}]
        result = scout._apply_filters(raw)
        assert len(result) == 0

    @patch("agents.news_scout.datetime")
    def test_priced_in_penalty(self, mock_dt, scout):
        # Mock to a weekday to avoid weekend penalty
        from datetime import datetime
        fake_now = datetime(2026, 3, 2, 12, 0, 0)  # Monday
        mock_dt.utcnow.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        raw = [{"signal_strength": 0.7, "headline": "Old news", "asset": "BTC", "sentiment": "bullish", "category": "macro", "new_information": "something", "urgency": "medium", "already_priced_in": True}]
        result = scout._apply_filters(raw)
        # 0.7 - 0.2 (priced in) = 0.5 — above default threshold 0.4
        assert len(result) == 1
        assert result[0].signal_strength == pytest.approx(0.5, abs=0.01)

    @patch("agents.news_scout.datetime")
    def test_speculation_penalty(self, mock_dt, scout):
        # Mock to a weekday to avoid weekend penalty
        from datetime import datetime
        fake_now = datetime(2026, 3, 2, 12, 0, 0)  # Monday
        mock_dt.utcnow.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        raw = [{"signal_strength": 0.8, "headline": "BTC could rally", "asset": "BTC", "sentiment": "bullish", "category": "crypto_specific", "new_information": "might go up", "urgency": "medium"}]
        result = scout._apply_filters(raw)
        # 0.8 - 0.3 (could/might) = 0.5 — above threshold
        assert len(result) == 1
        assert result[0].signal_strength == pytest.approx(0.5, abs=0.01)

    def test_dedup(self, scout):
        raw = [
            {"signal_strength": 0.8, "headline": "Fed cuts", "asset": "MACRO", "sentiment": "bullish", "category": "central_bank", "new_information": "50bp", "urgency": "critical"},
            {"signal_strength": 0.8, "headline": "Fed cuts", "asset": "MACRO", "sentiment": "bullish", "category": "central_bank", "new_information": "50bp", "urgency": "critical"},
        ]
        result = scout._apply_filters(raw)
        assert len(result) == 1

    def test_max_alerts_per_hour(self, scout):
        scout._params = {"min_signal_threshold": 0.3, "max_alerts_per_hour": 2, "weekend_signal_penalty": 0.15}
        raw = [
            {"signal_strength": 0.9, "headline": f"Signal {i}", "asset": "BTC", "sentiment": "bullish", "category": "macro", "new_information": "new", "urgency": "high"}
            for i in range(5)
        ]
        result = scout._apply_filters(raw)
        assert len(result) <= 2
