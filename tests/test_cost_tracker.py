"""Tests for core/cost_tracker.py"""

from unittest.mock import patch

import pytest

from core.cost_tracker import CostTracker


@pytest.fixture
def tracker(tmp_path):
    with patch("core.cost_tracker.STATE_FILE", str(tmp_path / "cost.json")):
        return CostTracker()


class TestCheckBudget:
    def test_under_limit_returns_true(self, tracker):
        assert tracker.check_budget("anthropic") is True

    def test_over_limit_returns_false(self, tracker):
        # Record enough calls to exceed $0.15 daily limit for anthropic
        # Anthropic pricing: $15/M input + $75/M output
        # A ~2700 char input + ~2700 char output ≈ 675 tokens each
        # Cost ≈ 675*15/1M + 675*75/1M ≈ $0.06
        # Need 3 calls to exceed $0.15
        for _ in range(5):
            tracker.record("anthropic", "test", "x" * 3000, "y" * 3000)

        assert tracker.check_budget("anthropic") is False

    def test_different_provider_independent(self, tracker):
        # Exhaust anthropic budget
        for _ in range(5):
            tracker.record("anthropic", "test", "x" * 3000, "y" * 3000)

        # deepseek should still be under budget
        assert tracker.check_budget("deepseek") is True

    def test_unknown_provider_uses_default_limit(self, tracker):
        # Unknown provider defaults to $1.00 limit
        assert tracker.check_budget("unknown_provider") is True

    def test_only_counts_today(self, tracker):
        # Record a call then tamper with timestamp to be yesterday
        tracker.record("anthropic", "test", "x" * 3000, "y" * 3000)
        # Manually change all timestamps to yesterday
        for call in tracker._calls:
            call["timestamp"] = "2020-01-01T00:00:00+00:00"

        # Budget should be available since today has $0 spend
        assert tracker.check_budget("anthropic") is True


class TestRecord:
    def test_records_call(self, tracker):
        record = tracker.record("deepseek", "news_scout", "input text", "output text")
        assert record["provider"] == "deepseek"
        assert record["agent"] == "news_scout"
        assert record["cost_usd"] > 0

    def test_summary_reflects_calls(self, tracker):
        tracker.record("deepseek", "pipeline", "x" * 1000, "y" * 1000)
        s = tracker.summary()
        assert s["call_count"] == 1
        assert s["total_usd"] > 0
        assert "deepseek" in s["by_provider"]
