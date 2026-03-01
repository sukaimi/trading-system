"""Tests for agents/weekly_strategist.py"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from agents.weekly_strategist import WeeklyStrategist
from core.llm_client import LLMClient
from core.schemas import WeeklyDirective
from core.self_optimizer import SelfOptimizer


@pytest.fixture
def mock_llm():
    return LLMClient(mock_mode=True)


@pytest.fixture
def mock_optimizer():
    return MagicMock(spec=SelfOptimizer)


@pytest.fixture
def mock_telegram():
    return MagicMock()


@pytest.fixture
def strategist(mock_llm, mock_optimizer, mock_telegram):
    return WeeklyStrategist(
        llm_client=mock_llm,
        optimizer=mock_optimizer,
        telegram=mock_telegram,
    )


@pytest.fixture
def sample_weekly_package():
    return {
        "week_ending": "2026-03-01",
        "trade_summary": {
            "total_trades": 5,
            "wins": 3,
            "losses": 2,
            "no_trades": 10,
            "killed_trades": 3,
            "win_rate": 0.6,
            "avg_win_pct": 2.5,
            "avg_loss_pct": -1.2,
        },
        "portfolio_summary": {
            "equity": 105.0,
            "initial_capital": 100.0,
            "return_pct": 5.0,
            "drawdown_from_peak_pct": 2.0,
            "open_positions": 1,
        },
        "current_strategy_params": {"news_scout": {"min_signal_threshold": 0.4}},
        "current_risk_params": {"max_position_pct": 7.0},
        "full_trade_journal": [],
    }


class TestReviewWeek:
    def test_returns_weekly_directive(self, strategist, sample_weekly_package):
        result = strategist.review_week(sample_weekly_package)
        assert isinstance(result, WeeklyDirective)
        assert result.week_reviewed == "2026-03-01"

    def test_calls_optimizer_when_changes(self, strategist, sample_weekly_package, mock_optimizer):
        # Mock LLM to return directive with changes
        with patch.object(LLMClient, "call_anthropic") as mock_call:
            mock_call.return_value = {
                "week_reviewed": "2026-03-01",
                "assessment": {"overall": "Good week"},
                "parameter_changes": [
                    {"target": "news_scout", "parameter": "min_signal_threshold",
                     "old_value": 0.4, "new_value": 0.35, "reason": "Too many missed signals"}
                ],
                "next_week_focus": ["Monitor Fed"],
                "risk_adjustments": {},
            }
            strategist._llm = LLMClient(mock_mode=False)
            strategist.review_week(sample_weekly_package)
            mock_optimizer.apply_directives.assert_called_once()

    def test_sends_telegram_report(self, strategist, sample_weekly_package, mock_telegram):
        strategist.review_week(sample_weekly_package)
        mock_telegram.send_weekly_report.assert_called_once()

    def test_saves_directive_to_file(self, strategist, sample_weekly_package, tmp_path):
        with patch("agents.weekly_strategist.REVIEWS_DIR", str(tmp_path)):
            strategist.review_week(sample_weekly_package)
            files = list(tmp_path.iterdir())
            assert len(files) == 1
            assert files[0].suffix == ".json"

    def test_handles_opus_error(self, strategist, sample_weekly_package):
        with patch.object(LLMClient, "call_anthropic") as mock_call:
            mock_call.return_value = {"error": "API timeout"}
            strategist._llm = LLMClient(mock_mode=False)
            result = strategist.review_week(sample_weekly_package)
            assert isinstance(result, WeeklyDirective)
            assert "failed" in result.assessment.get("overall", "").lower()


class TestAssessRegime:
    def test_risk_off(self, strategist):
        assert strategist.assess_regime({"vix": 30, "gold_change_7d": 2.0}) == "risk_off"

    def test_risk_on(self, strategist):
        assert strategist.assess_regime({"vix": 15, "btc_change_7d": 5.0}) == "risk_on"

    def test_transitional(self, strategist):
        assert strategist.assess_regime({"vix": 22}) == "transitional"

    def test_unknown(self, strategist):
        assert strategist.assess_regime({}) == "unknown"
