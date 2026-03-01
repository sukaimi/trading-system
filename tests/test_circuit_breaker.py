"""Tests for agents/circuit_breaker_agent.py"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from agents.circuit_breaker_agent import CircuitBreakerAgent
from core.llm_client import LLMClient
from core.portfolio import PortfolioState
from core.schemas import CircuitBreakerAction, CircuitBreakerDecision


@pytest.fixture
def mock_llm():
    return LLMClient(mock_mode=True)


@pytest.fixture
def mock_portfolio():
    p = PortfolioState()
    p.add_position({
        "trade_id": "t1", "asset": "BTC", "direction": "long",
        "entry_price": 65000, "position_size_pct": 5.0,
    })
    return p


@pytest.fixture
def mock_executor():
    return MagicMock()


@pytest.fixture
def mock_telegram():
    return MagicMock()


@pytest.fixture
def cb(mock_llm, mock_executor, mock_portfolio, mock_telegram):
    return CircuitBreakerAgent(
        llm_client=mock_llm,
        executor=mock_executor,
        portfolio=mock_portfolio,
        telegram=mock_telegram,
    )


@pytest.fixture
def healthy_state():
    return {
        "daily_pnl_pct": 1.0,
        "drawdown_from_peak_pct": 3.0,
        "consecutive_losses": 0,
        "open_positions": [],
    }


@pytest.fixture
def crisis_state():
    return {
        "daily_pnl_pct": -6.0,
        "drawdown_from_peak_pct": 16.0,
        "consecutive_losses": 6,
        "open_positions": [
            {"trade_id": "t1", "asset": "BTC", "direction": "long", "pnl": -500},
        ],
    }


class TestCheck:
    def test_no_triggers(self, cb, healthy_state):
        result = cb.check(healthy_state, {"vix": 20})
        assert result is None

    def test_daily_loss_trigger(self, cb, healthy_state):
        healthy_state["daily_pnl_pct"] = -5.5
        result = cb.check(healthy_state, {})
        assert result is not None
        assert "daily_loss_limit" in result["triggers_fired"]

    def test_max_drawdown_trigger(self, cb, healthy_state):
        healthy_state["drawdown_from_peak_pct"] = 16.0
        result = cb.check(healthy_state, {})
        assert result is not None
        assert "max_drawdown" in result["triggers_fired"]

    def test_losing_streak_trigger(self, cb, healthy_state):
        healthy_state["consecutive_losses"] = 5
        result = cb.check(healthy_state, {})
        assert result is not None
        assert "losing_streak" in result["triggers_fired"]

    def test_vix_extreme_trigger(self, cb, healthy_state):
        result = cb.check(healthy_state, {"vix": 40})
        assert result is not None
        assert "vix_extreme" in result["triggers_fired"]

    def test_flash_crash_trigger(self, cb, healthy_state):
        result = cb.check(healthy_state, {"flash_crash_detected": True})
        assert result is not None
        assert "flash_crash" in result["triggers_fired"]

    def test_api_failure_trigger(self, cb, healthy_state):
        result = cb.check(healthy_state, {"api_failure_count": 4})
        assert result is not None
        assert "api_failures" in result["triggers_fired"]

    def test_multiple_triggers(self, cb, crisis_state):
        result = cb.check(crisis_state, {"vix": 40})
        assert result is not None
        assert len(result["triggers_fired"]) >= 3


class TestEscalateToOpus:
    def test_halts_portfolio(self, cb, crisis_state, mock_portfolio):
        decision = cb.escalate_to_opus(["daily_loss_limit"], crisis_state, {})
        assert mock_portfolio.halted is True

    def test_returns_decision(self, cb, crisis_state):
        decision = cb.escalate_to_opus(["daily_loss_limit"], crisis_state, {})
        assert isinstance(decision, CircuitBreakerDecision)
        assert decision.decision in list(CircuitBreakerAction)

    def test_handles_opus_error(self, cb, crisis_state):
        with patch.object(LLMClient, "call_anthropic") as mock_call:
            mock_call.return_value = {"error": "timeout"}
            cb._llm = LLMClient(mock_mode=False)
            decision = cb.escalate_to_opus(["daily_loss_limit"], crisis_state, {})
            assert decision.decision == CircuitBreakerAction.HOLD
            assert "unavailable" in decision.reasoning.lower()


class TestExecuteDecision:
    def test_hold_no_position_changes(self, cb, mock_executor):
        decision = CircuitBreakerDecision(
            triggers_fired=["losing_streak"],
            decision=CircuitBreakerAction.HOLD,
            reasoning="Variance, not broken",
            resume_conditions="Manual review",
            telegram_message="HOLD",
        )
        cb.execute_decision(decision)
        mock_executor.execute.assert_not_called()

    def test_close_all_calls_executor(self, cb, mock_executor, mock_portfolio):
        mock_executor.execute.return_value = {"type": "order_confirmation"}
        decision = CircuitBreakerDecision(
            triggers_fired=["max_drawdown"],
            decision=CircuitBreakerAction.CLOSE_ALL,
            reasoning="Max drawdown",
            resume_conditions="Manual review",
            telegram_message="CLOSE ALL",
        )
        cb.execute_decision(decision)
        # Should have tried to close the 1 position
        assert mock_executor.execute.called

    def test_sends_telegram(self, cb, mock_telegram):
        decision = CircuitBreakerDecision(
            triggers_fired=["vix_extreme"],
            decision=CircuitBreakerAction.HOLD,
            reasoning="VIX spike",
            resume_conditions="VIX < 30",
            telegram_message="VIX extreme",
        )
        cb.execute_decision(decision)
        mock_telegram.send_circuit_breaker_alert.assert_called_once()

    def test_logs_decision(self, cb, tmp_path):
        with patch("agents.circuit_breaker_agent.CB_LOG_FILE", str(tmp_path / "cb.json")), \
             patch("agents.circuit_breaker_agent.DATA_DIR", str(tmp_path)):
            decision = CircuitBreakerDecision(
                triggers_fired=["test"],
                decision=CircuitBreakerAction.HOLD,
                reasoning="test",
                resume_conditions="test",
                telegram_message="test",
            )
            cb.execute_decision(decision)
            assert os.path.exists(tmp_path / "cb.json")
