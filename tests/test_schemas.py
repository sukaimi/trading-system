"""Tests for Pydantic v2 schemas — validation, serialization, defaults."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from core.schemas import (
    Asset,
    CircuitBreakerAction,
    CircuitBreakerDecision,
    ConfirmingSignal,
    ConfirmingSignals,
    DevilsChallenges,
    DevilsVerdict,
    Direction,
    ExecutionOrder,
    HeartbeatStatus,
    JournalEntry,
    OptimizationApplied,
    OptimizationChange,
    OrderConfirmation,
    OrderError,
    OrderStatus,
    ParameterChange,
    Sentiment,
    SignalAlert,
    SignalCategory,
    TradeThesis,
    Urgency,
    Verdict,
    WeeklyDirective,
)


class TestSignalAlert:
    def test_valid_signal(self):
        alert = SignalAlert(
            asset=Asset.BTC,
            signal_strength=0.85,
            headline="Fed signals rate pause",
            sentiment=Sentiment.BULLISH,
            category=SignalCategory.CENTRAL_BANK,
            new_information="First explicit pause signal from Chair",
            urgency=Urgency.HIGH,
            confidence_in_classification=0.9,
        )
        assert alert.type == "signal_alert"
        assert alert.asset == Asset.BTC
        assert alert.signal_strength == 0.85
        assert not alert.already_priced_in

    def test_signal_strength_bounds(self):
        with pytest.raises(ValidationError):
            SignalAlert(
                asset=Asset.BTC,
                signal_strength=1.5,
                headline="Too strong",
                sentiment=Sentiment.BULLISH,
                category=SignalCategory.MACRO,
                new_information="test",
                urgency=Urgency.LOW,
                confidence_in_classification=0.5,
            )

    def test_headline_max_length(self):
        with pytest.raises(ValidationError):
            SignalAlert(
                asset=Asset.ETH,
                signal_strength=0.5,
                headline="x" * 101,
                sentiment=Sentiment.NEUTRAL,
                category=SignalCategory.CRYPTO_SPECIFIC,
                new_information="test",
                urgency=Urgency.MEDIUM,
                confidence_in_classification=0.5,
            )

    def test_json_round_trip(self):
        alert = SignalAlert(
            asset=Asset.GLDM,
            signal_strength=0.6,
            headline="Gold rally",
            sentiment=Sentiment.BULLISH,
            category=SignalCategory.PRECIOUS_METALS,
            new_information="Safe haven flows",
            urgency=Urgency.MEDIUM,
            confidence_in_classification=0.7,
        )
        json_str = alert.model_dump_json()
        restored = SignalAlert.model_validate_json(json_str)
        assert restored.asset == alert.asset
        assert restored.signal_strength == alert.signal_strength


class TestTradeThesis:
    def test_valid_thesis(self):
        thesis = TradeThesis(
            asset=Asset.BTC,
            direction=Direction.LONG,
            confidence=0.72,
            thesis="Fed pause + RSI oversold + gold confirming",
            suggested_position_pct=3.5,
            confirming_signals=ConfirmingSignals(
                fundamental=ConfirmingSignal(present=True, description="Fed pause"),
                technical=ConfirmingSignal(present=True, description="RSI oversold"),
            ),
        )
        assert thesis.type == "trade_thesis"
        assert thesis.confidence == 0.72
        assert thesis.confirming_signals.fundamental.present

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            TradeThesis(
                asset=Asset.ETH,
                direction=Direction.SHORT,
                confidence=-0.1,
                thesis="test",
                suggested_position_pct=1.0,
            )


class TestDevilsVerdict:
    def test_approved_verdict(self):
        verdict = DevilsVerdict(
            original_thesis_id="thesis_001",
            verdict=Verdict.APPROVED,
            flags_raised=1,
            confidence_adjusted=0.7,
            final_reasoning="Thesis is sound",
        )
        assert verdict.verdict == Verdict.APPROVED
        assert verdict.flags_raised == 1

    def test_killed_verdict(self):
        verdict = DevilsVerdict(
            original_thesis_id="thesis_002",
            verdict=Verdict.KILLED,
            flags_raised=5,
            fatal_flaws=["Position would breach daily loss limit"],
            confidence_adjusted=0.2,
            final_reasoning="Too many red flags",
        )
        assert verdict.verdict == Verdict.KILLED
        assert len(verdict.fatal_flaws) == 1


class TestExecutionOrder:
    def test_valid_order(self):
        order = ExecutionOrder(
            thesis_id="thesis_001",
            asset=Asset.ETH,
            direction=Direction.LONG,
            quantity=0.01,
            stop_loss=3000.0,
            take_profit=3500.0,
            position_size_pct=5.0,
        )
        assert order.type == "execution_order"
        assert order.asset == Asset.ETH
        assert order.order_type == "market"

    def test_position_size_bounds(self):
        with pytest.raises(ValidationError):
            ExecutionOrder(
                thesis_id="test",
                asset=Asset.BTC,
                direction=Direction.LONG,
                quantity=1,
                position_size_pct=150.0,
            )


class TestOrderConfirmation:
    def test_filled_order(self):
        conf = OrderConfirmation(
            order_id=12345,
            asset=Asset.BTC,
            direction=Direction.LONG,
            quantity=0.001,
            fill_price=62500.0,
            status=OrderStatus.FILLED,
            thesis_id="thesis_001",
        )
        assert conf.status == OrderStatus.FILLED
        assert conf.fill_price == 62500.0


class TestOrderError:
    def test_error(self):
        err = OrderError(error="Connection refused", thesis_id="thesis_001")
        assert err.type == "order_error"
        assert "Connection" in err.error


class TestJournalEntry:
    def test_open_trade(self):
        entry = JournalEntry(
            trade_id="trade_20260301_143500",
            asset=Asset.BTC,
            direction=Direction.LONG,
            entry_price=62500.0,
            position_size_usd=3.50,
            position_size_pct=3.5,
            stop_loss_price=61250.0,
            thesis_summary="Fed pause + RSI oversold",
        )
        assert entry.exit_price is None
        assert entry.outcome.pnl_usd is None

    def test_closed_trade(self):
        entry = JournalEntry(
            trade_id="trade_20260301_143500",
            asset=Asset.BTC,
            direction=Direction.LONG,
            entry_price=62500.0,
            exit_price=64800.0,
        )
        assert entry.exit_price == 64800.0


class TestWeeklyDirective:
    def test_directive_with_changes(self):
        directive = WeeklyDirective(
            week_reviewed="2026-03-01 to 2026-03-07",
            parameter_changes=[
                ParameterChange(
                    target="news_scout",
                    parameter="signal_weights.regulatory_crypto",
                    old_value=0.8,
                    new_value=0.6,
                    reason="3 false positives",
                )
            ],
            next_week_focus=["Watch FOMC minutes"],
        )
        assert len(directive.parameter_changes) == 1
        assert directive.parameter_changes[0].new_value == 0.6


class TestCircuitBreakerDecision:
    def test_close_losing(self):
        decision = CircuitBreakerDecision(
            triggers_fired=["daily_loss_limit"],
            decision=CircuitBreakerAction.CLOSE_LOSING,
            positions_to_close=["trade_001"],
            reasoning="Cut losers, keep winners",
            resume_conditions="Resume tomorrow if VIX < 25",
        )
        assert decision.decision == CircuitBreakerAction.CLOSE_LOSING
        assert decision.notify_owner is True


class TestHeartbeatStatus:
    def test_healthy(self):
        status = HeartbeatStatus(
            checks={"cpu": True, "ram": True, "disk": True},
            all_healthy=True,
        )
        assert status.all_healthy
        assert len(status.failures) == 0

    def test_unhealthy(self):
        status = HeartbeatStatus(
            checks={"cpu": True, "ram": False},
            all_healthy=False,
            failures=["ram"],
        )
        assert not status.all_healthy
        assert "ram" in status.failures


class TestOptimizationApplied:
    def test_optimization_record(self):
        opt = OptimizationApplied(
            version=4,
            directive_week="2026-03-01 to 2026-03-07",
            changes=[
                OptimizationChange(
                    agent="news_scout",
                    parameter="signal_weights.regulatory_crypto",
                    old_value=0.8,
                    new_value=0.6,
                    reason="3 false positives",
                    version=4,
                )
            ],
            portfolio_value_at_change=103.50,
        )
        assert opt.version == 4
        assert len(opt.changes) == 1
