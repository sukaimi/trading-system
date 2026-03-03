"""Tests for agents/devils_advocate.py"""

from unittest.mock import MagicMock

import pytest

from agents.devils_advocate import DevilsAdvocate
from core.llm_client import LLMClient
from core.schemas import (
    ConfirmingSignal,
    ConfirmingSignals,
    DevilsVerdict,
    Direction,
    TradeThesis,
    Verdict,
)


@pytest.fixture
def mock_llm():
    return LLMClient(mock_mode=True)


@pytest.fixture
def devil(mock_llm):
    return DevilsAdvocate(llm_client=mock_llm)


@pytest.fixture
def valid_thesis():
    return TradeThesis(
        asset="BTC",
        direction=Direction.LONG,
        confidence=0.75,
        thesis="BTC breaking out on strong volume",
        confirming_signals=ConfirmingSignals(
            fundamental=ConfirmingSignal(present=True, description="ETF inflows"),
            technical=ConfirmingSignal(present=True, description="RSI divergence"),
            cross_asset=ConfirmingSignal(present=False),
        ),
        invalidation_level="62000",
        suggested_position_pct=5.0,
    )


@pytest.fixture
def weak_thesis():
    """Thesis with only 1 confirming signal — should be flagged."""
    return TradeThesis(
        asset="ETH",
        direction=Direction.LONG,
        confidence=0.6,
        thesis="ETH could go up",
        confirming_signals=ConfirmingSignals(
            fundamental=ConfirmingSignal(present=True, description="x"),
            technical=ConfirmingSignal(present=False),
            cross_asset=ConfirmingSignal(present=False),
        ),
        invalidation_level="3000",
        suggested_position_pct=3.0,
    )


@pytest.fixture
def healthy_portfolio():
    return {
        "equity": 100.0,
        "daily_pnl_pct": 0.5,
        "open_positions": [],
        "drawdown_from_peak_pct": 2.0,
    }


class TestCheckFatalFlaws:
    def test_no_flaws(self, devil, valid_thesis, healthy_portfolio):
        flaws = devil.check_fatal_flaws(valid_thesis, healthy_portfolio)
        assert flaws == []

    def test_daily_loss_breach(self, devil, valid_thesis, healthy_portfolio):
        healthy_portfolio["daily_pnl_pct"] = -4.5
        flaws = devil.check_fatal_flaws(valid_thesis, healthy_portfolio)
        assert any("daily loss" in f.lower() for f in flaws)

    def test_duplicate_asset(self, devil, valid_thesis, healthy_portfolio):
        healthy_portfolio["open_positions"] = [{"asset": "BTC"}]
        flaws = devil.check_fatal_flaws(valid_thesis, healthy_portfolio)
        assert any("duplicate" in f.lower() or "BTC" in f for f in flaws)

    def test_no_invalidation(self, devil, healthy_portfolio):
        thesis = TradeThesis(
            asset="BTC", direction=Direction.LONG, confidence=0.7,
            thesis="test", invalidation_level="",
            confirming_signals=ConfirmingSignals(
                fundamental=ConfirmingSignal(present=True, description="a"),
                technical=ConfirmingSignal(present=True, description="b"),
            ),
            suggested_position_pct=3.0,
        )
        flaws = devil.check_fatal_flaws(thesis, healthy_portfolio)
        assert any("invalidation" in f.lower() for f in flaws)

    def test_one_confirming_signal_ok(self, devil, weak_thesis, healthy_portfolio):
        # 1 confirming signal is now allowed (micro position)
        flaws = devil.check_fatal_flaws(weak_thesis, healthy_portfolio)
        assert not any("confirming" in f.lower() for f in flaws)

    def test_zero_confirming_signals_fatal(self, devil, healthy_portfolio):
        thesis = TradeThesis(
            asset="ETH", direction=Direction.LONG, confidence=0.5,
            thesis="test", invalidation_level="3000",
            confirming_signals=ConfirmingSignals(
                fundamental=ConfirmingSignal(present=False),
                technical=ConfirmingSignal(present=False),
                cross_asset=ConfirmingSignal(present=False),
            ),
            suggested_position_pct=2.0,
        )
        flaws = devil.check_fatal_flaws(thesis, healthy_portfolio)
        assert any("confirming" in f.lower() for f in flaws)


class TestChallenge:
    def test_weak_thesis_not_killed(self, devil, weak_thesis, healthy_portfolio):
        # weak_thesis has 1 confirming signal — no longer a fatal flaw, proceeds to LLM
        verdict = devil.challenge(weak_thesis, healthy_portfolio)
        assert isinstance(verdict, DevilsVerdict)
        # With 1 confirming signal, should pass fatal flaws and go to LLM challenge
        assert verdict.verdict in (Verdict.APPROVED, Verdict.APPROVED_WITH_MODIFICATION)

    def test_returns_verdict_object(self, devil, valid_thesis, healthy_portfolio):
        verdict = devil.challenge(valid_thesis, healthy_portfolio)
        assert isinstance(verdict, DevilsVerdict)
        assert verdict.verdict in (Verdict.APPROVED, Verdict.APPROVED_WITH_MODIFICATION, Verdict.KILLED)

    def test_approved_valid_thesis(self, devil, valid_thesis, healthy_portfolio):
        # In mock mode, Kimi returns a generic mock → challenges parse with defaults → 0 flags → APPROVED
        verdict = devil.challenge(valid_thesis, healthy_portfolio)
        # Could be APPROVED or APPROVED_WITH_MODIFICATION depending on mock
        assert verdict.verdict != Verdict.KILLED or len(verdict.fatal_flaws) > 0


class TestDetermineVerdict:
    def test_zero_flags_approved(self, devil):
        assert devil._determine_verdict(0, []) == Verdict.APPROVED

    def test_one_flag_approved(self, devil):
        assert devil._determine_verdict(1, []) == Verdict.APPROVED

    def test_two_flags_modified(self, devil):
        assert devil._determine_verdict(2, []) == Verdict.APPROVED_WITH_MODIFICATION

    def test_six_flags_killed(self, devil):
        # Kill threshold is now 6 (raised from 4)
        assert devil._determine_verdict(6, []) == Verdict.KILLED

    def test_four_flags_modified(self, devil):
        # 4 flags is now APPROVED_WITH_MODIFICATION (was KILLED at threshold 4)
        assert devil._determine_verdict(4, []) == Verdict.APPROVED_WITH_MODIFICATION

    def test_fatal_flaws_killed(self, devil):
        assert devil._determine_verdict(0, ["fatal"]) == Verdict.KILLED
