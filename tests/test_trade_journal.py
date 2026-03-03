"""Tests for agents/trade_journal.py"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from agents.trade_journal import TradeJournal
from core.llm_client import LLMClient
from core.schemas import (
    ConfirmingSignal,
    ConfirmingSignals,
    DevilsVerdict,
    Direction,
    JournalEntry,
    OrderConfirmation,
    OrderStatus,
    Sentiment,
    SignalAlert,
    SignalCategory,
    TradeThesis,
    Urgency,
    Verdict,
)


@pytest.fixture
def mock_llm():
    return LLMClient(mock_mode=True)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Redirect journal to a temp directory."""
    data_dir = str(tmp_path / "data")
    journal_file = os.path.join(data_dir, "trade_journal.json")
    return data_dir, journal_file


@pytest.fixture
def journal(mock_llm, tmp_data_dir):
    j = TradeJournal(llm_client=mock_llm)
    # Patch the file paths
    data_dir, journal_file = tmp_data_dir
    with patch("agents.trade_journal.DATA_DIR", data_dir), \
         patch("agents.trade_journal.JOURNAL_FILE", journal_file):
        yield j, journal_file


@pytest.fixture
def sample_thesis():
    return TradeThesis(
        asset="BTC",
        direction=Direction.LONG,
        confidence=0.75,
        thesis="BTC breaking out",
        confirming_signals=ConfirmingSignals(
            fundamental=ConfirmingSignal(present=True, description="ETF inflows"),
            technical=ConfirmingSignal(present=True, description="RSI divergence"),
        ),
        invalidation_level="62000",
        suggested_position_pct=5.0,
    )


@pytest.fixture
def sample_verdict():
    return DevilsVerdict(
        original_thesis_id="test123",
        verdict=Verdict.APPROVED,
        confidence_adjusted=0.7,
        modifications=["Reduce size to 4%"],
    )


@pytest.fixture
def sample_confirmation():
    return OrderConfirmation(
        order_id=42,
        asset="BTC",
        direction=Direction.LONG,
        quantity=0.01,
        fill_price=65000.0,
        status=OrderStatus.FILLED,
    )


class TestRecordEntry:
    def test_returns_journal_entry(self, journal, sample_thesis, sample_verdict, sample_confirmation):
        j, jf = journal
        with patch("agents.trade_journal.DATA_DIR", os.path.dirname(jf)), \
             patch("agents.trade_journal.JOURNAL_FILE", jf):
            entry = j.record_entry(sample_thesis, sample_verdict, sample_confirmation)
            assert isinstance(entry, JournalEntry)
            assert entry.asset == "BTC"
            assert entry.direction == Direction.LONG
            assert entry.entry_price == 65000.0

    def test_persists_to_file(self, journal, sample_thesis, sample_verdict, sample_confirmation):
        j, jf = journal
        with patch("agents.trade_journal.DATA_DIR", os.path.dirname(jf)), \
             patch("agents.trade_journal.JOURNAL_FILE", jf):
            j.record_entry(sample_thesis, sample_verdict, sample_confirmation)
            assert os.path.exists(jf)
            with open(jf) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["asset"] == "BTC"


class TestRecordNoTrade:
    def test_records_no_trade(self, journal):
        j, jf = journal
        with patch("agents.trade_journal.DATA_DIR", os.path.dirname(jf)), \
             patch("agents.trade_journal.JOURNAL_FILE", jf):
            signal = SignalAlert(
                asset="ETH", signal_strength=0.5, headline="Test",
                sentiment=Sentiment.NEUTRAL, category=SignalCategory.MACRO,
                new_information="", urgency=Urgency.LOW,
                confidence_in_classification=0.5,
            )
            record = j.record_no_trade(signal, "Low confidence")
            assert record["type"] == "no_trade"
            assert record["asset"] == "ETH"


class TestRecordKilledTrade:
    def test_records_killed(self, journal, sample_thesis):
        j, jf = journal
        with patch("agents.trade_journal.DATA_DIR", os.path.dirname(jf)), \
             patch("agents.trade_journal.JOURNAL_FILE", jf):
            verdict = DevilsVerdict(
                original_thesis_id="x", verdict=Verdict.KILLED,
                confidence_adjusted=0.0, fatal_flaws=["no edge"],
                flags_raised=5, final_reasoning="Too risky",
            )
            record = j.record_killed_trade(sample_thesis, verdict)
            assert record["type"] == "killed_trade"
            assert record["verdict"] == "KILLED"


class TestRecordExit:
    def test_exit_updates_entry(self, journal, sample_thesis, sample_verdict, sample_confirmation):
        j, jf = journal
        with patch("agents.trade_journal.DATA_DIR", os.path.dirname(jf)), \
             patch("agents.trade_journal.JOURNAL_FILE", jf):
            entry = j.record_entry(sample_thesis, sample_verdict, sample_confirmation)
            result = j.record_exit(entry.trade_id, {
                "exit_price": 67000.0,
                "pnl_usd": 20.0,
                "pnl_pct": 3.0,
                "exit_reason": "take_profit",
                "hold_duration_hours": 24.0,
            })
            assert result is not None
            assert result.exit_price == 67000.0

    def test_exit_not_found(self, journal):
        j, jf = journal
        with patch("agents.trade_journal.DATA_DIR", os.path.dirname(jf)), \
             patch("agents.trade_journal.JOURNAL_FILE", jf):
            result = j.record_exit("nonexistent", {})
            assert result is None


class TestWeeklyPackage:
    def test_empty_journal(self, journal):
        j, jf = journal
        with patch("agents.trade_journal.DATA_DIR", os.path.dirname(jf)), \
             patch("agents.trade_journal.JOURNAL_FILE", jf):
            pkg = j.assemble_weekly_package("2026-12-31")
            assert pkg["trade_summary"]["total_trades"] == 0
            assert pkg["trade_summary"]["win_rate"] == 0.0
