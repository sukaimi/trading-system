"""Tests for core/signal_tracker.py"""

import json
import os
import tempfile

import pytest

from core.schemas import Sentiment, SignalAlert, SignalCategory, Urgency
from core.signal_tracker import SignalAccuracyTracker


@pytest.fixture
def tmp_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def tracker(tmp_file):
    return SignalAccuracyTracker(data_file=tmp_file)


@pytest.fixture
def btc_signal():
    return SignalAlert(
        asset="BTC",
        signal_id="sig_test_BTC_0001",
        signal_strength=0.8,
        headline="BTC breaks 100k",
        sentiment=Sentiment.BULLISH,
        category=SignalCategory.CRYPTO_SPECIFIC,
        new_information="New ATH",
        urgency=Urgency.HIGH,
        confidence_in_classification=0.8,
    )


@pytest.fixture
def bearish_signal():
    return SignalAlert(
        asset="ETH",
        signal_id="sig_test_ETH_0002",
        signal_strength=0.6,
        headline="ETH whale dump detected",
        sentiment=Sentiment.BEARISH,
        category=SignalCategory.CRYPTO_SPECIFIC,
        new_information="Large sell orders",
        urgency=Urgency.MEDIUM,
        confidence_in_classification=0.7,
    )


class TestRecordSignalBasic:
    def test_record_signal_basic(self, tracker, btc_signal):
        tracker.record_signal("sig_test_BTC_0001", btc_signal, source_type="news_scan")

        summary = tracker.summary()
        assert summary["total_signals"] == 1
        assert len(summary["recent"]) == 1
        assert summary["recent"][0]["asset"] == "BTC"
        assert summary["recent"][0]["signal_id"] == "sig_test_BTC_0001"
        assert summary["recent"][0]["sentiment"] == "bullish"
        assert summary["recent"][0]["source_type"] == "news_scan"


class TestRecordOutcomeExecuted:
    def test_record_outcome_executed(self, tracker, btc_signal):
        tracker.record_signal("sig_test_BTC_0001", btc_signal, source_type="news_scan")
        tracker.record_outcome(
            "sig_test_BTC_0001", "executed",
            trade_id="order_123", direction="long", entry_price=100000.0,
        )

        summary = tracker.summary()
        assert summary["executed"] == 1
        sig = summary["recent"][0]
        assert sig["pipeline_outcome"] == "executed"
        assert sig["trade_id"] == "order_123"
        assert sig["direction"] == "long"
        assert sig["entry_price"] == 100000.0


class TestRecordOutcomeKilled:
    def test_record_outcome_killed(self, tracker, btc_signal):
        tracker.record_signal("sig_test_BTC_0001", btc_signal, source_type="news_scan")
        tracker.record_outcome(
            "sig_test_BTC_0001", "killed",
            killed_by="devils_advocate",
            kill_reason="Crowded trade risk too high",
        )

        summary = tracker.summary()
        sig = summary["recent"][0]
        assert sig["pipeline_outcome"] == "killed"
        assert sig["killed_by"] == "devils_advocate"
        assert sig["kill_reason"] == "Crowded trade risk too high"


class TestRecordTradeClose:
    def test_record_trade_close(self, tracker, btc_signal):
        tracker.record_signal("sig_test_BTC_0001", btc_signal, source_type="news_scan")
        tracker.record_outcome(
            "sig_test_BTC_0001", "executed",
            trade_id="order_123", direction="long", entry_price=100000.0,
        )
        tracker.record_trade_close(
            trade_id="order_123",
            exit_price=105000.0,
            pnl_usd=50.0,
            pnl_pct=5.0,
            exit_reason="take_profit",
            hold_duration_hours=24.0,
        )

        summary = tracker.summary()
        assert summary["closed"] == 1
        sig = summary["recent"][0]
        assert sig["exit_price"] == 105000.0
        assert sig["pnl_usd"] == 50.0
        assert sig["exit_reason"] == "take_profit"
        assert sig["hold_duration_hours"] == 24.0


class TestSignalCorrectBullishProfit:
    def test_signal_correct_bullish_profit(self, tracker, btc_signal):
        tracker.record_signal("sig_test_BTC_0001", btc_signal, source_type="news_scan")
        tracker.record_outcome(
            "sig_test_BTC_0001", "executed",
            trade_id="order_123", direction="long", entry_price=100000.0,
        )
        tracker.record_trade_close(
            trade_id="order_123",
            exit_price=105000.0,
            pnl_usd=50.0,
            pnl_pct=5.0,
            exit_reason="take_profit",
        )

        sig = tracker.summary()["recent"][0]
        assert sig["signal_correct"] is True


class TestSignalCorrectBearishProfit:
    def test_signal_correct_bearish_profit(self, tracker, bearish_signal):
        tracker.record_signal("sig_test_ETH_0002", bearish_signal, source_type="news_scan")
        tracker.record_outcome(
            "sig_test_ETH_0002", "executed",
            trade_id="order_456", direction="short", entry_price=3000.0,
        )
        tracker.record_trade_close(
            trade_id="order_456",
            exit_price=2800.0,
            pnl_usd=20.0,
            pnl_pct=6.67,
            exit_reason="take_profit",
        )

        sig = tracker.summary()["recent"][0]
        assert sig["signal_correct"] is True


class TestSignalCorrectBullishLoss:
    def test_signal_correct_bullish_loss(self, tracker, btc_signal):
        tracker.record_signal("sig_test_BTC_0001", btc_signal, source_type="news_scan")
        tracker.record_outcome(
            "sig_test_BTC_0001", "executed",
            trade_id="order_789", direction="long", entry_price=100000.0,
        )
        tracker.record_trade_close(
            trade_id="order_789",
            exit_price=95000.0,
            pnl_usd=-50.0,
            pnl_pct=-5.0,
            exit_reason="stop_loss",
        )

        sig = tracker.summary()["recent"][0]
        assert sig["signal_correct"] is False


class TestUnknownTradeIdIgnored:
    def test_unknown_trade_id_ignored(self, tracker):
        # Should not crash when closing a trade with no matching signal
        tracker.record_trade_close(
            trade_id="nonexistent_999",
            exit_price=50000.0,
            pnl_usd=-10.0,
            pnl_pct=-1.0,
            exit_reason="stop_loss",
        )
        summary = tracker.summary()
        assert summary["total_signals"] == 0
        assert summary["closed"] == 0


class TestRingBufferCap:
    def test_ring_buffer_cap(self, tracker):
        for i in range(550):
            sig = SignalAlert(
                asset="BTC",
                signal_id=f"sig_cap_{i:04d}",
                signal_strength=0.5,
                headline=f"Signal {i}",
                sentiment=Sentiment.BULLISH,
                category=SignalCategory.CRYPTO_SPECIFIC,
                new_information=f"Info {i}",
                urgency=Urgency.LOW,
                confidence_in_classification=0.5,
            )
            tracker.record_signal(f"sig_cap_{i:04d}", sig, source_type="test")

        summary = tracker.summary()
        assert summary["total_signals"] == 500


class TestPersistAndLoad:
    def test_persist_and_load(self, tmp_file, btc_signal):
        tracker1 = SignalAccuracyTracker(data_file=tmp_file)
        tracker1.record_signal("sig_test_BTC_0001", btc_signal, source_type="news_scan")
        tracker1.record_outcome(
            "sig_test_BTC_0001", "executed",
            trade_id="order_123", direction="long", entry_price=100000.0,
        )

        # Create new instance with same file
        tracker2 = SignalAccuracyTracker(data_file=tmp_file)
        summary = tracker2.summary()
        assert summary["total_signals"] == 1
        assert summary["executed"] == 1
        assert summary["recent"][0]["trade_id"] == "order_123"


class TestSummaryStats:
    def test_summary_stats(self, tracker):
        # Record mixed signals: 2 executed (1 win, 1 loss), 1 killed, 1 no_trade
        signals = [
            ("sig_1", "BTC", Sentiment.BULLISH, SignalCategory.CRYPTO_SPECIFIC, 0.9),
            ("sig_2", "ETH", Sentiment.BEARISH, SignalCategory.CRYPTO_SPECIFIC, 0.6),
            ("sig_3", "AAPL", Sentiment.BULLISH, SignalCategory.EQUITY, 0.4),
            ("sig_4", "SPY", Sentiment.NEUTRAL, SignalCategory.MACRO, 0.3),
        ]

        for sid, asset, sent, cat, strength in signals:
            sig = SignalAlert(
                asset=asset,
                signal_id=sid,
                signal_strength=strength,
                headline=f"Test {sid}",
                sentiment=sent,
                category=cat,
                new_information="test",
                urgency=Urgency.MEDIUM,
                confidence_in_classification=0.5,
            )
            tracker.record_signal(sid, sig, source_type="test")

        # sig_1: executed, won
        tracker.record_outcome("sig_1", "executed", trade_id="t1", direction="long", entry_price=100000.0)
        tracker.record_trade_close("t1", exit_price=105000.0, pnl_usd=50.0, pnl_pct=5.0, exit_reason="take_profit")

        # sig_2: executed, won (bearish + profitable short)
        tracker.record_outcome("sig_2", "executed", trade_id="t2", direction="short", entry_price=3000.0)
        tracker.record_trade_close("t2", exit_price=2800.0, pnl_usd=20.0, pnl_pct=6.67, exit_reason="take_profit")

        # sig_3: killed
        tracker.record_outcome("sig_3", "killed", killed_by="devils_advocate", kill_reason="too risky")

        # sig_4: no_trade
        tracker.record_outcome("sig_4", "no_trade")

        summary = tracker.summary()
        assert summary["total_signals"] == 4
        assert summary["executed"] == 2
        assert summary["closed"] == 2
        assert summary["wins"] == 2
        assert summary["losses"] == 0
        assert summary["win_rate"] == 100.0

        # By category
        assert summary["by_category"]["crypto_specific"]["total"] == 2
        assert summary["by_category"]["crypto_specific"]["executed"] == 2
        assert summary["by_category"]["equity"]["total"] == 1
        assert summary["by_category"]["macro"]["total"] == 1

        # By asset
        assert summary["by_asset"]["BTC"]["wins"] == 1
        assert summary["by_asset"]["ETH"]["wins"] == 1

        # By strength bucket
        assert summary["by_strength_bucket"]["high_0.8+"]["total"] == 1
        assert summary["by_strength_bucket"]["medium_0.5-0.8"]["total"] == 1
        assert summary["by_strength_bucket"]["low_<0.5"]["total"] == 2

        # Outcomes
        assert summary["outcomes"]["executed"] == 2
        assert summary["outcomes"]["killed"] == 1
        assert summary["outcomes"]["no_trade"] == 1
