"""Tests for core/session_analyzer.py — Session Analyzer."""

import json
from unittest.mock import patch

import pytest

from core.session_analyzer import SessionAnalyzer, SESSIONS


# ── Helpers ─────────────────────────────────────────────────────────


def _make_trade(
    asset: str = "BTC",
    pnl_pct: float = 2.0,
    hour: int = 8,
    timestamp_open: str | None = None,
) -> dict:
    if timestamp_open is None:
        timestamp_open = f"2026-01-15T{hour:02d}:30:00+00:00"
    return {
        "trade_id": "t1",
        "asset": asset,
        "direction": "long",
        "entry_price": 100.0,
        "exit_price": 100.0 + pnl_pct,
        "pnl_usd": pnl_pct,
        "pnl_pct": pnl_pct,
        "timestamp_open": timestamp_open,
        "timestamp_close": "2026-01-16T08:00:00+00:00",
        "exit_reason": "take_profit" if pnl_pct > 0 else "stop_loss",
        "mae_pct": -abs(pnl_pct) * 0.3,
        "mfe_pct": abs(pnl_pct) * 1.2,
    }


def _make_session_trades(
    asset: str,
    session: str,
    count: int,
    win_rate: float = 0.6,
    avg_pnl: float = 1.5,
) -> list[dict]:
    """Generate trades in a given session."""
    hour_map = {"asian": 2, "european": 9, "us": 15}
    hour = hour_map[session]
    trades = []
    wins = int(count * win_rate)
    for i in range(count):
        pnl = avg_pnl if i < wins else -abs(avg_pnl) * 0.5
        trades.append(_make_trade(asset=asset, pnl_pct=pnl, hour=hour))
    return trades


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def analyzer():
    return SessionAnalyzer(min_trades_per_session=3)


# ── Tests: Session classification ────────────────────────────────


class TestClassifySession:
    def test_asian_late_night(self, analyzer):
        assert analyzer.classify_session(22) == "asian"
        assert analyzer.classify_session(23) == "asian"

    def test_asian_early_morning(self, analyzer):
        assert analyzer.classify_session(0) == "asian"
        assert analyzer.classify_session(3) == "asian"
        assert analyzer.classify_session(5) == "asian"

    def test_european(self, analyzer):
        assert analyzer.classify_session(6) == "european"
        assert analyzer.classify_session(9) == "european"
        assert analyzer.classify_session(11) == "european"

    def test_us(self, analyzer):
        assert analyzer.classify_session(12) == "us"
        assert analyzer.classify_session(15) == "us"
        assert analyzer.classify_session(21) == "us"

    def test_boundary_asian_european(self, analyzer):
        """Hour 6 is European, hour 5 is Asian."""
        assert analyzer.classify_session(5) == "asian"
        assert analyzer.classify_session(6) == "european"

    def test_boundary_european_us(self, analyzer):
        """Hour 12 is US, hour 11 is European."""
        assert analyzer.classify_session(11) == "european"
        assert analyzer.classify_session(12) == "us"

    def test_boundary_us_asian(self, analyzer):
        """Hour 22 is Asian, hour 21 is US."""
        assert analyzer.classify_session(21) == "us"
        assert analyzer.classify_session(22) == "asian"

    def test_modulo_wrapping(self, analyzer):
        """Hours > 23 should wrap correctly."""
        assert analyzer.classify_session(24) == "asian"  # 24 % 24 = 0
        assert analyzer.classify_session(30) == "european"  # 30 % 24 = 6


# ── Tests: No data ───────────────────────────────────────────────


class TestNoData:
    def test_no_journal_file(self, analyzer):
        with patch("core.session_analyzer.JOURNAL_FILE", "/nonexistent.json"):
            result = analyzer.analyze()
        assert result["sufficient_data"] is False
        assert result["per_asset"] == {}

    def test_empty_journal(self, analyzer, tmp_path):
        journal_file = str(tmp_path / "journal.json")
        with open(journal_file, "w") as f:
            json.dump([], f)
        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            result = analyzer.analyze()
        assert result["sufficient_data"] is False

    def test_get_session_weight_no_data(self, analyzer):
        """With no data, weight should be 1.0 (neutral)."""
        with patch("core.session_analyzer.JOURNAL_FILE", "/nonexistent.json"):
            weight = analyzer.get_session_weight("BTC", "asian")
        assert weight == 1.0

    def test_corrupt_journal(self, analyzer, tmp_path):
        journal_file = str(tmp_path / "journal.json")
        with open(journal_file, "w") as f:
            f.write("{bad json")
        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            result = analyzer.analyze()
        assert result["sufficient_data"] is False


# ── Tests: With data ─────────────────────────────────────────────


class TestWithData:
    def test_session_stats_computed(self, tmp_path):
        analyzer = SessionAnalyzer(min_trades_per_session=3)
        journal_file = str(tmp_path / "journal.json")

        trades = (
            _make_session_trades("BTC", "asian", 5, win_rate=0.8, avg_pnl=3.0)
            + _make_session_trades("BTC", "european", 5, win_rate=0.4, avg_pnl=0.5)
            + _make_session_trades("BTC", "us", 5, win_rate=0.6, avg_pnl=1.5)
        )
        with open(journal_file, "w") as f:
            json.dump(trades, f)

        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            result = analyzer.analyze()

        assert result["sufficient_data"] is True
        btc = result["per_asset"]["BTC"]
        assert btc["asian"]["trades"] == 5
        assert btc["european"]["trades"] == 5
        assert btc["us"]["trades"] == 5
        assert btc["asian"]["win_rate"] >= btc["european"]["win_rate"]

    def test_best_session_selected(self, tmp_path):
        analyzer = SessionAnalyzer(min_trades_per_session=3)
        journal_file = str(tmp_path / "journal.json")

        # Asian has best performance
        trades = (
            _make_session_trades("BTC", "asian", 5, win_rate=0.8, avg_pnl=3.0)
            + _make_session_trades("BTC", "european", 5, win_rate=0.2, avg_pnl=-1.0)
            + _make_session_trades("BTC", "us", 5, win_rate=0.4, avg_pnl=0.5)
        )
        with open(journal_file, "w") as f:
            json.dump(trades, f)

        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            result = analyzer.analyze()

        assert result["per_asset"]["BTC"]["best_session"] == "asian"

    def test_overall_stats(self, tmp_path):
        analyzer = SessionAnalyzer(min_trades_per_session=2)
        journal_file = str(tmp_path / "journal.json")

        trades = (
            _make_session_trades("BTC", "asian", 3, win_rate=0.67, avg_pnl=2.0)
            + _make_session_trades("ETH", "asian", 3, win_rate=0.67, avg_pnl=1.0)
        )
        with open(journal_file, "w") as f:
            json.dump(trades, f)

        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            result = analyzer.analyze()

        assert result["overall"]["asian"]["trades"] == 6
        assert result["overall"]["european"]["trades"] == 0
        assert result["overall"]["us"]["trades"] == 0

    def test_multiple_assets(self, tmp_path):
        analyzer = SessionAnalyzer(min_trades_per_session=2)
        journal_file = str(tmp_path / "journal.json")

        trades = (
            _make_session_trades("BTC", "asian", 3)
            + _make_session_trades("ETH", "us", 3)
        )
        with open(journal_file, "w") as f:
            json.dump(trades, f)

        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            result = analyzer.analyze()

        assert "BTC" in result["per_asset"]
        assert "ETH" in result["per_asset"]

    def test_score_calculation(self, tmp_path):
        """Score is between 0 and 1."""
        analyzer = SessionAnalyzer(min_trades_per_session=2)
        journal_file = str(tmp_path / "journal.json")

        trades = _make_session_trades("BTC", "asian", 5, win_rate=1.0, avg_pnl=5.0)
        with open(journal_file, "w") as f:
            json.dump(trades, f)

        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            result = analyzer.analyze()

        score = result["per_asset"]["BTC"]["asian"]["score"]
        assert 0.0 <= score <= 1.0


# ── Tests: Session weight ────────────────────────────────────────


class TestSessionWeight:
    def test_weight_with_sufficient_data(self, tmp_path):
        analyzer = SessionAnalyzer(min_trades_per_session=3)
        journal_file = str(tmp_path / "journal.json")

        # Strong performance in Asian session
        trades = _make_session_trades("BTC", "asian", 5, win_rate=0.8, avg_pnl=3.0)
        with open(journal_file, "w") as f:
            json.dump(trades, f)

        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            weight = analyzer.get_session_weight("BTC", "asian")

        assert 0.5 <= weight <= 1.5
        assert weight > 1.0  # Good performance → weight > 1

    def test_weight_insufficient_data(self, tmp_path):
        analyzer = SessionAnalyzer(min_trades_per_session=10)
        journal_file = str(tmp_path / "journal.json")

        trades = _make_session_trades("BTC", "asian", 3)
        with open(journal_file, "w") as f:
            json.dump(trades, f)

        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            weight = analyzer.get_session_weight("BTC", "asian")

        assert weight == 1.0  # Insufficient data → neutral

    def test_weight_unknown_asset(self, tmp_path):
        analyzer = SessionAnalyzer(min_trades_per_session=3)
        journal_file = str(tmp_path / "journal.json")

        trades = _make_session_trades("BTC", "asian", 5)
        with open(journal_file, "w") as f:
            json.dump(trades, f)

        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            weight = analyzer.get_session_weight("UNKNOWN", "asian")

        assert weight == 1.0

    def test_weight_range(self, tmp_path):
        """Weight must be in [0.5, 1.5] range."""
        analyzer = SessionAnalyzer(min_trades_per_session=3)
        journal_file = str(tmp_path / "journal.json")

        # Best possible: all winners with high PnL
        trades = _make_session_trades("BTC", "asian", 5, win_rate=1.0, avg_pnl=10.0)
        with open(journal_file, "w") as f:
            json.dump(trades, f)

        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            weight = analyzer.get_session_weight("BTC", "asian")

        assert 0.5 <= weight <= 1.5


# ── Tests: Timestamp parsing ────────────────────────────────────


class TestTimestampParsing:
    def test_iso_with_timezone(self, tmp_path):
        analyzer = SessionAnalyzer(min_trades_per_session=1)
        journal_file = str(tmp_path / "journal.json")

        trades = [_make_trade(
            timestamp_open="2026-01-15T08:30:00+00:00",
            pnl_pct=1.0,
        )]
        with open(journal_file, "w") as f:
            json.dump(trades, f)

        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            result = analyzer.analyze()

        assert result["per_asset"]["BTC"]["european"]["trades"] == 1

    def test_iso_with_z_suffix(self, tmp_path):
        analyzer = SessionAnalyzer(min_trades_per_session=1)
        journal_file = str(tmp_path / "journal.json")

        trades = [_make_trade(
            timestamp_open="2026-01-15T15:00:00Z",
            pnl_pct=1.0,
        )]
        with open(journal_file, "w") as f:
            json.dump(trades, f)

        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            result = analyzer.analyze()

        assert result["per_asset"]["BTC"]["us"]["trades"] == 1

    def test_iso_without_timezone(self, tmp_path):
        analyzer = SessionAnalyzer(min_trades_per_session=1)
        journal_file = str(tmp_path / "journal.json")

        trades = [_make_trade(
            timestamp_open="2026-01-15T03:00:00",
            pnl_pct=1.0,
        )]
        with open(journal_file, "w") as f:
            json.dump(trades, f)

        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            result = analyzer.analyze()

        assert result["per_asset"]["BTC"]["asian"]["trades"] == 1

    def test_unparseable_timestamp_skipped(self, tmp_path):
        analyzer = SessionAnalyzer(min_trades_per_session=1)
        journal_file = str(tmp_path / "journal.json")

        trades = [_make_trade(timestamp_open="not-a-date", pnl_pct=1.0)]
        with open(journal_file, "w") as f:
            json.dump(trades, f)

        with patch("core.session_analyzer.JOURNAL_FILE", journal_file):
            result = analyzer.analyze()

        # Trade should be skipped, not crash
        total_trades = sum(
            s["trades"]
            for s in result.get("overall", {}).values()
        )
        assert total_trades == 0
