"""Tests for Kelly Criterion Position Sizing (core/kelly_sizer.py)."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from core.kelly_sizer import KellySizer


# ── Helpers ────────────────────────────────────────────────────────────


def _make_trade(asset: str, pnl_usd: float) -> dict[str, Any]:
    """Create a minimal trade journal entry with outcome.pnl_usd."""
    return {
        "asset": asset,
        "outcome": {"pnl_usd": pnl_usd, "pnl_pct": 0.0},
    }


def _write_journal(path, trades: list[dict[str, Any]]) -> None:
    """Write trades to a journal file."""
    with open(path, "w") as f:
        json.dump(trades, f)


def _make_sizer(
    journal_file: str = "/nonexistent/journal.json",
    **overrides: Any,
) -> KellySizer:
    """Create a KellySizer with default config, overrideable."""
    config: dict[str, Any] = {
        "kelly_alpha": 0.35,
        "min_kelly_trades": 20,
        "max_kelly_pct": 5.0,
        "kelly_enabled": True,
    }
    config.update(overrides)
    return _make_sizer_with_config(journal_file, config)


def _make_sizer_with_config(journal_file: str, config: dict[str, Any]) -> KellySizer:
    return KellySizer(journal_file=journal_file, config=config)


# ── Formula tests ─────────────────────────────────────────────────────


class TestKellyFormula:
    """Tests for Kelly formula computation via _apply_kelly."""

    def test_kelly_basic(self):
        """p=0.6, b=2.0 -> f*=(0.6*2-0.4)/2=0.4, fraction=0.35*0.4=0.14, capped at 0.05."""
        sizer = _make_sizer()
        stats = {"win_rate": 0.6, "avg_win": 200.0, "avg_loss": 100.0, "payoff_ratio": 2.0, "sample_size": 30}
        result = sizer._apply_kelly(stats, source="test", using_fallback=False)
        assert result["fraction"] == pytest.approx(0.05, abs=0.001)  # capped at max_kelly_pct=5%
        assert result["source"] == "test"

    def test_kelly_negative_edge(self):
        """p=0.3, b=1.0 -> f*=(0.3*1-0.7)/1=-0.4, fraction=0.0."""
        sizer = _make_sizer()
        stats = {"win_rate": 0.3, "avg_win": 100.0, "avg_loss": 100.0, "payoff_ratio": 1.0, "sample_size": 30}
        result = sizer._apply_kelly(stats, source="test", using_fallback=False)
        assert result["fraction"] == 0.0

    def test_kelly_cap(self):
        """Fraction never exceeds max_kelly_pct / 100."""
        sizer = _make_sizer(max_kelly_pct=3.0)
        # Very high edge: p=0.9, b=5.0 -> f*=(0.9*5-0.1)/5=0.88, fraction=0.35*0.88=0.308
        stats = {"win_rate": 0.9, "avg_win": 500.0, "avg_loss": 100.0, "payoff_ratio": 5.0, "sample_size": 30}
        result = sizer._apply_kelly(stats, source="test", using_fallback=False)
        assert result["fraction"] == 0.03  # max_kelly_pct=3.0 -> 0.03

    def test_kelly_alpha_scaling(self):
        """fraction = alpha * f_star, but verify scaling works with higher cap."""
        sizer = _make_sizer(kelly_alpha=0.5, max_kelly_pct=25.0)
        # p=0.6, b=2.0 -> f*=0.4, fraction=0.5*0.4=0.20
        stats = {"win_rate": 0.6, "avg_win": 200.0, "avg_loss": 100.0, "payoff_ratio": 2.0, "sample_size": 30}
        result = sizer._apply_kelly(stats, source="test", using_fallback=False)
        assert result["fraction"] == pytest.approx(0.20, abs=0.001)

    def test_kelly_perfect_record(self):
        """p=1.0, b=2.0 -> f*=(1.0*2-0)/2=1.0, fraction=0.35*1.0=0.35, capped at 0.05."""
        sizer = _make_sizer()
        stats = {"win_rate": 1.0, "avg_win": 200.0, "avg_loss": 0.0, "payoff_ratio": 2.0, "sample_size": 30}
        result = sizer._apply_kelly(stats, source="test", using_fallback=False)
        assert result["fraction"] == 0.05  # Capped at max_kelly_pct=5.0 -> 0.05

    def test_kelly_zero_payoff(self):
        """payoff_ratio=0 -> fraction=0.0 (guard against division by zero)."""
        sizer = _make_sizer()
        stats = {"win_rate": 0.5, "avg_win": 0.0, "avg_loss": 100.0, "payoff_ratio": 0.0, "sample_size": 30}
        result = sizer._apply_kelly(stats, source="test", using_fallback=False)
        assert result["fraction"] == 0.0

    def test_kelly_breakeven_edge(self):
        """p=0.5, b=1.0 -> f*=(0.5-0.5)/1=0.0, fraction=0.0."""
        sizer = _make_sizer()
        stats = {"win_rate": 0.5, "avg_win": 100.0, "avg_loss": 100.0, "payoff_ratio": 1.0, "sample_size": 30}
        result = sizer._apply_kelly(stats, source="test", using_fallback=False)
        assert result["fraction"] == 0.0


# ── Stats computation tests ───────────────────────────────────────────


class TestComputeStats:
    """Tests for _compute_stats."""

    def test_compute_stats_empty(self):
        sizer = _make_sizer()
        stats = sizer._compute_stats([])
        assert stats["sample_size"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["payoff_ratio"] == 0.0

    def test_compute_stats_mixed(self):
        trades = [
            {"asset": "AAPL", "pnl_usd": 100.0},
            {"asset": "AAPL", "pnl_usd": 200.0},
            {"asset": "AAPL", "pnl_usd": -50.0},
            {"asset": "AAPL", "pnl_usd": -100.0},
        ]
        sizer = _make_sizer()
        stats = sizer._compute_stats(trades)
        assert stats["sample_size"] == 4
        assert stats["win_rate"] == pytest.approx(0.5, abs=0.001)
        assert stats["avg_win"] == pytest.approx(150.0, abs=0.01)
        assert stats["avg_loss"] == pytest.approx(75.0, abs=0.01)
        assert stats["payoff_ratio"] == pytest.approx(2.0, abs=0.01)

    def test_compute_stats_all_wins(self):
        trades = [
            {"asset": "BTC", "pnl_usd": 50.0},
            {"asset": "BTC", "pnl_usd": 150.0},
        ]
        sizer = _make_sizer()
        stats = sizer._compute_stats(trades)
        assert stats["sample_size"] == 2
        assert stats["win_rate"] == 1.0
        # No losses: payoff = min(avg_win/1.0, 10.0) = min(100/1, 10) = 10.0
        assert stats["payoff_ratio"] == 10.0

    def test_compute_stats_all_losses(self):
        trades = [
            {"asset": "ETH", "pnl_usd": -30.0},
            {"asset": "ETH", "pnl_usd": -70.0},
        ]
        sizer = _make_sizer()
        stats = sizer._compute_stats(trades)
        assert stats["sample_size"] == 2
        assert stats["win_rate"] == 0.0
        assert stats["payoff_ratio"] == 0.0

    def test_compute_stats_skips_breakeven(self):
        """Trades with pnl_usd == 0 are skipped."""
        trades = [
            {"asset": "AAPL", "pnl_usd": 100.0},
            {"asset": "AAPL", "pnl_usd": 0.0},
            {"asset": "AAPL", "pnl_usd": -50.0},
        ]
        sizer = _make_sizer()
        stats = sizer._compute_stats(trades)
        assert stats["sample_size"] == 2  # Breakeven excluded

    def test_compute_stats_payoff_ratio_cap(self):
        """No losses: payoff_ratio capped at 10.0."""
        trades = [{"asset": "X", "pnl_usd": 500.0}]
        sizer = _make_sizer()
        stats = sizer._compute_stats(trades)
        assert stats["payoff_ratio"] == 10.0


# ── Fallback hierarchy tests ─────────────────────────────────────────


class TestFallbackHierarchy:

    def test_asset_sufficient(self, tmp_path):
        """Asset-level has enough data -> source='asset'."""
        trades = [_make_trade("AAPL", 10.0 if i % 3 != 0 else -5.0) for i in range(25)]
        journal = str(tmp_path / "journal.json")
        _write_journal(journal, trades)
        sizer = _make_sizer(journal_file=journal, min_kelly_trades=20)
        result = sizer.kelly_fraction("AAPL")
        assert result["source"] == "asset"
        assert result["using_fallback"] is False

    def test_fallback_asset_to_sector(self, tmp_path):
        """Asset n=5, sector has n=25 -> source='sector'."""
        trades = [_make_trade("AAPL", 10.0 if i % 3 != 0 else -5.0) for i in range(5)]
        # Add more tech trades from other assets
        trades += [_make_trade("NVDA", 15.0 if i % 3 != 0 else -8.0) for i in range(20)]
        journal = str(tmp_path / "journal.json")
        _write_journal(journal, trades)
        sizer = _make_sizer(journal_file=journal, min_kelly_trades=20)
        result = sizer.kelly_fraction("AAPL")
        assert result["source"] == "sector"
        assert result["using_fallback"] is True

    def test_fallback_sector_to_global(self, tmp_path):
        """Sector n=10, global n=30 -> source='global'."""
        # 10 tech trades
        trades = [_make_trade("AAPL", 10.0 if i % 3 != 0 else -5.0) for i in range(10)]
        # 20 crypto trades (different sector)
        trades += [_make_trade("BTC", 20.0 if i % 3 != 0 else -10.0) for i in range(20)]
        journal = str(tmp_path / "journal.json")
        _write_journal(journal, trades)
        sizer = _make_sizer(journal_file=journal, min_kelly_trades=20)
        result = sizer.kelly_fraction("AAPL")
        assert result["source"] == "global"
        assert result["using_fallback"] is True

    def test_fallback_insufficient_all(self, tmp_path):
        """Global n=5 -> source='insufficient_data'."""
        trades = [_make_trade("AAPL", 10.0) for _ in range(5)]
        journal = str(tmp_path / "journal.json")
        _write_journal(journal, trades)
        sizer = _make_sizer(journal_file=journal, min_kelly_trades=20)
        result = sizer.kelly_fraction("AAPL")
        assert result["source"] == "insufficient_data"
        assert result["fraction"] == 0.0


# ── Feature flag tests ────────────────────────────────────────────────


class TestFeatureFlag:

    def test_kelly_disabled(self, tmp_path):
        """kelly_enabled=false -> fraction=0.0, source='disabled'."""
        trades = [_make_trade("AAPL", 10.0 if i % 3 != 0 else -5.0) for i in range(30)]
        journal = str(tmp_path / "journal.json")
        _write_journal(journal, trades)
        sizer = _make_sizer(journal_file=journal, kelly_enabled=False)
        result = sizer.kelly_fraction("AAPL")
        assert result["fraction"] == 0.0
        assert result["source"] == "disabled"
        assert result["using_fallback"] is True


# ── Journal loading tests ─────────────────────────────────────────────


class TestJournalLoading:

    def test_load_missing_file(self):
        """Missing file returns empty list gracefully."""
        sizer = _make_sizer(journal_file="/nonexistent/path/journal.json")
        trades = sizer._load_closed_trades()
        assert trades == []

    def test_load_corrupt_json(self, tmp_path):
        """Corrupt JSON returns empty list gracefully."""
        journal = str(tmp_path / "journal.json")
        with open(journal, "w") as f:
            f.write("{invalid json!!!")
        sizer = _make_sizer(journal_file=journal)
        trades = sizer._load_closed_trades()
        assert trades == []

    def test_cache_ttl(self, tmp_path):
        """Cache is reused within TTL, refreshed after."""
        trades = [_make_trade("AAPL", 10.0)]
        journal = str(tmp_path / "journal.json")
        _write_journal(journal, trades)

        sizer = _make_sizer(journal_file=journal)
        # Set very short TTL for test
        sizer._cache_ttl = 0.1

        result1 = sizer._load_closed_trades()
        assert len(result1) == 1

        # Add more trades to file
        trades.append(_make_trade("NVDA", 20.0))
        _write_journal(journal, trades)

        # Within TTL — should return cached (still 1)
        result2 = sizer._load_closed_trades()
        assert len(result2) == 1

        # Wait for TTL to expire
        time.sleep(0.15)

        # After TTL — should re-read (now 2)
        result3 = sizer._load_closed_trades()
        assert len(result3) == 2

    def test_load_filters_open_trades(self, tmp_path):
        """Only trades with outcome.pnl_usd != None are loaded."""
        trades = [
            _make_trade("AAPL", 10.0),
            {"asset": "NVDA", "outcome": {"pnl_usd": None}},  # Open trade
            {"asset": "TSLA"},  # No outcome at all
        ]
        journal = str(tmp_path / "journal.json")
        _write_journal(journal, trades)
        sizer = _make_sizer(journal_file=journal)
        result = sizer._load_closed_trades()
        assert len(result) == 1
        assert result[0]["asset"] == "AAPL"


# ── Sector mapping tests ──────────────────────────────────────────────


class TestSectorMapping:

    def test_core_sectors(self):
        sizer = _make_sizer()
        assert sizer._get_sector("BTC") == "crypto"
        assert sizer._get_sector("ETH") == "crypto"
        assert sizer._get_sector("AAPL") == "tech"
        assert sizer._get_sector("NVDA") == "tech"
        assert sizer._get_sector("SPY") == "index"
        assert sizer._get_sector("TLT") == "bonds"
        assert sizer._get_sector("GLDM") == "commodities"
        assert sizer._get_sector("XLE") == "energy"
        assert sizer._get_sector("EWS") == "regional"
        assert sizer._get_sector("FXI") == "regional"

    def test_unknown_sector(self):
        sizer = _make_sizer()
        assert sizer._get_sector("UNKNOWN_TICKER") == "other"


# ── End-to-end test ───────────────────────────────────────────────────


class TestEndToEnd:

    def test_kelly_fraction_end_to_end(self, tmp_path):
        """Write 30 trades to tmp_path, verify full Kelly output."""
        trades = []
        # 20 wins at $15, 10 losses at $10
        for i in range(30):
            if i < 20:
                trades.append(_make_trade("AAPL", 15.0))
            else:
                trades.append(_make_trade("AAPL", -10.0))

        journal = str(tmp_path / "journal.json")
        _write_journal(journal, trades)

        sizer = _make_sizer(journal_file=journal, min_kelly_trades=20)
        result = sizer.kelly_fraction("AAPL")

        assert result["source"] == "asset"
        assert result["using_fallback"] is False
        assert result["fraction"] > 0
        assert result["fraction"] <= 0.05  # max_kelly_pct=5.0

        stats = result["stats"]
        assert stats["sample_size"] == 30
        assert stats["win_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert stats["avg_win"] == pytest.approx(15.0, abs=0.01)
        assert stats["avg_loss"] == pytest.approx(10.0, abs=0.01)
        assert stats["payoff_ratio"] == pytest.approx(1.5, abs=0.01)

        # Verify math: f* = (p*b - q) / b = (0.667*1.5 - 0.333)/1.5 = 0.4444
        # fraction = 0.35 * 0.4444 = 0.1556
        # Capped at 0.05 since max_kelly_pct=5.0
        assert result["fraction"] == 0.05

    def test_kelly_fraction_moderate_edge(self, tmp_path):
        """Moderate edge: fraction between 0 and cap."""
        trades = []
        # 12 wins at $8, 8 losses at $10 -> p=0.6, b=0.8
        for i in range(20):
            if i < 12:
                trades.append(_make_trade("BTC", 8.0))
            else:
                trades.append(_make_trade("BTC", -10.0))

        journal = str(tmp_path / "journal.json")
        _write_journal(journal, trades)

        sizer = _make_sizer(journal_file=journal, min_kelly_trades=20)
        result = sizer.kelly_fraction("BTC")

        assert result["source"] == "asset"
        # p=0.6, b=0.8, f*=(0.6*0.8-0.4)/0.8 = (0.48-0.4)/0.8 = 0.1
        # fraction=0.35*0.1=0.035
        assert result["fraction"] == pytest.approx(0.035, abs=0.001)


# ── get_all_stats test ────────────────────────────────────────────────


class TestGetAllStats:

    def test_get_all_stats(self, tmp_path):
        trades = [
            _make_trade("AAPL", 10.0),
            _make_trade("AAPL", -5.0),
            _make_trade("BTC", 20.0),
        ]
        journal = str(tmp_path / "journal.json")
        _write_journal(journal, trades)

        sizer = _make_sizer(journal_file=journal)
        result = sizer.get_all_stats()

        assert "global" in result
        assert result["global"]["sample_size"] == 3
        assert "per_asset" in result
        assert "AAPL" in result["per_asset"]
        assert "BTC" in result["per_asset"]
        assert "per_sector" in result
        assert "tech" in result["per_sector"]
        assert "crypto" in result["per_sector"]
        assert "config" in result
        assert result["config"]["kelly_alpha"] == 0.35

    def test_get_all_stats_empty(self):
        sizer = _make_sizer()
        result = sizer.get_all_stats()
        assert result["global"]["sample_size"] == 0
        assert result["per_asset"] == {}
        assert result["per_sector"] == {}
