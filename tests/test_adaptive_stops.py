"""Tests for core/adaptive_stops.py — Adaptive Stop Optimizer."""

import json
import os
from unittest.mock import patch

import pytest

from core.adaptive_stops import AdaptiveStopOptimizer, _percentile


# ── Helpers ─────────────────────────────────────────────────────────


def _make_trade(
    asset: str = "BTC",
    pnl_pct: float = 2.0,
    mae_pct: float = -1.0,
    mfe_pct: float = 3.0,
    exit_reason: str = "take_profit",
    direction: str = "long",
) -> dict:
    return {
        "trade_id": "t1",
        "asset": asset,
        "direction": direction,
        "entry_price": 100.0,
        "exit_price": 102.0,
        "pnl_usd": 2.0,
        "pnl_pct": pnl_pct,
        "timestamp_open": "2026-01-15T08:00:00Z",
        "timestamp_close": "2026-01-16T08:00:00Z",
        "exit_reason": exit_reason,
        "mae_pct": mae_pct,
        "mfe_pct": mfe_pct,
    }


def _make_winners(asset: str, count: int, mae_range=(-0.5, -3.0), mfe_range=(2.0, 6.0)):
    """Generate winner trades with linearly spaced MAE/MFE values."""
    trades = []
    for i in range(count):
        frac = i / max(count - 1, 1)
        mae = mae_range[0] + frac * (mae_range[1] - mae_range[0])
        mfe = mfe_range[0] + frac * (mfe_range[1] - mfe_range[0])
        trades.append(_make_trade(
            asset=asset,
            pnl_pct=abs(mae) + 1.0,  # Winners have positive PnL
            mae_pct=mae,
            mfe_pct=mfe,
            exit_reason="take_profit",
        ))
    return trades


def _make_losers(asset: str, count: int, mae_range=(-2.0, -5.0), mfe_range=(0.5, 1.5)):
    """Generate loser trades."""
    trades = []
    for i in range(count):
        frac = i / max(count - 1, 1)
        mae = mae_range[0] + frac * (mae_range[1] - mae_range[0])
        mfe = mfe_range[0] + frac * (mfe_range[1] - mfe_range[0])
        trades.append(_make_trade(
            asset=asset,
            pnl_pct=-(abs(mae) - 0.5),  # Losers have negative PnL
            mae_pct=mae,
            mfe_pct=mfe,
            exit_reason="stop_loss",
        ))
    return trades


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def optimizer():
    return AdaptiveStopOptimizer(min_trades=5)


@pytest.fixture
def risk_params():
    return {"default_stop_loss_pct": 3.0, "default_take_profit_pct": 5.0}


# ── Tests: No data / insufficient data ────────────────────────────


class TestNoData:
    def test_no_journal_file(self, optimizer):
        """When journal file doesn't exist, returns insufficient data."""
        with patch("core.adaptive_stops.JOURNAL_FILE", "/nonexistent/file.json"):
            result = optimizer.analyze()
        assert result["sufficient_data"] is False
        assert result["total_closed_trades"] == 0
        assert result["per_asset"] == {}

    def test_empty_journal(self, optimizer, tmp_path):
        journal_file = str(tmp_path / "journal.json")
        with open(journal_file, "w") as f:
            json.dump([], f)
        with patch("core.adaptive_stops.JOURNAL_FILE", journal_file):
            result = optimizer.analyze()
        assert result["sufficient_data"] is False
        assert result["total_closed_trades"] == 0

    def test_corrupt_journal(self, optimizer, tmp_path):
        journal_file = str(tmp_path / "journal.json")
        with open(journal_file, "w") as f:
            f.write("not json")
        with patch("core.adaptive_stops.JOURNAL_FILE", journal_file):
            result = optimizer.analyze()
        assert result["sufficient_data"] is False

    def test_insufficient_trades(self, tmp_path):
        """Below min_trades threshold, sufficient_data is False."""
        opt = AdaptiveStopOptimizer(min_trades=10)
        journal_file = str(tmp_path / "journal.json")
        trades = _make_winners("BTC", 5)
        with open(journal_file, "w") as f:
            json.dump(trades, f)
        with patch("core.adaptive_stops.JOURNAL_FILE", journal_file):
            result = opt.analyze()
        assert result["sufficient_data"] is False
        assert result["total_closed_trades"] == 5
        # Still produces per-asset data even if insufficient
        assert "BTC" in result["per_asset"]

    def test_trades_without_mae_mfe_ignored(self, optimizer, tmp_path):
        """Trades missing MAE/MFE fields are not counted as closed."""
        journal_file = str(tmp_path / "journal.json")
        trades = [{"asset": "BTC", "pnl_pct": 1.0, "exit_reason": "tp"}]  # No mae/mfe
        with open(journal_file, "w") as f:
            json.dump(trades, f)
        with patch("core.adaptive_stops.JOURNAL_FILE", journal_file):
            result = optimizer.analyze()
        assert result["total_closed_trades"] == 0


# ── Tests: With sufficient data ───────────────────────────────────


class TestWithData:
    def test_per_asset_recommendations(self, tmp_path, risk_params):
        """Per-asset recommendations are computed from MAE/MFE data."""
        opt = AdaptiveStopOptimizer(min_trades=5)
        journal_file = str(tmp_path / "journal.json")
        risk_file = str(tmp_path / "risk_params.json")

        trades = _make_winners("BTC", 7) + _make_losers("BTC", 3)
        with open(journal_file, "w") as f:
            json.dump(trades, f)
        with open(risk_file, "w") as f:
            json.dump(risk_params, f)

        with patch("core.adaptive_stops.JOURNAL_FILE", journal_file), \
             patch("core.adaptive_stops.RISK_PARAMS_FILE", risk_file):
            result = opt.analyze()

        assert result["sufficient_data"] is True
        assert result["total_closed_trades"] == 10
        btc = result["per_asset"]["BTC"]
        assert btc["trades"] == 10
        assert btc["winners"] == 7
        assert btc["losers"] == 3
        assert btc["recommended_stop_pct"] >= 1.0  # Floor enforced
        assert btc["recommended_tp_pct"] >= 2.0  # Floor enforced
        assert btc["current_stop_pct"] == 3.0
        assert btc["current_tp_pct"] == 5.0
        assert btc["median_winner_mae_pct"] is not None
        assert btc["p90_winner_mae_pct"] is not None
        assert btc["avg_winner_mfe_pct"] is not None

    def test_multiple_assets(self, tmp_path, risk_params):
        """Multiple assets are analyzed independently."""
        opt = AdaptiveStopOptimizer(min_trades=3)
        journal_file = str(tmp_path / "journal.json")
        risk_file = str(tmp_path / "risk_params.json")

        trades = (
            _make_winners("BTC", 4)
            + _make_losers("BTC", 2)
            + _make_winners("ETH", 3)
            + _make_losers("ETH", 1)
        )
        with open(journal_file, "w") as f:
            json.dump(trades, f)
        with open(risk_file, "w") as f:
            json.dump(risk_params, f)

        with patch("core.adaptive_stops.JOURNAL_FILE", journal_file), \
             patch("core.adaptive_stops.RISK_PARAMS_FILE", risk_file):
            result = opt.analyze()

        assert "BTC" in result["per_asset"]
        assert "ETH" in result["per_asset"]
        assert result["per_asset"]["BTC"]["trades"] == 6
        assert result["per_asset"]["ETH"]["trades"] == 4

    def test_portfolio_level_aggregation(self, tmp_path, risk_params):
        """Portfolio-level recommendations average across assets."""
        opt = AdaptiveStopOptimizer(min_trades=3)
        journal_file = str(tmp_path / "journal.json")
        risk_file = str(tmp_path / "risk_params.json")

        trades = _make_winners("BTC", 5) + _make_winners("ETH", 5)
        with open(journal_file, "w") as f:
            json.dump(trades, f)
        with open(risk_file, "w") as f:
            json.dump(risk_params, f)

        with patch("core.adaptive_stops.JOURNAL_FILE", journal_file), \
             patch("core.adaptive_stops.RISK_PARAMS_FILE", risk_file):
            result = opt.analyze()

        assert "recommended_stop_pct" in result["portfolio_level"]
        assert "recommended_tp_pct" in result["portfolio_level"]
        assert result["portfolio_level"]["recommended_stop_pct"] > 0
        assert result["portfolio_level"]["recommended_tp_pct"] > 0

    def test_all_winners(self, tmp_path, risk_params):
        """All winning trades produce valid recommendations."""
        opt = AdaptiveStopOptimizer(min_trades=3)
        journal_file = str(tmp_path / "journal.json")
        risk_file = str(tmp_path / "risk_params.json")

        trades = _make_winners("BTC", 10)
        with open(journal_file, "w") as f:
            json.dump(trades, f)
        with open(risk_file, "w") as f:
            json.dump(risk_params, f)

        with patch("core.adaptive_stops.JOURNAL_FILE", journal_file), \
             patch("core.adaptive_stops.RISK_PARAMS_FILE", risk_file):
            result = opt.analyze()

        btc = result["per_asset"]["BTC"]
        assert btc["losers"] == 0
        assert btc["median_loser_mfe_pct"] is None
        assert btc["recommended_stop_pct"] >= 1.0
        assert btc["recommended_tp_pct"] >= 2.0

    def test_all_losers(self, tmp_path, risk_params):
        """All losing trades widen stop from current config."""
        opt = AdaptiveStopOptimizer(min_trades=3)
        journal_file = str(tmp_path / "journal.json")
        risk_file = str(tmp_path / "risk_params.json")

        trades = _make_losers("BTC", 5)
        with open(journal_file, "w") as f:
            json.dump(trades, f)
        with open(risk_file, "w") as f:
            json.dump(risk_params, f)

        with patch("core.adaptive_stops.JOURNAL_FILE", journal_file), \
             patch("core.adaptive_stops.RISK_PARAMS_FILE", risk_file):
            result = opt.analyze()

        btc = result["per_asset"]["BTC"]
        assert btc["winners"] == 0
        assert btc["median_winner_mae_pct"] is None
        # All losers: stop widens by 20% from current
        assert btc["recommended_stop_pct"] == round(3.0 * 1.2, 2)
        assert btc["recommended_tp_pct"] == 5.0

    def test_single_asset(self, tmp_path, risk_params):
        """Single asset produces both per-asset and portfolio-level results."""
        opt = AdaptiveStopOptimizer(min_trades=3)
        journal_file = str(tmp_path / "journal.json")
        risk_file = str(tmp_path / "risk_params.json")

        trades = _make_winners("SPY", 5) + _make_losers("SPY", 2)
        with open(journal_file, "w") as f:
            json.dump(trades, f)
        with open(risk_file, "w") as f:
            json.dump(risk_params, f)

        with patch("core.adaptive_stops.JOURNAL_FILE", journal_file), \
             patch("core.adaptive_stops.RISK_PARAMS_FILE", risk_file):
            result = opt.analyze()

        assert len(result["per_asset"]) == 1
        assert "SPY" in result["per_asset"]
        spy_stop = result["per_asset"]["SPY"]["recommended_stop_pct"]
        assert result["portfolio_level"]["recommended_stop_pct"] == spy_stop


# ── Tests: Persistence ────────────────────────────────────────────


class TestPersistence:
    def test_persist_creates_file(self, tmp_path, risk_params):
        opt = AdaptiveStopOptimizer(min_trades=3)
        journal_file = str(tmp_path / "journal.json")
        risk_file = str(tmp_path / "risk_params.json")
        rec_file = str(tmp_path / "stop_recommendations.json")

        trades = _make_winners("BTC", 5)
        with open(journal_file, "w") as f:
            json.dump(trades, f)
        with open(risk_file, "w") as f:
            json.dump(risk_params, f)

        with patch("core.adaptive_stops.JOURNAL_FILE", journal_file), \
             patch("core.adaptive_stops.RISK_PARAMS_FILE", risk_file), \
             patch("core.adaptive_stops.RECOMMENDATIONS_FILE", rec_file), \
             patch("core.adaptive_stops.DATA_DIR", str(tmp_path)):
            opt.persist_recommendations()

        assert os.path.exists(rec_file)
        with open(rec_file) as f:
            data = json.load(f)
        assert "per_asset" in data
        assert data["total_closed_trades"] == 5

    def test_persist_with_no_data(self, tmp_path):
        opt = AdaptiveStopOptimizer(min_trades=5)
        rec_file = str(tmp_path / "stop_recommendations.json")

        with patch("core.adaptive_stops.JOURNAL_FILE", "/nonexistent.json"), \
             patch("core.adaptive_stops.RECOMMENDATIONS_FILE", rec_file), \
             patch("core.adaptive_stops.DATA_DIR", str(tmp_path)):
            opt.persist_recommendations()

        assert os.path.exists(rec_file)
        with open(rec_file) as f:
            data = json.load(f)
        assert data["sufficient_data"] is False


# ── Tests: Percentile helper ─────────────────────────────────────


class TestPercentile:
    def test_empty_list(self):
        assert _percentile([], 90) == 0.0

    def test_single_value(self):
        assert _percentile([5.0], 90) == 5.0

    def test_known_values(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        assert _percentile(data, 50) == 5.5
        assert _percentile(data, 0) == 1.0
        assert _percentile(data, 100) == 10.0

    def test_90th_percentile(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        p90 = _percentile(data, 90)
        assert 9.0 <= p90 <= 10.0
