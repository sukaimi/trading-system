"""Tests for core/phantom_analyzer.py"""

import json
import os
import tempfile

import pytest

from core.phantom_analyzer import PhantomAnalyzer


@pytest.fixture
def tmp_dirs():
    """Create temp files for phantom data and analysis output."""
    fd1, phantom_path = tempfile.mkstemp(suffix=".json")
    os.close(fd1)
    fd2, analysis_path = tempfile.mkstemp(suffix=".json")
    os.close(fd2)
    yield phantom_path, analysis_path
    for p in (phantom_path, analysis_path):
        if os.path.exists(p):
            os.unlink(p)


def _make_phantom(
    asset: str = "BTC",
    direction: str = "long",
    confidence: float = 0.7,
    killed_by: str = "devils_advocate",
    outcome_checked: bool = True,
    outcome_pnl_pct: float = 3.0,
    entry_price: float = 100.0,
) -> dict:
    return {
        "timestamp": "2026-03-01T00:00:00Z",
        "asset": asset,
        "direction": direction,
        "confidence": confidence,
        "entry_price": entry_price,
        "suggested_position_pct": 3.0,
        "killed_by": killed_by,
        "reason": "test reason",
        "thesis": "test thesis",
        "outcome_checked": outcome_checked,
        "outcome_price": entry_price * (1 + outcome_pnl_pct / 100) if outcome_checked else None,
        "outcome_pnl_pct": outcome_pnl_pct if outcome_checked else None,
    }


class TestNoData:
    def test_empty_file(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        with open(phantom_path, "w") as f:
            json.dump([], f)
        analyzer = PhantomAnalyzer(phantom_file=phantom_path, analysis_file=analysis_path)
        result = analyzer.analyze()
        assert result["sufficient_data"] is False
        assert result["total_phantoms"] == 0
        assert result["checked_phantoms"] == 0
        assert result["overall_false_kill_rate"] == 0.0

    def test_missing_file(self, tmp_dirs):
        _, analysis_path = tmp_dirs
        analyzer = PhantomAnalyzer(phantom_file="/nonexistent.json", analysis_file=analysis_path)
        result = analyzer.analyze()
        assert result["sufficient_data"] is False

    def test_unchecked_only(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        phantoms = [_make_phantom(outcome_checked=False) for _ in range(10)]
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)
        analyzer = PhantomAnalyzer(phantom_file=phantom_path, analysis_file=analysis_path)
        result = analyzer.analyze()
        assert result["sufficient_data"] is False
        assert result["total_phantoms"] == 10
        assert result["checked_phantoms"] == 0


class TestSufficientData:
    def test_basic_analysis(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        phantoms = []
        # 6 checked: 4 would have won, 2 would have lost
        for i in range(4):
            phantoms.append(_make_phantom(outcome_pnl_pct=3.0))
        for i in range(2):
            phantoms.append(_make_phantom(outcome_pnl_pct=-2.0))
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)

        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        result = analyzer.analyze()

        assert result["sufficient_data"] is True
        assert result["total_phantoms"] == 6
        assert result["checked_phantoms"] == 6
        # 4/6 = 0.6667
        assert abs(result["overall_false_kill_rate"] - 0.6667) < 0.01


class TestPerAsset:
    def test_per_asset_false_kill_rate(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        phantoms = []
        # BTC: 5 checked, 4 would have won (high false kill rate)
        for i in range(4):
            phantoms.append(_make_phantom(asset="BTC", outcome_pnl_pct=5.0))
        phantoms.append(_make_phantom(asset="BTC", outcome_pnl_pct=-3.0))
        # ETH: 5 checked, 1 would have won (low false kill rate)
        phantoms.append(_make_phantom(asset="ETH", outcome_pnl_pct=2.0))
        for i in range(4):
            phantoms.append(_make_phantom(asset="ETH", outcome_pnl_pct=-2.0))
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)

        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        result = analyzer.analyze()

        btc = result["per_asset"]["BTC"]
        eth = result["per_asset"]["ETH"]
        assert btc["kills"] == 5
        assert btc["checked"] == 5
        assert btc["would_have_won"] == 4
        assert btc["false_kill_rate"] == 0.8
        assert btc["avg_missed_pnl_pct"] == 5.0

        assert eth["kills"] == 5
        assert eth["false_kill_rate"] == 0.2
        assert eth["avg_missed_pnl_pct"] == 2.0


class TestPerKiller:
    def test_per_killer_analysis(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        phantoms = []
        # DA: 5 kills, 3 false
        for i in range(3):
            phantoms.append(_make_phantom(killed_by="devils_advocate", outcome_pnl_pct=4.0))
        for i in range(2):
            phantoms.append(_make_phantom(killed_by="devils_advocate", outcome_pnl_pct=-2.0))
        # RM: 5 kills, 1 false
        phantoms.append(_make_phantom(killed_by="risk_manager", outcome_pnl_pct=1.0))
        for i in range(4):
            phantoms.append(_make_phantom(killed_by="risk_manager", outcome_pnl_pct=-3.0))
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)

        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        result = analyzer.analyze()

        da = result["per_killer"]["devils_advocate"]
        rm = result["per_killer"]["risk_manager"]
        assert da["kills"] == 5
        assert da["false_kill_rate"] == 0.6
        assert rm["kills"] == 5
        assert rm["false_kill_rate"] == 0.2


class TestPerDirection:
    def test_per_direction_analysis(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        phantoms = []
        # Long: 6 checked, 4 false
        for i in range(4):
            phantoms.append(_make_phantom(direction="long", outcome_pnl_pct=3.0))
        for i in range(2):
            phantoms.append(_make_phantom(direction="long", outcome_pnl_pct=-2.0))
        # Short: 6 checked, 2 false
        for i in range(2):
            phantoms.append(_make_phantom(direction="short", outcome_pnl_pct=3.0))
        for i in range(4):
            phantoms.append(_make_phantom(direction="short", outcome_pnl_pct=-2.0))
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)

        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        result = analyzer.analyze()

        assert abs(result["per_direction"]["long"]["false_kill_rate"] - 0.6667) < 0.01
        assert abs(result["per_direction"]["short"]["false_kill_rate"] - 0.3333) < 0.01


class TestBiasAlerts:
    def test_high_false_kill_alerts(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        phantoms = []
        # BTC: 6 checked, 5 false kills
        for i in range(5):
            phantoms.append(_make_phantom(asset="BTC", outcome_pnl_pct=5.0))
        phantoms.append(_make_phantom(asset="BTC", outcome_pnl_pct=-1.0))
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)

        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        result = analyzer.analyze()

        assert len(result["bias_alerts"]) > 0
        # Should mention BTC
        btc_alerts = [a for a in result["bias_alerts"] if "BTC" in a]
        assert len(btc_alerts) > 0

    def test_accurate_killer_alert(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        phantoms = []
        # risk_manager: 6 kills, only 1 false (accurate)
        phantoms.append(_make_phantom(killed_by="risk_manager", outcome_pnl_pct=1.0))
        for i in range(5):
            phantoms.append(_make_phantom(killed_by="risk_manager", outcome_pnl_pct=-3.0))
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)

        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        result = analyzer.analyze()

        rm_alerts = [a for a in result["bias_alerts"] if "risk_manager" in a and "correctly" in a]
        assert len(rm_alerts) > 0

    def test_no_alerts_with_insufficient_data(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        # Only 2 phantoms - below min_phantoms threshold for alerts
        phantoms = [
            _make_phantom(outcome_pnl_pct=5.0),
            _make_phantom(outcome_pnl_pct=5.0),
        ]
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)

        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        result = analyzer.analyze()
        assert result["bias_alerts"] == []


class TestAssetKillBias:
    def test_high_false_kill_rate_returns_above_one(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        phantoms = []
        # BTC: 6 checked, 5 false kills (fkr=0.83)
        for i in range(5):
            phantoms.append(_make_phantom(asset="BTC", outcome_pnl_pct=4.0))
        phantoms.append(_make_phantom(asset="BTC", outcome_pnl_pct=-2.0))
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)

        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        bias = analyzer.get_asset_kill_bias("BTC")
        assert bias > 1.0
        assert bias <= 1.5

    def test_low_false_kill_rate_returns_below_one(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        phantoms = []
        # ETH: 6 checked, 1 false kill (fkr=0.17)
        phantoms.append(_make_phantom(asset="ETH", outcome_pnl_pct=2.0))
        for i in range(5):
            phantoms.append(_make_phantom(asset="ETH", outcome_pnl_pct=-3.0))
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)

        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        bias = analyzer.get_asset_kill_bias("ETH")
        assert bias < 1.0
        assert bias >= 0.5

    def test_insufficient_data_returns_one(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        with open(phantom_path, "w") as f:
            json.dump([], f)
        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        assert analyzer.get_asset_kill_bias("BTC") == 1.0

    def test_unknown_asset_returns_one(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        phantoms = [_make_phantom(asset="BTC", outcome_pnl_pct=3.0) for _ in range(6)]
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)
        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        assert analyzer.get_asset_kill_bias("DOGE") == 1.0


class TestPersistence:
    def test_persist_creates_file(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        if os.path.exists(analysis_path):
            os.unlink(analysis_path)
        phantoms = [_make_phantom(outcome_pnl_pct=3.0) for _ in range(6)]
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)

        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        analyzer.persist_analysis()

        assert os.path.exists(analysis_path)
        with open(analysis_path) as f:
            data = json.load(f)
        assert "total_phantoms" in data
        assert "per_asset" in data
        assert "bias_alerts" in data

    def test_persist_after_analyze(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        phantoms = [_make_phantom(outcome_pnl_pct=3.0) for _ in range(6)]
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)

        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        analyzer.analyze()
        analyzer.persist_analysis()

        with open(analysis_path) as f:
            data = json.load(f)
        assert data["total_phantoms"] == 6
        assert data["checked_phantoms"] == 6


class TestRecommendations:
    def test_generates_loosen_recommendation(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        phantoms = []
        # BTC: high false kill rate
        for i in range(5):
            phantoms.append(_make_phantom(asset="BTC", outcome_pnl_pct=5.0))
        phantoms.append(_make_phantom(asset="BTC", outcome_pnl_pct=-1.0))
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)

        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        result = analyzer.analyze()

        btc_recs = [r for r in result["recommendations"] if "BTC" in r]
        assert len(btc_recs) > 0
        assert any("loosen" in r.lower() for r in btc_recs)

    def test_no_recommendations_with_insufficient_data(self, tmp_dirs):
        phantom_path, analysis_path = tmp_dirs
        phantoms = [_make_phantom() for _ in range(2)]
        with open(phantom_path, "w") as f:
            json.dump(phantoms, f)

        analyzer = PhantomAnalyzer(min_phantoms=5, phantom_file=phantom_path, analysis_file=analysis_path)
        result = analyzer.analyze()
        assert result["recommendations"] == []
