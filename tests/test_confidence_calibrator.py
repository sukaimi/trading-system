"""Tests for core/confidence_calibrator.py"""

import json
import os
import tempfile

import pytest

from core.confidence_calibrator import ConfidenceCalibrator, _assign_bucket


@pytest.fixture
def tmp_dirs():
    """Create temp files for signal data and calibration output."""
    fd1, signal_path = tempfile.mkstemp(suffix=".json")
    os.close(fd1)
    fd2, cal_path = tempfile.mkstemp(suffix=".json")
    os.close(fd2)
    yield signal_path, cal_path
    for p in (signal_path, cal_path):
        if os.path.exists(p):
            os.unlink(p)


def _make_signal(confidence: float, correct: bool, source: str = "news_scan") -> dict:
    return {
        "signal_id": "test",
        "asset": "BTC",
        "confidence": confidence,
        "signal_strength": confidence,
        "signal_correct": correct,
        "pipeline_outcome": "executed",
        "source_type": source,
        "exit_price": 100.0,
        "pnl_pct": 2.0 if correct else -2.0,
    }


class TestBucketAssignment:
    def test_low_bucket(self):
        assert _assign_bucket(0.0) == "low"
        assert _assign_bucket(0.15) == "low"
        assert _assign_bucket(0.29) == "low"

    def test_medium_low_bucket(self):
        assert _assign_bucket(0.3) == "medium_low"
        assert _assign_bucket(0.49) == "medium_low"

    def test_medium_bucket(self):
        assert _assign_bucket(0.5) == "medium"
        assert _assign_bucket(0.69) == "medium"

    def test_high_bucket(self):
        assert _assign_bucket(0.7) == "high"
        assert _assign_bucket(0.84) == "high"

    def test_very_high_bucket(self):
        assert _assign_bucket(0.85) == "very_high"
        assert _assign_bucket(0.99) == "very_high"

    def test_exactly_one(self):
        assert _assign_bucket(1.0) == "very_high"

    def test_boundary_at_0_3(self):
        """0.3 should be in medium_low, not low."""
        assert _assign_bucket(0.3) == "medium_low"

    def test_boundary_at_0_7(self):
        """0.7 should be in high, not medium."""
        assert _assign_bucket(0.7) == "high"

    def test_boundary_at_0_85(self):
        """0.85 should be in very_high, not high."""
        assert _assign_bucket(0.85) == "very_high"


class TestNoData:
    def test_analyze_returns_insufficient(self, tmp_dirs):
        signal_path, cal_path = tmp_dirs
        # Write empty file
        with open(signal_path, "w") as f:
            json.dump([], f)
        cal = ConfidenceCalibrator(signal_file=signal_path, calibration_file=cal_path)
        result = cal.analyze()
        assert result["sufficient_data"] is False
        assert result["total_signals"] == 0
        assert result["overall_calibration"] == "insufficient_data"

    def test_calibrate_returns_raw_when_no_data(self, tmp_dirs):
        signal_path, cal_path = tmp_dirs
        with open(signal_path, "w") as f:
            json.dump([], f)
        cal = ConfidenceCalibrator(signal_file=signal_path, calibration_file=cal_path)
        assert cal.calibrate_confidence(0.75) == 0.75
        assert cal.calibrate_confidence(0.3) == 0.3

    def test_missing_file(self, tmp_dirs):
        _, cal_path = tmp_dirs
        cal = ConfidenceCalibrator(signal_file="/nonexistent/file.json", calibration_file=cal_path)
        result = cal.analyze()
        assert result["sufficient_data"] is False
        assert cal.calibrate_confidence(0.8) == 0.8


class TestOverconfidentSignals:
    def test_overconfident_detection(self, tmp_dirs):
        signal_path, cal_path = tmp_dirs
        # Create signals with high confidence but low win rate
        signals = []
        for i in range(10):
            # 0.75 confidence, only 30% win (3 wins, 7 losses)
            signals.append(_make_signal(0.75, i < 3))
        with open(signal_path, "w") as f:
            json.dump(signals, f)

        cal = ConfidenceCalibrator(min_signals_per_bucket=5, signal_file=signal_path, calibration_file=cal_path)
        result = cal.analyze()

        assert result["sufficient_data"] is True
        high_bucket = result["buckets"]["high"]
        assert high_bucket["signals"] == 10
        assert high_bucket["wins"] == 3
        assert high_bucket["actual_win_rate"] == 0.3
        # Calibration factor < 1 (overconfident)
        assert high_bucket["calibration_factor"] < 1.0
        assert result["overall_calibration"] == "overconfident"

    def test_calibration_reduces_confidence(self, tmp_dirs):
        signal_path, cal_path = tmp_dirs
        signals = []
        for i in range(10):
            signals.append(_make_signal(0.75, i < 3))  # 30% win rate
        with open(signal_path, "w") as f:
            json.dump(signals, f)

        cal = ConfidenceCalibrator(min_signals_per_bucket=5, signal_file=signal_path, calibration_file=cal_path)
        adjusted = cal.calibrate_confidence(0.75)
        assert adjusted < 0.75  # Should reduce overconfident signals


class TestUnderconfidentSignals:
    def test_underconfident_detection(self, tmp_dirs):
        signal_path, cal_path = tmp_dirs
        # Low confidence signals that mostly win
        signals = []
        for i in range(10):
            # 0.15 confidence (low bucket), 80% win rate
            signals.append(_make_signal(0.15, i < 8))
        with open(signal_path, "w") as f:
            json.dump(signals, f)

        cal = ConfidenceCalibrator(min_signals_per_bucket=5, signal_file=signal_path, calibration_file=cal_path)
        result = cal.analyze()

        assert result["sufficient_data"] is True
        low_bucket = result["buckets"]["low"]
        assert low_bucket["signals"] == 10
        assert low_bucket["wins"] == 8
        assert low_bucket["actual_win_rate"] == 0.8
        # Calibration factor > 1 (underconfident)
        assert low_bucket["calibration_factor"] > 1.0
        assert result["overall_calibration"] == "underconfident"


class TestClampedOutput:
    def test_calibrate_clamps_to_one(self, tmp_dirs):
        signal_path, cal_path = tmp_dirs
        # Very underconfident: low confidence but always wins
        signals = [_make_signal(0.15, True) for _ in range(10)]
        with open(signal_path, "w") as f:
            json.dump(signals, f)

        cal = ConfidenceCalibrator(min_signals_per_bucket=5, signal_file=signal_path, calibration_file=cal_path)
        # Even with huge correction factor, should clamp to 1.0
        adjusted = cal.calibrate_confidence(0.95)
        assert adjusted <= 1.0

    def test_calibrate_clamps_to_zero(self, tmp_dirs):
        signal_path, cal_path = tmp_dirs
        # Overconfident: high confidence but never wins
        signals = [_make_signal(0.75, False) for _ in range(10)]
        with open(signal_path, "w") as f:
            json.dump(signals, f)

        cal = ConfidenceCalibrator(min_signals_per_bucket=5, signal_file=signal_path, calibration_file=cal_path)
        adjusted = cal.calibrate_confidence(0.75)
        assert adjusted >= 0.0


class TestPerSource:
    def test_per_source_analysis(self, tmp_dirs):
        signal_path, cal_path = tmp_dirs
        signals = []
        # news_scan: 0.6 confidence, 80% win
        for i in range(10):
            signals.append(_make_signal(0.6, i < 8, source="news_scan"))
        # chart_scan: 0.6 confidence, 30% win
        for i in range(10):
            signals.append(_make_signal(0.6, i < 3, source="chart_scan"))
        with open(signal_path, "w") as f:
            json.dump(signals, f)

        cal = ConfidenceCalibrator(min_signals_per_bucket=5, signal_file=signal_path, calibration_file=cal_path)
        result = cal.analyze()

        assert "news_scan" in result["per_source"]
        assert "chart_scan" in result["per_source"]
        assert result["per_source"]["news_scan"]["actual_win_rate"] == 0.8
        assert result["per_source"]["chart_scan"]["actual_win_rate"] == 0.3

    def test_source_specific_calibration(self, tmp_dirs):
        signal_path, cal_path = tmp_dirs
        signals = []
        for i in range(10):
            signals.append(_make_signal(0.6, i < 8, source="news_scan"))
        for i in range(10):
            signals.append(_make_signal(0.6, i < 3, source="chart_scan"))
        with open(signal_path, "w") as f:
            json.dump(signals, f)

        cal = ConfidenceCalibrator(min_signals_per_bucket=5, signal_file=signal_path, calibration_file=cal_path)
        news_adj = cal.calibrate_confidence(0.6, source="news_scan")
        chart_adj = cal.calibrate_confidence(0.6, source="chart_scan")
        # news_scan wins more, so its adjusted confidence should be higher
        assert news_adj > chart_adj


class TestPersistence:
    def test_persist_creates_file(self, tmp_dirs):
        signal_path, cal_path = tmp_dirs
        # Remove cal_path to test creation
        if os.path.exists(cal_path):
            os.unlink(cal_path)

        signals = [_make_signal(0.6, True) for _ in range(6)]
        with open(signal_path, "w") as f:
            json.dump(signals, f)

        cal = ConfidenceCalibrator(min_signals_per_bucket=5, signal_file=signal_path, calibration_file=cal_path)
        cal.persist_calibration()

        assert os.path.exists(cal_path)
        with open(cal_path) as f:
            data = json.load(f)
        assert "buckets" in data
        assert "overall_calibration" in data

    def test_persist_after_analyze(self, tmp_dirs):
        signal_path, cal_path = tmp_dirs
        signals = [_make_signal(0.6, True) for _ in range(6)]
        with open(signal_path, "w") as f:
            json.dump(signals, f)

        cal = ConfidenceCalibrator(min_signals_per_bucket=5, signal_file=signal_path, calibration_file=cal_path)
        cal.analyze()
        cal.persist_calibration()

        with open(cal_path) as f:
            data = json.load(f)
        assert data["total_signals"] == 6


class TestWellCalibrated:
    def test_well_calibrated(self, tmp_dirs):
        signal_path, cal_path = tmp_dirs
        # 0.6 confidence, 60% win rate -> well calibrated
        signals = []
        for i in range(10):
            signals.append(_make_signal(0.6, i < 6))
        with open(signal_path, "w") as f:
            json.dump(signals, f)

        cal = ConfidenceCalibrator(min_signals_per_bucket=5, signal_file=signal_path, calibration_file=cal_path)
        result = cal.analyze()
        assert result["overall_calibration"] == "well_calibrated"

    def test_well_calibrated_returns_similar_confidence(self, tmp_dirs):
        signal_path, cal_path = tmp_dirs
        signals = []
        for i in range(10):
            signals.append(_make_signal(0.6, i < 6))
        with open(signal_path, "w") as f:
            json.dump(signals, f)

        cal = ConfidenceCalibrator(min_signals_per_bucket=5, signal_file=signal_path, calibration_file=cal_path)
        adjusted = cal.calibrate_confidence(0.6)
        # Should be close to original
        assert abs(adjusted - 0.6) < 0.1


class TestInsufficientBucketData:
    def test_bucket_with_few_signals_uses_factor_1(self, tmp_dirs):
        signal_path, cal_path = tmp_dirs
        # Only 2 signals in high bucket (below min_signals=5)
        signals = [_make_signal(0.75, True), _make_signal(0.75, False)]
        with open(signal_path, "w") as f:
            json.dump(signals, f)

        cal = ConfidenceCalibrator(min_signals_per_bucket=5, signal_file=signal_path, calibration_file=cal_path)
        result = cal.analyze()
        assert result["buckets"]["high"]["calibration_factor"] == 1.0
        assert result["buckets"]["high"]["sufficient_data"] is False

        # Calibration should return raw confidence
        assert cal.calibrate_confidence(0.75) == 0.75
