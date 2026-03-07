"""Confidence Calibrator — compares agent confidence vs actual outcomes.

Tier 0: Pure Python, no LLM calls, $0 cost.
Detects systematic over/under-confidence and applies correction factors.
Reads from data/signal_accuracy.json, persists calibration to data/confidence_calibration.json.
"""

from __future__ import annotations

import json
import os
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.confidence_calibrator")

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SIGNAL_FILE = os.path.join(_DATA_DIR, "signal_accuracy.json")
CALIBRATION_FILE = os.path.join(_DATA_DIR, "confidence_calibration.json")

CONFIDENCE_BUCKETS = [
    (0.0, 0.3, "low"),
    (0.3, 0.5, "medium_low"),
    (0.5, 0.7, "medium"),
    (0.7, 0.85, "high"),
    (0.85, 1.0, "very_high"),
]


def _bucket_midpoint(lo: float, hi: float) -> float:
    return round((lo + hi) / 2, 4)


def _assign_bucket(confidence: float) -> str | None:
    """Return the bucket label for a confidence value, or None if out of range."""
    for lo, hi, label in CONFIDENCE_BUCKETS:
        if lo <= confidence < hi:
            return label
    # Edge case: exactly 1.0 goes into the last bucket
    if confidence == 1.0:
        return CONFIDENCE_BUCKETS[-1][2]
    return None


class ConfidenceCalibrator:
    """Analyzes confidence calibration and applies correction factors."""

    def __init__(
        self,
        min_signals_per_bucket: int = 5,
        signal_file: str = SIGNAL_FILE,
        calibration_file: str = CALIBRATION_FILE,
    ) -> None:
        self._min_signals = min_signals_per_bucket
        self._signal_file = signal_file
        self._calibration_file = calibration_file
        self._cached_analysis: dict[str, Any] | None = None

    def _load_signals(self) -> list[dict[str, Any]]:
        """Load signal data from disk."""
        if not os.path.exists(self._signal_file):
            return []
        try:
            with open(self._signal_file) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def analyze(self) -> dict[str, Any]:
        """Analyze confidence calibration across all signals.

        Returns a dict with bucket-level calibration data, overall assessment,
        and per-source breakdowns.
        """
        signals = self._load_signals()

        # Only use signals that have been resolved (signal_correct is set)
        resolved = [
            s for s in signals
            if s.get("signal_correct") is not None
            and s.get("pipeline_outcome") == "executed"
        ]

        total = len(resolved)
        sufficient = total >= self._min_signals

        # Build bucket analysis
        buckets: dict[str, dict[str, Any]] = {}
        for lo, hi, label in CONFIDENCE_BUCKETS:
            mid = _bucket_midpoint(lo, hi)
            in_bucket = [
                s for s in resolved
                if _assign_bucket(s.get("confidence", s.get("signal_strength", 0))) == label
            ]
            wins = sum(1 for s in in_bucket if s.get("signal_correct") is True)
            count = len(in_bucket)
            actual_wr = round(wins / count, 4) if count > 0 else 0.0
            cal_factor = round(actual_wr / mid, 4) if mid > 0 and count >= self._min_signals else 1.0

            buckets[label] = {
                "signals": count,
                "wins": wins,
                "actual_win_rate": actual_wr,
                "expected_midpoint": mid,
                "calibration_factor": cal_factor,
                "sufficient_data": count >= self._min_signals,
            }

        # Per-source analysis
        sources: dict[str, list[dict[str, Any]]] = {}
        for s in resolved:
            src = s.get("source_type", s.get("source", "unknown"))
            sources.setdefault(src, []).append(s)

        per_source: dict[str, dict[str, Any]] = {}
        for src, src_signals in sources.items():
            wins = sum(1 for s in src_signals if s.get("signal_correct") is True)
            count = len(src_signals)
            avg_conf = round(
                sum(s.get("confidence", s.get("signal_strength", 0)) for s in src_signals) / count, 4
            ) if count > 0 else 0.0
            actual_wr = round(wins / count, 4) if count > 0 else 0.0
            per_source[src] = {
                "signals": count,
                "wins": wins,
                "actual_win_rate": actual_wr,
                "avg_confidence": avg_conf,
                "calibration_factor": round(actual_wr / avg_conf, 4) if avg_conf > 0 and count >= self._min_signals else 1.0,
            }

        # Overall calibration assessment
        weighted_conf = 0.0
        weighted_wr = 0.0
        total_weighted = 0
        for lo, hi, label in CONFIDENCE_BUCKETS:
            b = buckets[label]
            if b["signals"] > 0:
                mid = _bucket_midpoint(lo, hi)
                weighted_conf += mid * b["signals"]
                weighted_wr += b["actual_win_rate"] * b["signals"]
                total_weighted += b["signals"]

        if total_weighted > 0:
            avg_expected = weighted_conf / total_weighted
            avg_actual = weighted_wr / total_weighted
            if avg_actual < avg_expected * 0.85:
                overall = "overconfident"
            elif avg_actual > avg_expected * 1.15:
                overall = "underconfident"
            else:
                overall = "well_calibrated"
        else:
            overall = "insufficient_data"

        result = {
            "sufficient_data": sufficient,
            "total_signals": total,
            "buckets": buckets,
            "overall_calibration": overall,
            "per_source": per_source,
        }

        self._cached_analysis = result
        return result

    def calibrate_confidence(self, raw_confidence: float, source: str = "") -> float:
        """Apply calibration correction to a raw confidence score.

        Returns adjusted confidence clamped to [0, 1].
        Returns raw_confidence unchanged if insufficient data.
        """
        if self._cached_analysis is None:
            self.analyze()

        analysis = self._cached_analysis
        if not analysis or not analysis.get("sufficient_data"):
            return raw_confidence

        # Try source-specific calibration first
        if source and source in analysis.get("per_source", {}):
            src_data = analysis["per_source"][source]
            if src_data.get("signals", 0) >= self._min_signals:
                factor = src_data.get("calibration_factor", 1.0)
                adjusted = raw_confidence * factor
                return max(0.0, min(1.0, round(adjusted, 4)))

        # Fall back to bucket-level calibration
        bucket_label = _assign_bucket(raw_confidence)
        if bucket_label and bucket_label in analysis.get("buckets", {}):
            bucket = analysis["buckets"][bucket_label]
            if bucket.get("sufficient_data"):
                factor = bucket.get("calibration_factor", 1.0)
                adjusted = raw_confidence * factor
                return max(0.0, min(1.0, round(adjusted, 4)))

        return raw_confidence

    def persist_calibration(self) -> None:
        """Save calibration analysis to data/confidence_calibration.json."""
        if self._cached_analysis is None:
            self.analyze()
        try:
            os.makedirs(os.path.dirname(self._calibration_file), exist_ok=True)
            with open(self._calibration_file, "w") as f:
                json.dump(self._cached_analysis, f, indent=2)
            log.info("Calibration data persisted to %s", self._calibration_file)
        except Exception as e:
            log.warning("Failed to persist calibration: %s", e)
