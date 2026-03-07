"""Phantom Trade Analyzer — detects patterns in killed trades.

Tier 0: Pure Python, no LLM calls, $0 cost.
Goes beyond simple false kill rate to find per-asset, per-killer, per-direction
biases and generate actionable bias alerts.
Reads from data/phantom_trades.json, persists to data/phantom_analysis.json.
"""

from __future__ import annotations

import json
import os
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.phantom_analyzer")

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PHANTOM_FILE = os.path.join(_DATA_DIR, "phantom_trades.json")
ANALYSIS_FILE = os.path.join(_DATA_DIR, "phantom_analysis.json")

CONFIDENCE_BUCKETS = [
    (0.0, 0.3, "low"),
    (0.3, 0.5, "medium_low"),
    (0.5, 0.7, "medium"),
    (0.7, 0.85, "high"),
    (0.85, 1.0, "very_high"),
]


def _assign_bucket(confidence: float) -> str:
    """Return the bucket label for a confidence value."""
    for lo, hi, label in CONFIDENCE_BUCKETS:
        if lo <= confidence < hi:
            return label
    if confidence >= 1.0:
        return CONFIDENCE_BUCKETS[-1][2]
    return "low"


class PhantomAnalyzer:
    """Comprehensive analysis of killed/rejected trades."""

    def __init__(
        self,
        min_phantoms: int = 5,
        phantom_file: str = PHANTOM_FILE,
        analysis_file: str = ANALYSIS_FILE,
    ) -> None:
        self._min_phantoms = min_phantoms
        self._phantom_file = phantom_file
        self._analysis_file = analysis_file
        self._cached_analysis: dict[str, Any] | None = None

    def _load_phantoms(self) -> list[dict[str, Any]]:
        """Load phantom trade data from disk."""
        if not os.path.exists(self._phantom_file):
            return []
        try:
            with open(self._phantom_file) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def analyze(self) -> dict[str, Any]:
        """Comprehensive phantom trade analysis.

        Returns per-asset, per-killer, per-direction, per-confidence-bucket
        false kill rates plus bias alerts and recommendations.
        """
        phantoms = self._load_phantoms()
        total = len(phantoms)
        checked = [p for p in phantoms if p.get("outcome_checked")]
        checked_count = len(checked)
        sufficient = checked_count >= self._min_phantoms

        # Overall false kill rate
        would_have_won = [p for p in checked if (p.get("outcome_pnl_pct") or 0) > 0]
        overall_fkr = round(len(would_have_won) / checked_count, 4) if checked_count > 0 else 0.0

        # Per asset
        per_asset = self._analyze_group(phantoms, checked, "asset")

        # Per killer
        per_killer = self._analyze_group(phantoms, checked, "killed_by")

        # Per direction
        per_direction = self._analyze_group(phantoms, checked, "direction")

        # Per confidence bucket
        per_confidence = self._analyze_confidence_buckets(phantoms, checked)

        # Generate bias alerts and recommendations
        bias_alerts = self._generate_bias_alerts(per_asset, per_killer, per_direction, per_confidence)
        recommendations = self._generate_recommendations(per_asset, per_killer, per_direction)

        result = {
            "sufficient_data": sufficient,
            "total_phantoms": total,
            "checked_phantoms": checked_count,
            "overall_false_kill_rate": overall_fkr,
            "per_asset": per_asset,
            "per_killer": per_killer,
            "per_direction": per_direction,
            "per_confidence_bucket": per_confidence,
            "bias_alerts": bias_alerts,
            "recommendations": recommendations,
        }

        self._cached_analysis = result
        return result

    def _analyze_group(
        self,
        all_phantoms: list[dict[str, Any]],
        checked: list[dict[str, Any]],
        key: str,
    ) -> dict[str, dict[str, Any]]:
        """Analyze false kill rate grouped by a specific field."""
        groups: dict[str, dict[str, Any]] = {}

        # Count total kills per group
        for p in all_phantoms:
            val = p.get(key, "unknown")
            if val not in groups:
                groups[val] = {"kills": 0, "checked": 0, "would_have_won": 0, "false_kill_rate": 0.0, "avg_missed_pnl_pct": 0.0}
            groups[val]["kills"] += 1

        # Count checked + would-have-won per group
        for p in checked:
            val = p.get(key, "unknown")
            if val not in groups:
                groups[val] = {"kills": 0, "checked": 0, "would_have_won": 0, "false_kill_rate": 0.0, "avg_missed_pnl_pct": 0.0}
            groups[val]["checked"] += 1
            pnl = p.get("outcome_pnl_pct") or 0
            if pnl > 0:
                groups[val]["would_have_won"] += 1

        # Compute rates and avg missed PnL
        for val, data in groups.items():
            c = data["checked"]
            w = data["would_have_won"]
            data["false_kill_rate"] = round(w / c, 4) if c > 0 else 0.0

            # Avg missed PnL for would-have-won trades
            won_pnls = [
                p.get("outcome_pnl_pct", 0)
                for p in checked
                if p.get(key, "unknown") == val and (p.get("outcome_pnl_pct") or 0) > 0
            ]
            data["avg_missed_pnl_pct"] = round(sum(won_pnls) / len(won_pnls), 2) if won_pnls else 0.0

        return groups

    def _analyze_confidence_buckets(
        self,
        all_phantoms: list[dict[str, Any]],
        checked: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Analyze false kill rate by confidence bucket."""
        buckets: dict[str, dict[str, Any]] = {}

        for _, _, label in CONFIDENCE_BUCKETS:
            buckets[label] = {"kills": 0, "checked": 0, "would_have_won": 0, "false_kill_rate": 0.0}

        for p in all_phantoms:
            conf = p.get("confidence", 0)
            label = _assign_bucket(conf)
            buckets[label]["kills"] += 1

        for p in checked:
            conf = p.get("confidence", 0)
            label = _assign_bucket(conf)
            buckets[label]["checked"] += 1
            if (p.get("outcome_pnl_pct") or 0) > 0:
                buckets[label]["would_have_won"] += 1

        for label, data in buckets.items():
            c = data["checked"]
            w = data["would_have_won"]
            data["false_kill_rate"] = round(w / c, 4) if c > 0 else 0.0

        return buckets

    def _generate_bias_alerts(
        self,
        per_asset: dict[str, dict[str, Any]],
        per_killer: dict[str, dict[str, Any]],
        per_direction: dict[str, dict[str, Any]],
        per_confidence: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Generate human-readable bias alerts from analysis."""
        alerts: list[str] = []

        # Asset-level alerts
        for asset, data in per_asset.items():
            if data["checked"] >= self._min_phantoms and data["false_kill_rate"] > 0.5:
                pct = round(data["false_kill_rate"] * 100)
                alerts.append(
                    f"{pct}% of killed {asset} trades would have been profitable "
                    f"(avg missed: {data['avg_missed_pnl_pct']:.1f}%)"
                )

        # Killer-level alerts
        for killer, data in per_killer.items():
            if data["checked"] >= self._min_phantoms:
                if data["false_kill_rate"] > 0.5:
                    pct = round(data["false_kill_rate"] * 100)
                    alerts.append(
                        f"{killer} kills too aggressively — {pct}% of kills were false"
                    )
                elif data["false_kill_rate"] < 0.2:
                    pct = round((1 - data["false_kill_rate"]) * 100)
                    alerts.append(
                        f"{killer} correctly blocks most trades ({pct}% accuracy)"
                    )

        # Direction-level alerts
        for direction, data in per_direction.items():
            if data["checked"] >= self._min_phantoms and data["false_kill_rate"] > 0.5:
                pct = round(data["false_kill_rate"] * 100)
                alerts.append(
                    f"{direction} trades are over-killed — {pct}% false kill rate"
                )

        # High-confidence kills alert
        for label in ("high", "very_high"):
            b = per_confidence.get(label, {})
            if b.get("checked", 0) >= self._min_phantoms and b.get("false_kill_rate", 0) > 0.5:
                pct = round(b["false_kill_rate"] * 100)
                alerts.append(
                    f"High-confidence trades ({label}) have {pct}% false kill rate — "
                    f"consider trusting strong signals more"
                )

        return alerts

    def _generate_recommendations(
        self,
        per_asset: dict[str, dict[str, Any]],
        per_killer: dict[str, dict[str, Any]],
        per_direction: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recs: list[str] = []

        # Asset-specific recommendations
        for asset, data in per_asset.items():
            if data["checked"] >= self._min_phantoms:
                if data["false_kill_rate"] > 0.6:
                    recs.append(
                        f"Consider loosening DA threshold for {asset} trades"
                    )
                elif data["false_kill_rate"] < 0.15 and data["checked"] >= self._min_phantoms:
                    recs.append(
                        f"Kill filters working well for {asset} — maintain current thresholds"
                    )

        # Killer-specific recommendations
        for killer, data in per_killer.items():
            if data["checked"] >= self._min_phantoms and data["false_kill_rate"] > 0.5:
                recs.append(
                    f"Review {killer} rejection criteria — too many false kills"
                )

        # Direction-specific recommendations
        for direction, data in per_direction.items():
            if data["checked"] >= self._min_phantoms and data["false_kill_rate"] > 0.6:
                recs.append(
                    f"DA may be biased against {direction} trades — review criteria"
                )

        return recs

    def get_asset_kill_bias(self, asset: str) -> float:
        """Return kill bias factor for an asset (0.5-1.5).

        > 1.0 means too many false kills (should be more lenient).
        < 1.0 means kills are accurate (maintain or tighten).
        Returns 1.0 if insufficient data.
        """
        if self._cached_analysis is None:
            self.analyze()

        analysis = self._cached_analysis
        if not analysis or not analysis.get("sufficient_data"):
            return 1.0

        per_asset = analysis.get("per_asset", {})
        asset_data = per_asset.get(asset)
        if not asset_data or asset_data.get("checked", 0) < self._min_phantoms:
            return 1.0

        fkr = asset_data.get("false_kill_rate", 0.5)
        # Map false kill rate to bias factor:
        # fkr=0.0 -> 0.5 (kills are accurate, keep tight)
        # fkr=0.5 -> 1.0 (neutral)
        # fkr=1.0 -> 1.5 (too many false kills, loosen)
        bias = 0.5 + fkr
        return round(max(0.5, min(1.5, bias)), 4)

    def persist_analysis(self) -> None:
        """Save analysis to data/phantom_analysis.json."""
        if self._cached_analysis is None:
            self.analyze()
        try:
            os.makedirs(os.path.dirname(self._analysis_file), exist_ok=True)
            with open(self._analysis_file, "w") as f:
                json.dump(self._cached_analysis, f, indent=2)
            log.info("Phantom analysis persisted to %s", self._analysis_file)
        except Exception as e:
            log.warning("Failed to persist phantom analysis: %s", e)
