"""Regime Strategy Selector — selects strategy parameters based on market regime.

Tier 0: Pure Python, no LLM calls, $0 cost.

Goes beyond adaptive stop distances to select different position sizes,
holding periods, take-profit targets, and confidence thresholds per regime.
Presets are loaded from config/regime_strategy.json, falling back to defaults.
"""

from __future__ import annotations

import json
import os
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.regime_strategy")

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "regime_strategy.json")

# Default strategy presets (overridable via config/regime_strategy.json)
DEFAULT_PRESETS: dict[str, dict[str, Any]] = {
    "TRENDING_UP": {
        "position_size_mult": 1.2,
        "max_hold_hours_mult": 1.5,
        "take_profit_mult": 1.3,
        "min_confidence": 0.4,
        "prefer_direction": "long",
        "direction_bonus": 0.1,
    },
    "TRENDING_DOWN": {
        "position_size_mult": 1.2,
        "max_hold_hours_mult": 1.5,
        "take_profit_mult": 1.3,
        "min_confidence": 0.4,
        "prefer_direction": "short",
        "direction_bonus": 0.1,
    },
    "RANGING": {
        "position_size_mult": 0.7,
        "max_hold_hours_mult": 0.7,
        "take_profit_mult": 0.8,
        "min_confidence": 0.55,
        "prefer_direction": None,
        "direction_bonus": 0.0,
    },
    "HIGH_VOLATILITY": {
        "position_size_mult": 0.5,
        "max_hold_hours_mult": 0.5,
        "take_profit_mult": 1.5,
        "min_confidence": 0.6,
        "prefer_direction": None,
        "direction_bonus": 0.0,
    },
    "LOW_VOLATILITY": {
        "position_size_mult": 1.0,
        "max_hold_hours_mult": 1.0,
        "take_profit_mult": 1.0,
        "min_confidence": 0.45,
        "prefer_direction": None,
        "direction_bonus": 0.0,
    },
}

# Neutral adjustments for unknown regimes
_NEUTRAL: dict[str, Any] = {
    "position_size_mult": 1.0,
    "max_hold_hours_mult": 1.0,
    "take_profit_mult": 1.0,
    "min_confidence": 0.5,
    "prefer_direction": None,
    "direction_bonus": 0.0,
}


class RegimeStrategySelector:
    """Select strategy parameters based on the current market regime."""

    def __init__(self) -> None:
        self._presets = self._load_presets()

    def _load_presets(self) -> dict[str, dict[str, Any]]:
        """Load presets from config/regime_strategy.json, fall back to defaults."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    # Merge with defaults so missing keys get filled
                    merged: dict[str, dict[str, Any]] = {}
                    for regime, defaults in DEFAULT_PRESETS.items():
                        override = loaded.get(regime, {})
                        merged[regime] = {**defaults, **override}
                    # Include any extra regimes from config
                    for regime in loaded:
                        if regime not in merged:
                            merged[regime] = {**_NEUTRAL, **loaded[regime]}
                    log.info("Loaded regime strategy presets from %s", CONFIG_FILE)
                    return merged
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Failed to load regime_strategy.json: %s — using defaults", e)

        return {k: dict(v) for k, v in DEFAULT_PRESETS.items()}

    def get_adjustments(self, regime: str, direction: str = "") -> dict[str, Any]:
        """Get strategy adjustments for the given regime.

        Args:
            regime: Market regime string (e.g. "TRENDING_UP", "RANGING").
            direction: Trade direction ("long" or "short").

        Returns:
            Dict with position_size_mult, max_hold_hours_mult,
            take_profit_mult, and confidence_adjustment.
        """
        preset = self._presets.get(regime, _NEUTRAL)

        # Calculate direction bonus
        confidence_adj = 0.0
        preferred = preset.get("prefer_direction")
        bonus = preset.get("direction_bonus", 0.0)
        if preferred and direction and direction.lower() == preferred.lower():
            confidence_adj = bonus

        return {
            "position_size_mult": preset.get("position_size_mult", 1.0),
            "max_hold_hours_mult": preset.get("max_hold_hours_mult", 1.0),
            "take_profit_mult": preset.get("take_profit_mult", 1.0),
            "confidence_adjustment": confidence_adj,
        }

    def should_trade(
        self, regime: str, confidence: float, direction: str = ""
    ) -> tuple[bool, str]:
        """Check if minimum confidence is met for this regime.

        Args:
            regime: Market regime string.
            confidence: Current trade confidence (0-1).
            direction: Trade direction ("long" or "short").

        Returns:
            (should_trade, reason) tuple.
        """
        preset = self._presets.get(regime, _NEUTRAL)
        min_conf = preset.get("min_confidence", 0.5)

        # Include direction bonus in effective confidence
        adjustments = self.get_adjustments(regime, direction)
        effective_conf = confidence + adjustments["confidence_adjustment"]
        effective_conf = min(effective_conf, 1.0)

        if effective_conf >= min_conf:
            return True, f"Confidence {effective_conf:.2f} >= {min_conf:.2f} for {regime}"

        return (
            False,
            f"Confidence {effective_conf:.2f} < {min_conf:.2f} minimum for {regime} regime",
        )

    @property
    def presets(self) -> dict[str, dict[str, Any]]:
        """Return current presets (read-only copy)."""
        return {k: dict(v) for k, v in self._presets.items()}
