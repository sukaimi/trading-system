"""Tests for core/regime_strategy.py"""

import json
import os
from unittest.mock import patch

import pytest

from core.regime_strategy import (
    DEFAULT_PRESETS,
    RegimeStrategySelector,
    _NEUTRAL,
)


class TestGetAdjustments:
    def test_trending_up_returns_correct_multipliers(self):
        selector = RegimeStrategySelector()
        adj = selector.get_adjustments("TRENDING_UP")
        assert adj["position_size_mult"] == 1.2
        assert adj["max_hold_hours_mult"] == 1.5
        assert adj["take_profit_mult"] == 1.3

    def test_trending_down_returns_correct_multipliers(self):
        selector = RegimeStrategySelector()
        adj = selector.get_adjustments("TRENDING_DOWN")
        assert adj["position_size_mult"] == 1.2
        assert adj["take_profit_mult"] == 1.3

    def test_ranging_returns_reduced_multipliers(self):
        selector = RegimeStrategySelector()
        adj = selector.get_adjustments("RANGING")
        assert adj["position_size_mult"] == 0.7
        assert adj["max_hold_hours_mult"] == 0.7
        assert adj["take_profit_mult"] == 0.8

    def test_high_volatility_returns_small_position(self):
        selector = RegimeStrategySelector()
        adj = selector.get_adjustments("HIGH_VOLATILITY")
        assert adj["position_size_mult"] == 0.5
        assert adj["max_hold_hours_mult"] == 0.5
        assert adj["take_profit_mult"] == 1.5

    def test_low_volatility_returns_neutral(self):
        selector = RegimeStrategySelector()
        adj = selector.get_adjustments("LOW_VOLATILITY")
        assert adj["position_size_mult"] == 1.0
        assert adj["max_hold_hours_mult"] == 1.0
        assert adj["take_profit_mult"] == 1.0

    def test_direction_bonus_for_matching_direction(self):
        selector = RegimeStrategySelector()
        adj = selector.get_adjustments("TRENDING_UP", "long")
        assert adj["confidence_adjustment"] == 0.1

    def test_direction_bonus_zero_for_non_matching_direction(self):
        selector = RegimeStrategySelector()
        adj = selector.get_adjustments("TRENDING_UP", "short")
        assert adj["confidence_adjustment"] == 0.0

    def test_direction_bonus_for_trending_down_short(self):
        selector = RegimeStrategySelector()
        adj = selector.get_adjustments("TRENDING_DOWN", "short")
        assert adj["confidence_adjustment"] == 0.1

    def test_direction_bonus_zero_for_trending_down_long(self):
        selector = RegimeStrategySelector()
        adj = selector.get_adjustments("TRENDING_DOWN", "long")
        assert adj["confidence_adjustment"] == 0.0

    def test_no_direction_bonus_for_ranging(self):
        selector = RegimeStrategySelector()
        adj = selector.get_adjustments("RANGING", "long")
        assert adj["confidence_adjustment"] == 0.0

    def test_unknown_regime_returns_neutral_defaults(self):
        selector = RegimeStrategySelector()
        adj = selector.get_adjustments("UNKNOWN_REGIME")
        assert adj["position_size_mult"] == 1.0
        assert adj["max_hold_hours_mult"] == 1.0
        assert adj["take_profit_mult"] == 1.0
        assert adj["confidence_adjustment"] == 0.0

    def test_empty_direction_string(self):
        selector = RegimeStrategySelector()
        adj = selector.get_adjustments("TRENDING_UP", "")
        assert adj["confidence_adjustment"] == 0.0


class TestShouldTrade:
    def test_sufficient_confidence(self):
        selector = RegimeStrategySelector()
        should, reason = selector.should_trade("TRENDING_UP", 0.5, "long")
        assert should is True
        assert "0.60" in reason  # 0.5 + 0.1 bonus

    def test_rejection_with_low_confidence(self):
        selector = RegimeStrategySelector()
        should, reason = selector.should_trade("HIGH_VOLATILITY", 0.3)
        assert should is False
        assert "0.60" in reason  # min_confidence for HIGH_VOLATILITY

    def test_direction_bonus_can_push_above_threshold(self):
        """Test that direction bonus helps meet the threshold."""
        selector = RegimeStrategySelector()
        # TRENDING_UP min_confidence=0.4, direction_bonus=0.1
        # Confidence 0.35 + bonus 0.1 = 0.45 >= 0.4
        should, _ = selector.should_trade("TRENDING_UP", 0.35, "long")
        assert should is True

    def test_direction_bonus_not_applied_wrong_direction(self):
        selector = RegimeStrategySelector()
        # TRENDING_UP min_confidence=0.4, but short gets no bonus
        # 0.35 < 0.4
        should, _ = selector.should_trade("TRENDING_UP", 0.35, "short")
        assert should is False

    def test_ranging_higher_threshold(self):
        selector = RegimeStrategySelector()
        should, _ = selector.should_trade("RANGING", 0.5)
        assert should is False  # 0.5 < 0.55

    def test_ranging_passes_at_threshold(self):
        selector = RegimeStrategySelector()
        should, _ = selector.should_trade("RANGING", 0.55)
        assert should is True

    def test_unknown_regime_uses_neutral(self):
        selector = RegimeStrategySelector()
        should, _ = selector.should_trade("MYSTERY", 0.5)
        assert should is True  # neutral min_confidence=0.5

    def test_confidence_capped_at_1(self):
        selector = RegimeStrategySelector()
        should, reason = selector.should_trade("TRENDING_UP", 0.95, "long")
        assert should is True
        assert "1.00" in reason  # 0.95 + 0.1 = 1.05 -> capped at 1.0


class TestCustomConfig:
    def test_override_from_config_file(self, tmp_path):
        config = {
            "TRENDING_UP": {
                "position_size_mult": 2.0,
                "min_confidence": 0.3,
            },
        }
        config_file = tmp_path / "regime_strategy.json"
        config_file.write_text(json.dumps(config))

        with patch("core.regime_strategy.CONFIG_FILE", str(config_file)):
            selector = RegimeStrategySelector()
            adj = selector.get_adjustments("TRENDING_UP")
            assert adj["position_size_mult"] == 2.0
            # Other defaults should be preserved
            assert adj["take_profit_mult"] == 1.3

    def test_invalid_config_falls_back_to_defaults(self, tmp_path):
        config_file = tmp_path / "regime_strategy.json"
        config_file.write_text("not json!!!")

        with patch("core.regime_strategy.CONFIG_FILE", str(config_file)):
            selector = RegimeStrategySelector()
            adj = selector.get_adjustments("TRENDING_UP")
            assert adj["position_size_mult"] == 1.2

    def test_missing_config_uses_defaults(self, tmp_path):
        with patch("core.regime_strategy.CONFIG_FILE", str(tmp_path / "nope.json")):
            selector = RegimeStrategySelector()
            adj = selector.get_adjustments("RANGING")
            assert adj["position_size_mult"] == 0.7

    def test_presets_property_returns_copy(self):
        selector = RegimeStrategySelector()
        presets = selector.presets
        assert "TRENDING_UP" in presets
        # Mutating returned copy should not affect internal state
        presets["TRENDING_UP"]["position_size_mult"] = 999.0
        assert selector.presets["TRENDING_UP"]["position_size_mult"] == 1.2


class TestPipelineIntegration:
    """Test that RegimeStrategySelector can adjust position sizes as expected in pipeline context."""

    def test_position_size_adjustment_trending_up(self):
        selector = RegimeStrategySelector()
        base_position_pct = 5.0
        adj = selector.get_adjustments("TRENDING_UP", "long")
        adjusted = base_position_pct * adj["position_size_mult"]
        assert adjusted == 6.0  # 5.0 * 1.2

    def test_position_size_adjustment_high_vol(self):
        selector = RegimeStrategySelector()
        base_position_pct = 5.0
        adj = selector.get_adjustments("HIGH_VOLATILITY")
        adjusted = base_position_pct * adj["position_size_mult"]
        assert adjusted == 2.5  # 5.0 * 0.5

    def test_take_profit_multiplier_applied(self):
        selector = RegimeStrategySelector()
        current_price = 100.0
        base_tp = 105.0  # 5% take-profit
        adj = selector.get_adjustments("TRENDING_UP", "long")

        tp_distance = abs(base_tp - current_price)
        tp_distance *= adj["take_profit_mult"]
        new_tp = current_price + tp_distance
        assert new_tp == pytest.approx(106.5)  # 5 * 1.3 = 6.5
