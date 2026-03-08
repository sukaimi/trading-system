"""Tests for asymmetric Reward:Risk ratio enforcement in the pipeline.

The R:R check rejects trades where reward < min_ratio * risk (default 2:1).
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Must set MOCK_LLM before importing pipeline
os.environ["MOCK_LLM"] = "true"

from core.pipeline import TradingPipeline


# ── Helper: build a minimal pipeline with mocked dependencies ──────────


def _make_pipeline(risk_params_override: dict | None = None):
    """Create a TradingPipeline with mocked components for R:R testing."""
    base_params = {
        "max_position_pct": 7.0,
        "max_daily_loss_pct": 5.0,
        "max_total_drawdown_pct": 15.0,
        "max_open_positions": 20,
        "max_correlation": 0.50,
        "stop_loss_atr_mult": 2.0,
        "base_risk_per_trade_pct": 2.0,
        "default_stop_loss_pct": 3.0,
        "default_take_profit_pct": 5.0,
        "take_profit_atr_mult": 3.0,
        "min_reward_risk_ratio": 2.0,
        "trailing_stop_activation_pct": 2.0,
        "trailing_stop_distance_pct": 1.5,
        "trailing_stop_atr_mult": 1.0,
        "circuit_breaker_drawdown_pct": 25.0,
        "permanent_halt_drawdown_pct": 25.0,
        "trading_friction": {
            "enabled": False,
            "spread_pct": {"stock": 0.0, "etf": 0.0, "crypto": 0.0},
            "commission_pct": {"stock": 0.0, "etf": 0.0, "crypto": 0.0},
            "short_borrow_annual_pct": {"stock": 0.0, "etf": 0.0, "crypto": 0.0},
        },
    }
    if risk_params_override:
        base_params.update(risk_params_override)

    # Write temp risk_params.json
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="risk_params_"
    )
    json.dump(base_params, tmp)
    tmp.close()

    with patch("core.pipeline.Path") as mock_path_cls:
        # Make the risk_params.json read return our temp file
        mock_path_instance = MagicMock()
        mock_path_instance.resolve.return_value = mock_path_instance
        mock_path_instance.parent = mock_path_instance
        mock_path_instance.__truediv__ = lambda self, x: mock_path_instance
        mock_path_instance.read_text.return_value = json.dumps(base_params)
        mock_path_cls.return_value = mock_path_instance
        mock_path_cls.__call__ = lambda self, *a: mock_path_instance

        pipeline = TradingPipeline.__new__(TradingPipeline)
        pipeline._risk_params = base_params

    os.unlink(tmp.name)
    return pipeline


# ── Tests for _check_reward_risk_ratio ─────────────────────────────────


class TestRewardRiskRatioCheck:
    """Tests for the _check_reward_risk_ratio method."""

    def setup_method(self):
        self.pipeline = _make_pipeline()

    def test_long_good_rr_passes(self):
        """Long trade with 3:1 R:R should pass (above 2:1 minimum)."""
        order = {"stop_loss": 95.0, "take_profit": 115.0}
        # Risk = 100 - 95 = 5, Reward = 115 - 100 = 15, R:R = 3.0
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="long"
        )
        assert passes is True
        assert ratio == 3.0
        assert reason == "passed"

    def test_long_bad_rr_rejected(self):
        """Long trade with 1:1 R:R should be rejected."""
        order = {"stop_loss": 95.0, "take_profit": 105.0}
        # Risk = 100 - 95 = 5, Reward = 105 - 100 = 5, R:R = 1.0
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="long"
        )
        assert passes is False
        assert ratio == 1.0
        assert "R:R 1.00:1 below minimum 2.0:1" in reason

    def test_long_exact_minimum_passes(self):
        """Long trade with exactly 2:1 R:R should pass."""
        order = {"stop_loss": 95.0, "take_profit": 110.0}
        # Risk = 5, Reward = 10, R:R = 2.0
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="long"
        )
        assert passes is True
        assert ratio == 2.0

    def test_short_good_rr_passes(self):
        """Short trade with 3:1 R:R should pass."""
        order = {"stop_loss": 105.0, "take_profit": 85.0}
        # Risk = 105 - 100 = 5, Reward = 100 - 85 = 15, R:R = 3.0
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="short"
        )
        assert passes is True
        assert ratio == 3.0

    def test_short_bad_rr_rejected(self):
        """Short trade with 1:1 R:R should be rejected."""
        order = {"stop_loss": 105.0, "take_profit": 95.0}
        # Risk = 105 - 100 = 5, Reward = 100 - 95 = 5, R:R = 1.0
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="short"
        )
        assert passes is False
        assert ratio == 1.0
        assert "below minimum" in reason

    def test_missing_stop_loss_skips_check(self):
        """Missing stop_loss should skip the check (pass)."""
        order = {"stop_loss": None, "take_profit": 110.0}
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="long"
        )
        assert passes is True
        assert "skipped" in reason

    def test_missing_take_profit_skips_check(self):
        """Missing take_profit should skip the check (pass)."""
        order = {"stop_loss": 95.0, "take_profit": None}
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="long"
        )
        assert passes is True
        assert "skipped" in reason

    def test_zero_entry_price_skips_check(self):
        """Zero current_price should skip the check (pass)."""
        order = {"stop_loss": 95.0, "take_profit": 110.0}
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=0, direction="long"
        )
        assert passes is True
        assert "skipped" in reason

    def test_missing_both_skips_check(self):
        """Missing both stop_loss and take_profit should skip."""
        order = {}
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="long"
        )
        assert passes is True

    def test_stop_loss_equals_entry_rejects(self):
        """stop_loss == entry_price means zero risk, should reject."""
        order = {"stop_loss": 100.0, "take_profit": 110.0}
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="long"
        )
        assert passes is False
        assert "invalid risk" in reason

    def test_stop_loss_beyond_entry_rejects(self):
        """Long with stop_loss > entry_price means negative risk, should reject."""
        order = {"stop_loss": 105.0, "take_profit": 110.0}
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="long"
        )
        assert passes is False
        assert "invalid risk" in reason

    def test_configurable_min_ratio(self):
        """Custom min_reward_risk_ratio from risk_params should be honored."""
        pipeline = _make_pipeline({"min_reward_risk_ratio": 3.0})
        order = {"stop_loss": 95.0, "take_profit": 110.0}
        # R:R = 2.0 — below 3.0 minimum
        passes, ratio, reason = pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="long"
        )
        assert passes is False
        assert ratio == 2.0
        assert "3.0:1" in reason

    def test_configurable_min_ratio_lenient(self):
        """Lower min_reward_risk_ratio should accept trades with lower R:R."""
        pipeline = _make_pipeline({"min_reward_risk_ratio": 1.0})
        order = {"stop_loss": 95.0, "take_profit": 106.0}
        # R:R = 1.2 — above 1.0 minimum
        passes, ratio, reason = pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="long"
        )
        assert passes is True
        assert ratio == 1.2

    def test_large_rr_passes(self):
        """Very large R:R (10:1) should pass easily."""
        order = {"stop_loss": 99.0, "take_profit": 110.0}
        # Risk = 1, Reward = 10, R:R = 10.0
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="long"
        )
        assert passes is True
        assert ratio == 10.0

    def test_just_below_minimum_rejects(self):
        """R:R of 1.99 should be rejected when minimum is 2.0."""
        order = {"stop_loss": 90.0, "take_profit": 119.9}
        # Risk = 10, Reward = 19.9, R:R = 1.99
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="long"
        )
        assert passes is False
        assert ratio == 1.99

    def test_crypto_high_price_long(self):
        """R:R check works with high-price assets like BTC."""
        order = {"stop_loss": 95000.0, "take_profit": 115000.0}
        # Risk = 5000, Reward = 15000, R:R = 3.0
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=100000.0, direction="long"
        )
        assert passes is True
        assert ratio == 3.0

    def test_short_stop_equals_entry_rejects(self):
        """Short with stop_loss == entry_price means zero risk, should reject."""
        order = {"stop_loss": 100.0, "take_profit": 90.0}
        passes, ratio, reason = self.pipeline._check_reward_risk_ratio(
            order, current_price=100.0, direction="short"
        )
        assert passes is False
        assert "invalid risk" in reason


# ── Tests for TP floor after regime adjustment ─────────────────────────


def _make_pipeline_for_build(risk_params_override: dict | None = None):
    """Create a TradingPipeline with mocked components for _build_execution_order testing."""
    pipeline = _make_pipeline(risk_params_override)
    pipeline._risk = MagicMock()
    pipeline._risk.check_trade = MagicMock(return_value={"approved": True})
    pipeline._risk.calculate_position_size = MagicMock(return_value=100.0)
    pipeline._portfolio = MagicMock()
    pipeline._portfolio.equity = 10000.0
    pipeline._portfolio.consecutive_losses = 0
    pipeline._executor = MagicMock()
    pipeline._signal_tracker = MagicMock()
    pipeline._confidence_cal = MagicMock()
    pipeline._confidence_cal.calibrate_confidence = lambda c: c
    pipeline._earnings_cal = MagicMock()
    pipeline._earnings_cal.has_earnings_soon.return_value = False
    pipeline._session_analyzer = MagicMock()
    pipeline._session_analyzer.classify_session.return_value = "US"
    pipeline._session_analyzer.get_session_weight.return_value = 1.0
    pipeline._regime_classifier = MagicMock()
    pipeline._regime_classifier.classify.return_value = {
        "regime": "RANGING", "confidence": 0.7, "indicators": {}
    }
    pipeline._regime_classifier.get_initial_stop_multiplier.return_value = 1.0
    pipeline._regime_strategy = MagicMock()
    pipeline._regime_strategy.get_adjustments.return_value = {
        "position_size_mult": 0.7,
        "take_profit_mult": 0.8,
        "stop_loss_mult": 1.0,
        "trailing_stop_mult": 1.5,
        "confidence_adjustment": 0.0,
    }
    pipeline._regime_strategy.should_trade.return_value = (True, "ok")
    return pipeline


class TestTPFloorAfterRegimeAdjustment:
    """Tests that the TP floor ensures take-profit never drops below min R:R
    after regime-based TP multiplier is applied."""

    def test_long_tp_floor_kicks_in(self):
        """Long: regime adjustment pushes TP below min R:R, floor corrects it."""
        from core.schemas import TradeThesis, Direction, DevilsVerdict, Verdict

        pipeline = _make_pipeline_for_build({"min_reward_risk_ratio": 2.0})

        thesis = TradeThesis(
            asset="BTC", direction=Direction.LONG, confidence=0.8,
            thesis="test", suggested_position_pct=5.0,
            invalidation_level="62000", risk_reward_ratio="1.5",
            supporting_data={"current_price": 65000},
        )
        verdict = DevilsVerdict(
            original_thesis_id="x", verdict=Verdict.APPROVED,
            confidence_adjusted=0.75,
        )
        order = pipeline._build_execution_order(thesis, verdict)
        # stop = 62000, risk = 3000
        # Raw TP: reward = 3000 * 1.5 = 4500, TP = 69500
        # Regime 0.8x: TP distance = 4500 * 0.8 = 3600, TP = 68600
        # R:R = 3600/3000 = 1.2 — below min 2.0
        # TP floor: min_tp = 65000 + 3000*2 = 71000
        assert order["take_profit"] == 71000.0
        assert order["stop_loss"] == 62000.0

    def test_short_tp_floor_kicks_in(self):
        """Short: regime adjustment pushes TP too close to entry, floor corrects."""
        from core.schemas import TradeThesis, Direction, DevilsVerdict, Verdict

        pipeline = _make_pipeline_for_build({"min_reward_risk_ratio": 2.0})

        thesis = TradeThesis(
            asset="ETH", direction=Direction.SHORT, confidence=0.8,
            thesis="test", suggested_position_pct=5.0,
            invalidation_level="3500", risk_reward_ratio="1.5",
            supporting_data={"current_price": 3200},
        )
        verdict = DevilsVerdict(
            original_thesis_id="x", verdict=Verdict.APPROVED,
            confidence_adjusted=0.75,
        )
        order = pipeline._build_execution_order(thesis, verdict)
        # stop = 3500, risk = 300
        # Raw TP: reward = 300 * 1.5 = 450, TP = 2750
        # Regime 0.8x: TP distance = 450 * 0.8 = 360, TP = 2840
        # R:R = 360/300 = 1.2 — below min 2.0
        # TP floor: min_tp = 3200 - 300*2 = 2600
        assert order["take_profit"] == 2600.0
        assert order["stop_loss"] == 3500.0

    def test_long_tp_above_floor_not_adjusted(self):
        """Long: TP already above min R:R after regime adjustment, no floor applied."""
        from core.schemas import TradeThesis, Direction, DevilsVerdict, Verdict

        pipeline = _make_pipeline_for_build({"min_reward_risk_ratio": 1.0})

        thesis = TradeThesis(
            asset="BTC", direction=Direction.LONG, confidence=0.8,
            thesis="test", suggested_position_pct=5.0,
            invalidation_level="62000", risk_reward_ratio="1.5",
            supporting_data={"current_price": 65000},
        )
        verdict = DevilsVerdict(
            original_thesis_id="x", verdict=Verdict.APPROVED,
            confidence_adjusted=0.75,
        )
        order = pipeline._build_execution_order(thesis, verdict)
        # stop = 62000, risk = 3000
        # Raw TP: reward = 3000 * 1.5 = 4500, TP = 69500
        # Regime 0.8x: TP distance = 4500 * 0.8 = 3600, TP = 68600
        # R:R = 3600/3000 = 1.2 — above min 1.0, no floor
        assert order["take_profit"] == 68600.0
        assert order["stop_loss"] == 62000.0
