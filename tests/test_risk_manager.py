"""Tests for RiskManager — all 7 validation checks + position sizing."""

from unittest.mock import patch

import pytest

from core.risk_manager import RiskManager


class TestValidateOrder:
    """Test all 7 risk checks in validate_order."""

    @patch.object(RiskManager, "_check_correlation", return_value=(True, ""))
    def test_valid_order_approved(self, mock_corr, risk_config, sample_execution_order, sample_portfolio_state):
        rm = RiskManager(risk_config)
        approved, reason, order = rm.validate_order(sample_execution_order, sample_portfolio_state)
        assert approved is True
        assert reason == "APPROVED"
        assert order is not None

    def test_check1_position_size_exceeds_max(self, risk_config, sample_portfolio_state):
        rm = RiskManager(risk_config)
        order = {
            "asset": "ETH",
            "direction": "long",
            "position_size_pct": 10.0,
            "stop_loss": 3000.0,
        }
        approved, reason, adjusted = rm.validate_order(order, sample_portfolio_state)
        assert approved is False
        assert "exceeds max" in reason

    def test_check2_daily_loss_limit(self, risk_config, sample_execution_order):
        rm = RiskManager(risk_config)
        state = {
            "daily_pnl_pct": -5.0,
            "drawdown_from_peak_pct": 3.0,
            "open_positions": [],
        }
        approved, reason, _ = rm.validate_order(sample_execution_order, state)
        assert approved is False
        assert "DAILY LOSS LIMIT" in reason

    def test_check3_max_drawdown(self, risk_config, sample_execution_order):
        rm = RiskManager(risk_config)
        state = {
            "daily_pnl_pct": -1.0,
            "drawdown_from_peak_pct": 15.0,
            "open_positions": [],
        }
        approved, reason, _ = rm.validate_order(sample_execution_order, state)
        assert approved is False
        assert "MAX DRAWDOWN" in reason

    def test_check4_max_open_positions(self, risk_config, sample_execution_order):
        rm = RiskManager(risk_config)
        state = {
            "daily_pnl_pct": 0.0,
            "drawdown_from_peak_pct": 0.0,
            "open_positions": [
                {"asset": "BTC", "trade_id": "t1"},
                {"asset": "ETH", "trade_id": "t2"},
                {"asset": "GLDM", "trade_id": "t3"},
            ],
        }
        approved, reason, _ = rm.validate_order(sample_execution_order, state)
        assert approved is False
        assert "positions reached" in reason

    @patch.object(RiskManager, "_check_correlation", return_value=(True, ""))
    def test_check5_duplicate_asset_allowed_with_warning(self, mock_corr, risk_config, sample_portfolio_state):
        """Duplicate asset should be allowed through (DA handles sizing)."""
        rm = RiskManager(risk_config)
        order = {
            "asset": "BTC",
            "direction": "long",
            "position_size_pct": 3.0,
            "stop_loss": 60000.0,
        }
        approved, reason, _ = rm.validate_order(order, sample_portfolio_state)
        assert approved is True

    def test_check6_no_stop_loss_rejected(self, risk_config, sample_portfolio_state):
        rm = RiskManager(risk_config)
        order = {
            "asset": "SLV",
            "direction": "long",
            "position_size_pct": 3.0,
        }
        approved, reason, _ = rm.validate_order(order, sample_portfolio_state)
        assert approved is False
        assert "stop-loss" in reason.lower()

    def test_empty_portfolio_valid_order(self, risk_config, sample_execution_order, empty_portfolio_state):
        rm = RiskManager(risk_config)
        approved, reason, order = rm.validate_order(sample_execution_order, empty_portfolio_state)
        assert approved is True


class TestCalculatePositionSize:
    def test_basic_sizing(self, risk_config):
        rm = RiskManager(risk_config)
        size = rm.calculate_position_size(confidence=0.8, atr=500.0, portfolio_value=100.0)
        assert size > 0
        assert size <= 100.0 * (risk_config["max_position_pct"] / 100.0)

    def test_zero_atr_returns_zero(self, risk_config):
        rm = RiskManager(risk_config)
        assert rm.calculate_position_size(confidence=0.8, atr=0.0, portfolio_value=100.0) == 0.0

    def test_zero_portfolio_returns_zero(self, risk_config):
        rm = RiskManager(risk_config)
        assert rm.calculate_position_size(confidence=0.8, atr=500.0, portfolio_value=0.0) == 0.0

    def test_higher_confidence_larger_position(self, risk_config):
        rm = RiskManager(risk_config)
        low = rm.calculate_position_size(confidence=0.3, atr=500.0, portfolio_value=100.0)
        high = rm.calculate_position_size(confidence=0.9, atr=500.0, portfolio_value=100.0)
        assert high > low

    def test_capped_at_max_position(self, risk_config):
        rm = RiskManager(risk_config)
        # Very small ATR should produce a huge position — verify cap
        size = rm.calculate_position_size(confidence=1.0, atr=0.001, portfolio_value=100.0)
        max_value = 100.0 * (risk_config["max_position_pct"] / 100.0)
        assert size == pytest.approx(max_value)

    def test_confidence_clamped(self, risk_config):
        rm = RiskManager(risk_config)
        # Confidence > 1.0 should be clamped
        normal = rm.calculate_position_size(confidence=1.0, atr=500.0, portfolio_value=100.0)
        over = rm.calculate_position_size(confidence=1.5, atr=500.0, portfolio_value=100.0)
        assert normal == over


class TestExposureRatio:
    """Test Check 9: Exposure ratio gate."""

    def _make_config(self, max_exposure_ratio=0.30):
        return {
            "max_position_pct": 7.0,
            "max_daily_loss_pct": 5.0,
            "max_total_drawdown_pct": 15.0,
            "max_open_positions": 10,
            "max_correlation": 0.50,
            "stop_loss_atr_mult": 2.0,
            "base_risk_per_trade_pct": 2.0,
            "max_exposure_ratio": max_exposure_ratio,
        }

    def _make_order(self, asset="ETH"):
        return {
            "asset": asset,
            "direction": "long",
            "position_size_pct": 5.0,
            "stop_loss": 3000.0,
            "confidence": 0.5,
        }

    @patch.object(RiskManager, "_check_correlation", return_value=(True, ""))
    def test_exposure_below_threshold_approved(self, mock_corr):
        """ER=20% should be approved."""
        rm = RiskManager(self._make_config())
        portfolio = {
            "equity": 10000.0,
            "daily_pnl_pct": 0.0,
            "drawdown_from_peak_pct": 0.0,
            "open_positions": [
                {"asset": "AAPL", "quantity": 10, "current_price": 200.0},  # $2000 / $10000 = 20%
            ],
        }
        approved, reason, _ = rm.validate_order(self._make_order(), portfolio)
        assert approved is True
        assert reason == "APPROVED"

    @patch.object(RiskManager, "_check_correlation", return_value=(True, ""))
    def test_exposure_above_threshold_blocked(self, mock_corr):
        """ER=35% should be blocked."""
        rm = RiskManager(self._make_config())
        portfolio = {
            "equity": 10000.0,
            "daily_pnl_pct": 0.0,
            "drawdown_from_peak_pct": 0.0,
            "open_positions": [
                {"asset": "AAPL", "quantity": 20, "current_price": 175.0},  # $3500 / $10000 = 35%
            ],
        }
        approved, reason, _ = rm.validate_order(self._make_order(), portfolio)
        assert approved is False
        assert "Exposure ratio" in reason

    @patch.object(RiskManager, "_check_correlation", return_value=(True, ""))
    def test_exposure_at_threshold_blocked(self, mock_corr):
        """ER=exactly 30% should be blocked (uses >=)."""
        rm = RiskManager(self._make_config())
        portfolio = {
            "equity": 10000.0,
            "daily_pnl_pct": 0.0,
            "drawdown_from_peak_pct": 0.0,
            "open_positions": [
                {"asset": "AAPL", "quantity": 20, "current_price": 150.0},  # $3000 / $10000 = 30%
            ],
        }
        approved, reason, _ = rm.validate_order(self._make_order(), portfolio)
        assert approved is False
        assert "Exposure ratio" in reason

    @patch.object(RiskManager, "_check_correlation", return_value=(True, ""))
    def test_exposure_empty_portfolio_approved(self, mock_corr):
        """No positions means ER=0%, should be approved."""
        rm = RiskManager(self._make_config())
        portfolio = {
            "equity": 10000.0,
            "daily_pnl_pct": 0.0,
            "drawdown_from_peak_pct": 0.0,
            "open_positions": [],
        }
        approved, reason, _ = rm.validate_order(self._make_order(), portfolio)
        assert approved is True

    @patch.object(RiskManager, "_check_correlation", return_value=(True, ""))
    def test_exposure_zero_equity_blocked(self, mock_corr):
        """Zero equity should be blocked (returns 999.0)."""
        rm = RiskManager(self._make_config())
        portfolio = {
            "equity": 0.0,
            "daily_pnl_pct": 0.0,
            "drawdown_from_peak_pct": 0.0,
            "open_positions": [
                {"asset": "AAPL", "quantity": 10, "current_price": 150.0},
            ],
        }
        approved, reason, _ = rm.validate_order(self._make_order(), portfolio)
        assert approved is False
        assert "Exposure ratio" in reason

    @patch.object(RiskManager, "_check_correlation", return_value=(True, ""))
    def test_exposure_missing_current_price_uses_entry(self, mock_corr):
        """Should fall back to entry_price when current_price is missing."""
        rm = RiskManager(self._make_config())
        portfolio = {
            "equity": 10000.0,
            "daily_pnl_pct": 0.0,
            "drawdown_from_peak_pct": 0.0,
            "open_positions": [
                {"asset": "AAPL", "quantity": 10, "entry_price": 200.0},  # $2000 / $10000 = 20%
            ],
        }
        approved, reason, _ = rm.validate_order(self._make_order(), portfolio)
        assert approved is True
        assert reason == "APPROVED"

    @patch.object(RiskManager, "_check_correlation", return_value=(True, ""))
    def test_exposure_custom_config(self, mock_corr):
        """Custom max_exposure_ratio=0.50, ER=40% should be approved."""
        rm = RiskManager(self._make_config(max_exposure_ratio=0.50))
        portfolio = {
            "equity": 10000.0,
            "daily_pnl_pct": 0.0,
            "drawdown_from_peak_pct": 0.0,
            "open_positions": [
                {"asset": "AAPL", "quantity": 20, "current_price": 200.0},  # $4000 / $10000 = 40%
            ],
        }
        approved, reason, _ = rm.validate_order(self._make_order(), portfolio)
        assert approved is True
        assert reason == "APPROVED"
