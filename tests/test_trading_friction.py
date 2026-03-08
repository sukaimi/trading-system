"""Tests for the trading friction module in core/trading_friction.py.

Simulates realistic trading costs (spread, commission, borrow) during paper trading.
Auto-disables when paper_mode=False (live trading).
"""

import json
import os
from unittest.mock import mock_open, patch

import pytest


# ── Fixture: friction config ────────────────────────────────────────


FRICTION_CONFIG = {
    "trading_friction": {
        "enabled": True,
        "spread_pct": {"stock": 0.05, "etf": 0.03, "crypto": 0.15},
        "commission_pct": {"stock": 0.0, "etf": 0.0, "crypto": 0.15},
        "short_borrow_annual_pct": {"stock": 1.5, "etf": 1.0, "crypto": 5.0},
    }
}

FRICTION_CONFIG_DISABLED = {
    "trading_friction": {
        "enabled": False,
        "spread_pct": {"stock": 0.05, "etf": 0.03, "crypto": 0.15},
        "commission_pct": {"stock": 0.0, "etf": 0.0, "crypto": 0.15},
        "short_borrow_annual_pct": {"stock": 1.5, "etf": 1.0, "crypto": 5.0},
    }
}


def _load_risk_params(config_data):
    """Helper to mock risk_params.json loading."""
    full_config = {
        "max_position_pct": 7.0,
        "max_daily_loss_pct": 5.0,
        **config_data,
    }
    return full_config


def _make_friction(paper_mode=True, config=None):
    """Create a TradingFriction instance with mocked config."""
    if config is None:
        config = FRICTION_CONFIG
    risk_params = _load_risk_params(config)
    with patch("core.trading_friction.json.load", return_value=risk_params), \
         patch("builtins.open", mock_open(read_data=json.dumps(risk_params))):
        from core.trading_friction import TradingFriction
        return TradingFriction(paper_mode=paper_mode)


@pytest.fixture
def friction_paper():
    """TradingFriction in paper mode (friction enabled)."""
    return _make_friction(paper_mode=True)


@pytest.fixture
def friction_live():
    """TradingFriction in live mode (friction auto-disabled)."""
    return _make_friction(paper_mode=False)


@pytest.fixture
def friction_disabled():
    """TradingFriction in paper mode but config disabled."""
    return _make_friction(paper_mode=True, config=FRICTION_CONFIG_DISABLED)


# ── Enabled / Disabled Logic ────────────────────────────────────────


class TestEnabledProperty:

    def test_paper_mode_enabled(self, friction_paper):
        """Paper mode with config enabled should report enabled=True."""
        assert friction_paper.enabled is True

    def test_live_mode_disabled(self, friction_live):
        """Live mode should always report enabled=False regardless of config."""
        assert friction_live.enabled is False

    def test_config_disabled_in_paper_mode(self, friction_disabled):
        """Paper mode with config.enabled=False should report enabled=False."""
        assert friction_disabled.enabled is False


# ── Asset Type Classification ───────────────────────────────────────


class TestAssetTypeClassification:

    def test_btc_is_crypto(self, friction_paper):
        """BTC should use crypto rates."""
        cost = friction_paper.spread_cost("BTC", 100000.0, 0.01, "long")
        # Half spread = 0.15% / 2 = 0.075%
        # Cost = 100000 * 0.01 * 0.00075 = 0.75
        assert cost == pytest.approx(0.75, abs=1e-3)

    def test_eth_is_crypto(self, friction_paper):
        """ETH should use crypto rates."""
        cost = friction_paper.spread_cost("ETH", 3000.0, 1.0, "long")
        # Half spread = 0.15% / 2 = 0.075%
        # Cost = 3000 * 1.0 * 0.00075 = 2.25
        assert cost == pytest.approx(2.25, abs=1e-3)

    def test_aapl_is_stock(self, friction_paper):
        """AAPL should use stock rates."""
        cost = friction_paper.spread_cost("AAPL", 200.0, 10, "long")
        # Half spread = 0.05% / 2 = 0.025%
        # Cost = 200 * 10 * 0.00025 = 0.50
        assert cost == pytest.approx(0.50, abs=1e-3)

    def test_nvda_is_stock(self, friction_paper):
        """NVDA should use stock rates."""
        cost = friction_paper.spread_cost("NVDA", 800.0, 5, "long")
        # Half spread = 0.05% / 2 = 0.025%
        # Cost = 800 * 5 * 0.00025 = 1.00
        assert cost == pytest.approx(1.00, abs=1e-3)

    def test_tsla_is_stock(self, friction_paper):
        """TSLA should use stock rates."""
        cost = friction_paper.commission("TSLA", 250.0, 4)
        # Stock commission = 0.0%
        assert cost == 0.0

    def test_spy_is_etf(self, friction_paper):
        """SPY should use ETF rates."""
        cost = friction_paper.spread_cost("SPY", 500.0, 10, "long")
        # Half spread = 0.03% / 2 = 0.015%
        # Cost = 500 * 10 * 0.00015 = 0.75
        assert cost == pytest.approx(0.75, abs=1e-3)

    def test_gldm_is_etf(self, friction_paper):
        """GLDM should use ETF rates."""
        cost = friction_paper.spread_cost("GLDM", 50.0, 20, "long")
        # Half spread = 0.03% / 2 = 0.015%
        # Cost = 50 * 20 * 0.00015 = 0.15
        assert cost == pytest.approx(0.15, abs=1e-3)

    def test_slv_is_etf(self, friction_paper):
        """SLV should use ETF rates."""
        cost = friction_paper.spread_cost("SLV", 25.0, 40, "long")
        # Half spread = 0.03% / 2 = 0.015%
        # Cost = 25 * 40 * 0.00015 = 0.15
        assert cost == pytest.approx(0.15, abs=1e-3)

    def test_tlt_is_etf(self, friction_paper):
        """TLT should use ETF rates."""
        cost = friction_paper.spread_cost("TLT", 90.0, 10, "long")
        expected = 90.0 * 10 * (0.03 / 100 / 2)
        assert cost == pytest.approx(expected, abs=1e-3)

    def test_xle_is_etf(self, friction_paper):
        """XLE should use ETF rates."""
        cost = friction_paper.spread_cost("XLE", 85.0, 10, "long")
        expected = 85.0 * 10 * (0.03 / 100 / 2)
        assert cost == pytest.approx(expected, abs=1e-3)

    def test_ews_is_etf(self, friction_paper):
        """EWS should use ETF rates."""
        cost = friction_paper.spread_cost("EWS", 22.0, 50, "long")
        expected = 22.0 * 50 * (0.03 / 100 / 2)
        assert cost == pytest.approx(expected, abs=1e-3)

    def test_fxi_is_etf(self, friction_paper):
        """FXI should use ETF rates."""
        cost = friction_paper.spread_cost("FXI", 30.0, 30, "long")
        expected = 30.0 * 30 * (0.03 / 100 / 2)
        assert cost == pytest.approx(expected, abs=1e-3)

    def test_meta_is_stock(self, friction_paper):
        """META should use stock rates."""
        cost = friction_paper.spread_cost("META", 500.0, 2, "long")
        expected = 500.0 * 2 * (0.05 / 100 / 2)
        assert cost == pytest.approx(expected, abs=1e-3)

    def test_amzn_is_stock(self, friction_paper):
        """AMZN should use stock rates."""
        cost = friction_paper.spread_cost("AMZN", 180.0, 5, "long")
        expected = 180.0 * 5 * (0.05 / 100 / 2)
        assert cost == pytest.approx(expected, abs=1e-3)


# ── Spread Cost Calculations ────────────────────────────────────────


class TestSpreadCost:

    def test_half_spread_applied_per_side(self, friction_paper):
        """Spread cost should use half the bid/ask spread per trade side."""
        # Crypto: full spread = 0.15%, half = 0.075%
        cost = friction_paper.spread_cost("BTC", 60000.0, 0.1, "long")
        expected = 60000.0 * 0.1 * (0.15 / 100 / 2)
        assert cost == pytest.approx(expected, abs=1e-3)

    def test_spread_same_for_long_and_short(self, friction_paper):
        """Spread cost should be the same regardless of direction."""
        cost_long = friction_paper.spread_cost("BTC", 60000.0, 0.1, "long")
        cost_short = friction_paper.spread_cost("BTC", 60000.0, 0.1, "short")
        assert cost_long == cost_short

    def test_spread_scales_with_quantity(self, friction_paper):
        """Doubling quantity should double spread cost."""
        cost_1 = friction_paper.spread_cost("AAPL", 200.0, 10, "long")
        cost_2 = friction_paper.spread_cost("AAPL", 200.0, 20, "long")
        assert cost_2 == pytest.approx(cost_1 * 2, abs=1e-3)

    def test_spread_scales_with_price(self, friction_paper):
        """Doubling price should double spread cost."""
        cost_1 = friction_paper.spread_cost("AAPL", 100.0, 10, "long")
        cost_2 = friction_paper.spread_cost("AAPL", 200.0, 10, "long")
        assert cost_2 == pytest.approx(cost_1 * 2, abs=1e-3)


# ── Commission Calculations ─────────────────────────────────────────


class TestCommission:

    def test_stock_commission_is_zero(self, friction_paper):
        """Stock commission should be 0.0% (zero)."""
        cost = friction_paper.commission("AAPL", 200.0, 10)
        assert cost == 0.0

    def test_etf_commission_is_zero(self, friction_paper):
        """ETF commission should be 0.0% (zero)."""
        cost = friction_paper.commission("SPY", 500.0, 10)
        assert cost == 0.0

    def test_crypto_commission_applied(self, friction_paper):
        """Crypto commission should be 0.15%."""
        cost = friction_paper.commission("BTC", 60000.0, 0.1)
        # 0.15% of notional = 60000 * 0.1 * 0.0015 = 9.0
        expected = 60000.0 * 0.1 * (0.15 / 100)
        assert cost == pytest.approx(expected, abs=1e-3)

    def test_eth_commission(self, friction_paper):
        """ETH commission should use crypto rate."""
        cost = friction_paper.commission("ETH", 3000.0, 1.0)
        expected = 3000.0 * 1.0 * (0.15 / 100)
        assert cost == pytest.approx(expected, abs=1e-3)

    def test_commission_scales_with_notional(self, friction_paper):
        """Commission should scale linearly with notional value."""
        cost_1 = friction_paper.commission("BTC", 60000.0, 0.1)
        cost_2 = friction_paper.commission("BTC", 60000.0, 0.2)
        assert cost_2 == pytest.approx(cost_1 * 2, abs=1e-3)


# ── Borrow Cost Calculations ────────────────────────────────────────


class TestBorrowCostDaily:

    def test_stock_borrow_daily(self, friction_paper):
        """Stock borrow cost should be 1.5% annualized / 365."""
        cost = friction_paper.borrow_cost_daily("AAPL", 200.0, 10)
        # Notional = 2000, annual = 2000 * 0.015 = 30, daily = 30/365
        expected = round(200.0 * 10 * (1.5 / 100) / 365, 4)
        assert cost == pytest.approx(expected, abs=1e-4)

    def test_etf_borrow_daily(self, friction_paper):
        """ETF borrow cost should be 1.0% annualized / 365."""
        cost = friction_paper.borrow_cost_daily("SPY", 500.0, 10)
        expected = 500.0 * 10 * (1.0 / 100) / 365
        assert cost == pytest.approx(expected, abs=1e-3)

    def test_crypto_borrow_daily(self, friction_paper):
        """Crypto borrow cost should be 5.0% annualized / 365."""
        cost = friction_paper.borrow_cost_daily("BTC", 60000.0, 0.1)
        expected = 60000.0 * 0.1 * (5.0 / 100) / 365
        assert cost == pytest.approx(expected, abs=1e-3)

    def test_borrow_scales_with_notional(self, friction_paper):
        """Borrow cost should scale with position notional value."""
        cost_1 = friction_paper.borrow_cost_daily("AAPL", 200.0, 10)
        cost_2 = friction_paper.borrow_cost_daily("AAPL", 200.0, 20)
        assert cost_2 == pytest.approx(cost_1 * 2, abs=1e-3)


# ── Accrued Borrow Over Multiple Days ───────────────────────────────


class TestAccruedBorrowCost:

    def test_accrued_one_day(self, friction_paper):
        """Accrued borrow for 1 day should equal daily borrow cost."""
        daily = friction_paper.borrow_cost_daily("AAPL", 200.0, 10)
        accrued = friction_paper.accrued_borrow_cost("AAPL", 200.0, 10, 1)
        assert accrued == pytest.approx(daily, abs=1e-3)

    def test_accrued_multiple_days(self, friction_paper):
        """Accrued borrow for N days should be N * daily."""
        daily = friction_paper.borrow_cost_daily("BTC", 60000.0, 0.1)
        accrued = friction_paper.accrued_borrow_cost("BTC", 60000.0, 0.1, 30)
        assert accrued == pytest.approx(daily * 30, abs=1e-3)

    def test_accrued_zero_days(self, friction_paper):
        """Accrued borrow for 0 days should be 0."""
        accrued = friction_paper.accrued_borrow_cost("BTC", 60000.0, 0.1, 0)
        assert accrued == 0.0

    def test_accrued_365_days_equals_annual(self, friction_paper):
        """Accrued borrow for 365 days should equal the annual rate on notional."""
        notional = 60000.0 * 0.1  # 6000
        accrued = friction_paper.accrued_borrow_cost("BTC", 60000.0, 0.1, 365)
        expected_annual = notional * (5.0 / 100)
        assert accrued == pytest.approx(expected_annual, abs=0.1)  # rounding over 365 days


# ── Total Entry / Exit Costs ────────────────────────────────────────


class TestTotalEntryCost:

    def test_total_entry_is_spread_plus_commission(self, friction_paper):
        """Total entry cost should be spread + commission."""
        spread = friction_paper.spread_cost("BTC", 60000.0, 0.1, "long")
        commission = friction_paper.commission("BTC", 60000.0, 0.1)
        total = friction_paper.total_entry_cost("BTC", 60000.0, 0.1, "long")
        assert total == pytest.approx(spread + commission, abs=1e-3)

    def test_total_entry_stock_equals_spread_only(self, friction_paper):
        """For stocks (zero commission), total entry = spread only."""
        spread = friction_paper.spread_cost("AAPL", 200.0, 10, "long")
        total = friction_paper.total_entry_cost("AAPL", 200.0, 10, "long")
        assert total == pytest.approx(spread, abs=1e-3)

    def test_total_entry_crypto_includes_commission(self, friction_paper):
        """For crypto, total entry = spread + 0.15% commission."""
        total = friction_paper.total_entry_cost("BTC", 60000.0, 0.1, "long")
        # Spread: 60000 * 0.1 * 0.00075 = 4.50
        # Commission: 60000 * 0.1 * 0.0015 = 9.00
        # Total: 13.50
        expected = 60000.0 * 0.1 * (0.15 / 100 / 2 + 0.15 / 100)
        assert total == pytest.approx(expected, abs=1e-3)


class TestTotalExitCost:

    def test_total_exit_is_spread_plus_commission(self, friction_paper):
        """Total exit cost should be spread + commission."""
        spread = friction_paper.spread_cost("ETH", 3000.0, 1.0, "long")
        commission = friction_paper.commission("ETH", 3000.0, 1.0)
        total = friction_paper.total_exit_cost("ETH", 3000.0, 1.0, "long")
        assert total == pytest.approx(spread + commission, abs=1e-3)

    def test_total_exit_etf(self, friction_paper):
        """ETF exit cost should be spread only (zero commission)."""
        spread = friction_paper.spread_cost("SPY", 500.0, 10, "short")
        total = friction_paper.total_exit_cost("SPY", 500.0, 10, "short")
        assert total == pytest.approx(spread, abs=1e-3)

    def test_entry_and_exit_costs_equal_for_same_params(self, friction_paper):
        """Entry and exit costs should be equal for the same parameters."""
        entry = friction_paper.total_entry_cost("BTC", 60000.0, 0.1, "long")
        exit_ = friction_paper.total_exit_cost("BTC", 60000.0, 0.1, "long")
        assert entry == pytest.approx(exit_, abs=1e-3)


# ── Live Mode Returns Zero ──────────────────────────────────────────


class TestLiveModeReturnsZero:

    def test_spread_zero_in_live(self, friction_live):
        """Spread cost should return 0.0 in live mode."""
        assert friction_live.spread_cost("BTC", 60000.0, 0.1, "long") == 0.0

    def test_commission_zero_in_live(self, friction_live):
        """Commission should return 0.0 in live mode."""
        assert friction_live.commission("BTC", 60000.0, 0.1) == 0.0

    def test_borrow_zero_in_live(self, friction_live):
        """Borrow cost daily should return 0.0 in live mode."""
        assert friction_live.borrow_cost_daily("BTC", 60000.0, 0.1) == 0.0

    def test_accrued_borrow_zero_in_live(self, friction_live):
        """Accrued borrow cost should return 0.0 in live mode."""
        assert friction_live.accrued_borrow_cost("BTC", 60000.0, 0.1, 30) == 0.0

    def test_total_entry_zero_in_live(self, friction_live):
        """Total entry cost should return 0.0 in live mode."""
        assert friction_live.total_entry_cost("BTC", 60000.0, 0.1, "long") == 0.0

    def test_total_exit_zero_in_live(self, friction_live):
        """Total exit cost should return 0.0 in live mode."""
        assert friction_live.total_exit_cost("BTC", 60000.0, 0.1, "long") == 0.0


# ── Config Disabled Returns Zero ────────────────────────────────────


class TestConfigDisabledReturnsZero:

    def test_spread_zero_when_disabled(self, friction_disabled):
        """Spread cost should return 0.0 when config disabled."""
        assert friction_disabled.spread_cost("BTC", 60000.0, 0.1, "long") == 0.0

    def test_commission_zero_when_disabled(self, friction_disabled):
        """Commission should return 0.0 when config disabled."""
        assert friction_disabled.commission("BTC", 60000.0, 0.1) == 0.0

    def test_borrow_zero_when_disabled(self, friction_disabled):
        """Borrow cost should return 0.0 when config disabled."""
        assert friction_disabled.borrow_cost_daily("BTC", 60000.0, 0.1) == 0.0

    def test_accrued_borrow_zero_when_disabled(self, friction_disabled):
        """Accrued borrow should return 0.0 when config disabled."""
        assert friction_disabled.accrued_borrow_cost("BTC", 60000.0, 0.1, 30) == 0.0

    def test_total_entry_zero_when_disabled(self, friction_disabled):
        """Total entry should return 0.0 when config disabled."""
        assert friction_disabled.total_entry_cost("BTC", 60000.0, 0.1, "long") == 0.0

    def test_total_exit_zero_when_disabled(self, friction_disabled):
        """Total exit should return 0.0 when config disabled."""
        assert friction_disabled.total_exit_cost("BTC", 60000.0, 0.1, "long") == 0.0


# ── Edge Cases ──────────────────────────────────────────────────────


class TestEdgeCases:

    def test_zero_price_spread(self, friction_paper):
        """Zero price should produce zero spread cost."""
        assert friction_paper.spread_cost("BTC", 0.0, 0.1, "long") == 0.0

    def test_zero_quantity_spread(self, friction_paper):
        """Zero quantity should produce zero spread cost."""
        assert friction_paper.spread_cost("BTC", 60000.0, 0, "long") == 0.0

    def test_zero_price_commission(self, friction_paper):
        """Zero price should produce zero commission."""
        assert friction_paper.commission("BTC", 0.0, 0.1) == 0.0

    def test_zero_quantity_commission(self, friction_paper):
        """Zero quantity should produce zero commission."""
        assert friction_paper.commission("BTC", 60000.0, 0) == 0.0

    def test_zero_price_borrow(self, friction_paper):
        """Zero price should produce zero borrow cost."""
        assert friction_paper.borrow_cost_daily("BTC", 0.0, 0.1) == 0.0

    def test_zero_quantity_borrow(self, friction_paper):
        """Zero quantity should produce zero borrow cost."""
        assert friction_paper.borrow_cost_daily("BTC", 60000.0, 0) == 0.0

    def test_zero_price_total_entry(self, friction_paper):
        """Zero price should produce zero total entry cost."""
        assert friction_paper.total_entry_cost("BTC", 0.0, 0.1, "long") == 0.0

    def test_zero_quantity_total_exit(self, friction_paper):
        """Zero quantity should produce zero total exit cost."""
        assert friction_paper.total_exit_cost("BTC", 60000.0, 0, "short") == 0.0

    def test_very_small_quantity(self, friction_paper):
        """Very small quantity should still compute correct friction."""
        cost = friction_paper.spread_cost("BTC", 60000.0, 0.00001, "long")
        expected = 60000.0 * 0.00001 * (0.15 / 100 / 2)
        assert cost == pytest.approx(expected, abs=1e-3)

    def test_very_large_quantity(self, friction_paper):
        """Large quantity should scale linearly."""
        cost = friction_paper.spread_cost("AAPL", 200.0, 100000, "long")
        expected = 200.0 * 100000 * (0.05 / 100 / 2)
        assert cost == pytest.approx(expected, abs=1e-3)


# ── Numerical Correctness Spot Checks ───────────────────────────────


class TestNumericalCorrectness:
    """Concrete numerical checks for each asset type to catch regressions."""

    def test_btc_full_roundtrip_cost(self, friction_paper):
        """BTC $60k, 0.1 qty: verify full roundtrip (entry + exit) cost."""
        notional = 60000.0 * 0.1  # $6000
        half_spread = notional * (0.15 / 100 / 2)  # $4.50 per side
        commission = notional * (0.15 / 100)  # $9.00 per side
        entry = friction_paper.total_entry_cost("BTC", 60000.0, 0.1, "long")
        exit_ = friction_paper.total_exit_cost("BTC", 60000.0, 0.1, "long")
        roundtrip = entry + exit_
        expected = 2 * (half_spread + commission)  # 2 * 13.50 = 27.00
        assert roundtrip == pytest.approx(expected, abs=1e-3)

    def test_aapl_full_roundtrip_cost(self, friction_paper):
        """AAPL $200, 10 shares: verify full roundtrip cost."""
        notional = 200.0 * 10  # $2000
        half_spread = notional * (0.05 / 100 / 2)  # $0.50 per side
        commission = 0.0  # stocks = 0%
        entry = friction_paper.total_entry_cost("AAPL", 200.0, 10, "long")
        exit_ = friction_paper.total_exit_cost("AAPL", 200.0, 10, "long")
        roundtrip = entry + exit_
        expected = 2 * (half_spread + commission)  # 2 * 0.50 = 1.00
        assert roundtrip == pytest.approx(expected, abs=1e-3)

    def test_spy_full_roundtrip_cost(self, friction_paper):
        """SPY $500, 10 shares: verify full roundtrip cost."""
        notional = 500.0 * 10  # $5000
        half_spread = notional * (0.03 / 100 / 2)  # $0.75 per side
        entry = friction_paper.total_entry_cost("SPY", 500.0, 10, "long")
        exit_ = friction_paper.total_exit_cost("SPY", 500.0, 10, "long")
        roundtrip = entry + exit_
        expected = 2 * half_spread  # 2 * 0.75 = 1.50
        assert roundtrip == pytest.approx(expected, abs=1e-3)

    def test_btc_30day_borrow(self, friction_paper):
        """BTC $60k, 0.1 qty, 30-day short borrow cost."""
        notional = 60000.0 * 0.1  # $6000
        annual_rate = 5.0 / 100  # 5%
        expected = notional * annual_rate / 365 * 30
        accrued = friction_paper.accrued_borrow_cost("BTC", 60000.0, 0.1, 30)
        assert accrued == pytest.approx(expected, abs=1e-3)
