"""Tests for correlation-based position limits in RiskManager."""

import time
from unittest.mock import MagicMock, patch

import pytest

from core.risk_manager import RiskManager


def _make_config(**overrides):
    """Build a risk config dict with defaults."""
    cfg = {
        "max_position_pct": 7.0,
        "max_daily_loss_pct": 5.0,
        "max_total_drawdown_pct": 15.0,
        "max_open_positions": 10,
        "max_correlation": 0.8,
        "stop_loss_atr_mult": 2.0,
        "base_risk_per_trade_pct": 2.0,
        "max_portfolio_avg_correlation": 0.6,
    }
    cfg.update(overrides)
    return cfg


def _make_portfolio(positions):
    """Build a portfolio state dict."""
    return {
        "equity": 100.0,
        "daily_pnl_pct": 0.0,
        "drawdown_from_peak_pct": 0.0,
        "open_positions": positions,
    }


def _make_order(asset="ETH"):
    """Build a valid execution order."""
    return {
        "asset": asset,
        "direction": "long",
        "position_size_pct": 5.0,
        "stop_loss": 3000.0,
    }


class TestCorrelationLimits:
    """Test Check 8: correlation-based position limits."""

    @patch("tools.market_data.MarketDataFetcher")
    @patch("tools.correlation.CorrelationAnalyzer")
    def test_order_rejected_high_correlation(self, mock_analyzer_cls, mock_mdf_cls):
        """Order should be rejected when correlation exceeds max_correlation."""
        # Mock market data: return enough price data
        mock_mdf = MagicMock()
        mock_mdf.get_ohlcv.return_value = [{"close": float(i)} for i in range(30)]
        mock_mdf_cls.return_value = mock_mdf

        # Mock analyzer: return high correlation
        mock_analyzer = MagicMock()
        mock_analyzer.pairwise_correlation.return_value = 0.95
        mock_analyzer_cls.return_value = mock_analyzer

        rm = RiskManager(_make_config(max_correlation=0.8))
        portfolio = _make_portfolio([{"asset": "BTC"}])

        approved, reason, _ = rm.validate_order(_make_order("ETH"), portfolio)
        assert approved is False
        assert "Correlation limit" in reason
        assert "0.95" in reason

    @patch("tools.market_data.MarketDataFetcher")
    @patch("tools.correlation.CorrelationAnalyzer")
    def test_order_approved_low_correlation(self, mock_analyzer_cls, mock_mdf_cls):
        """Order should be approved when correlation is below threshold."""
        mock_mdf = MagicMock()
        mock_mdf.get_ohlcv.return_value = [{"close": float(i)} for i in range(30)]
        mock_mdf_cls.return_value = mock_mdf

        mock_analyzer = MagicMock()
        mock_analyzer.pairwise_correlation.return_value = 0.3
        mock_analyzer_cls.return_value = mock_analyzer

        rm = RiskManager(_make_config(max_correlation=0.8))
        portfolio = _make_portfolio([{"asset": "BTC"}])

        approved, reason, _ = rm.validate_order(_make_order("GLDM"), portfolio)
        assert approved is True
        assert reason == "APPROVED"

    @patch("tools.market_data.MarketDataFetcher")
    @patch("tools.correlation.CorrelationAnalyzer")
    def test_correlation_cache_works(self, mock_analyzer_cls, mock_mdf_cls):
        """Second call should use cached data, not re-fetch prices."""
        mock_mdf = MagicMock()
        mock_mdf.get_ohlcv.return_value = [{"close": float(i)} for i in range(30)]
        mock_mdf_cls.return_value = mock_mdf

        mock_analyzer = MagicMock()
        mock_analyzer.pairwise_correlation.return_value = 0.3
        mock_analyzer_cls.return_value = mock_analyzer

        rm = RiskManager(_make_config(max_correlation=0.8))
        portfolio = _make_portfolio([{"asset": "BTC"}])
        order = _make_order("GLDM")

        # First call — fetches prices
        rm.validate_order(order, portfolio)
        first_call_count = mock_mdf.get_ohlcv.call_count

        # Second call — should use cache
        rm.validate_order(order, portfolio)
        second_call_count = mock_mdf.get_ohlcv.call_count

        # No additional fetch calls on second validate
        assert second_call_count == first_call_count

    def test_single_position_no_correlation_check(self):
        """With no open positions, correlation check should pass trivially."""
        rm = RiskManager(_make_config())
        portfolio = _make_portfolio([])

        approved, reason, _ = rm.validate_order(_make_order("ETH"), portfolio)
        assert approved is True

    @patch("tools.market_data.MarketDataFetcher")
    @patch("tools.correlation.CorrelationAnalyzer")
    def test_insufficient_price_data_approves(self, mock_analyzer_cls, mock_mdf_cls):
        """When price data is insufficient (<5 points), approve by default."""
        mock_mdf = MagicMock()
        # Return only 3 data points for candidate asset
        mock_mdf.get_ohlcv.return_value = [{"close": 1.0}, {"close": 2.0}, {"close": 3.0}]
        mock_mdf_cls.return_value = mock_mdf

        rm = RiskManager(_make_config(max_correlation=0.5))
        portfolio = _make_portfolio([{"asset": "BTC"}])

        approved, reason, _ = rm.validate_order(_make_order("GLDM"), portfolio)
        assert approved is True

    @patch("tools.market_data.MarketDataFetcher")
    @patch("tools.correlation.CorrelationAnalyzer")
    def test_price_fetch_failure_approves(self, mock_analyzer_cls, mock_mdf_cls):
        """When price fetch throws exception, approve by default."""
        mock_mdf = MagicMock()
        mock_mdf.get_ohlcv.side_effect = Exception("API error")
        mock_mdf_cls.return_value = mock_mdf

        rm = RiskManager(_make_config(max_correlation=0.5))
        portfolio = _make_portfolio([{"asset": "BTC"}])

        # Should not crash — approve by default
        approved, reason, _ = rm.validate_order(_make_order("GLDM"), portfolio)
        assert approved is True

    @patch("tools.market_data.MarketDataFetcher")
    @patch("tools.correlation.CorrelationAnalyzer")
    def test_negative_correlation_allowed_through(self, mock_analyzer_cls, mock_mdf_cls):
        """Negative correlation = diversification, should be approved."""
        mock_mdf = MagicMock()
        mock_mdf.get_ohlcv.return_value = [{"close": float(i)} for i in range(30)]
        mock_mdf_cls.return_value = mock_mdf

        mock_analyzer = MagicMock()
        mock_analyzer.pairwise_correlation.return_value = -0.9  # Strong negative = diversification
        mock_analyzer_cls.return_value = mock_analyzer

        rm = RiskManager(_make_config(max_correlation=0.8))
        portfolio = _make_portfolio([{"asset": "BTC"}])

        approved, reason, _ = rm.validate_order(_make_order("GLDM"), portfolio)
        assert approved is True

    @patch("tools.market_data.MarketDataFetcher")
    @patch("tools.correlation.CorrelationAnalyzer")
    def test_cache_expires_after_ttl(self, mock_analyzer_cls, mock_mdf_cls):
        """Cache should expire after TTL and re-fetch prices."""
        mock_mdf = MagicMock()
        mock_mdf.get_ohlcv.return_value = [{"close": float(i)} for i in range(30)]
        mock_mdf_cls.return_value = mock_mdf

        mock_analyzer = MagicMock()
        mock_analyzer.pairwise_correlation.return_value = 0.3
        mock_analyzer_cls.return_value = mock_analyzer

        rm = RiskManager(_make_config(max_correlation=0.8))
        rm._correlation_ttl = 0  # Expire immediately
        portfolio = _make_portfolio([{"asset": "BTC"}])
        order = _make_order("GLDM")

        rm.validate_order(order, portfolio)
        first_count = mock_mdf.get_ohlcv.call_count

        rm.validate_order(order, portfolio)
        second_count = mock_mdf.get_ohlcv.call_count

        # Should have fetched again since cache expired
        assert second_count > first_count
