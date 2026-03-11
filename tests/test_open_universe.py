"""Tests for open universe trading — dynamic asset resolution and quality gates."""

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest


class TestAssetRegistryDynamic:
    """Test dynamic asset resolution in AssetRegistry."""

    def setup_method(self):
        """Reset singleton for each test."""
        from core.asset_registry import AssetRegistry
        AssetRegistry._instance = None

    def teardown_method(self):
        from core.asset_registry import AssetRegistry
        AssetRegistry._instance = None

    def test_core_assets_still_valid(self):
        from core.asset_registry import get_registry
        reg = get_registry()
        for sym in ["BTC", "ETH", "AAPL", "SPY", "GLDM"]:
            assert reg.is_valid(sym), f"{sym} should be valid"
            assert reg.is_tradeable(sym), f"{sym} should be tradeable"
            assert reg.is_core(sym), f"{sym} should be core"
            assert not reg.is_dynamic(sym), f"{sym} should not be dynamic"

    def test_macro_still_valid_not_tradeable(self):
        from core.asset_registry import get_registry
        reg = get_registry()
        assert reg.is_valid("MACRO")
        assert not reg.is_tradeable("MACRO")
        assert not reg.is_core("MACRO")

    def test_unknown_ticker_invalid_without_resolution(self):
        from core.asset_registry import get_registry
        reg = get_registry()
        # Without yfinance resolution, unknown ticker is invalid
        assert not reg.is_valid("ZZZZ")

    def test_invalid_ticker_format_rejected(self):
        from core.asset_registry import get_registry
        reg = get_registry()
        assert not reg.resolve_dynamic("123")
        assert not reg.resolve_dynamic("toolong")
        assert not reg.resolve_dynamic("")
        assert not reg.resolve_dynamic("a")  # lowercase

    @patch("yfinance.Ticker")
    def test_resolve_dynamic_passes_quality_gates(self, mock_ticker_cls):
        """Ticker with >$1B cap, >500K vol passes."""
        from core.asset_registry import get_registry
        mock_info = {
            "regularMarketPrice": 150.0,
            "marketCap": 2_000_000_000,
            "averageDailyVolume10Day": 1_000_000,
            "exchange": "NMS",
            "longName": "Test Corp",
            "sector": "Technology",
            "currency": "USD",
            "quoteType": "EQUITY",
        }
        mock_ticker_cls.return_value.info = mock_info

        reg = get_registry()
        assert reg.resolve_dynamic("GOOG")
        assert reg.is_valid("GOOG")
        assert reg.is_tradeable("GOOG")
        assert reg.is_dynamic("GOOG")
        assert not reg.is_core("GOOG")

    @patch("yfinance.Ticker")
    def test_reject_low_market_cap(self, mock_ticker_cls):
        from core.asset_registry import get_registry
        mock_ticker_cls.return_value.info = {
            "regularMarketPrice": 5.0,
            "marketCap": 500_000_000,  # < $1B
            "averageDailyVolume10Day": 1_000_000,
            "exchange": "NMS",
        }
        reg = get_registry()
        assert not reg.resolve_dynamic("SMOL")

    @patch("yfinance.Ticker")
    def test_reject_low_volume(self, mock_ticker_cls):
        from core.asset_registry import get_registry
        mock_ticker_cls.return_value.info = {
            "regularMarketPrice": 100.0,
            "marketCap": 5_000_000_000,
            "averageDailyVolume10Day": 100_000,  # < 500K
            "exchange": "NMS",
        }
        reg = get_registry()
        assert not reg.resolve_dynamic("THIN")

    @patch("yfinance.Ticker")
    def test_reject_otc_exchange(self, mock_ticker_cls):
        from core.asset_registry import get_registry
        mock_ticker_cls.return_value.info = {
            "regularMarketPrice": 10.0,
            "marketCap": 2_000_000_000,
            "averageDailyVolume10Day": 1_000_000,
            "exchange": "OTC",
        }
        reg = get_registry()
        assert not reg.resolve_dynamic("OTCX")

    @patch("yfinance.Ticker")
    def test_reject_spac(self, mock_ticker_cls):
        from core.asset_registry import get_registry
        mock_ticker_cls.return_value.info = {
            "regularMarketPrice": 10.0,
            "marketCap": 2_000_000_000,
            "averageDailyVolume10Day": 1_000_000,
            "exchange": "NMS",
            "longName": "SPECIAL PURPOSE ACQUISITION BLANK CHECK COMPANY",
        }
        reg = get_registry()
        assert not reg.resolve_dynamic("SPAC")

    @patch("yfinance.Ticker")
    def test_rejection_cache_prevents_repeated_calls(self, mock_ticker_cls):
        from core.asset_registry import get_registry
        mock_ticker_cls.return_value.info = {
            "regularMarketPrice": 5.0,
            "marketCap": 100,  # too small
        }
        reg = get_registry()
        assert not reg.resolve_dynamic("FAIL")
        # Second call should hit rejection cache, not yfinance
        mock_ticker_cls.reset_mock()
        assert not reg.resolve_dynamic("FAIL")
        mock_ticker_cls.assert_not_called()

    def test_core_symbols_property(self):
        from core.asset_registry import get_registry
        reg = get_registry()
        core = reg.core_symbols
        assert "BTC" in core
        assert "AAPL" in core
        assert len(core) == 14  # original 14 assets

    def test_get_core_assets_function(self):
        from core.asset_registry import get_core_assets
        core = get_core_assets()
        assert len(core) == 14
        assert "BTC" in core

    @patch("yfinance.Ticker")
    def test_validate_triggers_dynamic_resolution(self, mock_ticker_cls):
        """validate() should attempt dynamic resolution for unknown tickers."""
        from core.asset_registry import get_registry
        mock_ticker_cls.return_value.info = {
            "regularMarketPrice": 300.0,
            "marketCap": 10_000_000_000,
            "averageDailyVolume10Day": 5_000_000,
            "exchange": "NMS",
            "sector": "Healthcare",
            "currency": "USD",
            "quoteType": "EQUITY",
            "longName": "Big Pharma Inc",
        }
        reg = get_registry()
        result = reg.validate("JNJ")
        assert result == "JNJ"
        assert reg.is_dynamic("JNJ")

    def test_validate_rejects_bad_ticker(self):
        from core.asset_registry import get_registry
        reg = get_registry()
        with pytest.raises(ValueError, match="Unknown asset"):
            reg.validate("ZZZZZ")  # 5 chars, regex valid, but no yfinance data

    @patch("yfinance.Ticker")
    def test_get_sector_for_dynamic_asset(self, mock_ticker_cls):
        from core.asset_registry import get_registry
        mock_ticker_cls.return_value.info = {
            "regularMarketPrice": 200.0,
            "marketCap": 5_000_000_000,
            "averageDailyVolume10Day": 2_000_000,
            "exchange": "NMS",
            "sector": "Financial Services",
            "currency": "USD",
            "quoteType": "EQUITY",
            "longName": "Big Bank Inc",
        }
        reg = get_registry()
        reg.resolve_dynamic("JPM")
        assert reg.get_sector("JPM") == "Financial Services"


class TestRiskManagerDynamicSectors:
    """Test that RiskManager works with dynamic sector lookup."""

    def test_core_asset_sector_lookup(self):
        from core.risk_manager import RiskManager
        rm = RiskManager()
        assert rm._get_sector("AAPL") == "tech"
        assert rm._get_sector("BTC") == "crypto"
        assert rm._get_sector("GLDM") == "commodities"
        assert rm._get_sector("FXI") == "asia"

    def test_unknown_asset_returns_unknown(self):
        from core.risk_manager import RiskManager
        rm = RiskManager()
        assert rm._get_sector("ZZZZ") == "unknown"

    def test_sector_concentration_uses_dynamic_lookup(self):
        """Sector check should work for both core and dynamic assets."""
        from core.risk_manager import RiskManager
        rm = RiskManager()
        # Create portfolio with 4 tech positions
        portfolio = {
            "daily_pnl_pct": 0,
            "drawdown_from_peak_pct": 0,
            "open_positions": [
                {"asset": "AAPL", "direction": "long"},
                {"asset": "NVDA", "direction": "long"},
                {"asset": "TSLA", "direction": "long"},
                {"asset": "AMZN", "direction": "long"},
            ],
        }
        order = {
            "asset": "META",
            "position_size_pct": 5,
            "direction": "long",
            "stop_loss": 100,
        }
        ok, reason, _ = rm.validate_order(order, portfolio)
        assert not ok
        assert "Sector concentration" in reason


class TestSchemaValidation:
    """Test that schema validation works with open universe tickers."""

    def setup_method(self):
        from core.asset_registry import AssetRegistry
        AssetRegistry._instance = None

    def teardown_method(self):
        from core.asset_registry import AssetRegistry
        AssetRegistry._instance = None

    def test_core_asset_validates(self):
        from core.schemas import SignalAlert, Sentiment, SignalCategory, Urgency
        alert = SignalAlert(
            asset="BTC",
            signal_strength=0.8,
            headline="Test",
            sentiment=Sentiment.BULLISH,
            category=SignalCategory.CRYPTO_SPECIFIC,
            new_information="test",
            urgency=Urgency.HIGH,
            confidence_in_classification=0.9,
        )
        assert alert.asset == "BTC"

    def test_macro_validates(self):
        from core.schemas import SignalAlert, Sentiment, SignalCategory, Urgency
        alert = SignalAlert(
            asset="MACRO",
            signal_strength=0.5,
            headline="Fed meeting",
            sentiment=Sentiment.NEUTRAL,
            category=SignalCategory.MACRO,
            new_information="test",
            urgency=Urgency.MEDIUM,
            confidence_in_classification=0.5,
        )
        assert alert.asset == "MACRO"

    def test_invalid_format_rejected(self):
        from core.schemas import SignalAlert, Sentiment, SignalCategory, Urgency
        with pytest.raises(Exception):
            SignalAlert(
                asset="123invalid",
                signal_strength=0.5,
                headline="Test",
                sentiment=Sentiment.NEUTRAL,
                category=SignalCategory.EQUITY,
                new_information="test",
                urgency=Urgency.LOW,
                confidence_in_classification=0.5,
            )

    @patch("yfinance.Ticker")
    def test_dynamic_ticker_validates_via_resolution(self, mock_ticker_cls):
        """A new ticker like GOOG should validate if it passes quality gates."""
        mock_ticker_cls.return_value.info = {
            "regularMarketPrice": 150.0,
            "marketCap": 2_000_000_000_000,
            "averageDailyVolume10Day": 20_000_000,
            "exchange": "NMS",
            "sector": "Technology",
            "currency": "USD",
            "quoteType": "EQUITY",
            "longName": "Alphabet Inc",
        }
        from core.schemas import SignalAlert, Sentiment, SignalCategory, Urgency
        alert = SignalAlert(
            asset="GOOG",
            signal_strength=0.7,
            headline="Alphabet earnings beat",
            sentiment=Sentiment.BULLISH,
            category=SignalCategory.EQUITY,
            new_information="test",
            urgency=Urgency.HIGH,
            confidence_in_classification=0.8,
        )
        assert alert.asset == "GOOG"


class TestNewsScoutOpenUniverse:
    """Test that NewsScout prompt no longer constrains to fixed assets."""

    def test_prompt_uses_core_assets_not_valid_assets(self):
        """The classify prompt should reference core_assets, not valid_assets."""
        from agents.news_scout import CLASSIFY_PROMPT
        assert "{core_assets}" in CLASSIFY_PROMPT
        assert "{valid_assets}" not in CLASSIFY_PROMPT

    def test_prompt_allows_any_ticker(self):
        from agents.news_scout import CLASSIFY_PROMPT
        assert "ANY valid US-listed ticker" in CLASSIFY_PROMPT
        assert "NOT limited to our core assets" in CLASSIFY_PROMPT
