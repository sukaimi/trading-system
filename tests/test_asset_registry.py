"""Tests for core/asset_registry.py"""

import json
import os
from unittest.mock import patch

import pytest

from core.asset_registry import AssetRegistry, get_registry, get_tradeable_assets, validate_asset


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton between tests."""
    AssetRegistry._instance = None
    yield
    AssetRegistry._instance = None


class TestAssetRegistry:
    def test_loads_from_config(self):
        registry = get_registry()
        assets = registry.tradeable_symbols
        assert len(assets) >= 4  # At least BTC, ETH, GLDM, SLV
        assert "BTC" in assets
        assert "ETH" in assets

    def test_stocks_loaded(self):
        registry = get_registry()
        for stock in ("AAPL", "NVDA", "TSLA", "AMZN", "SPY", "META"):
            assert stock in registry.tradeable_symbols

    def test_macro_is_valid_not_tradeable(self):
        registry = get_registry()
        assert registry.is_valid("MACRO")
        assert not registry.is_tradeable("MACRO")

    def test_invalid_symbol(self):
        registry = get_registry()
        assert not registry.is_valid("FAKECOIN")

    def test_validate_valid_asset(self):
        assert validate_asset("BTC") == "BTC"
        assert validate_asset("MACRO") == "MACRO"
        assert validate_asset("AAPL") == "AAPL"

    def test_validate_invalid_asset(self):
        with pytest.raises(ValueError, match="Unknown asset"):
            validate_asset("INVALID")

    def test_get_config(self):
        config = get_registry().get_config("BTC")
        assert config.get("type") == "crypto"

    def test_get_config_stock(self):
        config = get_registry().get_config("AAPL")
        assert config.get("type") == "stock"

    def test_get_tradeable_assets(self):
        assets = get_tradeable_assets()
        assert isinstance(assets, list)
        assert "BTC" in assets
        assert "MACRO" not in assets

    def test_reload(self):
        registry = get_registry()
        original = registry.tradeable_symbols[:]
        registry.reload()
        assert registry.tradeable_symbols == original

    def test_all_valid_symbols_includes_macro(self):
        registry = get_registry()
        all_valid = registry.all_valid_symbols
        assert "MACRO" in all_valid
        assert "BTC" in all_valid

    def test_missing_config_file(self, tmp_path):
        """Registry gracefully handles missing config file."""
        AssetRegistry._instance = None
        with patch("core.asset_registry._ASSETS_FILE", str(tmp_path / "nonexistent.json")):
            registry = AssetRegistry()
            assert registry.tradeable_symbols == []
