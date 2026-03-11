"""Dynamic asset registry — loads core assets from config/assets.json + dynamic resolution.

Provides validation for Pydantic models and asset metadata lookup.
Singleton pattern: loaded once at import, reloadable for SelfOptimizer.

Open Universe: any valid US-listed ticker can be traded. Core assets come from
config/assets.json. Discovered assets are resolved on-demand via yfinance with
quality gates (market cap, volume, exchange, listing age). Resolved assets are
cached to data/dynamic_assets.json for fast restarts.

Tier 0: pure Python, no LLM.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.asset_registry")

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_ASSETS_FILE = os.path.join(_CONFIG_DIR, "assets.json")
_DYNAMIC_CACHE_FILE = os.path.join(_DATA_DIR, "dynamic_assets.json")

# Special non-tradeable signal categories (never in assets.json)
SPECIAL_CATEGORIES = frozenset({"MACRO"})

# Quality gate thresholds for dynamic asset resolution
_MIN_MARKET_CAP = 1_000_000_000  # $1B
_MIN_AVG_VOLUME = 500_000  # 500K shares/day
_MIN_LISTING_DAYS = 90
_BLOCKED_EXCHANGES = {"OTC", "PINK", "GREY"}
_CACHE_TTL_SECONDS = 86400  # 24 hours

# Valid US ticker regex (1-5 uppercase letters, optional dot class like BRK.B)
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")


class AssetRegistry:
    """Registry of tradeable assets — core (config) + dynamic (discovered)."""

    _instance: AssetRegistry | None = None

    def __new__(cls) -> AssetRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._assets = {}
            cls._instance._dynamic_assets: dict[str, dict[str, Any]] = {}
            cls._instance._rejected_cache: dict[str, float] = {}  # symbol -> timestamp
            cls._instance._load()
            cls._instance._load_dynamic_cache()
        return cls._instance

    def _load(self) -> None:
        try:
            with open(_ASSETS_FILE) as f:
                self._assets: dict[str, dict[str, Any]] = json.load(f)
        except Exception:
            self._assets = {}

    def _load_dynamic_cache(self) -> None:
        """Load previously resolved dynamic assets from disk cache."""
        try:
            if os.path.exists(_DYNAMIC_CACHE_FILE):
                with open(_DYNAMIC_CACHE_FILE) as f:
                    cached = json.load(f)
                now = time.time()
                # Keep entries that haven't expired
                self._dynamic_assets = {
                    sym: data for sym, data in cached.items()
                    if now - data.get("resolved_at", 0) < _CACHE_TTL_SECONDS * 7  # 7-day cache on disk
                }
        except Exception:
            self._dynamic_assets = {}

    def _save_dynamic_cache(self) -> None:
        """Persist dynamic assets to disk."""
        try:
            os.makedirs(os.path.dirname(_DYNAMIC_CACHE_FILE), exist_ok=True)
            with open(_DYNAMIC_CACHE_FILE, "w") as f:
                json.dump(self._dynamic_assets, f, indent=2)
        except Exception as e:
            log.warning("Failed to save dynamic asset cache: %s", e)

    def reload(self) -> None:
        """Reload from disk (called after SelfOptimizer edits assets.json)."""
        self._load()

    @property
    def core_symbols(self) -> list[str]:
        """Core tradeable asset symbols (from assets.json)."""
        return list(self._assets.keys())

    @property
    def tradeable_symbols(self) -> list[str]:
        """All tradeable asset symbols (core + dynamic)."""
        return list(set(list(self._assets.keys()) + list(self._dynamic_assets.keys())))

    @property
    def all_valid_symbols(self) -> list[str]:
        """All valid symbols (tradeable + special categories like MACRO)."""
        return self.tradeable_symbols + list(SPECIAL_CATEGORIES)

    @property
    def dynamic_symbols(self) -> list[str]:
        """Dynamically discovered asset symbols."""
        return list(self._dynamic_assets.keys())

    def is_valid(self, symbol: str) -> bool:
        """Check if a symbol is valid (tradeable or special category)."""
        return symbol in self._assets or symbol in self._dynamic_assets or symbol in SPECIAL_CATEGORIES

    def is_tradeable(self, symbol: str) -> bool:
        """Check if a symbol is a tradeable asset (not MACRO)."""
        return symbol in self._assets or symbol in self._dynamic_assets

    def is_core(self, symbol: str) -> bool:
        """Check if a symbol is a core asset (from assets.json)."""
        return symbol in self._assets

    def is_dynamic(self, symbol: str) -> bool:
        """Check if a symbol was dynamically discovered."""
        return symbol in self._dynamic_assets

    def get_config(self, symbol: str) -> dict[str, Any]:
        """Get asset configuration metadata."""
        return self._assets.get(symbol, self._dynamic_assets.get(symbol, {}))

    def get_sector(self, symbol: str) -> str:
        """Get sector for an asset. Uses cached config or falls back to yfinance."""
        config = self.get_config(symbol)
        if config.get("sector"):
            return config["sector"]
        # Core asset type-based fallback
        asset_type = config.get("type", "")
        if asset_type == "crypto":
            return "crypto"
        if asset_type == "etf":
            return "etf"
        return "unknown"

    def resolve_dynamic(self, symbol: str) -> bool:
        """Attempt to resolve and validate a dynamic ticker via yfinance.

        Returns True if the symbol passes quality gates and is now tradeable.
        Returns False if the symbol fails validation.
        """
        # Already known
        if symbol in self._assets or symbol in self._dynamic_assets:
            return True
        if symbol in SPECIAL_CATEGORIES:
            return False  # MACRO is not tradeable

        # Basic format check
        if not _TICKER_RE.match(symbol):
            return False

        # Check rejection cache (don't re-resolve recently rejected tickers)
        now = time.time()
        rejected_at = self._rejected_cache.get(symbol, 0)
        if now - rejected_at < _CACHE_TTL_SECONDS:
            return False

        # Resolve via yfinance
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            # Quality gate 1: Must have valid market data
            if not info or info.get("regularMarketPrice") is None:
                log.info("Dynamic resolve REJECT %s: no market data", symbol)
                self._rejected_cache[symbol] = now
                return False

            # Quality gate 2: Market cap
            market_cap = info.get("marketCap", 0) or 0
            if market_cap < _MIN_MARKET_CAP:
                log.info("Dynamic resolve REJECT %s: market cap $%s < $1B", symbol, f"{market_cap:,.0f}")
                self._rejected_cache[symbol] = now
                return False

            # Quality gate 3: Average volume
            avg_vol = info.get("averageDailyVolume10Day", 0) or info.get("averageVolume", 0) or 0
            if avg_vol < _MIN_AVG_VOLUME:
                log.info("Dynamic resolve REJECT %s: avg volume %s < 500K", symbol, f"{avg_vol:,.0f}")
                self._rejected_cache[symbol] = now
                return False

            # Quality gate 4: Exchange check (no OTC/pink sheets)
            exchange = info.get("exchange", "")
            if exchange.upper() in _BLOCKED_EXCHANGES:
                log.info("Dynamic resolve REJECT %s: blocked exchange %s", symbol, exchange)
                self._rejected_cache[symbol] = now
                return False

            # Quality gate 5: Not a SPAC (check for common SPAC indicators)
            long_name = (info.get("longName") or "").upper()
            if "ACQUISITION" in long_name and "BLANK CHECK" in long_name:
                log.info("Dynamic resolve REJECT %s: SPAC detected", symbol)
                self._rejected_cache[symbol] = now
                return False

            # Passed all gates — register as dynamic asset
            sector = info.get("sector", "unknown")
            asset_type = "stock"
            quote_type = (info.get("quoteType") or "").lower()
            if quote_type == "etf":
                asset_type = "etf"
            elif quote_type == "cryptocurrency":
                asset_type = "crypto"

            self._dynamic_assets[symbol] = {
                "exchange": exchange,
                "type": asset_type,
                "currency": info.get("currency", "USD"),
                "contract_type": "Stock" if asset_type != "crypto" else "Crypto",
                "trading_hours": "09:30-16:00 ET",
                "min_order_size": 1,
                "sector": sector,
                "market_cap": market_cap,
                "avg_volume": avg_vol,
                "long_name": info.get("longName", ""),
                "resolved_at": now,
                "is_dynamic": True,
            }
            self._save_dynamic_cache()
            log.info(
                "Dynamic resolve OK: %s (%s, %s, cap=$%s, vol=%s)",
                symbol, sector, exchange, f"{market_cap:,.0f}", f"{avg_vol:,.0f}",
            )
            return True

        except ImportError:
            log.warning("yfinance not installed — cannot resolve dynamic ticker %s", symbol)
            return False
        except Exception as e:
            log.warning("Dynamic resolve failed for %s: %s", symbol, e)
            self._rejected_cache[symbol] = now
            return False

    def validate(self, value: str) -> str:
        """Validate an asset symbol.

        For core assets and MACRO, validates immediately.
        For unknown symbols, attempts dynamic resolution via yfinance.
        Raises ValueError only if dynamic resolution also fails.
        """
        # Fast path: already known
        if self.is_valid(value):
            return value

        # Try dynamic resolution
        if self.resolve_dynamic(value):
            return value

        raise ValueError(
            f"Unknown asset '{value}'. Core: {self.core_symbols}. "
            f"Dynamic resolution failed (quality gates: >$1B cap, >500K vol, no OTC)."
        )


# Module-level convenience functions

def get_registry() -> AssetRegistry:
    """Get the singleton AssetRegistry instance."""
    return AssetRegistry()


def get_tradeable_assets() -> list[str]:
    """Get list of all tradeable asset symbols (core + dynamic)."""
    return AssetRegistry().tradeable_symbols


def get_core_assets() -> list[str]:
    """Get list of core asset symbols only (from assets.json)."""
    return AssetRegistry().core_symbols


def validate_asset(value: str) -> str:
    """Validate an asset symbol. Raises ValueError if unknown and fails quality gates."""
    return AssetRegistry().validate(value)
