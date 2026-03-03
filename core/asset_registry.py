"""Dynamic asset registry — loads tradeable assets from config/assets.json.

Provides validation for Pydantic models and asset metadata lookup.
Singleton pattern: loaded once at import, reloadable for SelfOptimizer.
Tier 0: pure Python, no LLM.
"""

from __future__ import annotations

import json
import os
from typing import Any

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
_ASSETS_FILE = os.path.join(_CONFIG_DIR, "assets.json")

# Special non-tradeable signal categories (never in assets.json)
SPECIAL_CATEGORIES = frozenset({"MACRO"})


class AssetRegistry:
    """Singleton registry of tradeable assets loaded from config."""

    _instance: AssetRegistry | None = None

    def __new__(cls) -> AssetRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._assets = {}
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        try:
            with open(_ASSETS_FILE) as f:
                self._assets: dict[str, dict[str, Any]] = json.load(f)
        except Exception:
            self._assets = {}

    def reload(self) -> None:
        """Reload from disk (called after SelfOptimizer edits assets.json)."""
        self._load()

    @property
    def tradeable_symbols(self) -> list[str]:
        """All tradeable asset symbols."""
        return list(self._assets.keys())

    @property
    def all_valid_symbols(self) -> list[str]:
        """All valid symbols (tradeable + special categories like MACRO)."""
        return self.tradeable_symbols + list(SPECIAL_CATEGORIES)

    def is_valid(self, symbol: str) -> bool:
        """Check if a symbol is valid (tradeable or special category)."""
        return symbol in self._assets or symbol in SPECIAL_CATEGORIES

    def is_tradeable(self, symbol: str) -> bool:
        """Check if a symbol is a tradeable asset (not MACRO)."""
        return symbol in self._assets

    def get_config(self, symbol: str) -> dict[str, Any]:
        """Get asset configuration metadata."""
        return self._assets.get(symbol, {})

    def validate(self, value: str) -> str:
        """Validate an asset symbol. Raises ValueError if unknown."""
        if not self.is_valid(value):
            raise ValueError(
                f"Unknown asset '{value}'. Valid: {self.all_valid_symbols}"
            )
        return value


# Module-level convenience functions

def get_registry() -> AssetRegistry:
    """Get the singleton AssetRegistry instance."""
    return AssetRegistry()


def get_tradeable_assets() -> list[str]:
    """Get list of all tradeable asset symbols."""
    return AssetRegistry().tradeable_symbols


def validate_asset(value: str) -> str:
    """Validate an asset symbol. Raises ValueError if unknown."""
    return AssetRegistry().validate(value)
