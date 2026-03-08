"""Trading friction simulator — realistic paper trading costs.

Simulates spread, commission, and short borrow costs during paper trading
to match real broker behavior. Auto-disables when executor is in live mode.
"""

import json
import os
from datetime import datetime, timezone
from core.logger import setup_logger

log = setup_logger("trading.friction")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "risk_params.json")


def _load_friction_config():
    """Load friction params from risk_params.json."""
    try:
        with open(CONFIG_PATH) as f:
            params = json.load(f)
        return params.get("trading_friction", {})
    except Exception:
        return {}


def _get_asset_type(asset: str) -> str:
    """Return 'crypto', 'etf', or 'stock' for the asset."""
    try:
        from core.asset_registry import get_registry
        registry = get_registry()
        config = registry.get_config(asset)
        if config:
            return config.get("type", "stock")
    except Exception:
        pass
    if asset in ("BTC", "ETH"):
        return "crypto"
    if asset in ("SPY", "GLDM", "SLV", "TLT", "XLE", "EWS", "FXI"):
        return "etf"
    return "stock"


class TradingFriction:
    """Calculates realistic trading friction for paper trading."""

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self._config = _load_friction_config()

    def reload_config(self):
        self._config = _load_friction_config()

    @property
    def enabled(self) -> bool:
        return self.paper_mode and self._config.get("enabled", True)

    def spread_cost(self, asset: str, price: float, quantity: float, direction: str) -> float:
        """Calculate spread cost (always negative -- cost to the trader).

        Spread is the difference between bid and ask. When you buy, you pay the ask (higher).
        When you sell, you get the bid (lower). This simulates that cost.
        Returns the dollar amount to deduct from equity.
        """
        if not self.enabled:
            return 0.0
        asset_type = _get_asset_type(asset)
        spread_pcts = self._config.get("spread_pct", {})
        spread_pct = spread_pcts.get(asset_type, 0.05) / 100  # Convert from pct to decimal
        # Half-spread applied to each side (buy or sell)
        cost = price * quantity * spread_pct / 2
        return round(cost, 4)

    def commission(self, asset: str, price: float, quantity: float) -> float:
        """Calculate commission cost. Returns dollar amount to deduct."""
        if not self.enabled:
            return 0.0
        asset_type = _get_asset_type(asset)
        comm_pcts = self._config.get("commission_pct", {})
        comm_pct = comm_pcts.get(asset_type, 0.0) / 100
        cost = price * quantity * comm_pct
        return round(cost, 4)

    def borrow_cost_daily(self, asset: str, price: float, quantity: float) -> float:
        """Calculate daily borrow cost for a short position.

        Annualized rate / 365 = daily cost.
        Returns dollar amount to deduct per day.
        """
        if not self.enabled:
            return 0.0
        asset_type = _get_asset_type(asset)
        borrow_rates = self._config.get("short_borrow_annual_pct", {})
        annual_pct = borrow_rates.get(asset_type, 1.5) / 100
        daily_cost = (price * quantity * annual_pct) / 365
        return round(daily_cost, 4)

    def total_entry_cost(self, asset: str, price: float, quantity: float, direction: str) -> float:
        """Total friction cost at entry (spread + commission). Returns dollar amount."""
        spread = self.spread_cost(asset, price, quantity, direction)
        comm = self.commission(asset, price, quantity)
        total = spread + comm
        if total > 0:
            log.info("FRICTION [entry %s %s]: spread=$%.4f + commission=$%.4f = $%.4f",
                     direction, asset, spread, comm, total)
        return total

    def total_exit_cost(self, asset: str, price: float, quantity: float, direction: str) -> float:
        """Total friction cost at exit (spread + commission). Returns dollar amount."""
        spread = self.spread_cost(asset, price, quantity, direction)
        comm = self.commission(asset, price, quantity)
        total = spread + comm
        if total > 0:
            log.info("FRICTION [exit %s %s]: spread=$%.4f + commission=$%.4f = $%.4f",
                     direction, asset, spread, comm, total)
        return total

    def accrued_borrow_cost(self, asset: str, price: float, quantity: float, days_held: float) -> float:
        """Total accrued borrow cost for a short position over days_held."""
        if not self.enabled:
            return 0.0
        daily = self.borrow_cost_daily(asset, price, quantity)
        return round(daily * days_held, 4)
