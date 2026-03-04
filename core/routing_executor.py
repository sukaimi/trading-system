"""Routing executor — routes orders to the correct executor by asset type.

crypto (BTC, ETH) → CoinbaseExecutor
stocks/ETFs (AAPL, NVDA, SPY, etc.) → AlpacaExecutor

Used when EXECUTOR_MODE=live. Transparent to the pipeline.
"""

from __future__ import annotations

from typing import Any

from core.alpaca_executor import AlpacaExecutor
from core.coinbase_executor import CoinbaseExecutor
from core.logger import setup_logger

log = setup_logger("trading.routing_executor")


class RoutingExecutor:
    """Routes orders to the correct executor based on asset type."""

    def __init__(self) -> None:
        self._alpaca = AlpacaExecutor()
        self._coinbase = CoinbaseExecutor()
        self.paper_mode = False

        log.info(
            "Routing executor initialized (alpaca_paper=%s)",
            self._alpaca.paper_mode,
        )

    def execute(self, execution_order: dict[str, Any]) -> dict[str, Any]:
        """Route execution to the appropriate executor based on asset type."""
        asset = execution_order.get("asset", "")
        executor = self._get_executor(asset)

        log.info("Routing %s → %s", asset, type(executor).__name__)

        return executor.execute(execution_order)

    def _get_executor(self, asset: str) -> AlpacaExecutor | CoinbaseExecutor:
        """Determine which executor handles this asset."""
        try:
            from core.asset_registry import get_registry
            config = get_registry().get_config(asset)
            if config and config.get("type") == "crypto":
                return self._coinbase
        except Exception:
            pass

        # Fallback for known crypto
        if asset in ("BTC", "ETH"):
            return self._coinbase

        return self._alpaca
