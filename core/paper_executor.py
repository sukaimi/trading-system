"""Paper trade executor — simulates fills at real market prices.

Drop-in replacement for the IBKR Executor. Uses MarketDataFetcher to get
live prices from CoinGecko/yfinance, applies small slippage, and returns
confirmation dicts in the same format as the IBKR executor.

Switch between executors with EXECUTOR_MODE=paper|ibkr in .env.
"""

from __future__ import annotations

import random
import threading
from datetime import datetime, timezone
from typing import Any

from core.logger import setup_logger
from tools.market_data import MarketDataFetcher

log = setup_logger("trading.paper_executor")

SLIPPAGE_PCT = 0.0005  # 0.05% simulated slippage


class PaperExecutor:
    """Simulated executor for paper trading with real market prices."""

    def __init__(self) -> None:
        self._market = MarketDataFetcher()
        self._order_counter = 0
        self._lock = threading.Lock()
        self.paper_mode = True

    def _next_order_id(self) -> int:
        with self._lock:
            self._order_counter += 1
            return self._order_counter

    def execute(self, execution_order: dict[str, Any]) -> dict[str, Any]:
        """Simulate a trade fill at current market price + slippage."""
        asset = execution_order.get("asset", "")
        direction = execution_order.get("direction", "long")
        quantity = execution_order.get("quantity", 0)
        thesis_id = execution_order.get("thesis_id", "")

        # Fetch live price
        try:
            price_data = self._market.get_price(asset)
            market_price = price_data.get("price", 0)
        except Exception as e:
            log.error("Failed to fetch price for %s: %s", asset, e)
            return {
                "type": "order_error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": f"Price fetch failed for {asset}: {e}",
                "thesis_id": thesis_id,
            }

        if not market_price or market_price <= 0:
            return {
                "type": "order_error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": f"No valid price for {asset}",
                "thesis_id": thesis_id,
            }

        # Apply slippage — buys fill slightly higher, sells slightly lower
        slippage = market_price * SLIPPAGE_PCT * random.uniform(0.5, 1.5)
        if direction == "long":
            fill_price = market_price + slippage
        else:
            fill_price = market_price - slippage

        order_id = self._next_order_id()

        log.info(
            "PAPER FILL: %s %s %.6f @ $%.2f (market: $%.2f, slippage: $%.2f)",
            direction.upper(), asset, quantity, fill_price, market_price, slippage,
        )

        return {
            "type": "order_confirmation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order_id": order_id,
            "asset": asset,
            "direction": direction,
            "quantity": quantity,
            "fill_price": round(fill_price, 2),
            "status": "Filled",
            "thesis_id": thesis_id,
        }
