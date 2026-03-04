"""Coinbase crypto executor — Advanced Trade API.

Supports BTC and ETH market orders via Coinbase's Advanced Trade API.
Used by RoutingExecutor in live mode (EXECUTOR_MODE=live).

Env vars: COINBASE_API_KEY, COINBASE_API_SECRET
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.coinbase_executor")


class CoinbaseExecutor:
    """Trade executor via Coinbase Advanced Trade API (live crypto only)."""

    def __init__(self) -> None:
        self._api_key = os.getenv("COINBASE_API_KEY", "")
        # python-dotenv stores \n as literal chars; PEM needs real newlines
        self._api_secret = os.getenv("COINBASE_API_SECRET", "").replace("\\n", "\n")
        self.paper_mode = False
        self._order_counter = 0
        self._lock = threading.Lock()
        self._client = None
        self._init_client()
        log.info("Coinbase executor initialized")

    def _init_client(self) -> None:
        """Initialize the Coinbase Advanced Trade SDK client."""
        if not self._api_key or not self._api_secret:
            log.warning("Coinbase API credentials not set")
            return
        try:
            from coinbase.rest import RESTClient
            self._client = RESTClient(
                api_key=self._api_key,
                api_secret=self._api_secret,
            )
        except ImportError:
            log.error("coinbase-advanced-py not installed: pip install coinbase-advanced-py")
        except Exception as e:
            log.error("Coinbase client init failed: %s", e)

    def execute(self, execution_order: dict[str, Any]) -> dict[str, Any]:
        """Execute a crypto trade via Coinbase. Returns confirmation or error dict."""
        asset = execution_order.get("asset", "")
        direction = execution_order.get("direction", "long")
        quantity = execution_order.get("quantity", 0)
        thesis_id = execution_order.get("thesis_id", "")

        if not self._client:
            return self._error("Coinbase client not initialized", thesis_id)

        product_id = f"{asset}-USD"
        side = "BUY" if direction == "long" else "SELL"
        client_order_id = str(uuid.uuid4())

        try:
            # Get current price for notional calculation
            current_price = self._get_price(asset)
            if current_price <= 0:
                return self._error(f"Cannot get price for {asset}", thesis_id)

            # Place market order
            if side == "BUY":
                quote_size = str(round(quantity * current_price, 2))
                order = self._client.market_order_buy(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    quote_size=quote_size,
                )
            else:
                base_size = str(round(quantity, 8))
                order = self._client.market_order_sell(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size=base_size,
                )

            # Check immediate response
            if not order.get("success"):
                error_resp = order.get("error_response", {})
                error_msg = error_resp.get("message", "") or order.get("failure_reason", "Unknown error")
                log.error("Coinbase order rejected: %s", error_msg)
                return self._error(f"Coinbase rejected: {error_msg}", thesis_id)

            order_id = order.get("order_id", client_order_id)

            # Wait for fill
            fill_price, filled_qty, status = self._wait_for_fill(order_id)

            if status != "FILLED":
                log.warning("Coinbase order %s status: %s", order_id, status)
                return self._error(f"Order not filled — status: {status}", thesis_id)

            log.info(
                "COINBASE FILL: %s %s %.8f @ $%.2f (order_id=%s)",
                direction.upper(), asset, filled_qty, fill_price, order_id,
            )

            return {
                "type": "order_confirmation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order_id": order_id,
                "asset": asset,
                "direction": direction,
                "quantity": filled_qty,
                "fill_price": round(fill_price, 2),
                "status": "Filled",
                "thesis_id": thesis_id,
            }

        except Exception as e:
            log.error("Coinbase execution failed for %s: %s", asset, e)
            return self._error(f"Coinbase API error: {e}", thesis_id)

    def _wait_for_fill(
        self, order_id: str, max_wait: int = 30
    ) -> tuple[float, float, str]:
        """Poll order status until filled or timeout."""
        for _ in range(max_wait):
            try:
                resp = self._client.get_order(order_id)
                order_data = resp.get("order", resp) if isinstance(resp, dict) else resp
                status = order_data.get("status", "") if isinstance(order_data, dict) else ""
                if status == "FILLED":
                    avg_price = float(order_data.get("average_filled_price", 0))
                    filled_size = float(order_data.get("filled_size", 0))
                    return avg_price, filled_size, "FILLED"
                if status in ("CANCELLED", "EXPIRED", "FAILED"):
                    return 0.0, 0.0, status
            except Exception:
                pass
            time.sleep(1)
        return 0.0, 0.0, "timeout"

    def _get_price(self, asset: str) -> float:
        """Get current price from Coinbase, fallback to CoinGecko."""
        try:
            product = self._client.get_product(f"{asset}-USD")
            price = float(product.get("price", 0))
            if price > 0:
                return price
        except Exception as e:
            log.warning("Coinbase price fetch failed for %s: %s", asset, e)

        try:
            from tools.market_data import MarketDataFetcher
            mdf = MarketDataFetcher()
            price_data = mdf.get_price(asset)
            return float(price_data.get("price", 0))
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _error(message: str, thesis_id: str) -> dict[str, Any]:
        return {
            "type": "order_error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": message,
            "thesis_id": thesis_id,
        }
