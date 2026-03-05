"""Alpaca paper/live trade executor — REST API.

Supports crypto (BTC, ETH) and stocks (GLDM, SLV) via Alpaca's
unified trading API. Drop-in replacement for PaperExecutor / IBKR Executor.

Set EXECUTOR_MODE=alpaca in .env to activate.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

import requests

from core.logger import setup_logger

log = setup_logger("trading.alpaca_executor")

def _get_alpaca_symbol(asset: str) -> str | None:
    """Map asset to Alpaca symbol using registry metadata."""
    try:
        from core.asset_registry import get_registry
        registry = get_registry()
        config = registry.get_config(asset)
        if config:
            asset_type = config.get("type", "")
            if asset_type == "crypto":
                return f"{asset}/USD"
            # Stocks and ETFs use the ticker directly
            return asset
    except Exception:
        pass
    # Fallback for crypto
    if asset in ("BTC", "ETH"):
        return f"{asset}/USD"
    return asset


class AlpacaExecutor:
    """Trade executor via Alpaca REST API (paper or live)."""

    def __init__(self) -> None:
        self._api_key = os.getenv("ALPACA_API_KEY", "")
        self._secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self._base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")

        # Strip trailing slash
        self._base_url = self._base_url.rstrip("/")

        # Ensure /v2 suffix
        if not self._base_url.endswith("/v2"):
            self._base_url += "/v2"

        self._headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Content-Type": "application/json",
        }

        self.paper_mode = "paper" in self._base_url
        self._order_counter = 0
        self._lock = threading.Lock()

        log.info(
            "Alpaca executor initialized (paper=%s, url=%s)",
            self.paper_mode, self._base_url,
        )

    def _next_order_id(self) -> int:
        with self._lock:
            self._order_counter += 1
            return self._order_counter

    def execute(self, execution_order: dict[str, Any]) -> dict[str, Any]:
        """Execute a trade via Alpaca API. Returns confirmation or error dict."""
        asset = execution_order.get("asset", "")
        direction = execution_order.get("direction", "long")
        quantity = execution_order.get("quantity", 0)
        order_type = execution_order.get("order_type", "market")
        stop_loss = execution_order.get("stop_loss")
        thesis_id = execution_order.get("thesis_id", "")

        symbol = _get_alpaca_symbol(asset)
        if not symbol:
            return {
                "type": "order_error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": f"Unknown asset: {asset}",
                "thesis_id": thesis_id,
            }

        side = "buy" if direction == "long" else "sell"
        is_crypto = "/" in symbol

        # Build order payload
        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "time_in_force": "gtc" if is_crypto else "day",
        }

        # Crypto uses notional for buys, qty for sells; stocks use qty
        if is_crypto:
            if side == "sell":
                # Close orders: send exact base quantity to avoid notional rounding mismatch
                payload["qty"] = str(quantity)
            else:
                # Open orders: use notional (dollar amount) for position sizing
                price = self._get_last_price(symbol)
                notional = round(quantity * price, 2) if price > 0 else 0

                if notional < 10:
                    # Alpaca minimum is $10 for crypto
                    notional = 10.0
                    log.info("Adjusted notional to $10 minimum for %s", symbol)

                payload["notional"] = str(notional)
        else:
            # Stocks: round to whole shares
            qty_int = max(1, int(round(quantity)))
            payload["qty"] = str(qty_int)

        # Submit main order
        try:
            resp = requests.post(
                f"{self._base_url}/orders",
                json=payload,
                headers=self._headers,
                timeout=15,
            )

            if resp.status_code not in (200, 201):
                error_msg = resp.text
                try:
                    error_msg = resp.json().get("message", resp.text)
                except Exception:
                    pass
                log.error("Alpaca order rejected: %s", error_msg)
                return {
                    "type": "order_error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": f"Alpaca rejected: {error_msg}",
                    "thesis_id": thesis_id,
                }

            order_data = resp.json()
        except requests.RequestException as e:
            log.error("Alpaca API request failed: %s", e)
            return {
                "type": "order_error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": f"Alpaca API error: {e}",
                "thesis_id": thesis_id,
            }

        # Wait for fill (market orders fill quickly)
        order_id = order_data.get("id", "")
        fill_price, filled_qty, status = self._wait_for_fill(order_id)

        if status != "filled":
            log.warning("Alpaca order %s status: %s", order_id, status)
            return {
                "type": "order_error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": f"Order not filled — status: {status}",
                "thesis_id": thesis_id,
            }

        # Place stop-loss as a separate order if provided
        if stop_loss and fill_price > 0:
            self._place_stop_loss(symbol, direction, filled_qty, stop_loss, is_crypto)

        log.info(
            "ALPACA FILL: %s %s %.6f @ $%.2f (order_id=%s)",
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

    def _wait_for_fill(
        self, order_id: str, max_wait: int = 30
    ) -> tuple[float, float, str]:
        """Poll order status until filled or timeout."""
        for _ in range(max_wait):
            try:
                resp = requests.get(
                    f"{self._base_url}/orders/{order_id}",
                    headers=self._headers,
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status", "")
                    if status == "filled":
                        fill_price = float(data.get("filled_avg_price", 0))
                        filled_qty = float(data.get("filled_qty", 0))
                        return fill_price, filled_qty, "filled"
                    if status in ("canceled", "expired", "rejected"):
                        return 0.0, 0.0, status
            except Exception:
                pass
            time.sleep(1)

        return 0.0, 0.0, "timeout"

    def _place_stop_loss(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        stop_price: float,
        is_crypto: bool,
    ) -> None:
        """Place a stop-loss order (opposite side)."""
        sl_side = "sell" if direction == "long" else "buy"
        payload = {
            "symbol": symbol,
            "side": sl_side,
            "type": "stop",
            "stop_price": str(round(stop_price, 2)),
            "time_in_force": "gtc",
            "qty": str(round(quantity, 8) if is_crypto else int(round(quantity))),
        }

        try:
            resp = requests.post(
                f"{self._base_url}/orders",
                json=payload,
                headers=self._headers,
                timeout=10,
            )
            if resp.status_code in (200, 201):
                log.info("Stop-loss placed at $%.2f for %s", stop_price, symbol)
            else:
                log.warning("Stop-loss failed: %s", resp.text)
        except Exception as e:
            log.error("Stop-loss request failed: %s", e)

    def _get_last_price(self, symbol: str) -> float:
        """Get latest price from Alpaca for notional calculation."""
        try:
            if "/" in symbol:
                # Crypto — try Alpaca data API
                clean = symbol.replace("/", "")
                resp = requests.get(
                    f"https://data.alpaca.markets/v1beta3/crypto/us/latest/trades?symbols={clean}",
                    headers=self._headers,
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    price = float(data.get("trades", {}).get(clean, {}).get("p", 0))
                    if price > 0:
                        return price

                # Fallback: use MarketDataFetcher (CoinGecko)
                from tools.market_data import MarketDataFetcher
                mdf = MarketDataFetcher()
                asset = symbol.split("/")[0]
                price_data = mdf.get_price(asset)
                return float(price_data.get("price", 0))
            else:
                # Stock
                resp = requests.get(
                    f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest",
                    headers=self._headers,
                    timeout=10,
                )
                if resp.status_code == 200:
                    return float(resp.json().get("trade", {}).get("p", 0))
        except Exception as e:
            log.warning("Failed to get Alpaca price for %s: %s", symbol, e)
        return 0.0
