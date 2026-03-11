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

    def get_broker_position(self, asset: str) -> dict[str, Any] | None:
        """Fetch a single position from Alpaca by asset symbol.

        Returns dict with 'qty' (float), 'side' ('long'/'short'), 'symbol',
        'current_price', 'unrealized_pl', or None if no position exists.
        """
        symbol = _get_alpaca_symbol(asset) or asset
        # Alpaca API uses URL-encoded symbol (BTC/USD → BTC%2FUSD)
        encoded = symbol.replace("/", "%2F")
        try:
            resp = requests.get(
                f"{self._base_url}/positions/{encoded}",
                headers=self._headers,
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "symbol": data.get("symbol", symbol),
                    "qty": abs(float(data.get("qty", 0))),
                    "side": data.get("side", "long"),
                    "current_price": float(data.get("current_price", 0)),
                    "unrealized_pl": float(data.get("unrealized_pl", 0)),
                }
            if resp.status_code == 404:
                # No position on broker
                return None
            log.warning("Alpaca position fetch for %s returned %d", symbol, resp.status_code)
        except Exception as e:
            log.error("Alpaca position fetch failed for %s: %s", symbol, e)
        return None

    def close_position(self, asset: str) -> dict[str, Any]:
        """Close a position using Alpaca's DELETE /positions/{symbol} endpoint.

        This avoids wash trade rejections that occur with counter-direction orders.
        Returns an order confirmation dict or order error dict.
        """
        symbol = _get_alpaca_symbol(asset) or asset
        encoded = symbol.replace("/", "%2F")

        try:
            resp = requests.delete(
                f"{self._base_url}/positions/{encoded}",
                headers=self._headers,
                timeout=15,
            )

            if resp.status_code not in (200, 204):
                error_msg = resp.text
                try:
                    error_msg = resp.json().get("message", resp.text)
                except Exception:
                    pass
                log.error("Alpaca close position rejected for %s: %s", symbol, error_msg)
                return {
                    "type": "order_error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": f"Alpaca close rejected: {error_msg}",
                }

            # HTTP 204 has no body — return synthetic confirmation
            if resp.status_code == 204 or not resp.content:
                log.info("ALPACA CLOSE: %s (204 no content)", asset)
                return {
                    "type": "order_confirmation",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "order_id": "",
                    "asset": asset,
                    "quantity": 0.0,
                    "fill_price": 0.0,
                    "status": "Closed",
                }

            order_data = resp.json()
            order_id = order_data.get("id", "")

            # Wait for fill
            fill_price, filled_qty, status = self._wait_for_fill(order_id)

            if status != "filled":
                log.warning("Alpaca close order %s status: %s", order_id, status)
                return {
                    "type": "order_error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": f"Close order not filled — status: {status}",
                }

            log.info(
                "ALPACA CLOSE: %s %.6f @ $%.2f (order_id=%s)",
                asset, filled_qty, fill_price, order_id,
            )

            return {
                "type": "order_confirmation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order_id": order_id,
                "asset": asset,
                "quantity": filled_qty,
                "fill_price": round(fill_price, 2),
                "status": "Filled",
            }

        except requests.RequestException as e:
            log.error("Alpaca close position API failed for %s: %s", symbol, e)
            return {
                "type": "order_error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": f"Alpaca API error: {e}",
            }

    def get_account_info(self) -> dict[str, Any] | None:
        """Fetch account details from Alpaca (equity, cash, positions)."""
        try:
            resp = requests.get(
                f"{self._base_url}/account",
                headers=self._headers,
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            log.warning("Alpaca account fetch returned %d", resp.status_code)
        except Exception as e:
            log.error("Alpaca account fetch failed: %s", e)
        return None

    @staticmethod
    def _alpaca_symbol_to_asset(symbol: str) -> str:
        """Convert Alpaca symbol back to normalized asset name.

        'BTC/USD' → 'BTC', 'BTCUSD' → 'BTC', 'AAPL' → 'AAPL'.
        """
        if "/" in symbol and symbol.endswith("/USD"):
            return symbol.split("/")[0]
        if symbol.endswith("USD") and len(symbol) > 3:
            # Handle BTCUSD, ETHUSD formats
            base = symbol[:-3]
            if base in ("BTC", "ETH"):
                return base
        return symbol

    def get_all_positions(self) -> dict[str, dict[str, Any]]:
        """Fetch all open positions from Alpaca.

        Returns dict keyed by normalized asset symbol (BTC not BTC/USD):
        {asset: {"symbol": str, "qty": float, "side": str,
                 "avg_entry_price": float, "current_price": float,
                 "unrealized_pl": float, "qty_available": float}}
        """
        try:
            resp = requests.get(
                f"{self._base_url}/positions",
                headers=self._headers,
                timeout=10,
            )
            if resp.status_code != 200:
                log.warning("Alpaca get_all_positions returned %d", resp.status_code)
                return {}

            positions: dict[str, dict[str, Any]] = {}
            for pos in resp.json():
                raw_symbol = pos.get("symbol", "")
                asset = self._alpaca_symbol_to_asset(raw_symbol)
                positions[asset] = {
                    "symbol": raw_symbol,
                    "qty": abs(float(pos.get("qty", 0))),
                    "side": pos.get("side", "long"),
                    "avg_entry_price": float(pos.get("avg_entry_price", 0)),
                    "current_price": float(pos.get("current_price", 0)),
                    "unrealized_pl": float(pos.get("unrealized_pl", 0)),
                    "qty_available": float(pos.get("qty_available", 0)),
                }
            return positions
        except Exception as e:
            log.error("Alpaca get_all_positions failed: %s", e)
            return {}

    def get_open_orders(self) -> list[dict[str, Any]]:
        """Fetch all open orders from Alpaca.

        Returns list of dicts with keys: id, symbol, asset, side, qty,
        type, status, created_at. Returns empty list on error.
        """
        try:
            resp = requests.get(
                f"{self._base_url}/orders",
                params={"status": "open"},
                headers=self._headers,
                timeout=10,
            )
            if resp.status_code != 200:
                log.warning("Alpaca get_open_orders returned %d", resp.status_code)
                return []

            orders: list[dict[str, Any]] = []
            for order in resp.json():
                raw_symbol = order.get("symbol", "")
                orders.append({
                    "id": order.get("id", ""),
                    "symbol": raw_symbol,
                    "asset": self._alpaca_symbol_to_asset(raw_symbol),
                    "side": order.get("side", ""),
                    "qty": float(order.get("qty", 0) or 0),
                    "type": order.get("type", ""),
                    "status": order.get("status", ""),
                    "created_at": order.get("created_at", ""),
                })
            return orders
        except Exception as e:
            log.error("Alpaca get_open_orders failed: %s", e)
            return []
