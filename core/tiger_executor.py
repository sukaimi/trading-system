"""Tiger Brokers trade executor — tigeropen SDK.

Supports stocks and ETFs listed on SGX, US exchanges, and HK via Tiger Brokers.
Drop-in replacement for AlpacaExecutor in the execution pipeline.

Set EXECUTOR_MODE=tiger in .env to activate.

Required env vars:
  TIGER_ID              — Tiger Brokers account ID (numeric string)
  TIGER_PRIVATE_KEY     — Path to private key PEM file (RSA, generated in Tiger Developer Portal)
  TIGER_ACCOUNT         — Trading account number (standard or paper)
  TIGER_PAPER_MODE      — Set to "true" to use Tiger paper trading account (default: false)

Install SDK:
  pip install tigeropen
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.tiger_executor")

# ---------------------------------------------------------------------------
# Asset → Tiger contract mapping helpers
# ---------------------------------------------------------------------------

# Tiger exchange identifiers for assets in the trading system
# SGX assets use SGD; all US-listed assets use USD
_TIGER_EXCHANGE_MAP: dict[str, dict[str, str]] = {
    # US stocks / ETFs — listed on US exchanges, priced in USD
    "AAPL":  {"exchange": "NASDAQ", "currency": "USD"},
    "NVDA":  {"exchange": "NASDAQ", "currency": "USD"},
    "TSLA":  {"exchange": "NASDAQ", "currency": "USD"},
    "AMZN":  {"exchange": "NASDAQ", "currency": "USD"},
    "META":  {"exchange": "NASDAQ", "currency": "USD"},
    "SPY":   {"exchange": "NYSE ARCA", "currency": "USD"},
    "TLT":   {"exchange": "NASDAQ",   "currency": "USD"},
    "GLDM":  {"exchange": "NYSE ARCA", "currency": "USD"},
    "SLV":   {"exchange": "NYSE ARCA", "currency": "USD"},
    "XLE":   {"exchange": "NYSE ARCA", "currency": "USD"},
    # ETFs listed on NYSE ARCA but tracking foreign markets — still USD-priced
    "EWS":   {"exchange": "NYSE ARCA", "currency": "USD"},
    "FXI":   {"exchange": "NYSE ARCA", "currency": "USD"},
}

# Crypto — Tiger Brokers supports crypto via separate product; we note these
# here so we can give an early rejection rather than submitting a bad order.
_UNSUPPORTED_ASSETS: frozenset[str] = frozenset({"BTC", "ETH"})


def _get_tiger_contract_params(asset: str) -> dict[str, str] | None:
    """Return the Tiger contract params for a given asset symbol.

    Tries the asset registry first, then falls back to the static map above.
    Returns None if the asset is unsupported or unknown.
    """
    if asset in _UNSUPPORTED_ASSETS:
        return None

    # Try dynamic registry (handles future assets added to assets.json)
    try:
        from core.asset_registry import get_registry
        registry = get_registry()
        config = registry.get_config(asset)
        if config:
            asset_type = config.get("type", "")
            if asset_type == "crypto":
                return None  # Tiger crypto not supported via this executor
            exchange = config.get("exchange", "NASDAQ")
            currency = config.get("currency", "USD")
            return {"exchange": exchange, "currency": currency}
    except Exception as exc:
        log.debug("Asset registry lookup failed for %s: %s", asset, exc)

    # Static fallback map
    return _TIGER_EXCHANGE_MAP.get(asset)


# ---------------------------------------------------------------------------
# TigerExecutor
# ---------------------------------------------------------------------------

class TigerExecutor:
    """Trade executor via Tiger Brokers (tigeropen SDK).

    Supports paper and live trading for stocks and ETFs on US, SGX, and HK
    exchanges. Crypto orders are rejected — use a separate executor for crypto.

    All public methods are exception-safe and will never raise: errors are
    returned as order_error dicts (for execute/close_position) or empty
    collections (for query methods).
    """

    def __init__(self) -> None:
        self._tiger_id = os.getenv("TIGER_ID", "")
        self._private_key_path = os.getenv("TIGER_PRIVATE_KEY", "")
        self._account = os.getenv("TIGER_ACCOUNT", "")

        paper_env = os.getenv("TIGER_PAPER_MODE", "false").strip().lower()
        self.paper_mode: bool = paper_env in ("1", "true", "yes")

        self._order_counter = 0
        self._lock = threading.Lock()

        # SDK clients — lazily initialised; None means "not available"
        self._client: Any = None  # tigeropen.trade.trade_client.TradeClient
        self._quote_client: Any = None  # tigeropen.quote.quote_client.QuoteClient

        self._init_client()
        log.info(
            "Tiger executor initialized (paper=%s, account=%s)",
            self.paper_mode,
            self._account or "<not set>",
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """Create the tigeropen TradeClient.  Logs and swallows all errors."""
        if not self._tiger_id:
            log.warning("TIGER_ID not set — Tiger executor will not be functional")
            return
        if not self._private_key_path:
            log.warning("TIGER_PRIVATE_KEY not set — Tiger executor will not be functional")
            return
        if not self._account:
            log.warning("TIGER_ACCOUNT not set — Tiger executor will not be functional")
            return

        try:
            from tigeropen.common.consts import Language, TigerEnvironment
            from tigeropen.tiger_open_client_config import TigerOpenClientConfig
            from tigeropen.trade.trade_client import TradeClient

            env = (
                TigerEnvironment.PAPER
                if self.paper_mode
                else TigerEnvironment.PROD
            )

            cfg = TigerOpenClientConfig()
            cfg.tiger_id = self._tiger_id
            cfg.private_key_path = self._private_key_path
            cfg.account = self._account
            cfg.env = env
            cfg.language = Language.en_US

            self._client = TradeClient(cfg)

            from tigeropen.quote.quote_client import QuoteClient
            self._quote_client = QuoteClient(cfg)

            log.info(
                "Tiger TradeClient connected (env=%s)",
                "PAPER" if self.paper_mode else "PROD",
            )
        except ImportError:
            log.error(
                "tigeropen not installed — run: pip install tigeropen"
            )
        except Exception as e:
            log.error("Tiger client init failed: %s", e)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_order_id(self) -> int:
        with self._lock:
            self._order_counter += 1
            return self._order_counter

    @staticmethod
    def _error(message: str, thesis_id: str = "") -> dict[str, Any]:
        """Build a standardised order_error response."""
        return {
            "type": "order_error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": message,
            "thesis_id": thesis_id,
        }

    def _build_stock_contract(self, asset: str, exchange: str, currency: str) -> Any:
        """Build a Tiger stock_contract object for the given asset."""
        from tigeropen.common.util.contract_utils import stock_contract
        return stock_contract(symbol=asset, currency=currency, exchange=exchange)

    def _get_last_price(self, asset: str) -> float:
        """Fetch the latest price from Tiger quote API, falling back to MarketDataFetcher."""
        try:
            if not self._quote_client:
                raise RuntimeError("QuoteClient not initialized")
            briefs = self._quote_client.get_stock_briefs([asset])
            if briefs is not None and len(briefs) > 0:
                row = briefs.iloc[0]
                price = float(row.get("latest_price", 0) or 0)
                if price > 0:
                    return price
        except Exception as e:
            log.warning("Tiger price fetch failed for %s: %s", asset, e)

        # Fallback: MarketDataFetcher (CoinGecko / yfinance)
        try:
            from tools.market_data import MarketDataFetcher
            mdf = MarketDataFetcher()
            price_data = mdf.get_price(asset)
            return float(price_data.get("price", 0))
        except Exception as e:
            log.warning("MarketDataFetcher fallback failed for %s: %s", asset, e)

        return 0.0

    def _wait_for_fill(
        self, order_id: int | str, max_wait: int = 30
    ) -> tuple[float, float, str]:
        """Poll Tiger order status until filled or timeout.

        Returns (fill_price, filled_qty, status_str).
        Terminal statuses: FILLED, CANCELLED, REJECTED, EXPIRED.
        """
        terminal_ok = {"FILLED"}
        terminal_bad = {"CANCELLED", "CANCELED", "REJECTED", "EXPIRED", "FAILED"}

        for _ in range(max_wait):
            try:
                orders = self._client.get_orders(
                    account=self._account, order_ids=[order_id]
                )
                if orders is not None and len(orders) > 0:
                    row = orders.iloc[0] if hasattr(orders, "iloc") else orders[0]
                    status = str(row.get("status", "") if isinstance(row, dict) else getattr(row, "status", "")).upper()

                    if status in terminal_ok:
                        fill_price = float(
                            (row.get("avg_fill_price", 0) if isinstance(row, dict) else getattr(row, "avg_fill_price", 0)) or 0
                        )
                        filled_qty = float(
                            (row.get("filled", 0) if isinstance(row, dict) else getattr(row, "filled", 0)) or 0
                        )
                        return fill_price, filled_qty, "FILLED"

                    if status in terminal_bad:
                        return 0.0, 0.0, status
            except Exception as e:
                log.debug("Tiger poll order %s: %s", order_id, e)

            time.sleep(1)

        return 0.0, 0.0, "timeout"

    # ------------------------------------------------------------------
    # Public interface (mirrors AlpacaExecutor)
    # ------------------------------------------------------------------

    def execute(self, execution_order: dict[str, Any]) -> dict[str, Any]:
        """Execute a trade via Tiger Brokers. Returns order_confirmation or order_error dict.

        Supports market orders for stocks and ETFs. Crypto is not supported.
        """
        asset = execution_order.get("asset", "")
        direction = execution_order.get("direction", "long")
        quantity = execution_order.get("quantity", 0)
        order_type = execution_order.get("order_type", "market")
        thesis_id = execution_order.get("thesis_id", "")

        if not self._client:
            return self._error("Tiger client not initialized — check TIGER_ID, TIGER_PRIVATE_KEY, TIGER_ACCOUNT", thesis_id)

        # Reject unsupported assets (crypto)
        if asset in _UNSUPPORTED_ASSETS:
            return self._error(
                f"Tiger executor does not support crypto asset {asset} — use CoinbaseExecutor",
                thesis_id,
            )

        # Resolve contract parameters
        contract_params = _get_tiger_contract_params(asset)
        if not contract_params:
            return self._error(f"Unknown or unsupported asset: {asset}", thesis_id)

        exchange = contract_params["exchange"]
        currency = contract_params["currency"]

        # Tiger uses integer share quantities for stocks and ETFs
        qty_int = max(1, int(round(quantity)))
        action = "BUY" if direction == "long" else "SELL"

        try:
            from tigeropen.common.util.contract_utils import stock_contract
            from tigeropen.common.util.order_utils import market_order

            contract = stock_contract(symbol=asset, currency=currency, exchange=exchange)

            if order_type == "market":
                order = market_order(
                    account=self._account,
                    contract=contract,
                    action=action,
                    quantity=qty_int,
                )
            else:
                # Limit order: fetch current price and place at market (best-effort)
                limit_price = execution_order.get("limit_price")
                if not limit_price:
                    limit_price = self._get_last_price(asset)
                if not limit_price or limit_price <= 0:
                    return self._error(f"Cannot determine limit price for {asset}", thesis_id)

                from tigeropen.common.util.order_utils import limit_order
                order = limit_order(
                    account=self._account,
                    contract=contract,
                    action=action,
                    quantity=qty_int,
                    limit_price=round(float(limit_price), 4),
                )

            # Submit order and capture Tiger order_id
            order_id = self._client.place_order(order)

            if order_id is None:
                log.error("Tiger place_order returned None for %s", asset)
                return self._error("Tiger rejected order: no order_id returned", thesis_id)

            log.info(
                "Tiger order submitted: %s %s %d qty (order_id=%s)",
                action, asset, qty_int, order_id,
            )

        except Exception as e:
            log.error("Tiger place_order failed for %s: %s", asset, e)
            return self._error(f"Tiger API error: {e}", thesis_id)

        # Wait for fill
        try:
            numeric_id = int(order_id)
        except (TypeError, ValueError):
            log.error("Tiger order_id is not numeric: %r", order_id)
            return self._error(f"Unexpected order_id format: {order_id}", thesis_id)
        fill_price, filled_qty, status = self._wait_for_fill(numeric_id)

        if status != "FILLED":
            log.warning("Tiger order %s for %s status: %s", order_id, asset, status)
            return self._error(f"Order not filled — status: {status}", thesis_id)

        # Best-effort fill price: Tiger may return 0 on paper for market orders
        if fill_price <= 0:
            fill_price = self._get_last_price(asset)

        actual_qty = filled_qty if filled_qty > 0 else float(qty_int)

        log.info(
            "TIGER FILL: %s %s %.4f @ $%.4f (order_id=%s, paper=%s)",
            direction.upper(), asset, actual_qty, fill_price, order_id, self.paper_mode,
        )

        return {
            "type": "order_confirmation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order_id": str(order_id),
            "asset": asset,
            "direction": direction,
            "quantity": actual_qty,
            "fill_price": round(fill_price, 4),
            "status": "Filled",
            "thesis_id": thesis_id,
        }

    def close_position(self, asset: str) -> dict[str, Any]:
        """Close an open position for the given asset.

        Cancels any open orders for the symbol first, then submits a
        market order on the opposite side to flatten the position.
        Returns order_confirmation or order_error dict.
        """
        if not self._client:
            return self._error("Tiger client not initialized")

        if asset in _UNSUPPORTED_ASSETS:
            return self._error(f"Tiger executor does not support crypto asset {asset}")

        contract_params = _get_tiger_contract_params(asset)
        if not contract_params:
            return self._error(f"Unknown or unsupported asset: {asset}")

        # Cancel any open orders for this symbol to free inventory
        canceled, stuck = self.cancel_orders_for_symbol(asset)
        if canceled > 0:
            log.info("Canceled %d open orders for %s before closing position", canceled, asset)
            time.sleep(0.5)
        if stuck > 0:
            log.warning("SKIP CLOSE %s: %d orders failed to cancel — inventory may be reserved", asset, stuck)
            return self._error(f"Cannot close {asset}: {stuck} orders failed to cancel")

        # Determine position side and quantity from broker
        broker_pos = self.get_broker_position(asset)
        if not broker_pos:
            log.info("No Tiger position found for %s — nothing to close", asset)
            return {
                "type": "order_confirmation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order_id": "",
                "asset": asset,
                "quantity": 0.0,
                "fill_price": 0.0,
                "status": "No position to close",
            }

        position_side = broker_pos.get("side", "long")
        position_qty = broker_pos.get("qty", 0.0)
        close_qty = max(1, int(round(position_qty)))

        # Close: sell if long, buy-to-cover if short
        close_action = "SELL" if position_side == "long" else "BUY"
        exchange = contract_params["exchange"]
        currency = contract_params["currency"]

        try:
            from tigeropen.common.util.contract_utils import stock_contract
            from tigeropen.common.util.order_utils import market_order

            contract = stock_contract(symbol=asset, currency=currency, exchange=exchange)
            order = market_order(
                account=self._account,
                contract=contract,
                action=close_action,
                quantity=close_qty,
            )
            order_id = self._client.place_order(order)

            if order_id is None:
                return self._error(f"Tiger close order rejected for {asset}: no order_id")

            log.info(
                "Tiger close order submitted: %s %s %d qty (order_id=%s)",
                close_action, asset, close_qty, order_id,
            )
        except Exception as e:
            log.error("Tiger close_position failed for %s: %s", asset, e)
            return self._error(f"Tiger API error: {e}")

        try:
            numeric_id = int(order_id)
        except (TypeError, ValueError):
            log.error("Tiger close order_id is not numeric: %r", order_id)
            return self._error(f"Unexpected order_id format: {order_id}")
        fill_price, filled_qty, status = self._wait_for_fill(numeric_id)

        if status != "FILLED":
            log.warning("Tiger close order %s for %s status: %s", order_id, asset, status)
            return self._error(f"Close order not filled — status: {status}")

        if fill_price <= 0:
            fill_price = self._get_last_price(asset)

        actual_qty = filled_qty if filled_qty > 0 else float(close_qty)

        log.info(
            "TIGER CLOSE: %s %.4f @ $%.4f (order_id=%s, paper=%s)",
            asset, actual_qty, fill_price, order_id, self.paper_mode,
        )

        return {
            "type": "order_confirmation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order_id": str(order_id),
            "asset": asset,
            "quantity": actual_qty,
            "fill_price": round(fill_price, 4),
            "status": "Filled",
        }

    def get_broker_position(self, asset: str) -> dict[str, Any] | None:
        """Fetch a single open position from Tiger by asset symbol.

        Returns a dict with keys: symbol, qty, side, avg_entry_price,
        current_price, unrealized_pl. Returns None if no position or on error.
        """
        if not self._client:
            return None

        try:
            positions_df = self._client.get_positions(account=self._account)
            if positions_df is None or len(positions_df) == 0:
                return None

            # tigeropen returns a DataFrame; iterate rows
            for _, row in positions_df.iterrows():
                row_symbol = str(row.get("contract", "") or "")
                # Tiger may return symbol as part of contract or as a 'symbol' column
                if not row_symbol:
                    row_symbol = str(row.get("symbol", "") or "")
                # Strip exchange suffix if present (e.g. "AAPL.US" → "AAPL")
                base_symbol = row_symbol.split(".")[0].upper()

                if base_symbol != asset.upper():
                    continue

                position = float(row.get("position", 0) or 0)
                side = "long" if position >= 0 else "short"
                qty = abs(position)
                avg_cost = float(row.get("average_cost", 0) or 0)
                latest_price = float(row.get("latest_price", 0) or 0)
                unrealized_pl = float(row.get("unrealized_pnl", 0) or 0)

                return {
                    "symbol": base_symbol,
                    "qty": qty,
                    "qty_available": qty,
                    "side": side,
                    "avg_entry_price": avg_cost,
                    "current_price": latest_price,
                    "unrealized_pl": unrealized_pl,
                }

            # Asset not found in positions
            return None

        except Exception as e:
            log.error("Tiger get_broker_position failed for %s: %s", asset, e)
            return None

    def get_all_positions(self) -> dict[str, dict[str, Any]]:
        """Fetch all open positions from Tiger.

        Returns dict keyed by normalized asset symbol:
        {asset: {"symbol": str, "qty": float, "side": str,
                 "avg_entry_price": float, "current_price": float,
                 "unrealized_pl": float}}
        """
        if not self._client:
            return {}

        try:
            positions_df = self._client.get_positions(account=self._account)
            if positions_df is None or len(positions_df) == 0:
                return {}

            result: dict[str, dict[str, Any]] = {}

            for _, row in positions_df.iterrows():
                raw_symbol = str(row.get("contract", "") or row.get("symbol", "") or "")
                # Strip exchange suffix if present (e.g. "AAPL.US" → "AAPL")
                asset = raw_symbol.split(".")[0].upper()
                if not asset:
                    continue

                position = float(row.get("position", 0) or 0)
                side = "long" if position >= 0 else "short"
                qty = abs(position)
                avg_cost = float(row.get("average_cost", 0) or 0)
                latest_price = float(row.get("latest_price", 0) or 0)
                unrealized_pl = float(row.get("unrealized_pnl", 0) or 0)

                result[asset] = {
                    "symbol": raw_symbol,
                    "qty": qty,
                    "qty_available": qty,
                    "side": side,
                    "avg_entry_price": avg_cost,
                    "current_price": latest_price,
                    "unrealized_pl": unrealized_pl,
                }

            return result

        except Exception as e:
            log.error("Tiger get_all_positions failed: %s", e)
            return {}

    def get_open_orders(self) -> list[dict[str, Any]]:
        """Fetch all open orders from Tiger.

        Returns list of dicts with keys: id, symbol, asset, side, qty,
        type, status, created_at. Returns empty list on error.
        """
        if not self._client:
            return []

        try:
            orders_df = self._client.get_orders(
                account=self._account, status_filter="working"
            )
            if orders_df is None or len(orders_df) == 0:
                return []

            orders: list[dict[str, Any]] = []

            for _, row in orders_df.iterrows():
                raw_symbol = str(row.get("contract", "") or row.get("symbol", "") or "")
                asset = raw_symbol.split(".")[0].upper()
                order_id = row.get("id", "") or row.get("order_id", "")
                side_raw = str(row.get("action", "") or "").upper()
                side = "buy" if side_raw == "BUY" else "sell"
                qty = float(row.get("quantity", 0) or 0)
                order_type = str(row.get("order_type", "") or "").lower()
                status = str(row.get("status", "") or "").upper()
                created_at = str(row.get("created_at", "") or "")

                orders.append({
                    "id": str(order_id),
                    "symbol": raw_symbol,
                    "asset": asset,
                    "side": side,
                    "qty": qty,
                    "type": order_type,
                    "status": status,
                    "created_at": created_at,
                })

            return orders

        except Exception as e:
            log.error("Tiger get_open_orders failed: %s", e)
            return []

    def cancel_orders_for_symbol(self, asset: str) -> tuple[int, int]:
        """Cancel all open orders for a specific asset symbol.

        Returns (canceled_count, failed_count).
        Failed means the cancel API call itself errored — the order may
        still be live and should be treated as stuck.
        """
        if not self._client:
            return 0, 0

        canceled = 0
        failed = 0

        try:
            open_orders = self.get_open_orders()
        except Exception as e:
            log.warning("Tiger cancel_orders_for_symbol(%s) — get_open_orders failed: %s", asset, e)
            return 0, 0

        for order in open_orders:
            if order.get("asset", "").upper() != asset.upper():
                continue

            order_id = order.get("id", "")
            if not order_id:
                continue

            try:
                self._client.cancel_order(
                    account=self._account, order_id=int(order_id)
                )
                log.info(
                    "Canceled Tiger order %s for %s (status was %s)",
                    order_id, asset, order.get("status", "?"),
                )
                canceled += 1
            except Exception as e:
                log.warning("Tiger cancel order %s for %s failed: %s", order_id, asset, e)
                failed += 1

        return canceled, failed

    def get_account_info(self) -> dict[str, Any] | None:
        """Fetch Tiger account summary (equity, cash, buying power).

        Returns a dict or None on error. Mirrors AlpacaExecutor.get_account_info().
        """
        if not self._client:
            return None

        try:
            summary = self._client.get_account_summary(
                account=self._account, currency="USD"
            )
            if summary is None:
                return None

            # summary may be a DataFrame or a dict depending on SDK version
            if hasattr(summary, "to_dict"):
                # Single-row DataFrame → dict
                row = summary.iloc[0].to_dict() if hasattr(summary, "iloc") else {}
            elif isinstance(summary, dict):
                row = summary
            else:
                row = {}

            return {
                "equity": float(row.get("equity", 0) or 0),
                "cash": float(row.get("cash", 0) or 0),
                "buying_power": float(row.get("buying_power", 0) or 0),
                "currency": row.get("currency", "USD"),
                "account": self._account,
                "paper_mode": self.paper_mode,
            }

        except Exception as e:
            log.error("Tiger get_account_info failed: %s", e)
            return None
