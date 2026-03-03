"""IBKR trade execution — pure Python, no AI.

Wraps ib_insync to execute validated orders. Gracefully handles
disconnected state (e.g. during development / paper testing setup).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.executor")


class Executor:
    """IBKR trade execution via ib_insync."""

    def __init__(self, ibkr_config: dict[str, Any] | None = None):
        if ibkr_config is None:
            ibkr_config = {}

        self.paper_mode: bool = ibkr_config.get(
            "paper_mode",
            os.getenv("IBKR_PAPER_MODE", "true").lower() == "true",
        )
        self.host: str = ibkr_config.get("host", os.getenv("IBKR_HOST", "127.0.0.1"))
        self.port: int = ibkr_config.get(
            "port",
            int(os.getenv("IBKR_PAPER_PORT", "7497"))
            if self.paper_mode
            else int(os.getenv("IBKR_LIVE_PORT", "7496")),
        )
        self.client_id: int = ibkr_config.get(
            "client_id", int(os.getenv("IBKR_CLIENT_ID", "1"))
        )
        self._ib = None

    def _connect(self) -> bool:
        """Attempt to connect to IBKR. Returns False if unavailable."""
        try:
            from ib_insync import IB

            if self._ib is None:
                self._ib = IB()
            if not self._ib.isConnected():
                self._ib.connect(self.host, self.port, clientId=self.client_id)
            return True
        except Exception as e:
            log.warning("IBKR connection failed: %s", e)
            return False

    def _disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

    def _build_contract(self, asset: str) -> Any:
        """Build an ib_insync contract for the given asset using registry metadata."""
        from ib_insync import Crypto, Stock

        try:
            from core.asset_registry import get_registry
            config = get_registry().get_config(asset)
            if config:
                contract_type = config.get("contract_type", "Stock")
                exchange = config.get("exchange", "SMART")
                if contract_type == "Crypto":
                    return Crypto(asset, exchange, "USD")
                return Stock(asset, exchange, "USD")
        except Exception:
            pass

        # Fallback for known assets
        if asset in ("BTC", "ETH"):
            return Crypto(asset, "PAXOS", "USD")
        return Stock(asset, "SMART", "USD")

    def execute(self, execution_order: dict[str, Any]) -> dict[str, Any]:
        """Execute a validated order. Returns confirmation or error dict."""
        asset = execution_order.get("asset", "")
        thesis_id = execution_order.get("thesis_id", "")

        if not self._connect():
            return {
                "type": "order_error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": "IBKR not connected",
                "thesis_id": thesis_id,
            }

        try:
            from ib_insync import LimitOrder, MarketOrder

            contract = self._build_contract(asset)
            direction = execution_order.get("direction", "long")
            action = "BUY" if direction == "long" else "SELL"
            quantity = execution_order.get("quantity", 0)
            order_type = execution_order.get("order_type", "market")

            if order_type == "limit" and execution_order.get("limit_price"):
                order = LimitOrder(action, quantity, execution_order["limit_price"])
            else:
                order = MarketOrder(action, quantity)

            trade = self._ib.placeOrder(contract, order)
            self._ib.sleep(2)

            # Place stop-loss if defined
            if execution_order.get("stop_loss"):
                self._place_stop_loss(contract, execution_order, action)

            # Place take-profit if defined
            if execution_order.get("take_profit"):
                self._place_take_profit(contract, execution_order, action)

            return {
                "type": "order_confirmation",
                "timestamp": datetime.utcnow().isoformat(),
                "order_id": trade.order.orderId,
                "asset": asset,
                "direction": direction,
                "quantity": quantity,
                "fill_price": trade.orderStatus.avgFillPrice,
                "status": trade.orderStatus.status,
                "thesis_id": thesis_id,
            }

        except Exception as e:
            log.error("Execution failed for %s: %s", asset, e)
            return {
                "type": "order_error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "thesis_id": thesis_id,
            }
        finally:
            self._disconnect()

    def _place_stop_loss(
        self, contract: Any, order: dict[str, Any], action: str
    ) -> None:
        """Place a stop-loss order (opposite direction)."""
        from ib_insync import StopOrder

        sl_action = "SELL" if action == "BUY" else "BUY"
        stop = StopOrder(sl_action, order["quantity"], order["stop_loss"])
        self._ib.placeOrder(contract, stop)
        log.info("Stop-loss placed at %s", order["stop_loss"])

    def _place_take_profit(
        self, contract: Any, order: dict[str, Any], action: str
    ) -> None:
        """Place a take-profit limit order (opposite direction)."""
        from ib_insync import LimitOrder

        tp_action = "SELL" if action == "BUY" else "BUY"
        tp = LimitOrder(tp_action, order["quantity"], order["take_profit"])
        self._ib.placeOrder(contract, tp)
        log.info("Take-profit placed at %s", order["take_profit"])
