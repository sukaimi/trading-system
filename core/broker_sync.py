"""Broker reconciliation engine — detect and fix portfolio drift.

Compares internal PortfolioState against broker (Alpaca) positions,
identifies ghosts (broker-only), orphans (internal-only), and quantity
mismatches, with optional auto-fix capability.

Reconciliation must never crash the system — all external calls are
wrapped in try/except.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

from core.logger import setup_logger

log = setup_logger("trading.broker_sync")

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


def _load_risk_params() -> dict[str, Any]:
    """Load risk parameters from config/risk_params.json."""
    try:
        path = os.path.join(CONFIG_DIR, "risk_params.json")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


@dataclass
class ReconciliationReport:
    """Result of a broker reconciliation run."""

    ghosts: list[dict[str, Any]] = field(default_factory=list)
    orphans: list[dict[str, Any]] = field(default_factory=list)
    qty_mismatches: list[dict[str, Any]] = field(default_factory=list)
    pending_orders: list[dict[str, Any]] = field(default_factory=list)
    is_clean: bool = True
    summary: str = "No discrepancies found."


class BrokerReconciler:
    """Compare internal portfolio state against broker positions."""

    def __init__(self, executor: Any, portfolio: Any, telegram: Any = None) -> None:
        self._executor = executor
        self._portfolio = portfolio
        self._telegram = telegram

    # ── Core reconciliation ─────────────────────────────────────────────

    def reconcile(self, auto_fix: bool = False) -> ReconciliationReport:
        """Run full reconciliation between internal state and broker.

        If auto_fix is True, discrepancies are corrected in the internal
        portfolio to match the broker (broker is source of truth).
        """
        # Guard: only works with AlpacaExecutor
        try:
            from core.alpaca_executor import AlpacaExecutor
        except ImportError:
            log.warning("AlpacaExecutor not available — skipping reconciliation")
            return ReconciliationReport()

        if not isinstance(self._executor, AlpacaExecutor):
            log.info("Executor is not AlpacaExecutor — skipping reconciliation")
            return ReconciliationReport()

        report = ReconciliationReport()

        # Step 1: Fetch broker positions
        try:
            broker_positions = self._executor.get_all_positions()
        except Exception as e:
            log.error("Failed to fetch broker positions: %s", e)
            report.is_clean = False
            report.summary = f"Broker position fetch failed: {e}"
            return report

        # Step 2: Fetch pending orders
        try:
            pending = self._executor.get_open_orders()
            report.pending_orders = pending
            if pending:
                log.info("Broker has %d pending orders", len(pending))
        except Exception as e:
            log.warning("Failed to fetch open orders: %s", e)

        # Step 3: Build internal positions grouped by asset
        try:
            internal_by_asset: dict[str, list[dict[str, Any]]] = {}
            for pos in list(self._portfolio.open_positions):
                asset = pos.get("asset", "")
                if asset:
                    internal_by_asset.setdefault(asset, []).append(pos)
        except Exception as e:
            log.error("Failed to read internal positions: %s", e)
            report.is_clean = False
            report.summary = f"Internal position read failed: {e}"
            return report

        # broker_positions is already dict keyed by asset
        broker_by_asset = broker_positions

        fixes_applied = False
        risk_params = _load_risk_params()

        # Step 4: Check each broker position against internal
        for asset, bp in broker_by_asset.items():
            broker_qty = float(bp.get("qty", 0))
            internal_positions = internal_by_asset.get(asset, [])

            if not internal_positions:
                # GHOST: broker has it, we don't
                ghost = {
                    "asset": asset,
                    "broker_qty": broker_qty,
                    "broker_side": bp.get("side", "long"),
                    "avg_entry_price": float(bp.get("avg_entry_price", 0)),
                    "current_price": float(bp.get("current_price", 0)),
                }
                report.ghosts.append(ghost)
                log.warning("GHOST detected: %s (broker has %.6f, internal has nothing)", asset, broker_qty)
                self._send_alert(f"GHOST: {asset} — broker has {broker_qty:.6f} but no internal record")

                if auto_fix:
                    self._fix_ghost(asset, bp, risk_params)
                    fixes_applied = True
                continue

            # Compare quantities
            internal_total = sum(float(p.get("quantity", 0)) for p in internal_positions)
            diff = abs(broker_qty - internal_total)

            if diff <= 0.001:
                # Quantities match — OK
                continue

            mismatch = {
                "asset": asset,
                "internal_qty": internal_total,
                "broker_qty": broker_qty,
                "diff": round(diff, 6),
            }
            report.qty_mismatches.append(mismatch)
            log.warning(
                "QTY MISMATCH: %s — internal=%.6f, broker=%.6f (diff=%.6f)",
                asset, internal_total, broker_qty, diff,
            )

            if auto_fix:
                if broker_qty < internal_total:
                    # OVER-TRACKED: remove oldest positions (FIFO) until match
                    self._fix_over_tracked(asset, internal_positions, broker_qty)
                else:
                    # UNDER-TRACKED: adjust newest position qty upward
                    self._fix_under_tracked(asset, internal_positions, broker_qty, internal_total)
                fixes_applied = True

        # Step 5: Check for orphans (internal positions not on broker)
        broker_assets = set(broker_by_asset.keys())
        for asset, positions in internal_by_asset.items():
            if asset in broker_assets:
                continue

            total_qty = sum(float(p.get("quantity", 0)) for p in positions)
            orphan = {
                "asset": asset,
                "internal_qty": total_qty,
                "trade_ids": [p.get("trade_id", "") for p in positions],
            }
            report.orphans.append(orphan)
            log.warning("ORPHAN detected: %s (internal has %.6f, broker has nothing)", asset, total_qty)
            self._send_alert(f"ORPHAN: {asset} — internal has {total_qty:.6f} but broker has no position")

            if auto_fix:
                self._fix_orphan(asset, positions)
                fixes_applied = True

        # Step 6: Persist if fixes were applied
        if auto_fix and fixes_applied:
            try:
                self._portfolio.persist()
                log.info("Portfolio persisted after reconciliation fixes")
            except Exception as e:
                log.error("Portfolio persist after reconciliation failed: %s", e)

        # Step 7: Build report
        report.is_clean = (
            len(report.ghosts) == 0
            and len(report.orphans) == 0
            and len(report.qty_mismatches) == 0
        )

        parts = []
        if report.ghosts:
            parts.append(f"{len(report.ghosts)} ghost(s)")
        if report.orphans:
            parts.append(f"{len(report.orphans)} orphan(s)")
        if report.qty_mismatches:
            parts.append(f"{len(report.qty_mismatches)} qty mismatch(es)")
        if report.pending_orders:
            parts.append(f"{len(report.pending_orders)} pending order(s)")

        if parts:
            prefix = "FIXED: " if auto_fix and fixes_applied else ""
            report.summary = f"{prefix}{', '.join(parts)}"
        else:
            report.summary = "No discrepancies found."

        log.info("Reconciliation complete: %s", report.summary)
        return report

    # ── Auto-fix helpers ────────────────────────────────────────────────

    def _fix_ghost(self, asset: str, broker_pos: dict, risk_params: dict) -> None:
        """Create internal position entry from broker-only position."""
        now = datetime.now(timezone.utc)
        entry_price = float(broker_pos.get("avg_entry_price", 0))
        direction = broker_pos.get("side", "long")
        qty = float(broker_pos.get("qty", 0))

        stop_loss_pct = risk_params.get("default_stop_loss_pct", 3.0) / 100
        take_profit_pct = risk_params.get("default_take_profit_pct", 6.0) / 100

        if direction == "long":
            stop_loss_price = round(entry_price * (1 - stop_loss_pct), 2)
            take_profit_price = round(entry_price * (1 + take_profit_pct), 2)
        else:
            stop_loss_price = round(entry_price * (1 + stop_loss_pct), 2)
            take_profit_price = round(entry_price * (1 - take_profit_pct), 2)

        position = {
            "trade_id": f"broker_sync_{asset}_{now.strftime('%Y%m%d%H%M%S')}",
            "asset": asset,
            "direction": direction,
            "quantity": qty,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "timestamp_open": now.isoformat(),
            "source": "broker_sync",
        }

        try:
            self._portfolio.add_position(position)
            log.info("GHOST FIXED: created internal position for %s (qty=%.6f)", asset, qty)
        except Exception as e:
            log.error("Failed to fix ghost for %s: %s", asset, e)

    def _fix_over_tracked(
        self, asset: str, positions: list[dict], broker_qty: float
    ) -> None:
        """Remove oldest internal positions (FIFO) until total matches broker."""
        # Sort by timestamp_open ascending (oldest first)
        sorted_positions = sorted(
            positions,
            key=lambda p: p.get("timestamp_open", ""),
        )

        remaining = broker_qty
        for pos in sorted_positions:
            pos_qty = float(pos.get("quantity", 0))
            trade_id = pos.get("trade_id", "")

            if remaining <= 0.001:
                # Remove this position entirely
                try:
                    self._portfolio.remove_position(trade_id)
                    self._portfolio.record_trade(0.0)
                    log.info("OVER-TRACKED FIX: removed %s (qty=%.6f)", trade_id, pos_qty)
                except Exception as e:
                    log.error("Failed to remove over-tracked position %s: %s", trade_id, e)
            elif pos_qty > remaining:
                # Trim this position to match remaining
                try:
                    self._portfolio.adjust_position_quantity(trade_id, remaining)
                    log.info(
                        "OVER-TRACKED FIX: trimmed %s from %.6f to %.6f",
                        trade_id, pos_qty, remaining,
                    )
                except Exception as e:
                    log.error("Failed to trim position %s: %s", trade_id, e)
                remaining = 0
            else:
                # This position fits entirely within remaining
                remaining -= pos_qty

    def _fix_under_tracked(
        self,
        asset: str,
        positions: list[dict],
        broker_qty: float,
        internal_total: float,
    ) -> None:
        """Adjust newest internal position qty upward to match broker total."""
        # Sort by timestamp_open descending (newest first)
        sorted_positions = sorted(
            positions,
            key=lambda p: p.get("timestamp_open", ""),
            reverse=True,
        )

        if not sorted_positions:
            return

        newest = sorted_positions[0]
        trade_id = newest.get("trade_id", "")
        old_qty = float(newest.get("quantity", 0))
        adjustment = broker_qty - internal_total
        new_qty = old_qty + adjustment

        try:
            self._portfolio.adjust_position_quantity(trade_id, new_qty)
            log.info(
                "UNDER-TRACKED FIX: adjusted %s from %.6f to %.6f (+%.6f)",
                trade_id, old_qty, new_qty, adjustment,
            )
        except Exception as e:
            log.error("Failed to adjust under-tracked position %s: %s", trade_id, e)

    def _fix_orphan(self, asset: str, positions: list[dict]) -> None:
        """Remove all internal positions for an asset not on broker."""
        for pos in positions:
            trade_id = pos.get("trade_id", "")
            try:
                self._portfolio.remove_position(trade_id)
                self._portfolio.record_trade(0.0)
                log.info("ORPHAN FIXED: removed %s for %s", trade_id, asset)
            except Exception as e:
                log.error("Failed to remove orphan %s: %s", trade_id, e)

    # ── Order management ────────────────────────────────────────────────

    def cancel_all_orders(self) -> int:
        """Cancel all open orders on the broker. Returns count canceled."""
        try:
            from core.alpaca_executor import AlpacaExecutor
        except ImportError:
            return 0

        if not isinstance(self._executor, AlpacaExecutor):
            log.info("Executor is not AlpacaExecutor — cannot cancel orders")
            return 0

        try:
            resp = requests.delete(
                f"{self._executor._base_url}/orders",
                headers=self._executor._headers,
                timeout=15,
            )

            if resp.status_code in (200, 207):
                # Alpaca returns list of canceled order objects
                try:
                    canceled = resp.json()
                    count = len(canceled) if isinstance(canceled, list) else 0
                except Exception:
                    count = 0
                log.info("Canceled %d open orders on broker", count)
                return count
            elif resp.status_code == 204:
                log.info("No open orders to cancel")
                return 0
            else:
                log.warning("Cancel all orders returned %d: %s", resp.status_code, resp.text)
                return 0
        except Exception as e:
            log.error("Cancel all orders failed: %s", e)
            return 0

    # ── Helpers ──────────────────────────────────────────────────────────

    def _send_alert(self, message: str) -> None:
        """Send Telegram alert if notifier is available."""
        if not self._telegram:
            return
        try:
            self._telegram.send_alert(f"[BrokerSync] {message}")
        except Exception as e:
            log.warning("Telegram alert failed: %s", e)
