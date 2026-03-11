"""Tests for BrokerReconciler — ghost/orphan/mismatch detection and auto-fix."""

from unittest.mock import MagicMock, patch

import pytest


def _make_executor():
    """Create a mock AlpacaExecutor with proper class identity."""
    with patch.dict("os.environ", {
        "ALPACA_API_KEY": "test-key",
        "ALPACA_SECRET_KEY": "test-secret",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets/v2",
    }):
        from core.alpaca_executor import AlpacaExecutor
        executor = MagicMock(spec=AlpacaExecutor)
        # Ensure isinstance check passes
        executor.__class__ = AlpacaExecutor
        return executor


def _make_portfolio(positions=None):
    """Create a mock PortfolioState with given open positions."""
    portfolio = MagicMock()
    portfolio.open_positions = list(positions or [])
    return portfolio


def _make_position(asset, qty, trade_id=None, direction="long", entry_price=100.0, timestamp="2026-03-10T00:00:00+00:00"):
    """Helper to build a position dict."""
    return {
        "trade_id": trade_id or f"test_{asset}_{qty}",
        "asset": asset,
        "direction": direction,
        "quantity": qty,
        "entry_price": entry_price,
        "timestamp_open": timestamp,
    }


# ── Clean state ─────────────────────────────────────────────────────────

class TestCleanState:
    """No discrepancies — broker and internal match perfectly."""

    def test_clean_state(self):
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor.get_all_positions.return_value = {
            "AAPL": {"qty": 10.0, "side": "long", "avg_entry_price": 150.0, "current_price": 155.0},
        }
        executor.get_open_orders.return_value = []

        portfolio = _make_portfolio([
            _make_position("AAPL", 10.0),
        ])

        reconciler = BrokerReconciler(executor, portfolio)
        report = reconciler.reconcile()

        assert report.is_clean is True
        assert len(report.ghosts) == 0
        assert len(report.orphans) == 0
        assert len(report.qty_mismatches) == 0
        assert "No discrepancies" in report.summary


# ── Ghost detection ─────────────────────────────────────────────────────

class TestGhostDetection:
    """Broker has position that internal doesn't know about."""

    def test_ghost_detected(self):
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor.get_all_positions.return_value = {
            "BTC": {"qty": 0.01, "side": "long", "avg_entry_price": 70000.0, "current_price": 71000.0},
        }
        executor.get_open_orders.return_value = []

        portfolio = _make_portfolio([])  # No internal positions

        reconciler = BrokerReconciler(executor, portfolio)
        report = reconciler.reconcile()

        assert report.is_clean is False
        assert len(report.ghosts) == 1
        assert report.ghosts[0]["asset"] == "BTC"
        assert report.ghosts[0]["broker_qty"] == 0.01

    @patch("core.broker_sync._load_risk_params", return_value={
        "default_stop_loss_pct": 3.0,
        "default_take_profit_pct": 6.0,
    })
    def test_ghost_auto_fixed(self, mock_risk):
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor.get_all_positions.return_value = {
            "ETH": {"qty": 0.5, "side": "long", "avg_entry_price": 3500.0, "current_price": 3600.0},
        }
        executor.get_open_orders.return_value = []

        portfolio = _make_portfolio([])

        reconciler = BrokerReconciler(executor, portfolio)
        report = reconciler.reconcile(auto_fix=True)

        # Should have created an internal position
        portfolio.add_position.assert_called_once()
        added = portfolio.add_position.call_args[0][0]
        assert added["asset"] == "ETH"
        assert added["quantity"] == 0.5
        assert added["entry_price"] == 3500.0
        assert added["direction"] == "long"
        assert added["source"] == "broker_sync"
        assert "broker_sync_ETH_" in added["trade_id"]

        # Stop loss/take profit derived from risk params
        assert added["stop_loss_price"] == round(3500.0 * 0.97, 2)
        assert added["take_profit_price"] == round(3500.0 * 1.06, 2)

        portfolio.persist.assert_called_once()
        assert len(report.ghosts) == 1


# ── Orphan detection ────────────────────────────────────────────────────

class TestOrphanDetection:
    """Internal has position that broker doesn't have."""

    def test_orphan_detected(self):
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor.get_all_positions.return_value = {}  # Broker empty
        executor.get_open_orders.return_value = []

        portfolio = _make_portfolio([
            _make_position("NVDA", 5.0, trade_id="orphan_nvda"),
        ])

        reconciler = BrokerReconciler(executor, portfolio)
        report = reconciler.reconcile()

        assert report.is_clean is False
        assert len(report.orphans) == 1
        assert report.orphans[0]["asset"] == "NVDA"
        assert report.orphans[0]["internal_qty"] == 5.0
        assert "orphan_nvda" in report.orphans[0]["trade_ids"]

    def test_orphan_auto_fixed(self):
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor.get_all_positions.return_value = {}
        executor.get_open_orders.return_value = []

        portfolio = _make_portfolio([
            _make_position("TSLA", 3.0, trade_id="orphan_tsla_1"),
            _make_position("TSLA", 2.0, trade_id="orphan_tsla_2"),
        ])

        reconciler = BrokerReconciler(executor, portfolio)
        report = reconciler.reconcile(auto_fix=True)

        # Both positions should be removed
        assert portfolio.remove_position.call_count == 2
        removed_ids = [call[0][0] for call in portfolio.remove_position.call_args_list]
        assert "orphan_tsla_1" in removed_ids
        assert "orphan_tsla_2" in removed_ids

        # Record zero P&L for each
        assert portfolio.record_trade.call_count == 2
        for call in portfolio.record_trade.call_args_list:
            assert call[0][0] == 0.0

        portfolio.persist.assert_called_once()
        assert len(report.orphans) == 1


# ── Quantity mismatch ───────────────────────────────────────────────────

class TestQtyMismatch:
    """Internal and broker quantities disagree."""

    def test_qty_mismatch_detected(self):
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor.get_all_positions.return_value = {
            "SPY": {"qty": 8.0, "side": "long", "avg_entry_price": 450.0, "current_price": 455.0},
        }
        executor.get_open_orders.return_value = []

        portfolio = _make_portfolio([
            _make_position("SPY", 10.0),
        ])

        reconciler = BrokerReconciler(executor, portfolio)
        report = reconciler.reconcile()

        assert report.is_clean is False
        assert len(report.qty_mismatches) == 1
        assert report.qty_mismatches[0]["asset"] == "SPY"
        assert report.qty_mismatches[0]["internal_qty"] == 10.0
        assert report.qty_mismatches[0]["broker_qty"] == 8.0

    def test_qty_mismatch_over_tracked_auto_fixed(self):
        """Broker has less than internal — remove oldest FIFO."""
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor.get_all_positions.return_value = {
            "AAPL": {"qty": 5.0, "side": "long", "avg_entry_price": 150.0, "current_price": 155.0},
        }
        executor.get_open_orders.return_value = []

        portfolio = _make_portfolio([
            _make_position("AAPL", 3.0, trade_id="aapl_old", timestamp="2026-03-01T00:00:00+00:00"),
            _make_position("AAPL", 7.0, trade_id="aapl_new", timestamp="2026-03-05T00:00:00+00:00"),
        ])

        reconciler = BrokerReconciler(executor, portfolio)
        report = reconciler.reconcile(auto_fix=True)

        # Internal total = 10, broker = 5
        # FIFO: oldest (aapl_old, qty=3) consumed first → remaining = 5-3 = 2
        # Next (aapl_new, qty=7) trimmed to 2
        portfolio.adjust_position_quantity.assert_called_once_with("aapl_new", 2.0)
        portfolio.persist.assert_called_once()

    def test_qty_mismatch_under_tracked_auto_fixed(self):
        """Broker has more than internal — adjust newest upward."""
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor.get_all_positions.return_value = {
            "META": {"qty": 15.0, "side": "long", "avg_entry_price": 500.0, "current_price": 510.0},
        }
        executor.get_open_orders.return_value = []

        portfolio = _make_portfolio([
            _make_position("META", 5.0, trade_id="meta_old", timestamp="2026-03-01T00:00:00+00:00"),
            _make_position("META", 5.0, trade_id="meta_new", timestamp="2026-03-05T00:00:00+00:00"),
        ])

        reconciler = BrokerReconciler(executor, portfolio)
        report = reconciler.reconcile(auto_fix=True)

        # Internal total = 10, broker = 15 → adjust newest by +5
        portfolio.adjust_position_quantity.assert_called_once_with("meta_new", 10.0)
        portfolio.persist.assert_called_once()


# ── Non-Alpaca executor ─────────────────────────────────────────────────

class TestNonAlpacaExecutor:
    """Reconciliation should skip non-Alpaca executors."""

    def test_non_alpaca_executor_skipped(self):
        from core.broker_sync import BrokerReconciler

        executor = MagicMock()  # Not an AlpacaExecutor
        portfolio = _make_portfolio([_make_position("BTC", 1.0)])

        reconciler = BrokerReconciler(executor, portfolio)
        report = reconciler.reconcile()

        assert report.is_clean is True
        assert report.summary == "No discrepancies found."
        executor.get_all_positions.assert_not_called()


# ── Error handling ──────────────────────────────────────────────────────

class TestErrorHandling:
    """API failures must not crash the system."""

    def test_reconcile_error_handling(self):
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor.get_all_positions.side_effect = Exception("Connection refused")

        portfolio = _make_portfolio([])

        reconciler = BrokerReconciler(executor, portfolio)
        report = reconciler.reconcile()

        # Should return a report with error, not raise
        assert report.is_clean is False
        assert "failed" in report.summary.lower()

    def test_open_orders_error_does_not_crash(self):
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor.get_all_positions.return_value = {}
        executor.get_open_orders.side_effect = Exception("Timeout")

        portfolio = _make_portfolio([])

        reconciler = BrokerReconciler(executor, portfolio)
        report = reconciler.reconcile()

        # Should still complete — open orders error is non-fatal
        assert report.is_clean is True


# ── Pending orders ──────────────────────────────────────────────────────

class TestPendingOrders:
    """Pending orders should be included in the report."""

    def test_pending_orders_reported(self):
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor.get_all_positions.return_value = {}
        executor.get_open_orders.return_value = [
            {"id": "order-1", "symbol": "AAPL", "side": "buy", "qty": 10.0, "type": "limit"},
        ]

        portfolio = _make_portfolio([])

        reconciler = BrokerReconciler(executor, portfolio)
        report = reconciler.reconcile()

        assert len(report.pending_orders) == 1
        assert "pending order" in report.summary.lower()


# ── Telegram alerts ─────────────────────────────────────────────────────

class TestTelegramAlerts:
    """Ghosts and orphans should trigger Telegram alerts."""

    def test_ghost_sends_telegram(self):
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor.get_all_positions.return_value = {
            "BTC": {"qty": 0.01, "side": "long", "avg_entry_price": 70000.0, "current_price": 71000.0},
        }
        executor.get_open_orders.return_value = []

        portfolio = _make_portfolio([])
        telegram = MagicMock()

        reconciler = BrokerReconciler(executor, portfolio, telegram)
        reconciler.reconcile()

        telegram.send_alert.assert_called_once()
        msg = telegram.send_alert.call_args[0][0]
        assert "GHOST" in msg
        assert "BTC" in msg

    def test_orphan_sends_telegram(self):
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor.get_all_positions.return_value = {}
        executor.get_open_orders.return_value = []

        portfolio = _make_portfolio([
            _make_position("NVDA", 5.0),
        ])
        telegram = MagicMock()

        reconciler = BrokerReconciler(executor, portfolio, telegram)
        reconciler.reconcile()

        telegram.send_alert.assert_called_once()
        msg = telegram.send_alert.call_args[0][0]
        assert "ORPHAN" in msg
        assert "NVDA" in msg


# ── Cancel all orders ───────────────────────────────────────────────────

class TestCancelAllOrders:
    """Test cancel_all_orders method."""

    @patch("core.broker_sync.requests")
    def test_cancel_all_orders(self, mock_requests):
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor._base_url = "https://paper-api.alpaca.markets/v2"
        executor._headers = {"APCA-API-KEY-ID": "test"}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"id": "order-1"}, {"id": "order-2"}]
        mock_requests.delete.return_value = mock_resp

        portfolio = _make_portfolio([])
        reconciler = BrokerReconciler(executor, portfolio)
        count = reconciler.cancel_all_orders()

        assert count == 2
        mock_requests.delete.assert_called_once()

    def test_cancel_non_alpaca_returns_zero(self):
        from core.broker_sync import BrokerReconciler

        executor = MagicMock()  # Not AlpacaExecutor
        portfolio = _make_portfolio([])
        reconciler = BrokerReconciler(executor, portfolio)

        assert reconciler.cancel_all_orders() == 0


# ── Quantity tolerance ──────────────────────────────────────────────────

class TestQuantityTolerance:
    """Quantities within 0.001 tolerance should not trigger mismatch."""

    def test_within_tolerance_is_clean(self):
        from core.broker_sync import BrokerReconciler

        executor = _make_executor()
        executor.get_all_positions.return_value = {
            "BTC": {"qty": 0.010001, "side": "long", "avg_entry_price": 70000.0, "current_price": 71000.0},
        }
        executor.get_open_orders.return_value = []

        portfolio = _make_portfolio([
            _make_position("BTC", 0.01),
        ])

        reconciler = BrokerReconciler(executor, portfolio)
        report = reconciler.reconcile()

        assert report.is_clean is True
        assert len(report.qty_mismatches) == 0
