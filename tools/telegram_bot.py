"""Telegram notification bot for trading alerts, summaries, and reports.

Operates as a no-op when TELEGRAM_BOT_TOKEN is not configured,
allowing the system to run without Telegram during development.
"""

from __future__ import annotations

import os
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.telegram")


class TelegramNotifier:
    """Send trading alerts and summaries via Telegram.

    If bot token is not configured, all methods are silent no-ops.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            log.info("Telegram not configured — notifications disabled")

    async def _send_message(self, text: str) -> bool:
        """Send a message via Telegram Bot API."""
        if not self.enabled:
            return False

        try:
            from telegram import Bot

            bot = Bot(token=self.bot_token)
            await bot.send_message(chat_id=self.chat_id, text=text, parse_mode="HTML")
            return True
        except Exception as e:
            log.error("Telegram send failed: %s", e)
            return False

    def _run_async(self, coro) -> None:
        """Run an async coroutine from sync context, handling both threaded and async callers."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            # Already inside an async context (e.g. FastAPI) — schedule as fire-and-forget task
            loop.create_task(coro)
        except RuntimeError:
            # No running event loop — create one (e.g. scheduler thread)
            asyncio.run(coro)

    def send_alert(self, message: str) -> None:
        """Send an alert message (sync wrapper — fires and forgets)."""
        if not self.enabled:
            log.debug("Telegram alert (disabled): %s", message[:80])
            return

        try:
            self._run_async(self._send_message(f"<b>ALERT</b>\n{message}"))
        except Exception as e:
            log.error("Failed to send alert: %s", e)

    def send_daily_summary(self, summary: dict[str, Any]) -> None:
        """Send end-of-day portfolio summary."""
        if not self.enabled:
            log.debug("Telegram daily summary (disabled)")
            return

        text = (
            "<b>Daily Summary</b>\n"
            f"Equity: ${summary.get('equity', 0):.2f}\n"
            f"Daily P&L: {summary.get('daily_pnl_pct', 0):.2f}%\n"
            f"Open positions: {summary.get('open_positions', 0)}\n"
            f"Drawdown: {summary.get('drawdown_from_peak_pct', 0):.2f}%"
        )

        try:
            self._run_async(self._send_message(text))
        except Exception as e:
            log.error("Failed to send daily summary: %s", e)

    def send_weekly_report(self, report: dict[str, Any]) -> None:
        """Send weekly strategy report."""
        if not self.enabled:
            log.debug("Telegram weekly report (disabled)")
            return

        text = (
            "<b>Weekly Report</b>\n"
            f"Week: {report.get('week_reviewed', 'N/A')}\n"
            f"Return: {report.get('weekly_return_pct', 0):.2f}%\n"
            f"Trades: {report.get('total_trades', 0)}\n"
            f"Win rate: {report.get('win_rate', 0):.0%}"
        )

        try:
            self._run_async(self._send_message(text))
        except Exception as e:
            log.error("Failed to send weekly report: %s", e)

    def send_circuit_breaker_alert(self, decision: dict[str, Any]) -> None:
        """Send circuit breaker alert with decision details."""
        if not self.enabled:
            log.debug("Telegram circuit breaker alert (disabled)")
            return

        triggers = ", ".join(decision.get("triggers_fired", []))
        text = (
            "<b>CIRCUIT BREAKER FIRED</b>\n\n"
            f"Triggers: {triggers}\n"
            f"Decision: <b>{decision.get('decision', 'UNKNOWN')}</b>\n"
            f"Reasoning: {decision.get('reasoning', 'N/A')}\n\n"
            f"Resume conditions: {decision.get('resume_conditions', 'N/A')}\n\n"
            f"{decision.get('telegram_message', '')}"
        )

        try:
            self._run_async(self._send_message(text))
        except Exception as e:
            log.error("Failed to send circuit breaker alert: %s", e)

    def send_optimization_summary(self, changes: list[dict[str, Any]]) -> None:
        """Send strategy optimization summary after parameter changes."""
        if not self.enabled:
            log.debug("Telegram optimization summary (disabled)")
            return

        lines = ["<b>STRATEGY UPDATE</b>\n"]
        for change in changes:
            agent = change.get("agent", "?")
            param = change.get("parameter", "?")
            old = change.get("old_value", "?")
            new = change.get("new_value", "?")
            reason = change.get("reason", "")
            lines.append(f"  {agent}.{param}: {old} → {new}")
            if reason:
                lines.append(f"    Reason: {reason}")

        version = changes[-1].get("version", "?") if changes else "?"
        lines.append(f"\nStrategy version: v{version}")

        try:
            self._run_async(self._send_message("\n".join(lines)))
        except Exception as e:
            log.error("Failed to send optimization summary: %s", e)
