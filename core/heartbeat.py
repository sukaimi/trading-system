"""System health monitor — runs every 5 minutes.

Checks CPU, RAM, disk, API connectivity, process health, and security.
Sends Telegram alerts on failures.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import psutil
import requests

from core.logger import setup_logger
from core.schemas import HeartbeatStatus

log = setup_logger("trading.heartbeat")


class Heartbeat:
    """System health monitor — pure Python, no LLM."""

    def __init__(self, telegram_notifier: Any = None, skip_ibkr: bool = False):
        self.telegram = telegram_notifier
        self.skip_ibkr = skip_ibkr

    def check(self) -> HeartbeatStatus:
        """Run all health checks and return status."""
        checks: dict[str, bool] = {}

        # 1. VPS / system health
        checks["cpu"] = psutil.cpu_percent(interval=1) < 80
        checks["ram"] = psutil.virtual_memory().percent < 85
        checks["disk"] = psutil.disk_usage("/").percent < 90

        # 2. API connectivity
        checks["deepseek"] = self._ping_api("https://api.deepseek.com/v1/models")
        checks["kimi"] = self._ping_api("https://api.moonshot.ai/v1/models")

        # 3. IBKR (skip in paper mode)
        if not self.skip_ibkr:
            checks["ibkr"] = self._ping_ibkr()

        # 4. Security
        checks["api_keys_present"] = self._check_env_keys()

        # 5. Trading state
        checks["not_halted"] = not self._is_halted()

        # Aggregate
        failures = [k for k, v in checks.items() if not v]
        all_healthy = len(failures) == 0

        status = HeartbeatStatus(
            timestamp=datetime.utcnow(),
            checks=checks,
            all_healthy=all_healthy,
            failures=failures,
        )

        if not all_healthy:
            msg = f"HEARTBEAT FAILURES: {', '.join(failures)}"
            log.warning(msg)
            if self.telegram:
                self.telegram.send_alert(msg)
        else:
            log.info("Heartbeat OK — all checks passed")

        return status

    @staticmethod
    def _ping_api(url: str, timeout: int = 5) -> bool:
        """Check if an API endpoint is reachable."""
        try:
            resp = requests.get(url, timeout=timeout)
            return resp.status_code < 500
        except Exception:
            return False

    @staticmethod
    def _ping_ibkr() -> bool:
        """Check if IBKR gateway is reachable on localhost."""
        import socket

        port = int(os.getenv("IBKR_PAPER_PORT", "7497"))
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except Exception:
            return False

    @staticmethod
    def _check_env_keys() -> bool:
        """Verify required API keys are present in environment."""
        required = ["DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"]
        return all(os.getenv(k) for k in required)

    @staticmethod
    def _is_halted() -> bool:
        """Check if circuit breaker has halted trading."""
        state_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "portfolio_state.json"
        )
        if os.path.exists(state_file):
            import json

            try:
                with open(state_file) as f:
                    state = json.load(f)
                return state.get("halted", False)
            except Exception:
                return False
        return False
